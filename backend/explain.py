"""Объяснение решения: SHAP → человеческий язык → сообщение клиенту.

Требование кейса — объяснимость. Мы понимаем его буквально: у каждого решения
три уровня объяснения, потому что читать их будут три разных человека.

  1. Вклады SHAP — для дата-сайентиста. Числа, которые можно проверить.
  2. Причины словами — для оператора антифрода. «Сумма в 12 раз больше
     обычной», «устройство появилось 4 минуты назад». По ним человек за
     полминуты понимает случай и принимает решение.
  3. Сообщение клиенту — то, что реально увидит человек в приложении.
     Для жертвы социальной инженерии это самая важная часть системы:
     единственный шанс разорвать разговор с мошенником — прямым текстом
     сказать, что происходит.

Показываем не только доводы против клиента, но и доводы за него. Оператору
важно видеть обе стороны, иначе объяснение превращается в оправдание.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from backend.cost import ALLOW, BLOCK, HOLD, STEP_UP

LANGUAGES = ("ru", "kk", "en")

# Технические коды в объяснении недопустимы: объяснение читает оператор и
# клиент, а не разработчик.
CATEGORY_NAMES: dict[str, dict[str, str]] = {
    "grocery": {"ru": "продукты", "kk": "азық-түлік", "en": "groceries"},
    "transport": {"ru": "транспорт", "kk": "көлік", "en": "transport"},
    "cafe": {"ru": "кафе", "kk": "дәмхана", "en": "cafés"},
    "pharmacy": {"ru": "аптека", "kk": "дәріхана", "en": "pharmacy"},
    "utilities": {"ru": "коммунальные платежи", "kk": "коммуналдық төлемдер", "en": "utilities"},
    "marketplace": {"ru": "маркетплейс", "kk": "маркетплейс", "en": "marketplace"},
    "clothing": {"ru": "одежда", "kk": "киім", "en": "clothing"},
    "electronics": {"ru": "электроника", "kk": "электроника", "en": "electronics"},
    "entertainment": {"ru": "развлечения", "kk": "ойын-сауық", "en": "entertainment"},
    "travel": {"ru": "путешествия", "kk": "саяхат", "en": "travel"},
    "jewelry": {"ru": "ювелирные изделия", "kk": "зергерлік бұйымдар", "en": "jewellery"},
    "crypto": {"ru": "криптовалюта", "kk": "криптовалюта", "en": "crypto"},
    "gambling": {"ru": "ставки и азартные игры", "kk": "бәс тігу және құмар ойындар", "en": "gambling"},
    "atm_withdrawal": {"ru": "снятие наличных", "kk": "қолма-қол ақша алу", "en": "cash withdrawal"},
    "transfer_p2p": {"ru": "перевод другому человеку", "kk": "басқа адамға аударым", "en": "peer-to-peer transfer"},
}

CHANNEL_NAMES: dict[str, dict[str, str]] = {
    "pos": {"ru": "оплата картой в магазине", "kk": "дүкенде картамен төлеу", "en": "card payment in store"},
    "ecom": {"ru": "оплата в интернете", "kk": "интернетте төлеу", "en": "online payment"},
    "mobile_app": {"ru": "мобильное приложение", "kk": "мобильді қосымша", "en": "mobile app"},
    "atm": {"ru": "банкомат", "kk": "банкомат", "en": "ATM"},
    "p2p": {"ru": "перевод по номеру", "kk": "нөмір бойынша аударым", "en": "transfer by phone number"},
}

TX_TYPE_NAMES: dict[str, dict[str, str]] = {
    "purchase": {"ru": "покупка", "kk": "сатып алу", "en": "purchase"},
    "transfer": {"ru": "перевод", "kk": "аударым", "en": "transfer"},
    "withdrawal": {"ru": "снятие наличных", "kk": "қолма-қол ақша алу", "en": "cash withdrawal"},
}


def localized(table: dict, key: str, lang: str) -> str:
    """Человеческое название кода на нужном языке."""
    return table.get(str(key), {}).get(lang, str(key))

# --------------------------------------------------------------------------
# Как признак превращается в фразу
# --------------------------------------------------------------------------
# Для каждого признака: как достать понятное значение и как назвать его на
# трёх языках. {v} подставляется из value_fn.

RISK_TEMPLATES: dict[str, dict] = {
    "amount_to_avg": {
        "value": lambda f, tx: f["amount_to_avg"],
        "fmt": "{v:.1f}",
        "ru": "Сумма в {v} раза больше обычного чека клиента",
        "kk": "Сома клиенттің әдеттегі чегінен {v} есе үлкен",
        "en": "Amount is {v}× the client's usual ticket",
    },
    "log_amount": {
        "value": lambda f, tx: float(tx.get("amount", 0) or 0),
        "fmt": "{v:,.0f}",
        "ru": "Крупная сумма операции — {v} ₸",
        "kk": "Операцияның сомасы үлкен — {v} ₸",
        "en": "Large transaction amount — {v} KZT",
    },
    "amount_z": {
        "value": lambda f, tx: f["amount_z"],
        "fmt": "{v:.1f}",
        "ru": "Сумма выходит за привычный разброс клиента на {v} стандартных отклонения",
        "kk": "Сома клиенттің әдеттегі ауқымынан {v} стандартты ауытқуға шығады",
        "en": "Amount is {v} standard deviations outside the client's usual range",
    },
    "amount_to_max": {
        "value": lambda f, tx: f["amount_to_max"],
        "fmt": "{v:.1f}",
        "ru": "Сумма в {v} раза больше самой крупной прошлой операции клиента",
        "kk": "Сома клиенттің бұрынғы ең үлкен операциясынан {v} есе көп",
        "en": "Amount is {v}× the client's largest previous transaction",
    },
    "is_amount_record": {
        "ru": "Это самая крупная операция за всю историю клиента",
        "kk": "Бұл — клиент тарихындағы ең ірі операция",
        "en": "Largest transaction in the client's entire history",
    },
    "is_round_amount": {
        "ru": "Ровная сумма, кратная 10 000 ₸ — характерно для перевода под диктовку",
        "kk": "10 000 ₸-ге еселік дөңгелек сома — айтқанмен жасалған аударымға тән",
        "en": "Round amount, a multiple of 10,000 KZT — typical of a dictated transfer",
    },
    "is_new_device": {
        "ru": "Устройство банк видит впервые",
        "kk": "Құрылғыны банк алғаш рет көріп тұр",
        "en": "The bank is seeing this device for the first time",
    },
    "device_age_days": {
        "value": lambda f, tx: f["device_age_days"] * 24.0,
        "fmt": "{v:.1f}",
        "ru": "Устройство появилось у клиента всего {v} часов назад",
        "kk": "Құрылғы клиентте небәрі {v} сағат бұрын пайда болды",
        "en": "The device first appeared only {v} hours ago",
    },
    "device_tx_share": {
        "ru": "С этого устройства клиент почти никогда не платит",
        "kk": "Клиент бұл құрылғыдан ешқашан дерлік төлемейді",
        "en": "The client almost never pays from this device",
    },
    "is_new_city": {
        "value": lambda f, tx: tx.get("city", ""),
        "fmt": "{v}",
        "ru": "Новый для клиента город — {v}",
        "kk": "Клиент үшін жаңа қала — {v}",
        "en": "A city new to this client — {v}",
    },
    "is_foreign": {
        "value": lambda f, tx: tx.get("city", ""),
        "fmt": "{v}",
        "ru": "Операция из-за рубежа — {v}",
        "kk": "Операция шетелден — {v}",
        "en": "Transaction from abroad — {v}",
    },
    "distance_from_home_km": {
        "value": lambda f, tx: f["distance_from_home_km"],
        "fmt": "{v:,.0f}",
        "ru": "До домашнего города клиента {v} км",
        "kk": "Клиенттің үй қаласына дейін {v} км",
        "en": "{v} km from the client's home city",
    },
    "travel_speed_kmh": {
        "value": lambda f, tx: f["travel_speed_kmh"],
        "fmt": "{v:,.0f}",
        "ru": "Невозможное перемещение: {v} км/ч с момента прошлой операции",
        "kk": "Мүмкін емес жылжу: өткен операциядан бері {v} км/сағ",
        "en": "Impossible travel: {v} km/h since the previous transaction",
    },
    "distance_from_prev_km": {
        "value": lambda f, tx: f["distance_from_prev_km"],
        "fmt": "{v:,.0f}",
        "ru": "Прошлая операция была за {v} км отсюда",
        "kk": "Алдыңғы операция осы жерден {v} км қашықтықта болған",
        "en": "The previous transaction was {v} km away",
    },
    "is_vpn": {
        "ru": "Подключение через VPN — настоящее местоположение скрыто",
        "kk": "VPN арқылы қосылым — нақты орналасқан жері жасырылған",
        "en": "Connection over VPN — the real location is hidden",
    },
    "clients_per_device": {
        "value": lambda f, tx: f["clients_per_device"],
        "fmt": "{v:.0f}",
        "ru": "С этого устройства платят {v} разных клиентов — признак кольца карт",
        "kk": "Бұл құрылғыдан {v} түрлі клиент төлейді — карта сақинасының белгісі",
        "en": "{v} different clients pay from this device — a card-ring signature",
    },
    "clients_per_ip": {
        "value": lambda f, tx: f["clients_per_ip"],
        "fmt": "{v:.0f}",
        "ru": "С этого IP-адреса работают {v} разных клиентов",
        "kk": "Бұл IP-мекенжайдан {v} түрлі клиент жұмыс істейді",
        "en": "{v} different clients operate from this IP address",
    },
    "clients_per_recipient": {
        "value": lambda f, tx: f["clients_per_recipient"],
        "fmt": "{v:.0f}",
        "ru": "На этого получателя уже переводили {v} других клиентов — похоже на дроппера",
        "kk": "Бұл алушыға {v} басқа клиент аударым жасаған — дроп-шот сияқты",
        "en": "{v} other clients have already sent money to this recipient — likely a mule account",
    },
    "recipient_clients_24h": {
        "value": lambda f, tx: f["recipient_clients_24h"],
        "fmt": "{v:.0f}",
        "ru": "За сутки на этот счёт перевели {v} разных клиентов — так выглядит дроп-счёт",
        "kk": "Тәулік ішінде бұл шотқа {v} түрлі клиент аударым жасаған — дроп-шот осылай көрінеді",
        "en": "{v} different clients sent money to this account within 24 hours — a drop-account pattern",
    },
    "recipient_clients_7d": {
        "value": lambda f, tx: f["recipient_clients_7d"],
        "fmt": "{v:.0f}",
        "ru": "За неделю на этот счёт перевели {v} разных клиентов",
        "kk": "Апта ішінде бұл шотқа {v} түрлі клиент аударым жасаған",
        "en": "{v} different clients sent money to this account over the past week",
    },
    "recipient_inbound_24h": {
        "value": lambda f, tx: f["recipient_inbound_24h"],
        "fmt": "{v:.0f}",
        "ru": "На счёт получателя за сутки пришло {v} переводов",
        "kk": "Алушының шотына тәулігіне {v} аударым түскен",
        "en": "The recipient's account received {v} transfers in 24 hours",
    },
    "recipient_amount_24h": {
        "value": lambda f, tx: math.expm1(f["recipient_amount_24h"]),
        "fmt": "{v:,.0f}",
        "ru": "За сутки на счёт получателя собрали {v} ₸",
        "kk": "Тәулік ішінде алушының шотына {v} ₸ жиналған",
        "en": "{v} KZT was collected into the recipient's account within 24 hours",
    },
    "recipient_age_days": {
        "value": lambda f, tx: f["recipient_age_days"],
        "fmt": "{v:.1f}",
        "ru": "Счёт получателя появился в поле зрения банка {v} дней назад",
        "kk": "Алушының шоты банк өрісінде {v} күн бұрын пайда болған",
        "en": "The recipient's account first appeared to the bank {v} days ago",
    },
    "recipient_is_new_to_bank": {
        "ru": "Банк видит этот счёт получателя впервые",
        "kk": "Банк бұл алушының шотын алғаш рет көріп тұр",
        "en": "The bank is seeing this recipient account for the first time",
    },
    "device_clients_24h": {
        "value": lambda f, tx: f["device_clients_24h"],
        "fmt": "{v:.0f}",
        "ru": "За сутки с этого устройства платили {v} разных клиентов",
        "kk": "Тәулік ішінде бұл құрылғыдан {v} түрлі клиент төлеген",
        "en": "{v} different clients paid from this device within 24 hours",
    },
    "ip_clients_24h": {
        "value": lambda f, tx: f["ip_clients_24h"],
        "fmt": "{v:.0f}",
        "ru": "За сутки с этого IP работали {v} разных клиентов",
        "kk": "Тәулік ішінде бұл IP-ден {v} түрлі клиент жұмыс істеген",
        "en": "{v} different clients operated from this IP within 24 hours",
    },
    "component_clients": {
        "value": lambda f, tx: f["component_clients"],
        "fmt": "{v:.0f}",
        "ru": "Клиент входит в связанную группу из {v} человек — общие устройства, адреса и получатели",
        "kk": "Клиент {v} адамнан тұратын байланысты топқа кіреді — ортақ құрылғылар, мекенжайлар, алушылар",
        "en": "The client belongs to a connected group of {v} people — shared devices, addresses and recipients",
    },
    "component_size": {
        "value": lambda f, tx: f["component_size"],
        "fmt": "{v:.0f}",
        "ru": "Размер связанной группы — {v} узлов графа",
        "kk": "Байланысты топтың мөлшері — графтың {v} түйіні",
        "en": "The connected group spans {v} graph nodes",
    },
    "is_new_recipient": {
        "ru": "Получатель новый, раньше клиент ему не переводил",
        "kk": "Алушы жаңа, бұрын клиент оған аударым жасамаған",
        "en": "New recipient — the client has never sent money here before",
    },
    "remote_access": {
        "ru": "На устройстве активна программа удалённого доступа — экраном клиента управляет кто-то ещё",
        "kk": "Құрылғыда қашықтан қол жеткізу бағдарламасы белсенді — экранды басқа біреу басқаруда",
        "en": "Remote-access software is active — someone else may be controlling the screen",
    },
    "call_minutes_before": {
        "value": lambda f, tx: f["call_minutes_before"],
        "fmt": "{v:.0f}",
        "ru": "Перед операцией клиент {v} минут говорил по телефону",
        "kk": "Операция алдында клиент {v} минут телефонмен сөйлескен",
        "en": "The client was on a phone call for {v} minutes before the transaction",
    },
    "has_long_call": {
        "ru": "Длинный телефонный разговор непосредственно перед переводом",
        "kk": "Аударым алдында ұзақ телефон әңгімесі",
        "en": "A long phone call immediately before the transfer",
    },
    "session_duration_s": {
        "value": lambda f, tx: f["session_duration_s"],
        "fmt": "{v:.0f}",
        "ru": "Нетипично короткая сессия — {v} секунд от входа до платежа",
        "kk": "Әдеттен тыс қысқа сессия — кіруден төлемге дейін {v} секунд",
        "en": "Unusually short session — {v} seconds from login to payment",
    },
    "tx_count_1h": {
        "value": lambda f, tx: f["tx_count_1h"],
        "fmt": "{v:.0f}",
        "ru": "Серия из {v} операций за последний час",
        "kk": "Соңғы сағатта {v} операциядан тұратын топтама",
        "en": "A burst of {v} transactions in the last hour",
    },
    "tx_count_24h": {
        "value": lambda f, tx: f["tx_count_24h"],
        "fmt": "{v:.0f}",
        "ru": "{v} операций за сутки — заметно чаще обычного",
        "kk": "Тәулігіне {v} операция — әдеттегіден жиі",
        "en": "{v} transactions in 24 hours — noticeably more often than usual",
    },
    "amount_24h_to_avg": {
        "value": lambda f, tx: f["amount_24h_to_avg"],
        "fmt": "{v:.1f}",
        "ru": "За сутки потрачено {v} обычных чеков клиента",
        "kk": "Тәулік ішінде клиенттің {v} әдеттегі чегі жұмсалған",
        "en": "{v} of the client's usual tickets spent within 24 hours",
    },
    "hours_since_prev": {
        "value": lambda f, tx: f["hours_since_prev"] * 60.0,
        "fmt": "{v:.0f}",
        "ru": "Всего {v} минут с предыдущей операции",
        "kk": "Алдыңғы операциядан бері небәрі {v} минут",
        "en": "Only {v} minutes since the previous transaction",
    },
    "category_risk": {
        "localized": ("merchant_category", "CATEGORY_NAMES"),
        "ru": "Рискованная категория операции — {v}",
        "kk": "Операцияның тәуекелді санаты — {v}",
        "en": "High-risk transaction category — {v}",
    },
    "is_new_category": {
        "localized": ("merchant_category", "CATEGORY_NAMES"),
        "ru": "Клиент никогда раньше не платил в категории «{v}»",
        "kk": "Клиент бұрын «{v}» санатында ешқашан төлемеген",
        "en": "The client has never paid in the “{v}” category before",
    },
    "channel_code": {
        "localized": ("channel", "CHANNEL_NAMES"),
        "ru": "Необычный для клиента способ оплаты — {v}",
        "kk": "Клиент үшін әдеттен тыс төлем тәсілі — {v}",
        "en": "Unusual payment channel for this client — {v}",
    },
    "tx_type_code": {
        "localized": ("tx_type", "TX_TYPE_NAMES"),
        "ru": "Тип операции повышает риск — {v}",
        "kk": "Операция түрі тәуекелді арттырады — {v}",
        "en": "The transaction type raises the risk — {v}",
    },
    "is_night": {
        "ru": "Ночная операция — клиент почти никогда не платит в это время",
        "kk": "Түнгі операция — клиент бұл уақытта дерлік төлемейді",
        "en": "Night-time transaction — the client almost never pays at this hour",
    },
    "is_transfer": {
        "ru": "Перевод другому человеку — деньги уходят безвозвратно",
        "kk": "Басқа адамға аударым — ақша қайтарымсыз кетеді",
        "en": "Peer-to-peer transfer — the money leaves irreversibly",
    },
    "client_age": {
        "value": lambda f, tx: f["client_age"],
        "fmt": "{v:.0f}",
        "ru": "Возраст клиента {v} лет — возрастная группа, по которой чаще всего работают телефонные мошенники",
        "kk": "Клиенттің жасы {v} — телефон алаяқтары ең жиі нысанаға алатын жас тобы",
        "en": "Client is {v} — the age group most often targeted by phone scammers",
    },
    "own_context": {
        "ru": "Своё устройство, свой город, без VPN — формально всё в порядке, и именно так выглядит обман клиента",
        "kk": "Өз құрылғысы, өз қаласы, VPN жоқ — сырттай бәрі дұрыс, дәл осылай көрінеді клиентті алдау",
        "en": "Own device, own city, no VPN — formally clean, which is exactly how a scammed client looks",
    },
    "hour": {
        "value": lambda f, tx: f["hour"],
        "fmt": "{v:.0f}",
        "ru": "Необычное для клиента время — {v} часов",
        "kk": "Клиент үшін әдеттен тыс уақыт — сағат {v}",
        "en": "Unusual hour for this client — {v}:00",
    },
    "is_weekend": {
        "ru": "Выходной день — клиент в это время обычно не платит",
        "kk": "Демалыс күні — клиент әдетте бұл уақытта төлемейді",
        "en": "Weekend — the client does not usually pay at this time",
    },
    "night_share": {
        "ru": "Клиент почти никогда не совершает ночных операций",
        "kk": "Клиент түнгі операцияларды дерлік жасамайды",
        "en": "The client almost never transacts at night",
    },
    "client_tenure_days": {
        "value": lambda f, tx: f["client_tenure_days"],
        "fmt": "{v:.0f}",
        "ru": "Клиент с банком всего {v} дней",
        "kk": "Клиент банкпен небәрі {v} күн",
        "en": "The client has been with the bank only {v} days",
    },
    "city_tx_share": {
        "ru": "Клиент почти никогда не платит из этого города",
        "kk": "Клиент бұл қаладан дерлік төлемейді",
        "en": "The client almost never pays from this city",
    },
    "history_len": {
        "value": lambda f, tx: f["history_len"],
        "fmt": "{v:.0f}",
        "ru": "Короткая история — всего {v} операций, сравнивать почти не с чем",
        "kk": "Тарихы қысқа — барлығы {v} операция, салыстыратын ештеңе жоқ",
        "en": "Short history — only {v} transactions to compare against",
    },
}

# Доводы в пользу клиента
SAFE_TEMPLATES: dict[str, dict] = {
    "device_age_days": {
        "ru": "Устройство давно знакомо банку",
        "kk": "Құрылғы банкке ежелден таныс",
        "en": "The device is long familiar to the bank",
    },
    "device_tx_share": {
        "ru": "Клиент постоянно платит именно с этого устройства",
        "kk": "Клиент үнемі осы құрылғыдан төлейді",
        "en": "The client consistently pays from this very device",
    },
    "is_new_device": {
        "ru": "Устройство привычное",
        "kk": "Құрылғы таныс",
        "en": "Known device",
    },
    "city_tx_share": {
        "ru": "Обычный для клиента город",
        "kk": "Клиент үшін әдеттегі қала",
        "en": "A city the client normally pays from",
    },
    "is_new_city": {
        "ru": "Город клиенту не новый",
        "kk": "Қала клиент үшін жаңа емес",
        "en": "Not a new city for this client",
    },
    "amount_to_avg": {
        "ru": "Сумма обычная для этого клиента",
        "kk": "Сома бұл клиент үшін әдеттегідей",
        "en": "The amount is normal for this client",
    },
    "amount_z": {
        "ru": "Сумма укладывается в привычный разброс",
        "kk": "Сома әдеттегі ауқымға сәйкес келеді",
        "en": "The amount fits the client's usual range",
    },
    "log_amount": {
        "ru": "Небольшая сумма",
        "kk": "Шағын сома",
        "en": "Small amount",
    },
    "category_risk": {
        "ru": "Бытовая категория покупки",
        "kk": "Күнделікті сатып алу санаты",
        "en": "Everyday merchant category",
    },
    "history_len": {
        "ru": "Длинная история операций — поведение хорошо изучено",
        "kk": "Операциялар тарихы ұзақ — мінез-құлық жақсы зерттелген",
        "en": "Long transaction history — behaviour is well understood",
    },
    "hours_since_prev": {
        "ru": "Спокойный ритм операций",
        "kk": "Операциялардың қалыпты ырғағы",
        "en": "Calm transaction rhythm",
    },
    "clients_per_device": {
        "ru": "Устройством пользуется только этот клиент",
        "kk": "Құрылғыны тек осы клиент пайдаланады",
        "en": "Only this client uses the device",
    },
    "is_vpn": {
        "ru": "Прямое подключение, без VPN",
        "kk": "VPN-сіз тікелей қосылым",
        "en": "Direct connection, no VPN",
    },
    "own_context": {
        "ru": "Знакомая обстановка операции",
        "kk": "Операцияның таныс жағдайы",
        "en": "Familiar transaction context",
    },
    "channel_code": {
        "ru": "Привычный способ оплаты",
        "kk": "Әдеттегі төлем тәсілі",
        "en": "Usual payment channel",
    },
    "tx_type_code": {
        "ru": "Обычный тип операции",
        "kk": "Әдеттегі операция түрі",
        "en": "Ordinary transaction type",
    },
    "is_transfer": {
        "ru": "Это покупка, а не перевод другому человеку",
        "kk": "Бұл — басқа адамға аударым емес, сатып алу",
        "en": "This is a purchase, not a transfer to another person",
    },
    "session_duration_s": {
        "ru": "Спокойная по длительности сессия",
        "kk": "Ұзақтығы қалыпты сессия",
        "en": "Normal session length",
    },
    "call_minutes_before": {
        "ru": "Перед операцией не было телефонного разговора",
        "kk": "Операция алдында телефон әңгімесі болмаған",
        "en": "No phone call preceded the transaction",
    },
    "remote_access": {
        "ru": "Программ удалённого доступа не обнаружено",
        "kk": "Қашықтан қол жеткізу бағдарламалары анықталмаған",
        "en": "No remote-access software detected",
    },
    "distance_from_home_km": {
        "ru": "Операция рядом с домашним городом клиента",
        "kk": "Операция клиенттің үй қаласына жақын",
        "en": "Transaction close to the client's home city",
    },
    "clients_per_ip": {
        "ru": "С этого IP работает только один клиент",
        "kk": "Бұл IP-ден тек бір клиент жұмыс істейді",
        "en": "Only one client operates from this IP",
    },
    "is_night": {
        "ru": "Операция в обычное дневное время",
        "kk": "Операция әдеттегі күндізгі уақытта",
        "en": "Transaction during normal daytime hours",
    },
}

SAFE_TEMPLATES.update({
    "amount_to_max": {
        "ru": "Сумма не превышает прошлых операций клиента",
        "kk": "Сома клиенттің бұрынғы операцияларынан аспайды",
        "en": "The amount does not exceed the client's past transactions",
    },
    "is_amount_record": {
        "ru": "Не рекорд по сумме для этого клиента",
        "kk": "Бұл клиент үшін сома бойынша рекорд емес",
        "en": "Not a record amount for this client",
    },
    "is_round_amount": {
        "ru": "Сумма не круглая — на перевод под диктовку не похоже",
        "kk": "Сома дөңгелек емес — айтқанмен жасалған аударымға ұқсамайды",
        "en": "Not a round amount — unlike a dictated transfer",
    },
    "hour": {
        "ru": "Обычное для клиента время операции",
        "kk": "Клиент үшін әдеттегі операция уақыты",
        "en": "A normal hour for this client",
    },
    "is_weekend": {
        "ru": "Будний день, как обычно у этого клиента",
        "kk": "Жұмыс күні, бұл клиент үшін әдеттегідей",
        "en": "A weekday, as usual for this client",
    },
    "tx_count_1h": {
        "ru": "Серии операций подряд нет",
        "kk": "Қатарынан операциялар топтамасы жоқ",
        "en": "No burst of consecutive transactions",
    },
    "tx_count_24h": {
        "ru": "За сутки операций не больше обычного",
        "kk": "Тәулік ішінде операциялар әдеттегіден көп емес",
        "en": "No more transactions in 24 hours than usual",
    },
    "amount_24h_to_avg": {
        "ru": "Суточные траты в пределах нормы",
        "kk": "Тәуліктік шығындар қалыпты шекте",
        "en": "Daily spending within the normal range",
    },
    "is_foreign": {
        "ru": "Операция внутри страны",
        "kk": "Операция ел ішінде",
        "en": "Domestic transaction",
    },
    "distance_from_prev_km": {
        "ru": "Прошлая операция была рядом",
        "kk": "Алдыңғы операция жақын жерде болған",
        "en": "The previous transaction was nearby",
    },
    "travel_speed_kmh": {
        "ru": "Перемещение физически возможно",
        "kk": "Жылжу физикалық тұрғыдан мүмкін",
        "en": "The movement is physically possible",
    },
    "clients_per_recipient": {
        "ru": "Получатель не связан с другими клиентами банка",
        "kk": "Алушы банктің басқа клиенттерімен байланысты емес",
        "en": "The recipient is not linked to other clients of the bank",
    },
    "is_new_recipient": {
        "ru": "Клиент уже переводил этому получателю",
        "kk": "Клиент бұл алушыға бұрын аударым жасаған",
        "en": "The client has sent money to this recipient before",
    },
    "is_new_category": {
        "ru": "Клиент регулярно платит в этой категории",
        "kk": "Клиент бұл санатта үнемі төлейді",
        "en": "The client pays in this category regularly",
    },
    "client_age": {
        "ru": "Возраст клиента вне типичной группы риска",
        "kk": "Клиенттің жасы әдеттегі тәуекел тобынан тыс",
        "en": "The client's age is outside the typical risk group",
    },
    "client_tenure_days": {
        "ru": "Клиент давно с банком",
        "kk": "Клиент банкпен ұзақ уақыттан бері",
        "en": "A long-standing client of the bank",
    },
    "recipient_clients_24h": {
        "ru": "На счёт получателя сегодня почти никто больше не переводил",
        "kk": "Алушының шотына бүгін басқа ешкім дерлік аударым жасамаған",
        "en": "Almost nobody else sent money to this account today",
    },
    "recipient_clients_7d": {
        "ru": "Получатель не собирает переводы от многих людей",
        "kk": "Алушы көп адамнан аударым жинамайды",
        "en": "The recipient is not collecting transfers from many people",
    },
    "recipient_inbound_24h": {
        "ru": "Поток переводов к получателю обычный",
        "kk": "Алушыға түсетін аударымдар ағыны қалыпты",
        "en": "The flow of transfers to the recipient is ordinary",
    },
    "recipient_amount_24h": {
        "ru": "На счёте получателя не скапливаются крупные суммы",
        "kk": "Алушының шотында ірі сомалар жиналмайды",
        "en": "No large sums are piling up in the recipient's account",
    },
    "recipient_age_days": {
        "ru": "Счёт получателя банку давно знаком",
        "kk": "Алушының шоты банкке ежелден таныс",
        "en": "The recipient's account is long known to the bank",
    },
    "recipient_is_new_to_bank": {
        "ru": "Счёт получателя банку уже знаком",
        "kk": "Алушының шоты банкке таныс",
        "en": "The recipient's account is already known to the bank",
    },
    "device_clients_24h": {
        "ru": "Сегодня с устройства платил только этот клиент",
        "kk": "Бүгін құрылғыдан тек осы клиент төледі",
        "en": "Only this client paid from the device today",
    },
    "ip_clients_24h": {
        "ru": "Сегодня с этого IP работал только этот клиент",
        "kk": "Бүгін бұл IP-ден тек осы клиент жұмыс істеді",
        "en": "Only this client operated from this IP today",
    },
    "component_clients": {
        "ru": "Клиент не связан общими устройствами и счетами с другими",
        "kk": "Клиент басқалармен ортақ құрылғылар мен шоттар арқылы байланыспаған",
        "en": "The client shares no devices or accounts with others",
    },
    "component_size": {
        "ru": "Окружение клиента в графе связей небольшое",
        "kk": "Клиенттің байланыс графындағы ортасы шағын",
        "en": "The client's neighbourhood in the link graph is small",
    },
    "has_long_call": {
        "ru": "Длинного разговора перед операцией не было",
        "kk": "Операция алдында ұзақ әңгіме болмаған",
        "en": "No long call preceded the transaction",
    },
    "night_share": {
        "ru": "Ночные операции для клиента обычны",
        "kk": "Түнгі операциялар клиент үшін әдеттегідей",
        "en": "Night-time activity is normal for this client",
    },
})

_GENERIC_SAFE = {
    "ru": "Признак «{f}» говорит в пользу клиента",
    "kk": "«{f}» белгісі клиенттің пайдасына",
    "en": "Feature “{f}” speaks in the client's favour",
}
_GENERIC_RISK = {
    "ru": "Нетипичное значение признака «{f}»",
    "kk": "«{f}» белгісінің әдеттен тыс мәні",
    "en": "Atypical value of feature “{f}”",
}


# --------------------------------------------------------------------------
# Сообщения клиенту
# --------------------------------------------------------------------------

# Главный текст всей системы. Человек в этот момент говорит по телефону с
# мошенником, который представился банком. Формальное «операция отклонена по
# соображениям безопасности» его не спасёт — мошенник тут же объяснит, что
# «банк глючит, переведите через другой». Спасает только прямая фраза.
ANTI_SCAM_WARNING = {
    "ru": (
        "Операция приостановлена на 15 минут. Если прямо сейчас вам звонят из банка, "
        "полиции или «службы безопасности» и просят перевести деньги на «безопасный счёт», "
        "оформить «зеркальный кредит» или назвать код из SMS — это мошенники. "
        "Банк никогда об этом не просит. Положите трубку и перезвоните нам сами по номеру "
        "на обороте карты. Мы свяжемся с вами в течение нескольких минут."
    ),
    "kk": (
        "Операция 15 минутқа тоқтатылды. Егер дәл қазір сізге банктен, полициядан немесе "
        "«қауіпсіздік қызметінен» қоңырау шалып, ақшаны «қауіпсіз шотқа» аударуды, "
        "«айна несие» ресімдеуді немесе SMS кодын айтуды сұрап жатса — бұл алаяқтар. "
        "Банк мұны ешқашан сұрамайды. Тұтқаны қойып, картаның артындағы нөмірмен өзіңіз "
        "қайта қоңырау шалыңыз. Біз сізбен бірнеше минут ішінде байланысамыз."
    ),
    "en": (
        "This transaction is on hold for 15 minutes. If someone is calling you right now "
        "claiming to be from the bank, the police or a “security service” and asking you to "
        "move money to a “safe account”, take out a “mirror loan” or read out an SMS code — "
        "they are criminals. The bank never asks for this. Hang up and call us back yourself "
        "using the number on the back of your card. We will contact you within a few minutes."
    ),
}

CLIENT_MESSAGES: dict[str, dict[str, dict[str, str]]] = {
    STEP_UP: {
        "default": {
            "ru": "Подтвердите операцию на {amount} ₸ в приложении. Если это не вы — нажмите «Это не я», мы заблокируем карту.",
            "kk": "Қосымшада {amount} ₸ операциясын растаңыз. Егер бұл сіз болмасаңыз — «Бұл мен емеспін» батырмасын басыңыз, картаны бұғаттаймыз.",
            "en": "Please confirm the {amount} KZT transaction in the app. If it wasn't you, tap “Not me” and we will block the card.",
        },
    },
    HOLD: {
        "social_eng": ANTI_SCAM_WARNING,
        "default": {
            "ru": "Мы приостановили операцию на {amount} ₸ — она не похожа на ваши обычные. Сейчас позвоним, чтобы подтвердить. Если это не вы, просто скажите оператору.",
            "kk": "{amount} ₸ операциясын тоқтаттық — ол сіздің әдеттегі операцияларыңызға ұқсамайды. Растау үшін қазір қоңырау шаламыз. Егер бұл сіз болмасаңыз, операторға айтыңыз.",
            "en": "We have paused a {amount} KZT transaction — it doesn't look like your usual activity. We're calling you now to confirm. If it wasn't you, just tell the operator.",
        },
    },
    BLOCK: {
        "social_eng": {
            "ru": "Операция на {amount} ₸ отклонена, деньги остались на вашем счёте. Если вас прямо сейчас убеждают перевести их — это мошенники, положите трубку. Карта работает, перезвоните нам сами по номеру на обороте.",
            "kk": "{amount} ₸ операциясы қабылданбады, ақша шотыңызда қалды. Егер сізді дәл қазір аударуға көндіріп жатса — бұл алаяқтар, тұтқаны қойыңыз. Карта жұмыс істейді, картаның артындағы нөмірмен өзіңіз қоңырау шалыңыз.",
            "en": "The {amount} KZT transaction was declined and the money stayed in your account. If someone is pressuring you to send it right now, they are criminals — hang up. Your card still works; call us back on the number on the back of it.",
        },
        "default": {
            "ru": "Операция на {amount} ₸ отклонена: она сильно отличается от ваших обычных. Деньги на месте. Если платили вы — подтвердите в приложении, и мы пропустим.",
            "kk": "{amount} ₸ операциясы қабылданбады: ол сіздің әдеттегі операцияларыңыздан қатты ерекшеленеді. Ақша орнында. Егер төлеген сіз болсаңыз — қосымшада растаңыз, біз өткіземіз.",
            "en": "The {amount} KZT transaction was declined — it differs sharply from your usual activity. Your money is safe. If it was you, confirm in the app and we'll let it through.",
        },
    },
}

OPERATOR_SCRIPT = {
    "social_eng": {
        "ru": "Спросить прямо: «Вам сейчас кто-нибудь звонит и объясняет, что нужно сделать перевод?» "
              "Не подсказывать ответ. Если клиент мнётся или пересказывает чужие слова — удерживать операцию "
              "и переводить на антифрод-линию.",
        "kk": "Тікелей сұрау: «Дәл қазір сізге біреу қоңырау шалып, аударым жасау керектігін түсіндіріп жатыр ма?» "
              "Жауапты айтып бермеу. Клиент күмілжісе немесе біреудің сөзін қайталаса — операцияны ұстап тұрып, "
              "антифрод желісіне ауыстыру.",
        "en": "Ask directly: “Is anyone on the phone with you right now telling you to make this transfer?” "
              "Do not lead the answer. If the client hesitates or repeats someone else's words, keep the hold "
              "and escalate to the anti-fraud line.",
    },
    "stolen_card": {
        "ru": "Подтвердить личность и спросить, где сейчас карта. При отрицании операции — блокировать карту "
              "и перевыпускать.",
        "kk": "Жеке басын растап, картаның қазір қайда екенін сұрау. Клиент операцияны танымаса — картаны бұғаттап, "
              "қайта шығару.",
        "en": "Verify identity and ask where the card physically is. If the client denies the transaction, block "
              "the card and reissue.",
    },
    "fraud_ring": {
        "ru": "Проверить связанные карты по устройству и IP — вероятно, это группа. Передать список в расследование.",
        "kk": "Құрылғы мен IP бойынша байланысты карталарды тексеру — бұл топ болуы мүмкін. Тізімді тергеуге беру.",
        "en": "Check related cards by device and IP — this is likely a group. Hand the list to investigations.",
    },
}


# --------------------------------------------------------------------------
# Объяснитель
# --------------------------------------------------------------------------


@dataclass
class Reason:
    feature: str
    contribution: float
    direction: str           # "risk" или "safe"
    text: dict[str, str]     # ru / kk / en

    def to_dict(self) -> dict:
        return asdict(self)


_LOCAL_TABLES = {
    "CATEGORY_NAMES": CATEGORY_NAMES,
    "CHANNEL_NAMES": CHANNEL_NAMES,
    "TX_TYPE_NAMES": TX_TYPE_NAMES,
}


def _format(template: dict, feats: dict, tx: dict, feature: str) -> dict[str, str]:
    """Собрать фразу на трёх языках."""
    # Значение, которое само переводится (категория, канал, тип операции)
    if "localized" in template:
        tx_field, table_name = template["localized"]
        table = _LOCAL_TABLES[table_name]
        key = str(tx.get(tx_field, "") or "")
        return {
            lang: (template.get(lang) or template.get("ru", "")).replace(
                "{v}", localized(table, key, lang)
            )
            for lang in LANGUAGES
        }

    value_fn = template.get("value")
    fmt = template.get("fmt")
    out: dict[str, str] = {}
    if value_fn is not None and fmt is not None:
        try:
            raw = value_fn(feats, tx)
        except Exception:
            raw = feats.get(feature, 0.0)
        try:
            shown = fmt.format(v=raw)
        except (ValueError, TypeError):
            shown = str(raw)
    else:
        shown = None
    for lang in LANGUAGES:
        text = template.get(lang) or template.get("ru", "")
        out[lang] = text.replace("{v}", shown) if shown is not None else text
    return out


def build_reasons(
    feats: dict[str, float],
    shap_values: dict[str, float],
    tx: dict,
    top_risk: int = 4,
    top_safe: int = 2,
) -> list[Reason]:
    """Перевести вклады SHAP в список причин на человеческом языке."""
    ordered = sorted(shap_values.items(), key=lambda kv: -abs(kv[1]))

    reasons: list[Reason] = []
    risks = 0
    safes = 0
    for feature, contribution in ordered:
        if abs(contribution) < 1e-4:
            continue
        if contribution > 0 and risks < top_risk:
            tpl = RISK_TEMPLATES.get(feature)
            text = (
                _format(tpl, feats, tx, feature)
                if tpl
                else {l: _GENERIC_RISK[l].format(f=feature) for l in LANGUAGES}
            )
            reasons.append(Reason(feature, round(float(contribution), 4), "risk", text))
            risks += 1
        elif contribution < 0 and safes < top_safe:
            tpl = SAFE_TEMPLATES.get(feature)
            text = (
                _format(tpl, feats, tx, feature)
                if tpl
                else {l: _GENERIC_SAFE[l].format(f=feature) for l in LANGUAGES}
            )
            reasons.append(Reason(feature, round(float(contribution), 4), "safe", text))
            safes += 1
        if risks >= top_risk and safes >= top_safe:
            break
    return reasons


def dominant_kind(type_probs: dict[str, float]) -> str:
    if not type_probs:
        return "stolen_card"
    return max(type_probs.items(), key=lambda kv: kv[1])[0]


def client_message(action: str, amount: float, type_probs: dict[str, float]) -> dict[str, str]:
    """Что увидит клиент. Пустая строка означает, что беспокоить его не нужно."""
    if action == ALLOW:
        return {lang: "" for lang in LANGUAGES}
    kind = dominant_kind(type_probs)
    block = CLIENT_MESSAGES.get(action, {})
    template = block.get(kind) or block.get("default") or {}
    shown = f"{amount:,.0f}".replace(",", " ")
    return {lang: (template.get(lang, "")).format(amount=shown) for lang in LANGUAGES}


def operator_script(action: str, type_probs: dict[str, float]) -> dict[str, str]:
    if action in (ALLOW, STEP_UP):
        return {lang: "" for lang in LANGUAGES}
    return OPERATOR_SCRIPT.get(dominant_kind(type_probs), OPERATOR_SCRIPT["stolen_card"])


class Explainer:
    """Обёртка над TreeSHAP.

    Точные вклады считает сам LightGBM (pred_contrib) — это тот же алгоритм
    TreeSHAP, только без лишнего слоя. Библиотека shap используется как основной
    путь, а pred_contrib остаётся запасным, если версии разойдутся по API.
    """

    def __init__(self, model, features: list[str]):
        self.model = model
        self.features = features
        self.booster = getattr(model, "booster_", model)
        self._shap = None
        try:
            import shap  # локальный импорт: без объяснений API всё равно поднимется

            self._shap = shap.TreeExplainer(self.booster)
            self.base_value = float(np.ravel(self._shap.expected_value)[0])
        except Exception:
            self.base_value = 0.0

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Матрица вкладов (n, n_features) в логарифме шансов."""
        X = np.asarray(X, dtype=float)
        if self._shap is not None:
            try:
                values = self._shap.shap_values(X, check_additivity=False)
                values = np.asarray(values)
                if values.ndim == 3:            # (n, features, classes)
                    values = values[..., -1]
                if values.shape[-1] == len(self.features):
                    return values
            except Exception:
                pass
        contrib = np.asarray(self.booster.predict(X, pred_contrib=True))
        return contrib[:, : len(self.features)]

    def explain_one(self, feats: dict[str, float], tx: dict) -> tuple[list[Reason], float]:
        row = np.array([[feats.get(f, 0.0) for f in self.features]], dtype=float)
        # TreeSHAP работает с голым массивом — имена колонок ему не нужны
        contrib = self.contributions(row)[0]
        shap_map = {f: float(contrib[i]) for i, f in enumerate(self.features)}
        return build_reasons(feats, shap_map, tx), float(contrib.sum() + self.base_value)


def reasons_from_stored(
    stored: list[dict], feats: dict[str, float], tx: dict
) -> list[Reason]:
    """Причины по заранее сохранённым вкладам — для ленты, где SHAP уже посчитан."""
    shap_map = {d["feature"]: float(d["contribution"]) for d in stored}
    return build_reasons(feats, shap_map, tx, top_risk=3, top_safe=1)
