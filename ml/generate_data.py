"""Генератор синтетического потока транзакций для Fraud Hunter.

Реального датасета у нас нет, поэтому месяц работы финтеха мы собираем сами:
3000 клиентов со своими привычками и 100 000 транзакций, из которых ~1.8% — фрод.

Ключевая мысль генератора: честные клиенты тоже «шумные». Они ездят в отпуск,
покупают новые телефоны, платят ночью, изредка тратят десять своих средних чеков
и переводят деньги незнакомым людям. Если бы честный поток был гладким, задача
решалась бы тремя if-ами и никакая модель была бы не нужна.

Запуск:
    python -m ml.generate_data --out data/transactions.csv
"""

from __future__ import annotations

import argparse
import math
import zlib
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ml.common import (
    CATEGORY_LIST,
    CITIES,
    FOREIGN_CITIES,
    MERCHANT_CATEGORIES,
    place_coords,
)

# --------------------------------------------------------------------------
# Параметры мира
# --------------------------------------------------------------------------

N_CLIENTS = 3000
N_TRANSACTIONS = 100_000
FRAUD_RATE = 0.018

# Месяц наблюдения
PERIOD_START = pd.Timestamp("2025-09-01 00:00:00")
PERIOD_DAYS = 30

# Распределение фрода по типам
FRAUD_MIX = {"stolen_card": 0.45, "social_eng": 0.35, "fraud_ring": 0.20}

# Доля «тихих» краж внутри stolen_card: та же страна, без VPN, мелкие суммы
QUIET_STOLEN_SHARE = 0.35


@dataclass
class Client:
    client_id: str
    age: int
    home_city: str
    avg_ticket: float          # средний чек, тенге
    ticket_sigma: float        # разброс чека (лог-нормальный)
    monthly_tx: int            # сколько транзакций делает за месяц
    primary_device: str
    secondary_device: str | None
    night_prop: float          # доля ночных операций
    travel_prop: float         # доля операций из другого города
    vpn_user: bool             # клиент регулярно сидит через VPN
    tenure_days: int           # сколько дней клиент с нами
    home_ip_prefix: str


# --------------------------------------------------------------------------
# Клиенты
# --------------------------------------------------------------------------


def make_clients(rng: np.random.Generator, n: int = N_CLIENTS) -> list[Client]:
    city_names = list(CITIES)
    city_weights = np.array([CITIES[c][2] for c in city_names], dtype=float)
    city_weights /= city_weights.sum()

    clients: list[Client] = []
    for i in range(n):
        age = int(np.clip(rng.normal(38, 14), 18, 85))
        home_city = str(rng.choice(city_names, p=city_weights))

        # Средний чек: лог-нормальный, медиана ~12 000 ₸, тяжёлый правый хвост
        avg_ticket = float(np.exp(rng.normal(math.log(12_000), 0.65)))
        avg_ticket = float(np.clip(avg_ticket, 1_500, 250_000))

        # Молодёжь платит чаще и мельче, старшие — реже и крупнее
        activity_base = 45 if age < 30 else (35 if age < 55 else 22)
        monthly_tx = int(np.clip(rng.poisson(activity_base), 3, 200))

        has_second = rng.random() < 0.30
        clients.append(
            Client(
                client_id=f"C{i:05d}",
                age=age,
                home_city=home_city,
                avg_ticket=avg_ticket,
                ticket_sigma=float(rng.uniform(0.45, 0.95)),
                monthly_tx=monthly_tx,
                primary_device=f"D{i:05d}A",
                secondary_device=f"D{i:05d}B" if has_second else None,
                night_prop=float(np.clip(rng.beta(1.6, 12), 0.0, 0.5)),
                travel_prop=float(np.clip(rng.beta(1.5, 12), 0.0, 0.6)),
                vpn_user=bool(rng.random() < 0.10),
                tenure_days=int(rng.integers(20, 3000)),
                home_ip_prefix=f"{rng.integers(2, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}",
            )
        )
    return clients


# --------------------------------------------------------------------------
# Вспомогательные генераторы
# --------------------------------------------------------------------------


def _timestamp(rng: np.random.Generator, night: bool) -> pd.Timestamp:
    """Момент операции внутри месяца с суточным ритмом."""
    day = int(rng.integers(0, PERIOD_DAYS))
    if night:
        hour = int(rng.choice([0, 1, 2, 3, 4, 5]))
    else:
        # два горба: обед и вечер
        hour = int(np.clip(round(rng.choice([12, 13, 19, 20]) + rng.normal(0, 2.4)), 6, 23))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return PERIOD_START + pd.Timedelta(days=day, hours=hour, minutes=minute, seconds=second)


def _amount(rng: np.random.Generator, client: Client, category: str) -> float:
    """Сумма покупки: средний чек клиента × профиль категории × шум."""
    mult = MERCHANT_CATEGORIES[category][0]
    raw = client.avg_ticket * mult * float(np.exp(rng.normal(0, client.ticket_sigma)))
    return float(np.clip(round(raw, -1), 200, 40_000_000))


def _human_round(rng: np.random.Generator, amount: float, tx_type: str) -> float:
    """Люди переводят круглыми суммами.

    Это важнее, чем кажется. Если округлять до десятков только у мошенников,
    признак «круглая сумма» в одиночку выдаёт весь фрод, модель цепляется
    за него и на реальных данных рассыпается. В жизни человек отправляет
    другу ровно 50 000, а банкомат вообще не умеет выдавать 43 271.
    """
    if tx_type == "withdrawal":
        step = int(rng.choice([5_000, 10_000, 20_000]))
        return float(max(step, round(amount / step) * step))
    if tx_type == "transfer" and rng.random() < 0.55:
        step = int(rng.choice([1_000, 5_000, 10_000, 50_000], p=[0.3, 0.3, 0.3, 0.1]))
        return float(max(step, round(amount / step) * step))
    return float(round(amount, -2))


def _pick_category(rng: np.random.Generator, client: Client) -> str:
    """Честные покупки смещены в бытовые категории."""
    weights = {
        "grocery": 26, "transport": 14, "cafe": 12, "pharmacy": 7, "utilities": 8,
        "marketplace": 11, "clothing": 6, "electronics": 2.0, "entertainment": 5,
        "travel": 1.5, "jewelry": 0.4, "crypto": 0.5, "gambling": 0.6,
        "atm_withdrawal": 3.5, "transfer_p2p": 6.0,
    }
    if client.age > 55:
        weights["crypto"] = 0.1
        weights["gambling"] = 0.1
        weights["utilities"] = 14
        weights["pharmacy"] = 12
    names = list(weights)
    p = np.array([weights[c] for c in names], dtype=float)
    p /= p.sum()
    return str(rng.choice(names, p=p))


def _channel(rng: np.random.Generator, category: str) -> tuple[str, str]:
    """Канал и тип операции по категории."""
    if category == "atm_withdrawal":
        return "atm", "withdrawal"
    if category == "transfer_p2p":
        return str(rng.choice(["mobile_app", "p2p"], p=[0.7, 0.3])), "transfer"
    if category in ("grocery", "transport", "cafe", "pharmacy"):
        return str(rng.choice(["pos", "mobile_app"], p=[0.82, 0.18])), "purchase"
    return str(rng.choice(["ecom", "mobile_app", "pos"], p=[0.55, 0.28, 0.17])), "purchase"


def _ip(rng: np.random.Generator, client: Client, foreign: bool = False) -> str:
    if foreign:
        return f"{rng.integers(2, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    return f"{client.home_ip_prefix}.{rng.integers(1, 254)}"


def _blank_tx() -> dict:
    """Пустая транзакция со всеми полями — чтобы схема была единой."""
    return {
        "tx_id": "",
        "timestamp": PERIOD_START,
        "client_id": "",
        "amount": 0.0,
        "merchant_category": "grocery",
        "merchant_id": "",
        "channel": "pos",
        "tx_type": "purchase",
        "device_id": "",
        "ip": "",
        "city": "",
        "country": "KZ",
        "is_vpn": 0,
        "remote_access": 0,          # активна программа удалённого доступа
        "call_minutes_before": 0.0,  # минут телефонного разговора перед операцией
        "session_duration_s": 0.0,
        "recipient_id": "",
        "is_fraud": 0,
        "fraud_type": "",
    }


# --------------------------------------------------------------------------
# Честный поток
# --------------------------------------------------------------------------


def make_legit(rng: np.random.Generator, clients: list[Client], n_target: int) -> list[dict]:
    """Честные транзакции. Намеренно шумные — иначе задача становится игрушечной."""
    # Раскладываем n_target операций между клиентами пропорционально их активности
    weights = np.array([c.monthly_tx for c in clients], dtype=float)
    weights /= weights.sum()
    counts = rng.multinomial(n_target, weights)

    rows: list[dict] = []
    for client, k in zip(clients, counts):
        if k == 0:
            continue
        # Клиент, который в этом месяце уехал в отпуск
        on_trip = rng.random() < 0.12
        trip_city = str(rng.choice(list(CITIES))) if on_trip else None
        trip_abroad = on_trip and rng.random() < 0.35
        if trip_abroad:
            trip_city = str(rng.choice(list(FOREIGN_CITIES)))
        # Клиент, который в этом месяце сменил телефон
        changed_phone = rng.random() < 0.14
        new_device = f"{client.primary_device}N" if changed_phone else None
        switch_at = int(k * rng.uniform(0.3, 0.8)) if changed_phone else k + 1

        for j in range(int(k)):
            tx = _blank_tx()
            category = _pick_category(rng, client)
            channel, tx_type = _channel(rng, category)
            night = rng.random() < client.night_prop
            ts = _timestamp(rng, night)

            # --- устройство
            if j >= switch_at and new_device:
                device = new_device
            elif client.secondary_device and rng.random() < 0.22:
                device = client.secondary_device
            elif rng.random() < 0.05:
                # чужой ноутбук, новый браузер, рабочий компьютер — устройство,
                # которого банк раньше не видел, но человек честный
                device = f"{client.primary_device}T{rng.integers(0, 999):03d}"
            else:
                device = client.primary_device

            # --- гео
            city = client.home_city
            country = "KZ"
            if on_trip and rng.random() < 0.55:
                city = trip_city or client.home_city
                country = FOREIGN_CITIES.get(city, (0, 0, "KZ"))[2]
            elif rng.random() < client.travel_prop:
                city = str(rng.choice(list(CITIES)))

            # --- сумма: изредка честный клиент делает крупную покупку
            amount = _amount(rng, client, category)
            if rng.random() < 0.025:
                amount = amount * rng.uniform(5, 20)
            amount = _human_round(rng, amount, tx_type)

            # --- VPN: часть людей просто всегда под VPN
            vpn = (rng.random() < 0.60) if client.vpn_user else (rng.random() < 0.03)

            # --- честные тоже переводят незнакомым и держат TeamViewer на ноутбуке
            recipient = ""
            if tx_type == "transfer":
                # crc32, а не hash(): встроенный hash от строки солится заново
                # в каждом процессе (PYTHONHASHSEED), и датасет с одним и тем же
                # сидом получался разным от запуска к запуску
                recipient = (
                    f"R{rng.integers(0, 40_000):05d}"
                    if rng.random() < 0.45
                    else f"R{zlib.crc32(client.client_id.encode()) % 40_000:05d}"
                )
            # Удалённый доступ бывает и у честных: сын настраивает матери
            # приложение через AnyDesk — и это выглядит ровно как мошенничество
            remote_p = 0.075 if client.age > 55 else 0.03
            remote = int(rng.random() < remote_p)
            # Перед переводом люди созваниваются с тем, кому переводят
            if tx_type == "transfer":
                call_before = float(round(rng.uniform(1, 60), 1)) if rng.random() < 0.38 else 0.0
            else:
                call_before = float(round(rng.uniform(1, 25), 1)) if rng.random() < 0.05 else 0.0

            # --- трудные отрицательные примеры
            # Примерно раз на полсотни операций честный клиент выглядит ровно
            # так, как в учебнике описан мошенник. Без таких примеров модель
            # выучивает не фрод, а его декорации, и в проде блокирует
            # командировки и помощь родителям.
            if rng.random() < 0.009:
                if rng.random() < 0.5:
                    # человек улетел в командировку с новым телефоном
                    city = str(rng.choice(list(FOREIGN_CITIES)))
                    country = FOREIGN_CITIES[city][2]
                    device = f"{client.primary_device}T{rng.integers(0, 999):03d}"
                    vpn = True
                    category = str(rng.choice(["electronics", "travel", "marketplace", "clothing"]))
                    channel, tx_type = _channel(rng, category)
                    amount = _human_round(rng, client.avg_ticket * rng.uniform(3, 12), tx_type)
                else:
                    # человек сам, добровольно и в трезвом уме переводит крупную
                    # круглую сумму незнакомому получателю после долгого звонка:
                    # покупка машины с рук, ремонт, помощь родственнику
                    category, tx_type = "transfer_p2p", "transfer"
                    channel = "mobile_app"
                    amount = _human_round(rng, client.avg_ticket * rng.uniform(4, 20), "transfer")
                    recipient = f"R{rng.integers(0, 40_000):05d}"
                    call_before = float(round(rng.uniform(5, 70), 1))
                    remote = int(rng.random() < 0.15)
                    city = client.home_city
                    country = "KZ"
                    device = client.primary_device
                    vpn = False

            lat, lon = place_coords(city)
            tx.update(
                timestamp=ts,
                client_id=client.client_id,
                amount=amount,
                merchant_category=category,
                merchant_id=f"M{rng.integers(0, 5000):04d}",
                channel=channel,
                tx_type=tx_type,
                device_id=device,
                ip=_ip(rng, client, foreign=(country != "KZ")),
                city=city,
                country=country,
                is_vpn=int(vpn),
                remote_access=remote,
                call_minutes_before=call_before,
                session_duration_s=float(round(abs(rng.normal(70, 45)) + 8, 1)),
                recipient_id=recipient,
                is_fraud=0,
                fraud_type="",
            )
            rows.append(tx)
    return rows


# --------------------------------------------------------------------------
# Фрод №1: украденная карта — «чужой платит за клиента»
# --------------------------------------------------------------------------


def make_stolen_card(rng: np.random.Generator, clients: list[Client], n_target: int) -> list[dict]:
    rows: list[dict] = []
    victims = rng.choice(len(clients), size=max(1, n_target // 3), replace=False)

    produced = 0
    for vi in victims:
        if produced >= n_target:
            break
        client = clients[int(vi)]
        quiet = rng.random() < QUIET_STOLEN_SHARE
        burst = int(rng.integers(1, 4)) if quiet else int(rng.integers(2, 6))
        burst = min(burst, n_target - produced)

        attacker_device = f"DX{rng.integers(0, 99999):05d}"
        attacker_ip_prefix = f"{rng.integers(2, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}"
        start = _timestamp(rng, night=rng.random() < (0.35 if quiet else 0.6))

        if quiet:
            # «Тихая» кража: своя страна, без VPN, суммы почти как у клиента.
            # Такие операции не ловятся правилом «заграница + новое устройство».
            city = client.home_city if rng.random() < 0.55 else str(rng.choice(list(CITIES)))
            country = "KZ"
            vpn_p = 0.05
            amount_mult = (0.8, 2.5)
            cats = ["marketplace", "grocery", "clothing", "entertainment", "transport", "cafe"]
        else:
            if rng.random() < 0.55:
                city = str(rng.choice(list(FOREIGN_CITIES)))
                country = FOREIGN_CITIES[city][2]
            else:
                far = [c for c in CITIES if c != client.home_city]
                city = str(rng.choice(far))
                country = "KZ"
            vpn_p = 0.55
            amount_mult = (3.0, 15.0)
            cats = ["electronics", "crypto", "gambling", "jewelry", "atm_withdrawal", "marketplace", "travel"]

        for b in range(burst):
            tx = _blank_tx()
            category = str(rng.choice(cats))
            channel, tx_type = _channel(rng, category)
            gap = rng.uniform(2, 45) if not quiet else rng.uniform(40, 900)
            ts = start + pd.Timedelta(minutes=float(gap) * b)
            amount = _human_round(rng, client.avg_ticket * rng.uniform(*amount_mult), tx_type)
            amount = float(np.clip(amount, 500, 40_000_000))
            lat, lon = place_coords(city)
            tx.update(
                timestamp=ts,
                client_id=client.client_id,
                amount=amount,
                merchant_category=category,
                merchant_id=f"M{rng.integers(0, 5000):04d}",
                channel=channel,
                tx_type=tx_type,
                device_id=attacker_device,
                ip=f"{attacker_ip_prefix}.{rng.integers(1, 254)}",
                city=city,
                country=country,
                is_vpn=int(rng.random() < vpn_p),
                remote_access=0,
                call_minutes_before=0.0,
                session_duration_s=float(round(abs(rng.normal(58, 38)) + 5, 1)),
                recipient_id=f"R{rng.integers(0, 40_000):05d}" if tx_type == "transfer" else "",
                is_fraud=1,
                fraud_type="stolen_card",
            )
            rows.append(tx)
            produced += 1
            if produced >= n_target:
                break
    return rows


# --------------------------------------------------------------------------
# Фрод №2: социальная инженерия — «клиента обманули, и он платит сам»
# --------------------------------------------------------------------------


def make_social_eng(rng: np.random.Generator, clients: list[Client], n_target: int) -> list[dict]:
    """Операция идёт со своего телефона, из своего города, без VPN.

    Обычные антифрод-правила её не видят: по всем формальным признакам это
    сам клиент. Отличают такую операцию поведенческие следы — длинный звонок
    перед переводом, включённая программа удалённого доступа, круглая сумма,
    новый получатель, нетипично быстрая сессия и возраст жертвы.
    """
    rows: list[dict] = []

    # Жертвы смещены в сторону старшего возраста, но не только они
    ages = np.array([c.age for c in clients], dtype=float)
    p = np.clip((ages - 20) / 60.0, 0.05, 1.0) ** 2
    p /= p.sum()
    victims = rng.choice(len(clients), size=max(1, int(n_target / 1.8)), replace=False, p=p)

    produced = 0
    for vi in victims:
        if produced >= n_target:
            break
        client = clients[int(vi)]
        # Мошенник часто дробит перевод на несколько частей
        parts = int(rng.choice([1, 2, 3, 4], p=[0.45, 0.28, 0.17, 0.10]))
        parts = min(parts, n_target - produced)

        # «Безопасный счёт», на который уговорили перевести
        mule = f"R{rng.integers(0, 40_000):05d}"
        # Не в каждом случае жертва говорит по телефону в момент операции и не
        # всегда мошенник ставит программу удалённого доступа. Часть схем идёт
        # через мессенджер и «инструкцию в переписке» — и следов почти нет.
        call_len = float(round(rng.uniform(8, 95), 1)) if rng.random() < 0.62 else 0.0
        remote = int(rng.random() < 0.38)
        start = _timestamp(rng, night=rng.random() < 0.12)
        total_mult = rng.uniform(3.0, 25.0)

        for b in range(parts):
            tx = _blank_tx()
            category = str(rng.choice(["transfer_p2p", "atm_withdrawal"], p=[0.85, 0.15]))
            channel, tx_type = _channel(rng, category)
            amount = client.avg_ticket * total_mult / parts
            amount = float(np.clip(_human_round(rng, amount, tx_type), 5_000, 40_000_000))
            ts = start + pd.Timedelta(minutes=float(rng.uniform(3, 25)) * b)
            lat, lon = place_coords(client.home_city)
            tx.update(
                timestamp=ts,
                client_id=client.client_id,
                amount=amount,
                merchant_category=category,
                merchant_id=f"M{rng.integers(0, 5000):04d}",
                channel=channel,
                tx_type=tx_type,
                # своё устройство, свой город, свой интернет — вот в чём сложность
                device_id=client.primary_device,
                ip=_ip(rng, client),
                city=client.home_city,
                country="KZ",
                is_vpn=int(rng.random() < 0.02),
                remote_access=remote,
                call_minutes_before=call_len if b == 0 else max(0.0, call_len - b * 5),
                session_duration_s=float(round(abs(rng.normal(65, 42)) + 10, 1)),
                recipient_id=mule if category == "transfer_p2p" else "",
                is_fraud=1,
                fraud_type="social_eng",
            )
            rows.append(tx)
            produced += 1
            if produced >= n_target:
                break
    return rows


# --------------------------------------------------------------------------
# Фрод №3: кольцо карт
# --------------------------------------------------------------------------


def make_fraud_ring(rng: np.random.Generator, clients: list[Client], n_target: int) -> list[dict]:
    """Кольца из 6–15 карт, работающие с одного устройства и одного IP."""
    rows: list[dict] = []
    produced = 0
    ring_idx = 0
    used: set[int] = set()

    while produced < n_target:
        size = int(rng.integers(6, 16))
        pool = [i for i in range(len(clients)) if i not in used]
        if len(pool) < size:
            break
        members = rng.choice(pool, size=size, replace=False)
        used.update(int(m) for m in members)

        ring_device = f"DR{ring_idx:03d}"
        ring_ip = f"{rng.integers(2, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
        ring_recipient = f"R{rng.integers(0, 40_000):05d}"
        ring_city = str(rng.choice(list(CITIES)))
        ring_start = _timestamp(rng, night=rng.random() < 0.5)
        ring_idx += 1

        for m in members:
            client = clients[int(m)]
            # Часть участников заходит со своих устройств и своих IP —
            # кольцо редко бывает настолько дисциплинированным
            member_device = ring_device if rng.random() < 0.72 else client.primary_device
            member_ip = (
                ring_ip
                if rng.random() < 0.62
                else f"{client.home_ip_prefix}.{rng.integers(1, 254)}"
            )
            k = int(rng.integers(1, 4))
            for b in range(k):
                if produced >= n_target:
                    break
                tx = _blank_tx()
                category = str(rng.choice(["transfer_p2p", "marketplace", "crypto", "atm_withdrawal"],
                                          p=[0.45, 0.25, 0.2, 0.1]))
                channel, tx_type = _channel(rng, category)
                amount = float(np.clip(
                    _human_round(rng, client.avg_ticket * rng.uniform(1.5, 6.0), tx_type),
                    1_000, 40_000_000))
                ts = ring_start + pd.Timedelta(minutes=float(rng.uniform(1, 240)))
                lat, lon = place_coords(ring_city)
                tx.update(
                    timestamp=ts,
                    client_id=client.client_id,
                    amount=amount,
                    merchant_category=category,
                    merchant_id=f"M{rng.integers(0, 5000):04d}",
                    channel=channel,
                    tx_type=tx_type,
                    device_id=member_device,
                    ip=member_ip,
                    city=ring_city,
                    country="KZ",
                    is_vpn=int(rng.random() < 0.35),
                    remote_access=0,
                    call_minutes_before=0.0,
                    session_duration_s=float(round(abs(rng.normal(52, 32)) + 5, 1)),
                    recipient_id=ring_recipient if tx_type == "transfer" else "",
                    is_fraud=1,
                    fraud_type="fraud_ring",
                )
                rows.append(tx)
                produced += 1
    return rows


# --------------------------------------------------------------------------
# Сборка
# --------------------------------------------------------------------------


def generate(
    n_clients: int = N_CLIENTS,
    n_tx: int = N_TRANSACTIONS,
    fraud_rate: float = FRAUD_RATE,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (транзакции, профили клиентов)."""
    rng = np.random.default_rng(seed)
    clients = make_clients(rng, n_clients)

    n_fraud = int(round(n_tx * fraud_rate))
    n_legit = n_tx - n_fraud

    rows = make_legit(rng, clients, n_legit)
    rows += make_stolen_card(rng, clients, int(round(n_fraud * FRAUD_MIX["stolen_card"])))
    rows += make_social_eng(rng, clients, int(round(n_fraud * FRAUD_MIX["social_eng"])))
    rows += make_fraud_ring(rng, clients, int(round(n_fraud * FRAUD_MIX["fraud_ring"])))

    df = pd.DataFrame(rows)
    # Ровно до секунды. Иначе часть строк уходит в CSV с наносекундами, часть без,
    # и pandas, угадав формат по первой строке, спотыкается на остальных.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.floor("s")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["tx_id"] = [f"T{i:07d}" for i in range(len(df))]

    # Координаты города — пригодятся карте на фронте
    coords = df["city"].map(lambda c: place_coords(c))
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]

    profiles = pd.DataFrame([asdict(c) for c in clients])
    return df, profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор данных Fraud Hunter")
    parser.add_argument("--out", default="data/transactions.csv")
    parser.add_argument("--profiles-out", default="data/clients.csv")
    parser.add_argument("--clients", type=int, default=N_CLIENTS)
    parser.add_argument("--transactions", type=int, default=N_TRANSACTIONS)
    parser.add_argument("--fraud-rate", type=float, default=FRAUD_RATE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, profiles = generate(args.clients, args.transactions, args.fraud_rate, args.seed)
    df.to_csv(args.out, index=False)
    profiles.to_csv(args.profiles_out, index=False)

    total = len(df)
    fraud = int(df["is_fraud"].sum())
    print(f"Транзакций:      {total:,}")
    print(f"Клиентов:        {profiles.shape[0]:,}")
    print(f"Фрод:            {fraud:,} ({fraud / total:.2%})")
    print("По типам:")
    for t, c in df.loc[df.is_fraud == 1, "fraud_type"].value_counts().items():
        print(f"  {t:<14} {c:>6,}  ({c / fraud:.0%})")
    print(f"Период:          {df.timestamp.min()} .. {df.timestamp.max()}")
    print(f"Оборот:          {df.amount.sum():,.0f} ₸")
    print(f"Файлы:           {args.out}, {args.profiles_out}")


if __name__ == "__main__":
    main()
