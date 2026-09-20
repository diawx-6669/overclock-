"""Контрфакты: что должно было измениться, чтобы решение стало другим.

SHAP отвечает на вопрос «почему система так решила» — показывает, какие
признаки толкнули оценку вверх. Но у оператора и клиента вопрос обычно
другой: «а что надо было сделать иначе». На него вклад признака не отвечает.

Контрфакт отвечает прямо: «до 87 000 ₸ эта операция прошла бы без вопросов»
или «без включённой программы удалённого доступа хватило бы подтверждения в
приложении». Такую фразу можно сказать клиенту вслух, и она не требует от
него понимать, что такое вклад признака в логарифм шансов.

Ищем только по тем полям, которые человек может осознать и изменить: сумма,
VPN, удалённый доступ, звонок перед операцией, устройство, город, получатель.
Перебирать все пятьдесят один признак бессмысленно — «уменьшите долю ночных
операций за последние тридцать дней» не совет, а издевательство.

Каждый кандидат прогоняется через полный тракт, поэтому контрфакты считаются
отдельной ручкой, а не на горячем пути авторизации.
"""

from __future__ import annotations

import math
from typing import Callable

from backend.cost import ACTIONS

# Порядок строгости. Контрфакт интересен, когда действие становится мягче.
SEVERITY = {a: i for i, a in enumerate(ACTIONS)}

# Сколько шагов двоичного поиска по сумме. Двенадцать дают точность около
# 0.03% от исходной суммы — заведомо больше, чем нужно для фразы оператору.
AMOUNT_STEPS = 12


def _text(field: str, value=None) -> dict[str, str]:
    """Человеческая формулировка контрфакта на трёх языках."""
    if field == "amount":
        shown = f"{value:,.0f}".replace(",", " ")
        return {
            "ru": f"Если бы сумма не превышала {shown} ₸, решение было бы другим",
            "kk": f"Егер сома {shown} ₸ аспаса, шешім басқа болар еді",
            "en": f"Had the amount stayed under {shown} KZT, the decision would differ",
        }
    table = {
        "is_vpn": {
            "ru": "Без VPN решение было бы другим",
            "kk": "VPN-сіз шешім басқа болар еді",
            "en": "Without VPN the decision would differ",
        },
        "remote_access": {
            "ru": "Без активной программы удалённого доступа решение было бы другим",
            "kk": "Қашықтан қол жеткізу бағдарламасы белсенді болмаса, шешім басқа болар еді",
            "en": "Without active remote-access software the decision would differ",
        },
        "call_minutes_before": {
            "ru": "Без телефонного разговора перед операцией решение было бы другим",
            "kk": "Операция алдында телефон әңгімесі болмаса, шешім басқа болар еді",
            "en": "Without the phone call before the transaction the decision would differ",
        },
        "device_id": {
            "ru": "С привычного устройства клиента решение было бы другим",
            "kk": "Клиенттің әдеттегі құрылғысынан шешім басқа болар еді",
            "en": "From the client's usual device the decision would differ",
        },
        "city": {
            "ru": "Из домашнего города клиента решение было бы другим",
            "kk": "Клиенттің үй қаласынан шешім басқа болар еді",
            "en": "From the client's home city the decision would differ",
        },
        "recipient_id": {
            "ru": "Получателю, которому клиент уже переводил, решение было бы другим",
            "kk": "Клиент бұрын аударым жасаған алушыға шешім басқа болар еді",
            "en": "To a recipient the client has paid before the decision would differ",
        },
    }
    return table.get(field, {lang: field for lang in ("ru", "kk", "en")})


def _candidates(tx: dict, client: dict | None) -> list[tuple[str, object, object]]:
    """Какие одиночные изменения вообще имеет смысл проверять.

    Возвращает (поле, было, стало). Проверяем только то, что в этой операции
    действительно стоит «неудобно»: предлагать выключить VPN там, где его и
    так нет, — пустая строка в объяснении.
    """
    out: list[tuple[str, object, object]] = []

    if float(tx.get("is_vpn") or 0):
        out.append(("is_vpn", 1, 0))
    if float(tx.get("remote_access") or 0):
        out.append(("remote_access", 1, 0))
    if float(tx.get("call_minutes_before") or 0) > 0:
        out.append(("call_minutes_before", tx["call_minutes_before"], 0))

    devices = (client or {}).get("devices") or []
    if devices and tx.get("device_id") not in devices:
        out.append(("device_id", tx.get("device_id"), devices[0]))

    home = (client or {}).get("home_city")
    if home and tx.get("city") != home:
        out.append(("city", tx.get("city"), home))

    return out


NOTES = {
    "already_softest": {
        "ru": "Операция и так пропускается — смягчать нечего",
        "kk": "Операция әлдеқашан өткізіледі — жұмсартатын ештеңе жоқ",
        "en": "The transaction is already allowed — nothing to soften",
    },
    "nothing_helps": {
        "ru": "Ни одно отдельное изменение не смягчает решение: даже при минимальной "
              "сумме подтверждение остаётся дешевле пропуска, потому что разбор "
              "случая мошенничества стоит одинаково независимо от суммы",
        "kk": "Бірде-бір жеке өзгеріс шешімді жұмсартпайды: ең аз сомада да растау "
              "өткізуден арзан болып қалады, өйткені алаяқтық жағдайын талдау сомаға "
              "қарамастан бірдей тұрады",
        "en": "No single change softens the decision: even at a minimal amount a "
              "confirmation stays cheaper than allowing, because handling a fraud "
              "case costs the same regardless of the amount",
    },
}


def find_counterfactuals(
    tx: dict,
    score: Callable[[dict], dict],
    baseline: dict | None = None,
    max_items: int = 3,
) -> dict:
    """Найти изменения, при которых система решила бы мягче.

    `score` — функция, прогоняющая транзакцию через полный тракт и
    возвращающая результат скоринга. Контрфакты считаются по тому же тракту,
    что и само решение: иначе объяснение описывало бы не ту систему, которая
    приняла решение.
    """
    base = baseline or score(tx)
    base_action = base["decision"]["action"]
    base_rank = SEVERITY[base_action]
    if base_rank == 0:                       # уже «пропустить», смягчать нечего
        return {"items": [], "note": NOTES["already_softest"]}

    client = base.get("client") or {}
    found: list[dict] = []

    # --- одиночные переключения
    for field, was, becomes in _candidates(tx, client):
        probe = dict(tx)
        probe[field] = becomes
        result = score(probe)
        action = result["decision"]["action"]
        if SEVERITY[action] < base_rank:
            found.append({
                "field": field,
                "from": was,
                "to": becomes,
                "action_from": base_action,
                "action_to": action,
                "p_from": base["p_fraud"],
                "p_to": result["p_fraud"],
                "text": _text(field),
            })

    # --- сумма: ищем границу, а не перебираем наугад
    amount = float(tx.get("amount") or 0)
    if amount > 0:
        lo, hi = amount * 0.01, amount      # при lo решение почти наверняка мягче
        probe = dict(tx)
        probe["amount"] = lo
        if SEVERITY[score(probe)["decision"]["action"]] < base_rank:
            # двоичный поиск по логарифму: шаг по сумме не линейный по смыслу
            log_lo, log_hi = math.log(lo), math.log(hi)
            best = lo
            best_action = base_action
            for _ in range(AMOUNT_STEPS):
                mid = math.exp((log_lo + log_hi) / 2)
                probe["amount"] = mid
                action = score(probe)["decision"]["action"]
                if SEVERITY[action] < base_rank:
                    best, best_action = mid, action
                    log_lo = math.log(mid)
                else:
                    log_hi = math.log(mid)
            probe["amount"] = best
            final = score(probe)
            found.append({
                "field": "amount",
                "from": amount,
                "to": round(best, -2),
                "action_from": base_action,
                "action_to": best_action,
                "p_from": base["p_fraud"],
                "p_to": final["p_fraud"],
                "text": _text("amount", round(best, -2)),
            })

    # Сначала те, что смягчают сильнее; при равенстве — где ниже вероятность
    found.sort(key=lambda c: (SEVERITY[c["action_to"]], c["p_to"]))
    # Пустой ответ без объяснения бесполезен оператору: он не понимает, то ли
    # система не искала, то ли искала и не нашла.
    return {
        "items": found[:max_items],
        "note": None if found else NOTES["nothing_helps"],
    }
