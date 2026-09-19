"""Решение по ожидаемым потерям — ядро Fraud Hunter.

Обычный антифрод устроен так: есть вероятность фрода и есть порог. Выше порога
блокируем, ниже пропускаем. Порог подбирают один на всех, и он всегда неверный:
для перевода на 3 000 000 ₸ он слишком мягкий, для покупки кофе за 900 ₸ —
слишком жёсткий.

Мы не выбираем порог. Для каждой транзакции считаем, во сколько тенге обойдётся
каждое из четырёх действий, и берём самое дешёвое:

    ПРОПУСТИТЬ    — ничего не делаем
    ПОДТВЕРДИТЬ   — push/3-D Secure, клиент подтверждает сам
    ЗАДЕРЖАТЬ     — пауза и живой звонок оператора по антифрод-скрипту
    ЗАБЛОКИРОВАТЬ — операция не проходит

Главное здесь — матрица эффективности. Подтверждение в push отлично останавливает
чужого человека с украденной картой: у него нет телефона клиента. И почти не
работает против социальной инженерии: там клиент сам, добровольно, нажимает
«подтвердить» — его же убедили, что так надо. Поэтому против обмана работает
только пауза и живой оператор. Ни один фиксированный порог такую разницу
выразить не может, а разница в деньгах — решающая.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Действия
# --------------------------------------------------------------------------

ALLOW = "allow"
STEP_UP = "step_up"
HOLD = "hold"
BLOCK = "block"

ACTIONS: list[str] = [ALLOW, STEP_UP, HOLD, BLOCK]

ACTION_LABELS_RU: dict[str, str] = {
    ALLOW: "Пропустить",
    STEP_UP: "Запросить подтверждение",
    HOLD: "Задержать и позвонить",
    BLOCK: "Заблокировать",
}

FRAUD_KINDS: list[str] = ["stolen_card", "social_eng", "fraud_ring"]


# --------------------------------------------------------------------------
# Экономика. Все суммы в тенге.
# --------------------------------------------------------------------------


@dataclass
class Economics:
    """Стоимостная модель. Все числа настраиваются — это параметры бизнеса,
    а не константы кода, и их видно на дашборде."""

    # --- потери при пропущенном фроде
    # Доля суммы, которую банк реально не возвращает. Перевод почти не отозвать,
    # покупку по карте иногда спасает chargeback.
    lgd_transfer: float = 0.92
    lgd_purchase: float = 0.60
    lgd_withdrawal: float = 1.00
    # Разбор одного случая: поддержка, расследование, бумаги
    case_handling_cost: float = 5_000.0

    # --- операционные расходы на само действие
    step_up_op_cost: float = 30.0      # push/SMS и обработка
    hold_op_cost: float = 1_500.0      # минуты оператора плюс очередь

    # --- цена трения для честного клиента
    # Трение считаем из двух частей. Первая — прямая потеря выручки: часть людей
    # бросает оплату на полпути. Вторая — ущерб отношениям, и он обязан расти
    # вместе с суммой: задержать оплату кофе и задержать платёж за квартиру —
    # это совершенно разная степень неудобства для человека.
    step_up_abandon_rate: float = 0.04
    step_up_annoyance: float = 120.0
    step_up_friction_rate: float = 0.0015
    hold_abandon_rate: float = 0.18
    hold_annoyance: float = 1_200.0
    hold_friction_rate: float = 0.006
    # Блокировка честной операции: потерянная маржа плюс риск, что клиент уйдёт
    merchant_margin: float = 0.015     # доля от суммы, которую банк зарабатывает
    block_churn_rate: float = 0.11     # вероятность, что клиент уйдёт после ложной блокировки
    clv_cap: float = 1_200_000.0       # потолок пожизненной ценности клиента
    clv_default: float = 180_000.0

    # --- эффективность действия против каждого вида фрода
    # (вероятность, что действие остановит именно эту схему)
    effectiveness: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            ALLOW: {"stolen_card": 0.00, "social_eng": 0.00, "fraud_ring": 0.00},
            # push не доходит до чужого человека — кража останавливается;
            # обманутый клиент подтверждает сам — почти не помогает
            STEP_UP: {"stolen_card": 0.85, "social_eng": 0.10, "fraud_ring": 0.55},
            # живой оператор задаёт вопрос «вам сейчас кто-то звонит и диктует?» —
            # единственное, что реально разрывает социальную инженерию
            HOLD: {"stolen_card": 0.93, "social_eng": 0.80, "fraud_ring": 0.85},
            # Блокировка останавливает конкретную операцию, но не всегда схему.
            # Чужого человека с картой она отсекает начисто. А жертва обмана
            # остаётся под влиянием мошенника: ей просто скажут «банк глючит,
            # переведите через другой банк» или отправят в отделение с наличными.
            # Поэтому против социальной инженерии блокировка слабее разговора.
            BLOCK: {"stolen_card": 0.99, "social_eng": 0.60, "fraud_ring": 0.97},
        }
    )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_ECONOMICS = Economics()


# --------------------------------------------------------------------------
# Расчёт
# --------------------------------------------------------------------------


def loss_given_fraud(amount: float, tx_type: str, econ: Economics) -> float:
    """Сколько теряем, если фрод прошёл."""
    lgd = {
        "transfer": econ.lgd_transfer,
        "withdrawal": econ.lgd_withdrawal,
    }.get(tx_type, econ.lgd_purchase)
    return amount * lgd + econ.case_handling_cost


def estimate_clv(client_summary: dict | None, econ: Economics, amount: float = 0.0) -> float:
    """Грубая пожизненная ценность клиента: средний чек × активность × маржа × 3 года.

    Если истории нет, единственный доступный признак масштаба клиента — размер
    самой операции: человек, который переводит миллион, стоит банку дороже
    среднего. Поэтому запасная оценка тоже зависит от суммы.
    """
    fallback = float(min(max(econ.clv_default, amount * 4.0), econ.clv_cap))
    if not client_summary:
        return fallback
    avg = float(client_summary.get("avg_amount") or 0.0)
    tx_count = float(client_summary.get("tx_count") or 0.0)
    if avg <= 0 or tx_count <= 0:
        return fallback
    monthly_turnover = avg * max(tx_count, 1.0)  # история у нас ровно за месяц
    clv = monthly_turnover * econ.merchant_margin * 36.0
    return float(min(max(clv, 25_000.0), econ.clv_cap))


def blended_effectiveness(action: str, type_probs: dict[str, float], econ: Economics) -> float:
    """Эффективность действия с учётом того, какая именно схема перед нами."""
    table = econ.effectiveness[action]
    total = sum(max(type_probs.get(k, 0.0), 0.0) for k in FRAUD_KINDS)
    if total <= 0:
        # Распределения типа нет — берём осторожное среднее
        return sum(table[k] for k in FRAUD_KINDS) / len(FRAUD_KINDS)
    return sum(table[k] * max(type_probs.get(k, 0.0), 0.0) for k in FRAUD_KINDS) / total


def expected_costs(
    p_fraud: float,
    amount: float,
    tx_type: str = "purchase",
    type_probs: dict[str, float] | None = None,
    client_summary: dict | None = None,
    econ: Economics | None = None,
) -> dict[str, float]:
    """Ожидаемые потери в тенге по каждому из четырёх действий."""
    econ = econ or DEFAULT_ECONOMICS
    type_probs = type_probs or {}
    p = min(max(float(p_fraud), 0.0), 1.0)
    q = 1.0 - p

    fraud_loss = loss_given_fraud(amount, tx_type, econ)
    clv = estimate_clv(client_summary, econ, amount)
    margin = amount * econ.merchant_margin

    costs: dict[str, float] = {}

    # Пропустить: платим только за пропущенный фрод
    costs[ALLOW] = p * fraud_loss

    # Подтвердить: часть фрода останавливаем, честного клиента слегка раздражаем
    eff = blended_effectiveness(STEP_UP, type_probs, econ)
    costs[STEP_UP] = (
        p * (1.0 - eff) * fraud_loss
        + econ.step_up_op_cost
        + q * (
            econ.step_up_abandon_rate * margin
            + econ.step_up_annoyance
            + econ.step_up_friction_rate * amount
        )
    )

    # Задержать и позвонить: дорого по операторам, зато ломает обман
    eff = blended_effectiveness(HOLD, type_probs, econ)
    costs[HOLD] = (
        p * (1.0 - eff) * fraud_loss
        + econ.hold_op_cost
        + q * (
            econ.hold_abandon_rate * margin
            + econ.hold_annoyance
            + econ.hold_friction_rate * amount
        )
    )

    # Заблокировать: фрод почти наверняка остановлен, но ложная блокировка
    # стоит маржи и куска пожизненной ценности клиента
    eff = blended_effectiveness(BLOCK, type_probs, econ)
    costs[BLOCK] = (
        p * (1.0 - eff) * fraud_loss
        + q * (margin + econ.block_churn_rate * clv)
    )

    return {a: float(round(c, 2)) for a, c in costs.items()}


@dataclass
class Decision:
    action: str
    action_label: str
    p_fraud: float
    expected_costs: dict[str, float]
    chosen_cost: float
    # Сколько сэкономили относительно «пропустить всё»
    saving_vs_allow: float
    # Насколько следующее по дешевизне действие дороже выбранного.
    # Маленький запас = пограничный случай, его полезно показать оператору.
    margin_to_next: float
    runner_up: str
    type_probs: dict[str, float]
    amount: float

    def to_dict(self) -> dict:
        return asdict(self)


def decide(
    p_fraud: float,
    amount: float,
    tx_type: str = "purchase",
    type_probs: dict[str, float] | None = None,
    client_summary: dict | None = None,
    econ: Economics | None = None,
) -> Decision:
    """Выбрать самое дешёвое действие для конкретной транзакции."""
    costs = expected_costs(p_fraud, amount, tx_type, type_probs, client_summary, econ)
    ranked = sorted(costs.items(), key=lambda kv: kv[1])
    best, best_cost = ranked[0]
    second, second_cost = ranked[1]

    return Decision(
        action=best,
        action_label=ACTION_LABELS_RU[best],
        p_fraud=round(float(p_fraud), 6),
        expected_costs=costs,
        chosen_cost=best_cost,
        saving_vs_allow=round(costs[ALLOW] - best_cost, 2),
        margin_to_next=round(second_cost - best_cost, 2),
        runner_up=second,
        type_probs={k: round(float(v), 4) for k, v in (type_probs or {}).items()},
        amount=float(amount),
    )


# --------------------------------------------------------------------------
# Сравнение с обычным порогом — чтобы выгода была видна в деньгах
# --------------------------------------------------------------------------


def realised_cost(
    action: str,
    is_fraud: bool,
    amount: float,
    tx_type: str,
    fraud_kind: str | None,
    client_summary: dict | None = None,
    econ: Economics | None = None,
) -> float:
    """Фактические потери по уже известному исходу.

    Ожидаемые потери мы считаем до того, как узнали правду. Эта функция —
    для отчёта постфактум: берём реальную метку и считаем, во сколько нам
    обошлось выбранное действие.
    """
    econ = econ or DEFAULT_ECONOMICS
    fraud_loss = loss_given_fraud(amount, tx_type, econ)
    margin = amount * econ.merchant_margin
    clv = estimate_clv(client_summary, econ, amount)

    if is_fraud:
        eff = econ.effectiveness[action].get(fraud_kind or "stolen_card", 0.0)
        # В среднем по множеству случаев доля (1 - eff) проходит
        cost = (1.0 - eff) * fraud_loss
        if action == STEP_UP:
            cost += econ.step_up_op_cost
        elif action == HOLD:
            cost += econ.hold_op_cost
        return cost

    # Честная операция
    if action == ALLOW:
        return 0.0
    if action == STEP_UP:
        return (
            econ.step_up_op_cost
            + econ.step_up_abandon_rate * margin
            + econ.step_up_annoyance
            + econ.step_up_friction_rate * amount
        )
    if action == HOLD:
        return (
            econ.hold_op_cost
            + econ.hold_abandon_rate * margin
            + econ.hold_annoyance
            + econ.hold_friction_rate * amount
        )
    return margin + econ.block_churn_rate * clv


def threshold_policy(p_fraud: float, threshold: float = 0.5) -> str:
    """Базовая политика «один порог на всех» — то, с чем мы сравниваемся."""
    return BLOCK if p_fraud >= threshold else ALLOW
