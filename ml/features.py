"""Feature engineering Fraud Hunter.

Здесь один и тот же код считает признаки и при обучении, и в онлайне.
Это не стилистическая прихоть: если офлайн и онлайн расходятся хоть на один
признак, модель в проде работает не так, как на валидации, и заметить это
почти невозможно.

Схема работы — поток. Транзакции обрабатываются строго по времени:
сначала по накопленному состоянию считаются признаки, и только потом
состояние обновляется этой транзакцией. Значит, ни один признак не видит
будущего, и утечки по времени быть не может.

Состояние двух уровней:
  * ClientState  — что мы знаем про конкретного клиента (суммы, устройства,
    города, получатели, ритм);
  * GlobalState  — что мы знаем про сеть (сколько разных клиентов ходит с
    одного устройства, с одного IP, на одного получателя). Именно это
    вскрывает кольца карт.
"""

from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.common import (
    CATEGORY_LIST,
    CHANNELS,
    TX_TYPES,
    category_risk,
    haversine_km,
    place_coords,
)

# --------------------------------------------------------------------------
# Список признаков. Порядок фиксирован: модель обучается и применяется по нему.
# --------------------------------------------------------------------------

FEATURES: list[str] = [
    # деньги
    "log_amount",
    "amount_to_avg",
    "amount_z",
    "amount_to_max",
    "is_amount_record",
    "is_round_amount",
    # время
    "hour",
    "is_night",
    "is_weekend",
    "hours_since_prev",
    "tx_count_1h",
    "tx_count_24h",
    "amount_24h_to_avg",
    # устройство
    "is_new_device",
    "device_tx_share",
    "device_age_days",
    # гео
    "is_new_city",
    "city_tx_share",
    "is_foreign",
    "distance_from_home_km",
    "distance_from_prev_km",
    "travel_speed_kmh",
    "is_vpn",
    # сеть
    "clients_per_device",
    "clients_per_ip",
    "clients_per_recipient",
    "is_new_recipient",
    # поведение сессии
    "remote_access",
    "call_minutes_before",
    "has_long_call",
    "session_duration_s",
    # торговая точка
    "category_risk",
    "is_new_category",
    "channel_code",
    "tx_type_code",
    "is_transfer",
    # клиент
    "client_age",
    "client_tenure_days",
    "history_len",
    "night_share",
    # контекст «всё своё» — ключ к социальной инженерии
    "own_context",
]

# Вес априорного среднего чека из профиля клиента. В проде это среднее за
# прошлые периоды, которое банк уже знает. Вес маленький: после трёх-четырёх
# операций признаки опираются уже на фактическую историю.
PRIOR_WEIGHT = 3.0

# Средний чек «по больнице» — запасной вариант для клиента без профиля
GLOBAL_FALLBACK_TICKET = 12_000.0

_CHANNEL_CODE = {c: i for i, c in enumerate(CHANNELS)}
_TXTYPE_CODE = {t: i for i, t in enumerate(TX_TYPES)}
_CATEGORY_CODE = {c: i for i, c in enumerate(CATEGORY_LIST)}

_HOUR = 3600.0
_DAY = 86400.0


# --------------------------------------------------------------------------
# Состояние
# --------------------------------------------------------------------------


@dataclass
class ClientState:
    """Всё, что система знает о клиенте к моменту очередной операции."""

    client_id: str
    age: int = 35
    home_city: str = "Алматы"
    prior_ticket: float = GLOBAL_FALLBACK_TICKET
    tenure_days: int = 365

    n_tx: int = 0
    sum_amount: float = 0.0
    sum_amount_sq: float = 0.0
    max_amount: float = 0.0
    night_count: int = 0

    devices: dict[str, list] = field(default_factory=dict)      # device -> [count, first_epoch]
    cities: dict[str, int] = field(default_factory=dict)
    categories: dict[str, int] = field(default_factory=dict)
    recipients: set[str] = field(default_factory=set)

    last_epoch: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None

    recent: deque = field(default_factory=lambda: deque())       # (epoch, amount) за 24 часа

    # ---- производные величины
    @property
    def mean_amount(self) -> float:
        """Ожидаемый чек: априорная оценка, размытая фактической историей."""
        return (self.prior_ticket * PRIOR_WEIGHT + self.sum_amount) / (PRIOR_WEIGHT + self.n_tx)

    @property
    def std_amount(self) -> float:
        if self.n_tx < 2:
            return self.mean_amount * 0.8
        mean = self.sum_amount / self.n_tx
        var = max(self.sum_amount_sq / self.n_tx - mean * mean, 0.0)
        return max(math.sqrt(var), self.mean_amount * 0.15)

    def trim(self, now_epoch: float) -> None:
        """Выбрасываем из окна скорости всё старше суток."""
        cutoff = now_epoch - _DAY
        while self.recent and self.recent[0][0] < cutoff:
            self.recent.popleft()


@dataclass
class GlobalState:
    """Сетевой срез: кто с кем делит устройство, IP и получателя."""

    device_clients: dict[str, set[str]] = field(default_factory=dict)
    ip_clients: dict[str, set[str]] = field(default_factory=dict)
    recipient_clients: dict[str, set[str]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Расчёт признаков
# --------------------------------------------------------------------------


def _s(value) -> str:
    """Строковое поле в str. Пустые значения из CSV приходят как NaN, а NaN
    в Python истинный — из-за этого `value or ""` молча превращает пропуск в
    строку "nan" и склеивает разных получателей в одну сущность."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    return "" if text in ("nan", "None", "NaT") else text


def _f(value, default: float = 0.0) -> float:
    """Числовое поле во float с аккуратной обработкой пропусков."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def _epoch(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    return pd.Timestamp(ts).timestamp()


def compute_features(tx: dict, st: ClientState, gs: GlobalState) -> dict[str, float]:
    """Признаки одной транзакции по состоянию «до» неё.

    Эта функция — единственный источник правды о признаках. И обучение,
    и API вызывают именно её.
    """
    ts = pd.Timestamp(tx["timestamp"])
    now = ts.timestamp()
    amount = _f(tx["amount"])
    device = _s(tx.get("device_id"))
    ip = _s(tx.get("ip"))
    city = _s(tx.get("city"))
    country = _s(tx.get("country")) or "KZ"
    category = _s(tx.get("merchant_category")) or "grocery"
    channel = _s(tx.get("channel")) or "pos"
    tx_type = _s(tx.get("tx_type")) or "purchase"
    recipient = _s(tx.get("recipient_id"))

    st.trim(now)
    mean_amount = st.mean_amount
    std_amount = st.std_amount

    # --- деньги
    amount_to_avg = amount / max(mean_amount, 1.0)
    amount_z = (amount - mean_amount) / max(std_amount, 1.0)
    amount_to_max = amount / max(st.max_amount, 1.0) if st.max_amount > 0 else 1.0
    is_amount_record = 1.0 if (st.n_tx >= 3 and amount > st.max_amount) else 0.0
    is_round_amount = 1.0 if amount >= 10_000 and abs(amount % 10_000) < 1e-6 else 0.0

    # --- время и скорость
    hours_since_prev = (now - st.last_epoch) / _HOUR if st.last_epoch is not None else 720.0
    hours_since_prev = float(min(hours_since_prev, 720.0))
    cutoff_1h = now - _HOUR
    tx_count_1h = float(sum(1 for e, _a in st.recent if e >= cutoff_1h))
    tx_count_24h = float(len(st.recent))
    amount_24h = float(sum(a for _e, a in st.recent)) + amount
    amount_24h_to_avg = amount_24h / max(mean_amount, 1.0)

    # --- устройство
    dev_rec = st.devices.get(device)
    is_new_device = 0.0 if dev_rec else 1.0
    device_tx_share = (dev_rec[0] / st.n_tx) if (dev_rec and st.n_tx) else 0.0
    device_age_days = ((now - dev_rec[1]) / _DAY) if dev_rec else 0.0

    # --- гео
    city_count = st.cities.get(city, 0)
    is_new_city = 0.0 if city_count else 1.0
    city_tx_share = (city_count / st.n_tx) if st.n_tx else 0.0
    is_foreign = 0.0 if country == "KZ" else 1.0
    lat, lon = place_coords(city)
    home_lat, home_lon = place_coords(st.home_city)
    distance_from_home_km = haversine_km(home_lat, home_lon, lat, lon)
    if st.last_lat is not None and st.last_lon is not None:
        distance_from_prev_km = haversine_km(st.last_lat, st.last_lon, lat, lon)
    else:
        distance_from_prev_km = 0.0
    # «Невозможное перемещение»: 900 км за 20 минут — это не самолёт
    travel_speed_kmh = distance_from_prev_km / max(hours_since_prev, 1 / 60.0)
    travel_speed_kmh = float(min(travel_speed_kmh, 100_000.0))

    # --- сеть
    clients_per_device = float(len(gs.device_clients.get(device, ())) or 1)
    clients_per_ip = float(len(gs.ip_clients.get(ip, ())) or 1)
    if recipient:
        clients_per_recipient = float(len(gs.recipient_clients.get(recipient, ())) or 1)
        is_new_recipient = 0.0 if recipient in st.recipients else 1.0
    else:
        clients_per_recipient = 0.0
        is_new_recipient = 0.0

    # --- торговая точка
    is_new_category = 0.0 if st.categories.get(category) else 1.0
    is_transfer = 1.0 if tx_type == "transfer" else 0.0

    # --- «всё своё»: знакомое устройство, знакомый город, без VPN.
    # Для кражи карты так почти не бывает, для социальной инженерии — всегда.
    is_vpn = _f(tx.get("is_vpn"))
    own_context = 1.0 if (is_new_device == 0.0 and is_new_city == 0.0 and is_vpn == 0.0) else 0.0

    call_minutes = _f(tx.get("call_minutes_before"))

    return {
        "log_amount": math.log1p(amount),
        "amount_to_avg": amount_to_avg,
        "amount_z": amount_z,
        "amount_to_max": amount_to_max,
        "is_amount_record": is_amount_record,
        "is_round_amount": is_round_amount,
        "hour": float(ts.hour),
        "is_night": 1.0 if ts.hour < 6 else 0.0,
        "is_weekend": 1.0 if ts.dayofweek >= 5 else 0.0,
        "hours_since_prev": hours_since_prev,
        "tx_count_1h": tx_count_1h,
        "tx_count_24h": tx_count_24h,
        "amount_24h_to_avg": amount_24h_to_avg,
        "is_new_device": is_new_device,
        "device_tx_share": device_tx_share,
        "device_age_days": device_age_days,
        "is_new_city": is_new_city,
        "city_tx_share": city_tx_share,
        "is_foreign": is_foreign,
        "distance_from_home_km": distance_from_home_km,
        "distance_from_prev_km": distance_from_prev_km,
        "travel_speed_kmh": travel_speed_kmh,
        "is_vpn": is_vpn,
        "clients_per_device": clients_per_device,
        "clients_per_ip": clients_per_ip,
        "clients_per_recipient": clients_per_recipient,
        "is_new_recipient": is_new_recipient,
        "remote_access": _f(tx.get("remote_access")),
        "call_minutes_before": call_minutes,
        "has_long_call": 1.0 if call_minutes >= 5 else 0.0,
        "session_duration_s": _f(tx.get("session_duration_s")),
        "category_risk": category_risk(category),
        "is_new_category": is_new_category,
        "channel_code": float(_CHANNEL_CODE.get(channel, 0)),
        "tx_type_code": float(_TXTYPE_CODE.get(tx_type, 0)),
        "is_transfer": is_transfer,
        "client_age": float(st.age),
        "client_tenure_days": float(st.tenure_days),
        "history_len": float(st.n_tx),
        "night_share": (st.night_count / st.n_tx) if st.n_tx else 0.0,
        "own_context": own_context,
    }


def update_state(tx: dict, st: ClientState, gs: GlobalState) -> None:
    """Учесть транзакцию в состоянии. Вызывается строго ПОСЛЕ compute_features."""
    ts = pd.Timestamp(tx["timestamp"])
    now = ts.timestamp()
    amount = _f(tx["amount"])
    device = _s(tx.get("device_id"))
    ip = _s(tx.get("ip"))
    city = _s(tx.get("city"))
    category = _s(tx.get("merchant_category")) or "grocery"
    recipient = _s(tx.get("recipient_id"))

    st.n_tx += 1
    st.sum_amount += amount
    st.sum_amount_sq += amount * amount
    st.max_amount = max(st.max_amount, amount)
    if ts.hour < 6:
        st.night_count += 1

    if device:
        rec = st.devices.get(device)
        if rec is None:
            st.devices[device] = [1, now]
        else:
            rec[0] += 1
    if city:
        st.cities[city] = st.cities.get(city, 0) + 1
    st.categories[category] = st.categories.get(category, 0) + 1
    if recipient:
        st.recipients.add(recipient)

    st.last_epoch = now
    lat, lon = place_coords(city)
    st.last_lat, st.last_lon = lat, lon
    st.recent.append((now, amount))
    st.trim(now)

    if device:
        gs.device_clients.setdefault(device, set()).add(st.client_id)
    if ip:
        gs.ip_clients.setdefault(ip, set()).add(st.client_id)
    if recipient:
        gs.recipient_clients.setdefault(recipient, set()).add(st.client_id)


# --------------------------------------------------------------------------
# Хранилище состояний
# --------------------------------------------------------------------------


class _OverlayIndex:
    """Доступ «база + правки» к сетевым счётчикам, не трогая базу.

    Нужен песочнице: она должна видеть всё, что накопила система, но её
    собственные записи обязаны остаться при ней.
    """

    def __init__(self, base: dict[str, set], overlay: dict[str, set]):
        self.base = base
        self.overlay = overlay

    def get(self, key, default=()):
        b = self.base.get(key)
        o = self.overlay.get(key)
        if b and o:
            return b | o
        return b if b else (o if o else default)

    def setdefault(self, key, default):
        if key not in self.overlay:
            self.overlay[key] = set(self.base.get(key, ()))
        return self.overlay[key]


class _OverlayGlobalState:
    def __init__(self, base: GlobalState):
        self.device_clients = _OverlayIndex(base.device_clients, {})
        self.ip_clients = _OverlayIndex(base.ip_clients, {})
        self.recipient_clients = _OverlayIndex(base.recipient_clients, {})


class FeatureStore:
    """Состояние всех клиентов и сети. Кладётся в joblib и поднимается в API."""

    def __init__(self, profiles: pd.DataFrame | None = None):
        self.clients: dict[str, ClientState] = {}
        self.global_state = GlobalState()
        if profiles is not None:
            for row in profiles.itertuples(index=False):
                self.clients[row.client_id] = ClientState(
                    client_id=row.client_id,
                    age=int(row.age),
                    home_city=str(row.home_city),
                    prior_ticket=float(row.avg_ticket),
                    tenure_days=int(row.tenure_days),
                )

    def state_for(self, client_id: str) -> ClientState:
        st = self.clients.get(client_id)
        if st is None:
            st = ClientState(client_id=client_id)
            self.clients[client_id] = st
        return st

    def features(self, tx: dict) -> dict[str, float]:
        st = self.state_for(str(tx["client_id"]))
        return compute_features(tx, st, self.global_state)

    def observe(self, tx: dict) -> None:
        st = self.state_for(str(tx["client_id"]))
        update_state(tx, st, self.global_state)

    def features_and_observe(self, tx: dict) -> dict[str, float]:
        feats = self.features(tx)
        self.observe(tx)
        return feats

    def sandbox(self) -> "SandboxStore":
        """Одноразовая копия состояния для проигрывания сценария.

        Симулятор должен уметь прогонять серию операций подряд: вторая
        транзакция обязана видеть первую, иначе всплеска не будет и разбор
        сценария теряет смысл. При этом настоящее состояние остаётся нетронутым,
        и демо можно повторять сколько угодно раз с тем же результатом.
        """
        return SandboxStore(self)

    # ---- удобные срезы для API и фронта
    def client_summary(self, client_id: str) -> dict:
        st = self.clients.get(client_id)
        if st is None:
            return {}
        return {
            "client_id": st.client_id,
            "age": st.age,
            "home_city": st.home_city,
            "tx_count": st.n_tx,
            "avg_amount": round(st.mean_amount, 2),
            "max_amount": round(st.max_amount, 2),
            "devices": sorted(st.devices, key=lambda d: -st.devices[d][0]),
            "cities": sorted(st.cities, key=lambda c: -st.cities[c]),
            "tenure_days": st.tenure_days,
            "night_share": round(st.night_count / st.n_tx, 3) if st.n_tx else 0.0,
        }


def build_matrix(
    df: pd.DataFrame, profiles: pd.DataFrame, store: FeatureStore | None = None
) -> tuple[pd.DataFrame, FeatureStore]:
    """Прогнать весь поток по времени и собрать матрицу признаков.

    Возвращает матрицу и состояние на конец потока — последнее уходит в API,
    чтобы онлайн-скоринг начинался не с чистого листа.
    """
    store = store or FeatureStore(profiles)
    df = df.sort_values("timestamp", kind="mergesort")
    records = df.to_dict("records")

    rows = [store.features_and_observe(tx) for tx in records]
    matrix = pd.DataFrame(rows, columns=FEATURES, index=df.index)
    return matrix, store


class SandboxStore(FeatureStore):
    """Песочница поверх основного хранилища: копирует только то, что тронули."""

    def __init__(self, base: FeatureStore):
        self.base = base
        self.clients = {}
        self.global_state = _OverlayGlobalState(base.global_state)

    def state_for(self, client_id: str) -> ClientState:
        st = self.clients.get(client_id)
        if st is None:
            source = self.base.clients.get(client_id)
            st = deepcopy(source) if source is not None else ClientState(client_id=client_id)
            self.clients[client_id] = st
        return st
