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
    # сеть: накопительные счётчики
    "clients_per_device",
    "clients_per_ip",
    "clients_per_recipient",
    "is_new_recipient",
    # сеть: оконные счётчики. Дроп-счёт узнаётся не по тому, что на него
    # когда-либо переводили многие, а по тому, что многие переводят СЕЙЧАС.
    "recipient_clients_24h",
    "recipient_clients_7d",
    "recipient_inbound_24h",
    "recipient_amount_24h",
    "recipient_age_days",
    "recipient_is_new_to_bank",
    "device_clients_24h",
    "ip_clients_24h",
    # сеть: связная компонента графа «клиент — устройство — IP — получатель»
    "component_clients",
    "component_size",
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
    """Сетевой срез: кто с кем делит устройство, IP и получателя.

    Помимо накопительных множеств здесь живут две вещи посложнее.

    Первая — события с отметкой времени. Накопительный счётчик «сколько разных
    клиентов переводили этому получателю» за месяц размывается: у популярного
    магазина он большой и у дроп-счёта большой. Разница в том, что к дропу
    двадцать разных людей приходят за сутки, а не за месяц. Поэтому считаем
    в окне.

    Вторая — связная компонента. Кольцо карт не всегда сидит на одном
    устройстве: участник A делит телефон с B, B выходит с того же IP, что C,
    C переводит туда же, куда D. Попарные счётчики такую цепочку не видят,
    а обход компоненты видит целиком.
    """

    device_clients: dict[str, set[str]] = field(default_factory=dict)
    ip_clients: dict[str, set[str]] = field(default_factory=dict)
    recipient_clients: dict[str, set[str]] = field(default_factory=dict)

    # ключ вида "r:R00042" / "d:D00001A" / "i:10.0.0.1" -> очередь (время, клиент, сумма)
    entity_events: dict[str, deque] = field(default_factory=dict)
    entity_first_seen: dict[str, float] = field(default_factory=dict)

    # union-find по узлам графа
    dsu_parent: dict[str, str] = field(default_factory=dict)
    dsu_rank: dict[str, int] = field(default_factory=dict)
    dsu_nodes: dict[str, int] = field(default_factory=dict)    # узлов в компоненте
    dsu_clients: dict[str, int] = field(default_factory=dict)  # из них клиентов

    # ---- граф
    def find(self, key: str) -> str:
        parent = self.dsu_parent
        if key not in parent:
            parent[key] = key
            self.dsu_rank[key] = 0
            self.dsu_nodes[key] = 1
            self.dsu_clients[key] = 1 if key.startswith("c:") else 0
            return key
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:      # сжатие пути
            parent[key], key = root, parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.dsu_rank[ra] < self.dsu_rank[rb]:
            ra, rb = rb, ra
        self.dsu_parent[rb] = ra
        if self.dsu_rank[ra] == self.dsu_rank[rb]:
            self.dsu_rank[ra] += 1
        self.dsu_nodes[ra] += self.dsu_nodes[rb]
        self.dsu_clients[ra] += self.dsu_clients[rb]

    def component(self, key: str) -> tuple[int, int]:
        """(клиентов в компоненте, узлов в компоненте)."""
        root = self.find(key)
        return self.dsu_clients[root], self.dsu_nodes[root]

    # ---- оконные счётчики
    def window(self, key: str, now: float, seconds: float) -> list:
        events = self.entity_events.get(key)
        if not events:
            return []
        cutoff = now - seconds
        while events and events[0][0] < cutoff:
            events.popleft()
        return list(events)

    def first_seen(self, key: str) -> float | None:
        return self.entity_first_seen.get(key)

    def note_event(self, key: str, now: float, client_id: str, amount: float) -> None:
        events = self.entity_events.get(key)
        if events is None:
            events = deque()
            self.entity_events[key] = events
            self.entity_first_seen[key] = now
        events.append((now, client_id, amount))
        # окно недели: длиннее держать незачем
        cutoff = now - 7 * _DAY
        while events and events[0][0] < cutoff:
            events.popleft()


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

    # --- сеть в окне времени
    r_key, d_key, i_key = f"r:{recipient}", f"d:{device}", f"i:{ip}"
    if recipient:
        ev_24h = gs.window(r_key, now, _DAY)
        ev_7d = gs.window(r_key, now, 7 * _DAY)
        recipient_clients_24h = float(len({c for _t, c, _a in ev_24h}))
        recipient_clients_7d = float(len({c for _t, c, _a in ev_7d}))
        recipient_inbound_24h = float(len(ev_24h))
        recipient_amount_24h = math.log1p(sum(a for _t, _c, a in ev_24h))
        first_seen = gs.first_seen(r_key)
        recipient_age_days = (now - first_seen) / _DAY if first_seen is not None else 0.0
        recipient_is_new_to_bank = 0.0 if first_seen is not None else 1.0
    else:
        recipient_clients_24h = recipient_clients_7d = 0.0
        recipient_inbound_24h = recipient_amount_24h = 0.0
        recipient_age_days = 0.0
        recipient_is_new_to_bank = 0.0

    device_clients_24h = float(len({c for _t, c, _a in gs.window(d_key, now, _DAY)}) or 1)
    ip_clients_24h = float(len({c for _t, c, _a in gs.window(i_key, now, _DAY)}) or 1)

    # --- связная компонента графа
    component_clients, component_size = gs.component(f"c:{st.client_id}")

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
        "recipient_clients_24h": recipient_clients_24h,
        "recipient_clients_7d": recipient_clients_7d,
        "recipient_inbound_24h": recipient_inbound_24h,
        "recipient_amount_24h": recipient_amount_24h,
        "recipient_age_days": recipient_age_days,
        "recipient_is_new_to_bank": recipient_is_new_to_bank,
        "device_clients_24h": device_clients_24h,
        "ip_clients_24h": ip_clients_24h,
        "component_clients": float(component_clients),
        "component_size": float(component_size),
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

    # события с отметкой времени — для оконных счётчиков
    client_node = f"c:{st.client_id}"
    for key in (f"d:{device}" if device else "", f"i:{ip}" if ip else "",
                f"r:{recipient}" if recipient else ""):
        if key:
            gs.note_event(key, now, st.client_id, amount)
            # и ребро графа: клиент связан с устройством, IP и получателем
            gs.union(client_node, key)
    if not (device or ip or recipient):
        gs.find(client_node)


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
    """Сетевое состояние песочницы поверх настоящего.

    Читает всё, что накопила система, пишет только к себе. Оконные счётчики
    отдаются объединением базы и правок без единой записи в базу: штатный
    `window` подчищает устаревшие события прямо в очереди, и для песочницы
    такой побочный эффект недопустим — он бы незаметно портил рабочее
    состояние при каждом прогоне демонстрации.

    Union-find копируется целиком при первой же записи (около шести
    миллисекунд на сто тысяч узлов). Наложение поверх union-find делать
    бессмысленно: сжатие путей всё равно переписывает ссылки, и «только
    для чтения» там не получается.
    """

    def __init__(self, base: GlobalState):
        self.base = base
        self.device_clients = _OverlayIndex(base.device_clients, {})
        self.ip_clients = _OverlayIndex(base.ip_clients, {})
        self.recipient_clients = _OverlayIndex(base.recipient_clients, {})
        self.overlay_events: dict[str, list] = {}
        self.overlay_first_seen: dict[str, float] = {}
        self._dsu: GlobalState | None = None   # копия появляется при первой записи

    # ---- оконные счётчики: база плюс правки, без мутаций базы
    def window(self, key: str, now: float, seconds: float) -> list:
        cutoff = now - seconds
        out = [e for e in self.base.entity_events.get(key, ()) if e[0] >= cutoff]
        out.extend(e for e in self.overlay_events.get(key, ()) if e[0] >= cutoff)
        return out

    def first_seen(self, key: str) -> float | None:
        seen = self.base.first_seen(key)
        return seen if seen is not None else self.overlay_first_seen.get(key)

    def note_event(self, key: str, now: float, client_id: str, amount: float) -> None:
        self.overlay_events.setdefault(key, []).append((now, client_id, amount))
        if self.base.first_seen(key) is None and key not in self.overlay_first_seen:
            self.overlay_first_seen[key] = now

    # ---- граф: копия при первой записи
    def _graph(self) -> GlobalState:
        if self._dsu is None:
            copy = GlobalState()
            copy.dsu_parent = self.base.dsu_parent.copy()
            copy.dsu_rank = self.base.dsu_rank.copy()
            copy.dsu_nodes = self.base.dsu_nodes.copy()
            copy.dsu_clients = self.base.dsu_clients.copy()
            self._dsu = copy
        return self._dsu

    def component(self, key: str) -> tuple[int, int]:
        if self._dsu is None:
            # правок ещё не было — отвечаем по базе, ничего не копируя.
            # Узел, которого база не знает, одинок по определению.
            root = self.base.dsu_parent.get(key)
            if root is None:
                return (1 if key.startswith("c:") else 0), 1
            return self.base.component(key)
        return self._dsu.component(key)

    def find(self, key: str) -> str:
        return self._graph().find(key)

    def union(self, a: str, b: str) -> None:
        self._graph().union(a, b)


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
    df: pd.DataFrame,
    profiles: pd.DataFrame,
    store: FeatureStore | None = None,
    chunk: int = 5_000,
) -> tuple[pd.DataFrame, FeatureStore]:
    """Прогнать весь поток по времени и собрать матрицу признаков.

    Возвращает матрицу и состояние на конец потока — последнее уходит в API,
    чтобы онлайн-скоринг начинался не с чистого листа.

    Поток разбирается кусками, а признаки пишутся сразу в готовый массив.
    Прямолинейный вариант — превратить весь датафрейм в список словарей и
    собрать из него второй список с признаками — на ста тысячах строк держит
    в памяти двести тысяч словарей разом и добавляет к пику около 250 МБ.
    Для сборки на машине с 512 МБ этого достаточно, чтобы всё упало.
    """
    store = store or FeatureStore(profiles)
    df = df.sort_values("timestamp", kind="mergesort")
    n = len(df)
    out = np.empty((n, len(FEATURES)), dtype=np.float64)

    pos = 0
    for start in range(0, n, chunk):
        part = df.iloc[start : start + chunk].to_dict("records")
        for tx in part:
            feats = store.features_and_observe(tx)
            row = out[pos]
            for j, name in enumerate(FEATURES):
                row[j] = feats[name]
            pos += 1
        del part

    matrix = pd.DataFrame(out, columns=FEATURES, index=df.index)
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
