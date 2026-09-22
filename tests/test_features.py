"""Проверки признаков: именно здесь ошибки самые дорогие и самые незаметные."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURES, FeatureStore


def _tx(**kw) -> dict:
    base = dict(
        client_id="C1",
        timestamp="2025-09-10 14:00:00",
        amount=10_000.0,
        merchant_category="grocery",
        channel="pos",
        tx_type="purchase",
        device_id="D1",
        ip="1.2.3.4",
        city="Алматы",
        country="KZ",
        is_vpn=0,
        remote_access=0,
        call_minutes_before=0.0,
        session_duration_s=90.0,
        recipient_id="",
    )
    base.update(kw)
    return base


def test_all_features_present_and_finite():
    store = FeatureStore()
    feats = store.features(_tx())
    assert set(feats) == set(FEATURES), "набор признаков разошёлся со списком FEATURES"
    for name, value in feats.items():
        assert np.isfinite(value), f"{name} не конечное: {value}"


def test_nan_recipient_does_not_merge_clients():
    """Пропущенный получатель не должен склеивать разных клиентов в одну сущность.

    Из CSV пустая строка приходит как NaN, а NaN в Python истинный. Ровно на
    этом система однажды решила, что все покупки в стране идут одному человеку.
    """
    store = FeatureStore()
    for i in range(5):
        store.observe(_tx(client_id=f"C{i}", recipient_id=float("nan")))
    feats = store.features(_tx(client_id="C9", recipient_id=float("nan")))
    assert feats["clients_per_recipient"] == 0.0


def test_no_future_leak():
    """Признаки считаются по состоянию ДО транзакции, а не после."""
    store = FeatureStore()
    tx = _tx(amount=50_000)
    first = store.features(tx)
    assert first["history_len"] == 0
    store.observe(tx)
    second = store.features(tx)
    assert second["history_len"] == 1
    assert second["is_new_device"] == 0.0


def test_velocity_window_is_24h():
    store = FeatureStore()
    store.observe(_tx(timestamp="2025-09-10 10:00:00"))
    store.observe(_tx(timestamp="2025-09-10 10:30:00"))
    near = store.features(_tx(timestamp="2025-09-10 10:45:00"))
    assert near["tx_count_1h"] == 2
    far = store.features(_tx(timestamp="2025-09-12 10:45:00"))
    assert far["tx_count_24h"] == 0


def test_impossible_travel_detected():
    store = FeatureStore()
    store.observe(_tx(timestamp="2025-09-10 10:00:00", city="Алматы"))
    feats = store.features(_tx(timestamp="2025-09-10 10:20:00", city="Лондон", country="GB"))
    assert feats["travel_speed_kmh"] > 3000
    assert feats["is_foreign"] == 1.0


def test_shared_device_reveals_ring():
    store = FeatureStore()
    for i in range(8):
        store.observe(_tx(client_id=f"R{i}", device_id="RING", ip="9.9.9.9"))
    feats = store.features(_tx(client_id="R99", device_id="RING", ip="9.9.9.9"))
    # Восемь, а не девять: текущая транзакция ещё не учтена. Так и должно быть —
    # признак обязан описывать то, что система знала ДО неё.
    assert feats["clients_per_device"] == 8
    assert feats["clients_per_ip"] == 8
    store.observe(_tx(client_id="R99", device_id="RING", ip="9.9.9.9"))
    assert store.features(_tx(client_id="R100", device_id="RING"))["clients_per_device"] == 9


def test_sandbox_does_not_touch_base():
    """Симулятор обязан быть повторяемым: демо гоняют десятки раз подряд."""
    store = FeatureStore()
    store.observe(_tx(client_id="C1"))
    before = store.clients["C1"].n_tx

    sb = store.sandbox()
    for _ in range(3):
        sb.observe(_tx(client_id="C1", device_id="SANDBOX", ip="8.8.8.8"))

    assert store.clients["C1"].n_tx == before
    assert "SANDBOX" not in store.global_state.device_clients
    assert sb.clients["C1"].n_tx == before + 3
    # песочница при этом видит накопленную базу
    assert sb.state_for("C1").devices.get("D1") is not None


def test_sandbox_sees_base_network_counters():
    store = FeatureStore()
    for i in range(4):
        store.observe(_tx(client_id=f"C{i}", device_id="SHARED"))
    sb = store.sandbox()
    feats = sb.features(_tx(client_id="CX", device_id="SHARED"))
    assert feats["clients_per_device"] == 4
    sb.observe(_tx(client_id="CX", device_id="SHARED"))
    assert sb.features(_tx(client_id="CY", device_id="SHARED"))["clients_per_device"] == 5
    # база по-прежнему знает только про четверых
    assert len(store.global_state.device_clients["SHARED"]) == 4


def test_offline_matches_online():
    """Главная проверка: пакетная сборка и онлайн дают одинаковые признаки.

    Если эти два пути разойдутся, модель в проде будет работать не так, как на
    валидации, и заметить это по метрикам практически невозможно.
    """
    from ml.features import build_matrix

    rows = [
        _tx(client_id="C1", timestamp="2025-09-10 10:00:00", amount=5_000),
        _tx(client_id="C1", timestamp="2025-09-10 11:00:00", amount=90_000, device_id="D2"),
        _tx(client_id="C2", timestamp="2025-09-10 12:00:00", amount=7_000, device_id="D2"),
        _tx(client_id="C1", timestamp="2025-09-11 09:00:00", amount=15_000, city="Актау"),
    ]
    df = pd.DataFrame(rows)
    profiles = pd.DataFrame(
        [{"client_id": "C1", "age": 40, "home_city": "Алматы", "avg_ticket": 10_000, "tenure_days": 500},
         {"client_id": "C2", "age": 30, "home_city": "Астана", "avg_ticket": 8_000, "tenure_days": 200}]
    )
    batch, _ = build_matrix(df, profiles)

    online = FeatureStore(profiles)
    stream = [online.features_and_observe(r) for r in rows]

    for i, row in enumerate(stream):
        for f in FEATURES:
            assert row[f] == pytest.approx(batch.iloc[i][f]), f"строка {i}, признак {f}"
