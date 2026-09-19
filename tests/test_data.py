"""Проверки генератора: данные должны быть трудными, иначе метрики врут."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.generate_data import generate


@pytest.fixture(scope="module")
def dataset():
    # Меньше полного датасета, но достаточно, чтобы доли были устойчивы:
    # на 12 000 строк социальной инженерии выходит меньше сотни, и любая
    # проверка доли начинает ловить шум выборки, а не свойство генератора.
    return generate(n_clients=900, n_tx=30_000, fraud_rate=0.018, seed=7)


def test_shape_and_fraud_rate(dataset):
    df, profiles = dataset
    assert len(profiles) == 900
    assert abs(len(df) - 30_000) <= 30
    rate = df["is_fraud"].mean()
    assert 0.014 < rate < 0.022


def test_all_three_schemes_present(dataset):
    df, _ = dataset
    kinds = df.loc[df.is_fraud == 1, "fraud_type"].value_counts(normalize=True)
    assert set(kinds.index) == {"stolen_card", "social_eng", "fraud_ring"}
    assert kinds["stolen_card"] == pytest.approx(0.45, abs=0.06)
    assert kinds["social_eng"] == pytest.approx(0.35, abs=0.06)


def test_timestamps_are_clean(dataset):
    """Единый формат времени: иначе pandas спотыкается при чтении CSV."""
    df, _ = dataset
    ts = pd.to_datetime(df["timestamp"])
    assert ts.notna().all()
    assert (ts.dt.nanosecond == 0).all()
    assert (ts.dt.microsecond == 0).all()
    assert df["timestamp"].is_monotonic_increasing


def test_social_engineering_looks_legitimate(dataset):
    """Обман клиента обязан выглядеть чисто: своё устройство, свой город, без VPN.

    Если бы эти операции приходили с чужих устройств, задача сводилась бы к
    обычной краже карты и весь смысл проекта пропадал.
    """
    df, profiles = dataset
    home = dict(zip(profiles.client_id, profiles.home_city))
    device = dict(zip(profiles.client_id, profiles.primary_device))
    se = df[df.fraud_type == "social_eng"]
    assert len(se) > 150
    assert (se.city == se.client_id.map(home)).mean() > 0.95
    assert (se.device_id == se.client_id.map(device)).mean() > 0.95
    # Генератор целится в 2%; граница оставляет запас на шум выборки
    assert se.is_vpn.mean() < 0.06


def test_quiet_stolen_cards_exist(dataset):
    """Часть краж должна идти тихо: своя страна, без VPN, небольшие суммы.

    Именно на таких операциях ломаются системы, построенные на правилах
    «заграница плюс новое устройство».
    """
    df, _ = dataset
    stolen = df[df.fraud_type == "stolen_card"]
    quiet = stolen[(stolen.country == "KZ") & (stolen.is_vpn == 0)]
    assert len(quiet) / len(stolen) > 0.20


def test_honest_traffic_is_noisy(dataset):
    """Честные клиенты обязаны шуметь, иначе задача решается тремя правилами."""
    df, profiles = dataset
    device = dict(zip(profiles.client_id, profiles.primary_device))
    legit = df[df.is_fraud == 0]

    # ездят
    assert legit.country.ne("KZ").mean() > 0.005
    # меняют устройства
    assert (legit.device_id != legit.client_id.map(device)).mean() > 0.10
    # пользуются VPN
    assert legit.is_vpn.mean() > 0.03
    # переводят круглыми суммами — как и мошенники
    round_legit = ((legit.amount % 10_000 == 0) & (legit.amount >= 10_000)).mean()
    assert round_legit > 0.01
    # звонят перед переводами
    transfers = legit[legit.tx_type == "transfer"]
    assert transfers.call_minutes_before.gt(0).mean() > 0.20
    # держат программы удалённого доступа
    assert legit.remote_access.mean() > 0.01


def test_rings_share_infrastructure(dataset):
    df, _ = dataset
    ring = df[df.fraud_type == "fraud_ring"]
    assert len(ring) > 20
    per_device = ring.groupby("device_id")["client_id"].nunique()
    assert per_device.max() >= 5


def test_fraud_is_not_trivially_separable(dataset):
    """Ни один одиночный признак не должен выдавать фрод целиком.

    Если такой признак есть, модель выучит его, метрики будут прекрасными,
    а на реальных данных система рассыплется.
    """
    df, _ = dataset
    fraud, legit = df[df.is_fraud == 1], df[df.is_fraud == 0]
    for column in ("is_vpn", "remote_access", "country"):
        if column == "country":
            share_f = fraud.country.ne("KZ").mean()
            share_l = legit.country.ne("KZ").mean()
        else:
            share_f = fraud[column].mean()
            share_l = legit[column].mean()
        # признак может быть сильным, но не абсолютным
        assert share_f < 0.95, f"{column} выдаёт почти весь фрод"
        assert share_l > 0.0, f"{column} никогда не встречается у честных"


def test_generation_is_reproducible_across_processes():
    """Один сид обязан давать один датасет в любом процессе.

    Встроенный hash() от строки солится заново при каждом запуске
    (PYTHONHASHSEED), и однажды из-за него датасет с фиксированным сидом
    получался разным от прогона к прогону: цифры в отчёте не сходились с тем,
    что выдавала свежая сборка. Поэтому в генераторе только crc32.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        from ml.generate_data import generate
        df, _ = generate(n_clients=150, n_tx=3_000, seed=5)
        print(int(df.amount.sum()), df.recipient_id.fillna("").str.cat())
        """
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        ).stdout
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, "датасет зависит от PYTHONHASHSEED"
