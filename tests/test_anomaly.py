"""Проверки второго канала — поиска аномалий без учителя."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.anomaly import ANOMALY_FEATURES, AnomalyChannel, NoisyOr
from ml.features import FEATURES


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)


def test_anomaly_features_exist_in_feature_set():
    missing = [f for f in ANOMALY_FEATURES if f not in FEATURES]
    assert not missing, f"канал просит признаки, которых нет: {missing}"


def test_channel_scores_outliers_higher():
    rng = np.random.default_rng(1)
    normal = _frame(500, seed=1)
    channel = AnomalyChannel(contamination=0.02).fit(normal)

    weird = normal.iloc[:20].copy()
    for f in ANOMALY_FEATURES[:8]:
        weird[f] = 25.0  # заведомо далеко от нормы

    assert channel.score(weird).mean() > channel.score(normal).mean()


def test_noisy_or_never_lowers_suspicion():
    """Связка «или» обязана только повышать подозрение.

    Это не стилистика, а суть страховки: если бы уверенность модели могла
    погасить аномалию, незнакомая схема снова проходила бы насквозь.
    """
    rng = np.random.default_rng(2)
    anomaly = rng.normal(size=400)
    y = (rng.random(400) < 0.2).astype(int)
    p_sup = rng.random(400)

    combiner = NoisyOr(cap=0.3).fit(anomaly, y)
    mixed = combiner.combine(p_sup, anomaly)
    assert (mixed >= p_sup - 1e-9).all()
    assert (mixed <= 1.0 + 1e-9).all()


def test_anomaly_contribution_respects_cap():
    rng = np.random.default_rng(3)
    anomaly = rng.normal(size=300)
    y = (anomaly > 0.5).astype(int)          # аномалия идеально предсказывает метку
    combiner = NoisyOr(cap=0.25).fit(anomaly, y)
    assert combiner.anomaly_probability(anomaly).max() <= 0.25 + 1e-9


def test_recalibration_restores_probability_scale():
    """После «или» вероятность обязана вернуться к честной шкале.

    Без второго прохода изотонической регрессии смесь систематически
    завышала вероятность, и стоимостная модель начинала перестраховываться:
    трение по честным клиентам выросло с 2.7% до 14.4%.
    """
    rng = np.random.default_rng(4)
    n = 4000
    y = (rng.random(n) < 0.05).astype(int)
    p_sup = np.clip(np.where(y == 1, rng.beta(5, 2, n), rng.beta(1, 25, n)), 0, 1)
    anomaly = np.where(y == 1, rng.normal(1.0, 1, n), rng.normal(0, 1, n))

    combiner = NoisyOr(cap=0.3).fit(anomaly, y)
    raw = combiner.combine(p_sup, anomaly)
    combiner.fit_final(p_sup, anomaly, y)
    final = combiner.predict(p_sup, anomaly)

    rate = y.mean()
    # сырая смесь завышает, откалиброванная держится рядом с базовой частотой
    assert raw.mean() > rate
    assert abs(final.mean() - rate) < abs(raw.mean() - rate)
    assert abs(final.mean() - rate) < 0.02


def test_recalibration_preserves_ranking():
    """Перекалибровка монотонна, поэтому страховка продолжает работать.

    Полнота по незнакомой схеме измеряется ранжированием, а не абсолютным
    значением вероятности. Если бы второй проход портил порядок, он бы
    сломал ровно то, ради чего канал добавлен.
    """
    rng = np.random.default_rng(5)
    n = 2000
    y = (rng.random(n) < 0.1).astype(int)
    p_sup = rng.random(n)
    anomaly = rng.normal(size=n)

    combiner = NoisyOr(cap=0.3).fit(anomaly, y).fit_final(p_sup, anomaly, y)
    raw = combiner.combine(p_sup, anomaly)
    final = combiner.predict(p_sup, anomaly)

    order_raw = np.argsort(raw)
    # изотоническая регрессия неубывающая: порядок сохраняется
    assert (np.diff(final[order_raw]) >= -1e-9).all()
