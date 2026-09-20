"""Проверки мониторинга дрейфа."""

from __future__ import annotations

import numpy as np

from ml.drift import ACCUMULATING, PSI_SIGNIFICANT, psi, verdict


def test_identical_samples_have_no_drift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-9


def test_shifted_distribution_is_detected():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 5000)
    shifted = rng.normal(2.5, 1, 5000)
    assert psi(base, shifted) > PSI_SIGNIFICANT
    assert verdict(psi(base, shifted)) == "significant"


def test_small_noise_is_not_flagged():
    rng = np.random.default_rng(2)
    assert verdict(psi(rng.normal(size=8000), rng.normal(size=8000))) == "stable"


def test_empty_bucket_does_not_give_infinity():
    """Отсутствующий диапазон не должен обнулять весь отчёт бесконечностью."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=4000)
    narrow = rng.normal(size=4000) * 0.01      # целые корзины окажутся пустыми
    value = psi(base, narrow)
    assert np.isfinite(value)
    assert value > 0


def test_constant_feature_is_not_drift():
    zeros = np.zeros(1000)
    assert psi(zeros, zeros) == 0.0
    assert psi(zeros, np.zeros(500)) == 0.0


def test_accumulating_features_are_declared():
    """Накопительные признаки должны быть помечены явно.

    Без этого списка дежурный каждый день видит красными признаки, которые
    красные по построению, перестаёт смотреть на список — и пропускает
    настоящий сдвиг.
    """
    from ml.features import FEATURES

    unknown = [f for f in ACCUMULATING if f not in FEATURES]
    assert not unknown, f"в списке признаки, которых нет: {unknown}"
    for expected in ("device_age_days", "history_len", "component_size"):
        assert expected in ACCUMULATING
