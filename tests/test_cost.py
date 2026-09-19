"""Проверки стоимостной модели — ядра, которое принимает решение."""

from __future__ import annotations

import pytest

from backend.cost import (
    ALLOW,
    BLOCK,
    HOLD,
    STEP_UP,
    Economics,
    blended_effectiveness,
    decide,
    estimate_clv,
    expected_costs,
    realised_cost,
)

STOLEN = {"stolen_card": 0.92, "social_eng": 0.04, "fraud_ring": 0.04}
SOCIAL = {"stolen_card": 0.04, "social_eng": 0.92, "fraud_ring": 0.04}
RING = {"stolen_card": 0.05, "social_eng": 0.05, "fraud_ring": 0.90}


def test_tiny_risk_is_allowed():
    d = decide(0.001, 1_500, "purchase", STOLEN)
    assert d.action == ALLOW


def test_obvious_fraud_is_never_allowed():
    for probs in (STOLEN, SOCIAL, RING):
        d = decide(0.97, 900_000, "transfer", probs)
        assert d.action != ALLOW


def test_social_engineering_prefers_a_call_over_a_push():
    """Ключевое утверждение проекта.

    Push-подтверждение против обмана почти бесполезно: клиент под влиянием
    мошенника подтверждает операцию сам. Разговор с оператором — единственное,
    что разрывает схему, и стоимостная модель обязана это видеть.
    """
    costs = expected_costs(0.6, 800_000, "transfer", SOCIAL)
    assert costs[HOLD] < costs[STEP_UP]
    assert decide(0.6, 800_000, "transfer", SOCIAL).action == HOLD


def test_same_probability_different_amount_gives_different_action():
    """Порога нет: при одной вероятности решение зависит от суммы."""
    small = decide(0.25, 3_000, "purchase", STOLEN)
    large = decide(0.25, 2_000_000, "transfer", STOLEN)
    assert small.action != large.action
    assert ACTION_ORDER.index(large.action) > ACTION_ORDER.index(small.action)


ACTION_ORDER = [ALLOW, STEP_UP, HOLD, BLOCK]


def test_same_amount_different_scheme_gives_different_action():
    """При одной сумме и вероятности вид схемы меняет выбор действия."""
    a = decide(0.45, 400_000, "transfer", STOLEN)
    b = decide(0.45, 400_000, "transfer", SOCIAL)
    assert a.action != b.action or a.expected_costs != b.expected_costs
    assert b.action == HOLD


def test_costs_are_monotonic_in_probability():
    """Чем выше вероятность фрода, тем дороже «пропустить»."""
    prev = -1.0
    for p in (0.0, 0.1, 0.3, 0.6, 0.9, 1.0):
        c = expected_costs(p, 100_000, "transfer", STOLEN)[ALLOW]
        assert c > prev
        prev = c


def test_chosen_action_is_the_cheapest():
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        for amount in (900, 25_000, 400_000, 5_000_000):
            d = decide(p, amount, "transfer", SOCIAL)
            assert d.chosen_cost == pytest.approx(min(d.expected_costs.values()))
            assert d.margin_to_next >= 0


def test_blended_effectiveness_matches_pure_type():
    econ = Economics()
    pure = {"stolen_card": 1.0, "social_eng": 0.0, "fraud_ring": 0.0}
    assert blended_effectiveness(STEP_UP, pure, econ) == pytest.approx(
        econ.effectiveness[STEP_UP]["stolen_card"]
    )


def test_blended_effectiveness_handles_empty_distribution():
    econ = Economics()
    value = blended_effectiveness(HOLD, {}, econ)
    assert 0.0 <= value <= 1.0


def test_clv_scales_with_client_value():
    econ = Economics()
    small = estimate_clv({"avg_amount": 3_000, "tx_count": 10}, econ)
    big = estimate_clv({"avg_amount": 300_000, "tx_count": 40}, econ)
    assert big > small
    assert big <= econ.clv_cap


def test_economics_overrides_change_the_decision():
    """Параметры бизнеса действительно управляют системой, а не украшают её."""
    cheap_calls = Economics(hold_op_cost=50.0, hold_annoyance=50.0, hold_friction_rate=0.0)
    dear_calls = Economics(hold_op_cost=80_000.0)
    p, amount = 0.3, 150_000
    a = decide(p, amount, "transfer", SOCIAL, None, cheap_calls)
    b = decide(p, amount, "transfer", SOCIAL, None, dear_calls)
    assert a.action == HOLD
    assert b.action != HOLD


def test_realised_cost_of_correct_allow_is_zero():
    assert realised_cost(ALLOW, False, 50_000, "purchase", None) == 0.0


def test_realised_cost_penalises_missed_fraud():
    missed = realised_cost(ALLOW, True, 500_000, "transfer", "social_eng")
    stopped = realised_cost(HOLD, True, 500_000, "transfer", "social_eng")
    assert missed > stopped


def test_false_block_is_expensive():
    assert realised_cost(BLOCK, False, 200_000, "transfer", None) > realised_cost(
        STEP_UP, False, 200_000, "transfer", None
    )
