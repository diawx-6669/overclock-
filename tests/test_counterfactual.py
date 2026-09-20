"""Проверки контрфактов: что надо было сделать иначе."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.counterfactual import SEVERITY, find_counterfactuals

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "fraud_hunter.joblib").exists(),
    reason="нет обученной модели",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


def _scorer(client):
    def score(tx: dict) -> dict:
        return client.post("/api/score", json=tx).json()
    return score


def test_counterfactual_always_softens(client):
    """Контрфакт обязан предлагать более мягкое действие, а не любое другое.

    Предложение «сделайте так, и мы вас заблокируем вместо звонка» — не
    объяснение, а издевательство.
    """
    for preset in client.get("/api/presets").json()["presets"]:
        tx = preset["sequence"][-1]
        res = client.post("/api/counterfactual", json=tx).json()
        base_rank = SEVERITY[res["baseline"]["action"]]
        for item in res["items"]:
            assert SEVERITY[item["action_to"]] < base_rank, preset["key"]


def test_empty_result_always_explains_itself(client):
    """Пустой список без объяснения бесполезен: оператор не понимает,
    искала система или нет."""
    for preset in client.get("/api/presets").json()["presets"]:
        res = client.post("/api/counterfactual", json=preset["sequence"][-1]).json()
        if not res["items"]:
            assert res["note"], preset["key"]
            for lang in ("ru", "kk", "en"):
                assert res["note"][lang].strip()


def test_amount_counterfactual_is_below_original(client):
    """Граница по сумме не может оказаться выше исходной: смягчение
    достигается уменьшением, а не увеличением."""
    for preset in client.get("/api/presets").json()["presets"]:
        tx = preset["sequence"][-1]
        res = client.post("/api/counterfactual", json=tx).json()
        for item in res["items"]:
            if item["field"] == "amount":
                assert item["to"] < item["from"]


def test_all_counterfactuals_speak_three_languages(client):
    for preset in client.get("/api/presets").json()["presets"]:
        res = client.post("/api/counterfactual", json=preset["sequence"][-1]).json()
        for item in res["items"]:
            for lang in ("ru", "kk", "en"):
                assert item["text"][lang].strip()
                assert "_" not in item["text"][lang], "сырое имя поля в тексте"


def test_allowed_transaction_has_nothing_to_soften(client):
    tx = dict(next(p for p in client.get("/api/presets").json()["presets"]
                   if p["key"] == "honest_big")["sequence"][-1])
    tx["amount"] = 900
    res = client.post("/api/counterfactual", json=tx).json()
    if res["baseline"]["action"] == "allow":
        assert res["items"] == []
        assert res["note"]


def test_counterfactual_does_not_mutate_state(client):
    """Поиск контрфактов прогоняет десятки кандидатов через скоринг —
    ни один не должен осесть в истории клиента."""
    tx = client.get("/api/presets").json()["presets"][0]["sequence"][-1]
    before = client.post("/api/score", json=tx).json()
    client.post("/api/counterfactual", json=tx)
    after = client.post("/api/score", json=tx).json()
    assert before["p_fraud"] == after["p_fraud"]
    assert before["decision"]["action"] == after["decision"]["action"]
