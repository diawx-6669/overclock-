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


def _cf(client, preset: dict) -> dict:
    """Контрфакты по всей серии — так их считает и интерфейс."""
    return client.post("/api/counterfactual",
                       json={"transactions": preset["sequence"]}).json()


def test_counterfactual_always_softens(client):
    """Контрфакт обязан предлагать более мягкое действие, а не любое другое.

    Предложение «сделайте так, и мы вас заблокируем вместо звонка» — не
    объяснение, а издевательство.
    """
    for preset in client.get("/api/presets").json()["presets"]:
        res = _cf(client, preset)
        base_rank = SEVERITY[res["baseline"]["action"]]
        for item in res["items"]:
            assert SEVERITY[item["action_to"]] < base_rank, preset["key"]


def test_empty_result_always_explains_itself(client):
    """Пустой список без объяснения бесполезен: оператор не понимает,
    искала система или нет."""
    for preset in client.get("/api/presets").json()["presets"]:
        res = _cf(client, preset)
        if not res["items"]:
            assert res["note"], preset["key"]
            for lang in ("ru", "kk", "en"):
                assert res["note"][lang].strip()


def test_amount_counterfactual_is_below_original(client):
    """Граница по сумме не может оказаться выше исходной: смягчение
    достигается уменьшением, а не увеличением."""
    for preset in client.get("/api/presets").json()["presets"]:
        res = _cf(client, preset)
        for item in res["items"]:
            if item["field"] == "amount":
                assert item["to"] < item["from"]


def test_all_counterfactuals_speak_three_languages(client):
    for preset in client.get("/api/presets").json()["presets"]:
        res = _cf(client, preset)
        for item in res["items"]:
            for lang in ("ru", "kk", "en"):
                assert item["text"][lang].strip()
                assert "_" not in item["text"][lang], "сырое имя поля в тексте"


def test_allowed_transaction_has_nothing_to_soften(client):
    preset = next(p for p in client.get("/api/presets").json()["presets"]
                  if p["key"] == "honest_big")
    tx = dict(preset["sequence"][-1])
    tx["amount"] = 900
    res = client.post("/api/counterfactual", json={"transactions": [tx]}).json()
    if res["baseline"]["action"] == "allow":
        assert res["items"] == []
        assert res["note"]


def test_counterfactual_does_not_mutate_state(client):
    """Поиск контрфактов прогоняет десятки кандидатов через скоринг —
    ни один не должен осесть в истории клиента."""
    preset = client.get("/api/presets").json()["presets"][0]
    tx = preset["sequence"][-1]
    before = client.post("/api/score", json=tx).json()
    client.post("/api/counterfactual", json={"transactions": preset["sequence"]})
    after = client.post("/api/score", json=tx).json()
    assert before["p_fraud"] == after["p_fraud"]
    assert before["decision"]["action"] == after["decision"]["action"]


# --------------------------------------------------------------------------
# Регрессии. Каждый тест ниже закрывает ошибку, которая реально была в коде.
# --------------------------------------------------------------------------


def _fake_scorer(boundaries):
    """Скоринг-заглушка: действие зависит только от суммы."""
    def score(tx):
        amount = float(tx["amount"])
        for limit, action in boundaries:
            if amount < limit:
                return {"decision": {"action": action},
                        "p_fraud": min(amount / 1e5, 1.0), "client": {}}
        return {"decision": {"action": "block"}, "p_fraud": 1.0, "client": {}}
    return score


def test_reported_action_matches_the_probed_amount():
    """Действие в ответе должно быть измерено, а не унаследовано от исходного.

    Было: best_action начинался со значения base_action, и если ни одна точка
    двоичного поиска не оказывалась мягче, контрфакт сообщал то же действие,
    что и в исходном решении — то есть «изменится на то же самое».
    """
    score = _fake_scorer([(1_000, "allow"), (5_000, "step_up"), (10**9, "hold")])
    res = find_counterfactuals({"amount": 40_000}, score)
    for item in res["items"]:
        assert item["action_to"] != item["action_from"]
        assert SEVERITY[item["action_to"]] < SEVERITY[item["action_from"]]


def test_amount_search_brackets_far_enough():
    """Вилка не должна упираться в фиксированный процент от суммы.

    Было: нижняя граница жёстко 1% от суммы. Если и там решение прежнее,
    поиск пропускался и система заявляла, что сумма не помогает вообще —
    хотя граница существовала ниже.
    """
    # граница на 500 ₸ лежит сильно ниже 1% от миллиона
    score = _fake_scorer([(500, "allow"), (10**9, "hold")])
    res = find_counterfactuals({"amount": 1_000_000}, score)
    amounts = [i for i in res["items"] if i["field"] == "amount"]
    assert amounts, "граница существует, но не найдена"
    assert amounts[0]["to"] < 500


def test_rounded_amount_is_rechecked_and_never_zero():
    """Округление не должно перешагивать найденную границу и обнулять сумму.

    Было: round(best, -2) уходил в ответ без повторной проверки и мог как
    перешагнуть границу вверх, так и превратиться в «не превышала 0 ₸».
    """
    score = _fake_scorer([(150, "allow"), (10**9, "hold")])
    res = find_counterfactuals({"amount": 900_000}, score)
    for item in res["items"]:
        if item["field"] == "amount":
            assert item["to"] > 0, "сумма в объяснении не может быть нулевой"
            probe = {"amount": item["to"]}
            assert score(probe)["decision"]["action"] == item["action_to"]


def test_city_change_also_changes_country():
    """Смена города обязана менять страну.

    Было: probe получал «Шымкент» со страной TR, признак «операция из-за
    рубежа» оставался поднятым, и контрфакт объяснял несуществующую операцию.
    """
    from backend.counterfactual import _apply

    probe = _apply({"city": "Стамбул", "country": "TR"}, "city", "Шымкент")
    assert probe["country"] == "KZ"
    probe = _apply({"city": "Алматы", "country": "KZ"}, "city", "Дубай")
    assert probe["country"] == "AE"


def test_counterfactual_explains_the_decision_it_was_shown_with(client):
    """Контрфакт обязан объяснять именно то решение, что показано в разборе.

    Было: контрфакт считался по одной операции против глобального состояния,
    а решение в серии принималось в песочнице с учётом предыдущих операций.
    На тихой краже карты разбор показывал «задержать» при вероятности 0.85,
    а контрфакт рассуждал про «подтвердить» при 0.03 — то есть объяснял
    другое решение.
    """
    for preset in client.get("/api/presets").json()["presets"]:
        body = {"transactions": preset["sequence"]}
        last = client.post("/api/score-sequence", json=body).json()["steps"][-1]
        cf = client.post("/api/counterfactual", json=body).json()
        assert cf["baseline"]["action"] == last["decision"]["action"], preset["key"]
        assert cf["baseline"]["p_fraud"] == pytest.approx(last["p_fraud"]), preset["key"]


def test_counterfactual_honours_custom_economics(client):
    """Ползунки экономики должны доходить и до контрфактов.

    Было: переопределение экономики уходило в скоринг серии, но не в
    контрфакт, и панель объясняла решение, принятое при других параметрах.
    """
    body = {"transactions": next(
        p for p in client.get("/api/presets").json()["presets"]
        if p["key"] == "social_eng")["sequence"]}
    base = client.post("/api/counterfactual", json=body).json()
    body["economics"] = {"hold_op_cost": 900_000.0}
    changed = client.post("/api/counterfactual", json=body).json()
    assert base["baseline"]["action"] == "hold"
    assert changed["baseline"]["action"] != "hold"
