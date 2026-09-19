"""Проверки API и объяснений. Требуют обученной модели в models/."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HAS_MODEL = (ROOT / "models" / "fraud_hunter.joblib").exists()

pytestmark = pytest.mark.skipif(
    not HAS_MODEL, reason="нет обученной модели: выполните ml.generate_data и ml.train"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["clients"] > 0


@pytest.mark.parametrize(
    "path",
    ["/api/stats", "/api/metrics", "/api/feed?limit=5", "/api/graph",
     "/api/map", "/api/presets", "/api/economics",
     "/api/what-if?p_fraud=0.4&amount=250000"],
)
def test_endpoints_respond(client, path):
    assert client.get(path).status_code == 200


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Fraud" in r.text


def test_unknown_client_is_404(client):
    assert client.post("/api/score", json={"client_id": "НЕТ", "amount": 1000}).status_code == 404


def test_score_returns_full_explanation(client):
    presets = client.get("/api/presets").json()["presets"]
    tx = presets[0]["sequence"][0]
    body = client.post("/api/score", json=tx).json()

    assert 0.0 <= body["p_fraud"] <= 1.0
    assert body["decision"]["action"] in ("allow", "step_up", "hold", "block")
    assert body["reasons"], "решение без объяснения недопустимо"
    for lang in ("ru", "kk", "en"):
        assert lang in body["client_message"]
        assert lang in body["reasons"][0]["text"]
        assert body["reasons"][0]["text"][lang].strip()


def test_explanations_never_leak_raw_feature_names(client):
    """Объяснение читают оператор и клиент — технические коды туда попадать не должны."""
    feed = client.get("/api/feed?limit=60").json()["items"]
    assert feed
    for item in feed:
        for reason in item["reasons"]:
            for lang in ("ru", "kk", "en"):
                text = reason["text"][lang]
                assert "_" not in text.replace("—", ""), f"сырое имя признака в тексте: {text}"


def test_scoring_does_not_mutate_state(client):
    """Один и тот же запрос обязан давать один и тот же ответ."""
    tx = client.get("/api/presets").json()["presets"][1]["sequence"][0]
    first = client.post("/api/score", json=tx).json()
    for _ in range(3):
        again = client.post("/api/score", json=tx).json()
        assert again["p_fraud"] == first["p_fraud"]
        assert again["decision"]["action"] == first["decision"]["action"]


def test_sequence_is_repeatable_and_escalates(client):
    """Серия операций должна повышать оценку — и не портить состояние системы."""
    preset = next(p for p in client.get("/api/presets").json()["presets"]
                  if p["key"] == "stolen_card")
    first = client.post("/api/score-sequence", json={"transactions": preset["sequence"]}).json()
    second = client.post("/api/score-sequence", json={"transactions": preset["sequence"]}).json()

    ps = [s["p_fraud"] for s in first["steps"]]
    assert [s["p_fraud"] for s in second["steps"]] == ps, "сценарий не повторяем"
    assert ps[-1] > ps[0], "серия операций должна повышать подозрение"


def test_social_engineering_is_held_not_blocked(client):
    """Главный сценарий проекта: обманутому клиенту звонят, а не просто отказывают."""
    preset = next(p for p in client.get("/api/presets").json()["presets"]
                  if p["key"] == "social_eng")
    res = client.post("/api/score-sequence", json={"transactions": preset["sequence"]}).json()
    last = res["steps"][-1]
    assert last["dominant_kind"] == "social_eng"
    assert last["decision"]["action"] == "hold"
    # и клиент получает прямое предупреждение, а не формальную отписку
    assert "мошенник" in last["client_message"]["ru"].lower()
    assert last["operator_script"]["ru"]


def test_honest_big_purchase_is_not_blocked(client):
    preset = next(p for p in client.get("/api/presets").json()["presets"]
                  if p["key"] == "honest_big")
    res = client.post("/api/score-sequence", json={"transactions": preset["sequence"]}).json()
    assert res["steps"][-1]["decision"]["action"] in ("allow", "step_up")


def test_economics_override_changes_decision(client):
    tx = dict(next(p for p in client.get("/api/presets").json()["presets"]
                   if p["key"] == "social_eng")["sequence"][0])
    base = client.post("/api/score", json=tx).json()
    tx["economics"] = {"hold_op_cost": 900_000.0}
    changed = client.post("/api/score", json=tx).json()
    assert base["decision"]["action"] == "hold"
    assert changed["decision"]["action"] != "hold"


def test_what_if_curve_has_no_single_threshold(client):
    """Кривая решений должна содержать несколько переломов, а не один порог."""
    curve = client.get(
        "/api/what-if?p_fraud=0.3&amount=400000&tx_type=transfer&kind=stolen_card"
    ).json()["curve"]
    actions = [c["action"] for c in curve]
    switches = sum(1 for a, b in zip(actions, actions[1:]) if a != b)
    assert switches >= 2, f"ожидали несколько переломов, получили {switches}"


def test_decision_map_boundary_moves_with_amount(client):
    """Граница между действиями обязана ползти влево с ростом суммы.

    Это формальная проверка главного утверждения проекта: фиксированного
    порога нет. Если бы он был, граница была бы вертикальной линией и индекс
    первого «не пропускать» не зависел бы от суммы.
    """
    data = client.get("/api/decision-map?kind=social_eng&tx_type=transfer").json()
    grid, amounts = data["grid"], data["amounts"]

    def first_intervention(row):
        for j, action in enumerate(row):
            if action != "allow":
                return j
        return len(row)

    boundary = [first_intervention(r) for r in grid]
    assert amounts[0] < amounts[-1], "суммы должны идти по возрастанию"
    # чем крупнее сумма, тем раньше система вмешивается
    assert boundary[0] > boundary[-1]
    assert boundary == sorted(boundary, reverse=True), "граница обязана быть монотонной"
    # и это именно кривая, а не одна ступенька
    assert len(set(boundary)) >= 4


def test_report_endpoint_is_safe_without_report(client):
    """Сайт не должен падать, если отчёт ещё не считали."""
    body = client.get("/api/report").json()
    assert isinstance(body, dict)


def test_report_matches_claims_when_present(client):
    """Если отчёт есть, его числа должны быть осмысленными.

    Вкладка «Модель» показывает эти значения как доказательство, поэтому
    пустые или бессмысленные числа там недопустимы.
    """
    report = client.get("/api/report").json()
    if not report:
        pytest.skip("отчёт не считали: python -m ml.report")

    variants = report["ablation"]["variants"]
    assert len(variants) == 2
    assert variants[1]["n_features"] > variants[0]["n_features"]
    # сетевые признаки обязаны помогать, иначе их незачем держать
    assert variants[1]["pr_auc"] > variants[0]["pr_auc"]

    for row in report["cap_sweep"]["rows"]:
        assert 0.0 <= row["pr_auc_known"] <= 1.0
        assert 0.0 <= row["novel_recall"] <= 1.0

    kinds = {r["hidden"] for r in report["novel_scheme"]["results"]}
    assert kinds == {"stolen_card", "social_eng", "fraud_ring"}
    social = next(r for r in report["novel_scheme"]["results"]
                  if r["hidden"] == "social_eng")
    # главный заявленный результат: спрятанную соцтнженерию одна модель не видит,
    # а два канала ловят заметную долю
    assert social["blended_recall"] > social["supervised_recall"] + 0.2


def test_metrics_carry_training_curve(client):
    curve = client.get("/api/metrics").json().get("training_curve", {})
    assert len(curve.get("average_precision", [])) > 10
    assert curve["best_iteration"] > 0
