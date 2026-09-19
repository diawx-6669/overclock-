"""FastAPI Fraud Hunter: онлайн-скоринг, объяснения и данные для дашборда.

Бэкенд отдаёт и API, и сам фронтенд — это один сервис, который поднимается
одной командой и одинаково работает локально и на Render.

Важная деталь онлайн-скоринга: POST /api/score не меняет состояние клиента.
Симулятор на демо можно гонять сколько угодно, и один и тот же сценарий всегда
даст один и тот же ответ. Реальный поток учитывается отдельной ручкой
/api/observe — в проде именно она вызывалась бы после подтверждения операции.
"""

from __future__ import annotations

import json
import math
import zlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.cost import (
    ACTION_LABELS_RU,
    ACTIONS,
    ALLOW,
    BLOCK,
    DEFAULT_ECONOMICS,
    Economics,
    decide,
    expected_costs,
)
from backend.explain import (
    Explainer,
    client_message,
    dominant_kind,
    operator_script,
    reasons_from_stored,
)
from ml.common import (
    ALL_PLACES,
    CATEGORY_LIST,
    CHANNELS,
    CITIES,
    TX_TYPES,
    place_coords,
    read_transactions,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
FRONTEND = ROOT / "frontend"

STATE: dict[str, Any] = {}


# --------------------------------------------------------------------------
# Загрузка артефактов
# --------------------------------------------------------------------------


def load_artifacts() -> None:
    bundle_path = MODELS_DIR / "fraud_hunter.joblib"
    store_path = MODELS_DIR / "feature_store.joblib"
    scored_path = DATA_DIR / "scored_test.csv"

    missing = [p.name for p in (bundle_path, store_path, scored_path) if not p.exists()]
    if missing:
        raise RuntimeError(
            "Не хватает артефактов: "
            + ", ".join(missing)
            + ". Сначала выполните: python -m ml.generate_data && python -m ml.train"
        )

    bundle = joblib.load(bundle_path)
    STATE["model"] = bundle["model"]
    STATE["calibrator"] = bundle["calibrator"]
    STATE["anomaly_channel"] = bundle.get("anomaly_channel")
    STATE["combiner"] = bundle.get("combiner")
    STATE["type_model"] = bundle["type_model"]
    STATE["type_classes"] = bundle["type_classes"]
    STATE["features"] = bundle["features"]
    STATE["metrics"] = bundle["metrics"]
    STATE["store"] = joblib.load(store_path)
    STATE["explainer"] = Explainer(bundle["model"], bundle["features"])

    feed = read_transactions(scored_path)
    feed = feed.sort_values("timestamp").reset_index(drop=True)
    STATE["feed"] = feed

    report_path = MODELS_DIR / "report.json"
    STATE["report"] = (
        json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    )

    graph_path = MODELS_DIR / "graph.json"
    STATE["graph"] = (
        json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {}
    )
    STATE["economics"] = DEFAULT_ECONOMICS


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="Fraud Hunter",
    description="Антифрод с объяснимыми решениями и выбором действия по ожидаемым потерям",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Схемы запросов
# --------------------------------------------------------------------------


class TransactionIn(BaseModel):
    client_id: str = Field(..., description="Идентификатор клиента, например C00042")
    amount: float = Field(..., gt=0, description="Сумма в тенге")
    merchant_category: str = "transfer_p2p"
    channel: str = "mobile_app"
    tx_type: str = "transfer"
    city: str = ""
    country: str = "KZ"
    device_id: str = ""
    ip: str = ""
    is_vpn: int = 0
    remote_access: int = 0
    call_minutes_before: float = 0.0
    session_duration_s: float = 60.0
    recipient_id: str = ""
    timestamp: str = ""
    merchant_id: str = "M0001"
    # Необязательное переопределение экономики — для ползунков на дашборде
    economics: dict[str, float] | None = None


class ObserveIn(TransactionIn):
    pass


class SequenceIn(BaseModel):
    """Серия операций подряд — так работает настоящая атака."""

    transactions: list[TransactionIn] = Field(..., min_length=1, max_length=10)
    economics: dict[str, float] | None = None


# --------------------------------------------------------------------------
# Скоринг
# --------------------------------------------------------------------------


def _economics_from(overrides: dict | None) -> Economics:
    """Экономику можно переопределить прямо из запроса — этим живут ползунки."""
    if not overrides:
        return STATE["economics"]
    base = DEFAULT_ECONOMICS.to_dict()
    base.update({k: v for k, v in overrides.items() if k in base})
    return Economics(**base)


def _resolve_defaults(tx: dict) -> dict:
    """Дозаполнить поля, которые клиент API мог не прислать."""
    store = STATE["store"]
    st = store.clients.get(tx["client_id"])
    if st is None:
        raise HTTPException(404, f"Клиент {tx['client_id']} не найден")

    if not tx.get("timestamp"):
        tx["timestamp"] = str(pd.Timestamp("2025-10-01 12:00:00"))
    if not tx.get("city"):
        tx["city"] = st.home_city
    if not tx.get("device_id"):
        known = sorted(st.devices, key=lambda d: -st.devices[d][0])
        tx["device_id"] = known[0] if known else f"{tx['client_id']}-dev"
    if not tx.get("ip"):
        # crc32 вместо hash(): один и тот же клиент обязан получать один и тот
        # же IP-заполнитель после каждого перезапуска сервиса
        tx["ip"] = f"10.{zlib.crc32(tx['client_id'].encode()) % 250}.0.1"
    if tx.get("tx_type") == "transfer" and not tx.get("recipient_id"):
        tx["recipient_id"] = "R00000"
    return tx


def score_transaction(tx: dict, econ: Economics | None = None, store=None) -> dict:
    """Полный цикл: признаки → вероятность → тип схемы → решение → объяснение."""
    store = store or STATE["store"]
    features = STATE["features"]
    econ = econ or STATE["economics"]

    feats = store.features(tx)
    # Именно DataFrame с теми же колонками, что и при обучении: иначе
    # sklearn справедливо предупреждает о потере имён признаков
    row = pd.DataFrame([[feats[f] for f in features]], columns=features)

    raw = float(STATE["model"].predict_proba(row)[0, 1])
    p_supervised = float(STATE["calibrator"].predict([raw])[0])
    p_supervised = min(max(p_supervised, 0.0), 1.0)

    # Второй канал: «не похоже на нормальное поведение». Он не знает разметки
    # и потому не слепнет на схеме, которой не было в обучении.
    channel = STATE.get("anomaly_channel")
    combiner = STATE.get("combiner")
    anomaly_score = None
    p_anomaly = None
    p = p_supervised
    if channel is not None and combiner is not None:
        anomaly_score = float(channel.score(row)[0])
        p_anomaly = float(combiner.anomaly_probability([anomaly_score])[0])
        p = float(combiner.predict([p_supervised], [anomaly_score])[0])
    p = min(max(p, 0.0), 1.0)

    type_probs_arr = STATE["type_model"].predict_proba(row)[0]
    type_probs = {c: float(type_probs_arr[i]) for i, c in enumerate(STATE["type_classes"])}

    summary = store.client_summary(tx["client_id"])
    decision = decide(p, float(tx["amount"]), str(tx.get("tx_type", "purchase")),
                      type_probs, summary, econ)

    reasons, _logit = STATE["explainer"].explain_one(feats, tx)

    return {
        "transaction": {k: v for k, v in tx.items() if k != "economics"},
        "p_fraud": round(p, 6),
        "p_fraud_raw": round(raw, 6),
        "p_supervised": round(p_supervised, 6),
        "p_anomaly": round(p_anomaly, 6) if p_anomaly is not None else None,
        "anomaly_score": round(anomaly_score, 4) if anomaly_score is not None else None,
        "fraud_type_probs": {k: round(v, 4) for k, v in type_probs.items()},
        "dominant_kind": dominant_kind(type_probs),
        "decision": decision.to_dict(),
        "reasons": [r.to_dict() for r in reasons],
        "client_message": client_message(decision.action, float(tx["amount"]), type_probs),
        "operator_script": operator_script(decision.action, type_probs),
        "client": summary,
        "features": {k: round(float(v), 4) for k, v in feats.items()},
    }


@app.post("/api/score")
def api_score(payload: TransactionIn) -> dict:
    """Оценить транзакцию и выбрать действие. Состояние клиента не меняется."""
    tx = payload.model_dump()
    econ = _economics_from(tx.pop("economics", None))
    tx = _resolve_defaults(tx)
    return score_transaction(tx, econ)


@app.post("/api/score-sequence")
def api_score_sequence(payload: SequenceIn) -> dict:
    """Проиграть серию операций так, как её увидела бы система в реальном времени.

    Одна изолированная покупка за границей с нового устройства — это с равным
    успехом и кража, и командировка, и система честно отвечает «не знаю». Но три
    покупки подряд за двадцать минут — это уже совсем другая картина. Здесь
    видно, как оценка растёт от шага к шагу.

    Считается в песочнице: настоящее состояние клиентов не меняется, один и
    тот же сценарий всегда даёт один и тот же результат.
    """
    econ = _economics_from(payload.economics)
    sandbox = STATE["store"].sandbox()

    steps = []
    for item in payload.transactions:
        tx = _resolve_defaults(item.model_dump(exclude={"economics"}))
        result = score_transaction(tx, econ, store=sandbox)
        sandbox.observe(tx)
        steps.append(result)

    return {
        "steps": steps,
        "final_action": steps[-1]["decision"]["action"],
        "max_p_fraud": max(s["p_fraud"] for s in steps),
        "total_amount": sum(float(s["transaction"]["amount"]) for s in steps),
    }


@app.post("/api/observe")
def api_observe(payload: ObserveIn) -> dict:
    """Учесть транзакцию в истории клиента (в проде — после проведения)."""
    tx = _resolve_defaults(payload.model_dump(exclude={"economics"}))
    STATE["store"].observe(tx)
    return {"ok": True, "client": STATE["store"].client_summary(tx["client_id"])}


# --------------------------------------------------------------------------
# Дашборд
# --------------------------------------------------------------------------


@app.get("/api/health")
def api_health() -> dict:
    return {
        "status": "ok" if "model" in STATE else "loading",
        "clients": len(STATE.get("store").clients) if STATE.get("store") else 0,
        "feed_rows": int(len(STATE["feed"])) if "feed" in STATE else 0,
    }


@app.get("/api/metrics")
def api_metrics() -> dict:
    """Метрики качества и экономики с последнего обучения."""
    return STATE["metrics"]


@app.get("/api/report")
def api_report() -> dict:
    """Замеры, которыми подкреплены заявления: абляция, подбор потолка,
    эксперимент со схемой, которой модель не видела.

    Собирается командой `python -m ml.report`. Пустой ответ означает, что
    отчёт ещё не считали — сайт от этого не ломается, вкладка просто скажет,
    что данных нет.
    """
    return STATE.get("report") or {}


@app.get("/api/stats")
def api_stats() -> dict:
    """Агрегаты по тестовому периоду — это то, что видно на главном экране."""
    feed: pd.DataFrame = STATE["feed"]
    metrics = STATE["metrics"]

    total = int(len(feed))
    fraud = int(feed["is_fraud"].sum())
    by_action = feed["action"].value_counts().to_dict()

    caught = feed[(feed.is_fraud == 1) & (feed.action != ALLOW)]
    missed = feed[(feed.is_fraud == 1) & (feed.action == ALLOW)]
    friction = feed[(feed.is_fraud == 0) & (feed.action != ALLOW)]

    # Динамика по дням
    daily = (
        feed.assign(day=feed["timestamp"].dt.date)
        .groupby("day")
        .agg(
            transactions=("tx_id", "count"),
            fraud=("is_fraud", "sum"),
            amount=("amount", "sum"),
            blocked=("action", lambda s: int((s == BLOCK).sum())),
        )
        .reset_index()
    )
    daily["day"] = daily["day"].astype(str)

    return {
        "period": {
            "from": str(feed["timestamp"].min()),
            "to": str(feed["timestamp"].max()),
        },
        "transactions": total,
        "turnover": float(feed["amount"].sum()),
        "fraud_count": fraud,
        "fraud_rate": fraud / total if total else 0.0,
        "fraud_amount": float(feed.loc[feed.is_fraud == 1, "amount"].sum()),
        "actions": {a: int(by_action.get(a, 0)) for a in ACTIONS},
        "caught_count": int(len(caught)),
        "caught_amount": float(caught["amount"].sum()),
        "missed_count": int(len(missed)),
        "missed_amount": float(missed["amount"].sum()),
        "friction_count": int(len(friction)),
        "friction_rate": float(len(friction) / max((feed.is_fraud == 0).sum(), 1)),
        "detection_rate": float(len(caught) / max(fraud, 1)),
        "cost_ours": metrics["cost_ours"],
        "cost_baselines": metrics["cost_baselines"],
        "saving_vs_threshold": float(
            min(v for k, v in metrics["cost_baselines"].items() if k != "no_system")
            - metrics["cost_ours"]
        ),
        "saving_vs_nothing": float(
            metrics["cost_baselines"]["no_system"] - metrics["cost_ours"]
        ),
        "per_type": metrics["per_type"],
        "daily": daily.to_dict("records"),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
    }


@app.get("/api/feed")
def api_feed(
    limit: int = Query(40, ge=1, le=300),
    action: Literal["all", "allow", "step_up", "hold", "block"] = "all",
    only_fraud: bool = False,
) -> dict:
    """Лента последних решений — то, что видит дежурный аналитик."""
    feed: pd.DataFrame = STATE["feed"]
    view = feed
    if action != "all":
        view = view[view["action"] == action]
    if only_fraud:
        view = view[view["is_fraud"] == 1]
    # Сначала интересное: всё, что не «пропустить», затем по времени
    view = view.sort_values(["timestamp"], ascending=False).head(limit)

    store = STATE["store"]
    items = []
    for row in view.to_dict("records"):
        stored = json.loads(row.get("top_reasons") or "[]")
        feats = store.features(row)
        reasons = reasons_from_stored(stored, feats, row)
        type_probs = {
            c: float(row.get(f"p_{c}", 0.0)) for c in STATE["type_classes"]
        }
        items.append(
            {
                "tx_id": row["tx_id"],
                "timestamp": str(row["timestamp"]),
                "client_id": row["client_id"],
                "amount": float(row["amount"]),
                "city": row["city"],
                "country": row["country"],
                "merchant_category": row["merchant_category"],
                "channel": row["channel"],
                "tx_type": row["tx_type"],
                "p_fraud": float(row["p_fraud"]),
                "action": row["action"],
                "action_label": ACTION_LABELS_RU.get(row["action"], row["action"]),
                "is_fraud": int(row["is_fraud"]),
                "fraud_type": row.get("fraud_type") if isinstance(row.get("fraud_type"), str) else "",
                "dominant_kind": dominant_kind(type_probs),
                "fraud_type_probs": {k: round(v, 3) for k, v in type_probs.items()},
                "reasons": [r.to_dict() for r in reasons],
            }
        )
    return {"count": len(items), "items": items}


@app.get("/api/clients/{client_id}")
def api_client(client_id: str) -> dict:
    summary = STATE["store"].client_summary(client_id)
    if not summary:
        raise HTTPException(404, f"Клиент {client_id} не найден")
    feed: pd.DataFrame = STATE["feed"]
    history = feed[feed["client_id"] == client_id].sort_values("timestamp").tail(30)
    summary["recent"] = [
        {
            "tx_id": r["tx_id"],
            "timestamp": str(r["timestamp"]),
            "amount": float(r["amount"]),
            "city": r["city"],
            "merchant_category": r["merchant_category"],
            "p_fraud": float(r["p_fraud"]),
            "action": r["action"],
            "is_fraud": int(r["is_fraud"]),
        }
        for r in history.to_dict("records")
    ]
    return summary


@app.get("/api/graph")
def api_graph(min_clients: int = Query(3, ge=2, le=20), limit: int = Query(6, ge=1, le=20)) -> dict:
    """Граф связей: устройства и IP, за которыми стоит больше одного клиента.

    Именно так на экране проявляются кольца карт — одна точка, из которой
    расходятся лучи к десятку разных людей.

    Сводка считается на этапе обучения по всему месяцу, а не по тестовому
    срезу. Кольцо живёт во времени и не обязано целиком попасть в последнюю
    четверть периода: построив граф только по тестовым дням, мы теряли часть
    колец и показывали у остальных нулевую долю фрода, потому что их операции
    остались в обучающей выборке.
    """
    summary = STATE.get("graph") or {}
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_clients: set[str] = set()

    for kind in ("device", "ip"):
        hubs = [h for h in summary.get(kind, []) if h["clients"] >= min_clients][:limit]
        for hub in hubs:
            hub_id = f"{kind}:{hub['key']}"
            nodes.append(
                {
                    "id": hub_id,
                    "type": kind,
                    "label": hub["key"],
                    "clients": hub["clients"],
                    "transactions": hub["transactions"],
                    "amount": hub["amount"],
                    "fraud_share": hub["fraud_share"],
                }
            )
            for member in hub["members"]:
                cid = member["client_id"]
                if cid not in seen_clients:
                    seen_clients.add(cid)
                    nodes.append(
                        {
                            "id": f"client:{cid}",
                            "type": "client",
                            "label": cid,
                            "transactions": member["transactions"],
                            "amount": member["amount"],
                            "fraud_share": member["fraud_share"],
                        }
                    )
                edges.append(
                    {
                        "source": hub_id,
                        "target": f"client:{cid}",
                        "weight": member["transactions"],
                        "amount": member["amount"],
                    }
                )

    return {"nodes": nodes, "edges": edges}


@app.get("/api/map")
def api_map() -> dict:
    """Города с объёмом, фродом и заблокированными суммами."""
    feed: pd.DataFrame = STATE["feed"]
    grouped = feed.groupby("city").agg(
        transactions=("tx_id", "count"),
        amount=("amount", "sum"),
        fraud=("is_fraud", "sum"),
        fraud_amount=("amount", lambda s: float(s[feed.loc[s.index, "is_fraud"] == 1].sum())),
        blocked=("action", lambda s: int((s == BLOCK).sum())),
    )
    points = []
    for city, row in grouped.iterrows():
        lat, lon = place_coords(str(city))
        points.append(
            {
                "city": str(city),
                "lat": lat,
                "lon": lon,
                # Зарубежные точки рисуются отдельно: иначе Лагос и Бангкок
                # растягивают рамку так, что весь Казахстан сжимается в пятно
                "foreign": ALL_PLACES.get(str(city), (0, 0, "KZ"))[2] != "KZ",
                "country": ALL_PLACES.get(str(city), (0, 0, "KZ"))[2],
                "transactions": int(row["transactions"]),
                "amount": float(row["amount"]),
                "fraud": int(row["fraud"]),
                "fraud_amount": float(row["fraud_amount"]),
                "blocked": int(row["blocked"]),
                "fraud_rate": float(row["fraud"] / max(row["transactions"], 1)),
            }
        )
    points.sort(key=lambda p: -p["transactions"])
    return {"points": points}


@app.get("/api/what-if")
def api_what_if(
    p_fraud: float = Query(..., ge=0.0, le=1.0),
    amount: float = Query(..., gt=0),
    tx_type: str = "transfer",
    kind: str = "social_eng",
) -> dict:
    """Кривая решений: как меняется выбор при росте вероятности.

    Это ответ на главный вопрос жюри — «покажите, что порога правда нет».
    """
    type_probs = {k: 0.05 for k in ("stolen_card", "social_eng", "fraud_ring")}
    type_probs[kind] = 0.9
    grid = [round(x / 100, 2) for x in range(0, 101, 2)]
    curve = []
    for p in grid:
        costs = expected_costs(p, amount, tx_type, type_probs)
        best = min(costs, key=costs.get)
        curve.append({"p": p, "action": best, "costs": costs})
    current = decide(p_fraud, amount, tx_type, type_probs)
    return {"curve": curve, "current": current.to_dict()}


@app.get("/api/decision-map")
def api_decision_map(
    tx_type: str = "transfer",
    kind: str = "social_eng",
    amount_min: float = Query(1_000, gt=0),
    amount_max: float = Query(5_000_000, gt=0),
    n_amount: int = Query(26, ge=4, le=60),
    n_p: int = Query(41, ge=5, le=101),
) -> dict:
    """Карта решений: какое действие дешевле при каждой паре (вероятность, сумма).

    Это самый прямой ответ на вопрос «а где у вас порог». Порога нет: граница
    между действиями — кривая, которая ползёт влево с ростом суммы. При 3 000 ₸
    система терпит риск в 40%, при 3 000 000 ₸ вмешивается уже на 2%.
    """
    type_probs = {k: 0.05 for k in ("stolen_card", "social_eng", "fraud_ring")}
    if kind in type_probs:
        type_probs[kind] = 0.9

    lo, hi = math.log(amount_min), math.log(max(amount_max, amount_min * 2))
    amounts = [math.exp(lo + (hi - lo) * i / (n_amount - 1)) for i in range(n_amount)]
    ps = [i / (n_p - 1) for i in range(n_p)]

    grid = []
    for amount in amounts:
        row = []
        for p in ps:
            costs = expected_costs(p, amount, tx_type, type_probs)
            row.append(min(costs, key=costs.get))
        grid.append(row)

    return {
        "amounts": [round(a, 2) for a in amounts],
        "probabilities": [round(p, 4) for p in ps],
        "grid": grid,
        "tx_type": tx_type,
        "kind": kind,
    }


@app.get("/api/presets")
def api_presets() -> dict:
    """Готовые сценарии для симулятора.

    Каждый сценарий — это последовательность, а не одна операция. Так и
    происходит в жизни: мошенник почти никогда не делает ровно один платёж,
    а система принимает решение по накопленной картине.
    """
    store = STATE["store"]

    def pick(predicate, fallback: str = "C00000") -> str:
        for cid, st in store.clients.items():
            if st.n_tx >= 15 and predicate(st):
                return cid
        return fallback

    elderly = pick(lambda st: st.age >= 60)
    typical = pick(lambda st: 25 <= st.age <= 45)

    def device_of(cid: str) -> str:
        st = store.clients[cid]
        known = sorted(st.devices, key=lambda d: -st.devices[d][0])
        return known[0] if known else f"{cid}-dev"

    elderly_st = store.clients[elderly]
    typical_st = store.clients[typical]
    e_avg = elderly_st.mean_amount
    t_avg = typical_st.mean_amount

    def at(minute: int) -> str:
        base = pd.Timestamp("2025-10-01 13:00:00") + pd.Timedelta(minutes=minute)
        return str(base)

    def tx(**kw) -> dict:
        base = {
            "client_id": typical,
            "merchant_category": "transfer_p2p",
            "channel": "mobile_app",
            "tx_type": "transfer",
            "country": "KZ",
            "is_vpn": 0,
            "remote_access": 0,
            "call_minutes_before": 0,
            "session_duration_s": 90,
        }
        base.update(kw)
        return base

    presets = [
        {
            "key": "social_eng",
            "title": "«Звонят из банка» — клиент платит сам",
            "hint": (
                "Своё устройство, свой город, без VPN. Формально придраться не к чему — "
                "и именно поэтому обычные системы такое пропускают. Мошенник дробит "
                "сумму на две части."
            ),
            "expect": "Задержать и позвонить",
            "sequence": [
                tx(client_id=elderly, amount=round(e_avg * 7, -4) or 150_000,
                   city=elderly_st.home_city, device_id=device_of(elderly),
                   remote_access=1, call_minutes_before=38, session_duration_s=55,
                   recipient_id="R77777", timestamp=at(0)),
                tx(client_id=elderly, amount=round(e_avg * 9, -4) or 200_000,
                   city=elderly_st.home_city, device_id=device_of(elderly),
                   remote_access=1, call_minutes_before=52, session_duration_s=44,
                   recipient_id="R77777", timestamp=at(11)),
            ],
        },
        {
            "key": "stolen_card",
            "title": "Украденная карта — громкая",
            "hint": (
                "Чужое устройство, заграница, VPN, дорогая электроника. Первая покупка "
                "сама по себе спорная: так же выглядит командировка. Серия расставляет всё "
                "по местам."
            ),
            "expect": "Заблокировать",
            "sequence": [
                tx(client_id=typical, amount=round(t_avg * 6, -3) or 120_000,
                   merchant_category="electronics", channel="ecom", tx_type="purchase",
                   city="Стамбул", country="TR", device_id="DX00001", ip="185.42.7.19",
                   is_vpn=1, session_duration_s=38, timestamp=at(0)),
                tx(client_id=typical, amount=round(t_avg * 9, -3) or 180_000,
                   merchant_category="electronics", channel="ecom", tx_type="purchase",
                   city="Стамбул", country="TR", device_id="DX00001", ip="185.42.7.19",
                   is_vpn=1, session_duration_s=25, timestamp=at(9)),
                tx(client_id=typical, amount=round(t_avg * 14, -3) or 280_000,
                   merchant_category="crypto", channel="ecom", tx_type="purchase",
                   city="Стамбул", country="TR", device_id="DX00001", ip="185.42.7.19",
                   is_vpn=1, session_duration_s=21, timestamp=at(17)),
            ],
        },
        {
            "key": "quiet_stolen",
            "title": "Украденная карта — тихая",
            "hint": (
                "Своя страна, без VPN, суммы почти обычные, обычный маркетплейс. "
                "Ни одно правило не срабатывает — держится только на новом устройстве "
                "и ритме операций."
            ),
            "expect": "Подтверждение, затем звонок",
            "sequence": [
                tx(client_id=typical, amount=round(t_avg * 1.6, -3) or 22_000,
                   merchant_category="marketplace", channel="ecom", tx_type="purchase",
                   city=typical_st.home_city, device_id="DX04242", ip="95.56.12.44",
                   session_duration_s=52, timestamp=at(0)),
                tx(client_id=typical, amount=round(t_avg * 2.2, -3) or 30_000,
                   merchant_category="marketplace", channel="ecom", tx_type="purchase",
                   city=typical_st.home_city, device_id="DX04242", ip="95.56.12.44",
                   session_duration_s=41, timestamp=at(26)),
            ],
        },
        {
            "key": "fraud_ring",
            "title": "Кольцо карт",
            "hint": "Одно устройство и один IP на десяток разных клиентов. Видно только по графу связей.",
            "expect": "Задержать и позвонить",
            "sequence": [
                tx(client_id=typical, amount=round(t_avg * 4, -3) or 60_000,
                   merchant_category="crypto", channel="ecom", tx_type="purchase",
                   city=typical_st.home_city, device_id="DR001", ip="77.88.99.100",
                   is_vpn=1, session_duration_s=40, timestamp=at(0)),
            ],
        },
        {
            "key": "honest_big",
            "title": "Честная крупная покупка",
            "hint": "Клиент сам, со своего телефона, долго выбирал и купил дорогое. Мешать нельзя.",
            "expect": "Пропустить",
            "sequence": [
                tx(client_id=typical, amount=round(t_avg * 8, -3) or 110_000,
                   merchant_category="electronics", channel="mobile_app", tx_type="purchase",
                   city=typical_st.home_city, device_id=device_of(typical),
                   session_duration_s=240, timestamp=at(0)),
            ],
        },
        {
            "key": "honest_trip",
            "title": "Честная командировка",
            "hint": (
                "Новый телефон, другая страна, VPN — набор признаков как у кражи, "
                "но это правда клиент. Проверка на ложные срабатывания."
            ),
            "expect": "Пропустить или подтверждение",
            "sequence": [
                tx(client_id=typical, amount=round(t_avg * 3, -3) or 40_000,
                   merchant_category="travel", channel="ecom", tx_type="purchase",
                   city="Дубай", country="AE", device_id=device_of(typical) + "T007",
                   is_vpn=1, session_duration_s=210, timestamp=at(0)),
            ],
        },
    ]

    return {
        "presets": presets,
        "reference": {
            "categories": CATEGORY_LIST,
            "channels": CHANNELS,
            "tx_types": TX_TYPES,
            "cities": list(CITIES) + ["Стамбул", "Дубай", "Москва", "Бангкок", "Лондон"],
        },
    }


@app.get("/api/economics")
def api_economics() -> dict:
    """Текущая стоимостная модель — её показываем и объясняем на дашборде."""
    return STATE["economics"].to_dict()


# --------------------------------------------------------------------------
# Фронтенд
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/favicon.ico")
def favicon() -> JSONResponse:
    return JSONResponse({}, status_code=204)


def run() -> None:
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
