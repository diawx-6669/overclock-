"""Замер задержки: сколько система думает над одной операцией.

Банку этот вопрос важен сразу после точности. Авторизация карты укладывается
в сотни миллисекунд на весь путь, и антифрод — только одна из его частей:
если модель думает полсекунды, её не поставят в онлайн, какой бы точной она
ни была.

Меряем не «предсказание модели», а полный путь, как он идёт в проде: сборка
признаков по состоянию клиента и сети, обе модели, расчёт ожидаемых потерь
по четырём действиям и построение объяснения. Отдельно показываем разбивку
по этапам, потому что оптимизировать имеет смысл только то, что реально
занимает время.

Запуск:
    python -m ml.bench
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.cost import decide
from backend.explain import Explainer
from ml.common import read_transactions

MODELS_DIR = Path("models")


def percentiles(samples: list[float]) -> dict[str, float]:
    a = np.array(samples, dtype=float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Замер задержки решения")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", default=str(MODELS_DIR / "bench.json"))
    args = parser.parse_args()

    bundle = joblib.load(MODELS_DIR / "fraud_hunter.joblib")
    store = joblib.load(MODELS_DIR / "feature_store.joblib")
    model, calibrator = bundle["model"], bundle["calibrator"]
    type_model, features = bundle["type_model"], bundle["features"]
    channel, combiner = bundle.get("anomaly_channel"), bundle.get("combiner")
    explainer = Explainer(model, features)

    # Берём настоящие операции, а не одну и ту же по кругу: разные клиенты
    # имеют разную историю, и время сборки признаков от неё зависит
    df = read_transactions("data/transactions.csv").tail(args.runs + args.warmup)
    rows = df.to_dict("records")

    stages: dict[str, list[float]] = {
        "признаки": [], "модели": [], "решение": [], "объяснение": [], "всего": []
    }

    for i, tx in enumerate(rows):
        t_all = time.perf_counter()

        t = time.perf_counter()
        feats = store.features(tx)
        row = pd.DataFrame([[feats[f] for f in features]], columns=features)
        t_features = time.perf_counter() - t

        t = time.perf_counter()
        raw = float(model.predict_proba(row)[0, 1])
        p = float(calibrator.predict([raw])[0])
        type_probs_arr = type_model.predict_proba(row)[0]
        if channel is not None and combiner is not None:
            p = float(combiner.predict([p], [float(channel.score(row)[0])])[0])
        t_models = time.perf_counter() - t

        t = time.perf_counter()
        type_probs = {c: float(type_probs_arr[j])
                      for j, c in enumerate(bundle["type_classes"])}
        decide(p, float(tx["amount"]), str(tx.get("tx_type", "purchase")),
               type_probs, store.client_summary(str(tx["client_id"])))
        t_decide = time.perf_counter() - t

        t = time.perf_counter()
        explainer.explain_one(feats, tx)
        t_explain = time.perf_counter() - t

        if i < args.warmup:
            continue
        stages["признаки"].append(t_features * 1000)
        stages["модели"].append(t_models * 1000)
        stages["решение"].append(t_decide * 1000)
        stages["объяснение"].append(t_explain * 1000)
        stages["всего"].append((time.perf_counter() - t_all) * 1000)

    report = {
        "runs": len(stages["всего"]),
        "stages": {name: percentiles(vals) for name, vals in stages.items()},
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }
    # Сколько операций в час выдержит один процесс на этой машине
    report["throughput_per_hour"] = int(3600 / (report["stages"]["всего"]["mean"] / 1000))

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print(f"Замерено операций: {report['runs']}\n")
    print(f"{'этап':<14}{'p50':>9}{'p95':>9}{'p99':>9}{'макс':>9}")
    for name, v in report["stages"].items():
        mark = "  <-" if name == "всего" else ""
        print(f"{name:<14}{v['p50']:>8.2f}м{v['p95']:>8.2f}м{v['p99']:>8.2f}м"
              f"{v['max']:>8.2f}м{mark}")
    print(f"\nПропускная способность одного процесса: "
          f"{report['throughput_per_hour']:,} операций в час")
    print(f"Месячный поток кейса — 100 000 операций, то есть около 140 в час.")
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
