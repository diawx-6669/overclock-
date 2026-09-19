"""Полный отчёт по модели: всё, чем в README подкреплены заявления.

Каждое число в README должно быть воспроизводимо одной командой, иначе это
не результат, а обещание. Здесь собраны три замера:

  1. Абляция признаков — сколько дают сетевые признаки на одних и тех же
     данных и том же разбиении.
  2. Подбор потолка канала аномалий — цена страховки против её пользы.
  3. Схема, которой модель не видела — главный ответ на «вы ловите только
     то, что уже случалось».

Отчёт складывается в models/report.json и показывается на сайте во вкладке
«Модель»: жюри видит не пересказ, а те же самые цифры.

Запуск (несколько минут — здесь обучается около десятка моделей):
    python -m ml.report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ml.anomaly import ANOMALY_FEATURES, AnomalyChannel, NoisyOr
from ml.common import FRAUD_TYPES, read_transactions
from ml.features import FEATURES, build_matrix
from ml.train import time_split, train_binary

MODELS_DIR = Path("models")

# Признаки, добавленные ради сетевого сигнала: их вклад и меряет абляция
NETWORK_FEATURES = [
    "recipient_clients_24h", "recipient_clients_7d", "recipient_inbound_24h",
    "recipient_amount_24h", "recipient_age_days", "recipient_is_new_to_bank",
    "device_clients_24h", "ip_clients_24h", "component_clients", "component_size",
]

ALERT_BUDGET = 0.02
CAP_GRID = [0.10, 0.15, 0.20, 0.30, 0.50]


def recall_at_budget(scores: np.ndarray, mask: np.ndarray, budget: float) -> float:
    if mask.sum() == 0:
        return float("nan")
    k = max(1, int(round(len(scores) * budget)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:k]] = True
    return float(flagged[mask].mean())


def recall_at_precision(y_true, scores, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= target
    return float(np.max(recall[:-1] * ok)) if ok.any() else 0.0


def supervised(X, y, splits, cols=None):
    """Обучить и откалибровать основную модель. Возвращает (p_valid, p_test)."""
    tr, va, te = splits
    Xc = X if cols is None else X[cols]
    model = train_binary(Xc.iloc[tr], y[tr], Xc.iloc[va], y[va])
    raw_va = model.predict_proba(Xc.iloc[va])[:, 1]
    raw_te = model.predict_proba(Xc.iloc[te])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, y[va])
    return cal.predict(raw_va), cal.predict(raw_te)


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный отчёт по модели")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--profiles", default="data/clients.csv")
    parser.add_argument("--out", default=str(MODELS_DIR / "report.json"))
    args = parser.parse_args()

    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    df = read_transactions(args.data).sort_values("timestamp", kind="mergesort")
    df = df.reset_index(drop=True)
    X, _ = build_matrix(df, pd.read_csv(args.profiles))
    X = X.reset_index(drop=True)
    y = df["is_fraud"].to_numpy()
    kinds = df["fraud_type"].fillna("").astype(str).to_numpy()
    splits = time_split(len(df))
    tr, va, te = splits
    y_te = y[te]

    report: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat()}

    # ------------------------------------------------ 1. абляция признаков
    print("1/3  Абляция признаков…")
    old_cols = [f for f in FEATURES if f not in NETWORK_FEATURES]
    _, p_old = supervised(X, y, splits, old_cols)
    _, p_new = supervised(X, y, splits)
    report["ablation"] = {
        "rows_test": int(len(y_te)),
        "variants": [
            {
                "name": "без сетевых признаков",
                "n_features": len(old_cols),
                "pr_auc": float(average_precision_score(y_te, p_old)),
                "roc_auc": float(roc_auc_score(y_te, p_old)),
                "recall_at_precision90": recall_at_precision(y_te, p_old, 0.90),
            },
            {
                "name": "с сетевыми признаками",
                "n_features": len(FEATURES),
                "pr_auc": float(average_precision_score(y_te, p_new)),
                "roc_auc": float(roc_auc_score(y_te, p_new)),
                "recall_at_precision90": recall_at_precision(y_te, p_new, 0.90),
            },
        ],
        "network_features": NETWORK_FEATURES,
    }
    for v in report["ablation"]["variants"]:
        print(f"     {v['name']:26} PR-AUC={v['pr_auc']:.4f} "
              f"recall@prec90={v['recall_at_precision90']:.3f}")

    # ------------------------------------------------ 2. потолок страховки
    print("2/3  Подбор потолка канала аномалий…")
    channel = AnomalyChannel().fit(X.iloc[tr])
    anom_va, anom_te = channel.score(X.iloc[va]), channel.score(X.iloc[te])
    p_va_known, p_te_known = supervised(X, y, splits)

    # та же модель, но соцтнженерия спрятана — на ней мерим пользу
    y_blind = y.copy()
    y_blind[kinds == "social_eng"] = 0
    p_va_blind, p_te_blind = supervised(X, y_blind, splits)
    target = (y_te == 1) & (kinds[te] == "social_eng")

    rows = [{
        "cap": None,
        "pr_auc_known": float(average_precision_score(y_te, p_te_known)),
        "novel_recall": recall_at_budget(p_te_blind, target, ALERT_BUDGET),
    }]
    for cap in CAP_GRID:
        known = NoisyOr(cap=cap).fit(anom_va, y[va]).fit_final(p_va_known, anom_va, y[va])
        blind = NoisyOr(cap=cap).fit(anom_va, y_blind[va]).fit_final(
            p_va_blind, anom_va, y_blind[va])
        rows.append({
            "cap": cap,
            "pr_auc_known": float(average_precision_score(
                y_te, known.predict(p_te_known, anom_te))),
            "novel_recall": recall_at_budget(
                blind.predict(p_te_blind, anom_te), target, ALERT_BUDGET),
        })
    report["cap_sweep"] = {"alert_budget": ALERT_BUDGET, "chosen": NoisyOr.DEFAULT_CAP,
                           "rows": rows}
    for r in rows:
        label = "выключен" if r["cap"] is None else f"{r['cap']:.2f}"
        print(f"     потолок {label:>9}  PR-AUC знакомых={r['pr_auc_known']:.4f}  "
              f"незнакомая схема={r['novel_recall']:.1%}")

    # ------------------------------------------------ 3. незнакомая схема
    print("3/3  Схема, которой модель не видела…")
    novel = []
    for kind in FRAUD_TYPES:
        yb = y.copy()
        yb[kinds == kind] = 0
        p_va_b, p_te_b = supervised(X, yb, splits)
        combiner = NoisyOr().fit(anom_va, yb[va]).fit_final(p_va_b, anom_va, yb[va])
        p_mix = combiner.predict(p_te_b, anom_te)
        hidden = (y_te == 1) & (kinds[te] == kind)
        known_mask = (y_te == 1) & (kinds[te] != kind)
        novel.append({
            "hidden": kind,
            "n_hidden_in_test": int(hidden.sum()),
            "supervised_recall": recall_at_budget(p_te_b, hidden, ALERT_BUDGET),
            "anomaly_recall": recall_at_budget(anom_te, hidden, ALERT_BUDGET),
            "blended_recall": recall_at_budget(p_mix, hidden, ALERT_BUDGET),
            "supervised_recall_known": recall_at_budget(p_te_b, known_mask, ALERT_BUDGET),
            "blended_recall_known": recall_at_budget(p_mix, known_mask, ALERT_BUDGET),
        })
        r = novel[-1]
        print(f"     спрятана {kind:14} с учителем {r['supervised_recall']:.1%}"
              f" -> два канала {r['blended_recall']:.1%}")
    report["novel_scheme"] = {"alert_budget": ALERT_BUDGET, "results": novel}

    report["architecture"] = {
        "n_features": len(FEATURES),
        "n_anomaly_features": len(ANOMALY_FEATURES),
        "split": {"train": int(len(range(*tr.indices(len(df))))),
                  "valid": int(len(range(*va.indices(len(df))))),
                  "test": int(len(range(*te.indices(len(df)))))},
    }
    report["elapsed_seconds"] = round(time.time() - t0, 1)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nСохранено: {args.out}  (за {report['elapsed_seconds']:.0f} с)")


if __name__ == "__main__":
    main()
