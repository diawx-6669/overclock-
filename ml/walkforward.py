"""Скользящая валидация по времени: метрики с доверительным интервалом.

Одно разбиение на обучение и тест даёт одно число. Звучит оно убедительно,
но проверить его нечем: попади в тестовое окно на десяток колец больше — и
доля по кольцам изменится на пятнадцать процентных пунктов. Ровно это у нас
и было: в тестовом срезе оказалось 32 операции колец, и любая оценка по ним
болталась в широких пределах.

Скользящая валидация решает это правильно. Окно обучения расширяется, окно
проверки едет вперёд по времени, и каждый сдвиг даёт независимый замер:

    складка 1:  обучение [....]           проверка [..]
    складка 2:  обучение [......]         проверка   [..]
    складка 3:  обучение [........]       проверка     [..]
    складка 4:  обучение [..........]     проверка       [..]

Окно обучения растёт, а не едет целиком: банк не выбрасывает прошлогодние
данные, он к ним прибавляет. И ни одна складка не подглядывает в будущее.

Результат — не одно число, а распределение: медиана и границы. По нему сразу
видно, устойчив результат или держится на удачном разбиении.

Запуск:
    python -m ml.walkforward
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ml.anomaly import AnomalyChannel, NoisyOr
from ml.common import FRAUD_TYPES, read_transactions
from ml.features import FEATURES, build_matrix
from ml.train import train_binary

MODELS_DIR = Path("models")

# Складок. Четыре при месяце данных дают проверочные окна примерно по неделе:
# короче — и в окно не попадает достаточно фрода, длиннее — и складок не
# остаётся. Ограничение здесь в объёме данных, а не в методе.
N_FOLDS = 4

# Доля потока на первое обучающее окно. Дальше оно только растёт.
FIRST_TRAIN = 0.40
VALID_SHARE = 0.10


def folds(n: int, n_folds: int = N_FOLDS) -> list[tuple[slice, slice, slice]]:
    """Расширяющееся окно обучения, едущие вперёд валидация и проверка."""
    out = []
    test_share = (1.0 - FIRST_TRAIN - VALID_SHARE) / n_folds
    for k in range(n_folds):
        train_end = FIRST_TRAIN + test_share * k
        valid_end = train_end + VALID_SHARE
        test_end = min(valid_end + test_share, 1.0)
        out.append((
            slice(0, int(n * train_end)),
            slice(int(n * train_end), int(n * valid_end)),
            slice(int(n * valid_end), int(n * test_end)),
        ))
    return out


def recall_at_precision(y_true, scores, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= target
    return float(np.max(recall[:-1] * ok)) if ok.any() else 0.0


def spread(values: list[float]) -> dict:
    a = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if a.size == 0:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "n": int(a.size)}


def run_fold(X, y, kinds, split, with_anomaly: bool = True) -> dict:
    tr, va, te = split
    model = train_binary(X.iloc[tr], y[tr], X.iloc[va], y[va])
    raw_va = model.predict_proba(X.iloc[va])[:, 1]
    raw_te = model.predict_proba(X.iloc[te])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, y[va])
    p_va, p_te = cal.predict(raw_va), cal.predict(raw_te)

    if with_anomaly:
        channel = AnomalyChannel().fit(X.iloc[tr])
        anom_va, anom_te = channel.score(X.iloc[va]), channel.score(X.iloc[te])
        combiner = NoisyOr().fit(anom_va, y[va]).fit_final(p_va, anom_va, y[va])
        p_te = combiner.predict(p_te, anom_te)
        del channel, anom_va, anom_te, combiner

    y_te = y[te]
    result = {
        "rows_train": int(tr.stop - tr.start),
        "rows_test": int(te.stop - te.start),
        "fraud_in_test": int(y_te.sum()),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "recall_at_precision90": recall_at_precision(y_te, p_te, 0.90),
        "per_type": {},
    }
    kt = kinds[te]
    for kind in FRAUD_TYPES:
        mask = (y_te == 1) & (kt == kind)
        if mask.sum() >= 5:
            result["per_type"][kind] = {
                "n": int(mask.sum()),
                "share_above_half": float((p_te[mask] > 0.5).mean()),
            }
    del model, cal, raw_va, raw_te, p_va, p_te
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Скользящая валидация по времени")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--profiles", default="data/clients.csv")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--out", default=str(MODELS_DIR / "walkforward.json"))
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    df = read_transactions(args.data).sort_values("timestamp", kind="mergesort")
    df = df.reset_index(drop=True)
    X, store = build_matrix(df, pd.read_csv(args.profiles))
    X = X.reset_index(drop=True)
    y = df["is_fraud"].to_numpy()
    kinds = df["fraud_type"].fillna("").astype(str).to_numpy()
    stamps = df["timestamp"]
    del store, df
    gc.collect()

    print(f"Скользящая валидация: {args.folds} складки, окно обучения расширяется\n")
    results = []
    for i, split in enumerate(folds(len(X), args.folds), start=1):
        tr, _va, te = split
        r = run_fold(X, y, kinds, split)
        r["fold"] = i
        r["test_from"] = str(stamps.iloc[te.start].date())
        r["test_to"] = str(stamps.iloc[te.stop - 1].date())
        results.append(r)
        print(f"  складка {i}: обучение {r['rows_train']:>6,} строк, "
              f"проверка {r['test_from']}..{r['test_to']} "
              f"({r['fraud_in_test']:>3} фрода)  "
              f"PR-AUC={r['pr_auc']:.4f}  recall@prec90={r['recall_at_precision90']:.3f}")

    summary = {
        "folds": args.folds,
        "pr_auc": spread([r["pr_auc"] for r in results]),
        "roc_auc": spread([r["roc_auc"] for r in results]),
        "recall_at_precision90": spread([r["recall_at_precision90"] for r in results]),
        "per_type": {
            kind: spread([r["per_type"][kind]["share_above_half"]
                          for r in results if kind in r["per_type"]])
            for kind in FRAUD_TYPES
        },
        "fraud_total_in_tests": int(sum(r["fraud_in_test"] for r in results)),
    }

    print(f"\n{'метрика':<24}{'медиана':>10}{'минимум':>10}{'максимум':>11}")
    for name, key in (("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"),
                      ("полнота при точн. 90%", "recall_at_precision90")):
        v = summary[key]
        print(f"{name:<24}{v['median']:>10.4f}{v['min']:>10.4f}{v['max']:>11.4f}")
    print(f"\nВсего фрода в проверочных окнах: {summary['fraud_total_in_tests']} "
          f"(против одного окна в обычном разбиении)")
    for kind, v in summary["per_type"].items():
        if v["n"]:
            print(f"  {kind:<14} доля p>0.5: медиана {v['median']:.3f} "
                  f"[{v['min']:.3f}–{v['max']:.3f}] по {v['n']} складкам")

    Path(args.out).write_text(
        json.dumps({"summary": summary, "folds_detail": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
