"""Эксперимент: что будет со схемой, которой модель никогда не видела.

Главное возражение к любому антифроду на разметке звучит так: «вы ловите то,
что уже случалось, а мошенники придумают новое». Возражение справедливое, и
отвечать на него надо не словами, а замером.

Постановка. Берём один вид мошенничества и полностью вычёркиваем его из
обучения: в обучающей и валидационной выборках его метки обнуляются, будто
банк про такую схему ничего не знает и считает эти операции обычными. Затем
смотрим, сколько этой схемы поймает каждый канал на тестовом периоде.

Сравнение идёт при одинаковой нагрузке на операторов: обе системы поднимают
тревогу по одной и той же доле потока. Иначе выиграть можно было бы просто
тем, что подозреваешь всех подряд.

Запуск:
    python -m ml.experiment_novel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ml.anomaly import AnomalyChannel, NoisyOr
from ml.common import FRAUD_TYPES, read_transactions
from ml.features import FEATURES, build_matrix
from ml.train import time_split, train_binary

ALERT_BUDGET = 0.02  # доля потока, которую система имеет право отправить на проверку


def recall_at_budget(scores: np.ndarray, mask: np.ndarray, budget: float) -> float:
    """Какую долю целевых операций поймаем, подняв тревогу по budget потока."""
    if mask.sum() == 0:
        return float("nan")
    k = max(1, int(round(len(scores) * budget)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:k]] = True
    return float(flagged[mask].mean())


def run_one(X, y, kinds, splits, hidden: str) -> dict:
    tr, va, te = splits

    # Банк не знает про эту схему: в обучении её метки обнулены
    y_blind = y.copy()
    y_blind[kinds == hidden] = 0

    model = train_binary(X.iloc[tr], y_blind[tr], X.iloc[va], y_blind[va])
    raw_va = model.predict_proba(X.iloc[va])[:, 1]
    raw_te = model.predict_proba(X.iloc[te])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(raw_va, y_blind[va])
    p_sup = cal.predict(raw_te)

    # Канал без учителя разметку не видел вообще, его прятать не от чего
    channel = AnomalyChannel().fit(X.iloc[tr])
    anom_va, anom_te = channel.score(X.iloc[va]), channel.score(X.iloc[te])

    # ровно та же сборка, что и в проде: «или» плюс перекалибровка
    p_sup_va = cal.predict(raw_va)
    combiner = NoisyOr().fit(anom_va, y_blind[va])
    combiner.fit_final(p_sup_va, anom_va, y_blind[va])
    p_mix = combiner.predict(p_sup, anom_te)

    target = (y[te] == 1) & (kinds[te] == hidden)
    known = (y[te] == 1) & (kinds[te] != hidden)

    return {
        "hidden": hidden,
        "n_hidden_in_test": int(target.sum()),
        "supervised_recall": recall_at_budget(p_sup, target, ALERT_BUDGET),
        "anomaly_recall": recall_at_budget(anom_te, target, ALERT_BUDGET),
        "blended_recall": recall_at_budget(p_mix, target, ALERT_BUDGET),
        # на знакомых схемах смесь не должна проседать
        "supervised_recall_known": recall_at_budget(p_sup, known, ALERT_BUDGET),
        "blended_recall_known": recall_at_budget(p_mix, known, ALERT_BUDGET),
        "anomaly_p_mean": float(combiner.anomaly_probability(anom_te).mean()),
        "anomaly_p_max": float(combiner.anomaly_probability(anom_te).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент с незнакомой схемой")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--profiles", default="data/clients.csv")
    parser.add_argument("--out", default="models/experiment_novel.json")
    args = parser.parse_args()

    df = read_transactions(args.data).sort_values("timestamp", kind="mergesort")
    df = df.reset_index(drop=True)
    X, _ = build_matrix(df, pd.read_csv(args.profiles))
    X = X.reset_index(drop=True)
    y = df["is_fraud"].to_numpy()
    kinds = df["fraud_type"].fillna("").astype(str).to_numpy()
    splits = time_split(len(df))

    print("=" * 74)
    print("СХЕМА, КОТОРОЙ МОДЕЛЬ НЕ ВИДЕЛА")
    print(f"Тревога поднимается по {ALERT_BUDGET:.0%} потока — нагрузка одинакова у всех")
    print("=" * 74)

    results = []
    for kind in FRAUD_TYPES:
        r = run_one(X, y, kinds, splits, kind)
        results.append(r)
        if not r["n_hidden_in_test"]:
            print(f"\n{kind}: в тесте нет примеров, пропускаем")
            continue
        print(f"\nСпрятана схема: {kind}  ({r['n_hidden_in_test']} операций в тесте)")
        print(f"  только с учителем     {r['supervised_recall']:.1%}")
        print(f"  только аномалии       {r['anomaly_recall']:.1%}")
        print(f"  два канала вместе     {r['blended_recall']:.1%}"
              f"   ({r['blended_recall'] - r['supervised_recall']:+.1%})")
        print(f"  на знакомых схемах    {r['supervised_recall_known']:.1%}"
              f" -> {r['blended_recall_known']:.1%}")
        print(f"  вклад канала аномалий: в среднем {r['anomaly_p_mean']:.3f},"
              f" максимум {r['anomaly_p_max']:.3f}")

    Path(args.out).write_text(
        json.dumps({"alert_budget": ALERT_BUDGET, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
