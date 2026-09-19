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
import gc
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

# Замеры канала аномалий усредняются по нескольким зёрнам.
# Это не перестраховка: Isolation Forest строит деревья на случайных
# подвыборках, и полнота по спрятанной схеме гуляла от 18.9% до 46.5% в
# зависимости от зерна. Одиночный прогон здесь — не результат, а один
# розыгрыш, и подавать его как результат было бы враньём.
SEEDS = [0, 1, 2, 3, 4]


def recall_at_budget(scores: np.ndarray, mask: np.ndarray, budget: float) -> float:
    if mask.sum() == 0:
        return float("nan")
    k = max(1, int(round(len(scores) * budget)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-scores)[:k]] = True
    return float(flagged[mask].mean())


def spread(values: list[float]) -> dict:
    """Медиана и границы разброса — так честнее, чем одно число."""
    a = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if a.size == 0:
        return {"median": float("nan"), "min": float("nan"),
                "max": float("nan"), "n_seeds": 0}
    return {"median": float(np.median(a)), "min": float(a.min()),
            "max": float(a.max()), "n_seeds": int(a.size)}


def recall_at_precision(y_true, scores, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= target
    return float(np.max(recall[:-1] * ok)) if ok.any() else 0.0


def supervised(X, y, splits, cols=None):
    """Обучить и откалибровать основную модель. Возвращает (p_valid, p_test).

    Модель здесь одноразовая: наружу уходят только предсказания. Отпускаем её
    сразу и зовём сборщик мусора — за один отчёт обучается около десяти
    моделей, и если держать их все, пик памяти подбирается к пределу
    free-тарифа Render, где сборка просто умирает без внятного сообщения.
    """
    tr, va, te = splits
    Xc = X if cols is None else X[cols]
    model = train_binary(Xc.iloc[tr], y[tr], Xc.iloc[va], y[va])
    raw_va = model.predict_proba(Xc.iloc[va])[:, 1]
    raw_te = model.predict_proba(Xc.iloc[te])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, y[va])
    out = cal.predict(raw_va), cal.predict(raw_te)
    del model, cal, raw_va, raw_te
    if cols is not None:
        del Xc
    gc.collect()
    return out


# --------------------------------------------------------------------------
# Контекст: данные, признаки и разбиение — общие для всех шагов
# --------------------------------------------------------------------------


class Context:
    """Данные, признаки и разбиение — всё, что нужно любому шагу.

    После сборки признаков исходный датафрейм отпускается целиком. Сто тысяч
    строк со строковыми колонками — идентификаторы, города, устройства, IP —
    весят больше самой матрицы признаков, а дальше от них нужны только метка
    и вид схемы. На free-тарифе Render с его 512 МБ эта разница решает,
    доживёт сборка до конца или нет.
    """

    def __init__(self, data: str, profiles: str):
        df = read_transactions(data).sort_values("timestamp", kind="mergesort")
        df = df.reset_index(drop=True)
        X, store = build_matrix(df, pd.read_csv(profiles))
        self.X = X.reset_index(drop=True)
        self.y = df["is_fraud"].to_numpy()
        self.kinds = df["fraud_type"].fillna("").astype(str).to_numpy()
        self.n_rows = len(df)
        del store, df, X
        gc.collect()
        self.splits = time_split(self.n_rows)
        self.y_te = self.y[self.splits[2]]


# --------------------------------------------------------------------------
# Шаги. Каждый самодостаточен и возвращает свой кусок отчёта.
# --------------------------------------------------------------------------


def step_ablation(ctx: Context) -> dict:
    """Сколько дают сетевые признаки на одних и тех же данных."""
    print("Абляция признаков…")
    old_cols = [f for f in FEATURES if f not in NETWORK_FEATURES]
    _, p_old = supervised(ctx.X, ctx.y, ctx.splits, old_cols)
    _, p_new = supervised(ctx.X, ctx.y, ctx.splits)

    def variant(name, cols, p):
        return {
            "name": name,
            "n_features": len(cols),
            "pr_auc": float(average_precision_score(ctx.y_te, p)),
            "roc_auc": float(roc_auc_score(ctx.y_te, p)),
            "recall_at_precision90": recall_at_precision(ctx.y_te, p, 0.90),
        }

    out = {
        "ablation": {
            "rows_test": int(len(ctx.y_te)),
            "variants": [
                variant("без сетевых признаков", old_cols, p_old),
                variant("с сетевыми признаками", FEATURES, p_new),
            ],
            "network_features": NETWORK_FEATURES,
        }
    }
    for v in out["ablation"]["variants"]:
        print(f"  {v['name']:26} PR-AUC={v['pr_auc']:.4f} "
              f"recall@prec90={v['recall_at_precision90']:.3f}")
    return out


def step_caps(ctx: Context) -> dict:
    """Цена страховки против её пользы при разных потолках."""
    print("Подбор потолка канала аномалий…")
    tr, va, te = ctx.splits
    p_va_known, p_te_known = supervised(ctx.X, ctx.y, ctx.splits)
    y_blind = ctx.y.copy()
    y_blind[ctx.kinds == "social_eng"] = 0
    p_va_blind, p_te_blind = supervised(ctx.X, y_blind, ctx.splits)
    target = (ctx.y_te == 1) & (ctx.kinds[te] == "social_eng")

    per_cap: dict = {cap: {"pr": [], "novel": []} for cap in CAP_GRID}
    for seed in SEEDS:
        channel = AnomalyChannel(random_state=seed).fit(ctx.X.iloc[tr])
        anom_va = channel.score(ctx.X.iloc[va])
        anom_te = channel.score(ctx.X.iloc[te])
        for cap in CAP_GRID:
            known = NoisyOr(cap=cap).fit(anom_va, ctx.y[va]).fit_final(
                p_va_known, anom_va, ctx.y[va])
            blind = NoisyOr(cap=cap).fit(anom_va, y_blind[va]).fit_final(
                p_va_blind, anom_va, y_blind[va])
            per_cap[cap]["pr"].append(float(average_precision_score(
                ctx.y_te, known.predict(p_te_known, anom_te))))
            per_cap[cap]["novel"].append(recall_at_budget(
                blind.predict(p_te_blind, anom_te), target, ALERT_BUDGET))
        del channel, anom_va, anom_te
        gc.collect()

    rows = [{
        "cap": None,
        "pr_auc_known": float(average_precision_score(ctx.y_te, p_te_known)),
        "pr_auc_known_spread": None,
        "novel_recall": recall_at_budget(p_te_blind, target, ALERT_BUDGET),
        "novel_recall_spread": None,
    }]
    for cap in CAP_GRID:
        pr, nov = spread(per_cap[cap]["pr"]), spread(per_cap[cap]["novel"])
        rows.append({"cap": cap, "pr_auc_known": pr["median"],
                     "pr_auc_known_spread": pr,
                     "novel_recall": nov["median"], "novel_recall_spread": nov})
    for r in rows:
        label = "выключен" if r["cap"] is None else f"{r['cap']:.2f}"
        sp = r["novel_recall_spread"]
        tail = f" [{sp['min']:.1%}-{sp['max']:.1%}]" if sp else ""
        print(f"  потолок {label:>9}  PR-AUC знакомых={r['pr_auc_known']:.4f}  "
              f"незнакомая схема={r['novel_recall']:.1%}{tail}")
    return {"cap_sweep": {"alert_budget": ALERT_BUDGET, "seeds": SEEDS,
                          "chosen": NoisyOr.DEFAULT_CAP, "rows": rows}}


def step_novel(ctx: Context) -> dict:
    """Что будет со схемой, которой модель никогда не видела."""
    print("Схема, которой модель не видела…")
    tr, va, te = ctx.splits
    channels = []
    for seed in SEEDS:
        ch = AnomalyChannel(random_state=seed).fit(ctx.X.iloc[tr])
        channels.append((seed, ch.score(ctx.X.iloc[va]), ch.score(ctx.X.iloc[te])))
        del ch
    gc.collect()

    results = []
    for kind in FRAUD_TYPES:
        yb = ctx.y.copy()
        yb[ctx.kinds == kind] = 0
        p_va_b, p_te_b = supervised(ctx.X, yb, ctx.splits)
        hidden = (ctx.y_te == 1) & (ctx.kinds[te] == kind)
        known = (ctx.y_te == 1) & (ctx.kinds[te] != kind)

        anom_only, blended, blended_known = [], [], []
        for _seed, anom_va, anom_te in channels:
            combiner = NoisyOr().fit(anom_va, yb[va]).fit_final(p_va_b, anom_va, yb[va])
            p_mix = combiner.predict(p_te_b, anom_te)
            anom_only.append(recall_at_budget(anom_te, hidden, ALERT_BUDGET))
            blended.append(recall_at_budget(p_mix, hidden, ALERT_BUDGET))
            blended_known.append(recall_at_budget(p_mix, known, ALERT_BUDGET))
            del combiner, p_mix

        sup = recall_at_budget(p_te_b, hidden, ALERT_BUDGET)
        bl = spread(blended)
        results.append({
            "hidden": kind,
            "n_hidden_in_test": int(hidden.sum()),
            "supervised_recall": sup,
            "anomaly_recall": spread(anom_only)["median"],
            "anomaly_recall_spread": spread(anom_only),
            "blended_recall": bl["median"],
            "blended_recall_spread": bl,
            "supervised_recall_known": recall_at_budget(p_te_b, known, ALERT_BUDGET),
            "blended_recall_known": spread(blended_known)["median"],
        })
        print(f"  спрятана {kind:14} с учителем {sup:.1%}"
              f" -> два канала {bl['median']:.1%} [{bl['min']:.1%}-{bl['max']:.1%}]")
        del p_va_b, p_te_b
        gc.collect()

    return {
        "novel_scheme": {"alert_budget": ALERT_BUDGET, "seeds": SEEDS,
                         "results": results},
        "architecture": {
            "n_features": len(FEATURES),
            "n_anomaly_features": len(ANOMALY_FEATURES),
            "split": {"train": len(range(*tr.indices(ctx.n_rows))),
                      "valid": len(range(*va.indices(ctx.n_rows))),
                      "test": len(range(*te.indices(ctx.n_rows)))},
        },
    }


STEPS = {"ablation": step_ablation, "caps": step_caps, "novel": step_novel}


def run_in_subprocesses(args) -> dict:
    """Прогнать шаги отдельными процессами и склеить результат.

    За один отчёт обучается около десяти моделей. В одном процессе их пик
    складывается и подбирается к 512 МБ free-тарифа Render, где сборка умирает
    без внятного сообщения. Отдельный процесс на шаг — и система забирает
    память обратно после каждого: пик равен самому тяжёлому шагу, а не сумме.
    Сборщик мусора так не умеет: аллокатор Python не возвращает освобождённые
    страницы операционной системе.

    Упавший шаг не роняет отчёт целиком — остальные разделы всё равно
    соберутся, а сайт умеет показывать неполный отчёт.
    """
    import subprocess
    import sys

    merged: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat()}
    for step in STEPS:
        part = MODELS_DIR / f".report_{step}.json"
        code = subprocess.run(
            [sys.executable, "-m", "ml.report", "--step", step,
             "--data", args.data, "--profiles", args.profiles, "--out", str(part)],
            check=False,
        ).returncode
        if code != 0 or not part.exists():
            print(f"  шаг «{step}» не выполнен (код {code}), пропускаем")
            continue
        merged.update(json.loads(part.read_text(encoding="utf-8")))
        part.unlink()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный отчёт по модели")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--profiles", default="data/clients.csv")
    parser.add_argument("--out", default=str(MODELS_DIR / "report.json"))
    parser.add_argument("--step", choices=tuple(STEPS),
                        help="выполнить один шаг (внутренний запуск по процессам)")
    parser.add_argument("--single-process", action="store_true",
                        help="всё в одном процессе: быстрее, но пик памяти выше")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    if args.step:
        report = STEPS[args.step](Context(args.data, args.profiles))
    elif args.single_process:
        ctx = Context(args.data, args.profiles)
        report = {"generated_at": pd.Timestamp.now("UTC").isoformat()}
        for fn in STEPS.values():
            report.update(fn(ctx))
    else:
        report = run_in_subprocesses(args)

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    have = [k for k in ("ablation", "cap_sweep", "novel_scheme") if k in report]
    print(f"\nСохранено: {args.out}  разделы: {', '.join(have) or 'нет'}"
          f"  (за {report['elapsed_seconds']:.0f} с)")


if __name__ == "__main__":
    main()
