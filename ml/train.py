"""Обучение и честная оценка Fraud Hunter.

Что здесь важно:

1. Разрез по времени, а не случайный. Антифрод всегда работает «в будущее»:
   учимся на первых днях месяца, проверяемся на последних. Случайное
   перемешивание завысило бы метрики и обмануло бы нас самих.

2. Две модели. Первая отвечает на вопрос «это фрод?». Вторая — «какая это
   схема?». Вторая нужна не ради красоты: от вида схемы зависит, какое
   действие сработает, а значит и решение по деньгам.

3. Калибровка. Модель по умолчанию выдаёт рейтинг, а не вероятность. Для
   выбора по ожидаемым потерям нужна именно вероятность: 0.3 должно означать,
   что из тысячи таких операций фродом окажутся примерно триста. Иначе
   стоимостная модель считает не то.

4. Метрика — деньги. ROC-AUC мы печатаем, но решает сравнение в тенге с
   обычным фиксированным порогом.

Запуск:
    python -m ml.train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from ml.anomaly import AnomalyChannel, NoisyOr
from ml.common import FRAUD_TYPES, read_transactions
from ml.features import FEATURES, build_matrix
from backend.cost import (
    ACTIONS,
    ALLOW,
    BLOCK,
    DEFAULT_ECONOMICS,
    decide,
    realised_cost,
)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

TRAIN_FRAC = 0.60
VALID_FRAC = 0.15
# остальное — тест


# --------------------------------------------------------------------------
# Разбиение
# --------------------------------------------------------------------------


def time_split(n: int) -> tuple[slice, slice, slice]:
    i1 = int(n * TRAIN_FRAC)
    i2 = int(n * (TRAIN_FRAC + VALID_FRAC))
    return slice(0, i1), slice(i1, i2), slice(i2, n)


# --------------------------------------------------------------------------
# Модели
# --------------------------------------------------------------------------


def train_binary(X_tr, y_tr, X_va, y_va) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        # Без этих двух флагов LightGBM строит гистограммы в несколько потоков и
        # результат гуляет в последних знаках от запуска к запуску. На метриках
        # это почти незаметно, но пороговые решения по отдельным операциям
        # начинают прыгать, и воспроизвести цифры из отчёта уже нельзя.
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    callbacks = [lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)]
    # LightGBM 4.7 переименовал eval_set в eval_X/eval_y, но старые версии
    # нового имени не знают. Поддерживаем оба, чтобы сборка не зависела от
    # того, какая версия окажется на сервере.
    try:
        model.fit(
            X_tr, y_tr, eval_X=X_va, eval_y=y_va,
            eval_metric="average_precision", callbacks=callbacks,
        )
    except TypeError:
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            eval_metric="average_precision", callbacks=callbacks,
        )
    return model


def train_type_model(X, kinds) -> tuple[lgb.LGBMClassifier, list[str]]:
    """Классификатор схемы. Учится только на фродовых операциях."""
    classes = sorted(set(kinds))
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[k] for k in kinds])
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(classes),
        n_estimators=250,
        learning_rate=0.07,
        num_leaves=15,
        min_child_samples=15,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    model.fit(X, y)
    return model, classes


# --------------------------------------------------------------------------
# Оценка
# --------------------------------------------------------------------------


def reliability_curve(y_true, p, n_bins: int = 10) -> list[dict]:
    """Проверка калибровки: в корзине с предсказанием 0.3 фрода должно быть ~30%.

    Без этой проверки стоимостная модель считает ожидаемые потери по числу,
    которое вероятностью только притворяется.
    """
    edges = np.concatenate([[0.0], np.quantile(p, np.linspace(0.9, 1.0, n_bins))])
    edges = np.unique(np.round(edges, 6))
    out: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < edges[-1] else (p >= lo) & (p <= hi)
        if mask.sum() < 20:
            continue
        out.append({
            "p_lo": float(lo),
            "p_hi": float(hi),
            "count": int(mask.sum()),
            "predicted": float(p[mask].mean()),
            "actual": float(np.asarray(y_true)[mask].mean()),
        })
    return out


def recall_at_precision(y_true, scores, target_precision: float) -> tuple[float, float]:
    """Какую долю фрода поймаем, если держать заданную точность."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = precision[:-1] >= target_precision
    if not ok.any():
        return 0.0, 1.0
    idx = int(np.argmax(recall[:-1] * ok))
    return float(recall[idx]), float(thresholds[idx])


def best_threshold_by_cost(
    p: np.ndarray, frame: pd.DataFrame, summaries: dict, grid: np.ndarray
) -> tuple[float, float]:
    """Подобрать лучший фиксированный порог для базовой политики.

    Порог подбираем на валидации, а не на тесте — иначе сравнение было бы
    нечестным в нашу пользу.
    """
    best_t, best_cost = 0.5, float("inf")
    amounts = frame["amount"].to_numpy()
    types = frame["tx_type"].to_numpy()
    frauds = frame["is_fraud"].to_numpy().astype(bool)
    kinds = frame["fraud_type"].fillna("").to_numpy()
    clients = frame["client_id"].to_numpy()

    for t in grid:
        total = 0.0
        for i in range(len(p)):
            action = BLOCK if p[i] >= t else ALLOW
            total += realised_cost(
                action, bool(frauds[i]), float(amounts[i]), str(types[i]),
                str(kinds[i]) or None, summaries.get(clients[i]),
            )
        if total < best_cost:
            best_cost, best_t = total, float(t)
    return best_t, best_cost


def evaluate_policies(
    p: np.ndarray,
    type_probs: np.ndarray,
    type_classes: list[str],
    frame: pd.DataFrame,
    summaries: dict,
    fixed_thresholds: dict[str, float],
) -> dict:
    """Сравнить нашу политику с «ничего не делаем» и с фиксированными порогами."""
    amounts = frame["amount"].to_numpy(dtype=float)
    types = frame["tx_type"].astype(str).to_numpy()
    frauds = frame["is_fraud"].to_numpy().astype(bool)
    kinds = frame["fraud_type"].fillna("").astype(str).to_numpy()
    clients = frame["client_id"].astype(str).to_numpy()

    n = len(p)
    chosen: list[str] = []
    cost_ours = 0.0

    for i in range(n):
        tp = {c: float(type_probs[i, j]) for j, c in enumerate(type_classes)}
        d = decide(
            float(p[i]), float(amounts[i]), str(types[i]), tp, summaries.get(clients[i])
        )
        chosen.append(d.action)
        cost_ours += realised_cost(
            d.action, bool(frauds[i]), float(amounts[i]), str(types[i]),
            str(kinds[i]) or None, summaries.get(clients[i]),
        )

    chosen_arr = np.array(chosen)

    baselines: dict[str, float] = {}
    # Ничего не делаем — вся сумма фрода наша потеря
    baselines["no_system"] = sum(
        realised_cost(ALLOW, bool(frauds[i]), float(amounts[i]), str(types[i]),
                      str(kinds[i]) or None, summaries.get(clients[i]))
        for i in range(n)
    )
    for name, t in fixed_thresholds.items():
        baselines[name] = sum(
            realised_cost(
                BLOCK if p[i] >= t else ALLOW, bool(frauds[i]), float(amounts[i]),
                str(types[i]), str(kinds[i]) or None, summaries.get(clients[i]),
            )
            for i in range(n)
        )

    # Операционная нагрузка и трение
    honest = ~frauds
    action_counts = {a: int((chosen_arr == a).sum()) for a in ACTIONS}
    friction_honest = float((chosen_arr[honest] != ALLOW).mean()) if honest.any() else 0.0
    caught = float((chosen_arr[frauds] != ALLOW).mean()) if frauds.any() else 0.0

    per_type: dict[str, dict] = {}
    for kind in FRAUD_TYPES:
        mask = frauds & (kinds == kind)
        if not mask.any():
            continue
        acts = chosen_arr[mask]
        per_type[kind] = {
            "count": int(mask.sum()),
            "detected_share": float((acts != ALLOW).mean()),
            "actions": {a: int((acts == a).sum()) for a in ACTIONS},
            "mean_p": float(p[mask].mean()),
        }

    return {
        "cost_ours": float(cost_ours),
        "baselines": {k: float(v) for k, v in baselines.items()},
        "action_counts": action_counts,
        "friction_rate_honest": friction_honest,
        "detection_rate": caught,
        "per_type": per_type,
        "chosen": chosen_arr,
    }


# --------------------------------------------------------------------------
# Главный сценарий
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение Fraud Hunter")
    parser.add_argument("--data", default=str(DATA_DIR / "transactions.csv"))
    parser.add_argument("--profiles", default=str(DATA_DIR / "clients.csv"))
    parser.add_argument("--out", default=str(MODELS_DIR / "fraud_hunter.joblib"))
    parser.add_argument(
        "--no-anomaly",
        action="store_true",
        help="отключить канал поиска аномалий без учителя (страховку от незнакомых схем)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("Загрузка данных…")
    df = read_transactions(args.data)
    profiles = pd.read_csv(args.profiles)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    print("Сборка признаков (поток по времени, без утечек)…")
    X_all, store = build_matrix(df, profiles)
    X_all = X_all.reset_index(drop=True)
    y_all = df["is_fraud"].to_numpy()

    tr, va, te = time_split(len(df))
    X_tr, X_va, X_te = X_all.iloc[tr], X_all.iloc[va], X_all.iloc[te]
    y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]
    print(
        f"  обучение {len(X_tr):,} (фрод {y_tr.sum()})  "
        f"валидация {len(X_va):,} (фрод {y_va.sum()})  "
        f"тест {len(X_te):,} (фрод {y_te.sum()})"
    )

    print("Обучение модели «это фрод?»…")
    model = train_binary(X_tr, y_tr, X_va, y_va)

    # Настоящая история обучения: по ней видно, где модель перестала
    # улучшаться и почему ранняя остановка сработала именно там
    history = getattr(model, "evals_result_", {}) or {}
    curve = next(iter(history.values()), {})
    training_curve = {
        "average_precision": [float(v) for v in curve.get("average_precision", [])],
        "logloss": [float(v) for v in curve.get("binary_logloss", [])],
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
    }
    print(f"  итераций {len(training_curve['average_precision'])}, "
          f"лучшая {training_curve['best_iteration']}")

    raw_va = model.predict_proba(X_va)[:, 1]
    raw_te = model.predict_proba(X_te)[:, 1]

    print("Калибровка вероятностей на валидации…")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_va, y_va)
    p_te = calibrator.predict(raw_te)
    p_va = calibrator.predict(raw_va)

    # --- второй канал: страховка от схем, которых не было в обучении
    anomaly_channel = None
    combiner = None
    p_te_supervised = p_te.copy()
    if not args.no_anomaly:
        print("Обучение канала аномалий (без учителя)…")
        anomaly_channel = AnomalyChannel().fit(X_tr)
        anom_va = anomaly_channel.score(X_va)
        anom_te = anomaly_channel.score(X_te)
        combiner = NoisyOr().fit(anom_va, y_va)
        # второй проход: вернуть смеси смысл вероятности
        combiner.fit_final(p_va, anom_va, y_va)
        p_te = combiner.predict(p_te, anom_te)
        p_va = combiner.predict(p_va, anom_va)

    print("Обучение модели «какая это схема?»…")
    fraud_mask_fit = (y_all[tr] == 1)
    fit_idx = np.arange(len(df))[tr][fraud_mask_fit]
    fraud_mask_va = (y_all[va] == 1)
    fit_idx = np.concatenate([fit_idx, np.arange(len(df))[va][fraud_mask_va]])
    type_model, type_classes = train_type_model(
        X_all.iloc[fit_idx], df.loc[fit_idx, "fraud_type"].astype(str).tolist()
    )
    type_probs_te = type_model.predict_proba(X_te)

    # ---------------------------------------------------------------- метрики
    print("\n" + "=" * 68)
    print("КАЧЕСТВО МОДЕЛИ (на отложенном по времени тесте)")
    print("=" * 68)
    roc = roc_auc_score(y_te, p_te)
    pr = average_precision_score(y_te, p_te)
    brier_raw = brier_score_loss(y_te, raw_te)
    brier_cal = brier_score_loss(y_te, p_te)
    print(f"ROC-AUC                {roc:.4f}")
    print(f"PR-AUC (average prec.) {pr:.4f}   базовый уровень {y_te.mean():.4f}")
    print(f"Brier до калибровки    {brier_raw:.5f}")
    print(f"Brier после калибровки {brier_cal:.5f}")
    if combiner is not None:
        pr_sup = average_precision_score(y_te, p_te_supervised)
        print(f"PR-AUC без канала аномалий {pr_sup:.4f} "
              f"(страховка стоит {pr - pr_sup:+.4f}; что она даёт — ml/experiment_novel.py)")

    reliability = reliability_curve(y_te, p_te)
    print("\nНадёжность вероятностей (предсказано / фактически):")
    for b in reliability:
        print(
            f"  [{b['p_lo']:.3f} … {b['p_hi']:.3f})  n={b['count']:>6,}  "
            f"предсказано {b['predicted']:.3f}  фактически {b['actual']:.3f}"
        )

    rec_ops = {}
    for target in (0.90, 0.75, 0.50, 0.25):
        rec, thr = recall_at_precision(y_te, p_te, target)
        rec_ops[f"recall@precision{int(target * 100)}"] = {
            "recall": rec, "threshold": float(thr)
        }
        print(f"Полнота при точности {target:.0%}:  {rec:.3f}  (порог {thr:.4f})")

    # Попадание в тип схемы
    te_fraud = y_te == 1
    type_acc = None
    if te_fraud.any():
        pred_kind = np.array(type_classes)[type_probs_te.argmax(axis=1)]
        true_kind = df.loc[X_te.index, "fraud_type"].astype(str).to_numpy()
        type_acc = float((pred_kind[te_fraud] == true_kind[te_fraud]).mean())
        print(f"\nТип схемы угадан у {type_acc:.1%} фродовых операций")
        for kind in type_classes:
            m = te_fraud & (true_kind == kind)
            if m.any():
                print(f"  {kind:<14} {(pred_kind[m] == kind).mean():.1%}  ({int(m.sum())} шт.)")

    # ---------------------------------------------------------------- деньги
    print("\n" + "=" * 68)
    print("ДЕНЬГИ: наша политика против фиксированного порога")
    print("=" * 68)
    summaries = {cid: store.client_summary(cid) for cid in df["client_id"].unique()}

    va_frame = df.iloc[va]
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_t, _ = best_threshold_by_cost(p_va, va_frame, summaries, grid)
    print(f"Лучший фиксированный порог, подобранный на валидации: {best_t:.2f}")

    te_frame = df.iloc[te]
    report = evaluate_policies(
        p_te, type_probs_te, type_classes, te_frame, summaries,
        {"threshold_0.50": 0.5, f"threshold_best_{best_t:.2f}": best_t},
    )
    chosen = report.pop("chosen")

    ours = report["cost_ours"]
    print(f"\nПотери за тестовый период ({len(te_frame):,} операций):")
    for name, value in report["baselines"].items():
        delta = value - ours
        print(f"  {name:<22} {value:>14,.0f} ₸   мы дешевле на {delta:>12,.0f} ₸")
    print(f"  {'наша политика':<22} {ours:>14,.0f} ₸")

    print(f"\nДоля фрода, на который система отреагировала: {report['detection_rate']:.1%}")
    print(f"Доля честных операций с трением:              {report['friction_rate_honest']:.2%}")
    print("Что система решила сделать:")
    for a, c in report["action_counts"].items():
        print(f"  {a:<10} {c:>7,}  ({c / len(te_frame):.2%})")

    print("\nПо видам мошенничества:")
    for kind, info in report["per_type"].items():
        print(
            f"  {kind:<14} поймано {info['detected_share']:.1%} из {info['count']:>4}  "
            f"средняя вероятность {info['mean_p']:.3f}  {info['actions']}"
        )

    # ---------------------------------------------------------------- важность
    gain = model.booster_.feature_importance(importance_type="gain")
    importance = sorted(zip(FEATURES, gain), key=lambda kv: -kv[1])
    print("\nЧто больше всего влияет на решение:")
    for name, value in importance[:12]:
        print(f"  {name:<24} {value:>12,.0f}")

    # ---------------------------------------------------------------- сохранение
    metrics = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_tr)),
        "rows_valid": int(len(X_va)),
        "rows_test": int(len(X_te)),
        "fraud_rate": float(y_all.mean()),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "pr_auc_baseline": float(y_te.mean()),
        "brier_raw": float(brier_raw),
        "brier_calibrated": float(brier_cal),
        "operating_points": rec_ops,
        "reliability": reliability,
        "training_curve": training_curve,
        "type_accuracy": type_acc,
        "best_fixed_threshold": float(best_t),
        "cost_ours": ours,
        "cost_baselines": report["baselines"],
        "detection_rate": report["detection_rate"],
        "friction_rate_honest": report["friction_rate_honest"],
        "action_counts": report["action_counts"],
        "per_type": report["per_type"],
        "feature_importance": [
            {"feature": n, "gain": float(v)} for n, v in importance
        ],
        "economics": DEFAULT_ECONOMICS.to_dict(),
        "anomaly_enabled": combiner is not None,
        "anomaly_cap": combiner.cap if combiner is not None else None,
        "pr_auc_supervised_only": (
            float(average_precision_score(y_te, p_te_supervised))
            if combiner is not None else None
        ),
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Сводка графа связей за весь месяц. Считается здесь, а не в API:
    # кольцо живёт во времени и не обязано попасть в тестовый срез, а метки
    # для отчёта по прошлому периоду банку доступны честно.
    graph_summary = {"device": [], "ip": []}
    for kind, column in (("device", "device_id"), ("ip", "ip")):
        grouped = df.groupby(column).agg(
            clients=("client_id", "nunique"),
            transactions=("tx_id", "count"),
            amount=("amount", "sum"),
            fraud=("is_fraud", "sum"),
        )
        grouped = grouped[grouped["clients"] >= 2].sort_values("clients", ascending=False)
        for key, row in grouped.head(40).iterrows():
            members = (
                df.loc[df[column] == key]
                .groupby("client_id")
                .agg(transactions=("tx_id", "count"),
                     amount=("amount", "sum"),
                     fraud=("is_fraud", "sum"))
            )
            graph_summary[kind].append({
                "key": str(key),
                "clients": int(row["clients"]),
                "transactions": int(row["transactions"]),
                "amount": float(row["amount"]),
                "fraud_share": float(row["fraud"] / max(row["transactions"], 1)),
                "members": [
                    {
                        "client_id": str(cid),
                        "transactions": int(m["transactions"]),
                        "amount": float(m["amount"]),
                        "fraud_share": float(m["fraud"] / max(m["transactions"], 1)),
                    }
                    for cid, m in members.iterrows()
                ],
            })
    (MODELS_DIR / "graph.json").write_text(
        json.dumps(graph_summary, ensure_ascii=False), encoding="utf-8"
    )

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "anomaly_channel": anomaly_channel,
        "combiner": combiner,
        "type_model": type_model,
        "type_classes": type_classes,
        "features": FEATURES,
        "metrics": metrics,
    }
    joblib.dump(bundle, args.out, compress=3)
    joblib.dump(store, MODELS_DIR / "feature_store.joblib", compress=3)

    # Размеченный тестовый срез — им живёт лента и дашборд
    contrib = model.booster_.predict(X_te.to_numpy(), pred_contrib=True)
    scored = te_frame.copy()
    scored["p_fraud"] = p_te
    for j, c in enumerate(type_classes):
        scored[f"p_{c}"] = type_probs_te[:, j]
    scored["action"] = chosen
    # Три признака, сильнее всего толкнувшие оценку вверх
    top_idx = np.argsort(-contrib[:, : len(FEATURES)], axis=1)[:, :3]
    scored["top_reasons"] = [
        json.dumps(
            [
                {"feature": FEATURES[j], "contribution": round(float(contrib[i, j]), 4)}
                for j in top_idx[i]
                if contrib[i, j] > 0
            ],
            ensure_ascii=False,
        )
        for i in range(len(scored))
    ]
    scored.to_csv(DATA_DIR / "scored_test.csv", index=False)

    print(f"\nСохранено: {args.out}")
    print(f"           {MODELS_DIR / 'feature_store.joblib'}")
    print(f"           {MODELS_DIR / 'metrics.json'}")
    print(f"           {MODELS_DIR / 'graph.json'}")
    print(f"           {DATA_DIR / 'scored_test.csv'}")
    print(f"Готово за {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
