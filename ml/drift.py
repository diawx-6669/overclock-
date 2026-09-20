"""Мониторинг дрейфа: когда модели пора переобучаться.

В ограничениях проекта записано честно: модель обучена на одном месяце, и в
проде нужен мониторинг дрейфа. Этот модуль закрывает ту строчку.

Дрейф бывает двух видов, и путать их дорого.

Первый — сдвиг данных. Поток стал другим: появился новый эквайер, сменился
формат идентификаторов устройств, выросла доля переводов. Модель ещё может
работать правильно, но она видит не то, на чём училась. Меряем индексом
PSI (population stability index) по каждому признаку.

Второй — сдвиг качества. Вероятности перестали соответствовать реальности:
модель говорит «тридцать процентов», а фродом оказывается пять. Это опаснее,
потому что стоимостная модель считает ожидаемые потери именно по вероятности,
и все решения тихо разъезжаются. Меряем разницей между предсказанной и
фактической долей фрода по неделям.

Пороги PSI взяты отраслевые: до 0.1 — сдвига нет, 0.1-0.25 — умеренный,
выше 0.25 — значимый, признак стоит смотреть руками.

Запуск:
    python -m ml.drift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common import read_transactions
from ml.features import FEATURES, build_matrix
from ml.train import time_split

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

# Накопительные признаки. Они растут по ходу потока просто потому, что у
# клиента со временем прибавляется истории: устройство «стареет», список
# операций удлиняется, компонента графа обрастает связями. Между началом и
# концом месяца они сдвигаются всегда, и PSI на них срабатывает не на дрейф
# модели, а на разогрев состояния.
#
# Путать эти две вещи дорого: если дежурный каждый день видит шесть красных
# признаков, которые красные по построению, он перестаёт смотреть на список
# вообще — и пропустит седьмой, настоящий.
ACCUMULATING = {
    "device_age_days", "history_len", "component_size", "component_clients",
    "client_tenure_days", "recipient_age_days", "amount_to_max", "amount_z",
    "clients_per_device", "clients_per_ip", "clients_per_recipient",
    "device_tx_share", "city_tx_share", "night_share",
}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index между двумя выборками одного признака.

    Границы корзин берём по обучающей выборке: вопрос стоит «насколько новый
    поток не похож на тот, на котором училась модель», а не наоборот.
    Пустые корзины сглаживаем — иначе один отсутствующий диапазон даёт
    бесконечность и прячет всё остальное.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:                      # почти константный признак
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_share = np.histogram(expected, bins=edges)[0] / expected.size
    a_share = np.histogram(actual, bins=edges)[0] / actual.size
    floor = 1e-4                            # сглаживание пустых корзин
    e_share = np.clip(e_share, floor, None)
    a_share = np.clip(a_share, floor, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def verdict(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def calibration_by_period(
    timestamps: pd.Series, p: np.ndarray, y: np.ndarray, freq: str = "W"
) -> list[dict]:
    """Предсказанная доля фрода против фактической, по неделям.

    Именно эта таблица первой показывает, что модель поплыла: столбцы
    «предсказано» и «фактически» начинают расходиться.
    """
    frame = pd.DataFrame({"ts": pd.to_datetime(timestamps), "p": p, "y": y})
    out: list[dict] = []
    for period, part in frame.groupby(pd.Grouper(key="ts", freq=freq)):
        if len(part) < 200:
            continue
        predicted = float(part["p"].mean())
        actual = float(part["y"].mean())
        out.append({
            "period": str(period.date()),
            "count": int(len(part)),
            "predicted": predicted,
            "actual": actual,
            "gap": predicted - actual,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Мониторинг дрейфа")
    parser.add_argument("--data", default=str(DATA_DIR / "transactions.csv"))
    parser.add_argument("--profiles", default=str(DATA_DIR / "clients.csv"))
    parser.add_argument("--scored", default=str(DATA_DIR / "scored_test.csv"))
    parser.add_argument("--out", default=str(MODELS_DIR / "drift.json"))
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    df = read_transactions(args.data).sort_values("timestamp", kind="mergesort")
    df = df.reset_index(drop=True)
    X, _ = build_matrix(df, pd.read_csv(args.profiles))
    X = X.reset_index(drop=True)
    tr, _va, te = time_split(len(df))

    # Сравниваем окна одинаковой длины и подряд идущие: последний кусок
    # обучающего периода против тестового. Брать весь обучающий период против
    # тестового нечестно — в него попадает начало месяца, когда истории у
    # клиентов ещё нет, и накопительные признаки расходятся по построению.
    n_test = len(range(*te.indices(len(df))))
    ref = slice(max(0, te.start - n_test), te.start)

    print("Сдвиг данных: два соседних окна равной длины")
    rows = []
    for feature in FEATURES:
        value = psi(X[feature].to_numpy()[ref], X[feature].to_numpy()[te])
        rows.append({
            "feature": feature,
            "psi": value,
            "verdict": verdict(value),
            "accumulating": feature in ACCUMULATING,
        })
    rows.sort(key=lambda r: -r["psi"])

    real = [r for r in rows if not r["accumulating"]]
    counts = {v: sum(1 for r in real if r["verdict"] == v)
              for v in ("stable", "moderate", "significant")}
    for r in rows[:8]:
        mark = {"stable": " ", "moderate": "~", "significant": "!"}[r["verdict"]]
        tag = "  (накопительный — сдвиг ожидаем)" if r["accumulating"] else ""
        print(f"  {mark} {r['feature']:26} PSI={r['psi']:.4f}{tag}")
    print(f"\n  Среди признаков, которые обязаны быть стабильными: "
          f"стабильны {counts['stable']}, умеренный сдвиг {counts['moderate']}, "
          f"значимый {counts['significant']}")

    report = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "thresholds": {"moderate": PSI_MODERATE, "significant": PSI_SIGNIFICANT},
        "window": {"reference_rows": int(ref.stop - ref.start), "current_rows": n_test},
        "feature_psi": rows,
        "counts": counts,
        "accumulating_features": sorted(ACCUMULATING),
    }

    scored_path = Path(args.scored)
    if scored_path.exists():
        scored = read_transactions(scored_path)
        report["calibration"] = calibration_by_period(
            scored["timestamp"], scored["p_fraud"].to_numpy(),
            scored["is_fraud"].to_numpy(),
        )
        print("\nСдвиг качества: предсказано против фактического, по неделям")
        for c in report["calibration"]:
            print(f"  {c['period']}  n={c['count']:>6,}  предсказано {c['predicted']:.4f}"
                  f"  фактически {c['actual']:.4f}  расхождение {c['gap']:+.4f}")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
