"""Второй канал: поиск аномалий без учителя.

Зачем он нужен. Модель с учителем знает ровно те схемы, которые были в
обучающей выборке. Мошенники меняют схемы быстрее, чем банк переобучает
модель, и новая схема проходит мимо: она не похожа ни на один размеченный
пример фрода. Зато она почти всегда не похожа и на нормальное поведение
клиента — а вот это можно заметить без единой метки.

Поэтому в системе два независимых канала:

  1. «Похоже на известное мошенничество» — LightGBM по разметке.
  2. «Не похоже на нормальное поведение» — Isolation Forest без разметки.

Второй канал слабее первого на знакомых схемах и не должен их перебивать.
Его работа — не дать совсем незнакомой схеме пройти как ни в чём не бывало.

Как каналы НЕ надо сводить. Первая версия обучала логистическую регрессию на
валидации: пусть данные сами решат вес каждого канала. Эксперимент с
незнакомой схемой показал, что так делать нельзя. Регрессия учится на
размеченной валидации, где спрятанная схема помечена как «не фрод» — то есть
её просят подобрать вес для распознавания того, что ей назвали нормой. Она
закономерно обнуляет канал аномалий, а на кольцах карт и вовсе даёт ему
отрицательный вес: полнота по спрятанной схеме падала с 15.6% до 3.1%, то
есть смесь работала хуже одной модели с учителем.

Как надо. Каналы соединяются по схеме «или»:

    p = 1 - (1 - p_модели) * (1 - p_аномалии)

Такая связка умеет только повышать подозрение и никогда не понижать.
Аномалию нельзя «переспорить» уверенностью модели в том, что всё нормально —
а именно этого мы и хотим от страховки на случай незнакомой схемы. Оценка
аномальности при этом переводится в вероятность изотонической регрессией по
известным меткам, так что в стоимостную модель по-прежнему попадает
настоящая вероятность, а не произвольное число.

Насколько это работает, проверяется экспериментом: из обучения полностью
убирается один вид мошенничества, и замеряется, сколько его ловит каждый
канал. См. ml/report.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

# Признаки, по которым имеет смысл искать аномалию: отклонение операции от
# привычного поведения самого клиента и от нормальной структуры связей.
# Сырые величины вроде часа суток или кода канала сюда не берём — они
# описывают операцию, а не её необычность.
ANOMALY_FEATURES: list[str] = [
    "amount_to_avg",
    "amount_z",
    "amount_to_max",
    "is_amount_record",
    "hours_since_prev",
    "tx_count_1h",
    "tx_count_24h",
    "amount_24h_to_avg",
    "is_new_device",
    "device_tx_share",
    "device_age_days",
    "is_new_city",
    "city_tx_share",
    "distance_from_home_km",
    "travel_speed_kmh",
    "is_vpn",
    "clients_per_device",
    "clients_per_ip",
    "recipient_clients_24h",
    "recipient_inbound_24h",
    "recipient_amount_24h",
    "recipient_is_new_to_bank",
    "component_clients",
    "is_new_recipient",
    "is_new_category",
    "remote_access",
    "call_minutes_before",
    "session_duration_s",
    "own_context",
]


class AnomalyChannel:
    """Isolation Forest над «отклонением от нормы» плюс приведение к шкале."""

    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.features = ANOMALY_FEATURES
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            # Сто деревьев, а не триста. Замер показал, что качество от их
            # числа тут не зависит (ROC-AUC 0.930 против 0.934, PR-AUC в
            # пределах шума), а задержка растёт линейно: обход леса занимал
            # 18.9 мс из 23.3 мс всего решения — восемьдесят процентов
            # времени уходило на канал, который ничего от этого не выигрывал.
            n_estimators=100,
            contamination=contamination,
            max_samples=min(4096, 100_000),
            random_state=random_state,
            n_jobs=4,
        )

    def fit(self, X: pd.DataFrame) -> "AnomalyChannel":
        """Учится на обычном потоке. Разметка не используется вообще.

        Небольшая доля фрода в обучающем потоке — это нормально и даже
        правильно: в проде никто не выдаёт чистую выборку honest-only.
        """
        Z = self.scaler.fit_transform(X[self.features].to_numpy())
        self.forest.fit(Z)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Чем больше, тем необычнее. Ноль примерно соответствует границе нормы."""
        Z = self.scaler.transform(X[self.features].to_numpy())
        # score_samples: чем меньше, тем аномальнее. Меняем знак, чтобы
        # «больше» означало «подозрительнее» — так читать проще.
        return -self.forest.score_samples(Z)


class NoisyOr:
    """Связка двух каналов по «или»: подозрение можно только усилить.

    Оценку аномальности сначала переводим в вероятность фрода изотонической
    регрессией на валидации — иначе в стоимостную модель поедет число, которое
    вероятностью только притворяется, и весь расчёт ожидаемых потерь поплывёт.
    """

    # Значение по умолчанию подобрано перебором, а не на глаз. Замеры на
    # отложенном тесте (цена — PR-AUC по знакомым схемам, польза — полнота по
    # схеме, спрятанной от обучения):
    #
    #     потолок   PR-AUC знакомых   незнакомая схема
    #     нет            0.9329              0.5%
    #     0.10           0.9295              1.1%
    #     0.15           0.9265             22.7%
    #     0.20           0.9244             60.0%     <- выбран
    #     0.30           0.9200             63.8%
    #     0.60           0.9118             63.8%
    #
    # Ниже 0.15 страховка не срабатывает, выше 0.30 платишь больше и не
    # получаешь ничего.
    DEFAULT_CAP = 0.20

    def __init__(self, cap: float | None = None):
        # Потолок вклада канала аномалий. Канал без учителя не должен в
        # одиночку доводить операцию до блокировки: он говорит «это странно»,
        # а не «это мошенничество». Выше потолка поднимает только разметка.
        cap = self.DEFAULT_CAP if cap is None else cap
        self.cap = cap
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, anomaly: np.ndarray, y: np.ndarray) -> "NoisyOr":
        self.calibrator.fit(np.asarray(anomaly, dtype=float), np.asarray(y))
        return self

    def anomaly_probability(self, anomaly: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.predict(np.asarray(anomaly, dtype=float)),
                       0.0, self.cap)

    def combine(self, p_sup: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        """Сырое «или» до перекалибровки."""
        p_sup = np.clip(np.asarray(p_sup, dtype=float), 0.0, 1.0)
        p_anom = self.anomaly_probability(anomaly)
        return 1.0 - (1.0 - p_sup) * (1.0 - p_anom)

    def fit_final(self, p_sup: np.ndarray, anomaly: np.ndarray,
                  y: np.ndarray) -> "NoisyOr":
        """Вернуть смеси смысл вероятности.

        «Или» считает каналы независимыми, а они смотрят на одни и те же
        признаки и потому сильно зависимы. Сырая смесь систематически
        завышает вероятность: в первом прогоне трение по честным клиентам
        выросло с 2.7% до 14.4%, потому что стоимостная модель принимала
        завышенное число за настоящую вероятность и начинала перестраховываться.

        Лечится вторым проходом изотонической регрессии по валидации.
        Преобразование монотонное, поэтому порядок операций по подозрительности
        не меняется — страховка от незнакомых схем продолжает работать, она
        ведь опирается на ранжирование, — а шкала снова становится честной.
        """
        self.final = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.final.fit(self.combine(p_sup, anomaly), np.asarray(y))
        return self

    def predict(self, p_sup: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        raw = self.combine(p_sup, anomaly)
        final = getattr(self, "final", None)
        return raw if final is None else np.clip(final.predict(raw), 0.0, 1.0)
