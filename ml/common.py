"""Справочники, общие для генератора данных, обучения и онлайн-скоринга."""

from __future__ import annotations

# ---------------------------------------------------------------- гео
# Города Казахстана: (название, широта, долгота, вес по населению)
CITIES: dict[str, tuple[float, float, float]] = {
    "Алматы": (43.2220, 76.8512, 2.2),
    "Астана": (51.1694, 71.4491, 1.4),
    "Шымкент": (42.3417, 69.5901, 1.1),
    "Караганда": (49.8047, 73.1094, 0.5),
    "Актобе": (50.2839, 57.1670, 0.5),
    "Тараз": (42.9000, 71.3667, 0.36),
    "Павлодар": (52.2871, 76.9674, 0.33),
    "Усть-Каменогорск": (49.9787, 82.6014, 0.33),
    "Семей": (50.4111, 80.2275, 0.32),
    "Атырау": (47.0945, 51.9238, 0.29),
    "Костанай": (53.2198, 63.6354, 0.25),
    "Кызылорда": (44.8479, 65.4823, 0.24),
    "Уральск": (51.2333, 51.3667, 0.24),
    "Петропавловск": (54.8667, 69.1500, 0.22),
    "Актау": (43.6410, 51.1980, 0.20),
}

# Зарубежные точки, откуда обычно прилетает фрод по украденным картам
FOREIGN_CITIES: dict[str, tuple[float, float, str]] = {
    "Москва": (55.7558, 37.6173, "RU"),
    "Стамбул": (41.0082, 28.9784, "TR"),
    "Дубай": (25.2048, 55.2708, "AE"),
    "Бангкок": (13.7563, 100.5018, "TH"),
    "Киев": (50.4501, 30.5234, "UA"),
    "Лагос": (6.5244, 3.3792, "NG"),
    "Лондон": (51.5074, -0.1278, "GB"),
    "Бишкек": (42.8746, 74.5698, "KG"),
}

ALL_PLACES: dict[str, tuple[float, float, str]] = {
    **{name: (lat, lon, "KZ") for name, (lat, lon, _w) in CITIES.items()},
    **FOREIGN_CITIES,
}


def place_coords(city: str) -> tuple[float, float]:
    """Координаты города; неизвестный город падает в центр страны."""
    rec = ALL_PLACES.get(city)
    if rec is None:
        return (48.0196, 66.9237)
    return (rec[0], rec[1])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками по большому кругу, км."""
    import math

    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# ---------------------------------------------------------------- торговля
# Категория -> (типичная доля от среднего чека, риск-вес 0..1)
MERCHANT_CATEGORIES: dict[str, tuple[float, float]] = {
    "grocery": (0.6, 0.05),
    "transport": (0.15, 0.05),
    "cafe": (0.4, 0.08),
    "pharmacy": (0.3, 0.05),
    "utilities": (0.5, 0.05),
    "marketplace": (1.0, 0.25),
    "clothing": (1.2, 0.20),
    "electronics": (3.5, 0.55),
    "entertainment": (0.8, 0.20),
    "travel": (4.0, 0.45),
    "jewelry": (6.0, 0.70),
    "crypto": (5.0, 0.90),
    "gambling": (2.0, 0.85),
    "atm_withdrawal": (2.0, 0.60),
    "transfer_p2p": (2.5, 0.50),
}

CATEGORY_LIST: list[str] = list(MERCHANT_CATEGORIES)

CHANNELS: list[str] = ["pos", "ecom", "mobile_app", "atm", "p2p"]

TX_TYPES: list[str] = ["purchase", "transfer", "withdrawal"]

FRAUD_TYPES: list[str] = ["stolen_card", "social_eng", "fraud_ring"]

# Валюта одна на всю систему — тенге
CURRENCY = "KZT"


def category_risk(category: str) -> float:
    return MERCHANT_CATEGORIES.get(category, (1.0, 0.3))[1]


# ---------------------------------------------------------------- загрузка


def read_transactions(path):
    """Прочитать CSV с транзакциями, надёжно разобрав время.

    pandas угадывает формат даты по первой строке файла и падает, если ниже
    формат хоть немного другой. Здесь разбираем явно и не даём одной строке
    уронить весь датасет.
    """
    import pandas as pd

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
        bad = int(df["timestamp"].isna().sum())
        if bad:
            df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df
