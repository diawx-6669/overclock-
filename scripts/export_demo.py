"""Сборка статичной витрины: один HTML-файл без сервера.

Зачем. Free-тариф Render усыпляет сервис после простоя, и первый посетитель
ждёт пробуждения контейнера. Для защиты проекта это плохо. Витрина открывается
мгновенно и не требует ничего, кроме браузера.

Как. Мы поднимаем настоящее приложение, обходим все ручки, складываем ответы в
один набор и вшиваем его в ту же самую страницу интерфейса. Код интерфейса не
дублируется: `api()` во фронтенде просто видит вшитый набор и берёт данные
оттуда. Значит витрина не может разойтись с продуктом по внешнему виду и
поведению — расходится только источник данных.

Чего в витрине нет: ручного ввода произвольной операции и пересчёта экономики
ползунками. И то и другое требует живой модели, поэтому в витрине они честно
помечены, а не имитируются.

Запуск:
    python -m scripts.export_demo
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "index.html"

# Ровно те запросы, которые делает интерфейс
GET_PATHS = [
    "/api/stats",
    "/api/metrics",
    "/api/report",
    "/api/bench",
    "/api/drift",
    "/api/economics",
    "/api/graph?min_clients=3&limit=6",
    "/api/map",
    "/api/presets",
]
FEED_ACTIONS = ["all", "block", "hold", "step_up", "allow"]
DECISION_KINDS = ["stolen_card", "social_eng", "fraud_ring"]
DECISION_TX_TYPES = ["purchase", "transfer", "withdrawal"]


def collect(client) -> dict:
    """Обойти приложение и собрать все ответы, которые нужны витрине."""
    bundle: dict = {"get": {}, "posts": []}

    for path in GET_PATHS:
        bundle["get"][path] = client.get(path).json()

    for action in FEED_ACTIONS:
        for only_fraud in ("false", "true"):
            path = f"/api/feed?limit=60&action={action}&only_fraud={only_fraud}"
            bundle["get"][path] = client.get(path).json()

    for kind in DECISION_KINDS:
        for tx_type in DECISION_TX_TYPES:
            path = f"/api/decision-map?kind={kind}&tx_type={tx_type}"
            bundle["get"][path] = client.get(path).json()

    # Сценарии симулятора и контрфакты к ним. Тело кладём структурой, а не
    # строкой: сравнивать их будет браузер, приведя обе стороны к своему
    # представлению чисел.
    for preset in bundle["get"]["/api/presets"]["presets"]:
        body = {"transactions": preset["sequence"]}
        sequence = client.post("/api/score-sequence", json=body).json()
        bundle["posts"].append({
            "path": "/api/score-sequence", "body": body, "result": sequence,
        })
        # Интерфейс спрашивает контрфакты для той транзакции, которую вернул
        # скоринг, а она уже дозаполнена значениями по умолчанию. Если взять
        # исходную запись пресета, тела запросов не совпадут и витрина
        # решит, что ответа нет.
        last = sequence["steps"][-1]["transaction"]
        bundle["posts"].append({
            "path": "/api/counterfactual", "body": last,
            "result": client.post("/api/counterfactual", json=last).json(),
        })
    return bundle


def build_page(bundle: dict) -> str:
    """Вшить набор в страницу интерфейса.

    Артефакт сам оборачивает файл в каркас документа, поэтому собственные
    теги html/head/body нужно убрать, а заголовок и стили оставить наверху.
    """
    html = FRONTEND.read_text(encoding="utf-8")

    html = re.sub(r"^\s*<!doctype html>\s*", "", html, flags=re.IGNORECASE)
    html = re.sub(r"</?html[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"</?head[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r"</?body[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(r'<meta[^>]*charset[^>]*>', "", html, flags=re.IGNORECASE)
    html = re.sub(r'<meta[^>]*viewport[^>]*>', "", html, flags=re.IGNORECASE)

    # Ссылка на /docs в подвале без сервера никуда не ведёт
    html = html.replace('<a href="/docs">API</a>', "GitHub")

    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # чтобы не закрыть тег раньше времени
    injection = (
        "<script>\n"
        "/* Ответы настоящего приложения, снятые на этапе сборки витрины. */\n"
        f"window.FH_STATIC = {payload};\n"
        "</script>\n"
    )
    # Набор должен появиться раньше кода интерфейса
    marker = "<script>"
    idx = html.index(marker)
    return html[:idx] + injection + html[idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Статичная витрина Fraud Hunter")
    parser.add_argument("--out", default=str(ROOT / "dist" / "demo.html"))
    args = parser.parse_args()

    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as client:
        bundle = collect(client)

    page = build_page(bundle)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")

    size = out.stat().st_size
    print(f"Ответов GET:      {len(bundle['get'])}")
    print(f"POST-ответов:     {len(bundle['posts'])}")
    print(f"Файл:             {out}  ({size / 1024:.0f} КБ)")
    if size > 16 * 1024 * 1024:
        raise SystemExit("Страница больше 16 МБ — витрина столько не выдержит")


if __name__ == "__main__":
    main()
