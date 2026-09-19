.PHONY: install data train report bench run test all clean demo

install:
	pip install -r requirements.txt

data:
	python -m ml.generate_data

train:
	python -m ml.train

report:
	python -m ml.report

bench:
	python -m ml.bench

run:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

test:
	python -m pytest tests -q

all: install data train report bench

demo:
	python -m scripts.export_demo

clean:
	rm -f data/*.csv models/*.joblib models/metrics.json
