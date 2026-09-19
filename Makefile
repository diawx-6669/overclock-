.PHONY: install data train run test all clean

install:
	pip install -r requirements.txt

data:
	python -m ml.generate_data

train:
	python -m ml.train

run:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

test:
	python -m pytest tests -q

all: install data train

clean:
	rm -f data/*.csv models/*.joblib models/metrics.json
