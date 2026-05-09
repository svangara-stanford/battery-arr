.PHONY: install test lint demo clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src scripts tests

demo:
	python scripts/generate_demo_data.py --out data/demo --n-cells 72 --n-cycles 100 --seed 42
	python scripts/train_baseline.py --processed-dir data/demo --out runs/demo_baseline --max-cycle 100 --seed 42
	python scripts/evaluate_submission.py --truth runs/demo_baseline/labels_test.csv --pred runs/demo_baseline/predictions.csv --out runs/demo_baseline/eval_metrics.json

clean:
	rm -rf data/demo/* data/processed/* runs/* .pytest_cache .ruff_cache htmlcov .coverage
	touch data/demo/.gitkeep data/processed/.gitkeep runs/.gitkeep
