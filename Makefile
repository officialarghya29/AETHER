# AETHER-ML — reproducible workflows.
SHELL := /bin/bash
PY ?= python3
PIP ?= $(PY) -m pip
export AETHER_CONFIG ?= configs/aether.yaml

.PHONY: help install data train calibrate reference benchmarks figures test lint format typecheck api frontend migrate docker-build docker-up docker-down clean

help:
	@echo "AETHER-ML targets:"
	@echo "  make install      install the package and dev extras"
	@echo "  make data         materialise CIFAR-10 into data/raw"
	@echo "  make train        fine-tune MobileNetV3-Small and ResNet50"
	@echo "  make calibrate    fit temperatures and the learned risk models"
	@echo "  make reference    freeze the drift reference window"
	@echo "  make benchmarks   run the full reliability study"
	@echo "  make figures      render the README figures from recorded results"
	@echo "  make test         run the test suite"
	@echo "  make lint         ruff check"
	@echo "  make format       ruff format"
	@echo "  make typecheck    mypy over aether/"
	@echo "  make api          start the API on :8000"
	@echo "  make frontend     start the dashboard on :5173"
	@echo "  make migrate      apply database migrations"
	@echo "  make docker-up    build and start the full stack"

install:
	$(PIP) install -e ".[dev]"

data:
	$(PY) scripts/download_cifar.py --from-parquet

train:
	$(PY) training/train.py --role lightweight --epochs 10
	$(PY) training/train.py --role advanced --epochs 6

calibrate:
	$(PY) training/calibrate.py --config $(AETHER_CONFIG)

reference:
	$(PY) training/build_reference.py --config $(AETHER_CONFIG)

benchmarks:
	$(PY) experiments/run_benchmarks.py --config $(AETHER_CONFIG)

figures:
	$(PY) experiments/make_figures.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check aether backend training experiments tests scripts

format:
	$(PY) -m ruff format aether backend training experiments tests scripts

typecheck:
	$(PY) -m mypy aether

api:
	$(PY) -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm install && npm run dev

migrate:
	$(PY) -m alembic upgrade head

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
