.PHONY: help install lint test data train run-api run-app docker-build docker-up

help:
	@echo "Available commands:"
	@echo "  install      : Install dependencies"
	@echo "  lint         : Lint code with ruff and black"
	@echo "  test         : Run test suite with coverage"
	@echo "  data         : Download and process raw datasets"
	@echo "  train        : Train all ML models (churn, segmentation, recommender, forecast)"
	@echo "  run-api      : Run the FastAPI backend service"
	@echo "  run-app      : Run the Streamlit executive dashboard"
	@echo "  docker-build : Build Docker container images"
	@echo "  docker-up    : Launch all services with docker-compose"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

lint:
	ruff check .
	black --check .

test:
	pytest tests/ -v --cov=src

data:
	python scripts/download_data.py
	python scripts/run_etl.py

train:
	python scripts/train_all.py

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-app:
	streamlit run app/streamlit_app.py --server.port 8501

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d
