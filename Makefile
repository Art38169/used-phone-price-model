install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

train:
	uv run python src/models/train.py

predict:
	uv run python src/models/predict.py