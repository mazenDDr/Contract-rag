.PHONY: setup check lint fmt test

setup:
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,ui]"

check: lint test

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

test:
	pytest -q
