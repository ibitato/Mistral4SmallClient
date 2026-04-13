.PHONY: help sync lock format format-check lint typecheck test check run clean

UV ?= uv

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync       - create or update the uv virtual environment' \
		'  make lock       - refresh uv.lock' \
		'  make format     - format Python code with ruff' \
		'  make format-check - verify formatting without changing files' \
		'  make lint       - lint Python code with ruff' \
		'  make typecheck  - run mypy' \
		'  make test       - run pytest' \
		'  make check      - verify format, lint, and typecheck' \
		'  make run        - run the CLI entrypoint' \
		'  make clean      - remove local caches and the virtual environment'

sync:
	$(UV) sync --group dev

lock:
	$(UV) lock

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy src

test:
	$(UV) run pytest

check: format-check lint typecheck

run:
	$(UV) run mistral4cli

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache __pycache__ build dist htmlcov
