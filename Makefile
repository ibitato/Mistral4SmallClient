.PHONY: help sync lock build format format-check lint typecheck typecheck-mypy typecheck-pyright test docs docs-check check run cancel-probe clean

UV ?= uv

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync       - create or update the uv virtual environment' \
		'  make lock       - refresh uv.lock' \
		'  make build      - build wheel and source distribution in dist/' \
		'  make format     - format Python code with ruff' \
		'  make format-check - verify formatting without changing files' \
		'  make lint       - lint Python code with ruff' \
		'  make typecheck  - run mypy and pyright' \
		'  make typecheck-mypy - run mypy' \
		'  make typecheck-pyright - run pyright' \
		'  make test       - run pytest' \
		'  make docs       - generate docs/reference.md from docstrings' \
		'  make docs-check - verify generated docs and language hygiene' \
		'  make cancel-probe - probe stream cancellation and follow-up recovery' \
		'  make check      - verify format, lint, typecheck, and docs-check' \
		'  make run        - run the CLI entrypoint' \
		'  make clean      - remove local caches and the virtual environment'

sync:
	$(UV) sync --group dev

lock:
	$(UV) lock

build:
	$(UV) build

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

typecheck-mypy:
	$(UV) run mypy

typecheck-pyright:
	$(UV) run pyright

typecheck: typecheck-mypy typecheck-pyright

test:
	$(UV) run pytest

docs:
	$(UV) run python scripts/check_docs.py --fix

docs-check:
	$(UV) run python scripts/check_docs.py

cancel-probe:
	$(UV) run python scripts/cancel_probe.py

check: format-check lint typecheck docs-check

run:
	$(UV) run mistral4cli

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache __pycache__ build dist htmlcov
