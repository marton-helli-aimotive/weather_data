# Weather Data Challenge

This repository contains the scaffolding for M01 (project setup and quality baseline).

- Python 3.11+
- Dependency manager: uv
- CI: GitHub Actions runs lint, format checks, type checks, and tests (see `.github/workflows/ci.yml`).

Local usage with uv:

1) Install dev deps
	uv sync --dev

2) Run quality suite
	uv run ruff check .
	uv run ruff format --check .
	uv run black --check .
	uv run isort --check-only .
	uv run mypy
	uv run pytest -q

Optional: enable pre-commit hooks
	uv run pre-commit install
	uv run pre-commit run --all-files
