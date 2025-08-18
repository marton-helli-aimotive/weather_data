# ADR 002: Quality Tooling Baseline

- Status: accepted
- Context: Keep the repo clean and enforce quality early.
- Decision: Ruff for lint+format, Black for check, isort for import order, mypy for typing. pytest (+ asyncio) for tests.
- Consequences: Pre-commit configured; CI runs these checks.
