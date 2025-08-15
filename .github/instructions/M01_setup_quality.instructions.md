# M01 – Project setup and quality baseline

Goal: Establish repo hygiene, tooling, and CI scaffolding to keep velocity high.

In-scope
- Decide dependency manager and Python version, define minimal env bootstrap.
- Lint/format/type: ruff, black, isort, mypy (config only; no code changes yet).
- Test harness: pytest + pytest-asyncio skeleton; one placeholder test.
- Pre-commit configuration (hooks for ruff/black/isort/mypy).
- Logging baseline decision (structured JSON via stdlib or structlog).
- Settings approach (Pydantic Settings v2) decision doc.

Out-of-scope
- Implementing features; only scaffolding decisions and configs.

Deliverables
- Written configs plan and CI placeholder file list (no actual code changes now).

Success criteria
- A documented plan enabling commands to run locally and in CI with zero errors once implemented.

Dependencies: —
Risks
- Over-configuring early. Mitigation: keep config minimal and incremental.

Open questions
- uv approved as package manager? Python 3.11 or 3.12?
