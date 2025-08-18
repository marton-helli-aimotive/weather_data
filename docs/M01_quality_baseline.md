# Decisions and Baselines for M01

This document captures the minimal quality scaffolding to keep velocity high.

- Python: 3.11+
- Package manager: uv (pin via CI action)
- Code layout: flat for now; `src/` may be introduced by M02/M04.
- Lint/format/type: Ruff (lint+format), Black (check), isort, mypy (config only now)
- Tests: pytest + pytest-asyncio; placeholder test added
- Pre-commit: hooks for ruff/format, black, isort, mypy
- Logging: use stdlib `logging` with JSON formatter later (decided in M12); for now no code changes
- Settings: Pydantic Settings v2 in M04; no code changes now

Local workflow (once tools are installed):
- ruff check .
- ruff format .
- black .
- isort .
- mypy .
- pytest

CI: GitHub Actions workflow runs above checks on pushes/PRs.

Files added by M01:
- `.github/workflows/ci.yml` – CI workflow for quality gates
- `.pre-commit-config.yaml` – local hooks for ruff/black/isort/mypy and hygiene
- `.editorconfig` – consistent editor settings
- `.gitignore` – ignores caches, envs, and scaffolding files

Open questions tracked in `INDEX.md` have answers; no new ones introduced by M01.
