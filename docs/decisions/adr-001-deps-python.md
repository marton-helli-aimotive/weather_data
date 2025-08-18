# ADR 001: Dependency and Python Version

- Status: accepted
- Context: Need a fast, reproducible Python workflow.
- Decision: Use uv for dependency management; target Python 3.11.
- Consequences: CI includes setup-uv and uses `uv run` for tools.
