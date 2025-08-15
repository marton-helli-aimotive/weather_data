# M13 – Containerization and CI/CD

Goal: Multi-stage Containerfile, podman-compose dev env, and GitHub Actions CI.

In-scope
- Slim Python base, uv/pip caching, non-root user, healthchecks.
- Compose services: app, Redis, optional DuckDB volume.
- CI: lint/type/test; optional build & publish artifact.

Out-of-scope
- Production-grade CD beyond example.

Deliverables
- Containerization plan and CI workflow outline.

Success criteria
- Image builds reproducibly; compose boots locally; CI passes on PRs.

Depends on: M01, integrates M10–M12
Risks
- Windows path quirks. Mitigation: cross-platform paths and env handling.

Open questions
- Registry and naming conventions? Podman only or Docker too?
