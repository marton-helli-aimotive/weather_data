# Project Milestones Index

Purpose: Provide an ordered plan, dependencies, success criteria, and open questions for the Advanced Weather Data Engineering Challenge.

## Legend
- ID: Milestone identifier
- Order: Recommended execution order (lower runs earlier)
- Depends on: IDs that must be done first

## Ordered Milestones

1. M01 – Project setup and quality baseline (Depends: —)
2. M02 – API client abstraction and reliability infra (rate limit, retries, circuit breaker) (Depends: M01)
3. M03 – 7timer provider MVP (vertical slice fetch → validate → process → stats) (Depends: M02, M04)
4. M04 – Data models and configuration (Pydantic v2, settings, schema evolution) (Depends: M01)
5. M05 – Multi-API support (OpenWeatherMap, WeatherAPI) with unified interface (Depends: M02, M04, M03)
6. M06 – Data quality checks, lineage, freshness monitoring (Depends: M03, M04)
7. M07 – Processing engine enhancements (pandas + Polars; time series/feature eng foundation) (Depends: M03, M04)
8. M08 – Caching strategy (Redis) for API responses and derived data (Depends: M02, M05)
9. M09 – Streaming patterns (async queues; simulated real-time ingestion) (Depends: M07, M08)
10. M10 – Interactive dashboard (Streamlit) with Plotly visualizations (Depends: M07, M06)
11. M11 – Comprehensive testing strategy (unit/integration/property/perf/load) (Depends: M01; spans M02–M10)
12. M12 – Logging, metrics, monitoring (structured JSON, Prometheus) (Depends: M01; spans M02–M10)
13. M13 – Containerization and CI/CD (Containerfile, podman-compose, GitHub Actions) (Depends: M01; integrates M10–M12)
14. M14 – Documentation and performance report (Depends: most milestones; finalize at end, draft early)

## Rationale
- Start with quality gates (M01), then deliver a working vertical slice (M02/M04/M03) to prove the architecture. Expand providers (M05), then data quality (M06) and processing depth (M07), followed by cache (M08) and streaming (M09). Add UI (M10), deepen tests/observability (M11/M12), and package/deploy (M13). Wrap with docs and perf (M14).

## Open Questions (confirm before coding)
1. Dashboard choice: Streamlit vs Dash? Answer: Streamlit.
2. Which additional provider(s) beyond 7timer: OpenWeatherMap, WeatherAPI, or both? Keys available? Answer:
  - 7timer key is: not required
  - WeatherAPI key is: 0e919c6354e74601a7b131057251508
  - OpenWeatherMap key is: currently unavailable
3. Provider rate limits/backoff specifics to honor? Answer: must remain within free limits
4. Data freshness thresholds/SLA per city/provider? Answer: up to you
5. Dashboard auth: basic password vs OAuth/SSO? Answer: basic password
6. Approve Redis and DuckDB for local dev (via podman-compose)? Answer: yes
7. Dependency manager preference: uv vs pip/venv; Python version constraints? Answer: uv, python 3.11+
8. Ensure Docker compatibility in addition to Podman? Answer: not necessary
9. Geospatial basemap access policy (online vs local)? Regions of interest? Answer: up to you, roi = Budapest
10. Performance targets for ingestion latency and dashboard update cadence? Answer: up to you
