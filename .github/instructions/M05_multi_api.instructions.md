# M05 – Multi-API support

Goal: Add OpenWeatherMap and/or WeatherAPI behind a unified provider interface and factory.

In-scope
- Provider adapters; factory pattern; per-provider policy overrides.
- Response normalization to canonical model; error mapping.

Out-of-scope
- Advanced analytics or caching.

Deliverables
- Adapter specs and test cases (with mocking) per provider.

Success criteria
- Swap providers via config without code changes; contract tests pass.

Depends on: M02, M04, M03
Risks
- API key quotas. Mitigation: record/replay or stubbed tests; backoff.

Open questions
- Which provider(s) to prioritize? Keys availability and quotas?
