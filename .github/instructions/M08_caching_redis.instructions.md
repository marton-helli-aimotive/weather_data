# M08 – Caching strategy (Redis)

Goal: Reduce API load and improve latency with response/result caching.

In-scope
- Redis as cache for provider responses and normalized records.
- Keys, TTLs, invalidation policies; cache versioning by model schema.

Out-of-scope
- Persistent databases beyond cache.

Deliverables
- Cache design doc and test scenarios (hit/miss/stale).

Success criteria
- Stable hit rates in repeated runs; correctness preserved under cache.

Depends on: M02, M05
Risks
- Stale data surfacing. Mitigation: freshness-aware keys and TTL.

Open questions
- Coarse vs per-city TTL? Cache in CI allowed or bypassed?
