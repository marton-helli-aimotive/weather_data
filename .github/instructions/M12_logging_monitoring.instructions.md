# M12 – Logging, metrics, and monitoring

Goal: Structured JSON logging and Prometheus-style metrics with health endpoints.

In-scope
- Log structure, correlation IDs, and sampling.
- Metrics: request counts, latencies, cache hits, quality errors, freshness.
- Health and readiness semantics.

Out-of-scope
- External observability stacks beyond local scrape/export.

Deliverables
- Observability spec and metric names; alert conditions.

Success criteria
- Logs parsable; metrics scrapeable; alerts fire under simulated faults.

Depends on: M01; spans M02–M10
Risks
- Noise overload. Mitigation: levels and sampling.

Open questions
- Preferred metric namespace and labels convention?
