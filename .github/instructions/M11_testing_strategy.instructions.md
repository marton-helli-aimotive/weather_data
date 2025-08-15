# M11 – Comprehensive testing strategy

Goal: Testing pyramid with unit, integration, property-based, performance, load, and contract tests.

In-scope
- Unit tests for models, utilities, and provider adapters.
- Integration tests with HTTP mocking; record/replay for provider contracts.
- Hypothesis property-based tests for parsers/transformations.
- Benchmarks and lightweight load tests.

Out-of-scope
- External paid load infrastructure.

Deliverables
- Test strategy doc, coverage targets, CI gates.

Success criteria
- >90% unit coverage; green builds with deterministic tests; perf baselines captured.

Depends on: M01; spans M02–M10
Risks
- Flaky tests from live APIs. Mitigation: strict mocking and time control.

Open questions
- Minimum coverage gate in CI? Accept flakes quarantine lane?
