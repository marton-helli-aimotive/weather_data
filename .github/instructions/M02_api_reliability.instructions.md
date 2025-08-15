# M02 – API abstraction and reliability

Goal: A robust async HTTP abstraction with rate limiting, retries (expo backoff), and circuit breaker.

In-scope
- Define interface for providers (async methods, request contract, error types).
- Reliability policies: retry/backoff, timeouts, jitter, concurrency limits.
- Circuit breaker semantics (open/half-open/closed) and metrics hooks.

Out-of-scope
- Concrete providers beyond a mock/stub.

Deliverables
- Interface spec and policy definitions ready for implementation.

Success criteria
- Clear contracts for error handling and backpressure; test plan defined.

Depends on: M01
Risks
- Hidden provider-specific quirks. Mitigation: adapter pattern with per-provider policies.

Open questions
- Global vs per-host rate limits? Default budgets?
