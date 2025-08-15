# M09 – Streaming patterns

Goal: Support near-real-time ingestion/processing via async queues and backpressure.

In-scope
- Async producer/consumer pattern; batching; graceful shutdown.
- Sliding-window metrics in streaming mode.

Out-of-scope
- External brokers (Kafka); use in-process queues for now.

Deliverables
- Streaming design and minimal simulation plan.

Success criteria
- Demonstrated end-to-end with simulated bursts; no data loss under normal backpressure.

Depends on: M07, M08
Risks
- Queue overflows; Mitigation: bounded queues and shed/slow strategies.

Open questions
- Target update cadence (e.g., per minute) and acceptable lag?
