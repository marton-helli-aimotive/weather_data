# ADR 003: Logging & Settings Approach

- Status: accepted
- Context: Need structured logs and 12-factor config later.
- Decision: Use stdlib `logging` with a JSON formatter in M12; Use Pydantic Settings v2 in M04.
- Consequences: No code yet; keep configs minimal. Documented for future milestones.
