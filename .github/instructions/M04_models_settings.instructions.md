# M04 – Data models and configuration

Goal: Canonical Pydantic v2 models and Pydantic Settings-based configuration.

In-scope
- Weather domain models; provider-normalized record; enums; validators.
- Schema evolution strategy (versioned models, adapters).
- Settings: API keys, endpoints, rate limits; env-var and .env support.

Out-of-scope
- Persistence stores beyond in-memory for now.

Deliverables
- Model and settings specs with migration notes.

Success criteria
- Models cover current and near-term providers; mypy-friendly type hints.

Depends on: M01
Risks
- Overfitting models to one provider. Mitigation: normalized core + provider mappers.

Open questions
- Any compliance constraints on storing API keys locally?
