# M07 – Processing engine enhancements

Goal: Extend processing to time series, feature engineering, and geospatial basics using pandas/Polars.

In-scope
- Rolling windows, lag features, basic anomaly detection.
- Time zone normalization and resampling.
- Geospatial basics: clustering by proximity; groundwork for maps.

Out-of-scope
- Heavy ML models; advanced geostatistics.

Deliverables
- Transformation specs and evaluation metrics.

Success criteria
- Deterministic transformations with tests; measurable speed on Polars vs pandas for key ops.

Depends on: M03, M04
Risks
- Performance regressions. Mitigation: benchmark harness and guardrails.

Open questions
- Minimum time granularity (hourly?) and window sizes to standardize?
