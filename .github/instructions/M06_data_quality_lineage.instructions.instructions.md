# M06 – Data quality, lineage, and freshness

Goal: Ensure correctness and traceability of data with automated checks and lineage tracking.

In-scope
- Data quality rules: missing values, outliers, temporal consistency.
- Freshness monitors with thresholds; alerts interface contract.
- Lineage metadata: source, fetch time, transformation steps, versions.

Out-of-scope
- Full-fledged data catalog or external lineage store.

Deliverables
- Spec for quality checks and lineage fields; test plan for failures.

Success criteria
- Failing records are flagged/quantified; pipeline returns quality summary per run.

Depends on: M03, M04
Risks
- Over-aggressive rules causing false positives. Mitigation: configurable severity and thresholds.

Open questions
- Acceptable data loss vs. lenient acceptance? Thresholds per metric/city?
