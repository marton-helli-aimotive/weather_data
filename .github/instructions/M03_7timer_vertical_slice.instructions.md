# M03 – 7timer MVP vertical slice

Goal: End-to-end slice: fetch 7timer → validate with Pydantic → process (basic stats) → expose results to CLI.

In-scope
- Use existing starter code pathways; harden around the new abstractions.
- Minimal cities set; concurrent fetch; graceful failure path.
- Stats: min/max/mean/std by city; expose JSON/print for early validation.

Out-of-scope
- Dashboard, caching, streaming, multi-API.

Deliverables
- Slice plan and acceptance tests outline.

Success criteria
- Deterministic run with non-empty results in most cases; error paths logged; unit/integration tests planned.

Depends on: M02, M04
Risks
- 7timer schema variability. Mitigation: strict models with tolerant parsing and schema guards.

Open questions
- Use UTC timestamps from API or localize? Timezone handling standard?
