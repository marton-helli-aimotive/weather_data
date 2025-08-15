# M10 – Interactive dashboard and visualizations

Goal: Streamlit-based dashboard with Plotly: time series, geographic heatmaps, and basic auth.

In-scope
- Time series plots with zoom/pan; city selectors.
- Geographic heatmap/choropleth; optional 3D surface for temp/pressure.
- Export: CSV/Excel and PDF (simple template) plan.
- Basic auth/session management approach.

Out-of-scope
- Complex multi-tenant auth; heavy theming.

Deliverables
- UI/UX skeleton plan; data contract between backend and UI; export strategy.

Success criteria
- Interactive filters update plots within target latency; exports produce expected files.

Depends on: M07, M06
Risks
- Plot performance on large data. Mitigation: downsampling and server-side aggregation.

Open questions
- Branding requirements? Preferred map tiles/basemaps?
