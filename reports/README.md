# Reports

Use this directory for the written technical report, ablation notes, and experiment summaries.

Current report files:

- `technical_report.md`: full research-style report draft covering motivation, method, diagnostics, experiments, limitations, and next steps

Generated experiment artifacts:

- `outputs/<run>/ablation_summary.md`: physical-only versus physical+neural metric table
- `outputs/<run>/phase*/diagnostics/decomposition_grid.png`: component visualizations
- `outputs/<run>/phase*/diagnostics/loss_curves.svg`: optimization curves
- `outputs/<run>/phase*/diagnostics/component_statistics.json`: tensor statistics
- `outputs/<run>/phase*/diagnostics/lighting_coefficients.json`: recovered illumination coefficients
- `outputs/<run>/phase*/diagnostics/diagnostic_summary.md`: phase-level summary
