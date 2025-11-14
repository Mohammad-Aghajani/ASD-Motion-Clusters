# Export Samples

This folder ships lightweight samples that mirror the CSV schemas described in the paper. They are derived from the full analysis but truncated to the first few rows for illustration.

- `example_aggregated_features.csv` — window-level kinematic descriptors (actor, emotion, ROI, feature type, statistic, value).
- `example_cluster_memberships.csv` — cluster assignments per (actor, emotion, ROI, method, k) with diagnostic scores.
- `example_social_score_summary_fdr.csv` — ANOVA and correlation outputs with Benjamini–Hochberg q-values.

When you run `python kbc_cleaned.py --force` the full-resolution tables will be written under `results/exports/` using the same column layouts.
