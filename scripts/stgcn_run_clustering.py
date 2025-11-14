"""Compute quick clustering diagnostics for ST-GCN features.

This mirrors the Euclidean baseline stats so we can compare silhouettes
and number of significant social-score associations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SOCIAL_COLS = [
    "AQ",
    "AQ_SocialSkills",
    "AQ_AttnSwitch",
    "AQ_AttntoDetail",
    "AQ_Communication",
    "AQ_Imagination",
    "RMET",
    "Alexithymia",
    "Cambridge_Behavior",
    "Cambridge_Friendship",
    "PerceivedSocialSupport",
    "TIPI_EXT",
    "TIPI_AGR",
    "TIPI_CON",
    "TIPI_NEU",
    "TIPI_OM",
]


def merge_features(stgcn_csv: Path, agg_csv: Path) -> pd.DataFrame:
    stgcn = pd.read_csv(stgcn_csv)
    agg = pd.read_csv(agg_csv)
    agg["participant"] = agg["actor"].astype(str) + "_" + agg["emotion"]
    merged = stgcn.merge(agg[["participant"] + SOCIAL_COLS], on="participant", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping participants between ST-GCN and aggregated data")
    return merged


def significant_social_counts(df: pd.DataFrame, labels: np.ndarray) -> int:
    counts = 0
    for col in SOCIAL_COLS:
        groups = [df.loc[labels == lab, col].dropna() for lab in np.unique(labels)]
        if any(len(g) < 2 for g in groups):
            continue
        fstat, pval = f_oneway(*groups)
        if np.isfinite(pval) and pval <= 0.05:
            counts += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stgcn", type=Path, default=Path("exports/stgcn_features_summary.csv"))
    parser.add_argument("--aggregated", type=Path, default=Path("exports/aggregated_features_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("exports/stgcn_clustering_summary.csv"))
    args = parser.parse_args()

    df = merge_features(args.stgcn, args.aggregated)
    feature_cols = [c for c in df.columns if c.startswith("stgcn_")]
    X = df[feature_cols].values

    rows = []
    for k in (4, 6, 8):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        sig = significant_social_counts(df, labels)
        singleton_clusters = sum((labels == lab).sum() == 1 for lab in np.unique(labels))
        rows.append({
            "k": k,
            "silhouette": sil,
            "significant_social": sig,
            "singleton_clusters": singleton_clusters,
        })

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(out_df)
    print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
