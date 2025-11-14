"""
Run clustering and social-score analysis on ST-GCN embeddings to mirror the
handcrafted feature pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SOCIAL_COLS: List[str] = [
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

METRIC_NAME = "embedding"
ROI_NAME = "stgcn"
METHOD_NAME = "Cosine"


def merge_features(stgcn_csv: Path, agg_csv: Path) -> pd.DataFrame:
    stgcn = pd.read_csv(stgcn_csv)
    agg = pd.read_csv(agg_csv)
    agg["participant"] = agg["actor"].astype(str) + "_" + agg["emotion"]
    merged = stgcn.merge(
        agg[["participant", "emotion"] + SOCIAL_COLS], on="participant", how="inner"
    )
    if "emotion_x" in merged.columns:
        merged = merged.drop(columns=["emotion_x"]).rename(columns={"emotion_y": "emotion"})
    elif "emotion" not in merged.columns:
        merged = merged.rename(columns={"emotion_y": "emotion"})
    if merged.empty:
        raise RuntimeError("No overlapping participants between ST-GCN and demographics.")
    return merged


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GCN clustering analysis.")
    parser.add_argument(
        "--stgcn", type=Path, default=Path("exports/stgcn_features_summary.csv")
    )
    parser.add_argument(
        "--aggregated", type=Path, default=Path("exports/aggregated_features_clean.csv")
    )
    parser.add_argument(
        "--cluster-out",
        type=Path,
        default=Path("exports/stgcn_cluster_summary.csv"),
    )
    parser.add_argument(
        "--social-out",
        type=Path,
        default=Path("exports/stgcn_social_score_summary.csv"),
    )
    parser.add_argument(
        "--membership-out",
        type=Path,
        default=Path("exports/stgcn_cluster_membership.csv"),
    )
    parser.add_argument("--k", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = merge_features(args.stgcn, args.aggregated)
    feature_cols = [c for c in df.columns if c.startswith("stgcn_")]

    cluster_rows = []
    social_rows = []
    membership_rows = []

    for emotion, group in pd.concat(
        [df, df.assign(emotion="ALL")], axis=0
    ).groupby("emotion"):
        for k in args.k:
            if len(group) <= k or k < 2:
                cluster_rows.append(
                    {
                        "emotion": emotion,
                        "k": k,
                        "metric": METRIC_NAME,
                        "roi": ROI_NAME,
                        "method": METHOD_NAME,
                        "silhouette": float("nan"),
                        "singleton_clusters": None,
                        "min_cluster_size": None,
                        "max_cluster_size": None,
                        "n_participants": len(group),
                        "note": "insufficient members",
                    }
                )
                continue

            kmeans = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
            labels = kmeans.fit_predict(group[feature_cols].values)

            unique, counts = np.unique(labels, return_counts=True)
            singleton_clusters = int(np.sum(counts == 1))

            cluster_rows.append(
                    {
                        "emotion": emotion,
                        "k": k,
                        "metric": METRIC_NAME,
                        "roi": ROI_NAME,
                        "method": METHOD_NAME,
                        "silhouette": compute_silhouette(
                            group[feature_cols].values, labels
                        ),
                    "singleton_clusters": singleton_clusters,
                    "min_cluster_size": int(counts.min()),
                    "max_cluster_size": int(counts.max()),
                    "n_participants": len(group),
                    "note": "",
                }
            )

            membership_rows.extend(
                {
                    "Participant": pid,
                    "Metric": METRIC_NAME,
                    "ROI": ROI_NAME,
                    "Method": METHOD_NAME,
                    "emotion": emotion,
                    "k": k,
                    "Cluster": int(label),
                }
                for pid, label in zip(group["participant"], labels)
            )

            for score in SOCIAL_COLS:
                values = group[[score]].copy()
                valid_mask = values[score].notna()
                if valid_mask.sum() < 4:
                    continue
                groups = [group.loc[(labels == lab) & valid_mask, score] for lab in np.unique(labels)]
                if any(len(g) < 2 for g in groups):
                    continue
                f_stat, p_val = f_oneway(*groups)
                corr = group.loc[valid_mask, score].corr(pd.Series(labels, index=group.index)[valid_mask])
                social_rows.append(
                    {
                        "Score": score,
                        "k": k,
                        "ANOVA_F": float(f_stat),
                        "ANOVA_p": float(p_val),
                        "ANOVA_Sig": "*" if p_val <= 0.05 else "",
                        "Correlation": float(corr) if pd.notna(corr) else np.nan,
                        "Corr_Sig": "*" if pd.notna(corr) and abs(corr) > 0.3 else "",
                        "emotion": emotion,
                        "roi": ROI_NAME,
                        "method": METHOD_NAME,
                    }
                )

    cluster_df = pd.DataFrame(cluster_rows)
    social_df = pd.DataFrame(social_rows)
    membership_df = pd.DataFrame(membership_rows)

    args.cluster_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(args.cluster_out, index=False)
    social_df.to_csv(args.social_out, index=False)
    membership_df.to_csv(args.membership_out, index=False)

    print("Cluster summary:")
    print(cluster_df.head())
    print(f"\nSaved cluster summary to {args.cluster_out}")
    print(f"Saved social-score summary to {args.social_out}")
    print(f"Saved memberships to {args.membership_out}")


if __name__ == "__main__":
    main()
