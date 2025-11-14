from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import adjusted_rand_score

RESULTS_DIR = Path('results')
EXPORT_DIR = RESULTS_DIR / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------- #
# Helper utilities
# ---------------------------------------------------------------------------- #

def load_csv(name: str) -> pd.DataFrame:
    path = EXPORT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_fig_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------- #
# Multiple comparison correction utilities
# ---------------------------------------------------------------------------- #

def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted q-values."""
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty_like(p)
    q[order] = (p[order] * n) / ranks

    # Enforce monotonicity from largest to smallest p-value
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0.0, 1.0)


def apply_multiple_comparison_correction() -> None:
    """Augment social_score_summary with FDR-adjusted columns and exports."""
    path = EXPORT_DIR / 'social_score_summary.csv'
    if not path.exists():
        return

    social = pd.read_csv(path)
    mask = social['ANOVA_p'].notna()
    qvals = benjamini_hochberg(social.loc[mask, 'ANOVA_p'])
    social.loc[mask, 'ANOVA_q'] = qvals
    social['ANOVA_FDR_Sig'] = ''
    social.loc[mask & (social['ANOVA_q'] <= 0.05), 'ANOVA_FDR_Sig'] = '*'

    social.to_csv(path, index=False)
    social.to_csv(EXPORT_DIR / 'social_score_summary_fdr.csv', index=False)

    summary_rows: List[Dict] = []
    for (method, k), group in social.groupby(['method', 'k']):
        summary_rows.append({
            'method': method,
            'k': k,
            'anova_hits_raw': int((group['ANOVA_Sig'] == '*').sum()),
            'anova_hits_fdr': int((group['ANOVA_FDR_Sig'] == '*').sum()),
            'corr_hits': int((group['Corr_Sig'] == '*').sum())
        })
    pd.DataFrame(summary_rows).to_csv(
        EXPORT_DIR / 'social_score_hits_summary.csv',
        index=False
    )


# ---------------------------------------------------------------------------- #
# 1. Method comparison: social hits + singleton counts
# ---------------------------------------------------------------------------- #

def method_comparison() -> None:
    cluster = load_csv('cluster_size_summary.csv')
    social = load_csv('social_score_summary.csv')

    methods = sorted(cluster['method'].unique())
    ks = sorted(cluster['k'].unique())

    rows: List[Dict] = []
    for method in methods:
        for k in ks:
            sub_c = cluster[(cluster['method'] == method) & (cluster['k'] == k)]
            sub_s = social[(social['method'] == method) & (social['k'] == k)]
            singleton_count = int((sub_c['size'] == 1).sum())
            corr_mask = sub_s['Corr_Sig'].fillna('') == '*'
            hit_mask = (sub_s['ANOVA_Sig'] == '*') | corr_mask
            if 'ANOVA_FDR_Sig' in sub_s.columns:
                fdr_mask = (sub_s['ANOVA_FDR_Sig'] == '*') | corr_mask
            else:
                fdr_mask = hit_mask
            social_hits = int(hit_mask.sum())
            rows.append({
                'method': method,
                'k': k,
                'social_hits': social_hits,
                'social_hits_fdr': int(fdr_mask.sum()),
                'singleton_clusters': singleton_count,
                'total_clusters': len(sub_c),
            })

    summary = pd.DataFrame(rows)
    summary_path = EXPORT_DIR / 'method_comparison_summary.csv'
    summary.to_csv(summary_path, index=False)

    # Plot
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    width = 0.25
    x = np.arange(len(ks))
    method_to_offset = {m: i - (len(methods) - 1) / 2 for i, m in enumerate(methods)}

    for method in methods:
        method_data = summary[summary['method'] == method]
        offsets = method_to_offset[method]
        axes[0].bar(x + offsets * width, method_data['social_hits'],
                    width=width, label=method)
        axes[1].bar(x + offsets * width, method_data['singleton_clusters'],
                    width=width, label=method)

    for ax, ylabel, title in zip(
        axes,
        ['# significant social-score findings', '# singleton clusters'],
        ['Social-score hits by distance method', 'Singleton clusters by method']
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel('k')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend(title='Method')

    fig.tight_layout()
    fig_path = EXPORT_DIR / 'method_comparison.png'
    ensure_fig_dir(fig_path)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# 2. ROI summary of social hits
# ---------------------------------------------------------------------------- #

def roi_social_hits() -> None:
    social = load_csv('social_score_summary.csv')
    hits = social[(social['ANOVA_Sig'] == '*') | (social['Corr_Sig'] == '*')]
    pivot = hits.groupby(['method', 'roi'])['Score'].count().reset_index(name='hit_count')
    pivot_path = EXPORT_DIR / 'roi_social_hits.csv'
    pivot.to_csv(pivot_path, index=False)

    heatmap = pivot.pivot(index='roi', columns='method', values='hit_count').fillna(0)
    sns.set_theme(style='white')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(heatmap, annot=True, fmt='.0f', cmap='Blues', ax=ax)
    ax.set_title('Significant social-score findings per ROI / method')
    fig.tight_layout()
    fig_path = EXPORT_DIR / 'roi_social_hits_heatmap.png'
    ensure_fig_dir(fig_path)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# 3. Singleton participant list
# ---------------------------------------------------------------------------- #

def singleton_participants() -> None:
    cluster = load_csv('cluster_size_summary.csv')
    membership = pd.read_csv(RESULTS_DIR / 'cluster_memberships_all.csv')
    singleton_info = cluster[cluster['size'] == 1]
    merged = singleton_info.merge(
        membership,
        left_on=['emotion', 'metric', 'roi', 'method', 'k', 'cluster'],
        right_on=['Metric', 'ROI', 'Method', 'Method', 'k', 'Cluster'],
        how='left'
    )
    # Fix duplicated columns from merge (Method appears twice)
    merged = merged.rename(columns={'Participant': 'participant'})
    keep_cols = ['participant', 'emotion', 'metric', 'roi', 'method', 'k', 'cluster']
    merged = merged[keep_cols].dropna(subset=['participant']).sort_values(['method', 'emotion', 'k'])
    merged.to_csv(EXPORT_DIR / 'singleton_participants.csv', index=False)


# ---------------------------------------------------------------------------- #
# 4. Emotion-wise social score deltas
# ---------------------------------------------------------------------------- #

def emotion_score_deltas() -> None:
    cos_membership = pd.read_csv(RESULTS_DIR / 'cluster_memberships_all.csv')
    cos_membership = cos_membership[cos_membership['Method'] == 'Cosine']
    features = pd.read_csv(EXPORT_DIR / 'participant_clusters_with_features_clean.csv')
    merged = cos_membership.merge(
        features,
        left_on=['Participant', 'Metric', 'ROI', 'Method', 'k', 'Cluster'],
        right_on=['Participant', 'Metric', 'ROI', 'Method', 'k', 'Cluster'],
        how='left'
    )
    social_cols = ['AQ', 'AQ_SocialSkills', 'AQ_AttnSwitch', 'AQ_AttntoDetail',
                   'AQ_Communication', 'AQ_Imagination', 'RMET', 'Alexithymia',
                   'Cambridge_Behavior', 'Cambridge_Friendship',
                   'PerceivedSocialSupport', 'TIPI_EXT', 'TIPI_AGR', 'TIPI_CON',
                   'TIPI_NEU', 'TIPI_OM']

    rows: List[Dict] = []
    for (emotion, roi, metric, k), group in merged.groupby(['emotion', 'ROI', 'Metric', 'k']):
        emotion_means = group.groupby('emotion')[social_cols].mean().iloc[0]
        for cluster_id, cluster_group in group.groupby('Cluster'):
            cluster_means = cluster_group[social_cols].mean()
            deltas = cluster_means - emotion_means
            for score, value in deltas.items():
                rows.append({
                    'emotion': emotion,
                    'roi': roi,
                    'metric': metric,
                    'k': k,
                    'cluster': cluster_id,
                    'score': score,
                    'delta': value
                })
    df = pd.DataFrame(rows)
    df.to_csv(EXPORT_DIR / 'emotion_score_deltas.csv', index=False)


# ---------------------------------------------------------------------------- #
# 5. Cross-emotion stability (ARI)
# ---------------------------------------------------------------------------- #

def cross_emotion_stability() -> None:
    membership = pd.read_csv(RESULTS_DIR / 'cluster_memberships_all.csv')
    membership['actor'] = membership['Participant'].str.split('_', n=1).str[0]
    membership['emotion'] = membership['Participant'].str.split('_', n=1).str[1]

    rows: List[Dict] = []
    group_cols = ['Method', 'Metric', 'ROI', 'k']
    for (method, metric, roi, k), group in membership.groupby(group_cols):
        emotions = sorted(group['emotion'].unique())
        for idx, emo_i in enumerate(emotions):
            data_i = group[group['emotion'] == emo_i][['actor', 'Cluster']].dropna()
            data_i = data_i.drop_duplicates(subset='actor').set_index('actor')
            for emo_j in emotions[idx + 1:]:
                data_j = group[group['emotion'] == emo_j][['actor', 'Cluster']].dropna()
                data_j = data_j.drop_duplicates(subset='actor').set_index('actor')
                common = data_i.index.intersection(data_j.index)
                if len(common) < 2:
                    continue
                ari = adjusted_rand_score(
                    data_i.loc[common, 'Cluster'],
                    data_j.loc[common, 'Cluster']
                )
                rows.append({
                    'Method': method,
                    'Metric': metric,
                    'ROI': roi,
                    'k': k,
                    'emotion_i': emo_i,
                    'emotion_j': emo_j,
                    'n_common': len(common),
                    'ARI': ari,
                })

    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(EXPORT_DIR / 'cross_emotion_ari.csv', index=False)

    focus_config = {
        'Method': 'Cosine',
        'Metric': 'acceleration',
        'ROI': 'head',
        'k': 6,
    }
    mask = (
        (df['Method'] == focus_config['Method']) &
        (df['Metric'] == focus_config['Metric']) &
        (df['ROI'] == focus_config['ROI']) &
        (df['k'] == focus_config['k'])
    )
    subset = df[mask]
    if subset.empty:
        return

    emotions = sorted(set(subset['emotion_i']).union(subset['emotion_j']))
    matrix = pd.DataFrame(
        np.eye(len(emotions)),
        index=emotions,
        columns=emotions,
        dtype=float
    )
    for _, row in subset.iterrows():
        matrix.loc[row['emotion_i'], row['emotion_j']] = row['ARI']
        matrix.loc[row['emotion_j'], row['emotion_i']] = row['ARI']

    sns.set_theme(style='white')
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        vmin=0,
        vmax=1,
        cmap='Purples',
        annot_kws={'fontsize': 7},
        ax=ax,
    )
    ax.tick_params(labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Cross-emotion ARI (Cosine, head acceleration, k=6)')
    fig.tight_layout()
    fig_path = EXPORT_DIR / 'cross_emotion_ari_heatmap.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# 6. Cluster centroid curves (example: admiration, head, acceleration, k=6)
# ---------------------------------------------------------------------------- #

def resample_series(series: Iterable[float], new_length: int = 100) -> np.ndarray:
    arr = np.asarray(list(series), dtype=float)
    if arr.size == 0:
        return np.zeros(new_length)
    x_old = np.linspace(0, 1, num=arr.size)
    x_new = np.linspace(0, 1, num=new_length)
    return np.interp(x_new, x_old, arr)


def cluster_centroid_plot() -> None:
    membership = pd.read_csv(RESULTS_DIR / 'cluster_memberships_all.csv')
    membership['emotion'] = membership['Participant'].str.split('_', n=1).str[1]
    config = {
        'emotion': 'admiration',
        'Metric': 'acceleration',
        'ROI': 'head',
        'Method': 'Cosine',
        'k': 6,
    }
    subset = membership[
        (membership['emotion'] == config['emotion']) &
        (membership['Metric'] == config['Metric']) &
        (membership['ROI'] == config['ROI']) &
        (membership['Method'] == config['Method']) &
        (membership['k'] == config['k'])
    ]

    ts_path = RESULTS_DIR / config['emotion'] / 'time_series' / 'time_series_data.json'
    if not ts_path.exists():
        return
    with ts_path.open() as f:
        ts_data = json.load(f)[config['Metric']][config['ROI']]

    clusters: Dict[int, List[np.ndarray]] = {cid: [] for cid in sorted(subset['Cluster'].unique())}
    for _, row in subset.iterrows():
        pid = row['Participant']
        cluster_id = row['Cluster']
        series = ts_data.get(pid)
        if series is None or len(series) < 2:
            continue
        clusters[cluster_id].append(resample_series(series))

    sns.set_theme(style='white')
    fig, ax = plt.subplots(figsize=(8, 4))
    for cluster_id, seqs in clusters.items():
        if not seqs:
            continue
        arr = np.vstack(seqs)
        mean_curve = arr.mean(axis=0)
        ax.plot(mean_curve, label=f'Cluster {cluster_id} (n={len(seqs)})')
    ax.set_title('Mean head acceleration curves (admiration, Cosine k=6)')
    ax.set_xlabel('Normalised time')
    ax.set_ylabel('Acceleration (arbitrary units)')
    ax.legend()
    fig.tight_layout()
    fig_path = EXPORT_DIR / 'admiration_head_acceleration_centroids.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# 7. Cluster size distribution plot
# ---------------------------------------------------------------------------- #

def cluster_size_distribution() -> None:
    cluster = load_csv('cluster_size_summary.csv')
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=cluster, x='method', y='size', ax=ax)
    ax.set_title('Cluster size distribution by method')
    ax.set_xlabel('Method')
    ax.set_ylabel('Cluster size')
    fig.tight_layout()
    fig_path = EXPORT_DIR / 'cluster_size_distribution.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# 8. Feature variance heatmap (Cosine, head, k=6)
# ---------------------------------------------------------------------------- #

def feature_variation_heatmap() -> None:
    data = pd.read_csv(EXPORT_DIR / 'participant_clusters_with_features_clean.csv')
    subset = data[(data['Method'] == 'Cosine') & (data['ROI'] == 'head') & (data['Metric'] == 'acceleration') & (data['k'] == 6)]
    feature_cols = [c for c in subset.columns if c not in ['Participant', 'Metric', 'ROI', 'Method', 'k', 'Cluster', 'actor', 'emotion', 'Sex', 'Age']]
    z = subset[feature_cols].apply(zscore, nan_policy='omit')
    subset_z = subset[['Cluster']].join(z)
    cluster_means = subset_z.groupby('Cluster').mean()
    variances = cluster_means.var(axis=0).sort_values(ascending=False)
    preferred = [
        'Head_v_max',
        'TIPI_OM',
        'Head_a_max',
        'Head_v_ent',
        'Head_a_ent',
        'Head_v_mean'
    ]
    available = [c for c in preferred if c in variances.index]
    remaining = [c for c in variances.index if c not in available]
    top_cols = available + remaining[: max(0, 8 - len(available))]
    sns.set_theme(style='white')
    fig, ax = plt.subplots(figsize=(8, 4))
    display = cluster_means[top_cols].rename(columns={'TIPI_OM': 'TIPI_O'})
    sns.heatmap(display, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Top feature deviations (Cosine, head, acceleration, k=6)')
    fig.tight_layout()
    fig_path = EXPORT_DIR / 'cosine_head_k6_feature_heatmap.png'
    fig.savefig(fig_path, dpi=300)
    fig.savefig(EXPORT_DIR / 'cosine_head_k6_feature_heatmap.pdf', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Main entry point
# ---------------------------------------------------------------------------- #

def main() -> None:
    apply_multiple_comparison_correction()
    method_comparison()
    roi_social_hits()
    singleton_participants()
    emotion_score_deltas()
    cross_emotion_stability()
    cluster_centroid_plot()
    cluster_size_distribution()
    feature_variation_heatmap()


if __name__ == '__main__':
    main()
