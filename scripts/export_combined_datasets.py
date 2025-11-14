import pandas as pd
from pathlib import Path
import os

RESULTS_DIR = Path('results')
EXPORT_DIR = RESULTS_DIR / 'exports'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


EMOTION_CANON = {
    'happy': 'happiness',
    'sad': 'sadness',
    'sad - part (2)': 'sadness',
    'sad - part 1': 'sadness',
    'admiraton': 'admiration',
    'guilit': 'guilt'
}

def canonical_emotion(name: str) -> str:
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    return EMOTION_CANON.get(key, key)
# ---------- Helper: load or build combined membership ----------
combined_membership_path = RESULTS_DIR / 'cluster_memberships_all.csv'
if combined_membership_path.exists():
    memberships = pd.read_csv(combined_membership_path)
else:
    parts = []
    for csv_path in RESULTS_DIR.rglob('cluster_membership_*.csv'):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        if 'Participant' not in df.columns:
            continue
        if 'k' not in df.columns:
            if 'K' in df.columns:
                df = df.rename(columns={'K': 'k'})
            else:
                continue
        df['Emotion'] = csv_path.parts[csv_path.parts.index('results') + 1]
        parts.append(df)
    if not parts:
        raise SystemExit('No cluster membership files found.')
    memberships = pd.concat(parts, ignore_index=True)
    memberships.to_csv(combined_membership_path, index=False)

# ---------- Build participant-level dataset with features ----------
features_path = RESULTS_DIR / 'aggregated' / 'aggregated_features.csv'
if not features_path.exists():
    raise SystemExit('aggregated_features.csv not found.')
features = pd.read_csv(features_path)

# normalise actor/emotion in features
def _clean_actor(val):
    try:
        return int(val)
    except Exception:
        try:
            return int(str(val).strip())
        except Exception:
            return None
features['actor_clean'] = features['actor'].apply(_clean_actor)
features['emotion_clean'] = features['emotion'].astype(str).apply(canonical_emotion)

# prepare membership identifiers
m = memberships.copy()
m['Participant'] = m['Participant'].astype(str)
m['actor_clean'] = m['Participant'].str.split('_').str[0].astype(int)
m['emotion_clean'] = m['Participant'].str.split('_').str[1].apply(canonical_emotion)

participant_dataset = m.merge(
    features.drop(columns=['actor', 'emotion']),
    on=['actor_clean', 'emotion_clean'],
    how='left',
    suffixes=('', '_feature')
)
participant_dataset.rename(columns={'actor_clean': 'actor', 'emotion_clean': 'emotion'}, inplace=True)
participant_dataset.to_csv(EXPORT_DIR / 'participant_clusters_with_features.csv', index=False)

# ---------- Cluster size summaries ----------
cluster_rows = []
for csv_path in RESULTS_DIR.rglob('cluster_sizes_*.csv'):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    parts = csv_path.parts
    emotion = canonical_emotion(parts[parts.index('results') + 1])
    metric_roi = parts[parts.index('clustering') + 1]
    if '_' in metric_roi:
        metric, roi = metric_roi.split('_', 1)
    else:
        metric, roi = metric_roi, ''
    method = csv_path.stem.replace('cluster_sizes_', '')
    df = df.copy()
    df['emotion'] = emotion
    df['metric'] = metric
    df['roi'] = roi
    df['method'] = method
    cluster_rows.append(df)
if cluster_rows:
    cluster_sizes = pd.concat(cluster_rows, ignore_index=True)
    cluster_sizes.to_csv(EXPORT_DIR / 'cluster_size_summary.csv', index=False)

# ---------- Social score summaries ----------
social_rows = []
for csv_path in RESULTS_DIR.rglob('*_social_scores_summary.csv'):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    parts = csv_path.parts
    emotion = canonical_emotion(parts[parts.index('results') + 1])
    try:
        roi = parts[parts.index('social_scores') + 1]
    except ValueError:
        roi = ''
    method = csv_path.stem.replace('_social_scores_summary', '')
    df = df.copy()
    df['emotion'] = emotion
    df['roi'] = roi
    df['method'] = method
    social_rows.append(df)
if social_rows:
    social_summary = pd.concat(social_rows, ignore_index=True)
    social_summary.to_csv(EXPORT_DIR / 'social_score_summary.csv', index=False)

print('Exports written to', EXPORT_DIR)
