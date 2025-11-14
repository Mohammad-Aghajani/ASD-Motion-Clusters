import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

RESULTS_ROOT = Path('results')
SOCIAL_SUFFIX = '_social_scores_summary.csv'
CLUSTER_SIZE_PREFIX = 'cluster_sizes_'
STABILITY_PREFIX = 'stability_ari_'
HIT_THRESHOLD = 0.05
DEFAULT_JSON_NAME = '_summary_report.json'
DEFAULT_TEXT_NAME = '_summary_report.txt'


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_social_summary(path: Path) -> Dict:
    hits: List[Dict] = []
    total = 0
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            score = row.get('Score', '')
            k = row.get('k', '')
            p_val = _parse_float(row.get('ANOVA_p'))
            corr = _parse_float(row.get('Correlation'))
            sig_anova = (row.get('ANOVA_Sig') == '*') or (not math.isnan(p_val) and p_val <= HIT_THRESHOLD)
            sig_corr = row.get('Corr_Sig') == '*'
            if sig_anova or sig_corr:
                hits.append({
                    'score': score,
                    'k': k,
                    'anova_p': p_val,
                    'anova_sig': row.get('ANOVA_Sig', ''),
                    'corr': corr,
                    'corr_sig': row.get('Corr_Sig', ''),
                })
    emotion = path.parts[1]
    roi = path.parts[4]
    method = path.stem.replace(SOCIAL_SUFFIX.replace('.csv', ''), '')
    return {
        'emotion': emotion,
        'roi': roi,
        'method': method,
        'file': str(path),
        'total_rows': total,
        'hits': hits,
    }


def _summarize_social(root: Path) -> List[Dict]:
    summaries = []
    for path in sorted(root.glob(f"*/statistical/social_scores/*/*{SOCIAL_SUFFIX}")):
        summary = _parse_social_summary(path)
        if summary['hits']:
            summaries.append(summary)
    return summaries


def _summarize_cluster_sizes(path: Path) -> Dict:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_participants: Dict[str, int] = defaultdict(int)
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row.get('k')
            cluster = row.get('cluster')
            size = int(row.get('size', 0))
            counts[k][cluster] = size
            total_participants[k] += size
    emotion = path.parts[1]
    metric_roi = path.parts[3]
    if '_' in metric_roi:
        metric, roi = metric_roi.split('_', 1)
    else:
        metric, roi = metric_roi, ''
    method = path.stem.replace(CLUSTER_SIZE_PREFIX, '')

    stability_path = path.with_name(path.name.replace(CLUSTER_SIZE_PREFIX, STABILITY_PREFIX))
    stability = []
    if stability_path.exists():
        with stability_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stability.append({
                    'k1': row.get('k1'),
                    'k2': row.get('k2'),
                    'ari': _parse_float(row.get('ARI')),
                })

    clusters = []
    for k, cluster_sizes in sorted(counts.items(), key=lambda x: int(x[0])):
        size_list = [cluster_sizes[c] for c in sorted(cluster_sizes, key=lambda x: int(x))]
        clusters.append({
            'k': k,
            'sizes': size_list,
            'min_size': min(size_list) if size_list else 0,
            'max_size': max(size_list) if size_list else 0,
            'n_participants': total_participants[k],
        })

    return {
        'emotion': emotion,
        'metric': metric,
        'roi': roi,
        'method': method,
        'file': str(path),
        'clusters': clusters,
        'stability': stability,
    }


def _summarize_clustering(root: Path) -> List[Dict]:
    summaries = []
    for path in sorted(root.glob(f"*/clustering/*/{CLUSTER_SIZE_PREFIX}*.csv")):
        summaries.append(_summarize_cluster_sizes(path))
    return summaries


def _format_text(report: Dict) -> str:
    lines: List[str] = []
    lines.append('')
    lines.append('=== Social-score hits (ANOVA p <= 0.05 or Corr_Sig == *) ===')
    social_hits = report['social_hits']
    if not social_hits:
        lines.append('None')
    else:
        for item in social_hits:
            lines.append(f"\n{item['emotion']} / {item['roi']} / {item['method']}")
            lines.append(f"  Source: {item['file']}")
            for hit in item['hits']:
                p_val = hit['anova_p']
                p_str = f"p={p_val:.3f}" if not math.isnan(p_val) else 'p=NA'
                corr = hit['corr']
                corr_str = f"corr={corr:.3f}" if not math.isnan(corr) else 'corr=NA'
                flags: List[str] = []
                if hit['anova_sig'] == '*':
                    flags.append('ANOVA*')
                elif not math.isnan(p_val) and p_val <= HIT_THRESHOLD:
                    flags.append('ANOVA<=0.05')
                if hit['corr_sig'] == '*':
                    flags.append('Corr*')
                flag_str = f" {', '.join(flags)}" if flags else ''
                lines.append(f"    k={hit['k']} score={hit['score']} {p_str} {corr_str}{flag_str}")
    lines.append('\n=== Clustering summaries (cluster_sizes_*.csv) ===')
    clustering = report['clustering']
    if not clustering:
        lines.append('None')
    else:
        for item in clustering:
            lines.append(f"\n{item['emotion']} / {item['metric']} / {item['roi']} / {item['method']}")
            for cluster in item['clusters']:
                sizes = ', '.join(str(s) for s in cluster['sizes'])
                note = ' (contains very small cluster)' if cluster['min_size'] <= 2 else ''
                lines.append(f"  k={cluster['k']}: sizes [{sizes}] n={cluster['n_participants']}{note}")
            if item['stability']:
                parts: List[str] = []
                for row in item['stability']:
                    ari = row.get('ari')
                    if ari is None or math.isnan(ari):
                        continue
                    parts.append(f"k{row['k1']}-k{row['k2']} ARI={ari:.3f}")
                if parts:
                    lines.append(f"  Stability ARI: {', '.join(parts)}")
    return '\n'.join(lines)


def _write_if_requested(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Summarize social-score significance and clustering structure.')
    parser.add_argument('--json', dest='as_json', action='store_true', help='Output raw JSON to stdout')
    parser.add_argument('--root', default='results', help='Results directory (default: results)')
    parser.add_argument('--json-out', help='Write JSON report to this path')
    parser.add_argument('--text-out', help='Write human-readable report to this path')
    parser.add_argument('--no-save', action='store_true', help='Do not save reports to disk')
    args = parser.parse_args()

    root = Path(args.root)
    social_hits = _summarize_social(root)
    clustering = _summarize_clustering(root)

    report = {
        'social_hits': social_hits,
        'social_total': len(list(root.glob(f"*/statistical/social_scores/*/*{SOCIAL_SUFFIX}"))),
        'social_hit_count': len(social_hits),
        'clustering': clustering,
        'clustering_total': len(clustering),
    }

    text_report = _format_text(report)
    json_report = json.dumps(report, indent=2)

    if args.as_json:
        print(json_report)
    else:
        print(text_report)

    if not args.no_save:
        json_out = Path(args.json_out) if args.json_out else root / DEFAULT_JSON_NAME
        text_out = Path(args.text_out) if args.text_out else root / DEFAULT_TEXT_NAME
        _write_if_requested(json_out, json_report)
        _write_if_requested(text_out, text_report + '\n')
        print(f"\nReports saved to:\n  JSON: {json_out}\n  Text: {text_out}")
    else:
        if args.json_out:
            _write_if_requested(Path(args.json_out), json_report)
        if args.text_out:
            _write_if_requested(Path(args.text_out), text_report + '\n')


if __name__ == '__main__':
    main()
