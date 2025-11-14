import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def parse_participant_emotion(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Expected <actor>_<emotion>.npy naming, got: {path.name}")
    actor, emotion = stem.split("_", 1)
    return actor, emotion


def summarise_embedding(arr: np.ndarray) -> Dict[str, float]:
    if arr.ndim == 1:
        arr = arr[None, :]
    mean = arr.mean(axis=0)
    max_ = arr.max(axis=0)
    std = arr.std(axis=0)

    summary = {}
    for idx, value in enumerate(mean):
        summary[f"stgcn_mean_{idx:03d}"] = float(value)
    for idx, value in enumerate(max_):
        summary[f"stgcn_max_{idx:03d}"] = float(value)
    for idx, value in enumerate(std):
        summary[f"stgcn_std_{idx:03d}"] = float(value)
    summary["stgcn_frames"] = int(arr.shape[0])
    return summary


def load_embeddings(paths: Iterable[Path]) -> pd.DataFrame:
    records = []
    for path in sorted(paths):
        actor, emotion = parse_participant_emotion(path)
        embedding = np.load(path)
        summary = summarise_embedding(embedding)
        summary.update(
            {
                "actor": actor,
                "emotion": emotion,
                "participant": f"{actor}_{emotion}",
                "source_file": str(path),
            }
        )
        records.append(summary)
    if not records:
        raise RuntimeError("No .npy files found to summarise.")
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ST-GCN embeddings into CSV summaries.")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help="Directory containing <actor>_<emotion>.npy embedding files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/stgcn_features_summary.csv"),
        help="Destination CSV for aggregated features.",
    )
    args = parser.parse_args()

    if not args.embedding_dir.exists():
        raise FileNotFoundError(args.embedding_dir)

    npy_files = list(args.embedding_dir.glob("*.npy"))
    df = load_embeddings(npy_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
