"""
Convert the raw pose text exports into NumPy tensors suitable for ST-GCN style
processing. Each output file stores an array with shape (T, V, C) where
T = number of frames, V = 25 joints, C = 2 (x, y). A single actor is assumed
per clip; the second person reported in the text headers is ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


# ST-GCN's default OpenPose layout uses 18 joints (excluding hand/foot extras).
JOINTS = 18


def parse_pose_txt(path: Path) -> np.ndarray:
    """Return a (C, T, V, M) tensor from a pose text file."""
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) <= 2:
        raise ValueError(f"{path} does not contain frame data.")

    frames = []
    for line in lines[2:]:
        values = [float(x) for x in line.split(",") if x.strip()]
        if len(values) < JOINTS * 2:
            raise ValueError(f"{path}: expected {JOINTS*2} values, found {len(values)}")
        coords = np.array(values[: JOINTS * 2], dtype=np.float32).reshape(JOINTS, 2)
        conf = np.ones((JOINTS, 1), dtype=np.float32)
        coords = np.concatenate([coords, conf], axis=1)  # (V, 3)
        frames.append(coords)

    arr = np.stack(frames, axis=0)  # (T, V, C)
    arr = np.transpose(arr, (2, 0, 1))  # (C=3, T, V)
    arr = arr[..., None]  # (C, T, V, M=1)
    return arr


def discover_pose_files(root: Path) -> Iterable[Tuple[Path, str, str]]:
    """Yield (path, actor_id, emotion) tuples for every pose file."""
    for actor_dir in sorted(root.glob("Actor *")):
        actor_id = actor_dir.name.split()[-1]
        for pose_file in actor_dir.glob("*.txt"):
            stem = pose_file.stem
            if "_" not in stem:
                continue
            _, emotion_raw = stem.split("_", 1)
            emotion = emotion_raw.replace(" ", "").lower()
            yield pose_file, actor_id, emotion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw pose text files into NumPy tensors."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains 'Actor <id>' folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("workdir/stgcn_input"),
        help="Directory to store <actor>_<emotion>.npy tensors.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for pose_file, actor_id, emotion in discover_pose_files(args.data_root):
        try:
            tensor = parse_pose_txt(pose_file)
        except ValueError as exc:
            print(f"Skipping {pose_file}: {exc}")
            continue

        output_name = f"{actor_id}_{emotion}.npy"
        output_path = args.output_dir / output_name
        np.save(output_path, tensor)
        count += 1

    print(f"Wrote {count} tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
