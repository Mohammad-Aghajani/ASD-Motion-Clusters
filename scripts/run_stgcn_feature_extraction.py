"""Run the pretrained ST-GCN model on exported pose sequences.

Example usage:

python scripts/run_stgcn_feature_extraction.py \
  --input-dir workdir/stgcn_input \
  --weights external/st-gcn/checkpoints/stgcn.kinetics-pretrained.pt \
  --output-dir workdir/stgcn_embeddings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def load_model(weights: Path, device: torch.device):
    repo_root = Path(__file__).resolve().parents[1] / "external" / "st-gcn"
    sys.path.insert(0, str(repo_root))

    from net.st_gcn import Model  # type: ignore

    model = Model(
        in_channels=3,
        num_class=400,
        graph_args={"layout": "openpose", "strategy": "spatial"},
        edge_importance_weighting=True,
    )

    state = torch.load(weights, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("Loaded with missing keys:", missing)
    if unexpected:
        print("Loaded with unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.weights, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_dir.glob("*.npy"))
    if not files:
        raise SystemExit("No .npy pose tensors found in input directory")

    with torch.no_grad():
        for path in files:
            data = np.load(path)  # (C, T, V, M)
            tensor = torch.from_numpy(data).unsqueeze(0).to(device)
            _, feature = model.extract_feature(tensor)
            embedding = feature.mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()
            np.save(args.output_dir / path.name, embedding)

    print(f"Processed {len(files)} clips -> {args.output_dir}")


if __name__ == "__main__":
    main()
