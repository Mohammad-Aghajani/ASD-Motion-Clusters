# ASD-Motion-Clusters

End-to-end pipeline for clustering emotion-specific body motion and linking kinematic signatures to social and psychometric measures in autism.

Official implementation of the CVPR 2026 paper  
**"Quantifying Emotion-Specific Motion Signatures and Social Correlates in Autism Spectrum Disorder."**

The repository reproduces the full pipeline used in the manuscript:

- normalize Kinect-style pose directories and align them with AQ/RMET/TIPI spreadsheets;
- extract velocity and acceleration features for head, left/right hands, and lower extremities with deterministic 15-frame windows;
- build Cosine, Euclidean, DTW, and soft-DTW distance matrices and run k-means for k in {4,6,8};
- compute stability diagnostics (cluster size stats, cross-emotion ARI, singleton rates);
- run ANOVAs and Spearman correlations against psychometric scores with Benjamini-Hochberg correction;
- optionally evaluate the ST-GCN baseline using the provided scripts.

The goal is to lower the barrier for ASD-focused affective computing by releasing all preprocessing code, clustering scripts, and export schemas referenced in the paper.

---

## Installation

```bash
git clone https://github.com/Mohammad-Aghajani/ASD-Motion-Clusters.git
cd ASD-Motion-Clusters
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

The requirements cover the handcrafted feature pipeline plus the optional ST-GCN baseline (PyTorch 2.1 + CUDA 11.8).

---

## Repository Layout

```
ASD-Motion-Clusters/
├── configs/                 # YAML presets used by the paper
├── data/README.md           # Instructions for provisioning raw pose/social data
├── exports/                 # Sample CSV schemas for aggregated features & statistics
├── figs/                    # Figures referenced in the manuscript
├── scripts/                 # Stage-specific utilities (ingest, exports, ST-GCN helpers)
├── src/asd_motion_clusters/ # Core implementation (kbc_cleaned + utils)
├── kbc_cleaned.py           # Compatibility shim -> src/asd_motion_clusters/kbc_cleaned.py
├── utils_cleaned.py         # Compatibility shim -> src/asd_motion_clusters/utils_cleaned.py
├── run_clustering.py        # Minimal entry point for re-running k-means sweeps
└── README.md
```

The compatibility shims keep historical import paths working (`python kbc_cleaned.py --force`) while exposing the packaged implementation under `asd_motion_clusters`.

---

## Quick Start

1. **Ingest data** (or point to existing normalized folders):
   ```bash
   python scripts/ingest_new_data.py --pose-root data/poses --scores data/social_scores.csv
   ```
2. **Regenerate all exports** (distance matrices, clusters, social links):
   ```bash
   python kbc_cleaned.py --force
   ```
3. **Optional:** recompute only social-score stats:
   ```bash
   python kbc_cleaned.py --force-social
   ```
4. **Inspect exports** under `results/exports/` (same schemas as `exports/example_*.csv`).
5. **ST-GCN baseline:** follow `scripts/stgcn_full_analysis.py` after cloning the upstream ST-GCN repo into `external/` and placing the pretrained Kinetics weights listed in `configs/stgcn.yaml`.

Key command-line flags are documented in `kbc_cleaned.py` (for example, `--k 4 6 8`, `--distance-metrics cosine dtw`, `--no-cache`).

---

## Pipeline Stages

| Stage | Script(s) | Output |
| --- | --- | --- |
| Pose ingestion | `scripts/ingest_new_data.py`, `utils_cleaned.py` | Harmonized directory structure under `data/poses/` |
| Feature extraction | `kbc_cleaned.py` (`compute_features`), `utils_cleaned.py` | Aggregated kinematic tables (`results/exports/participant_clusters_with_features_clean.csv`) |
| Distance matrices + clustering | `kbc_cleaned.py` (`run_distance_matrix`, `run_clustering`) | Per-emotion cluster summaries, singleton stats, cross-emotion ARI |
| Social-score linking | `kbc_cleaned.py` (`analyze_social_scores`) | `social_score_summary.csv` + BH-adjusted `social_score_summary_fdr.csv` |
| Visualization & ablations | `scripts/additional_analyses.py`, `scripts/summaries.py` | Figs in `figs/` + CSV digests |
| Neural baseline | `scripts/stgcn_*` | ST-GCN embeddings, baseline comparison tables |

`configs/default.yaml` mirrors the settings used in the paper (12 emotions, 4 ROIs, window = 15 @ 30 FPS, k in {4,6,8}).

---

## Data & Privacy

Raw ASD motion clips and psychometric scores cannot be redistributed. Use `data/README.md` to mirror the expected folder layout, or swap in your own dataset that exposes the same columns (`actor_id`, `emotion`, ROI-specific sequences, AQ/RMET/TIPI scores). The released code works with any dataset that follows this schema.

---

## Exports

`exports/` contains lightweight CSV samples so you can prototype without the full dataset. Running the pipeline produces high-resolution tables with identical columns under `results/exports/`:

- `participant_clusters_with_features_clean.csv`
- `cross_emotion_ari.csv`
- `social_score_summary_fdr.csv`
- `stgcn_social_score_summary.csv`

Each file is documented inline in `kbc_cleaned.py` and matches the manuscript descriptions (for example, Table 3, Figure 4, Figure 5).

---

## Citation

```bibtex
@inproceedings{Aghajani2026ASDMotion,
  title     = {Quantifying Emotion-Specific Motion Signatures and Social Correlates in Autism Spectrum Disorder},
  author    = {Aghajani, Mohammad and Lewis, Gregory F. and Jaime, Mark},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

---

## Contact

Questions about the code or dataset access:

- Mohammad Aghajani -- maghajan@iu.edu
- Open an issue at https://github.com/Mohammad-Aghajani/ASD-Motion-Clusters/issues
