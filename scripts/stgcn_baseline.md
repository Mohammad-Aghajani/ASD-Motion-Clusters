# ST-GCN Baseline Integration

This note documents the steps we used to benchmark ST-GCN features inside the ABAW/CVPR pipeline. The workflow mirrors the official [ST-GCN release](https://github.com/yysijie/st-gcn) and assumes you already have the pose JSON exports produced by the project.

## 1. Environment

```bash
conda create -n stgcn python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch
conda activate stgcn
pip install numpy pandas tqdm
```

Clone ST-GCN alongside this repository:

```bash
git clone https://github.com/yysijie/st-gcn.git external/st-gcn
cd external/st-gcn
python setup.py install
cd -
```

## 2. Prepare skeleton sequences

Use the helper script supplied with ST-GCN to convert pose JSON files into `.npy` skeleton clips (one file per actor/emotion):

```bash
python external/st-gcn/tools/generate_skeleton_json.py \
  --pose-root data/pose_sequences \
  --out-dir workdir/stgcn_raw
```

The emitted files should follow `<actor>_<emotion>.npy` naming.

## 3. Extract ST-GCN embeddings

Run the pretrained ST-GCN model to obtain 256-D embeddings for each clip:

```bash
python external/st-gcn/main.py recognition \
  --config config/stgcn_kinect_pretrained.yaml \
  --phase test \
  --weights checkpoints/stgcn.kinetics-pretrained.pt \
  --work-dir workdir/stgcn_features \
  --save-score True
```

Move the produced `*.npy` score files into a single directory such as `workdir/stgcn_embeddings/`. Each file has shape `(frames, 256)`.

## 4. Aggregate embeddings for clustering

Summarise the embeddings using the helper script included in this repository:

```bash
python scripts/stgcn_baseline_merge.py \
  --embedding-dir workdir/stgcn_embeddings \
  --output exports/stgcn_features_summary.csv
```

The output CSV stores per-clip statistics (`stgcn_mean_*`, `stgcn_max_*`, `stgcn_std_*`) together with the participant label. Keep the file in `exports/` so that downstream clustering can ingest it.

## 5. Re-run clustering with ST-GCN descriptors

Update the clustering configuration (e.g. `config/pipeline.yaml`) to point to the new feature table and execute:

```bash
python run_clustering.py --config config/pipeline.yaml --feature-set stgcn
```

This reproduces the metrics quoted in the manuscript: average silhouette `0.19 ± 0.03`, 11 significant social-score findings, and ARI values comparable to the Euclidean baseline yet below the multi-metric pipeline.

## Notes

- GPU resources are only needed for the ST-GCN forward pass; aggregation and clustering remain CPU-friendly.
- The merge script is deterministic and can be re-run whenever new embeddings are added.
- Keep the naming convention `<actor>_<emotion>.npy` to ensure participant labels align with the rest of the pipeline.
