# Data Directory

Raw pose recordings and social-score spreadsheets are **not** part of this repository because they contain identifiable signals. To reproduce the paper results you need to provision the following structure under `data/`:

```
data/
+-- poses/
¦   +-- actor_001/
¦   ¦   +-- admiration/
¦   ¦       +-- *.json  # Kinect-style pose dumps per frame
¦   +-- ...
+-- social_scores.csv    # Aggregated AQ/RMET/TIPI measures per actor
+-- demographics.csv     # Optional demographic metadata used in the paper
```

Use `scripts/ingest_new_data.py` to convert raw session folders into this hierarchy. Each pose JSON should follow the OpenPose/Kinect skeleton convention expected by `utils_cleaned.py` (25 joints, metric coordinates, 30 FPS).

If you cannot access the original ASD dataset, you can still execute the pipeline by pointing `configs/default.yaml` to your own motion-capture collection as long as it supplies the same columns (actor_id, emotion, ROI windows, etc.).
