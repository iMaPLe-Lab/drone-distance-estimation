# Distance Estimation Baseline

## What is this?

This repo trains a baseline distance estimator for drones.

The model uses:
- the full RGB frame
- a cropped patch around the drone (from YOLO bbox)
- simple bbox geometry features (width, height, area, aspect ratio)

We run both the full image and the crop through a frozen ResNet to get:
- `image_embedding`  (512 dim)
- `crop_embedding`   (512 dim)
- `bbox_feats`       (4 dims)

Concatenate those → a 1028-D feature vector per frame.

We DO NOT regress distance directly.  
Instead:
- we bucket distance into ranges (bins like 0–20 ft, 20–40 ft, …)
- we train a classifier to predict which bin you're in
- at eval time we map the predicted bin back to a single distance using that bin’s midpoint

Then we report:
- MAE / RMSE in feet
- % within 10 ft
- % within 20 ft

This is our baseline. Future backbones / heads must beat it.

---

## Repo layout (important parts)

src/
  config.py           # paths + hyperparams
  dataset.py          # LRDDDataset for one flight
  index_dataset.py    # scan raw data into a manifest
  extract_features.py # feature extraction to .npz
  binning.py          # bin edges, splits, pooling
  model_head.py       # small classifier head (MLP)
  train_eval.py       # train loop + metrics

scripts/
  run_extract.py      # STEP 1: extract features for all flights
  run_train_eval.py   # STEP 2: train + eval on those features

## How to run

1. Set the paths and hyperparameters in:
   `src/config.py`

   Make sure these are correct for your machine:
   - `DATA_ROOT`
   - `EMBED_DIR`
   - `CHECKPOINT_DIR`

2. Install requirements:
   ```bash
   pip install -r requirements.txt

3. Extract features from all flights (writes .npz files to EMBED_DIR):
   ```bash
   python scripts/run_extract.py

4. Train and evaluate the baseline model:
   ```bash
   python scripts/run_train_eval.py
