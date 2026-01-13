# Drone to Drone Distance Estimation

## What is this?

This repo helps train distance estimators for drones.

We are primarily training Resnet 18, 50 and 101 with LRDD V3 Dataset.

The model uses:
- the full RGB frame (optional, set by crop_only variable)
- a cropped patch around the drone (from YOLO bbox)
- simple bbox geometry features (width, height, area, aspect ratio)

We run the crop and optionally the full image through a ResNet backbone to get:
- `image_embedding`  (512 dim)
- `crop_embedding`   (512 dim)

Which we then concatonate with the bbox geometry features:
- `bbox_feats`       (4 dims)

We then regress distance from the features using a prediction head and report MAE / RMSE in feet.

---

## Repo layout (important parts)
```text
src/
  dataset.py          # dataset loading code
  binning.py          # bin edges, splits, pooling (UNUSED - TO DELETE)
  index_dataset.py    # (UNUSED - TO DELETE)
  model.py            # our model in DistancePredictor class, baseline model in DroneRanger class

drone_range.ipynb     # train and evaluate models. Uses functions from evalute.py and train.py
evaluate.py           # code for running batch inference on test data
train.py              # code for training models
```

## How to run

1. Set the paths and hyperparameters in:
   `drone_range.ipynb`

   Make sure these are correct for your machine:
   - `data_root`
   - `metadata_dir`

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Train and evaluate models by selecting parameters in
   ```bash
   drone_range.ipynb
   ```
