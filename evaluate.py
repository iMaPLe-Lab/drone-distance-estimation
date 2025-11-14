import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.model import DistancePredictor
from src.dataset import getLRDDDataLoader


@torch.no_grad()
def evaluate_model(
    data_root,
    metadata_dir,
    checkpoint_path,
    backbone="resnet50",
    batch_size=16,
    device="cuda",
    save_predictions=False
):

    test_data_root = data_root + "/test"
    test_metadata_dir = metadata_dir + "/test"
    test_loader = getLRDDDataLoader(test_data_root, test_metadata_dir, batch_size)

    model = DistancePredictor(
        backbone_name=backbone,
        bbox_feat_dim=test_ds.bbox_feature_dim
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds = []
    targets = []


    for full_img, crop_img, bbox_feat, distance in tqdm(test_loader):
        full_img = full_img.to(device)
        crop_img = crop_img.to(device)
        bbox_feat = bbox_feat.to(device)
        distance = distance.to(device).float()

        outputs = model(full_img, crop_img, bbox_feat)

        preds.append(outputs.cpu().numpy())
        targets.append(distance.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    # R² = 1 - SS_res / SS_tot
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("\n===== Test Metrics =====")
    print(f"MAE :  {mae:.4f}")
    print(f"MSE :  {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    if save_predictions:
        import pandas as pd
        df = pd.DataFrame({
            "target": targets,
            "prediction": preds
        })
        df.to_csv("test_predictions.csv", index=False)
        print("\nPredictions saved to test_predictions.csv")

    return preds, targets, {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
