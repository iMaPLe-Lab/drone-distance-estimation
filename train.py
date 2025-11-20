import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json

from src.model import DistancePredictor
from src.dataset import getLRDDDataLoader

def train(
    data_root="/mnt/active_storage/Knut/LRDD_v3",
    metadata_dir="/mnt/active_storage/Knut/LRDD_v3/metadata",
    backbone="resnet50",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    device="cuda",
    patience=5,
    min_delta=1e-4,
    checkpoint_path="best_model.pth",
    metrics_path="training_metrics.json",
    bbox_feature_dim=4,
    num_workers=4,
):

    train_data_root = data_root + "/train"
    train_metadata_dir = metadata_dir + "/train"
    val_data_root = data_root + "/val"
    val_metadata_dir = metadata_dir + "/val"

    train_loader = getLRDDDataLoader(train_data_root, train_metadata_dir, batch_size, num_workers=num_workers)
    val_loader = getLRDDDataLoader(val_data_root, val_metadata_dir, batch_size, num_workers=num_workers)

    # Model
    model = DistancePredictor(
        backbone_name=backbone,
        bbox_feat_dim=bbox_feature_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    patience_counter = 0

    # For saving metrics
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # --------------------------------------------------
        # Training loop with tqdm
        # --------------------------------------------------
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="Training", leave=False)

        for full_img, crop_img, bbox_feat, distance in pbar:
            full_img = full_img.to(device)
            crop_img = crop_img.to(device)
            bbox_feat = bbox_feat.to(device)
            distance = distance.to(device).float()

            optimizer.zero_grad()
            pred = model(full_img, crop_img, bbox_feat)
            loss = criterion(pred, distance)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss # * full_img.size(0)

            # Update tqdm info
            pbar.set_postfix({"batch_loss": batch_loss})

        train_loss /= len(train_loader.dataset)

        # --------------------------------------------------
        # Validation loop with tqdm
        # --------------------------------------------------
        model.eval()
        val_loss = 0
        pbar_val = tqdm(val_loader, desc="Validation", leave=False)

        preds = []
        targets = []

        with torch.no_grad():
            for full_img, crop_img, bbox_feat, distance in pbar_val:
                full_img = full_img.to(device)
                crop_img = crop_img.to(device)
                bbox_feat = bbox_feat.to(device)
                distance = distance.to(device).float()

                pred = model(full_img, crop_img, bbox_feat)
                loss = criterion(pred, distance)

                val_loss += loss.item() # * full_img.size(0)
                pbar_val.set_postfix({"batch_loss": loss.item()})

                preds.append(pred.cpu().numpy())
                targets.append(distance.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
    
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(mse)

        val_loss /= len(val_loader.dataset)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f}")

        # Save metrics
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(float(train_loss))
        metrics["val_loss"].append(float(val_loss))
        metrics["val_mae"].append(float(mae))
        metrics["val_rmse"].append(float(rmse))

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------
        torch.save(model.state_dict(), checkpoint_path+"_latest.pth")
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path+"_best.pth")
            print(f"  ✓ Improvement detected. Saving model to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  ✗ No improvement (Patience {patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Save metrics to JSON
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to {metrics_path}")

    return model, metrics


