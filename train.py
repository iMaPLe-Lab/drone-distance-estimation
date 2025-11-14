import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.model import DistancePredictor
from src.dataset import getLRDDDataLoader


def train(
    data_root,
    metadata_dir,
    backbone="resnet50",
    epochs=20,
    batch_size=16,
    lr=1e-4,
    device="cuda",
    patience=5,
    min_delta=1e-4,
    checkpoint_path="best_model.pth",
    bbox_feature_dim=4
):

    train_data_root = data_root + "/train"
    train_metadata_dir = metadata_dir + "/train"
    val_data_root = data_root + "/val"
    val_metadata_dir = metadata_dir + "/val"

    train_loader = getLRDDDataLoader(train_data_root, train_metadata_dir, batch_size)
    val_loader = getLRDDDataLoader(val_data_root, val_metadata_dir, batch_size)

    model = DistancePredictor(
        backbone_name=backbone,
        bbox_feat_dim=bbox_feature_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        print(f"==== Epoch {epoch+1} of {epochs} ====")

        # -------------------------
        # Training phase
        # -------------------------
        model.train()
        train_loss = 0

        for full_img, crop_img, bbox_feat, distance in tqdm(train_loader):
            full_img = full_img.to(device)
            crop_img = crop_img.to(device)
            bbox_feat = bbox_feat.to(device)
            distance = distance.to(device).float()

            optimizer.zero_grad()
            pred = model(full_img, crop_img, bbox_feat)
            loss = criterion(pred, distance)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * full_img.size(0)

        train_loss /= len(train_loader.dataset)

        # -------------------------
        # Validation phase
        # -------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for full_img, crop_img, bbox_feat, distance in tqdm(val_loader):
                full_img = full_img.to(device)
                crop_img = crop_img.to(device)
                bbox_feat = bbox_feat.to(device)
                distance = distance.to(device).float()

                pred = model(full_img, crop_img, bbox_feat)
                loss = criterion(pred, distance)
                val_loss += loss.item() * full_img.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # -------------------------
        # Early stopping check
        # -------------------------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0

            # save the best model
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Improvement detected. Saving model to {checkpoint_path}")

        else:
            patience_counter += 1
            print(f"  ✗ No improvement (Patience {patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}")
    return model

