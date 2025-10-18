import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from head import make_regressor


def load_npz_dataset(npz_file_path, y_min=0.5, y_max=300.0):
    """Load embeddings/labels, filter invalid targets, and return combined_features, distance_labels."""
    data = np.load(npz_file_path, allow_pickle=False)

    full_image_embeddings   = data["full"].astype(np.float32)
    crop_image_embeddings   = data["crop"].astype(np.float32)
    bounding_box_features   = data["bbox"].astype(np.float32)
    distance_labels         = data["y"].astype(np.float32).reshape(-1, 1)

    # mask: finite & within plausible range
    mask = np.isfinite(distance_labels) & (distance_labels >= y_min) & (distance_labels <= y_max)
    n_bad = int((~mask).sum())
    if n_bad:
        print(f"🧹 Filtering {n_bad} rows in {os.path.basename(npz_file_path)} "
              f"(kept {int(mask.sum())} / {len(distance_labels)})")

    # apply mask
    full_image_embeddings = full_image_embeddings[mask.ravel()]
    crop_image_embeddings = crop_image_embeddings[mask.ravel()]
    bounding_box_features = bounding_box_features[mask.ravel()]
    distance_labels       = distance_labels[mask]

    # Combine all features together (unchanged naming)
    combined_features = np.hstack([
        full_image_embeddings,
        crop_image_embeddings,
        bounding_box_features
    ]).astype(np.float32)

    return combined_features, distance_labels



def evaluate_model(model, data_loader, device):
    """Evaluate model performance on a given dataloader."""
    model.eval()
    total_squared_error, total_absolute_error, total_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for feature_batch, target_batch in data_loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)

            predictions = model(feature_batch)
            total_squared_error += ((predictions - target_batch) ** 2).sum().item()
            total_absolute_error += (predictions - target_batch).abs().sum().item()
            total_samples += target_batch.numel()

    mean_squared_error = total_squared_error / total_samples
    root_mean_squared_error = float(np.sqrt(mean_squared_error))
    mean_absolute_error = float(total_absolute_error / total_samples)

    return root_mean_squared_error, mean_absolute_error

def drop_nan_pairs(X, y, name):
    # y comes in as (N,1). Flatten mask ONLY, then reshape y back to (K,1).
    mask = ~np.isnan(y).ravel()
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"🧹 Dropping {dropped} NaN labels from {name} set")
    return X[mask], y[mask].reshape(-1, 1)



def train_distance_regressor(
    train_npz_path,
    val_npz_path,
    test_npz_path,
    hidden_units=128,
    dropout_rate=0.2,
    learning_rate=1e-3,
    weight_decay=1e-4,
    total_epochs=40,
    patience=6,
    batch_size_train=64,
    batch_size_eval=256
):
    """Train regression model using embeddings and evaluate on test split."""

    # Load precomputed features and distance labels
    X_train, y_train = load_npz_dataset(train_npz_path)
    X_val, y_val = load_npz_dataset(val_npz_path)
    X_test, y_test = load_npz_dataset(test_npz_path)
    
    X_train, y_train = drop_nan_pairs(X_train, y_train, "train")
    X_val, y_val = drop_nan_pairs(X_val, y_val, "val")
    X_test, y_test = drop_nan_pairs(X_test, y_test, "test")
    
    # right after you load/clean X_train, y_train, X_val, y_val
    y_train_mean = float(y_train.mean())
    mae_baseline = float(np.abs(y_val - y_train_mean).mean())
    rmse_baseline = float(np.sqrt(((y_val - y_train_mean)**2).mean()))
    print(f"baseline (val) MAE={mae_baseline:.2f} | RMSE={rmse_baseline:.2f}")

    
    print("shapes:",
      X_train.shape, y_train.shape,
      X_val.shape,   y_val.shape,
      X_test.shape,  y_test.shape)
    
    # after X_train, y_train = load_npz_dataset(...), etc.
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        assert np.isfinite(y).all(), f"{name}: found non-finite y"
        ymax = float(y.max()); ymin = float(y.min())
        assert 0.5 <= ymin and ymax <= 300, f"{name}: y out of range [{ymin}, {ymax}]"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize regression head
    input_dimension = X_train.shape[1]
    regressor_model = make_regressor(
        input_dim=input_dimension,
        hidden=hidden_units,
        dropout=dropout_rate
    ).to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        regressor_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Prepare data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size_train,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size_eval,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size_eval,
        shuffle=False
    )

    # Early stopping setup
    best_mean_absolute_error = float("inf")
    patience_limit = patience
    bad_epochs = 0

    # Training loop
    for current_epoch in range(1, total_epochs + 1):
        regressor_model.train()

        for feature_batch, target_batch in train_loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)

            predictions = regressor_model(feature_batch)
            loss = loss_function(predictions, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        root_mean_squared_error, mean_absolute_error = evaluate_model(
            regressor_model, val_loader, device
        )
        print(
            f"epoch {current_epoch:02d} | "
            f"val root_mean_squared_error={root_mean_squared_error:.3f} | "
            f"val mean_absolute_error={mean_absolute_error:.3f}"
        )

        # Early stopping logic
        if mean_absolute_error < best_mean_absolute_error - 1e-3:
            best_mean_absolute_error = mean_absolute_error
            bad_epochs = 0
            torch.save(regressor_model.state_dict(), "best_regressor_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience_limit:
                print("early stopping triggered.")
                break

    # Final evaluation on test set
    regressor_model.load_state_dict(torch.load("best_regressor_model.pt"))
    test_rmse, test_mae = evaluate_model(regressor_model, test_loader, device)
    print(f"test root_mean_squared_error={test_rmse:.3f} | test mean_absolute_error={test_mae:.3f}")

    return regressor_model
