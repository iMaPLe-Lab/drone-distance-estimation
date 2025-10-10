import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .head import make_regressor


def load_npz_dataset(npz_file_path):
    """Load embeddings and labels from a saved .npz file."""
    data = np.load(npz_file_path)
    full_image_embeddings = data["full"].astype(np.float32)
    crop_image_embeddings = data["crop"].astype(np.float32)
    bounding_box_features = data["bbox"].astype(np.float32)
    distance_labels = data["y"].astype(np.float32).reshape(-1, 1)

    # Combine all features together
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
