import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def make_loaders(
    X_train_all, y_train_cls_all,
    X_val_all, y_val_cls_all,
    X_test_all,  y_test_cls_all,
    batch_train, batch_val, batch_test,
    device
):
    X_train_t = torch.from_numpy(X_train_all).float()
    y_train_t = torch.from_numpy(y_train_cls_all).long()

    X_val_t = torch.from_numpy(X_val_all).float()
    y_val_t = torch.from_numpy(y_val_cls_all).long()

    X_test_t  = torch.from_numpy(X_test_all).float()
    y_test_t  = torch.from_numpy(y_test_cls_all).long()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_train,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_val,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_test,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def evaluate_classification(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


def train_classifier(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    lr,
    weight_decay,
    epochs,
    patience,
    checkpoint_path
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_acc = 0.0
    bad_epochs = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate_classification(model, val_loader, device)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc*100:.2f}%")
        
        print()

        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")

    # return best_acc
    test_acc = evaluate_classification(model, test_loader, device)

    return test_acc
    



def class_to_midpoint(bin_idx_array, bin_edges):
    lefts = bin_edges[bin_idx_array]
    rights = bin_edges[bin_idx_array + 1]
    return 0.5 * (lefts + rights)


def eval_distance_metrics(model, X_test_np, y_test_cont_np, bin_edges, device):
    model.eval()
    with torch.no_grad():
        Xb = torch.from_numpy(X_test_np).float().to(device)
        logits = model(Xb)
        pred_cls = torch.argmax(logits, dim=1).cpu().numpy()

    pred_dist_ft = class_to_midpoint(pred_cls, bin_edges)
    y_true = y_test_cont_np

    abs_err = np.abs(pred_dist_ft - y_true)
    mae = abs_err.mean()
    rmse = np.sqrt((abs_err ** 2).mean())
    pct10 = (abs_err <= 10).mean() * 100.0
    pct20 = (abs_err <= 20).mean() * 100.0

    print("--------- distance quality ---------")
    print(f"MAE (ft):        {mae:.2f}")
    print(f"RMSE (ft):       {rmse:.2f}")
    print(f"% within 10 ft:  {pct10:.2f}%")
    print(f"% within 20 ft:  {pct20:.2f}%")
    print("-------------------------------------")
