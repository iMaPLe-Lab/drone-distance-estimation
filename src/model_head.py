import torch.nn as nn

def build_classifier_head(input_dim: int, num_bins: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_bins)
    )
