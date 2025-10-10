import torch.nn as nn

def make_regressor(input_dim, hidden=64, dropout=0.2):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )
