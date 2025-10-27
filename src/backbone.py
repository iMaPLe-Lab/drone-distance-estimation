import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
)

def build_backbone(name: str, device: torch.device) -> nn.Module:
    """
    name in {"resnet18", "resnet50", "resnet101"}
    Returns a frozen model where .fc is Identity,
    and also returns the embedding dim.
    """
    name = name.lower()

    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        feat_dim = 512
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        feat_dim = 2048
    elif name == "resnet101":
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        feat_dim = 2048
    else:
        raise ValueError(f"Unknown backbone '{name}'")

    # remove classification head, leave pooled features
    model.fc = nn.Identity()

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)
    return model, feat_dim
