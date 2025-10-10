import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights

def load_resnet(arch="resnet18", weights="DEFAULT"):
    if arch == "resnet18":
        model = resnet18(weights=getattr(ResNet18_Weights, weights)); dim = 512
    elif arch == "resnet50":
        model = resnet50(weights=getattr(ResNet50_Weights, weights)); dim = 2048
    elif arch == "resnet101":
        model = resnet101(weights=getattr(ResNet101_Weights, weights)); dim = 2048
    else:
        raise ValueError(arch)
    
    model.fc = nn.Identity()
    
    for p in model.parameters():
        p.requires_grad = False
        
    model.eval()
    
    return model, dim
