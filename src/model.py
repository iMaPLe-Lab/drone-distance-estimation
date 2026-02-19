import torch
import torch.nn as nn
import torchvision.models as models


class DistancePredictor(nn.Module):
    def __init__(self, backbone_name="resnet18", alt_head=0, crop_only=True, bbox_feat_dim=4):
        super().__init__()
        self.crop_only = crop_only
        self.alt_head = alt_head

        # ------- Select ResNet backbone -------
        if backbone_name == "resnet18":
            self.backbone_full = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone_crop = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            backbone_out_dim = 512
        elif backbone_name == "resnet50":
            self.backbone_full = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone_crop = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone_out_dim = 2048
        elif backbone_name == "resnet101":
            self.backbone_full = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.backbone_crop = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            backbone_out_dim = 2048
        else:
            raise ValueError("Invalid backbone:", backbone_name)

        # Remove final classification layer
        self.backbone_full.fc = nn.Identity()
        self.backbone_crop.fc = nn.Identity()

        # ---- Final prediction head ----
        if crop_only:
            combined_dim = backbone_out_dim + bbox_feat_dim
        else:
            combined_dim = backbone_out_dim * 2 + bbox_feat_dim

        if self.alt_head==4 or self.alt_head==5 or self.alt_head==6:
            self.norm = nn.LayerNorm(combined_dim)

        if self.alt_head==5 or self.alt_head==6:
            self.bbox_gate = nn.Sequential(
                nn.Linear(bbox_feat_dim, backbone_out_dim),
                nn.Sigmoid()
            )

        if alt_head == 1: # alternate 4-layer prediction head
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # regression output
            )
        elif alt_head == 2: # 3-layer prediction head
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # regression output
            )
        elif alt_head == 3: #4-layer prediction head
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # regression output
            )
        elif alt_head == 4: # DR head + layer norm before head
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # regression output
            )
        elif alt_head == 5: # DR head + layer norm + bbox gate
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # regression output
            )
        elif alt_head == 6: # DR head + bbox gate
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # regression output
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # regression output
            )


    def forward(self, crop_img, bbox_features, full_img=None):
        if self.crop_only:
            crop_feat = self.backbone_crop(crop_img)
            
            if self.alt_head==5 or self.alt_head==6:
                gate = self.bbox_gate(bbox_features)
                crop_feat = crop_feat * gate

            x = torch.cat([crop_feat, bbox_features], dim=1)
            
            if self.alt_head==4 or self.alt_head==5:
                x = self.norm(x)
            
            out = self.head(x)
            return out.squeeze(1)
        else:
            full_feat = self.backbone_full(full_img)
            crop_feat = self.backbone_crop(crop_img)
            x = torch.cat([full_feat, crop_feat, bbox_features], dim=1)
            out = self.head(x)
            return out.squeeze(1)
            

class DroneRanger(nn.Module):
    def __init__(self, bbox_feat_dim=4):
        super().__init__()

        # ------- DroneRanger backbone -------
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2)
        )

        backbone_out_dim = 28800

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(backbone_out_dim+bbox_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # regression output
        )


    def forward(self, crop_img, bbox_features):
        crop_feat = self.backbone(crop_img)  
        x = torch.cat([crop_feat.view(crop_feat.size(0), -1), bbox_features], dim=1)
        out = self.head(x)
        return out.squeeze(1)
