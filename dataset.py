import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Preprocessing dataset to feed full images and cropped drone regions
class LRDDDataset(Dataset):
    def __init__(self, csv_file, img_folder, labels_folder):
        # reading metadata csv file
        self.metadata = pd.read_csv(csv_file, encoding="latin1", on_bad_lines="skip")
        
                # --- Clean and validate distance column ---
        if "distance_3d_ft" in self.metadata.columns:
            self.metadata["distance_3d_ft"] = pd.to_numeric(
                self.metadata["distance_3d_ft"], errors="coerce"
            )
            # Drop NaNs and unrealistic distances
            self.metadata = self.metadata[
                self.metadata["distance_3d_ft"].between(0.5, 300)
            ].reset_index(drop=True)
        else:
            raise ValueError(f"'distance_3d_ft' column missing in {csv_file}")


        self.img_folder = img_folder
        self.labels_folder = labels_folder  # can be None or missing

        # keep only rows whose images actually exist in this folder
        existing = []
        for name in self.metadata["img_name"].astype(str):
            p = os.path.join(self.img_folder, name)
            existing.append(os.path.exists(p))
        self.metadata = self.metadata.loc[existing].reset_index(drop=True)

        # defining transformations (resize, normalize, etc.)
        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # reading image path and opening image
        row = self.metadata.iloc[idx]
        img_name = row["img_name"]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")

        img_width, img_height = img.size
        stem, _ = os.path.splitext(img_name)
        label_path = os.path.join(self.labels_folder, stem + ".txt") if isinstance(self.labels_folder, str) else None

        # checking if bounding box file exists
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                line = f.readline().strip()
                if not line:
                    crop = img
                    width = height = area = aspect = 0.0
                else:
                    line = line.split()
                    _, x_center, y_center, width, height = map(float, line)

                    # calculating crop coordinates
                    x_center_pixels = x_center * img_width
                    y_center_pixels = y_center * img_height
                    width_pixels = width * img_width
                    height_pixels = height * img_height

                    x_min = x_center_pixels - width_pixels / 2
                    y_min = y_center_pixels - height_pixels / 2
                    x_max = x_center_pixels + width_pixels / 2
                    y_max = y_center_pixels + height_pixels / 2

                    # cropping the image
                    crop = img.crop((x_min, y_min, x_max, y_max))

                    # bounding box features
                    area = width * height
                    aspect = width / (height + 1e-6)
        else:
            crop = img
            width = height = area = aspect = 0.0

        # converting to tensors
        img_tensor = self.transform(img)
        crop_tensor = self.transform(crop)

        distance = torch.tensor(row["distance_3d_ft"], dtype=torch.float32)
        bbox_features = torch.tensor([width, height, area, aspect],
                                     dtype=torch.float32)

        return img_tensor, crop_tensor, bbox_features, distance
