import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision.transforms import v2
import pandas as pd

from src.index_dataset import scan_dataset


def getLRDDDataLoader(data_root, metadata_dir, batch_size, shuffle=True):

    manifest = scan_dataset(data_root, metadata_dir)
    manifest_df = pd.DataFrame(manifest)
    
    list_of_datasets = []
    for _, row in manifest_df.iterrows():
        try:
            list_of_datasets.append(LRDDDatasetChunk(row["csv_path"], row["img_dir"], row["label_dir"], transform=None))       
        except Exception as e:
            print(f"[extract:ERROR] {row['date']}/{row['flight_id']} failed: {e}")

    full_dataset = data.ConcatDataset(list_of_datasets)
    loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


class LRDDDatasetChunk(Dataset):
    """
    Dataset for a single flight.

    Inputs:
        csv_file: path to metadata CSV (must include columns like img_name, distance_3d_ft)
        img_folder: directory with the RGB frames
        labels_folder: directory with YOLO txt labels
        transform: (optional) torchvision v2 transform

    __getitem__ returns:
        full_img_tensor: (3,224,224) float32 normalized
        crop_tensor:     (3,224,224) float32 normalized (drone crop if we have bbox, else full img)
        bbox_features:   tensor([width, height, area, aspect]) in normalized coords
        distance:        scalar tensor (float32), distance_3d_ft
    """

    def __init__(self, csv_file, img_folder, labels_folder, transform=None):
        
        # assign first so they're available in checks
        self.imgs = img_folder
        self.yolo_labels = labels_folder

        df = pd.read_csv(csv_file)

        keep_mask = []
        missing_image_count = 0
        missing_distance_count = 0

        for _, row in df.iterrows():
            img_name = row["img_name"]
            img_path = os.path.join(self.imgs, img_name)

            # 1. check image exists
            if not os.path.exists(img_path):
                missing_image_count += 1
                keep_mask.append(False)
                continue

            # 2. check distance is valid numeric
            raw_dist = row.get("distance_3d_ft", None)

            # helper: is this usable distance?
            def valid_distance(val):
                # reject None / NaN
                if val is None or pd.isna(val):
                    return False
                # try to convert to float
                try:
                    float_val = float(val)
                except (TypeError, ValueError):
                    return False
                # optional sanity: distance should be >0
                # tweak if you ever have 0-ft ground truth
                if not np.isfinite(float_val):
                    return False
                return True

            if not valid_distance(raw_dist):
                missing_distance_count += 1
                keep_mask.append(False)
                continue

            # if we got here, keep the row
            keep_mask.append(True)

        df_clean = df[keep_mask].reset_index(drop=True)
        self.metadata = df_clean

        dropped_total = missing_image_count + missing_distance_count
        if dropped_total > 0:
            if missing_image_count > 0:
                print(
                    f"[LRDDDataset] WARN: {missing_image_count} rows dropped "
                    f"because image file was missing in {self.imgs}"
                )
            if missing_distance_count > 0:
                print(
                    f"[LRDDDataset] WARN: {missing_distance_count} rows dropped "
                    f"because distance_3d_ft was missing/invalid in {os.path.basename(csv_file)}"
                )
            print(
                f"[LRDDDataset] INFO: kept {len(df_clean)} / {len(df)} rows "
                f"for {os.path.basename(csv_file)}"
            )

        self.metadata = df_clean
        self.yolo_labels = labels_folder

        if transform is None:
            transform = v2.Compose([
                v2.Resize((224,224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['img_name']

        # --- open image ---
        img_path = os.path.join(self.imgs, img_name)
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size

        # --- try to read YOLO bbox from txt ---
        label_path = os.path.join(
            self.yolo_labels,
            os.path.splitext(img_name)[0] + ".txt"
        )

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                line = f.readline().strip()

            if not line:
                crop = img
                width = height = area = aspect = 0.0
            else:
                parts = line.split()
                # YOLO format: cls cx cy w h (all normalized 0-1)
                _, x_center, y_center, width, height = map(float, parts)

                # convert to pixels
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                w_px = width  * img_width
                h_px = height * img_height

                x_min = x_center_px - w_px/2
                y_min = y_center_px - h_px/2
                x_max = x_center_px + w_px/2
                y_max = y_center_px + h_px/2

                crop = img.crop((x_min, y_min, x_max, y_max))

                area = width * height
                aspect = width / (height + 1e-6)
        else:
            crop = img
            width = height = area = aspect = 0.0

        full_img_tensor = self.transform(img)
        crop_tensor     = self.transform(crop)

        distance = torch.tensor(row['distance_3d_ft'], dtype=torch.float32)
        bbox_features = torch.tensor([width, height, area, aspect], dtype=torch.float32)

        return full_img_tensor, crop_tensor, bbox_features, distance
