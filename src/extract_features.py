import os
import numpy as np
from typing import Dict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .dataset import LRDDDataset
from .backbone import build_backbone


def extract_flight_features(
    flight_row: Dict,
    batch_size: int,
    device: torch.device,
    out_dir: str,
    backbone_name: str,
):
    date_str   = flight_row["date"]
    flight_id  = flight_row["flight_id"]
    img_dir    = flight_row["img_dir"]
    label_dir  = flight_row["label_dir"]
    csv_path   = flight_row["csv_path"]

    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{date_str}_{flight_id}.npz".replace(" ", "_")
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path):
        print(f"[extract] skip {date_str}/{flight_id} (already exists: {out_name})")
        return out_path

    # dataset + loader
    ds = LRDDDataset(csv_file=csv_path,
                     img_folder=img_dir,
                     labels_folder=label_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # frozen backbone
    model, feat_dim = build_backbone(backbone_name, device)

    all_image_embs = []
    all_crop_embs  = []
    all_bbox_feats = []
    all_dists      = []
    all_names      = []

    names_col = ds.metadata['img_name'].tolist()
    idx_start = 0

    with torch.no_grad():
        for (imgs, crops, bboxf, dists) in tqdm(
            loader, desc=f"{date_str}/{flight_id}", leave=False
        ):
            imgs  = imgs.to(device)
            crops = crops.to(device)

            img_feats  = model(imgs)   # (B, feat_dim)
            crop_feats = model(crops)  # (B, feat_dim)

            all_image_embs.append(img_feats.cpu().numpy())
            all_crop_embs.append(crop_feats.cpu().numpy())

            all_bbox_feats.append(bboxf.numpy())
            all_dists.append(dists.numpy())

            bsz = imgs.shape[0]
            all_names.extend(names_col[idx_start : idx_start + bsz])
            idx_start += bsz

    image_embeddings = np.vstack(all_image_embs).astype(np.float32)
    crop_embeddings  = np.vstack(all_crop_embs).astype(np.float32)
    bbox_feats       = np.vstack(all_bbox_feats).astype(np.float32)
    distances        = np.concatenate(all_dists).astype(np.float32)

    np.savez_compressed(
        out_path,
        image_embeddings=image_embeddings,
        crop_embeddings=crop_embeddings,
        bbox_feats=bbox_feats,
        distances_feet=distances,
        img_names=np.array(all_names),
        backbone=np.array([backbone_name]),
        feat_dim=np.array([feat_dim], dtype=np.int32),
    )

    print(f"[extract] saved {out_name} ({image_embeddings.shape[0]} samples)")
    return out_path


def extract_all_flights(
    manifest_df,
    batch_size: int,
    device: torch.device,
    out_dir: str,
    backbone_name: str,
):
    for _, row in manifest_df.iterrows():
        row_dict = {
            "date": row["date"],
            "flight_id": row["flight_id"],
            "img_dir": row["img_dir"],
            "label_dir": row["label_dir"],
            "csv_path": row["csv_path"],
        }
        try:
            extract_flight_features(
                flight_row=row_dict,
                batch_size=batch_size,
                device=device,
                out_dir=out_dir,
                backbone_name=backbone_name,
            )
        except Exception as e:
            print(f"[extract:ERROR] {row['date']}/{row['flight_id']} failed: {e}")
