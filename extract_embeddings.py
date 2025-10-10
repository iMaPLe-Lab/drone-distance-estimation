import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LRDDDataset
from .backbone import load_resnet

def extract_npz(csv_file, img_folder, labels_folder,
                arch="resnet18",
                batch_size=64, num_workers=4,
                out_path="data/embeddings/run.npz"):
    dataset = LRDDDataset(csv_file=csv_file, img_folder=img_folder, labels_folder=labels_folder)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, emb_dim = load_resnet(arch=arch, weights="DEFAULT")
    model = model.to(device)

    image_embeddings, crop_embeddings, bbox_features, distances = [], [], [], []

    with torch.no_grad():
        for imgs, crops, bboxf, dists in tqdm(loader):
            imgs  = imgs.to(device)
            crops = crops.to(device)

            image_features = model(imgs)    # (B, D)
            crop_features  = model(crops)   # (B, D)

            image_embeddings.append(image_features.cpu().numpy())
            crop_embeddings.append(crop_features.cpu().numpy())
            bbox_features.append(bboxf.numpy())
            distances.append(dists.numpy())

    image_embeddings = np.vstack(image_embeddings).astype(np.float32)
    crop_embeddings  = np.vstack(crop_embeddings).astype(np.float32)
    bbox_features    = np.vstack(bbox_features).astype(np.float32)
    distances        = np.concatenate(distances).astype(np.float32)

    np.savez_compressed(out_path,
                        full=image_embeddings,
                        crop=crop_embeddings,
                        bbox=bbox_features,
                        y=distances,
                        arch=arch,
                        emb_dim=emb_dim)
    print("saved", out_path,
          "| full:", image_embeddings.shape,
          "| crop:", crop_embeddings.shape,
          "| bbox:", bbox_features.shape,
          "| y:", distances.shape)
    return out_path
