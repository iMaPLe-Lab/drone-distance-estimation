import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
from torchvision.transforms import v2

from src.model import DistancePredictor, DroneRanger
from src.dataset import getLRDDDataLoader


@torch.no_grad()
def evaluate_model(
    data_root,
    metadata_dir,
    checkpoint_path,
    backbone="resnet50",
    crop_only=True,
    use_droneranger=False,
    use_alt_head=0,
    batch_size=16,
    device="cuda",
    save_predictions=False,
    bbox_feature_dim=4,
    num_workers=4,
    max_dist=10000,
    crop_size=512,
    is_log = False,
    xy_features = False
):

    test_data_root = data_root + "/test"
    test_metadata_dir = metadata_dir + "/test"
    test_loader = getLRDDDataLoader(test_data_root, test_metadata_dir, crop_only, batch_size, num_workers=num_workers, crop_size=crop_size, max_dist=max_dist, is_log=is_log, xy_features=xy_features)

    if use_droneranger:
        model = DroneRanger(
            bbox_feat_dim=bbox_feature_dim
        ).to(device)
    else:
        model = DistancePredictor(
            backbone_name=backbone,
            alt_head=use_alt_head,
            crop_only=crop_only,
            bbox_feat_dim=bbox_feature_dim,
        ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds = []
    targets = []

    if crop_only:
        for crop_img, bbox_feat, distance in tqdm(test_loader):
            crop_img = crop_img.to(device)
            bbox_feat = bbox_feat.to(device)
            distance = distance.to(device).float()
    
            outputs = model(crop_img, bbox_feat)
    
            preds.append(outputs.cpu().numpy())
            targets.append(distance.cpu().numpy())
    else:
        for full_img, crop_img, bbox_feat, distance in tqdm(test_loader):
            full_img = full_img.to(device)
            crop_img = crop_img.to(device)
            bbox_feat = bbox_feat.to(device)
            distance = distance.to(device).float()
    
            outputs = model(crop_img, bbox_feat, full_img)
    
            preds.append(outputs.cpu().numpy())
            targets.append(distance.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    if is_log:
        preds = torch.exp(torch.tensor(preds)).numpy() # convert predicted log distance back to linear for evaluation
        targets = torch.exp(torch.tensor(targets)).numpy()



    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)

    rel_mae = np.mean(np.abs(preds / targets))

    # R² = 1 - SS_res / SS_tot
    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print("\n===== Test Metrics =====")
    print(f"MAE :  {mae:.4f}")
    print(f"Rel MAE :  {rel_mae:.4f}")
    print(f"MSE :  {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    if save_predictions:
        import pandas as pd
        df = pd.DataFrame({
            "target": targets,
            "prediction": preds
        })
        df.to_csv("test_predictions.csv", index=False)
        print("\nPredictions saved to test_predictions.csv")

    return preds, targets, {"mae": mae, "rel": rel_mae, "mse": mse, "rmse": rmse, "r2": r2}


@torch.no_grad()
def evaluate_model_single(
    data_folder,
    metadata_file,
    image_name,
    checkpoint_path,
    backbone="resnet50",
    crop_only=True,
    use_droneranger=False,
    use_alt_head=0,
    batch_size=16,
    device="cuda",
    bbox_feature_dim=4,
    crop_size=512
):
    # load image + yolo label + distance
    # test_data_root = data_root + "/test"
    # test_metadata_dir = metadata_dir + "/test"
    # "04-11-2025_DJI_0007_metadata.csv"

    img_folder = data_folder + "images/"
    yolo_labels = data_folder + "labels/"
    df = pd.read_csv(metadata_file)


    row = df[df['img_name'] == image_name]
    img_path = os.path.join(img_folder, image_name)
    print(img_path)

    if not os.path.exists(img_path):
        print("Image path not valid")
        return

    #raw_dist = row.get("distance_3d_ft", None)
    raw_dist = row['distance_3d_ft'].iloc[0]
    print(raw_dist)

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
        print("Image does not have a valid distance")
        return


    full_transform = v2.Compose([
        v2.Resize((512,512)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) # imagenet mean and std
    ])

    crop_transform = v2.Compose([
        v2.Resize((crop_size,crop_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) # imagenet mean and std
    ])

    # --- open image ---
    img = Image.open(img_path).convert('RGB')
    img_width, img_height = img.size

    # --- try to read YOLO bbox from txt ---
    label_path = os.path.join(
        yolo_labels,
        os.path.splitext(image_name)[0] + ".txt"
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

    if not crop_only:
        full_img_tensor = full_transform(img)
        
    crop_tensor     = crop_transform(crop)
    distance = torch.tensor(row['distance_3d_ft'], dtype=torch.float32)
    bbox_features = torch.tensor([width, height, area, aspect], dtype=torch.float32)

    
    if use_droneranger:
        model = DroneRanger(
            bbox_feat_dim=bbox_feature_dim
        ).to(device)
    else:
        model = DistancePredictor(
            backbone_name=backbone,
            alt_head=use_alt_head,
            crop_only=crop_only,
            bbox_feat_dim=bbox_feature_dim,
        ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if crop_only:
        crop_tensor = crop_tensor.to(device)
        bbox_features = bbox_features.to(device)
        distance = distance.to(device).float()

        outputs = model(crop_tensor.unsqueeze(0), bbox_features.unsqueeze(0))

        pred = outputs.cpu().numpy()
        target = distance.cpu().numpy()
    else:
        full_img_tensor = full_img_tensor.to(device)
        crop_tensor = crop_tensor.to(device)
        bbox_features = bbox_features.to(device)
        distance = distance.to(device).float()

        outputs = model(crop_tensor, bbox_features, full_img_tensor)

        pred = outputs.cpu().numpy()
        target = distance.cpu().numpy()

    return img, crop, target, pred



 




