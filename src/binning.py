import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

def load_all_flights(embed_dir):
    train_flights = []
    val_flights = []
    test_flights = []
    train_npz_files = sorted(glob.glob(os.path.join(embed_dir, 'train', "*.npz")))
    val_npz_files = sorted(glob.glob(os.path.join(embed_dir, 'val', "*.npz")))
    test_npz_files = sorted(glob.glob(os.path.join(embed_dir, 'test', "*.npz")))
    print(f"[load_all_flights] found {len(train_npz_files)+len(val_npz_files)+len(test_npz_files)} npz files")

    for path in train_npz_files:
        data = np.load(path)

        needed = ["image_embeddings", "crop_embeddings", "bbox_feats", "distances_feet"]
        if not all(k in data for k in needed):
            print(f"[skip] {os.path.basename(path)} (missing keys)")
            continue

        img_emb = data["image_embeddings"].astype(np.float32)
        crop_emb = data["crop_embeddings"].astype(np.float32)
        bbox_ft  = data["bbox_feats"].astype(np.float32)
        dist_ft  = data["distances_feet"].astype(np.float32)

        X = np.hstack([img_emb, crop_emb, bbox_ft]).astype(np.float32)

        flight_name = os.path.splitext(os.path.basename(path))[0]
        print(f"[load] {flight_name:30s} N={len(dist_ft):5d}  "
              f"dist[{dist_ft.min():.2f}, {dist_ft.max():.2f}] ft")

        train_flights.append({
            "flight_name": flight_name,
            "X": X,
            "y_cont": dist_ft,
        })

    for path in val_npz_files:
        data = np.load(path)

        needed = ["image_embeddings", "crop_embeddings", "bbox_feats", "distances_feet"]
        if not all(k in data for k in needed):
            print(f"[skip] {os.path.basename(path)} (missing keys)")
            continue

        img_emb = data["image_embeddings"].astype(np.float32)
        crop_emb = data["crop_embeddings"].astype(np.float32)
        bbox_ft  = data["bbox_feats"].astype(np.float32)
        dist_ft  = data["distances_feet"].astype(np.float32)

        X = np.hstack([img_emb, crop_emb, bbox_ft]).astype(np.float32)

        flight_name = os.path.splitext(os.path.basename(path))[0]
        print(f"[load] {flight_name:30s} N={len(dist_ft):5d}  "
              f"dist[{dist_ft.min():.2f}, {dist_ft.max():.2f}] ft")

        val_flights.append({
            "flight_name": flight_name,
            "X": X,
            "y_cont": dist_ft,
        })

    for path in test_npz_files:
        data = np.load(path)

        needed = ["image_embeddings", "crop_embeddings", "bbox_feats", "distances_feet"]
        if not all(k in data for k in needed):
            print(f"[skip] {os.path.basename(path)} (missing keys)")
            continue

        img_emb = data["image_embeddings"].astype(np.float32)
        crop_emb = data["crop_embeddings"].astype(np.float32)
        bbox_ft  = data["bbox_feats"].astype(np.float32)
        dist_ft  = data["distances_feet"].astype(np.float32)

        X = np.hstack([img_emb, crop_emb, bbox_ft]).astype(np.float32)

        flight_name = os.path.splitext(os.path.basename(path))[0]
        print(f"[load] {flight_name:30s} N={len(dist_ft):5d}  "
              f"dist[{dist_ft.min():.2f}, {dist_ft.max():.2f}] ft")

        test_flights.append({
            "flight_name": flight_name,
            "X": X,
            "y_cont": dist_ft,
        })

    return train_flights, val_flights, test_flights


def compute_global_bin_edges(train_flights, val_flights, test_flights, bin_step):
    train_d = np.concatenate([f["y_cont"] for f in train_flights]).astype(np.float32)
    val_d = np.concatenate([f["y_cont"] for f in val_flights]).astype(np.float32)
    test_d = np.concatenate([f["y_cont"] for f in test_flights]).astype(np.float32)
    
    all_d = np.concatenate([train_d, val_d, test_d]).astype(np.float32)
    
    y_min = float(all_d.min())
    y_max = float(all_d.max())

    y_min_aligned = np.floor(y_min / bin_step) * bin_step
    y_max_aligned = np.ceil(y_max / bin_step) * bin_step

    bin_edges = np.arange(y_min_aligned,
                          y_max_aligned + bin_step,
                          bin_step,
                          dtype=np.float32)

    print(f"[bins] raw min/max    {y_min:.2f}/{y_max:.2f} ft")
    print(f"[bins] aligned range  {y_min_aligned:.2f}/{y_max_aligned:.2f} ft")
    print(f"[bins] bin_step={bin_step} -> num_classes={len(bin_edges)-1}")

    return bin_edges


def distances_to_classes(distances_ft, bin_edges):
    cls = np.digitize(distances_ft, bins=bin_edges, right=False) - 1
    cls = np.clip(cls, 0, len(bin_edges)-2).astype(np.int64)
    return cls


def split_one_flight(flight_dict, bin_edges):
    X = flight_dict["X"]
    y_cont = flight_dict["y_cont"]
    y_cls = distances_to_classes(y_cont, bin_edges)

    return (X, y_cont, y_cls)


def build_global_pools(train_flights, val_flights, test_flights, bin_edges):
    train_X_list = []
    train_y_cont_list = []
    train_y_cls_list = []

    val_X_list = []
    val_y_cont_list = []
    val_y_cls_list = []

    test_X_list = []
    test_y_cont_list = []
    test_y_cls_list = []

    for f in train_flights:
        (X, y_cont, y_cls) = split_one_flight(f, bin_edges)

        print(f"[split] {f['flight_name']:30s} "
              f"train {len(y_cont):5d}")

        train_X_list.append(X)
        train_y_cont_list.append(y_cont)
        train_y_cls_list.append(y_cls)

    for f in val_flights:
        (X, y_cont, y_cls) = split_one_flight(f, bin_edges)

        print(f"[split] {f['flight_name']:30s} "
              f"val {len(y_cont):5d}")

        val_X_list.append(X)
        val_y_cont_list.append(y_cont)
        val_y_cls_list.append(y_cls)

    for f in test_flights:
        (X, y_cont, y_cls) = split_one_flight(f, bin_edges)

        print(f"[split] {f['flight_name']:30s} "
              f"test {len(y_cont):5d}")

        test_X_list.append(X)
        test_y_cont_list.append(y_cont)
        test_y_cls_list.append(y_cls)

    X_train_all = np.vstack(train_X_list).astype(np.float32)
    y_train_cont_all = np.concatenate(train_y_cont_list).astype(np.float32)
    y_train_cls_all  = np.concatenate(train_y_cls_list).astype(np.int64)

    X_val_all = np.vstack(val_X_list).astype(np.float32)
    y_val_cont_all = np.concatenate(val_y_cont_list).astype(np.float32)
    y_val_cls_all  = np.concatenate(val_y_cls_list).astype(np.int64)

    X_test_all = np.vstack(test_X_list).astype(np.float32)
    y_test_cont_all = np.concatenate(test_y_cont_list).astype(np.float32)
    y_test_cls_all  = np.concatenate(test_y_cls_list).astype(np.int64)

    print(f"[global] train total {len(y_train_cont_all)}, ",
          f"val total {len(y_val_cont_all)}, ", f"test total {len(y_test_cont_all)}")

    return (
        X_train_all, y_train_cont_all, y_train_cls_all,
        X_val_all, y_val_cont_all, y_val_cls_all,
        X_test_all,  y_test_cont_all,  y_test_cls_all
    )
