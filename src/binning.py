import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

def load_all_flights(embed_dir):
    flights = []
    npz_files = sorted(glob.glob(os.path.join(embed_dir, "*.npz")))
    print(f"[load_all_flights] found {len(npz_files)} npz files")

    for path in npz_files:
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

        flights.append({
            "flight_name": flight_name,
            "X": X,
            "y_cont": dist_ft,
        })

    return flights


def compute_global_bin_edges(flights, bin_step):
    all_d = np.concatenate([f["y_cont"] for f in flights]).astype(np.float32)
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


def split_one_flight(flight_dict, bin_edges, test_size):
    X = flight_dict["X"]
    y_cont = flight_dict["y_cont"]
    y_cls = distances_to_classes(y_cont, bin_edges)

    try:
        X_tr, X_te, ycont_tr, ycont_te, ycls_tr, ycls_te = train_test_split(
            X, y_cont, y_cls,
            test_size=test_size,
            random_state=42,
            stratify=y_cls
        )
    except ValueError:
        X_tr, X_te, ycont_tr, ycont_te, ycls_tr, ycls_te = train_test_split(
            X, y_cont, y_cls,
            test_size=test_size,
            random_state=42,
            stratify=None
        )

    return (X_tr, ycont_tr, ycls_tr), (X_te, ycont_te, ycls_te)


def build_global_pools(flights, bin_edges, test_size):
    train_X_list = []
    train_y_cont_list = []
    train_y_cls_list = []

    test_X_list = []
    test_y_cont_list = []
    test_y_cls_list = []

    for f in flights:
        (Xtr, ycont_tr, ycls_tr), (Xte, ycont_te, ycls_te) = split_one_flight(
            f, bin_edges, test_size
        )

        print(f"[split] {f['flight_name']:30s} "
              f"train {len(ycont_tr):5d} | test {len(ycont_te):5d}")

        train_X_list.append(Xtr)
        train_y_cont_list.append(ycont_tr)
        train_y_cls_list.append(ycls_tr)

        test_X_list.append(Xte)
        test_y_cont_list.append(ycont_te)
        test_y_cls_list.append(ycls_te)

    X_train_all = np.vstack(train_X_list).astype(np.float32)
    y_train_cont_all = np.concatenate(train_y_cont_list).astype(np.float32)
    y_train_cls_all  = np.concatenate(train_y_cls_list).astype(np.int64)

    X_test_all = np.vstack(test_X_list).astype(np.float32)
    y_test_cont_all = np.concatenate(test_y_cont_list).astype(np.float32)
    y_test_cls_all  = np.concatenate(test_y_cls_list).astype(np.int64)

    print(f"[global] train total {len(y_train_cont_all)}, "
          f"test total {len(y_test_cont_all)}")

    return (
        X_train_all, y_train_cont_all, y_train_cls_all,
        X_test_all,  y_test_cont_all,  y_test_cls_all
    )
