import os
import numpy as np
from extract_embeddings import extract_npz


def merge_npz_files(npz_paths, out_path):
    """Merge multiple .npz files into one combined dataset."""
    merged = {}
    for i, path in enumerate(npz_paths):
        data = np.load(path)
        if i == 0:
            merged = {k: data[k] for k in data.files}
        else:
            for k in data.files:
                merged[k] = np.concatenate([merged[k], data[k]], axis=0)
    np.savez_compressed(out_path, **merged)
    print(f"\n Merged {len(npz_paths)} files → {out_path}\n")


def summarize_flight_folder(img_folder, labels_folder):
    """Check how many images and label files exist and match."""
    if not os.path.exists(img_folder):
        return (0, 0, 0, "image folder missing!")
    if not os.path.exists(labels_folder):
        return (0, 0, 0, "labels folder missing!")

    all_images = [f for f in os.listdir(img_folder) if f.lower().endswith(".jpg")]
    all_labels = [f for f in os.listdir(labels_folder) if f.lower().endswith(".txt")]

    # match by frame_XXXXX naming
    img_basenames = set(os.path.splitext(f)[0] for f in all_images)
    label_basenames = set(os.path.splitext(f)[0] for f in all_labels)
    matched = len(img_basenames.intersection(label_basenames))
    note = "all matched" if matched == len(all_images) else "mismatch"

    return len(all_images), len(all_labels), matched, note


def prepare_split(base_path, split="train", arch="resnet50"):
    """
    Go through all metadata CSVs in LRDD-V3/metadata/{split},
    match them to corresponding date folders in LRDD-V3/{split},
    extract embeddings, and merge them.
    """
    meta_dir = os.path.join(base_path, "metadata", split)
    split_dir = os.path.join(base_path, split)
    out_dir = os.path.join(base_path, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nPreparing split: {split.upper()}")
    print(f"Metadata directory: {meta_dir}")
    print(f"Image root: {split_dir}\n")

    npz_files = []
    total_flights = 0
    total_images = 0
    total_labels = 0

    for csv_file in sorted(os.listdir(meta_dir)):
        if not csv_file.endswith(".csv"):
            continue

        date_prefix = csv_file.split("_metadata")[0]
        csv_path = os.path.join(meta_dir, csv_file)
        img_folder = os.path.join(split_dir, date_prefix)
        labels_folder = os.path.join(img_folder, "final_labels")
        out_path = os.path.join(out_dir, f"{split}_{date_prefix}.npz")

        # verify folder consistency
        n_imgs, n_lbls, matched, note = summarize_flight_folder(img_folder, labels_folder)
        print(f"{date_prefix}: {n_imgs} imgs, {n_lbls} labels, {matched} matched → {note}")

        if n_imgs == 0 or n_lbls == 0:
            print(f"Skipping {date_prefix} due to missing data.\n")
            continue

        total_flights += 1
        total_images += n_imgs
        total_labels += n_lbls

        # extract embeddings
        extract_npz(csv_path, img_folder, labels_folder,
                    arch=arch, out_path=out_path)
        npz_files.append(out_path)

    if not npz_files:
        print(f"No valid flights found for split: {split}")
        return None

    merged_out = os.path.join(out_dir, f"{split}_ALL.npz")
    merge_npz_files(npz_files, merged_out)

    print(f"Summary for {split.upper()} split:")
    print(f"   - Flights processed : {total_flights}")
    print(f"   - Total images      : {total_images}")
    print(f"   - Total labels      : {total_labels}\n")

    return merged_out
