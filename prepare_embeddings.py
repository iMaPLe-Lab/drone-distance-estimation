# prepare_embeddings.py
import os
import glob
import numpy as np
from extract_embeddings import extract_npz

ARRAY_KEYS = {"full", "crop", "bbox", "y"}  # only merge arrays


def merge_npz_files(npz_paths, out_path):
    """Merge multiple .npz files into one combined dataset (arrays only)."""
    merged = {}
    first = True
    for path in npz_paths:
        data = np.load(path, allow_pickle=False)
        if first:
            for k in data.files:
                if k in ARRAY_KEYS:
                    merged[k] = data[k]
            first = False
        else:
            for k in data.files:
                if k in ARRAY_KEYS:
                    merged[k] = np.concatenate([merged[k], data[k]], axis=0)
    np.savez_compressed(out_path, **merged)
    print(f"\nMerged {len(npz_paths)} files → {out_path}\n")


def _parse_meta_filename(fname: str):
    """
    Handles both single-flight (e.g. '04-11-2025_DJI_0003_metadata.csv')
    and combined-flight (e.g. '05-04-2025_DJI_0118_0121_metadata.csv') cases.
    Returns (date, flight_id or None).
    """
    stem = os.path.basename(fname).replace("_metadata.csv", "")
    parts = stem.split("_")

    # date is always first token
    date_only = parts[0]
    flight = None

    # find the 'DJI' or 'IMG' token and take everything from there up to the end
    if len(parts) >= 3:
        for i, token in enumerate(parts):
            if token.upper() in ("DJI", "IMG"):
                flight = "_".join(parts[i:])  # captures e.g. DJI_0118 or DJI_0118_0121
                break

    return date_only, flight


def summarize_flight_folder(flight_dir):
    """Count JPGs in images/ and TXT in labels/ inside one flight folder."""
    img_dir = os.path.join(flight_dir, "images")
    lbl_dir = os.path.join(flight_dir, "labels")

    if not os.path.isdir(img_dir):
        return (0, 0, 0, "images/ missing")
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

    all_labels = []
    if os.path.isdir(lbl_dir):
        all_labels = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]

    img_basenames = set(os.path.splitext(f)[0] for f in all_images)
    label_basenames = set(os.path.splitext(f)[0] for f in all_labels)
    matched = len(img_basenames & label_basenames)

    if not all_labels:
        note = "no labels"
    elif matched == len(all_images):
        note = "all matched"
    else:
        note = "mismatch"

    return len(all_images), len(all_labels), matched, note


def prepare_split(base_path, split, arch):
    """
    Walk LRDD structure:
      <base>/<split>/<date>/<FLIGHT>/images, labels
    For each metadata CSV in <base>/metadata/<split>, extract per-flight and merge.
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

    if not os.path.isdir(meta_dir):
        print(f"Missing metadata dir: {meta_dir}")
        return None

    for csv_file in sorted(os.listdir(meta_dir)):
        if not csv_file.endswith(".csv"):
            continue

        csv_path = os.path.join(meta_dir, csv_file)

        # parse date and (optional) flight id from metadata filename
        date_only, wanted_flight = _parse_meta_filename(csv_file)

        date_dir = os.path.join(split_dir, date_only)
        if not os.path.isdir(date_dir):
            print(f"skip {date_only}: missing date folder at {date_dir}")
            continue

        # collect flight folders; support DJI_*, IMG_*, MAX_*
        flight_dirs = []
        for pat in ("DJI_*", "IMG_*", "MAX_*"):
            flight_dirs.extend(
                d for d in glob.glob(os.path.join(date_dir, pat)) if os.path.isdir(d)
            )
        flight_dirs = sorted(flight_dirs)

        if not flight_dirs:
            print(f"skip {date_only}: no DJI_*/IMG_*/MAX_* flights in {date_dir}")
            continue

        # if metadata file names a specific flight, try restricting to it
        if wanted_flight is not None:
            filtered = [d for d in flight_dirs if os.path.basename(d) == wanted_flight]
            if filtered:
                flight_dirs = filtered
            else:
                print(f"  note: {wanted_flight} not found under {date_dir}; processing all flights for this date.")

        for flight_dir in flight_dirs:
            img_dir = os.path.join(flight_dir, "images")
            lbl_dir = os.path.join(flight_dir, "labels")

            out_path = os.path.join(
                out_dir, f"{split}_{date_only}_{os.path.basename(flight_dir)}.npz"
            )

            if os.path.exists(out_path):
                print(f"✓ Skipping {out_path} (already exists)\n")
                npz_files.append(out_path)

                # count rows from existing NPZ so summary reflects it
                try:
                    with np.load(out_path, allow_pickle=False) as d:
                        n_rows = int(d["y"].shape[0])
                except Exception:
                    # fallback: estimate from folder if NPZ is unreadable
                    n_imgs, _, _, _ = summarize_flight_folder(flight_dir)
                    n_rows = n_imgs

                total_flights += 1
                total_images  += n_rows
                total_labels  += n_rows
                continue  

            n_imgs, n_lbls, matched, note = summarize_flight_folder(flight_dir)
            print(f"{date_only}/{os.path.basename(flight_dir)}: {n_imgs} imgs, {n_lbls} labels, {matched} matched → {note}")
            if n_imgs == 0:
                print(f"Skipping {flight_dir} (no images)\n")
                continue

            labels_folder_to_use = lbl_dir if os.path.isdir(lbl_dir) else None

            extract_npz(
                csv_file=csv_path,                
                img_folder=img_dir,
                labels_folder=labels_folder_to_use,
                arch=arch,
                out_path=out_path,
                batch_size=64,
                num_workers=0,
            )

            # count rows from newly written NPZ
            with np.load(out_path, allow_pickle=False) as d:
                n_rows = int(d["y"].shape[0])

            total_flights += 1
            total_images  += n_rows
            total_labels  += n_rows
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
