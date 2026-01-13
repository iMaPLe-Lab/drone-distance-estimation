import os
import glob
from typing import List, Dict, Optional


def _find_csv_for_flight(
    metadata_root: str,
    date_str: str,
    flight_id: str
) -> Optional[str]:
    """
    Enforce date + flight_id match in the CSV name.

    Only accept CSVs that START WITH "{date_str}_{flight_id}_".
    Example valid:
        08-08-2025_MAX_0014_metadata.csv
    """
    wanted_prefix = f"{date_str}_{flight_id}_"

    candidates = glob.glob(os.path.join(metadata_root, "*_metadata.csv"))

    exact = [
        c for c in candidates
        if os.path.basename(c).startswith(wanted_prefix)
    ]

    if len(exact) == 1:
        return exact[0]

    if len(exact) > 1:
        print(
            f"[scan_dataset] !! {date_str}/{flight_id}: multiple CSVs matched {wanted_prefix}, "
            f"candidates={ [os.path.basename(x) for x in exact] } → skip"
        )
        return None

    # no match
    return None


def scan_dataset(
    root: str,
    metadata_dir: str = "metadata",
    flight_patterns: List[str] = ("DJI_*", "IMG_*", "MAX_*", "PXL_*")
) -> List[Dict]:
    """
    Walk the dataset root and build a manifest of usable flights.

    Returns a list of dicts:
        {
            "date": "04-18-2025",
            "flight_id": "DJI_0080",
            "img_dir":   ".../04-18-2025/DJI_0080/images",
            "label_dir": ".../04-18-2025/DJI_0080/labels",
            "csv_path":  ".../metadata/04-18-2025_DJI_0080_metadata.csv"
        }
    """

    manifest: List[Dict] = []

    # metadata_root = os.path.join(root, metadata_subdir)
    metadata_root = metadata_dir
    if not os.path.isdir(metadata_root):
        raise RuntimeError(f"metadata folder not found at {metadata_root}")

    # 1. loop over date folders under root
    for entry in sorted(os.listdir(root)):
        date_path = os.path.join(root, entry)
        if not os.path.isdir(date_path):
            continue

        # date dirs look like "MM-DD-YYYY" (two dashes)
        if entry.count("-") != 2:
            continue

        date_str = entry  # e.g. "08-08-2025"

        # 2. gather flight dirs (DJI_*, MAX_*, etc.)
        flight_dirs = []
        for pat in flight_patterns:
            flight_dirs.extend(
                d for d in glob.glob(os.path.join(date_path, pat))
                if os.path.isdir(d)
            )

        if not flight_dirs:
            print(f"[scan_dataset] note: no flights found under {date_str}")
            continue

        # 3. validate each flight dir
        for flight_dir in sorted(flight_dirs):
            flight_id = os.path.basename(flight_dir)  # e.g. "MAX_0014"

            img_dir = os.path.join(flight_dir, "images")
            if not os.path.isdir(img_dir):
                print(f"[scan_dataset] !! {date_str}/{flight_id}: missing images/ at {img_dir} → skip")
                continue

            # labels could be "labels" or "final_labels"
            label_dir = None
            for cand in ("labels", "final_labels"):
                cand_dir = os.path.join(flight_dir, cand)
                if os.path.isdir(cand_dir):
                    label_dir = cand_dir
                    break
            if label_dir is None:
                print(f"[scan_dataset] !! {date_str}/{flight_id}: no labels or final_labels → skip")
                continue

            # 4. find the matching metadata csv for THIS EXACT (date, flight)
            csv_path = _find_csv_for_flight(metadata_root, date_str, flight_id)
            if csv_path is None:
                print(f"[scan_dataset] !! {date_str}/{flight_id}: no matching CSV (strict) → skip")
                continue

            # 5. if we got here, we trust this record
            manifest.append({
                "date": date_str,
                "flight_id": flight_id,
                "img_dir": img_dir,
                "label_dir": label_dir,
                "csv_path": csv_path,
            })

    return manifest
