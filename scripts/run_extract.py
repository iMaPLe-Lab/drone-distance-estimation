import sys
# sys.path.append(r"C:\Project\Distance Estimation")  # so 'src' is importable
sys.path.append("/home/kp3275@drexel.edu/DroneRange/drone-distance-estimation")  # so 'src' is importable

from src.config import DATA_ROOT, METADATA_DIR, EMBED_DIR, BATCH_EXTRACT, DEVICE
from src.index_dataset import scan_dataset
from src.extract_features import extract_all_flights

# you can change this to "resnet50" / "resnet101"
BACKBONE = "resnet18"

def main():
    manifest = scan_dataset(DATA_ROOT, METADATA_DIR)
    # optional: pandas printout / sanity
    print(f"[run_extract] flights found: {len(manifest)}")
    # extract
    extract_all_flights(
        manifest_df = __import__("pandas").DataFrame(manifest),
        batch_size  = BATCH_EXTRACT,
        device      = DEVICE,
        out_dir     = EMBED_DIR,
        backbone_name = BACKBONE,
    )

if __name__ == "__main__":
    main()
