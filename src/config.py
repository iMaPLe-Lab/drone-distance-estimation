import torch
import os

# paths
DATA_ROOT   = "/mnt/active_storage/Knut/LRDD_v3/test" #r"C:\Project\Distance Estimation\LRDDv3"
METADATA_DIR = "/mnt/active_storage/Knut/LRDD_v3/metadata/test"
EMBED_DIR   = "/mnt/active_storage/Knut/DroneRange/LRDDv3_embeds" #r"C:\Project\Distance Estimation\LRDDv3_embeds"
CHECKPOINT_DIR = "/mnt/active_storage/Knut/DroneRange/LRDDv3_checkpoints" #r"C:\Project\Distance Estimation\checkpoints"

os.makedirs(EMBED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# extraction
BATCH_EXTRACT = 32   # batch size for feature extraction

# binning / split
BIN_STEP   = 20.0    # or 10.0 if you want hi-res mode
TEST_SIZE  = 0.30    # 30% holdout per flight

# training
BATCH_TRAIN   = 64
BATCH_VAL     = 64
BATCH_TEST    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
EPOCHS        = 30
PATIENCE      = 5

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
