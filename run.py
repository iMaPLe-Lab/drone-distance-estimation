import os
from prepare_embeddings import prepare_split
from train_regressor import train_distance_regressor

# Resolve BASE relative to this file, so you can just put the LRDDv3 folder next to run.py
HERE = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(HERE, "LRDDv3")   # <-- note: 'v' not '-V-'

ARCH = "resnet18"        


# Step 1: preprocess all flights → .npz
train_npz = prepare_split(BASE_PATH, split="train", arch=ARCH)
val_npz   = prepare_split(BASE_PATH, split="val",   arch=ARCH)
test_npz  = prepare_split(BASE_PATH, split="test",  arch=ARCH)

# Step 2: train regressor
if not train_npz or not val_npz:
    print("\nERROR: Could not find or build train/val NPZ files. Check BASE and metadata folders.")
else:
    train_distance_regressor(train_npz, val_npz, test_npz)
