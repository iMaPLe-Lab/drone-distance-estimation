from prepare_embeddings import prepare_split
from train_regressor import train_distance_regressor

BASE_PATH = "LRDD-V3"

# Step 1: preprocess all flights → .npz
train_npz = prepare_split(BASE_PATH, "train")
val_npz   = prepare_split(BASE_PATH, "val")
test_npz  = prepare_split(BASE_PATH, "test")

# Step 2: train regressor
train_distance_regressor(train_npz, val_npz, test_npz)
