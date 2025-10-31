import sys, os

# absolute path to the project root (the folder that contains /src and /scripts)
PROJECT_ROOT = "/home/kp3275@drexel.edu/DroneRange/drone-distance-estimation" #r"C:\Project\Distance Estimation"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    EMBED_DIR, BIN_STEP, TEST_SIZE,
    BATCH_TRAIN, BATCH_VAL, BATCH_TEST,
    LR, WEIGHT_DECAY, EPOCHS, PATIENCE,
    DEVICE, CHECKPOINT_DIR
)
from src.binning import (
    load_all_flights,
    compute_global_bin_edges,
    build_global_pools,
)
from src.model_head import build_classifier_head
from src.train_eval import (
    make_loaders,
    train_classifier,
    eval_distance_metrics,
)

def main():
    # 1. load flight .npz files
    train_flights, val_flights, test_flights = load_all_flights(EMBED_DIR)
    print("\n" + "="*80)
    print("[stage] finished loading per-flight .npz files")
    print("="*80 + "\n")

    # 2. compute bins
    bin_edges = compute_global_bin_edges(train_flights, val_flights, test_flights, BIN_STEP)
    num_bins = len(bin_edges) - 1
    
    print("\n" + "-"*80)
    print("[stage] global bin edges computed")
    print(f"  BIN_STEP={BIN_STEP} ft -> num_bins={num_bins}")
    print("-"*80 + "\n")

    # 3. split per flight & merge
    (X_train_all,
     y_train_cont_all,
     y_train_cls_all,
     X_val_all,
     y_val_cont_all,
     y_val_cls_all,
     X_test_all,
     y_test_cont_all,
     y_test_cls_all) = build_global_pools(train_flights, val_flights, test_flights, bin_edges)

    input_dim = X_train_all.shape[1]
    print("\n" + "="*80)
    print("[stage] finished per-flight splits + merged global pools")
    print(f"  input_dim: {input_dim}")
    print(f"  train total: {len(y_train_cont_all)}")
    print(f"  val total: {len(y_val_cont_all)}")
    print(f"  test  total: {len(y_test_cont_all)}")
    print("="*80 + "\n")

    # 4. loaders
    train_loader, val_loader, test_loader = make_loaders(
        X_train_all, y_train_cls_all,
        X_val_all, y_val_cls_all,
        X_test_all,  y_test_cls_all,
        BATCH_TRAIN, BATCH_VAL, BATCH_TEST,
        DEVICE
    )

    # 5. model head
    model = build_classifier_head(input_dim, num_bins).to(DEVICE)
    
    print("\n" + "#"*80)
    print("[train] starting training / validation loop")
    print("#"*80 + "\n")

    # 6. train
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_classifier.pt")
    train_classifier(
        model,
        train_loader,
        val_loader,
        test_loader,
        DEVICE,
        LR,
        WEIGHT_DECAY,
        EPOCHS,
        PATIENCE,
        ckpt_path
    )

    # 7. reload best model before eval
    best_model = build_classifier_head(input_dim, num_bins).to(DEVICE)
    best_model.load_state_dict(
        __import__("torch").load(ckpt_path, map_location=DEVICE)
    )
    
    print("\n" + "#"*80)
    print("[eval] reloading best checkpoint and computing distance metrics")
    print("#"*80 + "\n")


    # 8. distance metrics
    eval_distance_metrics(
        best_model,
        X_test_all,
        y_test_cont_all,
        bin_edges,
        DEVICE
    )
    
    print("\n" + "="*80)
    print("[done] pipeline finished successfully")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
