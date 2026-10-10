from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\rojin_lid_ai")

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "TinplateLidDATA2024"
)

RESULTS_PATH = PROJECT_ROOT / "results"


# ============================================================
# TRAINING CONFIG
# ============================================================

MODEL_NAME = "yolo11n-cls.pt"

IMAGE_SIZE = 640
EPOCHS = 40
BATCH_SIZE = 16
PATIENCE = 8
WORKERS = 4
SEED = 42


def check_environment():

    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training should not continue on CPU."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(f"GPU             : {gpu_name}")
    print(f"GPU memory      : {gpu_memory:.2f} GB")

    print(f"\nDataset path:\n{DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    print("\nEnvironment OK.")


def train():

    check_environment()

    print("\n" + "=" * 70)
    print("LOADING YOLO11N CLASSIFICATION MODEL")
    print("=" * 70)

    model = YOLO(MODEL_NAME)

    print("\nModel loaded.")
    print("Starting training...\n")

    results = model.train(

        # Dataset
        data=str(DATASET_PATH),

        # Training
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        # GPU
        device=0,

        # Early stopping
        patience=PATIENCE,

        # DataLoader
        workers=WORKERS,

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Pretrained ImageNet/YOLO weights
        pretrained=True,

        # Mixed precision
        amp=True,

        # Outputs
        project=str(RESULTS_PATH),
        name="yolo11n_cls_baseline",

        save=True,
        plots=True,

        # Do not overwrite an old experiment
        exist_ok=False,
    )

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        "\nBest model should be available at:\n"
        f"{RESULTS_PATH}\\yolo11n_cls_baseline\\weights\\best.pt"
    )

    return results


if __name__ == "__main__":
    train()