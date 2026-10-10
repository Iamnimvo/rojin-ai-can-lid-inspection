
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(r"D:\rojin_lid_ai")

MODEL_PATH = (
    ROOT
    / "results"
    / "yolo11n_cls_baseline"
    / "weights"
    / "best.pt"
)

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    print("=" * 50)
    print("ROJIN - MODEL INSPECTION")
    print("=" * 50)

    # Load our locally trained checkpoint
    model = YOLO(str(MODEL_PATH))

    print("\nModel task:", model.task)
    print("Model classes:", model.names)

    print("\nCUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nModel architecture:")
    model.info(detailed=False)

    print("\nMODEL LOADED SUCCESSFULLY")


if __name__ == "__main__":
    main()