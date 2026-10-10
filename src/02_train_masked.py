from pathlib import Path
from ultralytics import YOLO
import torch


ROOT = Path(__file__).resolve().parents[1]


DATASET = (
    ROOT
    / "data"
    / "processed"
    / "TinplateLidDATA2024_masked_random80"
)


MODEL = "yolo11n-cls.pt"


RESULTS = ROOT / "results"


print("=" * 60)
print("ROJIN LID AI - YOLO11n CLASSIFICATION TRAINING")
print("=" * 60)


print("Dataset:")
print(DATASET)


print()

print("Torch:")
print(torch.__version__)

print(
    "CUDA:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


print("=" * 60)


# Dataset check

for split in ["train", "val"]:

    print("\n", split.upper())

    for cls in ["good", "defective"]:

        count = len(
            list(
                (
                    DATASET
                    /
                    split
                    /
                    cls
                ).glob("*")
            )
        )

        print(
            cls,
            ":",
            count
        )


print("=" * 60)


model = YOLO(MODEL)


model.train(

    data=str(DATASET),

    epochs=100,

    imgsz=640,

    batch=32,

    device=0,

    workers=0,

    pretrained=True,

    optimizer="AdamW",

    lr0=0.001,

    patience=20,

    amp=True,

    cache=False,

    project=str(RESULTS),

    name="yolo11n_cls_masked_random80_v2",

    exist_ok=False,

    plots=True

)


print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)