
from pathlib import Path
import csv

import torch
from ultralytics import YOLO


ROOT = Path(r"D:\rojin_lid_ai")

MODEL_PATH = (
    ROOT / "results"
    / "yolo11n_cls_baseline"
    / "weights"
    / "best.pt"
)

TEST_DIR = ROOT / "data" / "rojin_test_png"

OUTPUT_DIR = ROOT / "results" / "external_test"

LOW_THRESHOLD = 0.15
HIGH_THRESHOLD = 0.45


def decide(p_defect):

    if p_defect <= LOW_THRESHOLD:
        return "GOOD"

    if p_defect >= HIGH_THRESHOLD:
        return "DEFECT"

    return "REVIEW"


def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    print("=" * 65)
    print("ROJIN - EXTERNAL IMAGE TEST")
    print("=" * 65)

    print("\nClasses:", model.names)

    device = 0 if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    class_map = {
        str(name).lower(): int(idx)
        for idx, name in model.names.items()
    }

    defect_idx = class_map["defective"]
    good_idx = class_map["good"]

    records = []

    for class_name in ["good", "defective"]:

        folder = TEST_DIR / class_name

        if not folder.exists():
            raise FileNotFoundError(folder)

        for image_path in sorted(folder.glob("*.png")):

            result = model.predict(
                source=str(image_path),
                imgsz=640,
                device=device,
                verbose=False
            )[0]

            probabilities = result.probs.data

            p_good = float(probabilities[good_idx].item())
            p_defect = float(probabilities[defect_idx].item())

            predicted = (
                "defective" if p_defect >= 0.5 else "good"
            )

            decision = decide(p_defect)

            record = {
                "image": image_path.name,
                "expected": class_name,
                "predicted": predicted,
                "p_good": round(p_good, 6),
                "p_defect": round(p_defect, 6),
                "decision": decision,
                "correct": predicted == class_name
            }

            records.append(record)

            print("-" * 65)
            print("Image:", image_path.name)
            print("Expected:", class_name)
            print(f"P(GOOD): {p_good:.4f}")
            print(f"P(DEFECT): {p_defect:.4f}")
            print("Prediction:", predicted)
            print("Decision:", decision)
            print("Correct:", record["correct"])

    expected_total = 7

    if len(records) != expected_total:
        raise RuntimeError(
            f"Expected 7 images, found {len(records)}"
        )

    csv_path = OUTPUT_DIR / "external_predictions.csv"

    with open(
        csv_path, "w", newline="", encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)

    correct = sum(r["correct"] for r in records)

    print("\n" + "=" * 65)
    print("EXTERNAL TEST SUMMARY")
    print("=" * 65)

    print("Total images:", len(records))
    print("Correct predictions:", correct)
    print("Incorrect predictions:", len(records) - correct)
    print("Review cases:",
          sum(r["decision"] == "REVIEW" for r in records))

    print("\nResults saved to:")
    print(csv_path)


if __name__ == "__main__":
    main()