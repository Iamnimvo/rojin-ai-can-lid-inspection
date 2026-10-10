
from pathlib import Path
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ======================================================
# PROJECT CONFIGURATION
# ======================================================

ROOT = Path(r"D:\rojin_lid_ai")

MODEL_PATH = (
    ROOT
    / "results"
    / "yolo11n_cls_baseline"
    / "weights"
    / "best.pt"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "raw"
    / "TinplateLidDATA2024"
)

TEST_PATH = DATASET_PATH / "test"

OUTPUT_PATH = ROOT / "results" / "evaluation_n"

IMAGE_SIZE = 640
DEVICE = 0

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


# ======================================================
# MAIN EVALUATION
# ======================================================

def main():

    print("=" * 60)
    print("ROJIN - YOLO11N CLASSIFICATION EVALUATION")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("\nLoading model...")

    model = YOLO(str(MODEL_PATH))

    print("Model:", MODEL_PATH.name)
    print("Task:", model.task)
    print("Classes:", model.names)
    print("GPU:", torch.cuda.get_device_name(0))

    # Do not assume class indices.
    class_indices = {
        str(name).strip().lower(): int(index)
        for index, name in model.names.items()
    }

    if not {"good", "defective"}.issubset(class_indices):
        raise RuntimeError(
            f"Unexpected classes: {model.names}"
        )

    defect_index = class_indices["defective"]

    # Evaluation labels:
    # 0 = good
    # 1 = defective

    image_records = []

    for class_name in ["good", "defective"]:

        folder = TEST_PATH / class_name

        if not folder.exists():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        true_label = 1 if class_name == "defective" else 0

        for path in files:
            image_records.append((path, true_label))

    if not image_records:
        raise RuntimeError("No test images found.")

    print("\nTotal test images:", len(image_records))

    y_true = []
    y_pred = []

    output_rows = []
    inference_times = []

    torch.cuda.reset_peak_memory_stats(0)

    # ==================================================
    # RUN PREDICTIONS
    # ==================================================

    print("\nRunning predictions...\n")

    for index, (image_path, true_label) in enumerate(
        image_records, start=1
    ):

        result = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )[0]

        if result.probs is None:
            raise RuntimeError(
                f"Classification probabilities missing: {image_path}"
            )

        p_defect = float(
            result.probs.data[defect_index].item()
        )

        predicted_label = 1 if p_defect >= 0.5 else 0

        y_true.append(true_label)
        y_pred.append(predicted_label)

        inference_times.append(
            result.speed.get("inference", 0.0)
        )

        output_rows.append({
            "image": str(image_path.relative_to(TEST_PATH)),
            "true_label": "defective" if true_label else "good",
            "predicted_label": (
                "defective" if predicted_label else "good"
            ),
            "p_defect": round(p_defect, 6),
            "correct": int(true_label == predicted_label),
        })

        print(
            f"[{index}/{len(image_records)}] "
            f"P(defect)={p_defect:.4f} | "
            f"Correct={true_label == predicted_label}"
        )

    # ==================================================
    # METRICS
    # ==================================================

    accuracy = accuracy_score(y_true, y_pred)

    defect_precision = precision_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    defect_recall = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    # Rows: actual / Columns: predicted
    #             GOOD  DEFECTIVE
    # GOOD         TN       FP
    # DEFECTIVE    FN       TP

    tn, fp, fn, tp = cm.ravel()

    false_accept_rate = fn / (fn + tp) if (fn + tp) else 0
    false_reject_rate = fp / (fp + tn) if (fp + tn) else 0

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["good", "defective"],
        zero_division=0,
    )

    # ==================================================
    # PRINT RESULTS
    # ==================================================

    summary = (
        "\n" + "=" * 60 +
        "\nFINAL TEST RESULTS\n" +
        "=" * 60 +
        f"\nTotal images       : {len(y_true)}"
        f"\nAccuracy           : {accuracy:.4%}"
        f"\nDefect Precision   : {defect_precision:.4%}"
        f"\nDefect Recall      : {defect_recall:.4%}"
        f"\nFalse Accept       : {fn}"
        f"\nFalse Accept Rate  : {false_accept_rate:.4%}"
        f"\nFalse Reject       : {fp}"
        f"\nFalse Reject Rate  : {false_reject_rate:.4%}"
        f"\nMean inference time: {np.mean(inference_times):.3f} ms"
        f"\nPeak PyTorch VRAM  : "
        f"{torch.cuda.max_memory_reserved(0) / 1024**3:.3f} GB"
        "\n\nCONFUSION MATRIX:\n" +
        str(cm) +
        "\n\nCLASSIFICATION REPORT:\n" +
        report
    )

    print(summary)

    # ==================================================
    # SAVE CSV
    # ==================================================

    csv_path = OUTPUT_PATH / "predictions.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "true_label",
                "predicted_label",
                "p_defect",
                "correct",
            ]
        )

        writer.writeheader()
        writer.writerows(output_rows)

    # ==================================================
    # SAVE CONFUSION MATRIX
    # ==================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["GOOD", "DEFECTIVE"]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("YOLO11n - Test Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH / "confusion_matrix.png",
        dpi=200
    )

    plt.close()

    # ==================================================
    # SAVE REPORT
    # ==================================================

    with open(
        OUTPUT_PATH / "evaluation_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(summary)

    # ==================================================
    # FALSE ACCEPT DETAILS
    # ==================================================

    print("\nFALSE ACCEPT IMAGES:")

    false_accepts = [
        row for row in output_rows
        if row["true_label"] == "defective"
        and row["predicted_label"] == "good"
    ]

    if not false_accepts:
        print("None found in this test set.")

    for row in false_accepts:
        print(row["image"], row["p_defect"])

    print("\nEvaluation completed.")
    print("Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()