from pathlib import Path
from ultralytics import YOLO
import cv2
import csv
import json
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==========================
# PATHS
# ==========================

ROOT = Path(__file__).resolve().parents[1]


MODEL_PATH = (
    ROOT
    /
    "results"
    /
    "yolo11n_cls_masked_random80_v2"
    /
    "weights"
    /
    "best.pt"
)


TEST_DIR = (
    ROOT
    /
    "data"
    /
    "processed"
    /
    "TinplateLidDATA2024_masked_random20"
    /
    "test"
)


OUTPUT = (
    ROOT
    /
    "results"
    /
    "test_random20_evaluation"
)


PREVIEW_DIR = OUTPUT / "previews"



# ==========================
# THRESHOLDS
# ==========================

GOOD_THRESHOLD = 0.2
DEFECT_THRESHOLD = 0.5



# ==========================
# Decision
# ==========================

def decide(p_defect):

    if p_defect < GOOD_THRESHOLD:
        return "GOOD"

    elif p_defect <= DEFECT_THRESHOLD:
        return "REVIEW"

    else:
        return "DEFECT"



# ==========================
# Save image preview
# ==========================

def save_preview(
        image_path,
        decision,
        p_good,
        p_defect):


    folder = PREVIEW_DIR / decision

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


    img = cv2.imread(
        str(image_path)
    )


    text = [
        f"Prediction: {decision}",
        f"Good: {p_good:.3f}",
        f"Defect: {p_defect:.3f}"
    ]


    y = 40

    for t in text:

        cv2.putText(
            img,
            t,
            (20,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        y += 40



    cv2.imwrite(
        str(folder / image_path.name),
        img
    )



# ==========================
# MAIN
# ==========================

def main():


    print("="*60)
    print("ROJIN LID AI - RANDOM20 TEST")
    print("="*60)


    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )


    model = YOLO(
        str(MODEL_PATH)
    )


    rows = []

    y_true = []

    y_pred = []



    images = []


    for cls in ["good","defective"]:

        folder = TEST_DIR / cls


        for img in folder.glob("*"):

            images.append(
                (
                    img,
                    cls
                )
            )


    print("Total images:",len(images))


    start = time.time()



    # ==========================
    # Inference
    # ==========================


    for image_path,true_label in images:


        result = model(
            str(image_path),
            verbose=False
        )[0]


        probs = result.probs

        names = result.names



        good_id = None

        defect_id = None


        for k,v in names.items():

            if v == "good":
                good_id = k

            if v == "defective":
                defect_id = k



        p_good = float(
            probs.data[good_id]
        )


        p_defect = float(
            probs.data[defect_id]
        )



        decision = decide(
            p_defect
        )



        rows.append(
            [
                image_path.name,
                true_label,
                round(p_good,5),
                round(p_defect,5),
                decision
            ]
        )



        y_true.append(
            true_label.upper()
        )


        y_pred.append(
            decision
        )



        save_preview(
            image_path,
            decision,
            p_good,
            p_defect
        )



    end = time.time()


    total_time = end-start


    fps = len(images)/total_time



    # ==========================
    # CSV
    # ==========================

    with open(
        OUTPUT/"predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.writer(f)


        writer.writerow(
            [
                "image",
                "true_label",
                "p_good",
                "p_defect",
                "decision"
            ]
        )


        writer.writerows(rows)



    # ==========================
    # Metrics
    # ==========================


    eval_true = []

    eval_pred = []


    for t,p in zip(y_true,y_pred):

        if p != "REVIEW":

            eval_true.append(t)

            eval_pred.append(p)



    metrics = {

        "total_images":
            len(images),

        "total_time_seconds":
            round(total_time,3),

        "fps":
            round(fps,2),

        "review_count":
            y_pred.count("REVIEW"),

        "accuracy_without_review":
            round(
                accuracy_score(
                    eval_true,
                    eval_pred
                ),
                4
            ),

        "precision_macro":
            round(
                precision_score(
                    eval_true,
                    eval_pred,
                    average="macro",
                    zero_division=0
                ),
                4
            ),


        "recall_macro":
            round(
                recall_score(
                    eval_true,
                    eval_pred,
                    average="macro",
                    zero_division=0
                ),
                4
            ),


        "f1_macro":
            round(
               f1_score(
                    eval_true,
                    eval_pred,
                    average="macro",
                    zero_division=0
                ),
                4
            )

    }



    with open(
        OUTPUT/"metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )



    with open(
        OUTPUT/"classification_report.txt",
        "w"
    ) as f:

        f.write(
            classification_report(
                eval_true,
                eval_pred
            )
        )



    print("="*60)

    print(
        json.dumps(
            metrics,
            indent=4
        )
    )

    print("="*60)



if __name__ == "__main__":

    main()