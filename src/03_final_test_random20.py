from pathlib import Path
from ultralytics import YOLO

import cv2
import csv
import json
import time
import shutil

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt



# ===============================
# PATH
# ===============================

ROOT = Path(__file__).resolve().parents[1]


MODEL_PATH = (
    ROOT /
    "results" /
    "yolo11n_cls_masked_random80_v2" /
    "weights" /
    "best.pt"
)


TEST_PATH = (
    ROOT /
    "data" /
    "processed" /
    "TinplateLidDATA2024_masked_random20" /
    "test"
)


OUTPUT = (
    ROOT /
    "results" /
    "random20_final_test"
)



# ===============================
# Threshold
# ===============================

GOOD_THRESHOLD = 0.2
REVIEW_THRESHOLD = 0.5



# ===============================
# Decision
# ===============================

def decide(p_defect):

    if p_defect < GOOD_THRESHOLD:
        return "good"

    elif p_defect <= REVIEW_THRESHOLD:
        return "review"

    else:
        return "defective"



# ===============================
# Main
# ===============================


def main():

    print("="*60)
    print("ROJIN LID AI FINAL RANDOM20 TEST")
    print("="*60)



    # create folders

    for folder in [
        "good",
        "defective",
        "review"
    ]:

        (OUTPUT / folder).mkdir(
            parents=True,
            exist_ok=True
        )



    print("Loading model...")

    model = YOLO(
        str(MODEL_PATH)
    )



    images = []


    for label in [
        "good",
        "defective"
    ]:

        for img in (
            TEST_PATH / label
        ).glob("*"):

            images.append(
                (
                    img,
                    label
                )
            )



    print(
        "Total images:",
        len(images)
    )



    start = time.time()



    predictions = []


    y_true=[]

    y_pred=[]



    # ===============================
    # Batch inference
    # ===============================


    batch_size = 32


    for i in range(
        0,
        len(images),
        batch_size
    ):


        batch = images[i:i+batch_size]


        paths = [
            str(x[0])
            for x in batch
        ]


        results = model.predict(

            paths,

            device=0,

            batch=batch_size,

            half=True,

            verbose=False

        )



        for item,result in zip(
            batch,
            results
        ):


            image_path,true_label = item


            probs = result.probs


            names = result.names



            good_id = None
            defect_id = None



            for k,v in names.items():

                if v=="good":
                    good_id=k

                if v=="defective":
                    defect_id=k



            p_good=float(
                probs.data[good_id]
            )


            p_defect=float(
                probs.data[defect_id]
            )



            decision = decide(
                p_defect
            )



            predictions.append(

                [
                    image_path.name,
                    true_label,
                    round(p_good,5),
                    round(p_defect,5),
                    decision
                ]

            )



            y_true.append(
                true_label
            )


            y_pred.append(
                decision
            )



            shutil.copy2(

                image_path,

                OUTPUT /
                decision /
                image_path.name

            )



    end=time.time()



    total_time=end-start


    fps=len(images)/total_time



    # ===============================
    # CSV
    # ===============================


    with open(
        OUTPUT/"predictions.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:


        writer=csv.writer(f)


        writer.writerow(

            [
                "image",
                "true_label",
                "p_good",
                "p_defect",
                "decision"
            ]

        )


        writer.writerows(
            predictions
        )




    # ===============================
    # Metrics
    # ===============================


    metrics={

        "images":
            len(images),

        "total_time_seconds":
            round(total_time,3),

        "fps":
            round(fps,2),


        "GOOD_predictions":
            y_pred.count("good"),

        "DEFECT_predictions":
            y_pred.count("defective"),

        "REVIEW_predictions":
            y_pred.count("review"),



        "accuracy_all":
            round(
                accuracy_score(
                    y_true,
                    y_pred
                ),
               4
            ),


        "precision_macro":
            round(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
               4
            ),


        "recall_macro":
            round(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
               4
            ),


        "f1_macro":
            round(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                ),
               4
            ),


        "false_accept":
            sum(
                1
                for t,p in zip(
                    y_true,
                    y_pred
                )
                if t=="defective"
                and
                p=="good"
            ),


        "false_reject":
            sum(
                1
                for t,p in zip(
                    y_true,
                    y_pred
                )
                if t=="good"
                and
                p=="defective"
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
                y_true,
                y_pred
            )

        )



    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "good",
            "defective",
            "review"
        ]
    )


    plt.figure(
        figsize=(6,5)
    )


    plt.imshow(cm)


    plt.xticks(
        range(3),
        [
            "good",
            "defective",
            "review"
        ]
    )


    plt.yticks(
        range(3),
        [
            "good",
            "defective",
            "review"
        ]
    )


    plt.colorbar()


    plt.savefig(
        OUTPUT/"confusion_matrix.png"
    )


    print("="*60)

    print(
        json.dumps(
            metrics,
            indent=4
        )
    )

    print("="*60)



if __name__=="__main__":

    main()