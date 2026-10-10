from pathlib import Path
import random
import shutil
import json


# =========================
# Paths
# =========================

ROOT = Path(__file__).resolve().parents[1]


DATASET = (
    ROOT
    / "data"
    / "processed"
    / "TinplateLidDATA2024_masked_random80"
)


TRAIN_DIR = DATASET / "train"
VAL_DIR = DATASET / "val"


# =========================
# Config
# =========================

SEED = 42

VAL_RATIO = 0.10

CLASSES = [
    "good",
    "defective"
]

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# =========================
# Functions
# =========================

def get_images(folder):

    return [
        f
        for f in folder.iterdir()
        if f.is_file()
        and f.suffix.lower() in EXTENSIONS
    ]



def copy_files(files, destination):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in files:

        shutil.copy2(
            file,
            destination / file.name
        )



# =========================
# Main
# =========================

def main():

    random.seed(SEED)


    print("=" * 60)
    print("CREATE VALIDATION FROM RANDOM80 TRAIN")
    print("=" * 60)


    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            TRAIN_DIR
        )


    # create val folders

    for cls in CLASSES:

        (
            VAL_DIR / cls
        ).mkdir(
            parents=True,
            exist_ok=True
        )


    report = {}


    for cls in CLASSES:

        source = TRAIN_DIR / cls


        images = get_images(source)


        print()
        print("Class:", cls)

        print(
            "Before:",
            len(images)
        )


        random.shuffle(images)


        val_count = int(
            len(images)
            *
            VAL_RATIO
        )


        val_images = images[:val_count]


        train_remaining = images[val_count:]


        # remove selected images from train

        for img in val_images:

            img.unlink()


        # copy to val

        copy_files(
            val_images,
            VAL_DIR / cls
        )


        print(
            "Train after split:",
            len(train_remaining)
        )

        print(
            "Val:",
            len(val_images)
        )


        report[cls] = {

            "original_train":
                len(images),

            "new_train":
                len(train_remaining),

            "val":
                len(val_images)

        }



    with open(
        DATASET / "val_split_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "Validation created:"
    )

    print(
        VAL_DIR
    )



if __name__ == "__main__":

    main()