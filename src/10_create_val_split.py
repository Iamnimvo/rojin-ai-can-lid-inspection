from pathlib import Path
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]


SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "TinplateLidDATA2024_masked_random80_20"
)


OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "TinplateLidDATA2024_masked_final_train"
)


SEED = 42
VAL_RATIO = 0.10

CLASSES = [
    "good",
    "defective"
]


random.seed(SEED)


def copy_files(files, dst):

    dst.mkdir(parents=True, exist_ok=True)

    for f in files:
        shutil.copy2(
            f,
            dst / f.name
        )


def main():

    # remove previous
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)


    # create folders
    for split in [
        "train",
        "val",
        "test"
    ]:
        for cls in CLASSES:
            (
                OUTPUT
                /
                split
                /
                cls
            ).mkdir(
                parents=True,
                exist_ok=True
            )


    # copy test untouched
    for cls in CLASSES:

        test_files = list(
            (
                SOURCE
                /
                "test"
                /
                cls
            ).glob("*")
        )

        copy_files(
            test_files,
            OUTPUT
            /
            "test"
            /
            cls
        )


    # split train into train/val

    for cls in CLASSES:

        files = list(
            (
                SOURCE
                /
                "train"
                /
                cls
            ).glob("*")
        )

        random.shuffle(files)


        val_count = int(
            len(files)
            *
            VAL_RATIO
        )


        val_files = files[:val_count]

        train_files = files[val_count:]


        print(cls)

        print(
            "train:",
            len(train_files)
        )

        print(
            "val:",
            len(val_files)
        )


        copy_files(
            train_files,
            OUTPUT
            /
            "train"
            /
            cls
        )


        copy_files(
            val_files,
            OUTPUT
            /
            "val"
            /
            cls
        )


    print("\nDONE")
    print(OUTPUT)


if __name__ == "__main__":
    main()