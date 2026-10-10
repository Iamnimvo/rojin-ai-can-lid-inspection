from pathlib import Path
import hashlib
from collections import defaultdict

DATASET = Path(
    r"D:\rojin_lid_ai\data\raw\TinplateLidDATA2024"
)

EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
}


def file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def main():

    print("=" * 70)
    print("DATASET LEAKAGE CHECK")
    print("=" * 70)

    splits = ["train", "val", "test"]

    hashes = defaultdict(list)

    total = 0

    for split in splits:

        split_path = DATASET / split

        images = [
            p for p in split_path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ]

        print(
            f"{split:10}: {len(images)} images"
        )

        for image in images:

            h = file_hash(image)

            hashes[h].append(
                (
                    split,
                    image.relative_to(DATASET)
                )
            )

            total += 1

    print("\nTotal:", total)

    duplicates = {
        h: items
        for h, items in hashes.items()
        if len(items) > 1
    }

    print("\n" + "=" * 70)
    print("DUPLICATES")
    print("=" * 70)

    if not duplicates:

        print("No exact duplicates found.")

    else:

        print(
            f"Found {len(duplicates)} duplicate groups."
        )

        for items in duplicates.values():

            print("\nDuplicate:")

            for split, path in items:

                print(
                    f"  [{split}] {path}"
                )

    print("\n" + "=" * 70)
    print("CROSS-SPLIT LEAKAGE")
    print("=" * 70)

    leakage_count = 0

    for items in duplicates.values():

        different_splits = set(
            split for split, _ in items
        )

        if len(different_splits) > 1:

            leakage_count += 1

            print("\nWARNING:")

            for split, path in items:

                print(
                    f"  [{split}] {path}"
                )

    if leakage_count == 0:

        print("No exact cross-split leakage detected.")

    else:

        print(
            f"\nCross-split leakage groups: "
            f"{leakage_count}"
        )


if __name__ == "__main__":
    main()