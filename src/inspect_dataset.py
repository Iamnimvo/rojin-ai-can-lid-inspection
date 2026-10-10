from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

DATASET_PATH = Path(
    r"D:\rojin_lid_ai\data\raw\TinplateLidDATA2024"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def main():
    print("=" * 70)
    print("TINPLATE LID DATASET INSPECTION")
    print("=" * 70)

    if not DATASET_PATH.exists():
        print("\nERROR: Dataset not found:")
        print(DATASET_PATH)
        return

    print(f"\nDataset path:\n{DATASET_PATH}")

    print("\n" + "=" * 70)
    print("TOP-LEVEL STRUCTURE")
    print("=" * 70)

    top_items = sorted(DATASET_PATH.iterdir())

    for item in top_items:
        if item.is_dir():
            print(f"[DIR]  {item.name}")
        else:
            print(f"[FILE] {item.name}")

    image_files = [
        p
        for p in DATASET_PATH.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("\n" + "=" * 70)
    print("IMAGE STATISTICS")
    print("=" * 70)

    print(f"Total images: {len(image_files)}")

    extension_counts = Counter(
        p.suffix.lower() for p in image_files
    )

    print("\nExtensions:")
    for ext, count in extension_counts.items():
        print(f"  {ext}: {count}")

    folder_counts = defaultdict(int)

    for img in image_files:
        relative_parent = img.parent.relative_to(DATASET_PATH)
        folder_counts[str(relative_parent)] += 1

    print("\n" + "=" * 70)
    print("IMAGES PER FOLDER")
    print("=" * 70)

    for folder, count in sorted(
        folder_counts.items(),
        key=lambda x: x[0]
    ):
        print(f"{folder:60} {count}")

    print("\n" + "=" * 70)
    print("IMAGE RESOLUTION SAMPLE")
    print("=" * 70)

    size_counts = Counter()
    corrupted = []

    sample_files = image_files[:200]

    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                size_counts[img.size] += 1
        except Exception as e:
            corrupted.append((img_path, str(e)))

    for size, count in size_counts.most_common(20):
        print(f"{size[0]} x {size[1]} : {count}")

    print("\n" + "=" * 70)
    print("SAMPLE FILES")
    print("=" * 70)

    for img in image_files[:20]:
        print(img.relative_to(DATASET_PATH))

    print("\n" + "=" * 70)
    print("CORRUPTED IMAGE CHECK")
    print("=" * 70)

    if not corrupted:
        print("No corrupted images found in inspected sample.")
    else:
        print(f"Found {len(corrupted)} corrupted images:")
        for path, error in corrupted:
            print(path)
            print(error)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()