from pathlib import Path
import random
import shutil
import json
import csv


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "TinplateLidDATA2024_masked_v1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "TinplateLidDATA2024_masked_random80_20"
)


# ==========================================
# Config
# ==========================================

SEED = 42
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

CLASSES = ["good", "defective"]
SPLITS_TO_MERGE = ["train", "val", "test"]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ==========================================
# Helpers
# ==========================================

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def collect_images():
    """
    Collect all images from:
      train/good, train/defective
      val/good, val/defective
      test/good, test/defective
    """
    data = {cls: [] for cls in CLASSES}

    for split in SPLITS_TO_MERGE:
        for cls in CLASSES:
            folder = SOURCE_ROOT / split / cls

            if not folder.exists():
                print(f"[WARNING] Missing folder: {folder}")
                continue

            files = [p for p in folder.iterdir() if p.is_file() and is_image_file(p)]
            # Keep original split info too, useful for manifest and name conflict handling
            for f in files:
                data[cls].append({
                    "path": f,
                    "original_split": split,
                    "class_name": cls,
                    "file_name": f.name,
                })

    return data


def ensure_clean_output():
    if OUTPUT_ROOT.exists():
        print(f"[INFO] Removing old output folder:\n{OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "test"]:
        for cls in CLASSES:
            (OUTPUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)


def copy_with_unique_name(src: Path, dst_dir: Path, prefix: str) -> str:
    """
    Copy file to destination.
    If duplicate name exists, add prefix based on original split.
    Returns final file name.
    """
    target = dst_dir / src.name

    if not target.exists():
        shutil.copy2(src, target)
        return target.name

    # Resolve conflict by prefixing original split
    target = dst_dir / f"{prefix}__{src.name}"

    if not target.exists():
        shutil.copy2(src, target)
        return target.name

    # If still duplicate, add index
    stem = target.stem
    suffix = target.suffix
    idx = 1
    while True:
        candidate = dst_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate.name
        idx += 1


def split_class_items(items, train_ratio=0.80):
    """
    Deterministic random split for a single class.
    """
    random.shuffle(items)

    n_total = len(items)
    n_train = round(n_total * train_ratio)
    n_test = n_total - n_train

    train_items = items[:n_train]
    test_items = items[n_train:]

    return train_items, test_items, n_total, n_train, n_test


# ==========================================
# Main
# ==========================================

def main():
    random.seed(SEED)

    print("=" * 70)
    print("ROJIN LID AI - CREATE FINAL RANDOM 80/20 DATASET SPLIT")
    print("=" * 70)
    print(f"Source dataset: {SOURCE_ROOT}")
    print(f"Output dataset: {OUTPUT_ROOT}")
    print(f"Random seed:    {SEED}")
    print()

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found:\n{SOURCE_ROOT}")

    data = collect_images()

    total_good = len(data["good"])
    total_defective = len(data["defective"])
    total_all = total_good + total_defective

    print("Collected images from existing train/val/test:")
    print(f"  good:       {total_good}")
    print(f"  defective:  {total_defective}")
    print(f"  TOTAL:      {total_all}")
    print()

    if total_all != 1307:
        print(f"[WARNING] Expected total 1307, but found {total_all}")
        print()

    ensure_clean_output()

    manifest_rows = []
    report = {
        "seed": SEED,
        "source_root": str(SOURCE_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "train_ratio": TRAIN_RATIO,
        "test_ratio": TEST_RATIO,
        "classes": {}
    }

    grand_train = 0
    grand_test = 0

    for cls in CLASSES:
        items = data[cls]

        train_items, test_items, n_total, n_train, n_test = split_class_items(
            items, train_ratio=TRAIN_RATIO
        )

        print(f"[CLASS: {cls}]")
        print(f"  Total: {n_total}")
        print(f"  Train: {n_train}")
        print(f"  Test:  {n_test}")
        print()

        # Copy train
        for item in train_items:
            dst_dir = OUTPUT_ROOT / "train" / cls
            final_name = copy_with_unique_name(
                src=item["path"],
                dst_dir=dst_dir,
                prefix=item["original_split"]
            )

            manifest_rows.append({
                "new_split": "train",
                "class_name": cls,
                "source_split": item["original_split"],
                "source_path": str(item["path"]),
                "copied_file_name": final_name,
            })

        # Copy test
        for item in test_items:
            dst_dir = OUTPUT_ROOT / "test" / cls
            final_name = copy_with_unique_name(
                src=item["path"],
                dst_dir=dst_dir,
                prefix=item["original_split"]
            )

            manifest_rows.append({
                "new_split": "test",
                "class_name": cls,
                "source_split": item["original_split"],
                "source_path": str(item["path"]),
                "copied_file_name": final_name,
            })

        report["classes"][cls] = {
            "total": n_total,
            "train": n_train,
            "test": n_test,
        }

        grand_train += n_train
        grand_test += n_test

    report["totals"] = {
        "all_images": total_all,
        "train_total": grand_train,
        "test_total": grand_test
    }

    # Save manifest CSV
    manifest_path = OUTPUT_ROOT / "split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "new_split",
                "class_name",
                "source_split",
                "source_path",
                "copied_file_name",
            ]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Save JSON report
    report_path = OUTPUT_ROOT / "split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("FINAL SPLIT CREATED SUCCESSFULLY")
    print("=" * 70)
    print("Final counts:")
    print(f"  Train total: {grand_train}")
    print(f"  Test total:  {grand_test}")
    print()
    print("Saved files:")
    print(f"  Dataset:  {OUTPUT_ROOT}")
    print(f"  Report:   {report_path}")
    print(f"  Manifest: {manifest_path}")
    print()

    print("Expected approximate result with current totals:")
    print("  good:       571  -> train ~457, test ~114")
    print("  defective:  736  -> train ~589, test ~147")
    print()


if __name__ == "__main__":
    main()