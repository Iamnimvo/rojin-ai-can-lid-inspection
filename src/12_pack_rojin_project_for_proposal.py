"""
ROJIN - Compact, read-only project package for proposal review.

Save as: D:\\rojin_lid_ai\\src\\12_pack_rojin_project_for_proposal.py
Run:     python .\\src\\12_pack_rojin_project_for_proposal.py

Python standard library only. No changes to training, testing, source data, or weights.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

# ------------------------ USER CONFIG ------------------------
ROOT = Path(r"D:\rojin_lid_ai")
ZIP_NAME = "rojin_proposal_review_package.zip"
MODEL_RUN = "yolo11n_cls_masked_random80_v2"
TEST_RUN = "random20_final_test"
SAMPLES_PER_CLASS = 3
MAX_TOTAL_MB = 90
MAX_REGULAR_FILE_MB = 8
# -------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CODE_EXTENSIONS = {".py", ".yaml", ".yml", ".toml", ".json", ".csv", ".md", ".txt", ".cfg", ".ini"}
REPORT_EXTENSIONS = {".yaml", ".yml", ".json", ".csv", ".md", ".txt"}
ZIP_FILE = ROOT / ZIP_NAME
MAX_TOTAL = MAX_TOTAL_MB * 1024 * 1024
MAX_REGULAR_FILE = MAX_REGULAR_FILE_MB * 1024 * 1024


def ensure_inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def select_spread(paths: list[Path], max_count: int) -> list[Path]:
    """Take distributed examples, rather than only the first few filenames."""
    paths = sorted(paths, key=lambda p: str(p).lower())
    if len(paths) <= max_count:
        return paths
    if max_count == 1:
        return [paths[len(paths) // 2]]
    indexes = [round(i * (len(paths) - 1) / (max_count - 1)) for i in range(max_count)]
    return [paths[i] for i in sorted(set(indexes))]


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


class PackageBuilder:
    def __init__(self, output: zipfile.ZipFile):
        self.output = output
        self.seen: set[str] = set()
        self.included: list[dict] = []
        self.skipped: list[dict] = []
        self.group_counts: Counter[str] = Counter()
        self.input_bytes = 0
        self.missing_important: list[str] = []
        self.sampled_masked: list[Path] = []

    def add(self, file: Path, group: str, size_limit: int = MAX_REGULAR_FILE) -> bool:
        if not file.is_file() or file.is_symlink() or not ensure_inside_project(file):
            return False
        rel = file.relative_to(ROOT).as_posix()
        if rel in self.seen or file == ZIP_FILE:
            return False
        self.seen.add(rel)
        size = file.stat().st_size
        if size > size_limit:
            self.skipped.append({"file": rel, "reason": "file_over_size_limit", "bytes": size})
            return False
        if self.input_bytes + size > MAX_TOTAL:
            self.skipped.append({"file": rel, "reason": "total_package_budget_exceeded", "bytes": size})
            return False
        digest = hashlib.sha256()
        with file.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        self.output.write(file, arcname=rel)
        self.input_bytes += size
        self.group_counts[group] += 1
        self.included.append({"file": rel, "group": group, "bytes": size, "sha256": digest.hexdigest()})
        return True

    def add_dir_filtered(self, folder: Path, group: str, extensions: set[str], max_files: int = 500):
        if not folder.is_dir():
            return
        candidates = sorted((p for p in folder.rglob("*") if p.is_file()
                             and p.suffix.lower() in extensions
                             and not any(part in {"__pycache__", ".venv", ".git"} for part in p.parts)),
                            key=lambda p: str(p).lower())
        for p in candidates[:max_files]:
            self.add(p, group)
        if len(candidates) > max_files:
            self.skipped.append({"folder": folder.relative_to(ROOT).as_posix(),
                                 "reason": "too_many_metadata_files", "count": len(candidates) - max_files})


def pack() -> int:
    if not ROOT.is_dir():
        print(f"ERROR: Project folder not found: {ROOT}")
        return 2

    if not (ROOT / "src").is_dir():
        print(f"ERROR: Source folder not found: {ROOT / 'src'}")
        return 2

    print("=" * 66)
    print("ROJIN - COMPACT PROJECT PACKAGE FOR PROPOSAL REVIEW")
    print("Mode: READ ONLY for project files (creates/updates ZIP only)")
    print("Project:", ROOT)
    print("Output: ", ZIP_FILE)
    print("=" * 66)

    try:
        with zipfile.ZipFile(ZIP_FILE, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as z:
            p = PackageBuilder(z)

            # 1. All Python source and small source configuration/report files.
            p.add_dir_filtered(ROOT / "src", "project_source", CODE_EXTENSIONS, max_files=1000)
            for f in sorted(ROOT.iterdir()):
                if f.is_file() and f.suffix.lower() in CODE_EXTENSIONS and not f.name.startswith(".env"):
                    p.add(f, "root_configuration")
            p.add_dir_filtered(ROOT / "models", "model_configuration", CODE_EXTENSIONS, max_files=100)

            # 2. Exact final training run: configuration, curves and pretrained-best output.
            run = ROOT / "results" / MODEL_RUN
            if not run.is_dir():
                p.missing_important.append(str(run.relative_to(ROOT)))
            for name in ("args.yaml", "results.csv", "results.png", "confusion_matrix.png",
                         "confusion_matrix_normalized.png", "labels.jpg", "labels_correlogram.jpg"):
                p.add(run / name, "final_training_results")
            for prefix in ("train_batch", "val_batch"):
                matches = sorted(f for f in run.glob(prefix + "*") if f.suffix.lower() in IMAGE_EXTENSIONS)
                for f in select_spread(matches, 3):
                    p.add(f, "training_visual_examples")
            best = run / "weights" / "best.pt"
            if not p.add(best, "final_model_best_pt", size_limit=15 * 1024 * 1024):
                p.missing_important.append(str(best.relative_to(ROOT)))
            # Deliberately exclude last.pt and all other model checkpoints.

            # 3. Current test, reports and a few predicted images in each class.
            test_run = ROOT / "results" / TEST_RUN
            if not test_run.is_dir():
                p.missing_important.append(str(test_run.relative_to(ROOT)))
            if test_run.is_dir():
                for f in sorted(test_run.iterdir()):
                    if f.is_file() and f.suffix.lower() in (REPORT_EXTENSIONS | {".png", ".jpg", ".jpeg", ".pdf"}):
                        p.add(f, "final_test_report")
            for cls in ("good", "defective", "review"):
                folder = test_run / cls
                for f in select_spread(image_files(folder), SAMPLES_PER_CLASS):
                    p.add(f, "final_test_predictions_sample")

            # 4. Integrity verification, optional older evaluation report.
            p.add_dir_filtered(ROOT / "results" / "dataset_integrity_audit",
                               "dataset_integrity_report", REPORT_EXTENSIONS, max_files=30)
            old_test = ROOT / "results" / "test_random20_evaluation"
            if old_test.is_dir():
                for f in sorted(old_test.iterdir()):
                    if f.is_file() and f.suffix.lower() in REPORT_EXTENSIONS | {".png"}:
                        p.add(f, "earlier_test_report_optional")

            # 5. Small split/masking reports; never package the full datasets.
            processed = ROOT / "data" / "processed"
            metadata_folders = [
                ("TinplateLidDATA2024_masked_random80_20", "reference_split_metadata"),
                ("TinplateLidDATA2024_masked_random80", "training_split_metadata"),
                ("TinplateLidDATA2024_masked_random20", "testing_split_metadata"),
                ("TinplateLidDATA2024_masked_v1", "original_masking_metadata"),
            ]
            for name, group in metadata_folders:
                p.add_dir_filtered(processed / name, group, REPORT_EXTENSIONS, max_files=80)

            # 6. Original (masked) dataset examples, distributed across train / val / test.
            sample_dirs = [
                ("TinplateLidDATA2024_masked_random80", "train"),
                ("TinplateLidDATA2024_masked_random80", "val"),
                ("TinplateLidDATA2024_masked_random20", "test"),
            ]
            for dataset_name, split in sample_dirs:
                for cls in ("good", "defective"):
                    folder = processed / dataset_name / split / cls
                    for f in select_spread(image_files(folder), 2):
                        if p.add(f, "masked_dataset_small_sample"):
                            p.sampled_masked.append(f)

            # 7. A few segmentation audit overlay pictures, if present.
            mask_audit = processed / "TinplateLidDATA2024_masked_v1" / "audit"
            overlay_dirs = [mask_audit / "overlays", mask_audit / "overlay"]
            for overlay_dir in overlay_dirs:
                if overlay_dir.is_dir():
                    examples = sorted((f for f in overlay_dir.rglob("*") if f.is_file()
                                       and f.suffix.lower() in IMAGE_EXTENSIONS), key=lambda x: str(x).lower())
                    for f in select_spread(examples, 4):
                        p.add(f, "masking_overlay_sample")
                    break

            # 8. Optional raw-image pairs, found by matching names with selected masked samples.
            # No exhaustive raw export; maximum 4 images from available raw source folders.
            raw_root = ROOT / "data" / "raw"
            if raw_root.is_dir() and p.sampled_masked:
                wanted = {f.name.lower() for f in select_spread(p.sampled_masked, 4)}
                found = set()
                for raw in raw_root.rglob("*"):
                    if (raw.is_file() and raw.suffix.lower() in IMAGE_EXTENSIONS
                            and raw.name.lower() in wanted and raw.name.lower() not in found):
                        if p.add(raw, "raw_image_pair_sample"):
                            found.add(raw.name.lower())
                    if found == wanted:
                        break

            readme = """ROJIN PROJECT REVIEW PACKAGE
============================
This ZIP was generated from D:\\rojin_lid_ai for technical review and an
English industrial Proof-of-Concept proposal concerning the final model:
  results/yolo11n_cls_masked_random80_v2/weights/best.pt

IMPORTANT: The TinplateLidDATA2024 dataset is PUBLIC/EXTERNAL, not a verified
Rojin factory dataset. Performance on that dataset does not establish field
performance on actual Rojin cans or a complete production line.

Selection policy:
- All small Python source/configuration files in src/ and some root/models metadata.
- The named final training run, best.pt, and charts if present.
- The named final test run: CSV/JSON/report/images at top level.
- Up to three example predicted images from each of good/defective/review.
- Dataset split manifests, verification reports, masking metadata.
- A small set of masked samples and optional raw/overlay examples.

NOT INCLUDED:
- .venv, full data folders, all 1307 training/test images, entire results history,
  redundant model weights, caches, or large temporary files.

The package is for review only. The original project is unchanged.
See package_manifest.json for included file SHA-256 hashes and any warnings.
"""
            z.writestr("__PACKAGE_INFO__/README.txt", readme)
            manifest = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "project_path": str(ROOT),
                "model_run": MODEL_RUN,
                "test_run": TEST_RUN,
                "external_public_dataset": True,
                "number_of_included_project_files": len(p.included),
                "total_uncompressed_bytes": p.input_bytes,
                "included_by_category": dict(sorted(p.group_counts.items())),
                "missing_important": p.missing_important,
                "skipped_by_size_or_limit": p.skipped,
                "included": p.included,
            }
            z.writestr("__PACKAGE_INFO__/package_manifest.json",
                       json.dumps(manifest, indent=2, ensure_ascii=False))

        # Verify the ZIP can be read and has no corrupted members.
        with zipfile.ZipFile(ZIP_FILE, "r") as z:
            corrupt = z.testzip()
            if corrupt is not None:
                raise RuntimeError(f"ZIP verification failed at: {corrupt}")
            members = len(z.namelist())

    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        print("ERROR:", exc)
        return 1

    print("\nSUCCESS - ZIP created and verified")
    print("ZIP:", ZIP_FILE)
    print("ZIP size: {:.2f} MB".format(ZIP_FILE.stat().st_size / (1024 * 1024)))
    print("Project files included:", len(p.included))
    print("Files inside ZIP (including README + manifest):", members)
    print("Included categories:")
    for key, count in sorted(p.group_counts.items()):
        print(f"  {key}: {count}")
    if p.missing_important:
        print("\nWARNING - expected files/folders missing:")
        for missing in p.missing_important:
            print(" -", missing)
    if p.skipped:
        print(f"\nINFO - {len(p.skipped)} items were omitted due to size/limit.")
    print("\nNo original project file was modified or deleted.")
    print("Review the ZIP for private information before uploading.")
    return 0


if __name__ == "__main__":
    sys.exit(pack())
