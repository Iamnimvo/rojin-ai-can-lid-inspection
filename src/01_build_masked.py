r"""Prepare train/val/test with black BG without changing raw files.

Run pilot first: python .\src\01_build_masked.py --pilot 20
Full: python .\src\01_build_masked.py --full
If rerunning a generated *processed* folder only: add --overwrite.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from rojin_pipeline import VALID_EXT, read_bgr, save_image, segment_lid, overlay

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "TinplateLidDATA2024"
PROCESSED_BASE = ROOT / "data" / "processed"
CLASSES = ("good", "defective")
SPLITS = ("train", "val", "test")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def file_inventory(splits):
    categories = {}
    for split in splits:
        for name in CLASSES:
            folder = RAW / split / name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class directory: {folder}")
            files = sorted((p for p in folder.iterdir()
                            if p.is_file() and p.suffix.lower() in VALID_EXT),
                           key=lambda p: p.name.lower())
            if not files:
                raise ValueError(f"No images found: {folder}")
            if len({p.stem.lower() for p in files}) != len(files):
                raise ValueError(f"Duplicate filename stems in {folder}; PNG destinations would collide")
            categories[(split, name)] = files
    return categories


def pilot_inventory(categories, count):
    groups = [list(v) for v in categories.values()]
    selected = []
    while groups and len(selected) < count:
        for group in list(groups):
            if group and len(selected) < count:
                selected.append(group.pop(0))
            if not group:
                groups.remove(group)
    return selected


def thumb(im, w, h):
    img = np.zeros((h, w, 3), np.uint8)
    if im is None:
        return img
    ih, iw = im.shape[:2]
    k = min(w / iw, h / ih)
    new = cv2.resize(im, (max(1, int(iw*k)), max(1, int(ih*k))), interpolation=cv2.INTER_AREA)
    nw, nh = new.shape[1], new.shape[0]
    img[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw] = new
    return img


def create_sheets(rows, out):
    """All images are represented on paginated audit sheets; user must inspect."""
    per_page, cols, tile_w, tile_h = 30, 5, 392, 214
    for start in range(0, len(rows), per_page):
        items = rows[start:start + per_page]
        sheet = np.full((6*tile_h, cols*tile_w, 3), 235, dtype=np.uint8)
        for i, row in enumerate(items):
            yy, xx = divmod(i, cols)
            x, y = xx*tile_w, yy*tile_h
            try:
                raw = read_bgr(RAW / row["source"])
                over_path = out / "audit" / "overlays" / Path(row["source"]).with_suffix(".jpg")
                left = read_bgr(over_path) if over_path.exists() else raw
                crop_path = out / row["crop"] if row["crop"] else None
                right = read_bgr(crop_path) if crop_path and crop_path.exists() else None
                sheet[y:y+174, x:x+189] = thumb(left, 189, 174)
                sheet[y:y+174, x+195:x+384] = thumb(right, 189, 174)
            except Exception:
                pass
            color = (20,105,15) if row["status"] == "OK" else (15,15,210)
            cv2.putText(sheet, Path(row["source"]).name[:32], (x+5,y+190),
                        cv2.FONT_HERSHEY_SIMPLEX, .37,(25,25,25),1,cv2.LINE_AA)
            cv2.putText(sheet, f"{row['status']} | {row['method'] or row['reason']}",
                        (x+5,y+207), cv2.FONT_HERSHEY_SIMPLEX, .39,color,1,cv2.LINE_AA)
        save_image(out / "audit" / "sheets" / f"sheet_{start//per_page+1:03d}.jpg", sheet)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int, metavar="N", help="Pilot selected images from train+val only")
    mode.add_argument("--full", action="store_true", help="Process all original splits, never merge them")
    ap.add_argument("--overwrite", action="store_true", help="Only delete the designated processed output directory")
    args = ap.parse_args()
    if args.pilot is not None and args.pilot < 1:
        ap.error("--pilot must be positive")
    categories = file_inventory(SPLITS if args.full else ("train", "val"))
    all_files = [p for values in categories.values() for p in values]
    files = all_files if args.full else pilot_inventory(categories, args.pilot)
    out = PROCESSED_BASE / ("TinplateLidDATA2024_masked_v1" if args.full else "TinplateLidDATA2024_masked_PILOT")
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out}\nUse --overwrite only when you intend to rebuild THIS processed directory")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    overrides_path = ROOT / "config" / "rojin_manual_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    rows = []
    passed, review = 0, 0
    print("Original:", RAW)
    print("Output  :", out)
    print("Images  :", len(files))
    for i, source in enumerate(files, start=1):
        relative = source.relative_to(RAW).as_posix()
        split, name = source.relative_to(RAW).parts[:2]
        row = {"source": relative, "split":split, "class":name,
               "sha256":sha256(source), "status":"REVIEW", "reason":"", "method":"",
               "edge_support":"", "area_ratio":"", "cx":"", "cy":"", "radius":"",
               "crop":"", "mask":""}
        image = None
        result = None
        try:
            image = read_bgr(source)
            result = segment_lid(image, override=overrides.get(relative))
            row.update(status=result.status, reason=result.reason, method=result.method,
                       edge_support=round(result.edge_support,4), area_ratio=round(result.area_ratio,5),
                       cx=round(result.cx,2), cy=round(result.cy,2), radius=round(result.radius,2))
            if result.status == "OK":
                crop_relative = Path(split) / name / (source.stem + ".png")
                mask_relative = Path("masks") / split / name / (source.stem + ".png")
                save_image(out / crop_relative, result.crop)
                save_image(out / mask_relative, result.mask)
                row["crop"], row["mask"] = crop_relative.as_posix(), mask_relative.as_posix()
                passed += 1
            else:
                review += 1
            vis = overlay(image, result)
            save_image(out / "audit" / "overlays" / Path(relative).with_suffix(".jpg"), vis)
        except Exception as exc:
            row.update(status="REVIEW", reason=f"EXCEPTION:{type(exc).__name__}:{str(exc)[:90]}")
            review += 1
            if image is not None:
                bad = np.copy(image)
                cv2.putText(bad,"EXCEPTION",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(10,10,255),2)
                save_image(out / "audit" / "overlays" / Path(relative).with_suffix(".jpg"), bad)
        rows.append(row)
        print(f"[{i:4d}/{len(files)}] {row['status']:6s} {relative} {row['reason']}")
    fields = list(rows[0])
    with (out / "manifest.csv").open("w", newline="",encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "review_required.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows([r for r in rows if r["status"] != "OK"])
    create_sheets(rows, out)
    summary = {"raw_root":str(RAW),"output":str(out), "expected_images":len(files),
               "ok":passed,"review":review,"pilot":not args.full,
               "all_splits_separate":True,"override_file":str(overrides_path)}
    (out / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nSUMMARY:",json.dumps(summary,ensure_ascii=False,indent=2))
    print("Review contact sheets in:",out / "audit" / "sheets")
    print("REVIEW failures:",out / "review_required.csv")
    if review:
        print("No automatic replacement images: correct the masks using 04_manual_override.py and rebuild.")
    else:
        print("All images processed. Human visual audit is still REQUIRED before approval/train.")

if __name__ == "__main__":
    main()
