"""Build two reproducible 25-image *demo* sets from the existing held-out Tinplate test split.

A: 13 original GOOD + 12 original DEFECTIVE.
B: 8 unchanged GOOD + 8 unchanged DEFECTIVE + 9 quality-degraded versions
   (5 originating from GOOD and 4 from DEFECTIVE), with required quality REVIEW.

These are not new independent external images: all 25 source photos come from
an existing test split that has already been used for evaluation. Do not report
these results as a new generalization benchmark or treat REVIEW as a third class.
"""
from pathlib import Path
from PIL import Image, ImageOps
import random
import csv
import hashlib

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'data' / 'raw' / 'TinplateLidDATA2024' / 'test'
OUT = ROOT / 'data' / 'demo25'
SEED = 42
EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def unique_samples(folder: Path, n: int, rng: random.Random):
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXT)
    rng.shuffle(files)
    selected, seen = [], set()
    for f in files:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        selected.append(f)
        if len(selected) == n:
            return selected
    raise RuntimeError(f'Need {n} unique photos under {folder}; found {len(selected)}')


def save_image(src: Path, dst: Path, degrade: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        img = ImageOps.exif_transpose(im).convert('RGB')
        if degrade:
            # Deliberately keep this file low-resolution. A quality gate should REVIEW it.
            img.thumbnail((120, 120), Image.Resampling.LANCZOS)
        img.save(dst, format='PNG')
    with Image.open(dst) as im:
        im.verify()


def add(rows, scenario, cls, index, src, degraded=False):
    filename = f'{cls}_{index:02d}.png'
    path = OUT / scenario / cls / filename
    save_image(src, path, degrade=degraded)
    rows.append(dict(scenario=scenario, relative_path=f'{cls}/{filename}',
                     actual_class='good' if cls == 'review' and src.parent.name == 'good' else
                                  'defective' if cls == 'review' else cls,
                     expected_action='REVIEW' if degraded else cls.upper().replace('DEFECTIVE', 'DEFECT'),
                     quality='low_resolution' if degraded else 'normal',
                     derived=str(bool(degraded)), source_file=src.name,
                     source_split='test'))


def main():
    rng = random.Random(SEED)
    good = unique_samples(SOURCE / 'good', 13, rng)
    defective = unique_samples(SOURCE / 'defective', 12, rng)
    a, b = [], []
    for i, src in enumerate(good, 1):
        add(a, 'A_binary_13_12', 'good', i, src)
    for i, src in enumerate(defective, 1):
        add(a, 'A_binary_13_12', 'defective', i, src)
    for i, src in enumerate(good[:8], 1):
        add(b, 'B_8_8_9_quality', 'good', i, src)
    for i, src in enumerate(defective[:8], 1):
        add(b, 'B_8_8_9_quality', 'defective', i, src)
    # 9 distinct underlying source photos, NOT an independently annotated third class.
    for i, src in enumerate(good[8:] + defective[8:], 1):
        add(b, 'B_8_8_9_quality', 'review', i, src, degraded=True)
    for scenario, rows in [('A_binary_13_12', a), ('B_8_8_9_quality', b)]:
        manifest = OUT / scenario / 'manifest.csv'
        with manifest.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(f'{scenario}: {len(rows)} image files; manifest: {manifest}')
    print('A: 13 good + 12 defective, all original images.')
    print('B: 8 good + 8 defective + 9 artificially low-res REVIEW-required images.')
    print('WARNING: Source images overlap across A and B. Do not treat these as independent tests.')
    print('WARNING: These 25 photos came from the already-evaluated source test split.')

if __name__ == '__main__':
    main()
