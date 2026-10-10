"""Download 25 real internet image sources for the Rojin external challenge.
No TinplateLidDATA2024/test image or AI-generated image is used.
Run from D:\\rojin_lid_ai: python .\\src\\download_and_prepare_internet25.py
"""
from __future__ import annotations
import csv
import hashlib
import io
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(r'D:\rojin_lid_ai')
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'internet25_manifest.csv'
DEST = ROOT / 'data' / 'internet25_external'
RAW = DEST / 'source_originals'
READY = DEST / 'ready_for_model'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
}

def image_from_bytes(content):
    with Image.open(io.BytesIO(content)) as im:
        im.load()
        return im.format, im.size


def main():
    if not MANIFEST.is_file():
        raise SystemExit(f'Missing: {MANIFEST}')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row['group'] for row in rows)
    if len(rows) != 25 or counts != {'good':8, 'defective':8, 'review_candidate':9}:
        raise SystemExit(f'Invalid manifest: {len(rows)} rows; {counts}')
    all_urls = [row['image_url'] for row in rows]
    if len(set(all_urls)) != 25:
        raise SystemExit('Duplicate image URLs in manifest!')

    RAW.mkdir(parents=True, exist_ok=True)
    READY.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    report = []
    failed = []

    for row in rows:
        ident, group, url = row['id'], row['group'], row['image_url']
        path_ext = Path(urlparse(url).path).suffix.lower() or '.img'
        raw_path = RAW / (ident + path_ext)
        png_path = READY / group / (ident + '.png')
        png_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'[{ident}] {group}: ', end='', flush=True)
        try:
            if raw_path.exists():
                content = raw_path.read_bytes()
                try:
                    image_from_bytes(content)
                except Exception:
                    raw_path.unlink()
                    content = None
            else:
                content = None
            if content is None:
                err = ''
                for attempt in range(3):
                    try:
                        h = dict(HEADERS)
                        h['Referer'] = row['source_page']
                        response = session.get(url, timeout=30, headers=h)
                        response.raise_for_status()
                        content = response.content
                        _, dims = image_from_bytes(content)  # Reject HTML / blocked / damaged responses.
                        if len(content) < 1024:
                            raise ValueError('Too few image bytes')
                        raw_path.write_bytes(content)
                        break
                    except Exception as e:
                        err = f'{type(e).__name__}: {e}'
                        content = None
                        time.sleep(1 + attempt)
                if content is None:
                    raise RuntimeError(err)

            with Image.open(raw_path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGBA')
                white = Image.new('RGBA', image.size, (255,255,255,255))
                white.alpha_composite(image)
                image = white.convert('RGB')
                orig_size = image.size
                box = row['crop_bbox_xyxy_normalized'].strip()
                if box:
                    coords = [float(v) for v in box.split(';')]
                    if len(coords) != 4 or not (0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1):
                        raise ValueError(f'Invalid crop bbox {box}')
                    w, h = image.size
                    crop = (int(coords[0]*w),int(coords[1]*h),int(coords[2]*w),int(coords[3]*h))
                    image = image.crop(crop)
                image.save(png_path, 'PNG')
                with Image.open(png_path) as check:
                    check.verify()
            sha = hashlib.sha256(content).hexdigest()
            report.append({'id':ident,'group':group,'raw_filename':raw_path.name,'prepared_filename':f'{group}/{png_path.name}',
                           'source_image_url':url,'source_page':row['source_page'],'sha256_original':sha,
                           'original_width':orig_size[0],'original_height':orig_size[1],
                           'prepared_width':image.size[0],'prepared_height':image.size[1],
                           'limitations':row['limitations']})
            print(f'OK  {orig_size} -> {image.size}')
        except Exception as e:
            failed.append((ident,url,str(e)))
            if png_path.exists():
                png_path.unlink()
            print(f'FAILED: {e}')

    hashes = [r['sha256_original'] for r in report]
    if len(set(hashes)) != len(hashes):
        print('WARNING: exact duplicate source image bytes detected; inspect hashes in download_report.csv')
    report_path = DEST / 'download_report.csv'
    if report:
        with report_path.open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(report[0]))
            writer.writeheader();writer.writerows(report)
    if report:
        # Preview sheet allows the user to inspect every crop before obtaining metrics.
        tw, th = 220, 180
        cols = 5
        rows_n = (len(report) + cols - 1) // cols
        sheet = Image.new('RGB', (cols*tw, rows_n*th), 'white')
        draw = ImageDraw.Draw(sheet)
        for idx, item in enumerate(report):
            img_path = READY / item['prepared_filename']
            with Image.open(img_path) as view:
                view = view.convert('RGB')
                view.thumbnail((tw-16, th-42))
                x = (idx%cols)*tw + (tw-view.width)//2
                y = (idx//cols)*th + 6
                sheet.paste(view, (x,y))
                label = f"{item['id']}  [{item['group']}]"
                draw.text(((idx%cols)*tw+7,(idx//cols)*th+th-31),label,fill='black')
        sheet.save(DEST/'preview_contact_sheet.jpg', quality=90)
    if failed:
        print('\nSome Internet hosts may block automatic requests. Download the ORIGINAL image manually via image_url')
        print('Save it under source_originals as <ID>.<original_extension>, then rerun this script.')
        for ident,url,error in failed: print(f'FAILED {ident}: {url}\n  {error}')
        raise SystemExit(f'INCOMPLETE: {len(report)}/25 readable image sources. Do not evaluate yet.')
    print('\nSUCCESS: 25 distinct URL entries prepared, in:', READY)
    print('Inspect each prepared PNG before testing. Some good examples required approximate single-lid crops.')
    print('A source URL does NOT establish independent QC approval for product photography.')

if __name__ == '__main__':
    main()
