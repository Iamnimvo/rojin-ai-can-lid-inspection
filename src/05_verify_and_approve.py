r"""Validate processed dataset before training; visually inspect ALL contact sheets first.
python .\src\05_verify_and_approve.py
python .\src\05_verify_and_approve.py --approve
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from rojin_pipeline import read_bgr

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'/'TinplateLidDATA2024'
DATA=ROOT/'data'/'processed'/'TinplateLidDATA2024_masked_v1'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def validate():
    manifest=DATA/'manifest.csv'
    if not manifest.exists():raise FileNotFoundError(manifest)
    with manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    issues=[]
    if not rows:issues.append('EMPTY_MANIFEST')
    counts=defaultdict(int)
    hashes=defaultdict(set)
    for row in rows:
        split,cls=row['split'],row['class']
        counts[(split,cls)]+=1
        original=RAW/row['source']
        if not original.exists() or sha(original)!=row['sha256']:
            issues.append('ORIGINAL_CHANGED_OR_MISSING:'+row['source'])
        hashes[row['sha256']].add(split)
        if row['status']!='OK':
            issues.append('MASK_REVIEW_REQUIRED:'+row['source']+':'+row['reason']);continue
        crop=DATA/row['crop'];mask=DATA/row['mask']
        if not crop.exists() or not mask.exists():
            issues.append('MISSING_MASK_OR_CROP:'+row['source']);continue
        try:
            im=read_bgr(crop)
            mm=read_bgr(mask)
            if im.shape!=(704,704,3):issues.append('BAD_OUTPUT_SHAPE:'+row['crop'])
            if len(np.unique(mm[:,:,0]))>2 or not set(np.unique(mm[:,:,0])).issubset({0,255}):
                issues.append('MASK_NOT_BINARY:'+row['mask'])
            # Images are cropped/recentered so zero padding exists near corners.
            if np.any(im[0,0]) or np.any(im[-1,-1]):issues.append('CORNERS_NOT_BLACK:'+row['crop'])
            if not np.any(im):issues.append('ALL_BLACK_CROP:'+row['crop'])
        except Exception as e:issues.append('BAD_FILE:'+row['source']+':'+str(e))
    for d,splits in hashes.items():
        if len(splits)>1:issues.append('EXACT_CROSS_SPLIT_DUPLICATE:'+d[:16]+':'+','.join(sorted(splits)))
    for split in ('train','val','test'):
        for cls in ('good','defective'):
            expected=len([p for p in (RAW/split/cls).iterdir() if p.is_file() and p.suffix.lower() in
                          {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}])
            if counts[(split,cls)]!=expected:
                issues.append(f'INCOMPLETE_{split}_{cls}:manifest={counts[(split,cls)]}:raw={expected}')
    print('Verified entries:',len(rows),'across splits:',dict(counts))
    print('Detected issues:',len(issues))
    for issue in issues[:40]:print('  !',issue)
    if len(issues)>40:print('  ...remaining',len(issues)-40)
    if issues:raise RuntimeError('Approval blocked; correct issues and rebuild. Similar/near duplicates also need human review.')
    return manifest,rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--approve',action='store_true')
    args=ap.parse_args()
    manifest,rows=validate()
    if not args.approve:
        print('Structural validation PASSED. Review ALL audit/sheets and overlays before approval.')
        return
    print('Inspect:',DATA/'audit'/'sheets')
    print('Approval attests you have visually checked the masks (particularly both classes, edge defects and all exceptions).')
    answer=input('Type exactly APPROVE to permit training: ').strip()
    if answer!='APPROVE':
        print('No approval was written.');return
    stamp={'manifest_sha256':sha(manifest),'entries':len(rows),
           'human_visual_review_declared':True,
           'method':'geometric/manual_mask_v1','classifier_input_px':704}
    (DATA/'.qa_approved.json').write_text(json.dumps(stamp,indent=2),encoding='utf-8')
    print('Approved:',DATA/'.qa_approved.json')

if __name__=='__main__':main()
