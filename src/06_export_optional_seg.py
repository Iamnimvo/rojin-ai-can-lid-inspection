r"""OPTIONAL. Export human-approved geometry/polygon masks to YOLO SEG polygons.
This trains a student to imitate the APPROVED masks. Pseudo-labels cannot by themselves
prove true mask IoU or improve beyond their teacher. Review real Rojin masks as well.
python .\src\06_export_optional_seg.py
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml
from rojin_pipeline import read_bgr

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'/'TinplateLidDATA2024_masked_v1'
RAW=ROOT/'data'/'raw'/'TinplateLidDATA2024'
OUT=ROOT/'data'/'processed'/'TinplateLid_lidseg_v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    approval=DATA/'.qa_approved.json'
    if not approval.exists() or json.loads(approval.read_text())['manifest_sha256']!=sha(DATA/'manifest.csv'):
        raise RuntimeError('Requires current human-approved masks; run 05_verify_and_approve.py --approve')
    if OUT.exists():raise FileExistsError(f'Existing dataset will not be overwritten: {OUT}')
    with (DATA/'manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    count=0
    for row in rows:
        if row['status']!='OK':raise RuntimeError(f'Unresolved mask: {row["source"]}')
        split=row['split'];label=row['class'];stem=Path(row['source']).stem
        img=RAW/row['source'];binary=cv2.imdecode(np.fromfile(str(DATA/row['mask']),dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if binary is None:raise FileNotFoundError(DATA/row['mask'])
        contour_list,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contour_list:raise RuntimeError('Empty segmentation: '+row['source'])
        cnt=max(contour_list,key=cv2.contourArea)
        poly=cv2.approxPolyDP(cnt,0.0017*cv2.arcLength(cnt,True),True).reshape(-1,2)
        if len(poly)<3:raise ValueError('Invalid polygon: '+row['source'])
        h,w=binary.shape;points=[]
        for x,y in poly:
            points.extend((min(1,max(0,float(x)/w)), min(1,max(0,float(y)/h))))
        unique_stem=f'{label}__{stem}'
        new_img=OUT/'images'/split/(unique_stem+img.suffix.lower())
        new_img.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(img,new_img)
        text=OUT/'labels'/split/(unique_stem+'.txt')
        text.parent.mkdir(parents=True,exist_ok=True)
        text.write_text('0 '+' '.join(f'{val:.6f}' for val in points)+'\n',encoding='utf-8')
        count+=1
    config={'path':OUT.resolve().as_posix(),'train':'images/train','val':'images/val',
            'test':'images/test','names':{0:'lid'}}
    (OUT/'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print(f'Exported {count} one-lid segmentation polygon labels to {OUT}')
    print('Remember: masks are approved geometric/manual annotations; benchmark true mask IoU on separately human-drawn references.')

if __name__=='__main__':main()
