r"""Human correction for difficult masks. Click center+outer edge (circle) or polygon vertices.
Examples:
  python .\src\04_manual_override.py --image train/defective/example.png --shape circle
  python .\src\04_manual_override.py --image val/good/example.png --shape polygon
Left click to add, Z undo, R reset, S save, ESC quit. Then rerun the builder.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from rojin_pipeline import read_bgr, segment_lid

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'TinplateLidDATA2024'
OVERRIDE_PATH = ROOT / 'config' / 'rojin_manual_overrides.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Source path relative to RAW, e.g. train/defective/a.png')
    parser.add_argument('--shape', choices=['circle','polygon'], default='circle')
    args = parser.parse_args()
    source = (RAW / args.image).resolve()
    if RAW.resolve() not in source.parents or not source.is_file():
        raise ValueError(f'Only an existing image inside the RAW folder is allowed: {source}')
    img = read_bgr(source)
    h,w = img.shape[:2]
    scale = min(1, 1100/w, 750/h)
    disp_w, disp_h = round(w*scale), round(h*scale)
    points = []
    window = 'ROJIN MANUAL MASK: left click; Z undo; R reset; S save; ESC quit'
    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if args.shape == 'circle' and len(points) >= 2:
                points.clear()
            points.append((float(x/scale), float(y/scale)))
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window,click)
    while True:
        preview = cv2.resize(img,(disp_w,disp_h),interpolation=cv2.INTER_AREA)
        p = [(round(x*scale),round(y*scale)) for x,y in points]
        if args.shape == 'circle' and p:
            cv2.circle(preview,p[0],3,(0,255,0),-1)
            if len(p)==2:
                cv2.circle(preview,p[0],round(np.linalg.norm(np.array(p[0])-p[1])),(0,255,0),2)
        if args.shape == 'polygon' and p:
            cv2.polylines(preview,[np.array(p,np.int32)],False,(0,255,0),2)
            for pt in p:cv2.circle(preview,pt,3,(0,255,0),-1)
        cv2.imshow(window,preview)
        k=cv2.waitKey(30) & 0xFF
        if k==27:
            print('Cancelled; no files changed.');break
        if k in (ord('r'),ord('R')): points.clear()
        if k in (ord('z'),ord('Z')) and points: points.pop()
        if k in (ord('s'),ord('S')):
            if args.shape=='circle' and len(points)==2:
                a,b=points
                override={'type':'circle','cx':a[0],'cy':a[1],
                          'r':float(np.linalg.norm(np.asarray(a)-np.asarray(b)))}
            elif args.shape=='polygon' and len(points)>=3:
                override={'type':'polygon','points':points}
            else:
                print('Add 2 circle points or at least 3 polygon points before saving.');continue
            result=segment_lid(img,override=override)
            if result.status!='OK':
                print('Rejected correction:',result.reason);continue
            OVERRIDE_PATH.parent.mkdir(parents=True,exist_ok=True)
            current=json.loads(OVERRIDE_PATH.read_text(encoding='utf-8')) if OVERRIDE_PATH.exists() else {}
            rel=source.relative_to(RAW.resolve()).as_posix()
            current[rel]=override
            OVERRIDE_PATH.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
            print('Saved manual correction:',rel)
            print('Rebuild processed dataset using 01_build_masked.py --full --overwrite (raw files remain untouched).')
            break
    cv2.destroyAllWindows()

if __name__=='__main__':main()
