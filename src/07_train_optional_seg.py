r"""OPTIONAL if geometry failed often and verified masks are available.
python .\src\07_train_optional_seg.py --kind n --batch 8
"""
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'data'/'processed'/'TinplateLid_lidseg_v1'/'dataset.yaml'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('n','s'),default='n')
    ap.add_argument('--batch',type=int,default=8);args=ap.parse_args()
    if not CFG.exists():raise FileNotFoundError('First run 06_export_optional_seg.py')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    name=f'lid_yolo11{args.kind}_seg_v1';dest=ROOT/'results'/name
    if dest.exists():raise FileExistsError(dest)
    model=YOLO(f'yolo11{args.kind}-seg.pt')
    model.train(data=str(CFG),imgsz=640,epochs=50,batch=args.batch,patience=10,
                workers=4,device=0,seed=42,deterministic=True,amp=True,
                project=str(ROOT/'results'),name=name,exist_ok=False,plots=True)
    print('Segmentation checkpoint:',dest/'weights'/'best.pt')
    print('Evaluate mask IoU against independently hand-annotated masks before deployment.')

if __name__=='__main__':main()
