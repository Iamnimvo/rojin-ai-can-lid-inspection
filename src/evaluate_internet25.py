"""Rojin: honest external 25-IMAGE evaluation, without refitting thresholds.
Run: python .\\src\\evaluate_internet25.py --repeats 3 --batch 8
Requires a complete downloaded set prepared by download_and_prepare_internet25.py.
"""
from __future__ import annotations
import argparse
import csv
import statistics
import time
from collections import Counter
from pathlib import Path

from PIL import Image
import torch
from ultralytics import YOLO

ROOT = Path(r'D:\rojin_lid_ai')
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'internet25_manifest.csv'
MODEL = ROOT / 'results' / 'yolo11n_cls_baseline' / 'weights' / 'best.pt'
READY = ROOT / 'data' / 'internet25_external' / 'ready_for_model'
OUTPUT = ROOT / 'results' / 'internet25_external'
LOW = 0.15
HIGH = 0.45
IMG_SIZE = 640


def action(p):
    if p <= LOW: return 'GOOD'
    if p >= HIGH: return 'DEFECT'
    return 'REVIEW'


def synchronize(device):
    if device != 'cpu': torch.cuda.synchronize(device)


def load_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def run(model, rows, device, defective_idx, repeat_batch_size):
    """The entire timed pass includes per-image file read/decode and model prediction."""
    outputs = []
    synchronize(device)
    start = time.perf_counter()
    for start_idx in range(0, len(rows), repeat_batch_size):
        chunk = rows[start_idx:start_idx+repeat_batch_size]
        images = [load_rgb(r['_path']) for r in chunk]
        # An actual image list lets Ultralytics process a batch rather than 25 single predicts.
        results = model.predict(source=images, imgsz=IMG_SIZE, device=device, verbose=False)
        if len(results) != len(chunk):
            raise RuntimeError(f'Result count mismatch: {len(results)} for {len(chunk)} inputs')
        for row, res in zip(chunk, results):
            prob = res.probs
            if prob is None: raise RuntimeError(f'No classification probabilities: {row["id"]}')
            p = float(prob.data[defective_idx].item())
            outputs.append({'id':row['id'], 'group':row['group'], 'p_defect':p,
                            'binary_pred': 'defective' if p>=0.5 else 'good',
                            'action':action(p), 'description':row['description'],
                            'limitations':row['limitations'], 'source_image_url':row['image_url']})
    synchronize(device)
    duration = time.perf_counter() - start
    return outputs, duration


def summarize(rows):
    known = [r for r in rows if r['group'] in ('good','defective')]
    review_candidates = [r for r in rows if r['group'] == 'review_candidate']
    binary_correct = sum(r['binary_pred'] == r['group'] for r in known)
    decisive = [r for r in known if r['action'] != 'REVIEW']
    decisive_correct = sum((r['action']=='GOOD' and r['group']=='good') or
                           (r['action']=='DEFECT' and r['group']=='defective') for r in decisive)
    fa = sum(r['group']=='defective' and r['action']=='GOOD' for r in known)
    fr = sum(r['group']=='good' and r['action']=='DEFECT' for r in known)
    known_review = sum(r['action']=='REVIEW' for r in known)
    review_hits = sum(r['action']=='REVIEW' for r in review_candidates)
    coverage = len(decisive)
    acc_decisive = (f'{decisive_correct}/{coverage} = {decisive_correct/coverage:.2%}'
                    if coverage else 'undefined (no decisive calls)')
    return (
        f'Thresholds: GOOD p<= {LOW}; REVIEW {LOW}<p<{HIGH}; DEFECT p>= {HIGH}\n'
        'This challenge uses 25 distinct public image URLs: 8 catalog photos visually normal, '
        '8 documented visible defects, 9 candidates that warrant contextual/manual review.\n'
        'REVIEW CANDIDATES have no independently verified third-class label.\n'
        f'Binary top-1 accuracy on 16 provisionally labeled images at p>=0.5: {binary_correct}/16 = {binary_correct/16:.2%}\n'
        f'Decisive GOOD/DEFECT coverage on these 16: {coverage}/16 = {coverage/16:.2%}\n'
        f'Correctness among decisive decisions: {acc_decisive}\n'
        f'Known-set GOOD/DEFECT sent to REVIEW: {known_review}/16\n'
        f'False accept (documented defect -> GOOD): {fa}/8\n'
        f'False reject (visually normal catalog lid -> DEFECT): {fr}/8\n'
        f'REVIEW-CANDIDATE sources receiving REVIEW: {review_hits}/9 (diagnostic, NOT model accuracy)\n'
        f'Actions over all 25: {dict(Counter(r["action"] for r in rows))}\n'
        'NOTE: Source photographs vary in lighting and viewpoint; some show partial ends, stacked lids or can-body edges.\n'
    )



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    args=parser.parse_args()
    if args.repeats<1 or args.batch<1: parser.error('repeats and batch must be >=1')
    if not MODEL.exists(): raise SystemExit(f'Model missing: {MODEL}')
    if not MANIFEST.exists(): raise SystemExit(f'Manifest missing: {MANIFEST}')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=25 or Counter(r['group'] for r in rows)!={'good':8,'defective':8,'review_candidate':9}:
        raise SystemExit('Manifest must have exactly 8 good, 8 defective and 9 review_candidate records.')
    for r in rows:
        r['_path']=READY / r['group'] / (r['id']+'.png')
        if not r['_path'].is_file(): raise SystemExit(f'MISSING: {r["_path"]}. Download/prepare the full 25 sources first.')
        with Image.open(r['_path']) as im:
            im.verify()
    device=0 if torch.cuda.is_available() else 'cpu'
    print('Loading model (excluded from timing):',MODEL)
    model=YOLO(str(MODEL))
    name_to_idx={str(name).lower():int(idx) for idx,name in model.names.items()}
    if {'good','defective'}-set(name_to_idx): raise SystemExit(f'Unexpected class names: {model.names}')
    defective_idx=name_to_idx['defective']
    # Initialize GPU kernels and predictor outside timing.
    sample=load_rgb(rows[0]['_path'])
    model.predict(source=sample,imgsz=IMG_SIZE,device=device,verbose=False)
    synchronize(device)
    print('GPU:',torch.cuda.get_device_name(0) if device!= 'cpu' else 'CPU')
    print('Thresholds:',LOW,HIGH)

    runs={}
    for name, bs in [('sequential',1),('batched',args.batch)]:
        print(f'\n{name}: {args.repeats} repeats; batch size {bs}')
        times=[];baseline=None
        for i in range(args.repeats):
            preds,seconds=run(model,rows,device,defective_idx,bs)
            if baseline is None: baseline=preds
            else:
                changed=[a['id'] for a,b in zip(baseline,preds) if a['action']!=b['action']]
                if changed: print('WARNING: nonidentical decisions between repeats:',changed)
            times.append(seconds)
            print(f'  Repeat {i+1}: {seconds:.4f}s for 25; {25/seconds:.2f} images/sec')
        runs[name]=(baseline,times)
    seq_preds, seq_times = runs['sequential']
    batch_preds, batch_times = runs['batched']
    different=[(a['id'],a['action'],b['action']) for a,b in zip(seq_preds,batch_preds) if a['action']!=b['action']]

    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['id','group','p_defect','binary_pred','action','description','limitations','source_image_url']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in seq_preds:
            w.writerow({**r,'p_defect':f'{r["p_defect"]:.8f}'})
    s_med=statistics.median(seq_times);b_med=statistics.median(batch_times)
    text=('ROJIN - 25 ORIGINAL INTERNET SOURCES / EXTERNAL CHALLENGE\n'
          'All timing includes image file read + decode + PyTorch model predict + final decision; '
          'excludes model loading, network downloads, crop preparation, localization and GUI.\n\n'
          +summarize(seq_preds)+'\n'
          +f'SEQUENTIAL run times seconds: {[round(x,4) for x in seq_times]}\n'
          +f'SEQUENTIAL median: {s_med:.4f}s /25; throughput {25/s_med:.2f} images/sec\n'
          +f'BATCH ({args.batch}) run times seconds: {[round(x,4) for x in batch_times]}\n'
          +f'BATCH median: {b_med:.4f}s /25; throughput {25/b_med:.2f} images/sec\n'
          +f'Batch-vs-sequential final decision differences: {different}\n')
    (OUTPUT/'report.txt').write_text(text,encoding='utf-8')
    print('\n'+text)
    print('Saved:',OUTPUT/'predictions.csv','and',OUTPUT/'report.txt')

if __name__=='__main__':
    main()
