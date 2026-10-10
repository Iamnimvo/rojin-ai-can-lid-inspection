r"""End-to-end masks + classification, with failed masks routed to REVIEW.

Examples:
 python .\src\03_evaluate_masked.py --source data/processed/TinplateLidDATA2024_masked_v1/test --already-masked
 python .\src\03_evaluate_masked.py --source data/external_new --batch 8 --repeats 3
 python .\src\03_evaluate_masked.py --source data/external_new --seg-model results/lid_seg_v1/weights/best.pt
IMPORTANT: --already-masked measures the classifier + file IO, NOT live segmentation.
"""
import argparse
import csv
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from rojin_pipeline import VALID_EXT, read_bgr, segment_lid, save_image, overlay

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MODEL=ROOT/'results'/'yolo11n_cls_masked_v1'/'weights'/'best.pt'
LOW,HIGH=0.15,0.45  # EXPLORATORY; choose production thresholds using masked VALIDATION only.


def label_from_path(p,source):
    relative=p.relative_to(source)
    components={part.lower() for part in relative.parts[:-1]}
    if 'defective' in components:return 'defective'
    if 'good' in components:return 'good'
    return ''


def decision(p, low, high):
    if p<=low:return 'GOOD'
    if p>=high:return 'DEFECT'
    return 'REVIEW'


def synchronize():
    if torch.cuda.is_available():torch.cuda.synchronize()


def process(files,source,model,defect_idx,device,batch_size,already_masked,seg_model,
            output_preview=None,low=LOW,high=HIGH):
    records=[]
    timing={'preprocess':0.0,'model':0.0}
    pending=[]

    def flush():
        nonlocal pending
        if not pending:return
        synchronize();t0=time.perf_counter()
        prediction=model.predict(source=[x['crop'] for x in pending],imgsz=640,
                                 device=device,verbose=False)
        synchronize();timing['model']+=time.perf_counter()-t0
        if len(prediction)!=len(pending):raise RuntimeError('Batch output count mismatch')
        for item,result in zip(pending,prediction):
            prob=float(result.probs.data[defect_idx].item())
            item['record'].update(p_defect=round(prob,6),
                                  binary_prediction=('defective' if prob>=0.5 else 'good'),
                                  action=decision(prob,low,high), segment_status='OK')
        pending=[]

    for p in files:
        t0=time.perf_counter()
        expected=label_from_path(p,source)
        record={'image':p.relative_to(source).as_posix(),'expected':expected,
                'segment_status':'REVIEW','segment_reason':'','p_defect':'',
                'binary_prediction':'','action':'REVIEW'}
        records.append(record)
        try:
            image=read_bgr(p)
            if already_masked:
                if image.shape[:2]!=(704,704):
                    raise ValueError('MASKED_INPUT_MUST_BE_704x704')
                crop=image
                seg=None
            else:
                seg=segment_lid(image,seg_model=seg_model)
                if seg.status!='OK':
                    record['segment_reason']=seg.reason
                    timing['preprocess']+=time.perf_counter()-t0
                    if output_preview:
                        save_image(output_preview/'review'/p.relative_to(source).with_suffix('.jpg'),overlay(image,seg))
                    continue
                crop=seg.crop
            timing['preprocess']+=time.perf_counter()-t0
            if output_preview:
                save_image(output_preview/'masked'/p.relative_to(source).with_suffix('.png'),crop)
            pending.append({'crop':crop,'record':record})
            if len(pending)>=batch_size:flush()
        except Exception as ex:
            record['segment_reason']='PREPROCESS_ERROR:'+str(ex)[:140]
            timing['preprocess']+=time.perf_counter()-t0
            continue
    flush()
    return records,timing


def summarize(records,times,stage_times,batch,already_masked,low,high):
    n=len(records)
    known=[r for r in records if r['expected'] in ('good','defective')]
    classified=[r for r in known if r['segment_status']=='OK']
    correct=sum(r['binary_prediction']==r['expected'] for r in classified)
    decisive=[r for r in known if r['action'] in ('GOOD','DEFECT')]
    correct_decisive=sum((r['expected']=='good' and r['action']=='GOOD') or
                         (r['expected']=='defective' and r['action']=='DEFECT') for r in decisive)
    fa=sum(r['expected']=='defective' and r['action']=='GOOD' for r in known)
    fr=sum(r['expected']=='good' and r['action']=='DEFECT' for r in known)
    re=sum(r['action']=='REVIEW' for r in known)
    median=statistics.median(times)
    prep_median=statistics.median(t["preprocess"] for t in stage_times)
    model_median=statistics.median(t["model"] for t in stage_times)
    lines=[
      f'Images: {n}; known labels: {len(known)}; segmented and classified with known labels: {len(classified)}',
      f'Binary accuracy at p>=0.5 on classified known labels: {correct}/{len(classified)} = {100*correct/len(classified):.2f}%' if classified else 'Binary accuracy: N/A (no labeled images classified)',
      f'Decisive known-set correctness: {correct_decisive}/{len(decisive)}' if decisive else 'Decisive correctness: N/A',
      f'Known-set review: {re}/{len(known)}; false accepts (defective->GOOD): {fa}; false rejects (good->DEFECT): {fr}',
      f'All actions: {dict(Counter(r["action"] for r in records))}',
      f'Mask/preprocess failures: {sum(r["segment_status"]!="OK" for r in records)}',
      f'Thresholds: GOOD <= {low}, DEFECT >= {high}, else REVIEW (must calibrate on masked val)',
      f'Batch size: {batch}; input already masked: {already_masked}',
      f'Wall times (s; warmed; files read each repeat): {[round(t,4) for t in times]}',
      f'Median full measured runtime: {median:.4f}s / {n} files',
      f'Median full measured throughput: {n/median:.2f} input images/second',
      f'Median preprocessing + file decode: {prep_median:.4f}s; median batched classification: {model_median:.4f}s',
      'Timing includes: original-file IO + decode + lid mask/black crop (unless already-masked) + classifier + threshold decision.',
      'Timing excludes: model loading, UI, image downloads, saving previews/results; image localization of MULTIPLE lids is not implemented.',
      'IMPORTANT: classification only works on ONE lid. REVIEW is a system decision, not a ground-truth third class.',
    ]
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True,help='Directory containing good/defective subfolders, or flat unlabeled photos')
    ap.add_argument('--model',default=str(DEFAULT_MODEL))
    ap.add_argument('--seg-model',help='Optional reviewed-data-trained YOLO11n-seg .pt; otherwise use fast geometric mask')
    ap.add_argument('--already-masked',action='store_true',help='Input has been processed as black-background 704 PNG')
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--low',type=float,default=LOW)
    ap.add_argument('--high',type=float,default=HIGH)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--save-previews',action='store_true',help='Write generated masks AFTER timed repetitions')
    ap.add_argument('--output',default=str(ROOT/'results'/'evaluate_masked_v1'))
    args=ap.parse_args()
    if args.batch<1 or args.repeats<1:ap.error('batch and repeats must be >=1')
    if not (0<=args.low<args.high<=1):ap.error('Require 0 <= --low < --high <= 1')
    if args.seg_model and args.already_masked:ap.error('Cannot combine seg-model and already-masked.')
    source=Path(args.source)
    if not source.is_absolute():source=ROOT/source
    source=source.resolve()
    files=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXT)
    if not files:raise ValueError(f'No images: {source}')
    model_path=Path(args.model)
    if not model_path.exists():raise FileNotFoundError(model_path)
    device=0 if torch.cuda.is_available() else 'cpu'
    model=YOLO(str(model_path))
    cmap={str(name).lower():int(index) for index,name in model.names.items()}
    if {'good','defective'}-set(cmap):raise ValueError(f'Unexpected classifier labels: {model.names}')
    seg_model=YOLO(str(Path(args.seg_model))) if args.seg_model else None
    # Warm up classifier and optional segmenter OUTSIDE measurements.
    dummy=np.zeros((704,704,3),np.uint8)
    model.predict(source=[dummy]*min(args.batch,8),imgsz=640,device=device,verbose=False)
    if seg_model:
        seg_model.predict(source=dummy,imgsz=640,device=device,verbose=False)
    synchronize()
    all_times=[];stage_times=[];rec=None
    for k in range(args.repeats):
        synchronize();start=time.perf_counter()
        records,stage=process(files,source,model,cmap['defective'],device,args.batch,
                          args.already_masked,seg_model,low=args.low,high=args.high)
        synchronize();elapsed=time.perf_counter()-start
        all_times.append(elapsed);stage_times.append(stage)
        if rec is None:rec=records
        else:
            decisions1=[(r['segment_status'],r['binary_prediction'],r['action']) for r in rec]
            decisions2=[(r['segment_status'],r['binary_prediction'],r['action']) for r in records]
            if decisions1!=decisions2:print('WARNING: Prediction decisions differed between repeats!')
        print(f'Repeat {k+1}: {elapsed:.4f}s / {len(files)} files, {len(files)/elapsed:.2f} images/sec')
    report=summarize(rec,all_times,stage_times,args.batch,args.already_masked,args.low,args.high)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rec[0]));writer.writeheader();writer.writerows(rec)
    (output/'report.txt').write_text(report,encoding='utf-8')
    if args.save_previews:
        process(files,source,model,cmap['defective'],device,args.batch,args.already_masked,
                seg_model,output_preview=output/'previews',low=args.low,high=args.high)
    print('\n'+report+'\nSaved in: '+str(output))

if __name__=='__main__':main()
