r"""Choose exploratory LOW/HIGH using masked VALIDATION ONLY, never TEST or external set.
Produces all candidates and proposes a zero-error candidate when the validation labels permit.

python .\src\08_calibrate_validation.py --model results/yolo11n_cls_masked_v1/weights/best.pt
"""
import argparse,csv,json
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
from rojin_pipeline import read_bgr

ROOT=Path(__file__).resolve().parents[1]
VAL=ROOT/'data'/'processed'/'TinplateLidDATA2024_masked_v1'/'val'
OUTPUT=ROOT/'results'/'masked_val_calibration_v1'

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model',default=str(ROOT/'results'/'yolo11n_cls_masked_v1'/'weights'/'best.pt'))
 ap.add_argument('--batch',type=int,default=8)
 ap.add_argument('--max-review',type=float,default=0.20,help='Maximum fraction to route to human review')
 args=ap.parse_args()
 if not (0<=args.max_review<1):ap.error('max-review must be in [0,1)')
 paths=[];true=[]
 for c in ('good','defective'):
  for p in sorted((VAL/c).glob('*.png')):
   paths.append(p);true.append(c)
 if not paths:raise RuntimeError('No masked validation photos')
 m=YOLO(args.model);index={str(v).lower():int(k) for k,v in m.names.items()}['defective']
 device=0 if torch.cuda.is_available() else 'cpu'
 probs=[]
 for start in range(0,len(paths),args.batch):
  ims=[read_bgr(p) for p in paths[start:start+args.batch]]
  pred=m.predict(source=ims,imgsz=640,device=device,verbose=False)
  probs.extend(float(q.probs.data[index].item()) for q in pred)
 OUTPUT.mkdir(parents=True,exist_ok=True)
 with (OUTPUT/'validation_probabilities.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.writer(f);writer.writerow(['source','actual','p_defect']);writer.writerows(zip([p.relative_to(VAL).as_posix() for p in paths],true,probs))
 candidates=[]
 # Coarse exploration; choose thresholds by validation constraints, not test feedback.
 for low in np.round(np.arange(0.02,0.50,0.02),2):
  for high in np.round(np.arange(max(.04,low+.02),.991,.02),2):
   actions=['GOOD' if p<=low else 'DEFECT' if p>=high else 'REVIEW' for p in probs]
   fa=sum(t=='defective' and a=='GOOD' for t,a in zip(true,actions))
   fr=sum(t=='good' and a=='DEFECT' for t,a in zip(true,actions))
   review=sum(a=='REVIEW' for a in actions)
   candidates.append({'low':float(low),'high':float(high),'false_accept':fa,'false_reject':fr,
                      'review':review,'review_rate':review/len(paths),
                      'decisive_correct':len(paths)-review-fa-fr})
 with (OUTPUT/'candidate_thresholds.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(candidates[0]));writer.writeheader();writer.writerows(candidates)
 feasible=[x for x in candidates if x['false_accept']==0 and x['false_reject']==0 and x['review_rate']<=args.max_review]
 feasible.sort(key=lambda x:(x['review'],-(x['high']-x['low'])))
 if feasible:
  best=feasible[0]
  (OUTPUT/'proposal.json').write_text(json.dumps({'validation_only':True,'num_val_images':len(paths),
    'proposed_thresholds':best,'external_validation_required':True},indent=2),encoding='utf-8')
  print('VALIDATION-ONLY candidate:',best)
  print('Test/new external images must NOT be used to tune these thresholds.')
 else:
  print('No zero-error threshold pair within the review budget; inspect candidate_thresholds.csv. Do not force a misleading proposal.')
 print('Calibration analysis saved:',OUTPUT)

if __name__=='__main__':main()
