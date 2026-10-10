"""Evaluate the two 25-image demo sets using current thresholds and time the local pipeline.

Latency includes disk image opening/quality check, Ultralytics predict, GPU sync,
and decision logic. It excludes loading the model, warmup, and any future desktop UI
or lid-localization stage. Scenario B's 9 tiny images are auto-REVIEW by a quality
gate without invoking the classifier; do not call this higher neural-net accuracy.
"""
from pathlib import Path
import argparse
import csv
import statistics
import time

from PIL import Image
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'results' / 'yolo11n_cls_baseline' / 'weights' / 'best.pt'
INPUT = ROOT / 'data' / 'demo25'
OUTPUT = ROOT / 'results' / 'demo25'
LOW, HIGH = 0.15, 0.45
MIN_DIM = 224


def sync(device):
    if device != 'cpu':
        torch.cuda.synchronize(0)


def decide(p):
    if p <= LOW: return 'GOOD'
    if p >= HIGH: return 'DEFECT'
    return 'REVIEW'


def evaluate(model, class_idx, scenario, device, repeats):
    root = INPUT / scenario
    manifest = root / 'manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 25: raise ValueError(f'{scenario}: expected 25 manifest entries, got {len(rows)}')
    paths = [root / row['relative_path'] for row in rows]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)

    # Warmup uses a normal-resolution image, outside timed segment.
    warmup = next(p for p in paths if p.parent.name != 'review')
    for _ in range(3):
        model.predict(source=str(warmup), imgsz=640, device=device, verbose=False)
    sync(device)

    total_runs = []
    first_results = []
    per_image_times = []
    for rep in range(repeats):
        sync(device)
        start_batch = time.perf_counter()
        current = []
        for item, path in zip(rows, paths):
            sync(device)
            t0 = time.perf_counter()
            with Image.open(path) as im:
                w, h = im.size
                im.load()
            # This is a deliberately simple DEMO policy, not a universal QC standard.
            if min(w, h) < MIN_DIM:
                p_defect, model_label, inference_ms, action = None, 'NOT_RUN', None, 'REVIEW'
                review_reason = 'LOW_RESOLUTION'
            else:
                r = model.predict(source=str(path), imgsz=640, device=device, verbose=False)[0]
                sync(device)
                p_defect = float(r.probs.data[class_idx['defective']].item())
                model_label = 'defective' if p_defect >= .5 else 'good'
                inference_ms = float(r.speed.get('inference', 0.0))
                action = decide(p_defect)
                review_reason = 'LOW_CONFIDENCE' if action == 'REVIEW' else ''
            sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            current.append(dict(image=item['relative_path'], actual=item['actual_class'],
                                expected_action=item['expected_action'], quality=item['quality'],
                                predicted_binary=model_label,
                                p_defect='' if p_defect is None else f'{p_defect:.6f}',
                                action=action, review_reason=review_reason,
                                model_inference_ms='' if inference_ms is None else f'{inference_ms:.3f}',
                                image_total_ms=f'{elapsed_ms:.3f}'))
        sync(device)
        total_ms = (time.perf_counter() - start_batch) * 1000
        total_runs.append(total_ms)
        if rep == 0:
            first_results = current
            per_image_times = [float(x['image_total_ms']) for x in current]

    results_dir = OUTPUT / scenario
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(first_results[0])); writer.writeheader(); writer.writerows(first_results)

    median_ms = statistics.median(total_runs)
    actions = {name: sum(r['action'] == name for r in first_results) for name in ['GOOD','REVIEW','DEFECT']}
    evaluated = [r for r in first_results if r['predicted_binary'] != 'NOT_RUN']
    binary_correct = sum(r['predicted_binary'] == r['actual'] for r in evaluated)
    false_accept = sum(r['actual'] == 'defective' and r['action'] == 'GOOD' for r in first_results)
    false_reject = sum(r['actual'] == 'good' and r['action'] == 'DEFECT' for r in first_results)
    quality_reviews = sum(r['review_reason'] == 'LOW_RESOLUTION' for r in first_results)
    if len(per_image_times) >= 2:
        p95 = sorted(per_image_times)[int(.95 * (len(per_image_times)-1))]
    else: p95 = per_image_times[0]
    expected_action_match = sum(r['action'] == r['expected_action'] for r in first_results)
    lines = [
        f'Scenario: {scenario}', f'Thresholds: GOOD <= {LOW}, DEFECT >= {HIGH}, otherwise REVIEW',
        f'Images: 25; classified: {len(evaluated)}; quality-gated: {quality_reviews}',
        f'Binary model accuracy ON CLASSIFIED images at P(defect)>=0.5: {binary_correct}/{len(evaluated)}',
        f'Actions: {actions}', f'False accept (defective -> GOOD): {false_accept}',
        f'False reject (good -> DEFECT): {false_reject}',
        f'Expected-action agreement (includes manually engineered quality cases): {expected_action_match}/25',
        f'Timed repetitions (model loaded + GPU warmed first): {repeats}',
        f'Wall-clock time for 25 files each run (ms): {[round(t,3) for t in total_runs]}',
        f'Median complete measured 25-image processing time: {median_ms:.3f} ms = {median_ms/1000:.4f} s',
        f'Median measured throughput: {25/(median_ms/1000):.2f} images/sec',
        f'Mean per-image measured processing: {statistics.mean(per_image_times):.3f} ms; p95: {p95:.3f} ms',
        'Scope: local file read, quality gate, model predict if eligible, decision; excludes model loading, GUI, lid localization.',
        'NOTE: review candidates in B are deliberately downsampled variants of underlying good/defective images.',
        'NOTE: these 25 source photos have previously appeared in your 65-image test set, not a new independent test.',
    ]
    report = '\n'.join(lines)
    (results_dir / 'report.txt').write_text(report, encoding='utf-8')
    print('\n'+report+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1: raise ValueError('repeats must be positive')
    if not MODEL.exists(): raise FileNotFoundError(MODEL)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print('Loading model (excluded from timing):', MODEL)
    model = YOLO(str(MODEL))
    names = {str(v).strip().lower(): int(k) for k,v in model.names.items()}
    if set(names) != {'good', 'defective'}: raise RuntimeError(f'Unexpected classes: {model.names}')
    print('Model classes:', model.names, 'device:', device)
    scenarios = {'A': 'A_binary_13_12', 'B': 'B_8_8_9_quality'}
    selection = [args.scenario] if args.scenario != 'both' else ['A', 'B']
    for key in selection: evaluate(model, names, scenarios[key], device, args.repeats)

if __name__ == '__main__':
    main()
