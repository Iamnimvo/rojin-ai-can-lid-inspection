"""Verify Rojin masked train/validation/test vs original random80_20 split.

Run: python src/11_verify_dataset_integrity.py
Read-only: does not modify any image or dataset directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'processed'
REFERENCE = BASE / 'TinplateLidDATA2024_masked_random80_20'
TRAIN_ROOT = BASE / 'TinplateLidDATA2024_masked_random80'
TEST_ROOT = BASE / 'TinplateLidDATA2024_masked_random20'
REPORT_DIR = ROOT / 'results' / 'dataset_integrity_audit'
CLASSES = ('good', 'defective')
EXT = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def scan(root: Path, splits: tuple[str, ...], group: str):
    rows = []
    for split in splits:
        for label in CLASSES:
            folder = root / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f'Missing required directory: {folder}')
            for f in sorted(folder.iterdir(), key=lambda x: x.name.casefold()):
                if f.is_file() and f.suffix.lower() in EXT:
                    rows.append({'group': group, 'split': split, 'label': label,
                                 'name': f.name, 'path': str(f), 'sha256': file_sha256(f)})
    return rows


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def counts(rows):
    return {'total': len(rows), 'by_split_class': dict(sorted(Counter(
        f"{r['split']}/{r['label']}" for r in rows).items()))}


def difference(a, b, key):
    return list((Counter(key(r) for r in a) - Counter(key(r) for r in b)).elements())


def main():
    print('ROJIN - DATASET INTEGRITY AUDIT (read-only)')
    print('Checking folders and SHA-256 hashes. This may take a little time...')
    ref = scan(REFERENCE, ('train', 'test'), 'reference')
    train = scan(TRAIN_ROOT, ('train', 'val'), 'training')
    test = scan(TEST_ROOT, ('test',), 'testing')
    combined = train + test

    # Identity by both content and label; detect missing copies even with renamed filenames.
    labeled_key = lambda r: (r['label'], r['sha256'])
    missing = difference(ref, combined, labeled_key)
    extra = difference(combined, ref, labeled_key)
    ref_train = [r for r in ref if r['split'] == 'train']
    ref_test = [r for r in ref if r['split'] == 'test']
    ref_train_missing = difference(ref_train, train, labeled_key)
    ref_train_extra = difference(train, ref_train, labeled_key)
    ref_test_missing = difference(ref_test, test, labeled_key)
    ref_test_extra = difference(test, ref_test, labeled_key)

    hashes_training = {r['sha256'] for r in train}
    hashes_testing = {r['sha256'] for r in test}
    cross_overlap = sorted(hashes_training & hashes_testing)
    hashes_train = {r['sha256'] for r in train if r['split'] == 'train'}
    hashes_val = {r['sha256'] for r in train if r['split'] == 'val'}
    train_val_overlap = sorted(hashes_train & hashes_val)

    # A checksum appearing more than once anywhere in the active split is suspicious,
    # even if it comes from distinct filenames (e.g. re-encoded copies only if byte-identical).
    dup_map = defaultdict(list)
    for r in combined:
        dup_map[r['sha256']].append(r)
    duplicates = {h: rows for h, rows in dup_map.items() if len(rows) > 1}
    # Duplicate filenames can exist across folders; report separately.
    name_map = defaultdict(list)
    for r in combined:
        name_map[r['name'].casefold()].append(r)
    same_names = {n: rs for n, rs in name_map.items() if len(rs) > 1}

    coverage_ok = not missing and not extra and len(ref) == len(combined)
    exact_partition_ok = (not ref_train_missing and not ref_train_extra
                          and not ref_test_missing and not ref_test_extra)
    no_leakage = not cross_overlap and not train_val_overlap
    count_ok = (len(ref) == 1307 and len(train) == 1046 and len(test) == 261)
    # This check is stronger than count-only; duplicated files also break validity.
    passed = coverage_ok and exact_partition_ok and no_leakage and not duplicates and count_ok

    summary = {
        'checked_at': datetime.now().isoformat(timespec='seconds'),
        'reference': str(REFERENCE), 'training_dataset': str(TRAIN_ROOT),
        'testing_dataset': str(TEST_ROOT),
        'reference_counts': counts(ref), 'training_counts': counts(train),
        'testing_counts': counts(test),
        'coverage_match': coverage_ok, 'original_80_20_membership_preserved': exact_partition_ok,
        'train_test_content_overlap_count': len(cross_overlap),
        'train_val_content_overlap_count': len(train_val_overlap),
        'duplicate_sha256_count': len(duplicates),
        'duplicate_filename_count': len(same_names),
        'missing_from_new_combined_count': len(missing),
        'extra_in_new_combined_count': len(extra),
        'reference_train_missing_from_train_val_count': len(ref_train_missing),
        'reference_train_extra_in_train_val_count': len(ref_train_extra),
        'reference_test_missing_from_new_test_count': len(ref_test_missing),
        'reference_test_extra_in_new_test_count': len(ref_test_extra),
        'expected_counts_ok': count_ok,
        'passed': passed,
        'limitations': [
            'SHA-256 finds byte-identical files; visually identical images with different encoding may need perceptual duplicate checking.',
            'Exact source coverage and no content overlap do not prove independence of near-adjacent frames or common recording sessions.',
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / 'integrity_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (REPORT_DIR / 'file_inventory.csv').open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=['group', 'split', 'label', 'name', 'path', 'sha256'])
        writer.writeheader()
        writer.writerows(ref + combined)
    with (REPORT_DIR / 'issues.txt').open('w', encoding='utf-8') as fh:
        for title, items in (
            ('MISSING FROM COMBINED (label,sha256)', missing),
            ('EXTRA IN COMBINED (label,sha256)', extra),
            ('REF TRAIN MISSING IN NEW TRAIN+VAL', ref_train_missing),
            ('NEW TRAIN+VAL NOT IN REF TRAIN', ref_train_extra),
            ('REF TEST MISSING IN NEW TEST', ref_test_missing),
            ('NEW TEST NOT IN REF TEST', ref_test_extra),
            ('TRAIN VS TEST HASH OVERLAP', cross_overlap),
            ('TRAIN VS VAL HASH OVERLAP', train_val_overlap),
        ):
            fh.write(f'\n## {title} ({len(items)})\n')
            for item in items:
                fh.write(f'{item}\n')
        fh.write(f'\n## DUPLICATE CONTENT WITHIN ACTIVE DATASET ({len(duplicates)})\n')
        for h, rs in duplicates.items():
            fh.write(h + '\n' + '\n'.join('  ' + r['path'] for r in rs) + '\n')
        fh.write(f'\n## DUPLICATE FILENAMES ({len(same_names)})\n')
        for name, rs in same_names.items():
            fh.write(name + '\n' + '\n'.join('  ' + r['path'] for r in rs) + '\n')

    print('\nReference:', summary['reference_counts'])
    print('Train/val:', summary['training_counts'])
    print('Test:', summary['testing_counts'])
    for k in ('coverage_match', 'original_80_20_membership_preserved',
              'train_test_content_overlap_count', 'train_val_content_overlap_count',
              'duplicate_sha256_count', 'missing_from_new_combined_count',
              'extra_in_new_combined_count', 'expected_counts_ok', 'passed'):
        print(f'{k}: {summary[k]}')
    print('\nReports saved to:', REPORT_DIR)
    print('RESULT:', 'PASS - dataset verified' if passed else 'FAIL - inspect issues.txt')


if __name__ == '__main__':
    main()
