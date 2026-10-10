ROJIN PROJECT REVIEW PACKAGE
============================
This ZIP was generated from D:\rojin_lid_ai for technical review and an
English industrial Proof-of-Concept proposal concerning the final model:
  results/yolo11n_cls_masked_random80_v2/weights/best.pt

IMPORTANT: The TinplateLidDATA2024 dataset is PUBLIC/EXTERNAL, not a verified
Rojin factory dataset. Performance on that dataset does not establish field
performance on actual Rojin cans or a complete production line.

Selection policy:
- All small Python source/configuration files in src/ and some root/models metadata.
- The named final training run, best.pt, and charts if present.
- The named final test run: CSV/JSON/report/images at top level.
- Up to three example predicted images from each of good/defective/review.
- Dataset split manifests, verification reports, masking metadata.
- A small set of masked samples and optional raw/overlay examples.

NOT INCLUDED:
- .venv, full data folders, all 1307 training/test images, entire results history,
  redundant model weights, caches, or large temporary files.

The package is for review only. The original project is unchanged.
See package_manifest.json for included file SHA-256 hashes and any warnings.
