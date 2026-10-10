r"""Quick preflight: project structure, image counts, CUDA, storage."""
from pathlib import Path
import platform
import shutil
import cv2
import torch
import ultralytics
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'/'TinplateLidDATA2024'
SUFFIXES={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
print('Project:',ROOT)
print('Python:',platform.python_version(),'OpenCV:',cv2.__version__,'Ultralytics:',ultralytics.__version__)
print('CUDA:',torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:',torch.cuda.get_device_name(0),'VRAM(GB):',round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))
print('Free disk GB:',round(shutil.disk_usage(ROOT).free/1024**3,2))
for split in ('train','val','test'):
 for label in ('good','defective'):
    folder=RAW/split/label
    if not folder.is_dir():raise FileNotFoundError(folder)
    files=[f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUFFIXES]
    print(f'{split}/{label}: {len(files)} images')
print('Preflight completed. Do a pilot and check the green outlines before processing the full dataset.')
