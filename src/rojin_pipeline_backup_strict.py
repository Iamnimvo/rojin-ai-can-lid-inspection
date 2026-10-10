r"""Rojin v1: one roughly circular, fully visible lid per image.

Input BGR image -> native-resolution binary mask -> centered 704x704 BGR
with exactly black background. A failure is a REVIEW, never a guessed mask.
The geometry method is not a learned pixel-perfect semantic segmenter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np
import torch

OUT_SIZE = 704
VALID_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class SegmentResult:
    status: str
    method: str = ""
    reason: str = ""
    mask: np.ndarray | None = None
    crop: np.ndarray | None = None
    edge_support: float = 0.0
    area_ratio: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0


def read_bgr(path: str | Path) -> np.ndarray:
    """Unicode-safe cv2 loader for Windows paths; detect failed reads."""
    p = Path(path)
    a = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(a, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise ValueError(f"Cannot decode image: {p}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Cannot encode: {p}")
    data.tofile(str(p))


def _ring_support(edge_dilated: np.ndarray, cx: float, cy: float, r: float) -> float:
    h, w = edge_dilated.shape
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    x = np.round(cx + r * np.cos(theta)).astype(np.int32)
    y = np.round(cy + r * np.sin(theta)).astype(np.int32)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.mean() < 0.99:
        return 0.0
    return float(np.mean(edge_dilated[y[inside], x[inside]] > 0))


def _auto_circle(image: np.ndarray) -> tuple[float, float, float, str, float] | str:
    h0, w0 = image.shape[:2]
    factor = min(1.0, 384.0 / max(h0, w0))
    w, h = max(1, round(w0 * factor)), max(1, round(h0 * factor))
    small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA) if factor < 1 else image
    short = min(h, w)
    if short < 150:
        return "IMAGE_TOO_SMALL_FOR_AUTO_MASK"
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    contrast = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(contrast, 65, 145)
    edge_dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    min_r, max_r = int(short * 0.15), int(short * 0.49)
    candidates: list[tuple[float,float,float,str]] = []
    # Hough may identify INNER concentric rings; select a supported OUTER ring below.
    for threshold in (38, 30, 23):
        circles = cv2.HoughCircles(
            contrast, cv2.HOUGH_GRADIENT, dp=1.25,
            minDist=max(15, int(short * 0.28)), param1=115,
            param2=threshold, minRadius=min_r, maxRadius=max_r,
        )
        if circles is not None:
            for cx, cy, r in circles[0]:
                candidates.append((float(cx), float(cy), float(r), "hough"))
        if len(candidates) >= 2:
            break
    # Threshold/contour fallback for high-contrast backgrounds.
    for binary in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, bw = cv2.threshold(contrast, 0, 255, binary | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.045 * h * w or area > 0.77 * h * w:
                continue
            perim = cv2.arcLength(contour, True)
            if perim <= 0 or 4 * np.pi * area / perim ** 2 < 0.56:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(contour)
            if not (min_r <= r <= max_r):
                continue
            candidates.append((cx, cy, r, "contour"))
    if not candidates:
        return "NO_CIRCULAR_LID_FOUND"

    # Cluster concentric rings by center. Distinct strong groups mean multiple lids.
    groups: list[list[tuple[float,float,float,str]]] = []
    for c in sorted(candidates, key=lambda z: -z[2]):
        for group in groups:
            if math.hypot(c[0] - group[0][0], c[1] - group[0][1]) < short * 0.105:
                group.append(c)
                break
        else:
            groups.append([c])
    scored: list[tuple[float,float,float,float,str]] = []
    for group in groups:
        center_x = float(np.median([c[0] for c in group]))
        center_y = float(np.median([c[1] for c in group]))
        safe_r = min(center_x, center_y, w - 1 - center_x, h - 1 - center_y)
        if safe_r <= min_r:
            continue
        radii = set()
        for cx, cy, r, kind in group:
            for multiple in (0.97, 1.0, 1.04, 1.09, 1.15, 1.22, 1.30):
                rr = round(r * multiple)
                if min_r <= rr <= min(max_r, safe_r * 0.985):
                    radii.add(rr)
        # Search for the outermost well-supported circular contour around this center.
        for rr in range(min_r, min(max_r, int(safe_r * 0.985)), 2):
            radii.add(rr)
        ring_candidates = []
        for rr in radii:
            support = _ring_support(edge_dilated, center_x, center_y, rr)
            if support >= 0.23:
                ring_candidates.append((rr, support))
        if not ring_candidates:
            continue
        peak_support = max(s for _, s in ring_candidates)
        eligible = [(r, s) for r, s in ring_candidates if s >= max(0.38, peak_support * 0.45)]
        if not eligible:
            eligible = ring_candidates
        # Choose outermost reliable ring, avoiding inner lid embossing whenever possible.
        radius, support = max(eligible, key=lambda rs: (rs[0], rs[1]))
        offset = math.hypot(center_x - w / 2, center_y - h / 2) / short
        quality = 0.78 * support + 0.28 * (radius / short) - 0.12 * offset
        scored.append((quality, center_x, center_y, float(radius), f"auto;support={support:.3f}"))
    if not scored:
        return "CIRCULAR_EDGE_NOT_RELIABLE"
    scored.sort(reverse=True)
    if len(scored) >= 2:
        # Do not silently pick one out of multiple separate similarly sized lids.
        a, b = scored[:2]
        dist = math.hypot(a[1] - b[1], a[2] - b[2])
        if dist > 0.55 * (a[3] + b[3]) and b[0] >= a[0] - 0.12:
            return "MULTIPLE_LID_CANDIDATES"
    _, cx, cy, r, method = scored[0]
    support = float(method.split("=")[-1])
    if support < 0.30:
        return "LOW_BOUNDARY_SUPPORT"
    # A small rim allowance avoids systematically clipping a metallic edge.
    r *= 1.015
    if cx-r < 2 or cy-r < 2 or cx+r > w-3 or cy+r > h-3:
        return "LID_TOUCHES_IMAGE_BORDER"
    if (np.pi * r**2) / (w*h) < 0.075:
        return "LID_TOO_SMALL_IN_FRAME"
    return cx / factor, cy / factor, r / factor, "geometric_circle", support


def _finish(image: np.ndarray, mask: np.ndarray, method: str,
            support: float = 0.0, radius: float = 0.0) -> SegmentResult:
    h, w = mask.shape
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    nonzero = cv2.countNonZero(mask)
    if not nonzero:
        return SegmentResult(status="REVIEW", reason="EMPTY_MASK")
    x1, y1, bw, bh = cv2.boundingRect(mask)
    x2, y2 = x1 + bw - 1, y1 + bh - 1
    if x1 <= 1 or y1 <= 1 or x2 >= w-2 or y2 >= h-2:
        return SegmentResult(status="REVIEW", reason="LID_TOUCHES_IMAGE_BORDER")
    area_ratio = nonzero/(w*h)
    if not (0.065 <= area_ratio <= 0.79):
        return SegmentResult(status="REVIEW", reason="IMPLAUSIBLE_MASK_AREA", area_ratio=area_ratio)
    if min(bw, bh)/max(bw, bh) < 0.67:
        return SegmentResult(status="REVIEW", reason="VERY_OBLIQUE_OR_NON_CIRCULAR", area_ratio=area_ratio)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = int(np.ceil(max(bw, bh) * 1.20))
    side += side % 2
    xx1, yy1 = int(round(cx-side/2)), int(round(cy-side/2))
    xx2, yy2 = xx1+side, yy1+side
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0,xx1), max(0,yy1), min(w,xx2), min(h,yy2)
    if sx2 <= sx1 or sy2 <= sy1:
        return SegmentResult(status="REVIEW", reason="INVALID_CROP")
    roi = image[sy1:sy2, sx1:sx2]
    cropped_mask = mask[sy1:sy2, sx1:sx2]
    dst_roi = canvas[(sy1-yy1):(sy2-yy1), (sx1-xx1):(sx2-xx1)]
    cv2.copyTo(roi, cropped_mask, dst_roi)
    interp = cv2.INTER_AREA if side >= OUT_SIZE else cv2.INTER_LINEAR
    crop = cv2.resize(canvas, (OUT_SIZE, OUT_SIZE), interpolation=interp)
    return SegmentResult(status="OK", method=method, mask=mask, crop=crop,
                         edge_support=support, area_ratio=area_ratio,
                         cx=cx, cy=cy, radius=radius)


def segment_lid(image: np.ndarray, override: dict | None = None,
                seg_model=None) -> SegmentResult:
    """Mutually exclusive geometric, human override, or trained YOLO-seg masks."""
    if image is None or image.ndim != 3:
        return SegmentResult(status="REVIEW", reason="INVALID_IMAGE")
    h, w = image.shape[:2]
    if seg_model is not None:
        # YOLO11n-seg must be trained on *reviewed lid masks*, not unedited COCO.
        pred = seg_model.predict(source=image, imgsz=640, conf=0.35,
                                 retina_masks=True, device=(0 if torch.cuda.is_available() else "cpu"), verbose=False)[0]
        if pred.masks is None or len(pred.masks.data) != 1:
            return SegmentResult(status="REVIEW", reason="SEG_MODEL_NO_UNIQUE_LID")
        mask = pred.masks.data[0].detach().cpu().numpy()
        mask = cv2.resize(mask.astype(np.float32), (w,h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)*255
        return _finish(image, mask, method="trained_yolo_seg")
    if override is not None:
        if override.get("type") == "circle":
            cx, cy, r = float(override["cx"]), float(override["cy"]), float(override["r"])
            if r < 10 or not (0 <= cx < w and 0 <= cy < h):
                return SegmentResult(status="REVIEW", reason="INVALID_MANUAL_CIRCLE")
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (round(cx), round(cy)), round(r), 255, -1, cv2.LINE_8)
            return _finish(image, mask, "manual_circle", radius=r)
        if override.get("type") == "polygon":
            pts = np.asarray(override["points"], dtype=np.float32).reshape(-1,2)
            if len(pts) < 3 or not np.isfinite(pts).all():
                return SegmentResult(status="REVIEW", reason="INVALID_MANUAL_POLYGON")
            mask = np.zeros((h,w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
            return _finish(image, mask, "manual_polygon")
        return SegmentResult(status="REVIEW", reason="UNKNOWN_OVERRIDE_TYPE")
    detected = _auto_circle(image)
    if isinstance(detected, str):
        return SegmentResult(status="REVIEW", reason=detected)
    cx, cy, r, method, support = detected
    mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(mask, (round(cx), round(cy)), round(r), 255, -1, cv2.LINE_8)
    return _finish(image, mask, method, support, r)


def overlay(image: np.ndarray, result: SegmentResult) -> np.ndarray:
    vis = image.copy()
    if result.mask is not None:
        contours, _ = cv2.findContours(result.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (40, 255, 40), max(2, round(min(image.shape[:2])/450)))
    cv2.putText(vis, result.method if result.status == "OK" else result.reason,
                (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (50,230,50) if result.status == "OK" else (20,20,255), 2, cv2.LINE_AA)
    return vis
