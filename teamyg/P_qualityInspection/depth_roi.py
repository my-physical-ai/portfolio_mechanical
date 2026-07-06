"""Depth ROI measurement for object existence and height validation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import ROI_MARGIN_RATIO, HEIGHT_THRESHOLD_MM, HEIGHT_PERCENTILE, MIN_OBJECT_PIXELS


@dataclass
class DepthRoiResult:
    ready: bool
    roi: Optional[Tuple[int, int, int, int]] = None
    height_mm: Optional[float] = None
    mean_diff_mm: Optional[float] = None
    min_diff_mm: Optional[float] = None
    max_diff_mm: Optional[float] = None
    valid_pct: Optional[float] = None
    object_pixels: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def expand_bbox(bbox, image_w: int, image_h: int, margin_ratio: float = ROI_MARGIN_RATIO):
    x, y, w, h = bbox
    margin = int(max(w, h) * margin_ratio)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image_w, x + w + margin)
    y2 = min(image_h, y + h + margin)
    return int(x1), int(y1), int(x2), int(y2)


def measure_depth_roi(background_depth, current_depth, depth_scale: float, bbox) -> DepthRoiResult:
    if bbox is None:
        return DepthRoiResult(False, reason="no ellipse bbox")
    if background_depth is None:
        return DepthRoiResult(False, reason="background is not captured")
    if current_depth is None:
        return DepthRoiResult(False, reason="empty depth frame")

    h, w = current_depth.shape[:2]
    if background_depth.shape[:2] != (h, w):
        return DepthRoiResult(
            False,
            reason="background size mismatch, recapture background",
        )
    x1, y1, x2, y2 = expand_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return DepthRoiResult(False, roi=(x1, y1, x2, y2), reason="invalid ROI")

    bg_roi = background_depth[y1:y2, x1:x2].astype(np.float64)
    now_roi = current_depth[y1:y2, x1:x2].astype(np.float64)

    valid = (bg_roi > 0) & (now_roi > 0)
    valid_pct = float(np.sum(valid) / max(valid.size, 1) * 100.0)
    if np.sum(valid) < MIN_OBJECT_PIXELS:
        return DepthRoiResult(False, roi=(x1, y1, x2, y2), valid_pct=valid_pct, reason="not enough valid depth")

    diff_mm = (bg_roi - now_roi) * float(depth_scale) * 1000.0
    diff_valid = diff_mm[valid]

    obj = diff_valid > HEIGHT_THRESHOLD_MM
    obj_pixels = int(np.sum(obj))
    if obj_pixels < MIN_OBJECT_PIXELS:
        return DepthRoiResult(
            False,
            roi=(x1, y1, x2, y2),
            valid_pct=valid_pct,
            object_pixels=obj_pixels,
            reason="object not separated from background",
        )

    obj_diff = diff_valid[obj]
    return DepthRoiResult(
        True,
        roi=(x1, y1, x2, y2),
        height_mm=float(np.percentile(obj_diff, HEIGHT_PERCENTILE)),
        mean_diff_mm=float(np.mean(obj_diff)),
        min_diff_mm=float(np.min(obj_diff)),
        max_diff_mm=float(np.max(obj_diff)),
        valid_pct=valid_pct,
        object_pixels=obj_pixels,
        reason="ok",
    )
