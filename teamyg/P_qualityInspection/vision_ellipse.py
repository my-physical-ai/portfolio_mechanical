"""RGB-based ellipse detection for an elliptical cylinder top view."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    MIN_CONTOUR_AREA,
    MAX_CONTOUR_AREA_RATIO,
    MAX_MAJOR_AXIS_RATIO,
    MIN_AXIS_RATIO,
    MAX_AXIS_RATIO,
    ELLIPSE_FIT_SCORE_MIN,
    MIN_SHAPE_SCORE,
    CANNY_LOW_FACTOR,
    CANNY_HIGH_FACTOR,
    HSV_SATURATION_MIN,
    HSV_VALUE_MIN,
)
from math_utils import normalize_angle_pm90, compute_rotate_to_target


@dataclass
class EllipseResult:
    found: bool
    center: Optional[Tuple[float, float]] = None
    major_axis_px: Optional[float] = None
    minor_axis_px: Optional[float] = None
    yaw_deg: Optional[float] = None
    rotate_deg: Optional[float] = None
    area_px: Optional[float] = None
    contour_score: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)


def _make_candidate_masks(gray: np.ndarray, color_bgr: np.ndarray) -> List[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)

    median = float(np.median(blur))
    canny_low = int(max(0, CANNY_LOW_FACTOR * median))
    canny_high = int(min(255, CANNY_HIGH_FACTOR * median))
    if canny_high <= canny_low:
        canny_high = min(255, canny_low + 30)
    edges = cv2.Canny(blur, canny_low, canny_high)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    masks = [_clean_mask(mask, kernel) for mask in (otsu, otsu_inv, edges_closed)]

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    sat_min = HSV_SATURATION_MIN
    val_min = HSV_VALUE_MIN
    color_ranges = [
        ((90, sat_min, val_min), (130, 255, 255)),
        ((0, sat_min, val_min), (15, 255, 255)),
        ((165, sat_min, val_min), (180, 255, 255)),
        ((35, sat_min, val_min), (85, 255, 255)),
    ]
    for lower, upper in color_ranges:
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masks.append(_clean_mask(color_mask, kernel))

    sat_mask = cv2.inRange(hsv, (0, sat_min, val_min), (180, 255, 255))
    masks.append(_clean_mask(sat_mask, kernel))
    return masks


def _ellipse_contour(ellipse: Tuple) -> np.ndarray:
    (cx, cy), (axis_a, axis_b), angle = ellipse
    pts = cv2.ellipse2Poly(
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(axis_a / 2))), max(1, int(round(axis_b / 2)))),
        int(round(angle)),
        0,
        360,
        1,
    )
    return pts.reshape(-1, 1, 2).astype(np.int32)


def _score_contour(
    cnt: np.ndarray,
    ellipse: Tuple,
    frame_area: float,
    frame_min_dim: float,
) -> Optional[Tuple[float, float, float]]:
    area = float(cv2.contourArea(cnt))
    if area < MIN_CONTOUR_AREA:
        return None

    area_ratio = area / max(frame_area, 1.0)
    if area_ratio > MAX_CONTOUR_AREA_RATIO:
        return None

    (cx, cy), (axis_a, axis_b), angle = ellipse
    major = float(max(axis_a, axis_b))
    minor = float(min(axis_a, axis_b))
    if major / max(frame_min_dim, 1.0) > MAX_MAJOR_AXIS_RATIO:
        return None

    axis_ratio = major / max(minor, 1e-6)
    if not (MIN_AXIS_RATIO <= axis_ratio <= MAX_AXIS_RATIO):
        return None

    ellipse_area = np.pi * (major / 2.0) * (minor / 2.0)
    fill_ratio = area / max(ellipse_area, 1.0)
    symmetry = min(fill_ratio, 1.0 / max(fill_ratio, 1e-6))
    if symmetry < ELLIPSE_FIT_SCORE_MIN:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / max(float(w * h), 1.0)

    try:
        ellipse_cnt = _ellipse_contour(ellipse)
        shape_dist = cv2.matchShapes(cnt, ellipse_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    except cv2.error:
        return None
    shape_score = 1.0 / (1.0 + shape_dist * 12.0)
    if shape_score < MIN_SHAPE_SCORE:
        return None

    if 0.02 <= area_ratio <= 0.28:
        size_score = 1.0
    elif area_ratio < 0.02:
        size_score = area_ratio / 0.02
    else:
        size_score = max(0.0, 1.0 - (area_ratio - 0.28) / 0.12)

    score = (
        symmetry * 55.0
        + shape_score * 35.0
        + extent * 10.0
        + size_score * 20.0
    )
    return score, symmetry, shape_score


def detect_ellipse(color_bgr: np.ndarray, target_yaw_deg: float) -> EllipseResult:
    if color_bgr is None:
        return EllipseResult(False, reason="empty color frame")

    h, w = color_bgr.shape[:2]
    frame_area = float(h * w)
    frame_min_dim = float(min(h, w))

    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    masks = _make_candidate_masks(gray, color_bgr)

    best = None
    best_score = -1.0

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 5:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            scored = _score_contour(cnt, ellipse, frame_area, frame_min_dim)
            if scored is None:
                continue

            score, _, _ = scored
            if score <= best_score:
                continue

            (cx, cy), (axis_a, axis_b), angle = ellipse
            major = float(max(axis_a, axis_b))
            minor = float(min(axis_a, axis_b))
            if axis_a >= axis_b:
                yaw = float(angle)
            else:
                yaw = float(angle + 90.0)
            yaw = normalize_angle_pm90(yaw)
            rotate = compute_rotate_to_target(yaw, target_yaw_deg)

            x, y, bw, bh = cv2.boundingRect(cnt)
            best_score = score
            best = EllipseResult(
                found=True,
                center=(float(cx), float(cy)),
                major_axis_px=major,
                minor_axis_px=minor,
                yaw_deg=float(yaw),
                rotate_deg=float(rotate),
                area_px=float(cv2.contourArea(cnt)),
                contour_score=float(score),
                bbox=(int(x), int(y), int(bw), int(bh)),
                reason="ok",
            )

    if best is None:
        return EllipseResult(False, reason="no ellipse-like contour")
    return best
