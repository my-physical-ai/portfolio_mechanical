# bottle_segmentor.py
# YOLOv8-seg로 생수병 영역 마스크를 추출 (COCO bottle 클래스 또는 커스텀 weights)

import cv2
import numpy as np
from ultralytics import YOLO

COCO_BOTTLE_CLASS_ID = 39


class BottleSegmentor:
    """컬러 이미지에서 생수병 세그멘테이션 마스크를 반환한다."""

    def __init__(self, model_path="yolov8n-seg.pt", device="cpu",
                 conf=0.25, mask_erode_px=10):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.mask_erode_px = mask_erode_px
        print(f"✅ YOLOv8-seg 로드: {model_path} | device={device}")

    def segment(self, bgr_image):
        """
        생수병 마스크(bool, H×W)를 반환한다.
        검출 실패 시 None.
        """
        if bgr_image is None:
            return None

        h, w = bgr_image.shape[:2]
        results = self.model.predict(
            bgr_image, conf=self.conf, verbose=False, device=self.device,
        )
        if not results or results[0].masks is None:
            return None

        r = results[0]
        best_mask = None
        best_score = 0.0

        for i, cls_id in enumerate(r.boxes.cls.cpu().numpy().astype(int)):
            if cls_id != COCO_BOTTLE_CLASS_ID:
                continue
            conf = float(r.boxes.conf[i])
            mask_raw = r.masks.data[i].cpu().numpy()
            if mask_raw.shape != (h, w):
                mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            area = float(mask_raw.sum())
            score = conf * area
            if score > best_score:
                best_score = score
                best_mask = mask_raw > 0.5

        if best_mask is None:
            return None

        return self._refine_mask(best_mask.astype(np.uint8))

    def _refine_mask(self, mask_u8):
        """경계 노이즈 제거 — 침식으로 병 표면 내부만 남긴다."""
        if self.mask_erode_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.mask_erode_px * 2 + 1, self.mask_erode_px * 2 + 1),
            )
            mask_u8 = cv2.erode(mask_u8, k, iterations=1)
        return mask_u8.astype(bool)

    @staticmethod
    def combine_masks(mask_a, mask_b):
        """양품·현재 마스크 교집합 (둘 다 None이면 None)."""
        if mask_a is None or mask_b is None:
            return None
        if mask_a.shape != mask_b.shape:
            return None
        return mask_a & mask_b

    @staticmethod
    def draw_overlay(bgr_image, mask, color=(0, 255, 0), alpha=0.35):
        """ROI 윤곽 + 반투명 채우기 (결과 시각화용)."""
        if mask is None or not mask.any():
            return bgr_image.copy()
        out = bgr_image.copy()
        overlay = out.copy()
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1 - alpha)
            + np.array(color, dtype=np.float32) * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        return overlay
