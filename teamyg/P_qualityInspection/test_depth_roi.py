"""Unit tests for depth ROI background subtraction."""
import numpy as np

from depth_roi import measure_depth_roi, expand_bbox


DEPTH_SCALE = 0.001


def _make_background(h=480, w=640, depth_mm=800.0):
  depth = np.full((h, w), depth_mm / (DEPTH_SCALE * 1000.0), dtype=np.float64)
  return depth


def test_no_background_returns_reason():
  current = np.ones((480, 640), dtype=np.uint16) * 800
  result = measure_depth_roi(None, current, DEPTH_SCALE, (200, 150, 120, 90))
  assert not result.ready
  assert result.reason == "background is not captured"


def test_shape_mismatch_returns_reason():
  bg = _make_background(480, 640)
  current = np.ones((240, 320), dtype=np.uint16)
  result = measure_depth_roi(bg, current, DEPTH_SCALE, (50, 50, 80, 60))
  assert not result.ready
  assert "mismatch" in result.reason


def test_object_closer_than_background_detected():
  h, w = 480, 640
  bg = _make_background(h, w, depth_mm=800.0)
  current = bg.copy().astype(np.float64)
  # Object 30mm closer in center ROI
  current[200:280, 260:380] -= 30.0 / (DEPTH_SCALE * 1000.0)

  result = measure_depth_roi(bg, current.astype(np.uint16), DEPTH_SCALE, (260, 200, 120, 80))
  assert result.ready, result.reason
  assert result.height_mm is not None
  assert 20.0 < result.height_mm < 40.0
  assert result.object_pixels >= 150


def test_flat_scene_not_detected_as_object():
  bg = _make_background()
  current = bg.copy().astype(np.uint16)
  result = measure_depth_roi(bg, current, DEPTH_SCALE, (200, 150, 120, 90))
  assert not result.ready
  assert result.reason == "object not separated from background"


def test_expand_bbox_respects_image_bounds():
  x1, y1, x2, y2 = expand_bbox((10, 10, 100, 80), 640, 480, margin_ratio=0.25)
  assert x1 >= 0 and y1 >= 0
  assert x2 <= 640 and y2 <= 480
  assert x2 > x1 and y2 > y1


if __name__ == "__main__":
  test_no_background_returns_reason()
  test_shape_mismatch_returns_reason()
  test_object_closer_than_background_detected()
  test_flat_scene_not_detected_as_object()
  test_expand_bbox_respects_image_bounds()
  print("All depth_roi tests passed.")
