"""API tests for background capture flow (camera mocked)."""
import sys
import types
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

# pyrealsense2 is optional for unit tests.
_rs = types.ModuleType("pyrealsense2")
_rs.pipeline = MagicMock
_rs.config = MagicMock
_rs.align = MagicMock
_rs.stream = MagicMock()
_rs.format = MagicMock()
_rs.option = MagicMock()
_rs.spatial_filter = MagicMock
_rs.temporal_filter = MagicMock
_rs.hole_filling_filter = MagicMock
sys.modules["pyrealsense2"] = _rs

import app as app_module


DEPTH_SCALE = 0.001


def _fake_depth(h=480, w=640, depth_mm=750.0):
  return np.full((h, w), depth_mm / (DEPTH_SCALE * 1000.0), dtype=np.float64)


def test_capture_background_stores_meta():
  app_module.state["background_depth"] = None
  app_module.state["background_meta"] = None
  app_module.state["running"] = False

  fake_bg = _fake_depth()
  client = app_module.app.test_client()

  with patch.object(type(app_module.camera), "is_running", new_callable=PropertyMock, return_value=False), \
       patch.object(app_module.camera, "start"), \
       patch.object(app_module.camera, "depth_scale", DEPTH_SCALE), \
       patch.object(app_module.camera, "capture_depth_average", return_value=(fake_bg, 10)):
    res = client.post("/api/capture_background")

  assert res.status_code == 200
  data = res.get_json()
  assert data["status"] == "ok"
  assert data["frames_used"] == 10
  assert data["center_depth_mm"] == 750.0
  assert app_module.state["background_depth"] is not None
  assert app_module.state["background_meta"]["center_depth_mm"] == 750.0
  assert app_module.state["running"] is True


def test_capture_background_rejects_empty_depth():
  app_module.state["background_depth"] = None
  empty_bg = np.zeros((480, 640), dtype=np.float64)
  client = app_module.app.test_client()

  with patch.object(type(app_module.camera), "is_running", new_callable=PropertyMock, return_value=True), \
       patch.object(app_module.camera, "depth_scale", DEPTH_SCALE), \
       patch.object(app_module.camera, "capture_depth_average", return_value=(empty_bg, 10)):
    res = client.post("/api/capture_background")

  assert res.status_code == 500
  assert app_module.state["background_depth"] is None


def test_status_reports_background():
  app_module.state["background_meta"] = {
    "center_depth_mm": 720.0,
    "valid_pct": 99.5,
    "frames_used": 10,
  }
  app_module.state["background_depth"] = _fake_depth(depth_mm=720.0)

  client = app_module.app.test_client()
  with patch.object(type(app_module.camera), "is_running", new_callable=PropertyMock, return_value=True):
    data = client.get("/api/status").get_json()

  assert data["has_background"] is True
  assert data["background"]["center_depth_mm"] == 720.0


if __name__ == "__main__":
  test_capture_background_stores_meta()
  test_capture_background_rejects_empty_depth()
  test_status_reports_background()
  print("All background API tests passed.")
