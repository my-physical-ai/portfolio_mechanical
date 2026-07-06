"""RealSense camera wrapper for color/depth acquisition."""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, List

import numpy as np
import pyrealsense2 as rs

from config import CAMERA_W, CAMERA_H, CAMERA_FPS, BACKGROUND_CAPTURE_FRAMES, BACKGROUND_WARMUP_FRAMES


class RealSenseCamera:
    def __init__(self) -> None:
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale: float = 0.001
        self.intrinsics = None
        self.lock = threading.Lock()

        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

        self.hole_filter = rs.hole_filling_filter()

    @property
    def is_running(self) -> bool:
        return self.pipeline is not None

    def start(self) -> None:
        if self.pipeline is not None:
            return

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_W, CAMERA_H, rs.format.bgr8, CAMERA_FPS)
        config.enable_stream(rs.stream.depth, CAMERA_W, CAMERA_H, rs.format.z16, CAMERA_FPS)

        profile = pipeline.start(config)
        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = stream_profile.get_intrinsics()

        # Warm-up frames improve auto-exposure/depth stability.
        for _ in range(10):
            try:
                self.pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                pass

    def stop(self) -> None:
        if self.pipeline is None:
            return
        try:
            self.pipeline.stop()
        finally:
            self.pipeline = None
            self.align = None
            self.intrinsics = None

    def _filter_depth(self, depth_frame):
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.hole_filter.process(depth_frame)
        return depth_frame

    def get_frames(self, apply_filter: bool = True):
        """Return (color_bgr, depth_frame, depth_array_uint16).

        Depth is aligned to color coordinates.
        """
        if self.pipeline is None or self.align is None:
            return None, None, None

        try:
            with self.lock:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    return None, None, None

                if apply_filter:
                    depth_frame = self._filter_depth(depth_frame)

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                return color, depth_frame, depth
        except Exception:
            return None, None, None

    def capture_depth_average(
        self,
        frames: int = BACKGROUND_CAPTURE_FRAMES,
        warmup_frames: int = BACKGROUND_WARMUP_FRAMES,
        apply_filter: bool = True,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Capture a stable average depth reference."""
        for _ in range(max(0, warmup_frames)):
            self.get_frames(apply_filter=apply_filter)
            time.sleep(0.04)

        collected: List[np.ndarray] = []
        for _ in range(frames):
            _, _, depth = self.get_frames(apply_filter=apply_filter)
            if depth is not None:
                collected.append(depth.astype(np.float64))
            time.sleep(0.04)

        if len(collected) < max(3, frames // 2):
            return None, len(collected)
        return np.mean(collected, axis=0), len(collected)
