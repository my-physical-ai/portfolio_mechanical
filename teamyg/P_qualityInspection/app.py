"""Ellipse Pillar Alignment Web Server.

Separate project from the previous mission-based RealSense Lab.
Purpose:
- Detect an elliptical cylinder from RGB only.
- Compute long/short axis and yaw angle.
- Compute required yaw rotation to target orientation.
- Use depth only for ROI-based object existence/height validation.
- Show results online through Flask MJPEG stream.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from config import (
    FLASK_HOST, FLASK_PORT, JPEG_QUALITY, STREAM_SLEEP_SEC,
    TARGET_YAW_DEG, MIN_OBJECT_PIXELS,
)
from realsense_camera import RealSenseCamera
from vision_ellipse import detect_ellipse
from depth_roi import measure_depth_roi
from overlay import draw_ellipse_overlay
from robot_interface import build_robot_command, send_alignment_command

app = Flask(__name__)
camera = RealSenseCamera()

state_lock = threading.Lock()
state: Dict[str, Any] = {
    "running": False,
    "target_yaw_deg": TARGET_YAW_DEG,
    "background_depth": None,
    "background_meta": None,
    "latest": {
        "ellipse": None,
        "depth_roi": None,
        "robot_command": None,
        "timestamp": None,
    },
}


def process_once():
    color, depth_frame, depth = camera.get_frames(apply_filter=True)
    if color is None:
        return None, {
            "ellipse": {"found": False, "reason": "camera frame unavailable"},
            "depth_roi": None,
            "robot_command": None,
            "timestamp": time.time(),
        }

    with state_lock:
        target_yaw = float(state["target_yaw_deg"])
        bg_depth = state["background_depth"]
        background_meta = state.get("background_meta")

    ellipse = detect_ellipse(color, target_yaw)
    depth_result = None
    if ellipse.found:
        depth_result = measure_depth_roi(bg_depth, depth, camera.depth_scale, ellipse.bbox)

    robot_command = build_robot_command(
        ellipse.center if ellipse.found else None,
        ellipse.yaw_deg if ellipse.found else None,
        ellipse.rotate_deg if ellipse.found else None,
    )

    overlay = draw_ellipse_overlay(
        color, ellipse, depth_result, robot_command, target_yaw, background_meta,
    )

    latest = {
        "ellipse": ellipse.to_dict(),
        "depth_roi": depth_result.to_dict() if depth_result else None,
        "robot_command": robot_command.to_dict(),
        "target_yaw_deg": target_yaw,
        "background": background_meta,
        "camera": {
            "running": camera.is_running,
            "depth_scale": camera.depth_scale,
        },
        "timestamp": time.time(),
    }

    with state_lock:
        state["latest"] = latest

    return overlay, latest


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    try:
        camera.start()
        with state_lock:
            state["running"] = True
        return jsonify({"status": "ok", "running": True, "depth_scale": camera.depth_scale})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        state["running"] = False
    camera.stop()
    return jsonify({"status": "ok", "running": False})


@app.route("/api/status")
def api_status():
    with state_lock:
        latest = state["latest"]
        background_meta = state.get("background_meta")
        has_background = state["background_depth"] is not None
        target_yaw = state["target_yaw_deg"]
    return jsonify({
        "server": "running",
        "camera_running": camera.is_running,
        "has_background": has_background,
        "background": background_meta,
        "target_yaw_deg": target_yaw,
        "latest": latest,
    })


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(force=True, silent=True) or {}
    if "target_yaw_deg" in data:
        with state_lock:
            state["target_yaw_deg"] = float(data["target_yaw_deg"])
    return api_status()


@app.route("/api/capture_background", methods=["POST"])
def api_capture_background():
    try:
        if not camera.is_running:
            camera.start()
            with state_lock:
                state["running"] = True

        bg, used = camera.capture_depth_average(apply_filter=True)
        if bg is None:
            return jsonify({
                "status": "error",
                "error": "not enough depth frames",
                "frames_used": used,
            }), 500

        valid = bg[bg > 0]
        if valid.size < MIN_OBJECT_PIXELS:
            return jsonify({
                "status": "error",
                "error": "not enough valid depth in background frame",
                "frames_used": used,
                "valid_pixels": int(valid.size),
            }), 500

        center_mm = float(bg[bg.shape[0] // 2, bg.shape[1] // 2] * camera.depth_scale * 1000.0)
        meta = {
            "captured_at": time.time(),
            "frames_used": used,
            "center_depth_mm": round(center_mm, 1),
            "shape": [int(bg.shape[0]), int(bg.shape[1])],
            "valid_pct": round(float(np.sum(bg > 0) / max(bg.size, 1) * 100.0), 1),
        }

        with state_lock:
            state["background_depth"] = bg
            state["background_meta"] = meta

        return jsonify({"status": "ok", **meta})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/latest")
def api_latest():
    with state_lock:
        return jsonify(state["latest"])


@app.route("/api/send_robot", methods=["POST"])
def api_send_robot():
    with state_lock:
        cmd_dict = state["latest"].get("robot_command")
    if not cmd_dict:
        return jsonify({"status": "error", "error": "no command preview"}), 400

    # Rebuild dataclass lightly through the existing dict shape is not necessary for preview.
    # Keep this endpoint safe: it does not move a robot until robot_interface.py is replaced.
    return jsonify({
        "status": "preview_only",
        "message": "Robot SDK is not connected. Edit robot_interface.py to enable motion.",
        "command": cmd_dict,
    })


def generate_stream():
    while True:
        with state_lock:
            should_run = bool(state["running"])

        if not should_run:
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.putText(frame, "Press START", (210, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else:
            frame, _ = process_once()
            if frame is None:
                frame = np.full((480, 640, 3), 255, dtype=np.uint8)
                cv2.putText(frame, "Waiting for RealSense...", (130, 235),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(STREAM_SLEEP_SEC)


@app.route("/video_feed")
def video_feed():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("=" * 60)
    print("Ellipse Pillar Alignment Server")
    print(f"http://localhost:{FLASK_PORT}")
    print("RGB: ellipse/yaw detection | Depth: ROI height validation")
    print("=" * 60)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
