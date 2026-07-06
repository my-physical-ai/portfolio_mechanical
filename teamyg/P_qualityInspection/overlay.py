"""Overlay drawing for web preview."""
from __future__ import annotations

import cv2
import numpy as np


def draw_ellipse_overlay(
    frame,
    ellipse_result,
    depth_result,
    robot_command,
    target_yaw_deg: float,
    background_meta=None,
):
    display = frame.copy()
    h, w = display.shape[:2]

    cv2.putText(display, "ELLIPSE PILLAR ALIGNER", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    if background_meta:
        bg_text = (
            f"BG OK {background_meta.get('center_depth_mm', '-')}mm "
            f"({background_meta.get('valid_pct', '-')}%)"
        )
        bg_color = (0, 220, 0)
    else:
        bg_text = "BG: not captured"
        bg_color = (0, 165, 255)
    cv2.putText(display, bg_text, (w - 310, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, bg_color, 2)

    if ellipse_result and ellipse_result.found:
        cx, cy = ellipse_result.center
        major = ellipse_result.major_axis_px
        minor = ellipse_result.minor_axis_px
        yaw = ellipse_result.yaw_deg

        # cv2.ellipse angle expects the ellipse local x-axis. We draw using yaw as visual major-axis direction.
        axes = (int(major / 2), int(minor / 2))
        cv2.ellipse(display, (int(cx), int(cy)), axes, float(yaw), 0, 360, (0, 255, 0), 2)
        cv2.circle(display, (int(cx), int(cy)), 5, (0, 255, 255), -1)

        # Draw major axis line.
        rad = np.deg2rad(yaw)
        dx = np.cos(rad) * major / 2
        dy = np.sin(rad) * major / 2
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        cv2.line(display, p1, p2, (255, 255, 0), 2)

        # Draw target vertical reference line through center.
        ref_len = int(max(major, 70))
        cv2.line(display, (int(cx), int(cy - ref_len / 2)), (int(cx), int(cy + ref_len / 2)), (255, 0, 255), 1)

        label_lines = [
            f"center=({cx:.0f},{cy:.0f})",
            f"major={major:.1f}px minor={minor:.1f}px",
            f"yaw={yaw:+.1f} deg target={target_yaw_deg:.1f} deg",
            f"rotate={ellipse_result.rotate_deg:+.1f} deg",
        ]
        if depth_result and depth_result.ready:
            label_lines.append(f"height={depth_result.height_mm:.1f} mm")
        elif depth_result:
            label_lines.append(f"depth: {depth_result.reason}")

        x0, y0 = 10, 55
        cv2.rectangle(display, (5, 36), (390, 168), (0, 0, 0), -1)
        for i, text in enumerate(label_lines):
            cv2.putText(display, text, (x0, y0 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1)
    else:
        reason = ellipse_result.reason if ellipse_result else "no result"
        cv2.putText(display, f"NO ELLIPSE: {reason}", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if depth_result and depth_result.roi:
        x1, y1, x2, y2 = depth_result.roi
        color = (0, 255, 0) if depth_result.ready else (0, 165, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    if robot_command and robot_command.ready:
        text = f"ROBOT PREVIEW: Z rotate {robot_command.robot_rotate_deg:+.1f} deg"
        cv2.rectangle(display, (5, h - 42), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.putText(display, text, (12, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

    return display
