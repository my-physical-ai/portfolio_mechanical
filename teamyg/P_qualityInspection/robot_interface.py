"""Robot command adapter.

This file intentionally contains a safe stub. Replace `send_alignment_command()`
with your robot arm SDK call after camera-to-robot calibration is finished.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

from config import ROBOT_ROTATION_OFFSET_DEG, ROBOT_ROTATION_SIGN


@dataclass
class RobotCommand:
    ready: bool
    center_px: Optional[Tuple[float, float]]
    yaw_deg: Optional[float]
    rotate_deg: Optional[float]
    robot_rotate_deg: Optional[float]
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_robot_command(center_px, yaw_deg, rotate_deg) -> RobotCommand:
    if center_px is None or yaw_deg is None or rotate_deg is None:
        return RobotCommand(False, None, None, None, None, "no valid ellipse")

    robot_rotate_deg = ROBOT_ROTATION_SIGN * rotate_deg + ROBOT_ROTATION_OFFSET_DEG
    return RobotCommand(
        ready=True,
        center_px=(float(center_px[0]), float(center_px[1])),
        yaw_deg=float(yaw_deg),
        rotate_deg=float(rotate_deg),
        robot_rotate_deg=float(robot_rotate_deg),
        reason="ok",
    )


def send_alignment_command(command: RobotCommand) -> Dict[str, Any]:
    """Safe placeholder.

    In production, replace this with a real robot command, for example:
    - convert pixel center to robot XY using calibrated homography
    - move above object
    - close gripper
    - rotate end-effector about Z by command.robot_rotate_deg
    - place object at aligned location
    """
    return {
        "sent": False,
        "message": "robot SDK is not connected; command preview only",
        "command": command.to_dict(),
    }
