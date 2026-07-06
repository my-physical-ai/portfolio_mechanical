"""Small geometry utilities."""
from __future__ import annotations


def normalize_angle_pm90(angle_deg: float) -> float:
    """Normalize angle to [-90, 90] degrees.

    Ellipse yaw has 180-degree symmetry: 0 and 180 degrees represent the same
    physical long-axis direction.
    """
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return angle_deg


def compute_rotate_to_target(current_yaw_deg: float, target_yaw_deg: float) -> float:
    """Return the shortest yaw rotation from current to target."""
    return normalize_angle_pm90(target_yaw_deg - current_yaw_deg)
