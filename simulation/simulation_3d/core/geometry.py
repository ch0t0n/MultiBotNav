"""Shared geometry helpers for wheeled robot simulation."""

from __future__ import annotations

import numpy as np
from shapely import Polygon


def get_robot_polygon(x, y, theta, robot_length, robot_width) -> Polygon:
    """Return a Shapely Polygon for the robot's rectangular footprint."""
    dx = robot_length / 2
    dy = robot_width / 2
    corners = np.array(
        [
            [dx, dy],
            [dx, -dy],
            [-dx, -dy],
            [-dx, dy],
        ]
    )
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = np.dot(corners, rot.T) + np.array([x, y])
    return Polygon(rotated)


def binary_list_to_decimal(bin_list) -> int:
    """Convert a list of 0/1 values to its decimal integer equivalent."""
    return int("".join(str(int(b)) for b in bin_list), 2)


def world_to_scene(x: float, y: float, world_width: float, world_height: float):
    """
    Map 2D training coordinates (origin bottom-left) to 3D scene coordinates
    centered on the field with Y-up in Ursina/Webots-style layouts.
    """
    return (
        x - world_width / 2.0,
        0.0,
        y - world_height / 2.0,
    )
