"""3D scene coordinate helpers for wheeled robot visualization."""

from __future__ import annotations


def world_to_scene(x: float, y: float, world_width: float, world_height: float):
    """
    Map 2D training coordinates (origin bottom-left) to 3D scene coordinates
    centered on the field with Y-up in Ursina-style layouts.
    """
    return (
        x - world_width / 2.0,
        0.0,
        y - world_height / 2.0,
    )
