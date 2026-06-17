"""3D wheeled ground-robot visuals using external models (bicycle kinematics)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.geometry import world_to_scene
from core.visuals.asset_paths import model_exists, model_relative

if TYPE_CHECKING:
    from ursina import Entity, Vec3


ROBOT_COLORS = [
    (0.15, 0.45, 0.85, 1),
    (0.85, 0.35, 0.15, 1),
    (0.20, 0.70, 0.30, 1),
    (0.75, 0.20, 0.55, 1),
    (0.55, 0.55, 0.15, 1),
]


def _resolve_robot_model(robot_cfg: dict) -> str | None:
    rel = robot_cfg.get("model", "models/robot/tractor.obj")
    parts = rel.replace("\\", "/").split("/")
    if model_exists(*parts):
        return model_relative(*parts)
    fallback = robot_cfg.get("fallback_model", "models/robot/mars_rover.obj")
    fb_parts = fallback.replace("\\", "/").split("/")
    if model_exists(*fb_parts):
        return model_relative(*fb_parts)
    return None


def _robot_uniform_scale(length: float, robot_cfg: dict) -> float:
    native = float(robot_cfg.get("model_length", 2.0))
    return length / max(native, 1e-3)


def create_wheeled_robot(
    Entity,
    Vec3,
    index: int,
    length: float,
    width: float,
    robot_cfg: dict,
    parent=None,
) -> dict:
    """Build a ground robot entity from a bundled OBJ/GLB model or procedural cubes."""
    model_path = _resolve_robot_model(robot_cfg)
    color = ROBOT_COLORS[index % len(ROBOT_COLORS)]

    if model_path is not None:
        scale = _robot_uniform_scale(length, robot_cfg)
        yaw_offset = float(robot_cfg.get("model_yaw_offset_deg", -90.0))
        y_lift = float(robot_cfg.get("ground_offset", 0.0))
        texture = robot_cfg.get("texture")
        tex_path = None
        if texture:
            tex_parts = texture.replace("\\", "/").split("/")
            if model_exists(*tex_parts):
                tex_path = model_relative(*tex_parts)

        body = Entity(
            parent=parent,
            model=model_path,
            scale=scale,
            color=color,
            texture=tex_path,
            collider=None,
        )
        body._yaw_offset = yaw_offset
        body._y_lift = y_lift
        return {
            "body": body,
            "cab": None,
            "wheels": [],
            "body_height": length * 0.25,
            "uses_mesh": True,
        }

    return _create_procedural_robot(Entity, Vec3, index, length, width, robot_cfg, parent=parent)


def _create_procedural_robot(Entity, Vec3, index, length, width, robot_cfg, parent=None):
    color = ROBOT_COLORS[index % len(ROBOT_COLORS)]
    body_h = float(robot_cfg["body_height"])
    cab_color = tuple(robot_cfg["cab_color"])
    wheel_color = tuple(robot_cfg["wheel_color"])
    wheel_scale = float(robot_cfg["wheel_scale"])

    body = Entity(
        parent=parent,
        model="cube",
        scale=(length, body_h, width),
        color=color,
        collider=None,
    )
    cab = Entity(
        parent=body,
        model="cube",
        scale=(0.35, 0.5, 0.8),
        position=(0.25, 0.35, 0),
        color=cab_color,
    )
    wheels = []
    for wx, wz in [(0.3, 0.45), (0.3, -0.45), (-0.3, 0.45), (-0.3, -0.45)]:
        wheels.append(
            Entity(
                parent=body,
                model="sphere",
                scale=(wheel_scale, wheel_scale, wheel_scale),
                position=(wx, -0.35, wz),
                color=wheel_color,
            )
        )
    return {
        "body": body,
        "cab": cab,
        "wheels": wheels,
        "body_height": body_h,
        "uses_mesh": False,
    }


def sync_wheeled_robot(
    robot_ent: dict,
    x: float,
    y: float,
    theta: float,
    delta: float,
    world_width: float,
    world_height: float,
    Vec3,
):
    """Update robot pose from bicycle-model state (x, y, theta, steer angle delta)."""
    px, _, pz = world_to_scene(x, y, world_width, world_height)
    body = robot_ent["body"]
    body_h = robot_ent["body_height"]

    if robot_ent.get("uses_mesh"):
        y_off = getattr(body, "_y_lift", 0.0)
        yaw_off = getattr(body, "_yaw_offset", 0.0)
        body.position = Vec3(px, y_off, pz)
        body.rotation_y = -math.degrees(theta) + yaw_off
        return

    body.position = Vec3(px, body_h / 2, pz)
    body.rotation_y = -math.degrees(theta)
    for wheel in robot_ent["wheels"]:
        wheel.rotation_y = -math.degrees(delta) if wheel.position.z > 0 else 0
