"""3D wheeled ground-robot visuals using bundled CC0 OBJ models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.geometry import world_to_scene
from core.visuals.asset_paths import model_exists, model_relative

if TYPE_CHECKING:
    from ursina import Entity, Vec3

# Distinct trail colors per robot (bodies use the model's own texture).
ROBOT_COLORS = [
    (0.15, 0.45, 0.85, 1),
    (0.85, 0.35, 0.15, 1),
    (0.20, 0.70, 0.30, 1),
    (0.75, 0.20, 0.55, 1),
    (0.55, 0.55, 0.15, 1),
]

DEFAULT_ROBOT_TYPES = {
    "tractor": {
        "label": "Tractor",
        "model": "models/robot/tractor.obj",
        "fallback_model": "models/robot/mars_rover.obj",
        "texture": "models/robot/Textures/colormap.png",
        "model_length": 2.0,
        "model_yaw_offset_deg": -90.0,
        "ground_offset": 0.0,
    },
    "delivery": {
        "label": "Delivery van",
        "model": "models/robot/delivery.obj",
        "fallback_model": "models/robot/tractor.obj",
        "texture": "models/robot/Textures/colormap.png",
        "model_length": 2.0,
        "model_yaw_offset_deg": -90.0,
        "ground_offset": 0.0,
    },
    "tractor_shovel": {
        "label": "Tractor with shovel",
        "model": "models/robot/tractor_shovel.obj",
        "fallback_model": "models/robot/tractor.obj",
        "texture": "models/robot/Textures/colormap.png",
        "model_length": 2.0,
        "model_yaw_offset_deg": -90.0,
        "ground_offset": 0.0,
    },
    "rover": {
        "label": "Research rover",
        "model": "models/robot/mars_rover.obj",
        "fallback_model": "models/robot/tractor.obj",
        "texture": None,
        "model_length": 1.8,
        "model_yaw_offset_deg": 0.0,
        "ground_offset": 0.0,
    },
}


def resolve_robot_cfg(robot_cfg: dict) -> dict:
    """Apply the selected ``type`` profile on top of shared robot settings."""
    merged = {k: v for k, v in robot_cfg.items() if k != "types"}
    robot_type = str(merged.get("type", "tractor"))
    types = dict(DEFAULT_ROBOT_TYPES)
    types.update(robot_cfg.get("types") or {})
    profile = types.get(robot_type)
    if profile:
        for key, value in profile.items():
            if key != "label":
                merged[key] = value
    merged["type"] = robot_type
    return merged


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
    _Vec3,
    length: float,
    _width: float,
    robot_cfg: dict,
    parent=None,
) -> dict:
    """Build a ground robot entity from a bundled OBJ model."""
    cfg = resolve_robot_cfg(robot_cfg)
    model_path = _resolve_robot_model(cfg)
    if model_path is None:
        raise FileNotFoundError(
            "No robot model found under assets/models/robot/. "
            "Run: python download_assets.py"
        )

    scale = _robot_uniform_scale(length, cfg)
    yaw_offset = float(cfg.get("model_yaw_offset_deg", -90.0))
    y_lift = float(cfg.get("ground_offset", 0.0))
    tex_path = None
    texture = cfg.get("texture")
    if texture:
        tex_parts = texture.replace("\\", "/").split("/")
        if model_exists(*tex_parts):
            tex_path = model_relative(*tex_parts)

    body = Entity(
        parent=parent,
        model=model_path,
        scale=scale,
        color=(1.0, 1.0, 1.0, 1.0),
        texture=tex_path,
        collider=None,
    )
    body._yaw_offset = yaw_offset
    body._y_lift = y_lift
    return {"body": body}


def sync_wheeled_robot(
    robot_ent: dict,
    x: float,
    y: float,
    theta: float,
    world_width: float,
    world_height: float,
    Vec3,
):
    """Update robot pose from bicycle-model state (x, y, heading theta)."""
    px, _, pz = world_to_scene(x, y, world_width, world_height)
    body = robot_ent["body"]
    y_off = getattr(body, "_y_lift", 0.0)
    yaw_off = getattr(body, "_yaw_offset", 0.0)
    body.position = Vec3(px, y_off, pz)
    body.rotation_y = -math.degrees(theta) + yaw_off + 180.0
