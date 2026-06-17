"""Corn field layout using external 3D plant models."""

from __future__ import annotations

import random

from core.visuals.asset_paths import model_exists, model_relative


def plantable_field_bounds(
    world_width: float,
    world_height: float,
    corn_cfg: dict,
    field_cfg: dict | None = None,
) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, z_min, z_max) excluding the untilled margin strip."""
    half_w = world_width / 2.0
    half_h = world_height / 2.0
    margin = float((field_cfg or {}).get("margin_size", 0.0))
    if margin > 0:
        return (
            -half_w + margin,
            half_w - margin,
            -half_h + margin,
            half_h - margin,
        )

    inset = float(corn_cfg.get("row_inset", 0.04))
    return (
        -half_w * (1.0 - inset),
        half_w * (1.0 - inset),
        -half_h * (1.0 - inset),
        half_h * (1.0 - inset),
    )


def _corn_scale(corn_cfg: dict) -> float:
    target_h = float(corn_cfg["stalk_height"])
    native_h = float(corn_cfg.get("model_height", 1.15))
    return target_h / max(native_h, 1e-3)


def _resolve_corn_model(corn_cfg: dict) -> str | None:
    candidates = [
        corn_cfg.get("model", "models/corn/corn.obj"),
        corn_cfg.get("fallback_model", "models/corn/Corn_4.obj"),
    ]
    for rel in candidates:
        parts = rel.replace("\\", "/").split("/")
        if model_exists(*parts):
            return model_relative(*parts)
    return None


def _try_load_corn_model(model_path: str) -> bool:
    """Return True if Ursina can load the OBJ (some Quaternius meshes need triangulation)."""
    try:
        from ursina import Entity, destroy

        probe = Entity(model=model_path, enabled=False, visible=False)
        ok = probe.model is not None
        destroy(probe)
        return ok
    except Exception:
        return False


def create_corn_field_entity(
    Entity,
    world_width: float,
    world_height: float,
    corn_cfg: dict,
    parent=None,
    field_cfg: dict | None = None,
):
    """
    Place corn plants in rows using a cached external OBJ/GLB model.

    Falls back to procedural boxes if bundled assets are missing.
    """
    model_path = _resolve_corn_model(corn_cfg)
    if model_path is not None and not _try_load_corn_model(model_path):
        alt = corn_cfg.get("fallback_model", "models/corn/corn.obj")
        alt_parts = alt.replace("\\", "/").split("/")
        if model_exists(*alt_parts):
            alt_path = model_relative(*alt_parts)
            if _try_load_corn_model(alt_path):
                model_path = alt_path

    if model_path is None:
        from ursina import Mesh

        return _create_procedural_corn_field(
            Entity, Mesh, world_width, world_height, corn_cfg, parent=parent, field_cfg=field_cfg
        )

    row_spacing = float(corn_cfg["row_spacing"])
    plant_spacing = float(corn_cfg["plant_spacing"])
    scale = _corn_scale(corn_cfg)
    yaw_jitter = float(corn_cfg.get("yaw_jitter_deg", 18.0))
    stalk = tuple(corn_cfg.get("stalk_color", (0.22, 0.55, 0.18, 1.0)))
    leaf = tuple(corn_cfg.get("leaf_color", (0.28, 0.62, 0.15, 1.0)))
    plant_color = (
        (stalk[0] + leaf[0]) / 2,
        (stalk[1] + leaf[1]) / 2,
        (stalk[2] + leaf[2]) / 2,
        1.0,
    )

    x_min, x_max, z_min, z_max = plantable_field_bounds(
        world_width, world_height, corn_cfg, field_cfg
    )

    field_root = Entity(parent=parent)
    x = x_min + row_spacing / 2
    while x < x_max:
        z = z_min
        while z <= z_max:
            yaw = random.uniform(-yaw_jitter, yaw_jitter)
            Entity(
                parent=field_root,
                model=model_path,
                position=(x, 0, z),
                scale=scale,
                rotation_y=yaw,
                color=plant_color,
                texture=None,
                collider=None,
            )
            z += plant_spacing
        x += row_spacing

    return field_root


def _create_procedural_corn_field(Entity, Mesh, world_width, world_height, corn_cfg, parent=None, field_cfg=None):
    """Legacy procedural fallback (boxes) when OBJ assets are unavailable."""
    from core.meshing import merge_meshes

    vertices, triangles = _build_procedural_mesh(world_width, world_height, corn_cfg, field_cfg)
    if not vertices:
        return None
    mesh = Mesh(vertices=vertices, triangles=triangles, mode="triangle")
    return Entity(
        parent=parent,
        model=mesh,
        color=tuple(corn_cfg["stalk_color"]),
        double_sided=True,
    )


def _build_procedural_mesh(world_width, world_height, corn_cfg, field_cfg=None):
    from core.meshing import merge_meshes

    def box_at(cx, cy, cz, sx, sy, sz):
        hx, hy, hz = sx / 2, sy / 2, sz / 2
        corners = [
            (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz + hz), (cx - hx, cy - hy, cz + hz),
            (cx - hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz - hz),
            (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
        ]
        faces = [
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ]
        tris = []
        for a, b, c in faces:
            tris.extend([a, b, c])
        return corners, tris

    h = float(corn_cfg["stalk_height"])
    r = float(corn_cfg["stalk_radius"])
    row_spacing = float(corn_cfg["row_spacing"])
    plant_spacing = float(corn_cfg["plant_spacing"])
    x_min, x_max, z_min, z_max = plantable_field_bounds(
        world_width, world_height, corn_cfg, field_cfg
    )
    parts = []
    x = x_min + row_spacing / 2
    while x < x_max:
        z = z_min
        while z <= z_max:
            parts.append(box_at(x, h / 2, z, r * 2, h, r * 2))
            z += plant_spacing
        x += row_spacing
    return merge_meshes(parts)
