"""Grass pasture inside the playable field boundary."""

from __future__ import annotations

import math
import random

from core.meshing import make_lit_mesh


def _rgb(r: float, g: float, b: float, a: float = 1.0):
    from ursina import Color

    return Color(r, g, b, a)


def field_interior_bounds(
    world_width: float,
    world_height: float,
    field_cfg: dict | None = None,
    inset_fraction: float = 0.02,
) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, z_min, z_max) for the mown interior."""
    half_w = world_width / 2.0
    half_h = world_height / 2.0
    field_cfg = field_cfg or {}
    margin = float(field_cfg.get("margin_size", 0.0))
    if margin > 0:
        return (
            -half_w + margin,
            half_w - margin,
            -half_h + margin,
            half_h - margin,
        )
    inset = float(field_cfg.get("interior_inset", inset_fraction))
    return (
        -half_w * (1.0 - inset),
        half_w * (1.0 - inset),
        -half_h * (1.0 - inset),
        half_h * (1.0 - inset),
    )


def _add_quad(
    vertices: list[tuple[float, float, float]],
    triangles: list[int],
    index: dict[tuple[float, float, float], int],
    corners: list[tuple[float, float, float]],
):
    def add_vertex(x: float, y: float, z: float) -> int:
        key = (round(x, 4), round(y, 4), round(z, 4))
        if key not in index:
            index[key] = len(vertices)
            vertices.append((x, y, z))
        return index[key]

    i0 = add_vertex(*corners[0])
    i1 = add_vertex(*corners[1])
    i2 = add_vertex(*corners[2])
    i3 = add_vertex(*corners[3])
    triangles.extend([i0, i1, i2, i0, i2, i3])


def _build_tuft_mesh_parts(
    cx: float,
    cz: float,
    base_y: float,
    height: float,
    width: float,
    yaw_deg: float,
) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Two crossed vertical quads forming a small grass clump."""
    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    index: dict[tuple[float, float, float], int] = {}
    half_w = width / 2.0
    top_y = base_y + height

    for angle in (yaw_deg, yaw_deg + 90.0):
        rad = math.radians(angle)
        dx = math.sin(rad) * half_w
        dz = math.cos(rad) * half_w
        _add_quad(
            vertices,
            triangles,
            index,
            [
                (cx - dx, base_y, cz - dz),
                (cx + dx, base_y, cz + dz),
                (cx + dx, top_y, cz + dz),
                (cx - dx, top_y, cz - dz),
            ],
        )
    return vertices, triangles


def _merge_mesh_parts(
    parts: list[tuple[list[tuple[float, float, float]], list[int]]],
) -> tuple[list[tuple[float, float, float]], list[int]]:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[int] = []
    offset = 0
    for verts, tris in parts:
        vertices.extend(verts)
        triangles.extend(i + offset for i in tris)
        offset += len(verts)
    return vertices, triangles


def build_grass_tufts(
    Entity,
    Mesh,
    world_width: float,
    world_height: float,
    grass_cfg: dict,
    field_cfg: dict | None = None,
    parent=None,
):
    """Scatter combined crossed-quad grass clumps across the field interior."""
    if not grass_cfg.get("tufts_enabled", False):
        return None

    field_cfg = field_cfg or {}
    field_y = float(field_cfg.get("surface_y", 0.1))
    base_y = field_y + float(grass_cfg.get("tuft_base_offset", 0.02))
    spacing = float(grass_cfg.get("tuft_spacing", 18.0))
    jitter = float(grass_cfg.get("tuft_jitter", 0.42))
    height_min = float(grass_cfg.get("tuft_height_min", 0.55))
    height_max = float(grass_cfg.get("tuft_height_max", 1.15))
    width_min = float(grass_cfg.get("tuft_width_min", 0.45))
    width_max = float(grass_cfg.get("tuft_width_max", 0.95))
    seed = int(grass_cfg.get("tuft_seed", 31))
    color = tuple(grass_cfg.get("tuft_color", (0.30, 0.58, 0.22, 0.92)))

    x_min, x_max, z_min, z_max = field_interior_bounds(world_width, world_height, field_cfg)
    rng = random.Random(seed)
    parts: list[tuple[list[tuple[float, float, float]], list[int]]] = []

    x = x_min + spacing / 2.0
    while x < x_max:
        z = z_min + spacing / 2.0
        while z < z_max:
            if rng.random() < float(grass_cfg.get("tuft_density", 0.82)):
                jx = rng.uniform(-jitter, jitter) * spacing
                jz = rng.uniform(-jitter, jitter) * spacing
                height = rng.uniform(height_min, height_max)
                width = rng.uniform(width_min, width_max)
                yaw = rng.uniform(0.0, 180.0)
                parts.append(
                    _build_tuft_mesh_parts(x + jx, z + jz, base_y, height, width, yaw)
                )
            z += spacing
        x += spacing

    if not parts:
        return None

    vertices, triangles = _merge_mesh_parts(parts)
    mesh = make_lit_mesh(Mesh, vertices, triangles, smooth=False)
    if mesh is None:
        return None

    return Entity(
        parent=parent,
        model=mesh,
        color=_rgb(*color),
        double_sided=True,
        collider=None,
        render_queue=2,
    )


def build_grass_ground(
    Entity,
    world_width: float,
    world_height: float,
    grass_cfg: dict,
    field_cfg: dict | None = None,
    parent=None,
):
    """Tiled tintable grass texture across the full playable field."""
    field_cfg = field_cfg or {}
    w, h = world_width, world_height
    field_y = float(field_cfg.get("surface_y", 0.1))
    tile_size = float(grass_cfg.get("tile_world_size", 14.0))
    texture = grass_cfg.get("texture", "grass_tintable")
    tint = tuple(grass_cfg.get("tint_color", (0.56, 0.70, 0.44, 1.0)))

    ground = Entity(
        parent=parent,
        model="plane",
        scale=(w, 1, h),
        position=(0, field_y, 0),
        texture=texture,
        texture_scale=(max(w / tile_size, 1.0), max(h / tile_size, 1.0)),
        color=_rgb(*tint),
        collider=None,
        render_queue=1,
    )

    if grass_cfg.get("patch_variation", False):
        patch_alpha = float(grass_cfg.get("patch_alpha", 0.14))
        patch_tint = tuple(grass_cfg.get("patch_tint_color", (0.55, 0.78, 0.42, patch_alpha)))
        patch_tile = float(grass_cfg.get("patch_tile_world_size", 28.0))
        Entity(
            parent=parent,
            model="plane",
            scale=(w, 1, h),
            position=(0, field_y + 0.005, 0),
            texture=grass_cfg.get("patch_texture", "noise"),
            texture_scale=(max(w / patch_tile, 1.0), max(h / patch_tile, 1.0)),
            color=_rgb(*patch_tint),
            collider=None,
            render_queue=1,
        )

    return ground


def build_grass_field(
    Entity,
    Mesh,
    world_width: float,
    world_height: float,
    grass_cfg: dict,
    field_cfg: dict | None = None,
    parent=None,
):
    """Grass pasture: tiled ground plus scattered clumps."""
    field_root = Entity(parent=parent)
    ground = build_grass_ground(
        Entity, world_width, world_height, grass_cfg, field_cfg, parent=field_root
    )
    tufts = build_grass_tufts(
        Entity,
        Mesh,
        world_width,
        world_height,
        grass_cfg,
        field_cfg,
        parent=field_root,
    )
    return field_root, ground, tufts
