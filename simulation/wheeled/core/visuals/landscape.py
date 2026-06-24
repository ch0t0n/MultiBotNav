"""Distant meadow backdrop, hills, sun, and perimeter fencing for Ursina scenes."""

from __future__ import annotations

import math
import random


def _rgb(r: float, g: float, b: float, a: float = 1.0):
    from ursina import Color

    return Color(r, g, b, a)


def _configure_render_pipeline(cfg: dict | None = None):
    """MSAA and depth-buffer settings (call before Ursina())."""
    from panda3d.core import loadPrcFileData

    cfg = cfg or {}
    if cfg.get("multisample", True):
        loadPrcFileData("", "framebuffer-multisample 1")
        samples = int(cfg.get("multisample_count", 4))
        loadPrcFileData("", f"multisamples {samples}")
    loadPrcFileData("", "gl-depth-bits 24")


def _disable_fog(scene):
    """Turn off Ursina/Panda3D scene fog completely."""
    scene.fog_density = 0
    scene.clearFog()
    try:
        from ursina import render

        render.clearFog()
    except Exception:
        pass


def setup_atmosphere(cfg: dict | None = None):
    """Sky, ambient light, and a directional sun (no fog)."""
    from ursina import AmbientLight, DirectionalLight, Entity, Sky, camera, scene

    cfg = cfg or {}
    sky_texture = cfg.get("sky_texture", "sky_default")
    Sky(texture=sky_texture)

    clip_near = float(cfg.get("clip_plane_near", 2.0))
    clip_far = float(cfg.get("clip_plane_far", 6000.0))
    camera.clip_plane_near = clip_near
    camera.clip_plane_far = clip_far

    _disable_fog(scene)

    ambient = tuple(cfg.get("ambient_color", (0.55, 0.58, 0.52, 1.0)))
    AmbientLight(color=_rgb(*ambient))

    sun_el = math.radians(float(cfg.get("sun_elevation_deg", 46.0)))
    sun_az = math.radians(float(cfg.get("sun_azimuth_deg", -32.0)))
    sun_color = tuple(cfg.get("sun_color", (1.0, 0.94, 0.78, 1.0)))
    sun = DirectionalLight(
        shadows=bool(cfg.get("sun_shadows", False)),
        color=_rgb(*sun_color),
    )
    sun.rotation_x = 90 - math.degrees(sun_el)
    sun.rotation_y = math.degrees(sun_az)

    if cfg.get("show_sun_disc", True):
        sun_dist = float(cfg.get("sun_disc_distance", 6500.0))
        disc_scale = float(cfg.get("sun_disc_scale", 180.0))
        sun_pos = (
            sun_dist * math.cos(sun_el) * math.sin(sun_az),
            sun_dist * math.sin(sun_el),
            sun_dist * math.cos(sun_el) * math.cos(sun_az),
        )
        from ursina.shaders import unlit_shader

        Entity(
            model="sphere",
            position=sun_pos,
            scale=disc_scale,
            color=_rgb(sun_color[0], sun_color[1], sun_color[2], 1.0),
            shader=unlit_shader,
            unlit=True,
            collider=None,
            render_queue=-2,
        )

    return sun


def build_meadow(Entity, cfg: dict | None = None, parent=None):
    """Vast ground plane with tiled grass texture extending to the horizon."""
    cfg = cfg or {}
    size = float(cfg.get("meadow_size", 14000.0))
    y = float(cfg.get("meadow_y", -0.4))
    tint = tuple(cfg.get("meadow_color", (0.28, 0.44, 0.20, 1.0)))
    texture = cfg.get("meadow_texture", "grass_tintable")
    tile_size = float(cfg.get("meadow_tile_world_size", 80.0))
    tile_repeat = max(size / tile_size, 1.0)

    ground = Entity(
        parent=parent,
        model="plane",
        scale=(size, 1, size),
        position=(0, y, 0),
        texture=texture,
        texture_scale=(tile_repeat, tile_repeat),
        color=_rgb(*tint),
        collider=None,
        render_queue=0,
    )

    if cfg.get("meadow_patch_variation", True):
        patch_alpha = float(cfg.get("meadow_patch_alpha", 0.16))
        patch_tint = tuple(
            cfg.get("meadow_patch_tint_color", (0.42, 0.58, 0.30, patch_alpha))
        )
        patch_tile = float(cfg.get("meadow_patch_tile_world_size", 200.0))
        patch_repeat = max(size / patch_tile, 1.0)
        Entity(
            parent=parent,
            model="plane",
            scale=(size, 1, size),
            position=(0, y + 0.003, 0),
            texture=cfg.get("meadow_patch_texture", "noise"),
            texture_scale=(patch_repeat, patch_repeat),
            color=_rgb(*patch_tint),
            collider=None,
            render_queue=0,
        )

    return ground


def build_distant_hills(Entity, cfg: dict | None = None, parent=None):
    """Low hills on the horizon (disabled when ``hill_count`` is 0)."""
    cfg = cfg or {}
    count = int(cfg.get("hill_count", 0))
    if count <= 0:
        return []
    base_dist = float(cfg.get("hill_distance", 2600.0))
    dist_spread = float(cfg.get("hill_distance_spread", 700.0))
    h_min = float(cfg.get("hill_height_min", 70.0))
    h_max = float(cfg.get("hill_height_max", 170.0))
    w_min = float(cfg.get("hill_width_min", 420.0))
    w_max = float(cfg.get("hill_width_max", 920.0))
    color = tuple(cfg.get("hill_color", (0.26, 0.40, 0.22, 1.0)))
    far_color = tuple(cfg.get("hill_color_far", (0.34, 0.48, 0.30, 1.0)))
    seed = int(cfg.get("hill_seed", 17))

    rng = random.Random(seed)
    hills = []
    for i in range(count):
        angle = (2.0 * math.pi * i / count) + rng.uniform(-0.22, 0.22)
        dist = base_dist + rng.uniform(-dist_spread, dist_spread)
        height = rng.uniform(h_min, h_max)
        width = rng.uniform(w_min, w_max)
        depth = width * rng.uniform(0.75, 1.15)
        x = dist * math.sin(angle)
        z = dist * math.cos(angle)
        t = min(1.0, dist / (base_dist + dist_spread))
        c = (
            color[0] + (far_color[0] - color[0]) * t,
            color[1] + (far_color[1] - color[1]) * t,
            color[2] + (far_color[2] - color[2]) * t,
            1.0,
        )
        hills.append(
            Entity(
                parent=parent,
                model="sphere",
                position=(x, height * 0.42, z),
                scale=(width, height, depth),
                color=_rgb(*c),
                collider=None,
                render_queue=0,
            )
        )
    return hills


def build_perimeter_fence(
    Entity,
    world_width: float,
    world_height: float,
    fence_cfg: dict,
    parent=None,
):
    """Wooden post-and-rail fence around the playable field boundary."""
    post_spacing = float(fence_cfg.get("post_spacing", 11.0))
    post_w = float(fence_cfg.get("post_width", 0.85))
    height = float(fence_cfg["height"])
    base_y = float(fence_cfg.get("base_y", 0.05))
    post_color = tuple(fence_cfg.get("post_color", (0.48, 0.32, 0.16, 1.0)))
    rail_color = tuple(fence_cfg.get("rail_color", (0.58, 0.40, 0.20, 1.0)))
    rail_h = float(fence_cfg.get("rail_height", 0.32))
    rail_t = float(fence_cfg.get("rail_thickness", 0.22))
    rail_levels = fence_cfg.get("rail_levels", (0.38, 0.72))

    half_w = world_width / 2.0
    half_h = world_height / 2.0
    inset = float(fence_cfg.get("perimeter_inset", 0.0))
    corners = [
        (-half_w + inset, -half_h + inset),
        (half_w - inset, -half_h + inset),
        (half_w - inset, half_h - inset),
        (-half_w + inset, half_h - inset),
    ]

    entities = []
    post_y = base_y + height / 2.0

    def _add_post(x: float, z: float):
        entities.append(
            Entity(
                parent=parent,
                model="cube",
                position=(x, post_y, z),
                scale=(post_w, height, post_w),
                color=_rgb(*post_color),
                collider=None,
            )
        )

    def _add_rail(x0: float, z0: float, x1: float, z1: float, level: float):
        mx, mz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
        dx, dz = x1 - x0, z1 - z0
        length = max(0.5, math.hypot(dx, dz))
        angle = math.degrees(math.atan2(dz, dx))
        rail_y = base_y + height * level
        entities.append(
            Entity(
                parent=parent,
                model="cube",
                position=(mx, rail_y, mz),
                scale=(length + post_w * 0.35, rail_h, rail_t),
                rotation_y=angle,
                color=_rgb(*rail_color),
                collider=None,
            )
        )

    for i in range(4):
        x0, z0 = corners[i]
        x1, z1 = corners[(i + 1) % 4]
        edge_len = math.hypot(x1 - x0, z1 - z0)
        segments = max(1, int(math.ceil(edge_len / post_spacing)))
        posts = []
        for s in range(segments + 1):
            t = s / segments
            px = x0 + (x1 - x0) * t
            pz = z0 + (z1 - z0) * t
            _add_post(px, pz)
            posts.append((px, pz))
        for j in range(len(posts) - 1):
            px0, pz0 = posts[j]
            px1, pz1 = posts[j + 1]
            for level in rail_levels:
                _add_rail(px0, pz0, px1, pz1, float(level))

    return entities


def build_landscape(Entity, world_width: float, world_height: float, cfg: dict, parent=None):
    """Meadow, hills, perimeter fence, and rural scenery."""
    from core.visuals.rural_scenery import build_rural_scenery

    env_cfg = cfg.get("environment", {})
    meadow_parent = Entity(parent=parent)
    build_meadow(Entity, env_cfg, parent=meadow_parent)
    hills = build_distant_hills(Entity, env_cfg, parent=meadow_parent)
    fence = build_perimeter_fence(
        Entity,
        world_width,
        world_height,
        cfg.get("fence", {}),
        parent=parent,
    )
    scenery_root, scenery = build_rural_scenery(
        Entity,
        world_width,
        world_height,
        cfg,
        parent=parent,
    )
    return meadow_parent, hills, fence, scenery_root
