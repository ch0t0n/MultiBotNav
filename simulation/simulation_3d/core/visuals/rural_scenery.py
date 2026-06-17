"""Procedural farm scenery: trees, buildings, livestock, paths, and ground detail."""

from __future__ import annotations

import math
import random


def _rgb(r: float, g: float, b: float, a: float = 1.0):
    from ursina import Color

    return Color(r, g, b, a)


def _lerp_color(a: tuple, b: tuple, t: float) -> tuple:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t if len(a) > 3 else 1.0,
    )


def _random_pasture_point(
    rng: random.Random,
    half_w: float,
    half_h: float,
    inner: float,
    outer: float,
) -> tuple[float, float]:
    """Pick a point in the pasture ring outside the fenced field."""
    for _ in range(64):
        side = rng.randint(0, 3)
        if side == 0:
            x = rng.uniform(-half_w - outer, half_w + outer)
            z = rng.uniform(half_h + inner, half_h + outer)
        elif side == 1:
            x = rng.uniform(-half_w - outer, half_w + outer)
            z = rng.uniform(-half_h - outer, -half_h - inner)
        elif side == 2:
            x = rng.uniform(-half_w - outer, -half_w - inner)
            z = rng.uniform(-half_h - outer, half_h + outer)
        else:
            x = rng.uniform(half_w + inner, half_w + outer)
            z = rng.uniform(-half_h - outer, half_h + outer)
        return x, z
    return half_w + inner + 10.0, 0.0


def _add_tree(
    Entity,
    parent,
    x: float,
    z: float,
    scale: float,
    rng: random.Random,
    scenery_cfg: dict,
):
    from core.visuals.tree_models import create_scenery_tree, scenery_variants

    variants = scenery_variants(scenery_cfg)
    if not variants:
        return []
    variant = variants[rng.randint(0, len(variants) - 1)]
    root, trunk, foliage = create_scenery_tree(Entity, parent, x, z, variant, scale, rng, scenery_cfg)
    return [root, trunk, foliage]


def _add_hay_bale(Entity, parent, x, z, scale, color):
    Entity(
        parent=parent,
        model="cube",
        position=(x, scale * 0.45, z),
        scale=(scale * 0.95, scale * 0.45, scale * 0.7),
        color=_rgb(*color),
        collider=None,
    )


def _add_barn(Entity, parent, x, z, rotation_y, cfg):
    body_c = tuple(cfg.get("barn_body_color", (0.58, 0.22, 0.14, 1.0)))
    roof_c = tuple(cfg.get("barn_roof_color", (0.32, 0.18, 0.10, 1.0)))
    trim_c = tuple(cfg.get("barn_trim_color", (0.72, 0.68, 0.58, 1.0)))
    w = float(cfg.get("barn_width", 42.0))
    d = float(cfg.get("barn_depth", 58.0))
    h = float(cfg.get("barn_height", 28.0))

    root = Entity(parent=parent, position=(x, 0, z), rotation_y=rotation_y, collider=None)
    Entity(
        parent=root,
        model="cube",
        position=(0, h * 0.5, 0),
        scale=(w, h, d),
        color=_rgb(*body_c),
        collider=None,
    )
    Entity(
        parent=root,
        model="cube",
        position=(0, h + 3.5, 0),
        scale=(w + 4, 7, d + 4),
        color=_rgb(*roof_c),
        collider=None,
    )
    Entity(
        parent=root,
        model="cube",
        position=(0, h * 0.35, d * 0.5 + 0.2),
        scale=(w * 0.35, h * 0.55, 1.5),
        color=_rgb(*trim_c),
        collider=None,
    )
    return root


def _add_farmhouse(Entity, parent, x, z, rotation_y, cfg):
    wall_c = tuple(cfg.get("house_wall_color", (0.86, 0.82, 0.74, 1.0)))
    roof_c = tuple(cfg.get("house_roof_color", (0.28, 0.24, 0.20, 1.0)))
    w = float(cfg.get("house_width", 34.0))
    d = float(cfg.get("house_depth", 30.0))
    h = float(cfg.get("house_height", 22.0))

    root = Entity(parent=parent, position=(x, 0, z), rotation_y=rotation_y, collider=None)
    Entity(
        parent=root,
        model="cube",
        position=(0, h * 0.5, 0),
        scale=(w, h, d),
        color=_rgb(*wall_c),
        collider=None,
    )
    Entity(
        parent=root,
        model="cube",
        position=(0, h + 2.5, 0),
        scale=(w + 3, 5, d + 3),
        color=_rgb(*roof_c),
        collider=None,
    )
    Entity(
        parent=root,
        model="cube",
        position=(w * 0.32, h * 0.55, d * 0.5 + 0.2),
        scale=(w * 0.22, h * 0.35, 1.2),
        color=_rgb(0.42, 0.30, 0.18, 1.0),
        collider=None,
    )
    return root


def _add_silo(Entity, parent, x, z, cfg):
    body_c = tuple(cfg.get("silo_color", (0.72, 0.72, 0.76, 1.0)))
    cap_c = tuple(cfg.get("silo_cap_color", (0.42, 0.40, 0.38, 1.0)))
    r = float(cfg.get("silo_radius", 7.5))
    h = float(cfg.get("silo_height", 48.0))
    Entity(
        parent=parent,
        model="cube",
        position=(x, h * 0.5, z),
        scale=(r * 2, h, r * 2),
        color=_rgb(*body_c),
        collider=None,
    )
    Entity(
        parent=parent,
        model="sphere",
        position=(x, h + r * 0.55, z),
        scale=r * 1.5,
        color=_rgb(*cap_c),
        collider=None,
    )


def _add_cow(Entity, parent, x, z, rotation_y, rng, cfg):
    body_c = tuple(cfg.get("cow_body_color", (0.22, 0.18, 0.14, 1.0)))
    spot_c = tuple(cfg.get("cow_spot_color", (0.88, 0.86, 0.82, 1.0)))
    scale = rng.uniform(3.8, 5.2)

    root = Entity(
        parent=parent,
        position=(x, 0, z),
        rotation_y=rotation_y,
        collider=None,
    )
    Entity(
        parent=root,
        model="cube",
        position=(0, scale * 0.55, 0),
        scale=(scale * 1.4, scale * 0.7, scale * 0.65),
        color=_rgb(*(spot_c if rng.random() < 0.45 else body_c)),
        collider=None,
    )
    Entity(
        parent=root,
        model="cube",
        position=(scale * 0.85, scale * 0.75, 0),
        scale=(scale * 0.45, scale * 0.4, scale * 0.38),
        color=_rgb(*body_c),
        collider=None,
    )
    for lx, lz in [(-0.35, 0.22), (-0.35, -0.22), (0.35, 0.22), (0.35, -0.22)]:
        Entity(
            parent=root,
            model="cube",
            position=(scale * lx, scale * 0.15, scale * lz),
            scale=(scale * 0.12, scale * 0.35, scale * 0.12),
            color=_rgb(*body_c),
            collider=None,
        )


def _add_sheep(Entity, parent, x, z, rotation_y, rng, cfg):
    wool_c = tuple(cfg.get("sheep_color", (0.90, 0.88, 0.84, 1.0)))
    scale = rng.uniform(2.4, 3.2)
    root = Entity(parent=parent, position=(x, 0, z), rotation_y=rotation_y, collider=None)
    Entity(
        parent=root,
        model="sphere",
        position=(0, scale * 0.55, 0),
        scale=(scale * 1.1, scale * 0.75, scale * 0.85),
        color=_rgb(*wool_c),
        collider=None,
    )
    Entity(
        parent=root,
        model="sphere",
        position=(scale * 0.55, scale * 0.65, 0),
        scale=scale * 0.42,
        color=_rgb(0.18, 0.16, 0.14, 1.0),
        collider=None,
    )


def build_dirt_paths(
    Entity,
    half_w: float,
    half_h: float,
    env_cfg: dict,
    scenery_cfg: dict,
    parent=None,
):
    """Farm lane from buildings toward the south field gate."""
    if not scenery_cfg.get("dirt_paths", True):
        return []

    from ursina.shaders import unlit_shader

    meadow_y = float(env_cfg.get("meadow_y", -0.4))
    path_c = tuple(scenery_cfg.get("path_color", (0.48, 0.38, 0.24, 1.0)))
    path_w = float(scenery_cfg.get("path_width", 14.0))
    path_thickness = float(scenery_cfg.get("path_thickness", 0.14))
    path_lift = float(scenery_cfg.get("path_lift", 0.06))
    path_y = meadow_y + path_lift + path_thickness * 0.5
    entities = []

    def _path_strip(x, z, width, length, rotation_y=0):
        ent = Entity(
            parent=parent,
            model="cube",
            position=(x, path_y, z),
            scale=(width, path_thickness, length),
            rotation_y=rotation_y,
            color=_rgb(*path_c),
            collider=None,
            shader=unlit_shader,
            unlit=True,
            render_queue=1,
        )
        entities.append(ent)
        return ent

    barn_x = -half_w - float(scenery_cfg.get("barn_offset_x", 58.0))
    barn_z = half_h + float(scenery_cfg.get("barn_offset_z", 42.0))
    gate_z = -half_h

    mx, mz = (barn_x + 0) / 2, (barn_z + gate_z) / 2
    length = math.hypot(barn_x, barn_z - gate_z)
    angle = math.degrees(math.atan2(gate_z - barn_z, -barn_x))
    _path_strip(mx * 0.55, mz, path_w, length * 0.95, angle)

    # Cross path along the south edge
    _path_strip(0, gate_z - 18, half_w * 2.4, path_w * 0.85)

    return entities


def build_trees(Entity, half_w, half_h, scenery_cfg, parent=None):
    count = int(scenery_cfg.get("tree_count", 72))
    inner = float(scenery_cfg.get("tree_inner_margin", 18.0))
    outer = float(scenery_cfg.get("tree_outer_margin", 320.0))
    seed = int(scenery_cfg.get("seed", 42))
    rng = random.Random(seed + 1)

    entities = []
    for _ in range(count):
        x, z = _random_pasture_point(rng, half_w, half_h, inner, outer)
        scale = rng.uniform(10.0, 22.0)
        placed = _add_tree(Entity, parent, x, z, scale, rng, scenery_cfg)
        if placed:
            entities.extend(placed)
    return entities


def build_hedgerow(
    Entity,
    half_w: float,
    half_h: float,
    scenery_cfg: dict,
    parent=None,
):
    if not scenery_cfg.get("hedgerow", True):
        return []

    offset = float(scenery_cfg.get("hedgerow_offset", 6.0))
    spacing = float(scenery_cfg.get("hedgerow_spacing", 9.0))
    bush_c = tuple(scenery_cfg.get("hedgerow_color", (0.16, 0.38, 0.12, 1.0)))
    bush_h = float(scenery_cfg.get("hedgerow_height", 5.5))
    seed = int(scenery_cfg.get("seed", 42))
    rng = random.Random(seed + 5)

    entities = []
    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    for i in range(4):
        x0, z0 = corners[i]
        x1, z1 = corners[(i + 1) % 4]
        edge_len = math.hypot(x1 - x0, z1 - z0)
        segments = max(1, int(edge_len / spacing))
        for s in range(segments + 1):
            t = s / segments
            px = x0 + (x1 - x0) * t
            pz = z0 + (z1 - z0) * t
            nx, nz = x1 - x0, z1 - z0
            ln = max(math.hypot(nx, nz), 1e-3)
            ox = -nz / ln * offset
            oz = nx / ln * offset
            scale = rng.uniform(3.5, 6.0)
            entities.append(
                Entity(
                    parent=parent,
                    model="sphere",
                    position=(px + ox, bush_h * 0.45, pz + oz),
                    scale=(scale * 1.2, bush_h, scale),
                    color=_rgb(*_lerp_color(bush_c, (0.22, 0.50, 0.16, 1.0), rng.random() * 0.4)),
                    collider=None,
                )
            )
    return entities


def build_farm_structures(Entity, half_w, half_h, scenery_cfg, parent=None):
    if not scenery_cfg.get("buildings", True):
        return []

    entities = []
    barn_x = -half_w - float(scenery_cfg.get("barn_offset_x", 58.0))
    barn_z = half_h + float(scenery_cfg.get("barn_offset_z", 42.0))
    entities.append(_add_barn(Entity, parent, barn_x, barn_z, 35, scenery_cfg))

    house_x = half_w + float(scenery_cfg.get("house_offset_x", 52.0))
    house_z = half_h + float(scenery_cfg.get("house_offset_z", 38.0))
    entities.append(_add_farmhouse(Entity, parent, house_x, house_z, -40, scenery_cfg))

    silo_x = barn_x + float(scenery_cfg.get("silo_offset_x", 28.0))
    silo_z = barn_z - float(scenery_cfg.get("silo_offset_z", 12.0))
    _add_silo(Entity, parent, silo_x, silo_z, scenery_cfg)

    hay_c = tuple(scenery_cfg.get("hay_color", (0.78, 0.64, 0.22, 1.0)))
    seed = int(scenery_cfg.get("seed", 42))
    rng = random.Random(seed + 7)
    for _ in range(int(scenery_cfg.get("hay_bale_count", 9))):
        bx = barn_x + rng.uniform(-20, 25)
        bz = barn_z + rng.uniform(-30, 10)
        _add_hay_bale(Entity, parent, bx, bz, rng.uniform(4.5, 6.5), hay_c)

    return entities


def build_livestock(Entity, half_w, half_h, scenery_cfg, parent=None):
    cow_n = int(scenery_cfg.get("cow_count", 10))
    sheep_n = int(scenery_cfg.get("sheep_count", 7))
    inner = float(scenery_cfg.get("livestock_inner_margin", 22.0))
    outer = float(scenery_cfg.get("livestock_outer_margin", 180.0))
    seed = int(scenery_cfg.get("seed", 42))
    rng = random.Random(seed + 11)

    entities = []
    for _ in range(cow_n):
        x, z = _random_pasture_point(rng, half_w, half_h, inner, outer)
        _add_cow(Entity, parent, x, z, rng.uniform(0, 360), rng, scenery_cfg)
    for _ in range(sheep_n):
        x, z = _random_pasture_point(rng, half_w, half_h, inner, outer)
        _add_sheep(Entity, parent, x, z, rng.uniform(0, 360), rng, scenery_cfg)
    return entities


def build_field_margin(Entity, world_width, world_height, field_cfg, parent=None):
    """Darker untilled strip along the inside of the fence."""
    from ursina.shaders import unlit_shader

    margin = float(field_cfg.get("margin_size", 0.0))
    if margin <= 0:
        return []

    half_w = world_width / 2.0
    half_h = world_height / 2.0
    field_y = float(field_cfg.get("surface_y", 0.1))
    strip_thickness = float(field_cfg.get("margin_thickness", 0.08))
    strip_y = field_y + strip_thickness * 0.5 + 0.04
    color = tuple(field_cfg.get("margin_color", (0.27, 0.40, 0.15, 1.0)))
    w, h = world_width, world_height
    m = margin
    entities = []
    strips = [
        ((0, strip_y, half_h - m / 2), (w, strip_thickness, m)),
        ((0, strip_y, -half_h + m / 2), (w, strip_thickness, m)),
        ((-half_w + m / 2, strip_y, 0), (m, strip_thickness, h - 2 * m)),
        ((half_w - m / 2, strip_y, 0), (m, strip_thickness, h - 2 * m)),
    ]
    for pos, scale in strips:
        entities.append(
            Entity(
                parent=parent,
                model="cube",
                position=pos,
                scale=scale,
                color=_rgb(*color),
                collider=None,
                shader=unlit_shader,
                unlit=True,
                render_queue=2,
            )
        )
    return entities


def build_crop_furrows(
    Entity,
    world_width,
    world_height,
    corn_cfg,
    field_cfg,
    parent=None,
):
    """Subtle darker strips between corn rows."""
    if not field_cfg.get("furrows", True):
        return []

    from ursina.shaders import unlit_shader

    from core.visuals.crop_models import plantable_field_bounds

    row_spacing = float(corn_cfg["row_spacing"])
    field_y = float(field_cfg.get("surface_y", 0.1))
    furrow_thickness = float(field_cfg.get("furrow_thickness", 0.06))
    furrow_y = field_y + furrow_thickness * 0.5 + 0.03
    furrow_c = tuple(field_cfg.get("furrow_color", (0.24, 0.36, 0.11, 1.0)))
    x_min, x_max, z_min, z_max = plantable_field_bounds(
        world_width, world_height, corn_cfg, field_cfg
    )
    row_len = max(z_max - z_min, row_spacing)
    width = row_spacing * 0.38

    entities = []
    x = x_min + row_spacing
    while x < x_max - row_spacing * 0.25:
        entities.append(
            Entity(
                parent=parent,
                model="cube",
                position=(x, furrow_y, 0),
                scale=(width, furrow_thickness, row_len),
                color=_rgb(*furrow_c),
                collider=None,
                shader=unlit_shader,
                unlit=True,
                render_queue=2,
            )
        )
        x += row_spacing
    return entities


def build_rural_scenery(
    Entity,
    world_width: float,
    world_height: float,
    cfg: dict,
    parent=None,
):
    """Trees, buildings, livestock, and paths outside the field."""
    scenery_cfg = dict(cfg.get("scenery", {}))
    if "trees" not in scenery_cfg and "trees" in cfg:
        scenery_cfg["trees"] = cfg["trees"]
    env_cfg = cfg.get("environment", {})
    half_w = world_width / 2.0
    half_h = world_height / 2.0

    root = Entity(parent=parent)
    entities = []
    entities.extend(build_dirt_paths(Entity, half_w, half_h, env_cfg, scenery_cfg, parent=root))
    entities.extend(build_trees(Entity, half_w, half_h, scenery_cfg, parent=root))
    entities.extend(build_hedgerow(Entity, half_w, half_h, scenery_cfg, parent=root))
    entities.extend(build_farm_structures(Entity, half_w, half_h, scenery_cfg, parent=root))
    entities.extend(build_livestock(Entity, half_w, half_h, scenery_cfg, parent=root))
    return root, entities
