"""Colored path traces for wheeled robots in the Ursina scene."""

from __future__ import annotations

from core.geometry import world_to_scene
from core.visuals.robot_model import ROBOT_COLORS


def _trail_alpha(trails_cfg: dict) -> float:
    """
    Resolve line opacity from config.

    ``alpha`` — 0 (invisible) to 1 (opaque).
    ``transparency`` — 0 (opaque) to 1 (invisible); inverted into alpha when set.
    """
    if "alpha" in trails_cfg:
        return max(0.0, min(1.0, float(trails_cfg["alpha"])))
    if "transparency" in trails_cfg:
        return max(0.0, min(1.0, 1.0 - float(trails_cfg["transparency"])))
    return 0.55


def _trail_color(trails_cfg: dict, robot_index: int, alpha: float) -> tuple[float, float, float, float]:
    colors = trails_cfg.get("colors")
    if colors and robot_index < len(colors):
        c = colors[robot_index]
        return (float(c[0]), float(c[1]), float(c[2]), alpha)

    if trails_cfg.get("use_robot_colors", True):
        base = ROBOT_COLORS[robot_index % len(ROBOT_COLORS)]
        return (float(base[0]), float(base[1]), float(base[2]), alpha)

    default = tuple(trails_cfg.get("color", (0.75, 0.25, 0.2, 1.0)))
    return (float(default[0]), float(default[1]), float(default[2]), alpha)


def draw_robot_trails(
    Entity,
    Mesh,
    Vec3,
    destroy,
    trail_state: list[dict],
    robot_paths,
    world_width: float,
    world_height: float,
    trails_cfg: dict | None,
    parent=None,
) -> None:
    """Update path line entities from ``robot_paths`` (one strip per robot)."""
    cfg = trails_cfg or {}
    if not cfg.get("enabled", True):
        for slot in trail_state:
            if slot.get("entity") is not None:
                destroy(slot["entity"])
        trail_state.clear()
        return

    alpha = _trail_alpha(cfg)
    if alpha <= 0.0:
        for slot in trail_state:
            if slot.get("entity") is not None:
                destroy(slot["entity"])
        trail_state.clear()
        return

    height = float(cfg.get("height", 0.22))
    thickness = float(cfg.get("thickness", 2.5))
    use_unlit = bool(cfg.get("unlit", True))
    render_queue = int(cfg.get("render_queue", 3))

    shader = None
    if use_unlit:
        from ursina.shaders import unlit_shader

        shader = unlit_shader

    while len(trail_state) < len(robot_paths):
        trail_state.append({"entity": None, "path_len": 0})

    # Trim extra slots when robot count decreases.
    while len(trail_state) > len(robot_paths):
        slot = trail_state.pop()
        if slot.get("entity") is not None:
            destroy(slot["entity"])

    for robot_index, path in enumerate(robot_paths):
        slot = trail_state[robot_index]
        path_len = len(path)
        if path_len < 2:
            if slot.get("entity") is not None:
                destroy(slot["entity"])
                slot["entity"] = None
            slot["path_len"] = 0
            continue

        if slot.get("entity") is not None and slot.get("path_len") == path_len:
            continue

        if slot.get("entity") is not None:
            destroy(slot["entity"])

        vertices = []
        for j in range(1, path_len):
            x0, y0 = path[j - 1]
            x1, y1 = path[j]
            px0, _, pz0 = world_to_scene(x0, y0, world_width, world_height)
            px1, _, pz1 = world_to_scene(x1, y1, world_width, world_height)
            vertices.append(Vec3(px0, height, pz0))
            vertices.append(Vec3(px1, height, pz1))

        mesh = Mesh(vertices=vertices, mode="line", thickness=thickness)
        color = _trail_color(cfg, robot_index, alpha)
        kwargs = {
            "parent": parent,
            "model": mesh,
            "color": color,
            "collider": None,
            "render_queue": render_queue,
        }
        if shader is not None:
            kwargs["shader"] = shader
            kwargs["unlit"] = True

        slot["entity"] = Entity(**kwargs)
        slot["path_len"] = path_len
