"""Large field trees at goal locations with plow (knock-down) animation."""

from __future__ import annotations

import math
from copy import copy

from core.geometry import world_to_scene
from core.visuals.asset_paths import model_exists, model_relative


def _resolve_model(rel: str | None) -> str | None:
    if not rel:
        return None
    parts = rel.replace("\\", "/").split("/")
    if model_exists(*parts):
        return model_relative(*parts)
    return None


def _resolve_goal_tree_models(goal_cfg: dict, corn_cfg: dict) -> tuple[str | None, str | None, str | None]:
    """Return (trunk_path, foliage_path, fallback_single_path)."""
    trunk = _resolve_model(goal_cfg.get("trunk_model", "models/trees/leaftree_trunk_ursina.obj"))
    foliage = _resolve_model(goal_cfg.get("foliage_model", "models/trees/leaftree_foliage_ursina.obj"))
    if trunk is not None and foliage is not None:
        return trunk, foliage, None

    single_candidates = [
        goal_cfg.get("model"),
        goal_cfg.get("fallback_model"),
        "models/trees/leaftree_ursina.obj",
        "models/trees/leaftree.obj",
        corn_cfg.get("model"),
        "models/corn/corn.obj",
    ]
    for rel in single_candidates:
        path = _resolve_model(rel)
        if path is not None:
            return None, None, path
    return None, None, None


def _unique_model(model_path: str):
    """Load an isolated mesh copy so color/plow changes never affect other instances."""
    from ursina import load_model

    return copy(load_model(model_path))


def _part_colors(goal_cfg: dict, corn_cfg: dict) -> dict[str, tuple[float, float, float, float]]:
    trunk = tuple(
        goal_cfg.get(
            "trunk_color",
            goal_cfg.get("stalk_color", corn_cfg.get("stalk_color", (0.45, 0.30, 0.15, 1.0))),
        )
    )
    leaf = tuple(goal_cfg.get("leaf_color", corn_cfg.get("leaf_color", (0.18, 0.65, 0.22, 1.0))))
    plowed = tuple(goal_cfg.get("plowed_color", (0.35, 0.28, 0.16, 1.0)))
    plowed_trunk = tuple(goal_cfg.get("plowed_trunk_color", plowed))
    plowed_leaf = tuple(goal_cfg.get("plowed_leaf_color", plowed))
    return {
        "trunk": trunk,
        "leaf": leaf,
        "plowed_trunk": plowed_trunk,
        "plowed_leaf": plowed_leaf,
    }


def _plant_color(goal_cfg: dict, corn_cfg: dict) -> tuple[float, float, float, float]:
    """Average standing color (used by callers that only need one tint)."""
    colors = _part_colors(goal_cfg, corn_cfg)
    trunk, leaf = colors["trunk"], colors["leaf"]
    return (
        (trunk[0] + leaf[0]) / 2,
        (trunk[1] + leaf[1]) / 2,
        (trunk[2] + leaf[2]) / 2,
        1.0,
    )


def _goal_scale(goal_cfg: dict, corn_cfg: dict, index: int) -> float:
    native_h = float(goal_cfg.get("model_height", corn_cfg.get("model_height", 14.0)))
    target_h = float(goal_cfg.get("plant_height", 36.0))
    base = target_h / max(native_h, 1e-3)
    spread = float(goal_cfg.get("scale_variation", 0.12))
    jitter = spread * math.sin(index * 2.17)
    return base * (1.0 + jitter)


def _goal_yaw(goal_cfg: dict, index: int) -> float:
    spread = float(goal_cfg.get("yaw_jitter_deg", 12.0))
    return spread * math.sin(index * 1.73 + 0.4)


def _mesh_y_offset(goal_cfg: dict, corn_cfg: dict, scale: float) -> float:
    if "model_base_y" in goal_cfg:
        return -float(goal_cfg["model_base_y"]) * scale
    native_h = float(goal_cfg.get("model_height", corn_cfg.get("model_height", 1.97)))
    return (native_h / 2.0) * scale


def _fall_side_tilt(goal_cfg: dict, index: int) -> float:
    spread = float(goal_cfg.get("fall_side_deg", 6.0))
    sign = 1.0 if index % 2 == 0 else -1.0
    return sign * spread * abs(math.sin(index * 1.31 + 0.2))


class GoalTree:
    """A standing tree that tips over when plowed (goal visited)."""

    def __init__(
        self,
        root,
        assembly,
        trunk_mesh,
        foliage_mesh,
        base_position: tuple[float, float, float],
        base_scale: float,
        tilt_deg: float,
        sink: float,
        plow_duration: float,
        trunk_color: tuple[float, float, float, float],
        leaf_color: tuple[float, float, float, float],
        plowed_trunk_color: tuple[float, float, float, float],
        plowed_leaf_color: tuple[float, float, float, float],
        fall_side_deg: float,
    ):
        self.root = root
        self.assembly = assembly
        self.trunk_mesh = trunk_mesh
        self.foliage_mesh = foliage_mesh
        self.base_position = base_position
        self.base_scale = base_scale
        self.tilt_deg = tilt_deg
        self.sink = sink
        self.plow_duration = plow_duration
        self.trunk_color = trunk_color
        self.leaf_color = leaf_color
        self.plowed_trunk_color = plowed_trunk_color
        self.plowed_leaf_color = plowed_leaf_color
        self.fall_side_deg = fall_side_deg
        self.visited = False
        self._plowing = False

    @property
    def mesh(self):
        """Backward-compatible handle for the visual assembly."""
        return self.assembly

    def _stop_animations(self):
        if hasattr(self.root, "animate_reset"):
            self.root.animate_reset()
        if hasattr(self.assembly, "animate_reset"):
            self.assembly.animate_reset()

    def reset(self):
        self.visited = False
        self._plowing = False
        self._stop_animations()
        self.root.position = self.base_position
        self.root.rotation_x = 0
        self.root.rotation_z = 0
        self.assembly.rotation_x = 0
        self.assembly.rotation_z = 0
        self.assembly.scale = (self.base_scale, self.base_scale, self.base_scale)
        self.assembly.enabled = True
        if self.trunk_mesh is not None:
            self.trunk_mesh.enabled = True
            self.trunk_mesh.color = self.trunk_color
        if self.foliage_mesh is not None:
            self.foliage_mesh.enabled = True
            self.foliage_mesh.color = self.leaf_color

    def update_visited(self, visited: bool):
        if visited and not self.visited:
            self._start_plow()
        elif not visited and self.visited:
            self.reset()
        self.visited = visited

    def _start_plow(self):
        if self._plowing:
            return
        self._plowing = True
        from ursina.curve import in_cubic, out_quad

        px, py, pz = self.base_position
        duration = self.plow_duration
        if self.trunk_mesh is not None:
            self.trunk_mesh.color = self.plowed_trunk_color
        if self.foliage_mesh is not None:
            self.foliage_mesh.color = self.plowed_leaf_color

        if abs(self.fall_side_deg) > 0.01:
            self.root.animate(
                "rotation_z",
                self.fall_side_deg,
                duration=duration * 0.18,
                curve=out_quad,
            )

        self.root.animate(
            "rotation_x",
            self.tilt_deg,
            duration=duration,
            delay=duration * 0.08,
            curve=in_cubic,
        )
        self.root.animate(
            "y",
            py - self.sink,
            duration=duration * 0.35,
            delay=duration * 0.72,
            curve=out_quad,
        )


def _create_tree_assembly(
    Entity,
    root,
    trunk_path: str | None,
    foliage_path: str | None,
    single_path: str | None,
    mesh_y: float,
    scale: float,
    colors: dict[str, tuple[float, float, float, float]],
):
    assembly = Entity(parent=root, position=(0, mesh_y, 0), scale=(scale, scale, scale), collider=None)
    trunk_mesh = None
    foliage_mesh = None

    if trunk_path and foliage_path:
        trunk_mesh = Entity(
            parent=assembly,
            model=_unique_model(trunk_path),
            color=colors["trunk"],
            texture=None,
            collider=None,
        )
        foliage_mesh = Entity(
            parent=assembly,
            model=_unique_model(foliage_path),
            color=colors["leaf"],
            texture=None,
            collider=None,
        )
    elif single_path:
        trunk_mesh = Entity(
            parent=assembly,
            model=_unique_model(single_path),
            color=_plant_color_from_dict(colors),
            texture=None,
            collider=None,
        )

    return assembly, trunk_mesh, foliage_mesh


def _plant_color_from_dict(colors: dict) -> tuple[float, float, float, float]:
    trunk, leaf = colors["trunk"], colors["leaf"]
    return (
        (trunk[0] + leaf[0]) / 2,
        (trunk[1] + leaf[1]) / 2,
        (trunk[2] + leaf[2]) / 2,
        1.0,
    )


def create_goal_plants(
    Entity,
    Vec3,
    goal_positions,
    world_width: float,
    world_height: float,
    goal_cfg: dict,
    corn_cfg: dict,
    parent=None,
) -> list[GoalTree]:
    trunk_path, foliage_path, single_path = _resolve_goal_tree_models(goal_cfg, corn_cfg)
    if not single_path and not (trunk_path and foliage_path):
        return []

    colors = _part_colors(goal_cfg, corn_cfg)
    tilt = float(goal_cfg.get("plow_tilt_deg", 88.0))
    sink = float(goal_cfg.get("plow_sink", 0.35))
    duration = float(goal_cfg.get("plow_duration", 1.1))

    plants: list[GoalTree] = []
    for index, goal in enumerate(goal_positions):
        gx, gy = float(goal[0]), float(goal[1])
        px, _, pz = world_to_scene(gx, gy, world_width, world_height)
        scale = _goal_scale(goal_cfg, corn_cfg, index)
        yaw = _goal_yaw(goal_cfg, index)
        mesh_y = _mesh_y_offset(goal_cfg, corn_cfg, scale)
        fall_side = _fall_side_tilt(goal_cfg, index)

        root = Entity(
            parent=parent,
            position=Vec3(px, 0.0, pz),
            rotation_y=yaw,
        )
        assembly, trunk_mesh, foliage_mesh = _create_tree_assembly(
            Entity,
            root,
            trunk_path,
            foliage_path,
            single_path,
            mesh_y,
            scale,
            colors,
        )
        plants.append(
            GoalTree(
                root=root,
                assembly=assembly,
                trunk_mesh=trunk_mesh,
                foliage_mesh=foliage_mesh,
                base_position=(px, 0.0, pz),
                base_scale=scale,
                tilt_deg=tilt,
                sink=sink,
                plow_duration=duration,
                trunk_color=colors["trunk"],
                leaf_color=colors["leaf"],
                plowed_trunk_color=colors["plowed_trunk"],
                plowed_leaf_color=colors["plowed_leaf"],
                fall_side_deg=fall_side,
            )
        )
    return plants
