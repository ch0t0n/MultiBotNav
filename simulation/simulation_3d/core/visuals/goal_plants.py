"""Large field trees at goal locations with plow (knock-down) animation."""

from __future__ import annotations

import math

from core.geometry import world_to_scene
from core.visuals.tree_models import (
    create_tree_entity,
    foliage_tint_from_cfg,
    goal_variant,
    mesh_base_offset,
    uniform_scale,
)


def _goal_scale(goal_cfg: dict, variant: str, index: int) -> float:
    target_h = float(goal_cfg.get("plant_height", 36.0))
    base = uniform_scale(variant, target_h)
    spread = float(goal_cfg.get("scale_variation", 0.12))
    jitter = spread * math.sin(index * 2.17)
    return base * (1.0 + jitter)


def _goal_yaw(goal_cfg: dict, index: int) -> float:
    spread = float(goal_cfg.get("yaw_jitter_deg", 12.0))
    return spread * math.sin(index * 1.73 + 0.4)


def _fall_side_tilt(goal_cfg: dict, index: int) -> float:
    spread = float(goal_cfg.get("fall_side_deg", 6.0))
    sign = 1.0 if index % 2 == 0 else -1.0
    return sign * spread * abs(math.sin(index * 1.31 + 0.2))


class GoalTree:
    """A standing tree that tips over when plowed (goal visited)."""

    def __init__(
        self,
        root,
        mesh_root,
        trunk_mesh,
        foliage_mesh,
        base_position: tuple[float, float, float],
        base_scale: float,
        tilt_deg: float,
        sink: float,
        plow_duration: float,
        fall_side_deg: float,
    ):
        self.root = root
        self.mesh_root = mesh_root
        self.assembly = mesh_root
        self.trunk_mesh = trunk_mesh
        self.foliage_mesh = foliage_mesh
        self.tree_mesh = trunk_mesh
        self.base_position = base_position
        self.base_scale = base_scale
        self.tilt_deg = tilt_deg
        self.sink = sink
        self.plow_duration = plow_duration
        self.fall_side_deg = fall_side_deg
        self.visited = False
        self._plowing = False

    @property
    def mesh(self):
        """Backward-compatible handle for the visual assembly."""
        return self.mesh_root

    def _stop_animations(self):
        if hasattr(self.root, "animate_reset"):
            self.root.animate_reset()
        if hasattr(self.mesh_root, "animate_reset"):
            self.mesh_root.animate_reset()

    def reset(self):
        self.visited = False
        self._plowing = False
        self._stop_animations()
        self.root.position = self.base_position
        self.root.rotation_x = 0
        self.root.rotation_z = 0
        self.mesh_root.rotation_x = 0
        self.mesh_root.rotation_z = 0
        self.mesh_root.enabled = True
        self.mesh_root.scale = (self.base_scale, self.base_scale, self.base_scale)
        if self.trunk_mesh is not None:
            self.trunk_mesh.enabled = True
        if self.foliage_mesh is not None:
            self.foliage_mesh.enabled = True

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


def create_goal_plants(
    Entity,
    Vec3,
    goal_positions,
    world_width: float,
    world_height: float,
    goal_cfg: dict,
    parent=None,
) -> list[GoalTree]:
    variant = goal_variant(goal_cfg)
    tilt = float(goal_cfg.get("plow_tilt_deg", 88.0))
    sink = float(goal_cfg.get("plow_sink", 0.35))
    duration = float(goal_cfg.get("plow_duration", 1.1))

    plants: list[GoalTree] = []
    for index, goal in enumerate(goal_positions):
        gx, gy = float(goal[0]), float(goal[1])
        px, _, pz = world_to_scene(gx, gy, world_width, world_height)
        scale = _goal_scale(goal_cfg, variant, index)
        yaw = _goal_yaw(goal_cfg, index)
        mesh_y = mesh_base_offset(variant, scale, goal_cfg)
        fall_side = _fall_side_tilt(goal_cfg, index)

        root, trunk_mesh, foliage_mesh = create_tree_entity(
            Entity,
            parent,
            variant,
            position=(px, 0.0, pz),
            rotation_y=yaw,
            scale=scale,
            y_offset=mesh_y,
            foliage_tint=foliage_tint_from_cfg(goal_cfg),
        )
        plants.append(
            GoalTree(
                root=root,
                mesh_root=trunk_mesh.parent,
                trunk_mesh=trunk_mesh,
                foliage_mesh=foliage_mesh,
                base_position=(px, 0.0, pz),
                base_scale=scale,
                tilt_deg=tilt,
                sink=sink,
                plow_duration=duration,
                fall_side_deg=fall_side,
            )
        )
    return plants
