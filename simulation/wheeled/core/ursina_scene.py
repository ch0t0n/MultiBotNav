"""Ursina 3D agricultural field renderer for trained wheeled-robot policies."""

from __future__ import annotations

import math

from core.meshing import extrude_polygon_mesh, make_lit_mesh
from core.multi_wheeled import MultiWheeled
from core.scene_config import load_scene_config
from core.visuals.asset_paths import configure_ursina_assets
from core.visuals.camera_control import StableEditorCamera
from core.visuals.goal_plants import create_goal_plants
from core.visuals.grass_field import build_grass_field
from core.visuals.landscape import _configure_render_pipeline, build_landscape, setup_atmosphere
from core.visuals.robot_model import create_wheeled_robot, sync_wheeled_robot
from core.visuals.robot_trails import draw_robot_trails


def _field_rotation_y(cfg: dict) -> float:
    """Yaw correction so the tilted camera sees the same layout as top-down training."""
    field_cfg = cfg.get("field", {})
    if "rotation_y_deg" in field_cfg:
        return float(field_cfg["rotation_y_deg"])
    cam_cfg = cfg.get("camera", {})
    if cam_cfg.get("orthographic", False):
        return 0.0
    return float(cam_cfg.get("azimuth_deg", 0.0))


def _camera_pose(
    world_width: float,
    world_height: float,
    cam_cfg: dict | None = None,
    window=None,
) -> dict:
    """Compute scene camera position and focus point from config."""
    cam_cfg = cam_cfg or {}
    margin = float(cam_cfg.get("margin", 1.12))
    max_dim = max(world_width, world_height) * margin
    look_y = float(cam_cfg.get("look_at_y", 0.0))
    focus = (0.0, look_y, 0.0)

    if cam_cfg.get("orthographic", False):
        aspect = max(window.aspect_ratio, 0.25) if window else 1.0
        vert_extent = max(world_height * margin, (world_width * margin) / aspect)
        height = max(world_width, world_height) * 1.5
        return {
            "position": (0.0, height, 0.0),
            "focus": focus,
            "orthographic": True,
            "fov": vert_extent,
            "perspective_fov": float(cam_cfg.get("fov", 42)),
        }

    elevation = float(cam_cfg.get("elevation_deg", 48))
    azimuth = float(cam_cfg.get("azimuth_deg", 0.0))
    distance = max_dim * float(cam_cfg.get("distance_scale", 1.45))

    elev_r = math.radians(elevation)
    az_r = math.radians(azimuth)
    horizontal = distance * math.cos(elev_r)
    cam_y = distance * math.sin(elev_r)
    cam_x = horizontal * math.sin(az_r)
    cam_z = -horizontal * math.cos(az_r)

    return {
        "position": (cam_x, cam_y, cam_z),
        "focus": focus,
        "orthographic": False,
        "fov": float(cam_cfg.get("fov", 42)),
        "perspective_fov": float(cam_cfg.get("fov", 42)),
    }


def _apply_fixed_camera(camera, pose: dict):
    """Apply a fixed camera pose (no mouse control)."""
    from ursina import Vec3

    focus = Vec3(*pose["focus"])
    if pose["orthographic"]:
        camera.orthographic = True
        camera.fov = pose["fov"]
        camera.position = Vec3(*pose["position"])
        camera.rotation_x = 90
        camera.rotation_y = 0
        camera.rotation_z = 0
        return

    camera.orthographic = False
    camera.fov = pose["fov"]
    camera.position = Vec3(*pose["position"])
    camera.look_at(focus)


def _enable_editor_camera(camera, pose: dict, cam_cfg: dict | None = None):
    """Attach orbit camera using the fixed overview as the initial view."""
    from ursina import Vec2, Vec3

    cam_cfg = cam_cfg or {}
    if pose["orthographic"]:
        _apply_fixed_camera(camera, pose)
        return None

    focus = Vec3(*pose["focus"])
    cam_pos = Vec3(*pose["position"])
    distance = (cam_pos - focus).length()
    elevation = float(cam_cfg.get("elevation_deg", 48))
    azimuth = float(cam_cfg.get("azimuth_deg", 0.0))
    pan = float(cam_cfg.get("pan_speed", 4.0))

    camera.orthographic = False
    camera.fov = pose["fov"]

    ec = StableEditorCamera(
        enabled=False,
        rotation_speed=float(cam_cfg.get("rotation_speed", 110)),
        pan_speed=Vec2(pan, pan),
        zoom_smoothing=float(cam_cfg.get("zoom_smoothing", 18)),
    )
    ec.position = focus
    ec.start_position = focus
    ec.rotation_x = elevation
    ec.rotation_y = azimuth
    ec.smoothing_helper.rotation_x = elevation
    ec.smoothing_helper.rotation_y = azimuth
    ec.perspective_fov = pose["perspective_fov"]

    camera.editor_position = Vec3(0, 0, -distance)
    ec.enabled = True
    ec.target_z = -distance
    camera.z = -distance
    return ec


def _setup_scene_camera(
    camera,
    window,
    world_width: float,
    world_height: float,
    cam_cfg: dict | None = None,
):
    """Place camera on the south edge center, looking north across the field."""
    cam_cfg = cam_cfg or {}
    pose = _camera_pose(world_width, world_height, cam_cfg, window)

    if cam_cfg.get("mouse_control_enabled", False):
        return _enable_editor_camera(camera, pose, cam_cfg)

    _apply_fixed_camera(camera, pose)
    return None


def _configure_window_ui(window, ui_cfg: dict | None = None):
    """Hide Ursina's redundant close button and tune overlay text size."""
    ui_cfg = ui_cfg or {}
    if hasattr(window, "exit_button"):
        window.exit_button.visible = False
        window.exit_button.enabled = False
    if hasattr(window, "fps_counter"):
        window.fps_counter.enabled = ui_cfg.get("show_fps", True)
        window.fps_counter.scale = float(ui_cfg.get("fps_text_scale", 0.55))
    for counter_name in ("entity_counter", "collider_counter"):
        counter = getattr(window, counter_name, None)
        if counter is None:
            continue
        counter.enabled = False
        label = getattr(counter, "text_entity", None)
        if label is not None:
            label.enabled = False


def _setup_scene_environment(cfg: dict | None = None):
    """Sky, sun, and ambient lighting."""
    setup_atmosphere((cfg or {}).get("environment"))


class AgriculturalScene3D:
    """Build and update a 3D agricultural visualization from env state."""

    def __init__(self, env: MultiWheeled, scene_config_path: str | None = None, robot_type: str | None = None):
        from ursina import (
            Entity,
            Mesh,
            Text,
            Vec3,
            camera,
            window,
        )

        self._Entity = Entity
        self._Mesh = Mesh
        self._Vec3 = Vec3
        self._Text = Text
        self.env = env
        self.cfg = load_scene_config(scene_config_path)
        if robot_type:
            self.cfg.setdefault("robots", {})["type"] = robot_type
        configure_ursina_assets()
        self.world_width = env.world_width
        self.world_height = env.world_height

        window.title = "MultiBotNav — Agricultural 3D Simulation"
        window.borderless = False
        _configure_window_ui(window, self.cfg.get("ui"))

        _setup_scene_environment(self.cfg)

        self.scene_root = Entity()
        self.field_rotation_y = _field_rotation_y(self.cfg)
        self.field_root = Entity(parent=self.scene_root, rotation_y=self.field_rotation_y)

        build_landscape(
            Entity,
            self.world_width,
            self.world_height,
            self.cfg,
            parent=self.scene_root,
        )
        self._build_field()
        self._build_obstacles()
        self._build_goals()
        self._build_robots()
        self._build_wind_indicator()
        self._build_hud()

        self.editor_camera = _setup_scene_camera(
            camera,
            window,
            self.world_width,
            self.world_height,
            self.cfg.get("camera"),
        )

        self.trail_state: list[dict] = []

    def _build_field(self):
        build_grass_field(
            self._Entity,
            self._Mesh,
            self.world_width,
            self.world_height,
            self.cfg.get("grass", {}),
            self.cfg.get("field", {}),
            parent=self.field_root,
        )

    def _build_obstacles(self):
        Entity = self._Entity
        Mesh = self._Mesh
        obs_cfg = self.cfg["obstacles"]
        base_y = float(obs_cfg["base_y"])
        styles = obs_cfg["styles"]
        default_color = tuple(obs_cfg.get("color", (0.50, 0.34, 0.16, 1.0)))
        self.obstacle_entities = []

        for idx, poly in enumerate(self.env.obstacles):
            style = styles[idx % len(styles)]
            height = float(style["height"])
            color = tuple(style["color"]) if "color" in style else default_color
            vertices, triangles = extrude_polygon_mesh(
                poly,
                self.world_width,
                self.world_height,
                height,
                base_y=base_y,
            )
            mesh = make_lit_mesh(Mesh, vertices, triangles)
            if mesh is None:
                continue
            ent = Entity(
                parent=self.field_root,
                model=mesh,
                color=color,
                double_sided=True,
            )
            self.obstacle_entities.append(ent)

    def _build_goals(self):
        goal_cfg = dict(self.cfg["goals"])
        if "trees" in self.cfg:
            goal_cfg["trees"] = self.cfg["trees"]
        self.goal_plants = create_goal_plants(
            self._Entity,
            self.env.goal_positions,
            self.world_width,
            self.world_height,
            goal_cfg,
            parent=self.field_root,
        )

    def reset_goal_plants(self):
        for plant in self.goal_plants:
            plant.reset()

    def _build_robots(self):
        self.robot_entities = []
        robot_cfg = self.cfg["robots"]
        length = self.env.ROBOT_LENGTH
        width = self.env.ROBOT_WIDTH
        for i in range(self.env.NUM_ROBOTS):
            self.robot_entities.append(
                create_wheeled_robot(
                    self._Entity,
                    self._Vec3,
                    length,
                    width,
                    robot_cfg,
                    parent=self.field_root,
                )
            )

    def _build_wind_indicator(self):
        Entity = self._Entity
        wind_cfg = self.cfg["wind"]
        wy = float(wind_cfg["arrow_height"])
        ox = float(wind_cfg["offset_x"])
        oz = float(wind_cfg["offset_z"])
        self.wind_arrow = Entity(
            parent=self.field_root,
            model="cube",
            scale=(20, 0.5, 2),
            color=(0.7, 0.8, 0.95, 0.8),
            position=(self.world_width / 2 - ox, wy, -self.world_height / 2 + oz),
        )

    def _build_hud(self):
        self.status_text = self._Text(
            text="",
            position=(-0.85, 0.48),
            scale=1.2,
            origin=(0, 0),
        )

    def sync_from_env(self, info: dict | None = None):
        env = self.env

        for i, robot_ent in enumerate(self.robot_entities):
            x, y, theta, _, _delta = env.robots[i]
            sync_wheeled_robot(
                robot_ent,
                x,
                y,
                theta,
                self.world_width,
                self.world_height,
                self._Vec3,
            )

        for plant, visited in zip(self.goal_plants, env.goal_visited):
            plant.update_visited(visited)

        self.wind_arrow.rotation_y = -env.wind_dir
        self.wind_arrow.scale_x = 10 + env.wind_mag * 15

        wind_line = f"Wind: {env.wind_mag:.2f} m/s @ {env.wind_dir:.0f}°"
        if info:
            self.status_text.text = (
                f"Step {info.get('step_count', 0)} | "
                f"Goals {info.get('goals_visited', 0)}/{len(env.goal_positions)} | "
                f"{wind_line}"
            )
        else:
            self.status_text.text = wind_line

    def draw_trails(self):
        from ursina import Mesh, destroy

        draw_robot_trails(
            self._Entity,
            Mesh,
            self._Vec3,
            destroy,
            self.trail_state,
            self.env.robot_paths,
            self.world_width,
            self.world_height,
            self.cfg.get("trails"),
            parent=self.field_root,
        )


def run_ursina_simulation(
    env: MultiWheeled,
    model,
    fps: int = 30,
    auto_reset: bool = True,
    scene_config_path: str | None = None,
    robot_type: str | None = None,
):
    """Run the trained (or random) policy with Ursina 3D rendering."""
    from ursina import Button, Entity, Ursina, color, destroy, held_keys, time

    scene_cfg = load_scene_config(scene_config_path)
    ui_cfg = scene_cfg.get("ui", {})
    _configure_render_pipeline(scene_cfg.get("environment"))

    app = Ursina(vsync=True, size=(1280, 720))
    scene = AgriculturalScene3D(
        env,
        scene_config_path=scene_config_path,
        robot_type=robot_type,
    )

    require_start = bool(ui_cfg.get("require_start_button", True))
    state = {
        "obs": env.reset()[0],
        "step_accum": 0.0,
        "total_reward": 0.0,
        "episode_steps": 0,
        "paused": False,
        "finished": False,
        "started": not require_start,
    }
    scene.sync_from_env()

    start_ui: list = []

    def _begin_simulation():
        state["started"] = True
        state["obs"] = env.reset()[0]
        state["total_reward"] = 0.0
        state["episode_steps"] = 0
        state["step_accum"] = 0.0
        scene.reset_goal_plants()
        scene.sync_from_env()
        scene.draw_trails()
        for ent in start_ui:
            destroy(ent)
        start_ui.clear()

    if require_start:
        hint = scene._Text(
            text=str(ui_cfg.get("start_hint_text", "Adjust the camera, then click Start")),
            origin=(0, 0),
            y=0.42,
            z=-1,
            scale=1.35,
            background=True,
        )
        start_btn = Button(
            text=str(ui_cfg.get("start_button_text", "Start Simulation")),
            scale=(0.28, 0.09),
            y=-0.42,
            z=-1,
            color=color.rgb(0.2, 0.55, 0.25),
            highlight_color=color.rgb(0.28, 0.68, 0.32),
            pressed_color=color.rgb(0.15, 0.42, 0.18),
            text_color=color.white,
        )
        start_btn.on_click = _begin_simulation
        start_ui.extend([hint, start_btn])

    class SimulationLoop(Entity):
        def update(self):
            if held_keys["escape"]:
                env.close()
                import sys

                sys.exit(0)

            if not state["started"]:
                return

            if held_keys["space"]:
                state["paused"] = not state["paused"]
                held_keys["space"] = False

            if state["finished"] or state["paused"]:
                return

            state["step_accum"] += time.dt
            interval = 1.0 / max(1, fps)
            if state["step_accum"] < interval:
                return
            state["step_accum"] = 0.0

            if model is not None:
                action, _ = model.predict(state["obs"], deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            state["obs"] = obs
            state["total_reward"] += reward
            state["episode_steps"] += 1
            scene.sync_from_env(info)
            scene.draw_trails()

            if terminated or truncated:
                print(
                    f"Episode ended | steps: {state['episode_steps']} | "
                    f"reward: {state['total_reward']:.1f} "
                    f"| term_cond: {info.get('term_cond', '?')}"
                )
                if info.get("term_cond") == "all_goals":
                    state["finished"] = True
                    return
                if auto_reset:
                    state["obs"] = env.reset()[0]
                    state["total_reward"] = 0.0
                    state["episode_steps"] = 0
                    scene.reset_goal_plants()
                    scene.sync_from_env()
                    scene.draw_trails()

    SimulationLoop()
    app.run()
