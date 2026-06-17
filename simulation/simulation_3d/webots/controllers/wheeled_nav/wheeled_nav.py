"""Webots supervisor controller — runs CrossQ policy and animates the 3D field."""

from __future__ import annotations

import os
import sys
import traceback

# ---------------------------------------------------------------------------
# Fast bootstrap: set PYTHONPATH before any heavy imports.
# Webots kills the controller if supervisor.step() is not called within ~1 s.
# ---------------------------------------------------------------------------
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
WEBOTS_ROOT = os.path.dirname(os.path.dirname(CONTROLLER_DIR))
SIM_ROOT = os.path.dirname(WEBOTS_ROOT)
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

LOG_PATH = os.path.join(CONTROLLER_DIR, "wheeled_nav.log")


def _log(message: str) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except OSError:
        pass
    print(message, flush=True)


def _load_config() -> dict:
    import json

    config_path = os.path.join(CONTROLLER_DIR, "sim_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _set_goal_visited(goal_node, visited: bool):
    if goal_node is None:
        return
    try:
        shape = goal_node.getField("children").getMFNode(0)
        appearance = shape.getField("appearance").getSFNode()
        color_field = appearance.getField("baseColor")
        if color_field is None:
            material = appearance.getField("material")
            if material is not None and material.getSFNode() is not None:
                color_field = material.getSFNode().getField("diffuseColor")
        if color_field is None:
            return
        if visited:
            color_field.setSFColor([0.35, 0.35, 0.35])
        else:
            color_field.setSFColor([0.15, 0.75, 0.25])
    except Exception:
        pass


def _set_flag_visible(goal_node, visible: bool):
    if goal_node is None:
        return
    try:
        flag = goal_node.getField("children").getMFNode(1)
        scale = 1.0 if visible else 0.001
        flag.getField("scale").setSFVec3f([scale, scale, scale])
    except Exception:
        pass


def _idle_loop(supervisor, timestep: int) -> None:
    """Keep simulation alive after a startup error (so the scene stays visible)."""
    _log("Controller entered idle loop — fix errors above and reload the world.")
    while supervisor.step(timestep) != -1:
        pass


def main():
    # Truncate log at each run
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("=== wheeled_nav controller start ===\n")
    except OSError:
        pass

    _log(f"Python: {sys.executable}")
    _log(f"SIM_ROOT: {SIM_ROOT}")

    from controller import Supervisor

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Satisfy Webots watchdog before TensorFlow / SB3 import (can take several seconds).
    if supervisor.step(timestep) == -1:
        return

    try:
        import json
        import math

        from core.geometry import world_to_scene
        from core.policy import prepare_env

        cfg = _load_config()

        env_key = os.environ.get("MULTIBOTNAV_ENV_KEY", cfg.get("env_key", "env1"))
        num_robots = os.environ.get("MULTIBOTNAV_NUM_ROBOTS")
        num_robots = int(num_robots) if num_robots else cfg.get("num_robots")
        weights = os.environ.get("MULTIBOTNAV_WEIGHTS", cfg.get("weights_path"))
        random_policy = bool(cfg.get("random_policy", False))
        if random_policy:
            weights = None
        json_path = cfg.get("json_path") or None
        if json_path == "":
            json_path = None
        max_steps = int(cfg.get("max_steps", 1000))

        _log(
            f"Config: env={env_key} num_robots={num_robots} "
            f"random={random_policy} weights={weights}"
        )

        env, model, num_robots = prepare_env(
            env_key=env_key,
            num_robots=num_robots,
            weights_path=weights if weights and os.path.isfile(weights) else None,
            json_path=json_path,
            max_steps=max_steps,
            load_weights=not random_policy,
        )

        _log(
            f"Ready: robots={num_robots} policy={'CrossQ' if model else 'random'}"
        )

        robot_nodes = [supervisor.getFromDef(f"ROBOT_{i}") for i in range(num_robots)]
        goal_nodes = [
            supervisor.getFromDef(f"GOAL_{i}") for i in range(len(env.goal_positions))
        ]
        wind_node = supervisor.getFromDef("WIND_ARROW")

        missing = [i for i, n in enumerate(robot_nodes) if n is None]
        if missing:
            _log(f"WARNING: missing DEF nodes for ROBOT indices: {missing}")

        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        def sync_scene():
            for i, node in enumerate(robot_nodes):
                if node is None:
                    continue
                x, y, theta, _, _ = env.robots[i]
                px, _, pz = world_to_scene(x, y, env.world_width, env.world_height)
                node.getField("translation").setSFVec3f([px, 2.25, pz])
                node.getField("rotation").setSFRotation([0, 1, 0, -theta])

            for i, visited in enumerate(env.goal_visited):
                _set_goal_visited(goal_nodes[i], visited)
                _set_flag_visible(goal_nodes[i], not visited)

            if wind_node is not None:
                wind_node.getField("rotation").setSFRotation(
                    [0, 1, 0, math.radians(-env.wind_dir)]
                )
                try:
                    sx = 10.0 + env.wind_mag * 15.0
                    geom = (
                        wind_node.getField("children")
                        .getMFNode(0)
                        .getField("geometry")
                        .getSFNode()
                    )
                    geom.getField("size").setSFVec3f([sx, 0.5, 2.0])
                except Exception:
                    pass

        sync_scene()

        while supervisor.step(timestep) != -1:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            sync_scene()

            if terminated or truncated:
                _log(
                    f"Episode ended | steps: {steps} | reward: {total_reward:.1f} "
                    f"| term_cond: {info.get('term_cond', '?')}"
                )
                obs, _ = env.reset()
                total_reward = 0.0
                steps = 0
                sync_scene()

    except Exception:
        _log("CONTROLLER ERROR:\n" + traceback.format_exc())
        _idle_loop(supervisor, timestep)


if __name__ == "__main__":
    main()
