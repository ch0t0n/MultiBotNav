#!/usr/bin/env python3
"""Run wheeled-robot navigation in a 3D agricultural environment."""

from __future__ import annotations

import argparse
import os
import sys

SIM_ROOT = os.path.dirname(os.path.abspath(__file__))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

from core.paths import default_weights_path, default_wheeled_json  # noqa: E402
from core.policy import prepare_env  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D agricultural simulation for trained wheeled-robot policies."
    )
    parser.add_argument(
        "--backend",
        choices=["webots", "ursina", "isaac", "headless"],
        default="webots",
        help="3D renderer backend (default: webots).",
    )
    parser.add_argument(
        "--env-key",
        default="env10",
        help="Environment key from wheeled_configs.json (env1 … env10).",
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=None,
        help="Robot count (2–5). Default: infer from checkpoint.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to CrossQ best_model.zip. Default: trained_models/wheeled/...",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Path to wheeled_configs.json. Default: exp_sets/wheeled/...",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30, help="Simulation render FPS.")
    parser.add_argument(
        "--random-policy",
        action="store_true",
        help="Ignore checkpoint and sample random actions.",
    )
    parser.add_argument(
        "--scene-config",
        default=None,
        help="Path to scene_config.json (3D heights, colors, crop layout).",
    )
    parser.add_argument(
        "--robot-type",
        default=None,
        choices=["tractor", "delivery", "tractor_shovel", "rover"],
        help="Override robots.type in scene_config.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = None if args.random_policy else (args.weights or default_weights_path(args.env_key))
    json_path = args.json or default_wheeled_json()

    if weights and not os.path.isfile(weights):
        print(f"Warning: checkpoint not found at {weights}")
        print("Running with random policy. Train a model or pass --weights explicitly.")
        weights = None

    if args.backend == "webots":
        from backends.webots_bridge import launch_webots

        num_robots = args.num_robots
        sys.exit(
            launch_webots(
                env_key=args.env_key,
                num_robots=num_robots,
                weights=weights,
                json_path=json_path,
                random_policy=args.random_policy,
                max_steps=args.max_steps,
            )
        )

    env, model, num_robots = prepare_env(
        env_key=args.env_key,
        num_robots=args.num_robots,
        weights_path=weights,
        json_path=json_path,
        max_steps=args.max_steps,
    )
    print(f"Environment: {args.env_key} | robots: {num_robots} | backend: {args.backend}")
    print(f"JSON: {json_path}")
    if model is not None:
        print(f"Policy: {weights}")
    else:
        print("Policy: random")

    if args.backend == "ursina":
        from backends.ursina_agri import run_ursina_simulation

        run_ursina_simulation(
            env,
            model,
            fps=args.fps,
            scene_config_path=args.scene_config,
            robot_type=args.robot_type,
        )
    elif args.backend == "isaac":
        from backends.isaac_bridge import run_isaac_instructions

        run_isaac_instructions(args.env_key, num_robots, weights)
    elif args.backend == "headless":
        from backends.headless import run_headless

        run_headless(env, model, episodes=3)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
