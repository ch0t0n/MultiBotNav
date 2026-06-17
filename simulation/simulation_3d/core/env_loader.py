"""Load wheeled environment parameters from the consolidated JSON file."""

from __future__ import annotations

import json

import numpy as np


def scale_polygon_about_centroid(poly_pts, scale: float):
    """Scale polygon vertices about their centroid by *scale* (< 1 shrinks)."""
    pts = np.asarray(poly_pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)
    scaled = centroid + scale * (pts - centroid)
    return [tuple(p) for p in scaled.tolist()]


def load_env_from_json(json_path: str, key: str = "env1") -> dict:
    """
    Load a single wheeled-robot environment config from the consolidated JSON.

    Returns a flat env_params dict with keys expected by MultiWheeled.
    Obstacles in env2-env9 are shrunk to 30% (matching training).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if key not in raw:
        raise KeyError(
            f"Environment key '{key}' not found in {json_path}. "
            f"Available keys: {list(raw.keys())}"
        )

    cfg = raw[key]
    robots = cfg["robots"]
    goals = cfg["goals"]
    obstacles = cfg["obstacles"]

    if key.startswith("env"):
        env_idx = int(key.replace("env", ""))
        if 2 <= env_idx <= 9:
            obstacles = [scale_polygon_about_centroid(poly, 0.30) for poly in obstacles]

    return {
        "SCREEN_WIDTH": float(cfg["screen"]["width"]),
        "SCREEN_HEIGHT": float(cfg["screen"]["height"]),
        "ROBOT_LENGTH": float(robots["length"]),
        "ROBOT_WIDTH": float(robots["width"]),
        "MAX_SPEED": float(robots["max_speed"]),
        "MAX_STEER": float(robots["max_steer"]),
        "NUM_ROBOTS": int(robots["num_robots"]),
        "ROBOT_INIT_CONFIGS": [
            (float(c[0]), float(c[1]), float(np.radians(c[2])))
            for c in robots["configs"]
        ],
        "NUM_GOALS": int(goals["num_goals"]),
        "GOAL_SIZE": float(goals["goal_size"]),
        "GOAL_POSITIONS": [tuple(p) for p in goals["positions"]],
        "NUM_OBSTACLES": len(obstacles),
        "OBSTACLES": [[tuple(v) for v in poly] for poly in obstacles],
    }
