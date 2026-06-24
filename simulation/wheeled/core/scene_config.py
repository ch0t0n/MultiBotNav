"""Load 3D scene visual parameters from ``config/scene_config.json``."""

from __future__ import annotations

import json
import os

from .paths import SIM_ROOT

_DEFAULT_PATH = os.path.join(SIM_ROOT, "config", "scene_config.json")


def default_scene_config_path() -> str:
    return _DEFAULT_PATH


def load_scene_config(path: str | None = None) -> dict:
    """Load scene visual config. Robot type profiles live in ``robot_model.DEFAULT_ROBOT_TYPES``."""
    path = path or _DEFAULT_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Scene config not found: {path}\n"
            "Expected config/scene_config.json in the simulation_3d folder."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
