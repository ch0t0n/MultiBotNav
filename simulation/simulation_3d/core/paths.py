"""Project path helpers for simulation_3d."""

from __future__ import annotations

import os


SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SIM_ROOT)


def resolve_path(*parts: str) -> str:
    """Resolve a path relative to the ICTAI_2026 project root."""
    return os.path.abspath(os.path.join(PROJECT_ROOT, *parts))


def default_wheeled_json() -> str:
    candidates = [
        resolve_path("exp_sets", "wheeled", "wheeled_configs.json"),
        resolve_path("github", "MultiBotNav", "exp_sets", "wheeled", "wheeled_configs.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def default_weights_path(env_key: str) -> str:
    env_id = int(env_key.replace("env", ""))
    candidates = [
        resolve_path(
            "trained_models",
            "wheeled",
            f"best_model_env{env_id}_stage2_robust_wind",
            "best_model.zip",
        ),
        resolve_path(
            "github",
            "MultiBotNav",
            "trained_models",
            "wheeled",
            f"best_model_env{env_id}_stage2_robust_wind",
            "best_model.zip",
        ),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]
