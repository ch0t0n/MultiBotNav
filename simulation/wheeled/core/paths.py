"""Path helpers for simulation/wheeled inside the MultiBotNav repository."""

from __future__ import annotations

import os
import sys

# simulation/wheeled/
SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MultiBotNav repo root (two levels above SIM_ROOT)
REPO_ROOT = os.path.normpath(os.path.join(SIM_ROOT, "..", ".."))


def ensure_repo_on_path() -> str:
    """Add the MultiBotNav repo root to ``sys.path`` for ``src`` imports."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


def resolve_path(*parts: str) -> str:
    """Resolve a path relative to the MultiBotNav repository root."""
    return os.path.abspath(os.path.join(REPO_ROOT, *parts))


def default_wheeled_json() -> str:
    """Default: ../../exp_sets/wheeled/wheeled_configs.json from simulation/wheeled."""
    return resolve_path("exp_sets", "wheeled", "wheeled_configs.json")


def default_weights_path(env_key: str) -> str:
    """Default: ../../trained_models/wheeled/best_model_env{N}_.../best_model.zip."""
    env_id = int(env_key.replace("env", ""))
    return resolve_path(
        "trained_models",
        "wheeled",
        f"best_model_env{env_id}_stage2_robust_wind",
        "best_model.zip",
    )
