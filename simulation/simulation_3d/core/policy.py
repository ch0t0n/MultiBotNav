"""Load CrossQ checkpoints and build simulation environments."""

from __future__ import annotations

import os

from .env_loader import load_env_from_json
from .multi_wheeled import MultiWheeled
from .paths import default_weights_path, default_wheeled_json


def infer_num_robots(observation_space) -> int:
    obs_dim = int(observation_space.shape[0])
    if (obs_dim - 1) % 5 != 0:
        raise ValueError(
            f"Cannot infer num_robots from observation size {obs_dim}; "
            "expected full-mode shape 5N+1."
        )
    return (obs_dim - 1) // 5


def resolve_num_robots(weights_path: str, num_robots: int | None) -> int:
    from sb3_contrib import CrossQ

    if num_robots is not None:
        return num_robots
    checkpoint = CrossQ.load(weights_path)
    inferred = infer_num_robots(checkpoint.observation_space)
    del checkpoint
    return inferred


def build_env(
    env_key: str,
    num_robots: int | None,
    json_path: str | None = None,
    max_steps: int = 1000,
    uncertainty_mode: str = "wind_only",
    dr_mode: str = "wind",
) -> tuple[MultiWheeled, dict]:
    json_path = json_path or default_wheeled_json()
    env_params = load_env_from_json(json_path, key=env_key)
    if num_robots is not None:
        env_params["NUM_ROBOTS"] = num_robots
        env_params["ROBOT_INIT_CONFIGS"] = env_params["ROBOT_INIT_CONFIGS"][
            :num_robots
        ]
    env = MultiWheeled(
        env_params=env_params,
        max_steps=max_steps,
        uncertainty_mode=uncertainty_mode,
        dr_mode=dr_mode,
    )
    return env, env_params


def prepare_env(
    env_key: str,
    num_robots: int | None,
    weights_path: str | None = None,
    json_path: str | None = None,
    max_steps: int = 1000,
    load_weights: bool = True,
) -> tuple[MultiWheeled, object | None, int]:
    json_path = json_path or default_wheeled_json()
    env_params = load_env_from_json(json_path, key=env_key)

    resolved_weights = weights_path
    if load_weights and not resolved_weights:
        candidate = default_weights_path(env_key)
        resolved_weights = candidate if os.path.isfile(candidate) else None

    if num_robots is None and resolved_weights and os.path.isfile(resolved_weights):
        num_robots = resolve_num_robots(resolved_weights, None)
    if num_robots is None:
        num_robots = env_params["NUM_ROBOTS"]

    env_params["NUM_ROBOTS"] = num_robots
    env_params["ROBOT_INIT_CONFIGS"] = env_params["ROBOT_INIT_CONFIGS"][:num_robots]

    env = MultiWheeled(
        env_params=env_params,
        max_steps=max_steps,
        uncertainty_mode="wind_only",
        dr_mode="wind",
    )

    model = None
    if load_weights and resolved_weights and os.path.isfile(resolved_weights):
        from sb3_contrib import CrossQ

        model = CrossQ.load(resolved_weights, env=env)
    return env, model, num_robots
