#!/usr/bin/env python3
"""
train.py — unified training script for all experiments.

Handles all experiment steps via command-line arguments:
  Step 1  --experiment main     (default HPs)
  Step 3  --experiment main     --hyperparams_json <path>   (tuned HPs)
  Step 4  --experiment ablation_reward  --ablation {full|no_term|no_path}
  Step 5  --experiment ablation_obs     --ablation {full|no_pos|no_inf_hist|pos_only}
  Step 6  --experiment ablation_uncertainty  --ablation {full|wind_only|act_only|deterministic}
  Step 7  --experiment dr       --ablation {none|wind|full}

Usage example:
  python train.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \
                  --steps 2000000 --device cuda --experiment main

  # Wheeled robot training:
  python train.py --algorithm CrossQ --set 1 --robot_type wheeled --seed 42 \
                  --steps 2000000 --device cuda --experiment main
"""

import os
import json
import argparse
import numpy as np

import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    EvalCallback, CallbackList, LogEveryNTimesteps)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiUAV, MultiWheeled
from src.utils import read_uav_json, read_wheeled_json

# ================================================================
# Constants
# ================================================================

ALGORITHMS = {
    "A2C":    (A2C,    "MlpPolicy"),
    "ARS":    (ARS,    "LinearPolicy"),
    "PPO":    (PPO,    "MlpPolicy"),
    "TRPO":   (TRPO,   "MlpPolicy"),
    "CrossQ": (CrossQ, "MlpPolicy"),
    "TQC":    (TQC,    "MlpPolicy"),
}

# Experiment → env constructor kwarg name
EXPERIMENT_MAP = {
    "main":                  None,               # no ablation kwarg
    "ablation_reward":       "reward_ablation",
    "ablation_obs":          "obs_mode",
    "ablation_uncertainty":  "uncertainty_mode",
    "dr":                    "dr_mode",
}

# Valid ablation choices per experiment — must match env.py assertions exactly
VALID_ABLATIONS = {
    "main":                  {None},
    "ablation_reward":       {"full", "no_term", "no_path"},
    "ablation_obs":          {"full", "no_pos", "no_inf_hist", "pos_only"},
    "ablation_uncertainty":  {"full", "wind_only", "act_only", "deterministic"},
    "dr":                    {"none", "wind", "full"},
}

# Default ablation value per experiment type (used when --ablation not given)
EXPERIMENT_DEFAULTS = {
    "main":                  None,
    "ablation_reward":       "full",
    "ablation_obs":          "full",
    "ablation_uncertainty":  "full",
    "dr":                    "none",
}

UAV_JSON_PATH      = os.path.join('exp_sets', 'uav', 'cont_sets.json')
WHEELED_JSON_PATH  = os.path.join('exp_sets', 'wheeled', 'wheeled_configs.json')

# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Unified RL training script")

    p.add_argument("--algorithm",     type=str,   required=True,
                   choices=list(ALGORITHMS.keys()))
    p.add_argument("--set",           type=int,   required=True,
                   help="Environment variation index (1–10)")
    p.add_argument("--robot_type",    type=str,   default="uav",
                   choices=["uav", "wheeled"],
                   help="Robot platform: 'uav' (MultiUAV) or 'wheeled' (MultiWheeled)")
    p.add_argument("--num_robots",    type=int,   default=3,
                   help="Number of robots (2-5 for both UAV and wheeled)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--steps",         type=int,   default=int(2e6))
    p.add_argument("--verbose",       type=int,   default=1)
    p.add_argument("--log_steps",     type=int,   default=10_000)
    p.add_argument("--device",        type=str,   default="cpu")
    p.add_argument("--num_envs",      type=int,   default=4,
                   help="Number of parallel training envs")
    p.add_argument("--max_steps",     type=int,   default=1000,
                   help="Maximum steps per episode")
    p.add_argument("--n_eval_eps",    type=int,   default=5,
                   help="Episodes per EvalCallback evaluation")

    # experiment-control arguments
    p.add_argument("--experiment",    type=str,   default="main",
                   choices=list(EXPERIMENT_MAP.keys()),
                   help="Which experiment this run belongs to")
    p.add_argument("--ablation",      type=str,   default=None,
                   help="Ablation condition within the selected experiment")
    p.add_argument("--hyperparams_json", type=str, default=None,
                   help="Path to best_hyperparams.json (tuned HPs). "
                        "If omitted, SB3 defaults are used.")
    p.add_argument("--log_root",      type=str,
                   default=os.path.join('logs'),
                   help="Root directory for all logs and saved models")

    args = p.parse_args()

    # Validate --ablation against the chosen experiment
    effective_ablation = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
    valid = VALID_ABLATIONS[args.experiment]
    if effective_ablation not in valid:
        p.error(
            f"--ablation '{effective_ablation}' is not valid for "
            f"--experiment '{args.experiment}'. "
            f"Choose from: {sorted(v for v in valid if v is not None)}"
        )

    return args


# ================================================================
# Utilities
# ================================================================

# def load_hyperparams(json_path: str, algorithm: str) -> dict:
#     """Load tuned hyperparameters for one algorithm from JSON."""
#     if json_path is None or not os.path.exists(json_path):
#         return {}
#     with open(json_path) as f:
#         data = json.load(f)
#     hp = data.get(algorithm, {}).get("params", {})
#     print(f"  Loaded tuned HPs for {algorithm}: {hp}")
#     return hp

def load_hyperparams(json_path: str, algorithm: str) -> dict:
    """Load tuned hyperparameters for one algorithm from JSON."""
    if json_path is None:
        return {}

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"--hyperparams_json was provided but does not exist: {json_path}"
        )

    with open(json_path) as f:
        data = json.load(f)

    if algorithm not in data or "params" not in data[algorithm]:
        raise KeyError(
            f"No tuned hyperparameters found for algorithm '{algorithm}' "
            f"in {json_path}."
        )

    hp = data[algorithm]["params"]
    if not hp:
        raise ValueError(
            f"Tuned hyperparameter entry for '{algorithm}' in {json_path} "
            "is empty."
        )

    print(f"  Loaded tuned HPs for {algorithm}: {hp}")
    return hp

def build_env_kwargs(args, env_config: dict) -> dict:
    """Assemble the keyword-argument dict for the chosen env class."""
    if args.robot_type == "uav":
        kwargs = dict(
            field_info=env_config,
            num_robots=args.num_robots,
            max_steps=args.max_steps,
            render_mode=None,
        )
    else:  # wheeled
        kwargs = dict(
            env_params=env_config,
            num_robots=args.num_robots,   # override JSON default; slices configs
            max_steps=args.max_steps,
            render_mode=None,
        )

    # Map the --ablation value to the correct env constructor kwarg
    kwarg_name = EXPERIMENT_MAP[args.experiment]
    if kwarg_name is not None:
        ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
        kwargs[kwarg_name] = ablation_val

    return kwargs


def build_log_dir(args) -> str:
    """Build a unique, stable log directory that encodes all run parameters."""
    if args.experiment == "main":
        hp_tag  = "tuned" if args.hyperparams_json else "default"
        version = f"main_{hp_tag}"
    elif args.experiment == "dr":
        version = f"dr_{args.ablation or 'none'}"
    else:
        version = f"{args.experiment}_{args.ablation or EXPERIMENT_DEFAULTS[args.experiment]}"

    tag = (f"{args.algorithm}_{args.robot_type}_N{args.num_robots}"
           f"_env{args.set}_seed{args.seed}")

    return os.path.join(args.log_root, version, tag)


# ================================================================
# Main training function
# ================================================================

def train(args):
    np.random.seed(args.seed)

    # Load environment configs
    if args.robot_type == "uav":
        json_dict  = read_uav_json(UAV_JSON_PATH)
        env_config = json_dict[f"set{args.set}"]
        env_class  = MultiUAV
        env_id     = "MultiUAV-v0"
    else:
        wheeled_dict = read_wheeled_json(WHEELED_JSON_PATH)
        env_config   = wheeled_dict[f"set{args.set}"]
        env_class    = MultiWheeled
        env_id       = "MultiWheeled-v0"
        # Validate: --num_robots must not exceed the available starting configs
        max_slots = len(env_config["ROBOT_INIT_CONFIGS"])
        if args.num_robots > max_slots:
            raise ValueError(
                f"--num_robots {args.num_robots} exceeds the {max_slots} "
                f"robot configs in wheeled set{args.set}."
            )

    env_kwargs = build_env_kwargs(args, env_config)

    # Gym registration
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=env_class,
                     max_episode_steps=args.max_steps)

    # Vectorised training env
    vec_env = make_vec_env(
        env_id,
        env_kwargs=env_kwargs,
        n_envs=args.num_envs,
        seed=args.seed,
    )

    # Vectorised evaluation env
    eval_vec = make_vec_env(
        env_id,
        env_kwargs=env_kwargs,
        n_envs=1,
        seed=args.seed + 1,
    )

    # Directories
    log_dir = build_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"  Algorithm   : {args.algorithm}")
    print(f"  Robot type  : {args.robot_type}")
    print(f"  Experiment  : {args.experiment}")
    print(f"  Ablation    : {args.ablation}")
    print(f"  Env set     : {args.set}  |  N robots: {args.num_robots}")
    print(f"  Seed        : {args.seed}")
    print(f"  Steps       : {args.steps:,}")
    print(f"  Log dir     : {log_dir}")
    print("=" * 60)

    # Callbacks
    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=max(args.log_steps // args.num_envs, 1),
        n_eval_episodes=args.n_eval_eps,
        deterministic=True,
        render=False,
    )
    log_cb   = LogEveryNTimesteps(n_steps=args.log_steps)
    callback = CallbackList([log_cb, eval_cb])

    # SB3 logger
    logger = configure(log_dir, ["stdout", "log", "csv", "tensorboard"])

    # Hyperparameters
    hp = load_hyperparams(args.hyperparams_json, args.algorithm)

    # Model
    AlgClass, policy = ALGORITHMS[args.algorithm]
    model = AlgClass(
        policy, vec_env,
        verbose=args.verbose,
        device=args.device,
        seed=args.seed,
        **hp,
    )
    model.set_logger(logger)

    # Training
    print(f"\nTraining {args.algorithm} ({args.robot_type}) ...")
    model.learn(total_timesteps=args.steps, callback=callback)

    # Save final model
    save_path = os.path.join(
        log_dir,
        f"{args.algorithm}_{args.robot_type}_N{args.num_robots}_env{args.set}",
    )
    model.save(save_path)
    print(f"Model saved -> {save_path}.zip")

    # Cleanup
    vec_env.close()
    eval_vec.close()
    del model


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)
