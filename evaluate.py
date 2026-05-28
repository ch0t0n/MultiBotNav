#!/usr/bin/env python3
"""
evaluate.py — post-training evaluation for ALL experiment types.

Loads the best saved model for a given (algorithm, robot_type, set,
num_robots, seed, experiment, ablation) combination, runs N_EVAL_EPISODES
episodes, and appends one row to a results CSV.

Terminal conditions tracked: visited_all, collision, max_steps.

Usage examples:
  # UAV main results:
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \
                     --experiment main --hp_tag default \
                     --log_root logs --output_csv logs/results/results_default_uav.csv

  # Wheeled robot:
  python evaluate.py --algorithm CrossQ --set 1 --robot_type wheeled --seed 42 \
                     --experiment main \
                     --output_csv logs/results/results_default_wheeled.csv

  # Reward ablation:
  python evaluate.py --algorithm CrossQ --set 1 --num_robots 3 --seed 42 \
                     --experiment ablation_reward --ablation no_path \
                     --output_csv logs/results/ablation_reward_uav.csv
"""

import os
import csv
import glob
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiUAV, MultiWheeled
from src.utils import read_uav_json, read_wheeled_json, flock_exclusive, flock_unlock

# ================================================================
# Constants
# ================================================================

UAV_JSON_PATH     = os.path.join('exp_sets', 'uav', 'cont_sets.json')
WHEELED_JSON_PATH = os.path.join('exp_sets', 'wheeled', 'wheeled_configs.json')
N_EVAL_EPISODES    = 50
MAX_STEPS          = 1000

ALGORITHMS = {
    "A2C":    A2C,
    "ARS":    ARS,
    "PPO":    PPO,
    "TRPO":   TRPO,
    "CrossQ": CrossQ,
    "TQC":    TQC,
}

EXPERIMENT_MAP = {
    "main":                 None,
    "ablation_reward":      "reward_ablation",
    "ablation_obs":         "obs_mode",
    "ablation_uncertainty": "uncertainty_mode",
    "dr":                   "dr_mode",
}

EXPERIMENT_DEFAULTS = {
    "main":                 None,
    "ablation_reward":      "full",
    "ablation_obs":         "full",
    "ablation_uncertainty": "full",
    "dr":                   "none",
}

# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Post-training evaluation")
    p.add_argument("--algorithm",    type=str, required=True, choices=list(ALGORITHMS.keys()))
    p.add_argument("--set",          type=int, required=True)
    p.add_argument("--robot_type",   type=str, default="uav", choices=["uav", "wheeled"])
    p.add_argument("--num_robots",   type=int, default=3,
                   help="Number of robots (2-5 for both UAV and wheeled)")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--experiment",   type=str, default="main", choices=list(EXPERIMENT_MAP.keys()))
    p.add_argument("--ablation",     type=str, default=None)
    p.add_argument("--hp_tag",       type=str, default="default",
                   help="For experiment=main: which HP set was used")
    p.add_argument("--log_root",     type=str, default="logs")
    p.add_argument("--output_csv",   type=str, required=True)
    p.add_argument("--n_eval_eps",   type=int, default=N_EVAL_EPISODES)
    p.add_argument("--device",       type=str, default="cpu")
    p.add_argument("--eval_wind_min",  type=float, default=None)
    p.add_argument("--eval_wind_max",  type=float, default=None)
    p.add_argument("--eval_uncertainty_mode", type=str, default=None)
    return p.parse_args()

# ================================================================
# Helpers
# ================================================================

def compute_iqm(rewards: np.ndarray) -> float:
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))

def _csv_str(v) -> str:
    return "" if v is None else str(v)

def already_evaluated(output_csv: str, args, ablation_val: str) -> bool:
    if not os.path.exists(output_csv):
        return False
    try:
        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("algorithm")             == args.algorithm
                        and row.get("robot_type")    == args.robot_type
                        and row.get("experiment")    == args.experiment
                        and row.get("ablation")      == ablation_val
                        and row.get("hp_tag")        == args.hp_tag
                        and row.get("num_robots")    == str(args.num_robots)
                        and row.get("env_set")       == str(args.set)
                        and row.get("seed")          == str(args.seed)
                        and row.get("eval_wind_min")         == _csv_str(args.eval_wind_min)
                        and row.get("eval_wind_max")         == _csv_str(args.eval_wind_max)
                        and row.get("eval_uncertainty_mode") == _csv_str(args.eval_uncertainty_mode)):
                    return True
    except Exception:
        pass
    return False

def find_model_path(log_root: str, algorithm: str, robot_type: str,
                    num_robots: int, env_set: int, experiment: str,
                    hp_tag: str, ablation: str, seed: int) -> str:
    if experiment == "main":
        version = f"main_{hp_tag}"
    elif experiment == "dr":
        version = f"dr_{ablation or 'none'}"
    else:
        version = f"{experiment}_{ablation or EXPERIMENT_DEFAULTS[experiment]}"

    tag = f"{algorithm}_{robot_type}_N{num_robots}_env{env_set}_seed{seed}"

    # Wheeled main runs use a two-stage curriculum; the best model ends up in
    # best_model_stage2/.  Fall back to best_model/ for all other cases.
    if robot_type == "wheeled" and experiment == "main":
        stage2_pattern = os.path.join(
            log_root, version, tag, "best_model_stage2", "best_model.zip")
        matches = sorted(glob.glob(stage2_pattern))
        if matches:
            path = matches[0]
            print(f"  Using model (stage 2): {path}")
            return os.path.splitext(path)[0]

    pattern = os.path.join(log_root, version, tag, "best_model", "best_model.zip")
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No model found for pattern:\n  {pattern}\n"
            "Have you completed training?")

    path = matches[0]
    print(f"  Using model: {path}")
    return os.path.splitext(path)[0]

def build_eval_env_kwargs(args, env_config: dict,
                          wind_speed: float = None, wind_dir: float = None) -> dict:
    if args.robot_type == "uav":
        kwargs = dict(
            field_info=env_config,
            num_robots=args.num_robots,
            max_steps=MAX_STEPS,
            render_mode=None,
        )
    else:
        kwargs = dict(
            env_params=env_config,
            num_robots=args.num_robots,   # override JSON default; slices configs
            max_steps=MAX_STEPS,
            render_mode=None,
        )

    kwarg_name = EXPERIMENT_MAP[args.experiment]
    if kwarg_name is not None:
        ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]
        if args.experiment == "dr" and wind_speed is not None:
            # When an explicit eval-wind range is provided, override dr_mode to
            # "none" so the env uses the fixed wind_par instead of resampling.
            # The model loaded is still the one trained with DR (ablation_val).
            kwargs[kwarg_name] = "none"
        else:
            kwargs[kwarg_name] = ablation_val

    if args.eval_uncertainty_mode is not None:
        kwargs["uncertainty_mode"] = args.eval_uncertainty_mode

    if wind_speed is not None:
        direction = wind_dir if wind_dir is not None else 0.0
        kwargs["wind_par"] = [wind_speed, direction]

    return kwargs

# ================================================================
# Main evaluation
# ================================================================

def evaluate(args):
    ablation_val = args.ablation or EXPERIMENT_DEFAULTS[args.experiment]

    if already_evaluated(args.output_csv, args, ablation_val):
        print(
            f"  [SKIP] Result already present in {args.output_csv} for\n"
            f"         {args.algorithm} | {args.robot_type} | {args.experiment} | "
            f"{ablation_val} | N={args.num_robots} | set={args.set} | seed={args.seed}\n"
            f"         Skipping re-evaluation."
        )
        return

    # Load env config
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
        # Validate: --num_robots must not exceed available starting configs
        max_slots = len(env_config["ROBOT_INIT_CONFIGS"])
        if args.num_robots > max_slots:
            raise ValueError(
                f"--num_robots {args.num_robots} exceeds the {max_slots} "
                f"robot configs in wheeled set{args.set}."
            )

    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=env_class, max_episode_steps=MAX_STEPS)

    model_path = find_model_path(
        args.log_root, args.algorithm, args.robot_type, args.num_robots,
        args.set, args.experiment, args.hp_tag, args.ablation, args.seed)
    AlgClass = ALGORITHMS[args.algorithm]
    model    = AlgClass.load(model_path, device=args.device)

    print(f"\n{'='*60}")
    print(f"  Evaluating  : {args.algorithm} | {args.robot_type} | "
          f"{args.experiment} | {ablation_val}")
    print(f"  Env set     : {args.set}  |  N robots: {args.num_robots}  |  Seed: {args.seed}")
    print(f"  Episodes    : {args.n_eval_eps}")
    print(f"{'='*60}")

    all_rewards    = []
    all_ep_lengths = []
    all_term_conds = []
    start = time.perf_counter()

    for ep in range(args.n_eval_eps):
        if args.eval_wind_min is not None and args.eval_wind_max is not None:
            ep_rng = np.random.default_rng(args.seed + ep)
            w     = ep_rng.uniform(args.eval_wind_min, args.eval_wind_max)
            w_dir = ep_rng.uniform(0, 360)
        else:
            w, w_dir = None, None

        env_kwargs = build_eval_env_kwargs(args, env_config, wind_speed=w, wind_dir=w_dir)
        eval_env   = make_vec_env(env_id, env_kwargs=env_kwargs, n_envs=1,
                                  seed=args.seed + ep + 1000)

        obs = eval_env.reset()
        ep_reward = 0.0
        ep_length = 0
        ep_term   = "max_steps"
        dones     = [False]

        while not dones[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_env.step(action)
            ep_reward += rewards[0]
            ep_length += 1
            if dones[0]:
                ep_term = infos[0].get("term_cond", "max_steps")

        all_rewards.append(ep_reward)
        all_ep_lengths.append(ep_length)
        all_term_conds.append(ep_term)
        eval_env.close()

    elapsed = time.perf_counter() - start

    rewards_arr = np.array(all_rewards,    dtype=np.float32)
    ep_lens_arr = np.array(all_ep_lengths, dtype=np.float32)
    mean_r      = float(np.mean(rewards_arr))
    std_r       = float(np.std(rewards_arr))
    max_r       = float(np.max(rewards_arr))
    iqm_r       = compute_iqm(rewards_arr)
    cutoff      = int(np.ceil(0.1 * len(rewards_arr)))
    cvar        = float(np.mean(np.sort(rewards_arr)[:cutoff]))
    mean_ep_len = float(np.mean(ep_lens_arr))
    iqm_ep_len  = compute_iqm(ep_lens_arr)

    # Terminal condition distributions
    # UAV success → "visited_all";  Wheeled success → "all_goals"
    success_pct   = (all_term_conds.count("visited_all")
                     + all_term_conds.count("all_goals")) / args.n_eval_eps * 100
    collision_pct = all_term_conds.count("collision")    / args.n_eval_eps * 100
    max_steps_pct = all_term_conds.count("max_steps")    / args.n_eval_eps * 100

    print(f"\n  mean={mean_r:.3f}  std={std_r:.3f}  max={max_r:.3f}  "
          f"IQM={iqm_r:.3f}  CVaR_0.1={cvar:.3f}")
    print(f"  mean_ep_len={mean_ep_len:.1f}  iqm_ep_len={iqm_ep_len:.1f}  elapsed={elapsed:.1f}s")
    print(f"  Terminal: Success ({success_pct:.1f}%), "
          f"Collision ({collision_pct:.1f}%), Max Steps ({max_steps_pct:.1f}%)")

    # Append to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "a", newline="") as f:
        flock_exclusive(f)
        try:
            write_header = os.fstat(f.fileno()).st_size == 0
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "algorithm", "robot_type", "experiment", "ablation", "hp_tag",
                    "num_robots", "env_set", "seed",
                    "eval_wind_min", "eval_wind_max", "eval_uncertainty_mode",
                    "mean_reward", "std_reward", "max_reward", "iqm",
                    "cvar_0.1", "mean_ep_length", "iqm_ep_length",
                    "success_pct", "collision_pct", "max_steps_pct",
                    "n_episodes", "elapsed_s",
                ])
            writer.writerow([
                args.algorithm, args.robot_type, args.experiment, ablation_val, args.hp_tag,
                args.num_robots, args.set, args.seed,
                args.eval_wind_min, args.eval_wind_max, args.eval_uncertainty_mode,
                f"{mean_r:.4f}", f"{std_r:.4f}", f"{max_r:.4f}", f"{iqm_r:.4f}",
                f"{cvar:.4f}", f"{mean_ep_len:.2f}", f"{iqm_ep_len:.2f}",
                f"{success_pct:.2f}", f"{collision_pct:.2f}", f"{max_steps_pct:.2f}",
                args.n_eval_eps, f"{elapsed:.1f}",
            ])
        finally:
            flock_unlock(f)
    print(f"  Appended to {args.output_csv}")

if __name__ == "__main__":
    evaluate(parse_args())
