#!/usr/bin/env python3
"""
sensitivity_hp.py — Hyperparameter sensitivity analysis for Table 6.

For each algorithm, sweeps each of its tunable hyperparameters over a
7-point grid while holding all others fixed at their Optuna-tuned values
(from logs/best_hyperparams_<robot_type>.json).  Trains a short policy for
each grid point, evaluates it, and records the IQM.

Reports the coefficient of variation  CV = σ(IQM) / |μ(IQM)|  across the
7-point grid for each (algorithm, hyperparameter) pair.  Lower CV means
the algorithm is more robust to that hyperparameter's choice.

Outputs
-------
results_dir/cv_table_<robot_type>.csv              — machine-readable
results_dir/sensitivity_hp_latex_rows_<robot_type>.txt — ready-to-paste LaTeX rows

Usage
-----
  python sensitivity_hp.py --algorithm CrossQ --results_dir results
  python sensitivity_hp.py --write_latex_only --results_dir results
"""

import os
import csv
import json
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO, TQC, CrossQ, ARS

from src.env import MultiUAV, MultiWheeled
from src.utils import read_uav_json, read_wheeled_json, flock_exclusive, flock_unlock


# ================================================================
# Constants
# ================================================================

UAV_JSON_PATH     = os.path.join('exp_sets', 'uav', 'cont_sets.json')
WHEELED_JSON_PATH = os.path.join('exp_sets', 'wheeled', 'wheeled_configs.json')
ENV_VAR     = 1
NUM_ROBOTS  = 3
NUM_ENVS    = 4
MAX_STEPS   = 1000
SEED        = 42
TRAIN_STEPS = 200_000
N_EVAL_EPS  = 20
N_GRID      = 7

ALGORITHMS = {
    "A2C":    (A2C,    "MlpPolicy"),
    "ARS":    (ARS,    "LinearPolicy"),
    "PPO":    (PPO,    "MlpPolicy"),
    "TRPO":   (TRPO,   "MlpPolicy"),
    "CrossQ": (CrossQ, "MlpPolicy"),
    "TQC":    (TQC,    "MlpPolicy"),
}


def _log_grid(lo, hi, n=N_GRID):
    return np.logspace(np.log10(lo), np.log10(hi), n).tolist()

def _lin_grid(lo, hi, n=N_GRID):
    return np.linspace(lo, hi, n).tolist()

HP_GRIDS = {
    "A2C": [
        ("learning_rate", _log_grid(1e-4, 1e-2),    False),
        ("gae_lambda",    _lin_grid(0.90, 1.00),     False),
        ("vf_coef",       _lin_grid(0.20, 0.70),     False),
        ("ent_coef",      _lin_grid(0.00, 0.05),     False),
        ("max_grad_norm", _lin_grid(0.30, 0.99),     False),
    ],
    "ARS": [
        ("learning_rate", _log_grid(1e-4, 1e-2),    False),
        ("delta_std",     _lin_grid(0.01, 0.30),     False),
        ("n_delta",       [int(v) for v in _lin_grid(8, 64)],  True),
    ],
    "PPO": [
        ("learning_rate", _log_grid(1e-4, 1e-2),    False),
        ("gae_lambda",    _lin_grid(0.90, 1.00),     False),
        ("vf_coef",       _lin_grid(0.20, 0.70),     False),
        ("ent_coef",      _lin_grid(0.00, 0.05),     False),
        ("max_grad_norm", _lin_grid(0.30, 0.99),     False),
        ("clip_range",    _lin_grid(0.10, 0.40),     False),
        ("n_epochs",      [int(v) for v in _lin_grid(3, 20)],  True),
    ],
    "TRPO": [
        ("learning_rate", _log_grid(1e-4, 1e-2),    False),
        ("gae_lambda",    _lin_grid(0.90, 1.00),     False),
        ("target_kl",     _log_grid(1e-3, 5e-2),     False),
        ("cg_max_steps",  [int(v) for v in _lin_grid(5, 20)],  True),
    ],
    "CrossQ": [
        ("learning_rate", _log_grid(1e-4, 1e-2),    False),
        ("buffer_size",   [int(v) for v in _lin_grid(1_000, 50_000)], True),
        ("batch_size",    [256, 256, 256, 512, 512, 1024, 1024],       True),
    ],
    "TQC": [
        ("learning_rate", _log_grid(1e-4, 1e-2),    False),
        ("buffer_size",   [int(v) for v in _lin_grid(1_000, 50_000)], True),
        ("batch_size",    [256, 256, 256, 512, 512, 1024, 1024],       True),
        ("tau",           _log_grid(1e-3, 5e-2),     False),
        ("top_quantiles_to_drop_per_net",
                          [int(v) for v in _lin_grid(0, 5)],           True),
    ],
}

# ================================================================
# Helpers
# ================================================================

def compute_iqm(rewards: np.ndarray) -> float:
    q25, q75 = np.percentile(rewards, [25, 75])
    mask = (rewards >= q25) & (rewards <= q75)
    return float(np.mean(rewards[mask])) if mask.any() else float(np.mean(rewards))


def compute_cv(iqm_values: list) -> float:
    arr = np.array(iqm_values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    mu = np.mean(arr)
    if abs(mu) < 1e-9:
        return float("nan")
    return float(np.std(arr) / abs(mu))


def load_tuned_hp(json_path: str, algorithm: str) -> dict:
    if not os.path.exists(json_path):
        print(f"  [WARN] {json_path} not found — using SB3 defaults as base.")
        return {}
    with open(json_path) as f:
        data = json.load(f)
    hp = data.get(algorithm, {}).get("params", {})
    print(f"  Loaded tuned base HPs for {algorithm}: {hp}")
    return hp


def run_one_trial(AlgClass, policy, env_id, env_kwargs,
                  hp_params: dict, device: str) -> float:
    """Train for TRAIN_STEPS and return IQM over N_EVAL_EPS episodes."""
    vec_env  = make_vec_env(env_id, env_kwargs=env_kwargs,
                            n_envs=NUM_ENVS, seed=SEED)
    eval_env = make_vec_env(env_id, env_kwargs=env_kwargs,
                            n_envs=1, seed=SEED + 1)
    model = None
    try:
        model = AlgClass(policy, vec_env,
                         verbose=0, device=device, seed=SEED,
                         **hp_params)
        model.learn(total_timesteps=TRAIN_STEPS)
        ep_r, _ = evaluate_policy(model, eval_env,
                                  n_eval_episodes=N_EVAL_EPS,
                                  deterministic=True,
                                  return_episode_rewards=True)
        return compute_iqm(np.array(ep_r, dtype=np.float32))
    except Exception as e:
        print(f"    [WARN] trial failed: {e}")
        return float("nan")
    finally:
        vec_env.close()
        eval_env.close()
        if model is not None:
            del model

# ================================================================
# Sweep one algorithm
# ================================================================

def sweep_algorithm(args, algorithm: str,
                    env_id: str, env_kwargs: dict,
                    raw_path: str) -> list:
    AlgClass, policy = ALGORITHMS[algorithm]
    base_hp = load_tuned_hp(args.hyperparams_json, algorithm)
    grids   = HP_GRIDS[algorithm]
    results = []

    for hp_name, grid_vals, is_int in grids:
        print(f"\n  [{algorithm}] sweeping {hp_name} over {N_GRID} points ...")
        iqm_list = []

        for i, val in enumerate(grid_vals):
            hp = dict(base_hp)
            hp[hp_name] = int(val) if is_int else float(val)

            iqm = run_one_trial(AlgClass, policy, env_id, env_kwargs, hp, args.device)
            print(f"    grid[{i}] {hp_name}={val:.6g}  IQM={iqm:.3f}")

            append_raw_csv(algorithm, hp_name, i, val, iqm, raw_path)
            iqm_list.append(iqm)

        cv = compute_cv(iqm_list)
        print(f"  -> CV({hp_name}) = {cv:.4f}")
        results.append({
            "algorithm":   algorithm,
            "hp_name":     hp_name,
            "grid_values": [round(v, 8) for v in grid_vals],
            "iqm_values":  [round(v, 4) for v in iqm_list],
            "cv":          round(cv, 4),
        })

    return results

# ================================================================
# CSV append (file-locked on Unix/HPC when fcntl is available)
# ================================================================

def append_cv_csv(results: list, csv_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    # Read existing rows so we can skip duplicates (algorithm, hp_name) pairs.
    existing_keys: set = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as rf:
                for row in csv.DictReader(rf):
                    existing_keys.add((row.get("algorithm", ""), row.get("hp_name", "")))
        except Exception:
            pass

    new_results = [r for r in results
                   if (r["algorithm"], r["hp_name"]) not in existing_keys]

    if not new_results:
        print(f"\n  All {len(results)} rows already present in {csv_path} — skipping.")
        return

    with open(csv_path, "a", newline="") as f:
        flock_exclusive(f)
        try:
            write_header = os.fstat(f.fileno()).st_size == 0
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "algorithm", "hp_name", "cv",
                    "iqm_mean", "iqm_std",
                    "grid_values", "iqm_values",
                ])
            for r in new_results:
                iqm_arr = [v for v in r["iqm_values"] if not np.isnan(v)]
                writer.writerow([
                    r["algorithm"],
                    r["hp_name"],
                    f"{r['cv']:.4f}",
                    f"{np.mean(iqm_arr):.4f}" if iqm_arr else "nan",
                    f"{np.std(iqm_arr):.4f}"  if iqm_arr else "nan",
                    ";".join(str(v) for v in r["grid_values"]),
                    ";".join(str(v) for v in r["iqm_values"]),
                ])
                f.flush()
                os.fsync(f.fileno())
        finally:
            flock_unlock(f)
    skipped = len(results) - len(new_results)
    print(f"\n  Appended {len(new_results)} new rows "
          f"({skipped} already present) -> {csv_path}")


def append_raw_csv(algorithm: str, hp_name: str,
                   grid_index: int, hp_value, iqm: float,
                   raw_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
    row = [algorithm, hp_name, grid_index, f"{hp_value:.8g}", f"{iqm:.4f}"]

    with open(raw_path, "a", newline="") as f:
        flock_exclusive(f)
        try:
            write_header = os.fstat(f.fileno()).st_size == 0
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["algorithm", "hp_name", "grid_index", "hp_value", "iqm"])
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            flock_unlock(f)

# ================================================================
# Expand raw CSV from an existing cv_table.csv
# ================================================================

def expand_raw_from_cv_table(cv_path: str, raw_path: str):
    import csv as csv_mod
    if not os.path.exists(cv_path):
        print(f"  [WARN] {cv_path} not found — nothing to expand.")
        return
    rows = []
    with open(cv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for rec in reader:
            grid_vals = rec["grid_values"].split(";")
            iqm_vals  = rec["iqm_values"].split(";")
            for i, (gv, iv) in enumerate(zip(grid_vals, iqm_vals)):
                rows.append({
                    "algorithm":  rec["algorithm"],
                    "hp_name":    rec["hp_name"],
                    "grid_index": i,
                    "hp_value":   gv.strip(),
                    "iqm":        iv.strip(),
                })
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "hp_name", "grid_index", "hp_value", "iqm"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Expanded {len(rows)} grid-point rows -> {raw_path}")

# ================================================================
# LaTeX table writer
# ================================================================

_HP_LATEX_LABELS = {
    "learning_rate":                    r"CV($\alpha$)",
    "gae_lambda":                       r"CV($\lambda_\mathrm{GAE}$)",
    "vf_coef":                          r"CV($c_\mathrm{v}$)",
    "ent_coef":                         r"CV($c_\mathrm{e}$)",
    "max_grad_norm":                    r"CV(clip$_\nabla$)",
    "clip_range":                       r"CV($\epsilon$)",
    "n_epochs":                         r"CV($K$)",
    "target_kl":                        r"CV($\delta_\mathrm{KL}$)",
    "cg_max_steps":                     r"CV($n_\mathrm{CG}$)",
    "delta_std":                        r"CV($\sigma_\delta$)",
    "n_delta":                          r"CV($n_\delta$)",
    "buffer_size":                      r"CV($|\mathcal{B}|$)",
    "batch_size":                       r"CV($B$)",
    "tau":                              r"CV($\tau$)",
    "top_quantiles_to_drop_per_net":    r"CV($q_\mathrm{drop}$)",
}


def write_latex(csv_path: str, out_path: str):
    import csv as csv_mod
    if not os.path.exists(csv_path):
        print(f"  [WARN] {csv_path} not found — cannot write LaTeX.")
        return

    data = {}
    hp_order_seen: list = []
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            alg = row["algorithm"]
            hp  = row["hp_name"]
            cv  = float(row["cv"]) if row["cv"] != "nan" else float("nan")
            data.setdefault(alg, {})[hp] = cv
            if hp not in hp_order_seen:
                hp_order_seen.append(hp)

    alg_order = ["A2C", "ARS", "PPO", "TRPO", "CrossQ", "TQC"]

    # Per-column minimum (best = lowest CV) for bolding
    col_mins: dict = {}
    for hp_name in hp_order_seen:
        vals   = {alg: data.get(alg, {}).get(hp_name, float("nan")) for alg in alg_order}
        finite = {a: v for a, v in vals.items() if not np.isnan(v)}
        col_mins[hp_name] = min(finite, key=lambda a: finite[a]) if finite else None

    col_headers = " & ".join(
        _HP_LATEX_LABELS.get(hp, f"CV({hp})") for hp in hp_order_seen)

    lines = [
        "% LaTeX table rows for tab:sensitivity_hp",
        "% Generated by sensitivity_hp.py",
        f"% Columns: Algorithm | {col_headers}",
        "",
    ]

    for alg in alg_order:
        alg_data = data.get(alg, {})
        cells = []
        for hp_name in hp_order_seen:
            cv = alg_data.get(hp_name, float("nan"))
            if np.isnan(cv):
                cells.append("---")
            else:
                s = f"${cv:.3f}$"
                if col_mins.get(hp_name) == alg:
                    s = rf"$\mathbf{{{cv:.3f}}}^\dagger$"
                cells.append(s)
        lines.append(f"{alg} & " + " & ".join(cells) + r" \\")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote LaTeX rows -> {out_path}")
    print("\n" + "\n".join(lines))

# ================================================================
# Argument parsing
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sensitivity sweep")
    p.add_argument("--algorithm",        type=str, default=None,
                   choices=list(ALGORITHMS.keys()))
    p.add_argument("--robot_type",       type=str, default="uav",
                   choices=["uav", "wheeled"],
                   help="Robot platform: 'uav' (MultiUAV) or 'wheeled' (MultiWheeled)")
    p.add_argument("--results_dir",      type=str, default="results")
    p.add_argument("--hyperparams_json", type=str,
                   default=None,
                   help="Path to best_hyperparams_<robot>.json. "
                        "Defaults to logs/best_hyperparams_<robot_type>.json")
    p.add_argument("--device",           type=str, default="cpu")
    p.add_argument("--train_steps",      type=int, default=TRAIN_STEPS)
    p.add_argument("--n_eval_eps",       type=int, default=N_EVAL_EPS)
    p.add_argument("--write_latex_only", action="store_true")
    p.add_argument("--write_raw_only",   action="store_true")
    return p.parse_args()

# ================================================================
# Entry point
# ================================================================

def main():
    args = parse_args()

    # Default hyperparams JSON depends on robot type
    if args.hyperparams_json is None:
        args.hyperparams_json = os.path.join(
            "logs", f"best_hyperparams_{args.robot_type}.json")

    # Outputs are tagged by robot type so UAV and wheeled don't overwrite each other
    csv_path   = os.path.join(args.results_dir, f"cv_table_{args.robot_type}.csv")
    raw_path   = os.path.join(args.results_dir, f"sensitivity_hp_raw_{args.robot_type}.csv")
    latex_path = os.path.join(args.results_dir, f"sensitivity_hp_latex_rows_{args.robot_type}.txt")

    if args.write_latex_only:
        print(f"Writing LaTeX from existing {csv_path} ...")
        write_latex(csv_path, latex_path)
        return

    if args.write_raw_only:
        print(f"Regenerating {raw_path} from {csv_path} ...")
        expand_raw_from_cv_table(csv_path, raw_path)
        return

    if args.algorithm is None:
        print("ERROR: --algorithm is required unless --write_latex_only "
              "or --write_raw_only is set.")
        raise SystemExit(1)

    global TRAIN_STEPS, N_EVAL_EPS
    TRAIN_STEPS = args.train_steps
    N_EVAL_EPS  = args.n_eval_eps

    # Environment setup (UAV or wheeled)
    if args.robot_type == "uav":
        json_dict  = read_uav_json(UAV_JSON_PATH)
        env_config = json_dict[f"set{ENV_VAR}"]
        env_kwargs = dict(
            field_info=env_config,
            num_robots=NUM_ROBOTS,
            max_steps=MAX_STEPS,
            render_mode=None,
        )
        env_id    = "MultiUAV-v0"
        env_class = MultiUAV
    else:
        wheeled_dict = read_wheeled_json(WHEELED_JSON_PATH)
        env_config   = wheeled_dict[f"set{ENV_VAR}"]
        env_kwargs   = dict(
            env_params=env_config,
            num_robots=NUM_ROBOTS,
            max_steps=MAX_STEPS,
            render_mode=None,
        )
        env_id    = "MultiWheeled-v0"
        env_class = MultiWheeled


    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=env_class, max_episode_steps=MAX_STEPS)

    print(f"\n{'='*60}")
    print(f"  Sensitivity sweep: {args.algorithm}  ({args.robot_type})")
    print(f"  Train steps / point: {TRAIN_STEPS:,}")
    print(f"  Eval episodes / point: {N_EVAL_EPS}")
    print(f"  Grid points per HP: {N_GRID}")
    print(f"{'='*60}")

    results = sweep_algorithm(args, args.algorithm, env_id, env_kwargs, raw_path)
    append_cv_csv(results, csv_path)
    write_latex(csv_path, latex_path)

    print(f"\nsensitivity_hp.py complete for {args.algorithm} ({args.robot_type}).")
    print(f"  Raw grid data  -> {raw_path}")
    print(f"  CV table       -> {csv_path}")
    print(f"  LaTeX rows     -> {latex_path}")


if __name__ == "__main__":
    main()
# end of file
