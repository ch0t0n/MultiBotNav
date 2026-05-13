# Learning to Navigate in a Stochastic and Uncertain Environment with Variable Motion Dynamics

This is the codebase for the paper *"Learning to Navigate in a Stochastic and
Uncertain Environment with Variable Motion Dynamics."*  We present a
reinforcement-learning solution for multi-robot navigation tasks under wind
disturbances, actuator noise, and observation noise.

The codebase is intentionally platform-agnostic: the same training pipeline
supports **two robot types** with very different motion dynamics, and every
experiment in the paper is run for **both** of them.

| Robot type  | Gym ID            | Class           | Dynamics                          | Configs                                  | Action     |
|------------:|:------------------|:----------------|:----------------------------------|:-----------------------------------------|:-----------|
| `uav`       | `MultiUAV-v0`     | `MultiUAV`      | Newtonian point-mass with wind     | `exp_sets/uav/cont_sets.json` (10 sets)  | (ax, ay)   |
| `wheeled`   | `MultiWheeled-v0` | `MultiWheeled`  | Bicycle kinematics with wind drift | `exp_sets/wheeled/env*.ini`   (8 sets)   | (accel, δ̇) |

Both environments support the same experimental knobs:

- **`reward_ablation`** ∈ `{full, no_term, no_path}`
- **`obs_mode`**        ∈ `{full, no_pos, no_inf_hist, pos_only}`
- **`uncertainty_mode`**∈ `{full, wind_only, act_only, deterministic}`
- **`dr_mode`**         ∈ `{none, wind, full}` (domain randomization)

A standalone reference implementation that bundles both environments, the
helpers, the gym registration, and the multiprocess training launcher is
available at `single_file/example_env_v1.py`.

---

## Repository Layout

```
learning_to_navigate/
├── src/
│   ├── env.py                  ← MultiUAV + MultiWheeled gym envs
│   ├── utils.py                ← seeding, geometry, config loaders
│   └── __init__.py             ← registers MultiUAV-v0 and MultiWheeled-v0
├── exp_sets/
│   ├── uav/cont_sets.json      ← 10 UAV field configurations
│   └── wheeled/env{1..8}.ini   ← 8 wheeled-robot configurations
├── train.py                    ← unified training (--robot_type uav|wheeled)
├── tune.py                     ← Optuna distributed tuning
├── evaluate.py                 ← post-training evaluation
├── sensitivity_hp.py           ← HP sensitivity sweep
├── analyze_results.py          ← aggregate NPZ + CSV into LaTeX-ready tables
├── plot_figures.py             ← all paper figures
├── sim2real.py                 ← CoppeliaSim observation-gap study (UAV only)
├── single_file/example_env_v1.py  ← all-in-one reference implementation
├── slurm/                      ← SLURM array scripts (one per step)
└── INSTRUCTIONS.MD             ← full reproduction recipe (local + HPC)
```

---

## Setup

It is recommended to run this codebase on Linux.  Install the dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for CrossQ and TQC, but not required.

---

## Quickstart

Train CrossQ for one UAV configuration with default hyperparameters:

```bash
python train.py \
    --algorithm   CrossQ \
    --robot_type  uav \
    --set         1 \
    --num_robots  3 \
    --seed        42 \
    --steps       2000000 \
    --experiment  main \
    --device      cpu
```

The trained model lands in:

```
logs/main_default/CrossQ_uav_N3_env1_seed42/
    best_model/best_model.zip
    eval_logs/evaluations.npz
```

Switching to the wheeled platform is a one-flag change (`--robot_type wheeled`);
the number of robots is read from the `.ini` config file in that case.

For the **full** reproduction pipeline — six algorithms, all env sets, three
seeds, both robot types, all ablation/DR conditions — follow
[`INSTRUCTIONS.MD`](INSTRUCTIONS.MD).  Locally this takes hundreds of
CPU-hours per algorithm; the included SLURM scripts parallelise it across
an HPC cluster.

---

## How the Single-File Reference (`example_env_v1.py`) Maps to This Repo

| `example_env_v1.py`        | This repository                                |
|----------------------------|------------------------------------------------|
| `set_seed`, helpers, ...   | `src/utils.py`                                 |
| `class MultiUAV`           | `src/env.py`                                   |
| `class MultiWheeled`       | `src/env.py`                                   |
| `read_uav_json`            | `src/utils.py`                                 |
| `read_wheeled_configs`     | `src/utils.py`                                 |
| Gym registration           | `src/__init__.py`                              |
| `train_single_env` worker  | `train.py` (with full CLI)                     |
| `main()` multi-seed loop   | `slurm/step1_*.sh` (job arrays) + `INSTRUCTIONS.MD` local loops |

The single-file version is the same logic flattened into one Python file so
you can read the entire pipeline top-to-bottom in one place — useful for
inspection or for running a quick smoke test on a desktop.

---

## Reproducing the Paper Tables and Figures

After training and (where required) evaluation completes, build the LaTeX
tables and figures with:

```bash
# Aggregate logs / CSVs -> table-ready summary files
python analyze_results.py --robot_type uav     --log_root logs --results_dir logs/results
python analyze_results.py --robot_type wheeled --log_root logs --results_dir logs/results

# Generate every figure for both robot types
python plot_figures.py --robot_type both \
    --log_root logs --results_dir logs/results --figures_dir figures
```

See [`INSTRUCTIONS.MD`](INSTRUCTIONS.MD) for the **exact submit order**, job
counts, and merge commands.

---

## Simulation (CoppeliaSim, UAV only)

Install CoppeliaSim from <https://coppeliarobotics.com/>, open the scene file
`simulation/sim_envs/coppeliasim_scene_for_spraying_v3.ttt`, then follow the
walkthrough in `simulation/new_env_sim_v3.ipynb`.  The Table 8 sim-to-real
observation-gap experiment is produced by `sim2real.py`.

> **IMPORTANT:** Reopen the CoppeliaSim scene before each run.  Never save
> changes to the scene file when closing.

---

## License

See [`LICENSE`](LICENSE).
