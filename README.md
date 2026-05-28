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
| `wheeled`   | `MultiWheeled-v0` | `MultiWheeled`  | Bicycle kinematics with wind drift | `exp_sets/wheeled/wheeled_configs.json` (10 sets)  | (accel, δ̇) |

Both environments support the same experimental knobs:

- **`reward_ablation`** ∈ `{full, no_term, no_path}`
- **`obs_mode`**        ∈ `{full, no_pos, no_vis_hist, pos_only}`
- **`uncertainty_mode`**∈ `{full, wind_only, act_only, deterministic}`
- **`dr_mode`**         ∈ `{none, wind, full}` (domain randomization)

A standalone reference implementation that bundles both environments, the
helpers, the gym registration, and the multiprocess training launcher is
available in `single_file_implementation/`.

---

## Repository Layout

```
MultiBotNav/
├── src/
│   ├── env.py                  ← MultiUAV + MultiWheeled gym envs
│   ├── utils.py                ← seeding, geometry, config loaders
│   └── __init__.py             ← registers MultiUAV-v0 and MultiWheeled-v0
├── exp_sets/
│   ├── uav/cont_sets.json      ← 10 UAV field configurations
│   └── wheeled/wheeled_configs.json  ← 10 wheeled-robot configurations
├── train.py                    ← unified training (--robot_type uav|wheeled)
├── tune.py                     ← Optuna distributed tuning
├── evaluate.py                 ← post-training evaluation
├── sensitivity_hp.py           ← HP sensitivity sweep
├── analyze_results.py          ← aggregate NPZ + CSV into LaTeX-ready tables
├── plot_figures.py             ← all paper figures
├── sim2real.py                 ← CoppeliaSim observation-gap study (UAV only)
├── single_file_implementation/    ← all-in-one reference implementations
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

> **Wheeled note:** The wheeled `main` experiment uses a two-stage curriculum.
> Artifacts land in `best_model_stage2/best_model.zip` and
> `eval_logs_stage2/evaluations.npz` rather than the standard paths above.
> `evaluate.py` and `analyze_results.py` detect and prefer the stage-2 paths
> automatically.

Switching to the wheeled platform is a one-flag change (`--robot_type wheeled`).

For the **full** reproduction pipeline — six algorithms, all env sets, three
seeds, both robot types, all ablation/DR conditions — follow
[`INSTRUCTIONS.MD`](INSTRUCTIONS.MD).  Locally this takes hundreds of
CPU-hours per algorithm; the included SLURM scripts parallelise it across
an HPC cluster.

---

## How the Single-File References Map to This Repo

| Single-file symbol          | This repository                                |
|-----------------------------|------------------------------------------------|
| `set_seed`, helpers, ...    | `src/utils.py`                                 |
| `class MultiUAV`            | `src/env.py`                                   |
| `class MultiWheeled`        | `src/env.py`                                   |
| `read_uav_json`             | `src/utils.py`                                 |
| `read_wheeled_json`         | `src/utils.py`                                 |
| Gym registration            | `src/__init__.py`                              |
| `train_single_env` worker   | `train.py` (with full CLI)                     |
| `main()` multi-seed loop    | `slurm/step1_*.sh` (job arrays) + `INSTRUCTIONS.MD` local loops |

The single-file versions contain the logic flattened into one Python file so
you can read the entire pipeline top-to-bottom in one place — useful for
inspection or for running a quick smoke test on a desktop.

> **Reference for the wheeled robot:** `single_file_implementation/v2_wheeled.py`
> reflects the production wheeled environment (modern reward shaping, two-stage
> curriculum) and is a closer reference than `example_training.py`.
> `example_training.py` covers the UAV environment and a simplified wheeled
> variant for initial exploration.

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

Install CoppeliaSim from <https://coppeliarobotics.com/>.  The scene file
(`simulation/sim_envs/coppeliasim_scene_for_spraying_v3.ttt`) and the
companion notebook (`simulation/new_env_sim_v3.ipynb`) are **not included
in this repository** — they are available as a separate download from the
paper's supplementary materials.  The Table 4 (`tab:obs_gap`) sim-to-real
observation-gap experiment is produced by `sim2real.py` (runs without
CoppeliaSim when `RENDER_COPPELIA = False`).

> **IMPORTANT:** Reopen the CoppeliaSim scene before each run.  Never save
> changes to the scene file when closing.

---

## License

A LICENSE file will be added before public release.
