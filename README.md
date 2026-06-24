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
├── sim2real.py                 ← observation-gap sim-to-real study (UAV + wheeled)
├── simulation/
│   ├── wheeled/                ← Ursina 3D demo (trained CrossQ policies)
│   │   ├── run_simulation.py
│   │   └── core/               ← 3D renderer (physics from src/env.py)
│   └── uav/                    ← CoppeliaSim inference notebook
│       ├── uav_sim_new.ipynb
│       └── coppeliasim_envs/uav_common_env.ttt
├── trained_models/             ← pre-trained wheeled checkpoints (see Simulation)
│   └── wheeled/
│       └── best_model_env{N}_stage2_robust_wind/best_model.zip
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

> **Reference for the wheeled robot:** `single_file_implementation/example_training.py`
> includes both UAV and wheeled environments.  The production wheeled
> environment in `src/env.py` adds two-stage curriculum support and matches
> the 3D Ursina simulator in `simulation/wheeled/`.

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

## Simulation

The `simulation/` folder contains visual demos for both robot platforms.  Each
simulator reuses the same environment configs and trained checkpoints as the
training pipeline.

### Wheeled robots (Ursina 3D)

`simulation/wheeled/` provides a standalone **Ursina** 3D visualizer for the
wheeled robot environment.  Physics and observations match `src/env.py`
(`MultiWheeled`); only the renderer differs.  It loads trained **CrossQ**
checkpoints and runs the policy in real time — no CoppeliaSim required.
Ursina is included in the root `requirements.txt`.

Optional: fetch CC0 tree/robot models if assets are missing (`cd simulation/wheeled` first):

```bash
python download_assets.py
```

**Trained models** are expected under:

```
trained_models/wheeled/best_model_env{N}_stage2_robust_wind/best_model.zip
```

where `{N}` is the environment index (`1` … `10`).  These correspond to the
stage-2 checkpoints produced by the wheeled two-stage curriculum (see the
wheeled note in Quickstart).  You can also point the script at your own
training output, e.g. `logs/.../best_model_stage2/best_model.zip`.

**Run with a trained policy** (random-action baseline uses `--random-policy`):

```bash
cd simulation/wheeled
python run_simulation.py --env-key env6
python run_simulation.py --env-key env10 --random-policy
```

With an explicit checkpoint:

```bash
python run_simulation.py --env-key env10 \
    --weights "../../trained_models/wheeled/best_model_env10_stage2_robust_wind/best_model.zip"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--env-key` | `env10` | Map to load (`env1` … `env10`) |
| `--num-robots` | inferred | Robot count (`2`–`5`); inferred from checkpoint when omitted |
| `--weights` | `trained_models/wheeled/best_model_env{N}_...` | Path to CrossQ `.zip` |
| `--random-policy` | off | Sample random actions instead of a checkpoint |
| `--max-steps` | `1000` | Episode length cap |
| `--robot-type` | from config | `tractor`, `delivery`, `tractor_shovel`, or `rover` |
| `--fps` | `30` | Render frame rate |

The simulator matches stage-2 training settings (`uncertainty_mode="wind_only"`,
`dr_mode="wind"`), applies the same 30% obstacle shrink for `env2`–`env9` as
`src/utils.py`, and auto-resets when an episode terminates.  See
[`simulation/wheeled/README.md`](simulation/wheeled/README.md) for coordinate
mapping and asset details.

### CoppeliaSim (UAV)

Install CoppeliaSim from <https://coppeliarobotics.com/>.  Open the scene file
`simulation/uav/coppeliasim_envs/uav_common_env.ttt` in the simulator, then
follow the instructions in `simulation/uav/uav_sim_new.ipynb`.

The Table 4 (`tab:obs_gap`) sim-to-real observation-gap experiment is produced
by `sim2real.py` for **both** UAV and wheeled robots:

```bash
python sim2real.py                    # both platforms (default)
python sim2real.py --robot_type uav   # UAV only
python sim2real.py --robot_type wheeled
```

Results land in `sim2real_uav.out` and `sim2real_wheeled.out`.  Batch runs need
no visual simulator; optional UAV CoppeliaSim rendering:
`python sim2real.py --robot_type uav --render_coppelia`.

> **IMPORTANT:** Reopen the CoppeliaSim scene before each visual UAV run.
> Never save changes to the scene file when closing.

---

## License

A LICENSE file will be added before public release.
