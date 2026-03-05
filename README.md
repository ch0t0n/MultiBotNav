# Efficient Environment Design for Multi-Robot Navigation via Continuous Control

<p align="center">
  <img src="./assets/images/two_robots_env1.gif" width="400" height="250">
  <img src="./assets/images/three_robot_env3.gif" width="400" height="250">
</p>

This repository contains the official implementation for the paper:

**Efficient Environment Design for Multi-Robot Navigation via Continuous Control**

We develop a scalable MDP-based simulation framework for multi-robot navigation and path planning under continuous control. The framework supports:

- Multi-UAV systems  
- Multi-wheeled robot systems  
- Parallel training with vectorized environments  
- Multiple deep reinforcement learning algorithms  

---

# Setup

We recommend running this project on Linux.

All required dependencies are provided in `environment.yaml`.

## Create the conda environment

```bash
conda env create -f environment.yaml
conda activate <environment_name>
```

If you modify the environment:

```bash
conda env export --no-builds > environment.yaml
```

---

# Experiment Sets

Experiment configurations are stored in:

```
exp_sets/
    ├── uav/
    │     └── icra_2026_cont_sets.json
    └── wheeled/
          └── envX.ini
```

- UAV environments are loaded from JSON files.
- Wheeled robot environments are loaded from `.ini` files.

---

# Training

Training supports both UAV and wheeled robot environments.

Supported algorithms:

- `A2C`
- `PPO`
- `TRPO`
- `ARS`
- `CrossQ`
- `TQC`

---

## Basic Training

### Train UAV

```bash
python train.py --algorithm CrossQ --robot_type uav --set 3 --num_robots 3
```

### Train Wheeled Robot

```bash
python train.py --algorithm PPO --robot_type wheeled_robot --set 1
```

---

## Full Training Command

```bash
python train.py \
    --algorithm {A2C,PPO,TRPO,ARS,CrossQ,TQC} \
    --robot_type {uav,wheeled_robot} \
    --set [set number] \
    --num_robots [int, UAV only] \
    --verbose {0,1,2} \
    --steps [training steps] \
    --num_envs [parallel envs] \
    --seed [seed] \
    --log_steps [logging interval] \
    --resume {True,False} \
    --use_tuned_params {True,False} \
    --device {cpu,cuda}
```

---

## Logs and Checkpoints

Training logs are saved in:

```
logs/
    ├── training_default_logs/
    └── training_best_logs/
```

Each run creates:

```
logs/.../<robot>_<algorithm>_setX_seedY_v0/
    ├── tensorboard/
    ├── checkpoints/trained_model.zip
    ├── log.txt
    ├── progress.csv
    └── progress.json
```

---

## View Training Results

```bash
tensorboard --logdir=logs
```

Training for 2M timesteps typically takes:

- 2–8 hours (CPU)
- 1–4 hours (GPU, depending on algorithm)

---

# Resume Training

```bash
python train.py \
    --algorithm CrossQ \
    --robot_type uav \
    --set 3 \
    --resume True
```

---

# Hyperparameter Tuning

Hyperparameter tuning uses `tune.py`.

## Example

```bash
python tune.py --algorithm PPO --set 1
```

Tuned hyperparameters are saved in:

```
logs/tuning_logs/
```

To train using tuned parameters:

```bash
python train.py \
    --algorithm PPO \
    --robot_type uav \
    --set 1 \
    --use_tuned_params True
```

---

# Simulation

You must train a model before running simulation.

---

## PyGame Simulation

```bash
python run.py \
    --algorithm CrossQ \
    --robot_type uav \
    --set 3 \
    --num_robots 3 \
    --simulate False
```

---

## CoppeliaSim Simulation

We use CoppeliaSim for realistic 3D simulation.

### Step 1: Install CoppeliaSim

Download from:  
https://coppeliarobotics.com/

### Step 2: Open Scene

Open:

```
simulation_env/drone_test_scene_aug14.ttt
```

Important:
- Reopen the scene before every run.
- Do NOT save changes when closing.

### Step 3: Run Simulation

```bash
python run.py \
    --algorithm CrossQ \
    --robot_type uav \
    --set 3 \
    --num_robots 3 \
    --simulate True
```

---

# Running on Compute Clusters (Slurm)

Slurm scripts are provided in:

```
slurm_scripts/
```

### Run all training jobs

```bash
sbatch slurm_scripts/train_all.sh
```

### Run a single job

Edit:

```
slurm_scripts/train_one.sh
```

Then:

```bash
sbatch slurm_scripts/train_one.sh
```

---

# Plotting

To visualize experiment layouts:

```bash
python plotting/plot_fields.py
```

To compare results:

```bash
python plotting/plot_results.py
```

Optional flags:

```
-a  Training with default hyperparameters
-b  Training with best hyperparameters
-c  Transfer learning
-o  Hyperparameter tuning
```

Plots are saved in:

```
plotting/plots/
```

---

# Project Structure

```
├── train.py
├── run.py
├── tune.py
├── transfer.py
├── exp_sets/
├── logs/
├── slurm_scripts/
├── plotting/
└── src/
```