# Learning to Navigate in a Stochastic and Uncertain Environment with Variable Motion Dynamics

This is the codebase for the paper *"Learning to Navigate in a Stochastic and
Uncertain Environment with Variable Motion Dynamics."*  It supports **two robot
platforms** with completely different motion dynamics, sharing the same
training/evaluation pipeline:

| Robot type | Gym ID            | Class           | Dynamics            | Config                                         |
|-----------:|:------------------|:----------------|:--------------------|:-----------------------------------------------|
| `uav`      | `MultiUAV-v0`     | `MultiUAV`      | Newtonian (ax, ay)  | `exp_sets/uav/cont_sets.json` (10 sets)         |
| `wheeled`  | `MultiWheeled-v0` | `MultiWheeled`  | Bicycle (accel, steer-rate) | `exp_sets/wheeled/wheeled_configs.json` (10 sets) |

**Every experiment is performed for both robot types and for 2–5 robots each.**
The CLI scripts (`train.py`, `evaluate.py`, `tune.py`, `sensitivity_hp.py`)
all accept `--robot_type {uav,wheeled}` and `--num_robots {2,3,4,5}`.  The
SLURM scripts accept the robot type as their first positional argument.

### Configuration files

| Robot type | Format | Path |
|:-----------|:-------|:-----|
| UAV        | JSON (one file, 10 sets) | `exp_sets/uav/cont_sets.json` |
| Wheeled    | JSON (one file, 10 sets) | `exp_sets/wheeled/wheeled_configs.json` |

The wheeled JSON stores **5 robot starting positions per environment** (the
maximum).  Passing `--num_robots N` (N ∈ {2,3,4,5}) tells `MultiWheeled` to
use only the first N of those positions, so the same config file covers all
robot-count experiments without duplication.

## Setup

It is recommended to run this codebase on Linux. The packages required to run
the code are listed in `requirements.txt`.  Install with:

```bash
pip install -r requirements.txt
```

## Summary (TLDR)

Follow **Part A** to run the experiments on a local machine (this may take a
long time).  Follow **Part B** if you have access to an HPC cluster with
SLURM.  A **single-file reference implementation** that bundles both
environments, the helpers, the gym registration, and the training launcher
is available in `single_file_implementation/`.  `example_training.py` bundles
both UAV and wheeled environments for quick inspection.  Once policies are
trained, see the **Simulation** section near the end of this document to run
trained models in the Ursina 3D wheeled demo or the CoppeliaSim UAV notebook.

---

# Part A — Local Machine (No HPC Cluster)

> **Device note:** Replace `--device cpu` with `--device cuda` anywhere below
> if you have a CUDA-capable GPU.  CrossQ and TQC benefit the most from a GPU.
>
> **Reproduction time warning:** The full grid (3 seeds × 10 sets × 4 robot
> counts for both UAV and wheeled, plus 6 algorithms) takes hundreds of
> CPU-hours.  The commands below show a **minimal single run** for sanity
> checking and **loop variants** for full coverage — start with the single
> run and only expand the loops once the pipeline is verified end-to-end.

## A-1. Step 1 — Default Hyperparameters (Local)

### Minimal single run — one for each robot type

```bash
# UAV, env set 1, 3 robots, seed 42, 2M steps
python train.py \
    --algorithm   CrossQ \
    --robot_type  uav \
    --set         1 \
    --num_robots  3 \
    --seed        42 \
    --steps       2000000 \
    --experiment  main \
    --device      cpu

# Wheeled, env set 1, 3 robots, seed 42
python train.py \
    --algorithm   CrossQ \
    --robot_type  wheeled \
    --set         1 \
    --num_robots  3 \
    --seed        42 \
    --steps       2000000 \
    --experiment  main \
    --device      cpu
```

Outputs land in:

```
logs/main_default/CrossQ_uav_N3_env1_seed42/        ← UAV
logs/main_default/CrossQ_wheeled_N3_env1_seed42/    ← wheeled
```

### Loop over both robot types, all algorithms, and 2–5 robots

```bash
for ROBOT in uav wheeled; do
  for ALG in CrossQ TQC PPO A2C TRPO ARS; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        for N in 2 3 4 5; do
          python train.py \
              --algorithm   $ALG \
              --robot_type  $ROBOT \
              --set         $SET \
              --num_robots  $N \
              --seed        $SEED \
              --steps       2000000 \
              --experiment  main \
              --device      cpu
        done
      done
    done
  done
done
```

---

## A-2. Step 2 — Hyperparameter Tuning (Local)

Tune each algorithm independently for each robot type.  The tuned
hyperparameters are written to a robot-type-tagged JSON so that UAV and
wheeled tunings do not overwrite each other.

```bash
mkdir -p logs/optuna_studies

for ROBOT in uav wheeled; do
  mkdir -p logs/optuna_studies/$ROBOT
  for ALG in A2C ARS PPO TRPO CrossQ TQC; do
    python tune.py \
        --algorithm   $ALG \
        --robot_type  $ROBOT \
        --device      cpu \
        --n_trials    50 \
        --tune_steps  500000 \
        --storage     logs/optuna_studies/${ROBOT}/${ALG}_journal.log \
        --study_name  ${ALG}_${ROBOT}_tune \
        --output_json logs/best_hyperparams_${ROBOT}.json
  done
done
```

**What this produces:**

```
logs/best_hyperparams_uav.json             ← consumed by Step A-3 (UAV)
logs/best_hyperparams_wheeled.json         ← consumed by Step A-3 (wheeled)
logs/optuna_studies/<robot>/<ALG>_journal.log
```

> **Resuming:** Re-run the same `tune.py` command — `load_if_exists=True`
> picks up where the study left off.

---

## A-3. Step 3 — Tuned Hyperparameters (Local)

Pass `--hyperparams_json` pointing at the matching robot-type JSON.

```bash
for ROBOT in uav wheeled; do
  for ALG in CrossQ TQC PPO A2C TRPO ARS; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        for N in 2 3 4 5; do
          python train.py \
              --algorithm        $ALG \
              --robot_type       $ROBOT \
              --set              $SET \
              --num_robots       $N \
              --seed             $SEED \
              --steps            2000000 \
              --experiment       main \
              --hyperparams_json logs/best_hyperparams_${ROBOT}.json \
              --device           cpu
        done
      done
    done
  done
done
```

Outputs land in `logs/main_tuned/<ALG>_<robot>_N<N>_env<SET>_seed<SEED>/`.

---

## A-4 to A-6. Ablation Training — Reward, Observation, Uncertainty (Local)

All three ablations use `train.py` with different `--experiment` and
`--ablation` flags and fix CrossQ and N=3.

| Experiment             | `--experiment`        | Valid `--ablation` values                        |
|------------------------|-----------------------|--------------------------------------------------|
| Reward ablation        | `ablation_reward`     | `full`, `no_term`, `no_path`                     |
| Observation ablation   | `ablation_obs`        | `full`, `no_pos`, `no_vis_hist`, `pos_only`      |
| Uncertainty ablation   | `ablation_uncertainty`| `full`, `wind_only`, `act_only`, `deterministic` |

### Step 4 — Reward ablation

```bash
for ROBOT in uav wheeled; do
  for CONDITION in full no_term no_path; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        python train.py \
            --algorithm  CrossQ \
            --robot_type $ROBOT \
            --set        $SET \
            --num_robots 3 \
            --seed       $SEED \
            --steps      2000000 \
            --experiment ablation_reward \
            --ablation   $CONDITION \
            --device     cpu
      done
    done
  done
done
```

### Step 5 — Observation ablation

```bash
for ROBOT in uav wheeled; do
  for OBS in full no_pos no_vis_hist pos_only; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        python train.py \
            --algorithm  CrossQ \
            --robot_type $ROBOT \
            --set        $SET \
            --num_robots 3 \
            --seed       $SEED \
            --steps      2000000 \
            --experiment ablation_obs \
            --ablation   $OBS \
            --device     cpu
      done
    done
  done
done
```

### Step 6 — Uncertainty ablation

```bash
for ROBOT in uav wheeled; do
  for MODE in full wind_only act_only deterministic; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        python train.py \
            --algorithm  CrossQ \
            --robot_type $ROBOT \
            --set        $SET \
            --num_robots 3 \
            --seed       $SEED \
            --steps      2000000 \
            --experiment ablation_uncertainty \
            --ablation   $MODE \
            --device     cpu
      done
    done
  done
done
```

---

## A-7. Step 7 — Domain Randomization (Local)

Three DR conditions: `none`, `wind`, `full`.  Both robot types sweep 2–5 robots.

```bash
for ROBOT in uav wheeled; do
  for DR in none wind full; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        for N in 2 3 4 5; do
          python train.py \
              --algorithm  CrossQ \
              --robot_type $ROBOT \
              --set        $SET \
              --num_robots $N \
              --seed       $SEED \
              --steps      2000000 \
              --experiment dr \
              --ablation   $DR \
              --device     cpu
        done
      done
    done
  done
done
```

---

## A-8. Post-Training Evaluation (Local)

Main results (Table 2 — UAV and Wheeled side-by-side, `tab:main_results`) and
the observation ablation block inside the combined ablation table (Table 3,
`tab:ablations_and_dr`) are read **directly from training NPZ files** — no
separate eval pass is required for those.  The experiments below do require
`evaluate.py`.

### A-8-A. Reward ablation (Table 3, left block of `tab:ablations_and_dr`)

```bash
mkdir -p logs/results
for ROBOT in uav wheeled; do
  for CONDITION in full no_term no_path; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        python evaluate.py \
            --algorithm  CrossQ \
            --robot_type $ROBOT \
            --set        $SET \
            --num_robots 3 \
            --seed       $SEED \
            --experiment ablation_reward \
            --ablation   $CONDITION \
            --output_csv logs/results/ablation_reward_${ROBOT}.csv \
            --n_eval_eps 50
      done
    done
  done
done
```

### A-8-B. Uncertainty ablation cross-evaluation matrix (Table 3, third block of `tab:ablations_and_dr`)

Each policy trained under one noise condition is evaluated under all four:

```bash
mkdir -p logs/results
for ROBOT in uav wheeled; do
  for TRAIN_MODE in full wind_only act_only deterministic; do
    for EVAL_MODE in full wind_only act_only deterministic; do
      for SET in 1 2 3 4 5 6 7 8 9 10; do
        for SEED in 42 123 9999; do
          python evaluate.py \
              --algorithm              CrossQ \
              --robot_type             $ROBOT \
              --set                    $SET \
              --num_robots             3 \
              --seed                   $SEED \
              --experiment             ablation_uncertainty \
              --ablation               $TRAIN_MODE \
              --eval_uncertainty_mode  $EVAL_MODE \
              --output_csv             logs/results/ablation_uncertainty_${ROBOT}.csv \
              --n_eval_eps             50
        done
      done
    done
  done
done
```

### A-8-C. Domain randomization — in-distribution & OOD (Table 3, right block of `tab:ablations_and_dr`)

```bash
mkdir -p logs/results
for ROBOT in uav wheeled; do
  for DR in none wind full; do
    for SET in 1 2 3 4 5 6 7 8 9 10; do
      for SEED in 42 123 9999; do
        for N in 2 3 4 5; do
          python evaluate.py \
              --algorithm     CrossQ --robot_type $ROBOT \
              --set $SET --num_robots $N --seed $SEED \
              --experiment dr --ablation $DR \
              --eval_wind_min 0.0 --eval_wind_max 0.5 \
              --output_csv logs/results/dr_inDist_${ROBOT}.csv \
              --n_eval_eps 50

          python evaluate.py \
              --algorithm     CrossQ --robot_type $ROBOT \
              --set $SET --num_robots $N --seed $SEED \
              --experiment dr --ablation $DR \
              --eval_wind_min 0.5 --eval_wind_max 2.0 \
              --output_csv logs/results/dr_OOD_${ROBOT}.csv \
              --n_eval_eps 50
        done
      done
    done
  done
done
```

### A-8-D. Wind sweep (Figure 4)

```bash
mkdir -p logs/results
wind_mins=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8)
wind_maxs=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

for ROBOT in uav wheeled; do
  for DR in none full; do
    for BIN in 0 1 2 3 4 5 6 7 8 9; do
      for SEED in 42 123 9999; do
        python evaluate.py \
            --algorithm     CrossQ \
            --robot_type    $ROBOT \
            --set           1 \
            --num_robots    3 \
            --seed          $SEED \
            --experiment    dr \
            --ablation      $DR \
            --eval_wind_min ${wind_mins[$BIN]} \
            --eval_wind_max ${wind_maxs[$BIN]} \
            --output_csv    logs/results/wind_sweep_${ROBOT}.csv \
            --n_eval_eps    50
      done
    done
  done
done
```

---

## A-9. Aggregate Results and Generate Table CSVs (Local)

**Table 2** (`tab:main_results`) is populated entirely from training NPZ files —
no separate `evaluate.py` run is needed.  Run the aggregation script for both
platforms, then run it once more with `--robot_type both` to emit the merged
LaTeX snippet:

```bash
# Step 1: aggregate per-platform results
python analyze_results.py --robot_type uav     --log_root logs --results_dir logs/results
python analyze_results.py --robot_type wheeled --log_root logs --results_dir logs/results

# Step 2: generate the merged Table 2 LaTeX rows (UAV + Wheeled side-by-side)
python analyze_results.py --robot_type both    --log_root logs --results_dir logs/results
```

**Output file:** `logs/results/main_combined_latex_rows_both.txt`

**How to paste into `writings/0_main.tex`:**

The file contains two clearly labelled blocks:

```
% ── (a) UAV minipage rows ───
    A2C    & ... \\
    ...
% ── (b) Wheeled minipage rows ───
    A2C    & ... \\
    ...
```

1. Open `writings/0_main.tex` and locate the merged Table 2 (`tab:main_results`).
2. In the **first** minipage (UAV, labelled `(a)`), replace the six placeholder
   rows (`A2C & $-$ & ...`) with the lines from the `(a) UAV minipage rows` block.
3. In the **second** minipage (Wheeled, labelled `(b)`), replace the six placeholder
   rows with the lines from the `(b) Wheeled minipage rows` block.
4. Each block of rows goes between `\midrule` and `\bottomrule` of its minipage.

The other output files from this step feed the other tables:

| Output file | Target |
|---|---|
| `main_combined_latex_rows_both.txt` | Table 2 (`tab:main_results`) |
| `combined_ablations_dr_latex_rows_uav.txt` | Table 3 UAV columns (`tab:ablations_and_dr`) |
| `combined_ablations_dr_latex_rows_wheeled.txt` | Table 3 Wheeled columns |

## A-10. Generate All Figures (Local)

```bash
# Generates UAV-tagged and wheeled-tagged figures side by side
python plot_figures.py --robot_type both \
    --log_root logs --results_dir logs/results --figures_dir figures
```

## A-11. Hyperparameter Sensitivity (supplementary / not yet in 0_main.tex)

```bash
for ROBOT in uav wheeled; do
  for ALG in A2C ARS PPO TRPO CrossQ TQC; do
    python sensitivity_hp.py \
        --algorithm  $ALG \
        --robot_type $ROBOT \
        --results_dir logs/results
  done
  python sensitivity_hp.py --robot_type $ROBOT --write_latex_only --results_dir logs/results
done
```

---

## Tips for Local Runs

| Goal | Suggestion |
|---|---|
| Smoke-test end-to-end quickly | Use `--steps 50000` and one seed/set/N |
| Speed up CPU training | Increase `--num_envs` to match your CPU cores |
| Use a GPU | Set `--device cuda`; most helpful for CrossQ and TQC |
| Resume interrupted tuning | Re-run the same `tune.py` command (`load_if_exists=True` continues) |
| Avoid re-running finished evals | `evaluate.py` skips rows already present in the output CSV |

---

# Part B — HPC Cluster (SLURM)

All commands assume your working directory is the project root.  Every
SLURM script takes the robot type as its **first positional argument**
(default `uav`).  Submit each step twice — once for `uav`, once for
`wheeled`.

## B-1. Step 1 — Default Hyperparameters

Both robot types now run 4 robot counts × 10 sets × 3 seeds = 120 jobs for CrossQ.

```bash
# CrossQ (GPU) — 120 jobs each
sbatch --array=0-119 slurm/step1_crossq_default.sh uav
sbatch --array=0-119 slurm/step1_crossq_default.sh wheeled

# Others (CPU) — 600 jobs each
sbatch --array=0-599 slurm/step1_others_default.sh uav
sbatch --array=0-599 slurm/step1_others_default.sh wheeled
```

**Output layout:**
```
logs/main_default/{ALG}_{robot}_N{N}_env{SET}_seed{SEED}/
    best_model/best_model.zip
    eval_logs/evaluations.npz
    *.csv  (SB3 training log)
```

## B-2. Step 2 — Optuna Hyperparameter Tuning

```bash
sbatch --array=0-299 slurm/step2_tune_hyperparameters.sh uav
sbatch --array=0-299 slurm/step2_tune_hyperparameters.sh wheeled
```

**Output:** `logs/best_hyperparams_uav.json`, `logs/best_hyperparams_wheeled.json`,
journal logs under `logs/optuna_studies/{uav,wheeled}/`.

## B-3. Step 3 — Tuned Hyperparameters

```bash
# CrossQ (GPU) — 120 jobs each
sbatch --array=0-119 slurm/step3_crossq_tuned.sh uav
sbatch --array=0-119 slurm/step3_crossq_tuned.sh wheeled

# Others (CPU) — 600 jobs each
sbatch --array=0-599 slurm/step3_others_tuned.sh uav
sbatch --array=0-599 slurm/step3_others_tuned.sh wheeled
```

## B-4–6. Ablation Training — Reward, Observation, Uncertainty

The merged script accepts the experiment as `arg1` and robot type as `arg2`.
Ablations fix N=3 robots.

```bash
# UAV  (3 cond × 10 sets × 3 seeds = 90 jobs for reward; 4 cond × 10 × 3 = 120 for obs/uncertainty)
sbatch --array=0-89  slurm/step4_5_6_ablations.sh ablation_reward       uav
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_obs          uav
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_uncertainty  uav

# Wheeled  (same job counts as UAV)
sbatch --array=0-89  slurm/step4_5_6_ablations.sh ablation_reward       wheeled
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_obs          wheeled
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_uncertainty  wheeled
```

> `ablation_reward` has 3 conditions (90 jobs); `ablation_obs` and
> `ablation_uncertainty` each have 4 conditions (120 jobs each).

## B-7. Step 7 — Domain Randomization

Both robot types now sweep 4 robot counts → 360 jobs each.

```bash
sbatch --array=0-359 slurm/step7_dr.sh uav      # 3 DR × 10 sets × 4 robots × 3 seeds = 360
sbatch --array=0-359 slurm/step7_dr.sh wheeled  # 3 DR × 10 sets × 4 robots × 3 seeds = 360
```

## B-8. Post-Training Evaluation

`eval_ablations.sh` accepts experiment as `arg1` and robot type as `arg2`.

### B-8-A. Reward ablation (Table 3, left block of `tab:ablations_and_dr`)

```bash
sbatch --array=0-89 slurm/eval_ablations.sh ablation_reward uav      # 90 jobs
sbatch --array=0-89 slurm/eval_ablations.sh ablation_reward wheeled  # 90 jobs
```

Merge per-job CSVs (one merge per robot type):

```bash
for ROBOT in uav wheeled; do
    OUT="logs/results/ablation_reward_${ROBOT}.csv"
    SRC="logs/results/tmp/ablation_reward_${ROBOT}"
    first_file=$(ls ${SRC}/*/set*/result_*.csv | head -n 1)
    head -n 1 "$first_file" > "$OUT"
    tail -n +2 -q ${SRC}/*/set*/result_*.csv >> "$OUT"
    echo "Merged -> $OUT"
done
```

### B-8-B. Uncertainty ablation cross-eval matrix (Table 3, third block of `tab:ablations_and_dr`)

```bash
sbatch --array=0-479 slurm/eval_ablations.sh ablation_uncertainty uav      # 4×4×10×3 = 480
sbatch --array=0-479 slurm/eval_ablations.sh ablation_uncertainty wheeled  # 4×4×10×3 = 480
```

Merge:

```bash
for ROBOT in uav wheeled; do
    OUT="logs/results/ablation_uncertainty_${ROBOT}.csv"
    SRC="logs/results/tmp/ablation_uncertainty_${ROBOT}"
    first_file=$(ls ${SRC}/train_*_eval_*/set*/result_*.csv | head -n 1)
    head -n 1 "$first_file" > "$OUT"
    tail -n +2 -q ${SRC}/train_*_eval_*/set*/result_*.csv >> "$OUT"
    echo "Merged -> $OUT"
done
```

### B-8-C. Domain randomization eval (Table 3, right block of `tab:ablations_and_dr`)

Both robot types now require 360 jobs each.

```bash
sbatch --array=0-359 slurm/eval_ablations.sh dr uav      # 3×10×4×3 = 360
sbatch --array=0-359 slurm/eval_ablations.sh dr wheeled  # 3×10×4×3 = 360
```

Merge in-distribution / OOD:

```bash
for ROBOT in uav wheeled; do
    for KIND in inDist OOD; do
        OUT="logs/results/dr_${KIND}_${ROBOT}.csv"
        SRC="logs/results/tmp/dr_${ROBOT}"
        first_file=$(ls ${SRC}/*/*/${KIND}_*.csv | head -n 1)
        head -n 1 "$first_file" > "$OUT"
        tail -n +2 -q ${SRC}/*/*/${KIND}_*.csv >> "$OUT"
        echo "Merged -> $OUT"
    done
done
```

### B-8-D. Wind sweep (Figure 4)

```bash
sbatch --array=0-59 slurm/eval_wind_sweep.sh uav      # 2×10×3 = 60
sbatch --array=0-59 slurm/eval_wind_sweep.sh wheeled  # 2×10×3 = 60
```

> `eval_wind_sweep.sh` appends directly to `logs/results/wind_sweep_<robot>.csv`
> (guarded by `fcntl` file locking in `evaluate.py`).  No merge step required.

### B-8-E. Hyperparameter sensitivity (supplementary / not yet in 0_main.tex)

```bash
sbatch --array=0-5 slurm/step8_sensitivity_hp.sh uav
sbatch --array=0-5 slurm/step8_sensitivity_hp.sh wheeled

# After all jobs finish, regenerate the LaTeX rows:
python sensitivity_hp.py --robot_type uav     --write_latex_only --results_dir logs/results
python sensitivity_hp.py --robot_type wheeled --write_latex_only --results_dir logs/results
```

## B-9. Aggregate Results

**Table 2** (`tab:main_results`) is populated from training NPZ files — no
separate evaluation step is needed.

```bash
# Aggregate per-platform results
python analyze_results.py --robot_type uav     --log_root logs --results_dir logs/results
python analyze_results.py --robot_type wheeled --log_root logs --results_dir logs/results

# Generate the merged Table 2 LaTeX rows (UAV + Wheeled side-by-side)
python analyze_results.py --robot_type both    --log_root logs --results_dir logs/results
```

**Output:** `logs/results/main_combined_latex_rows_both.txt`

Paste the `(a) UAV minipage rows` block between `\midrule` and `\bottomrule` of
the first minipage in `tab:main_results`, and the `(b) Wheeled minipage rows`
block into the second minipage.  See **A-9** for the full paste instructions.

## B-10. Generate Figures

```bash
python plot_figures.py --robot_type both \
    --log_root logs --results_dir logs/results --figures_dir figures
```

---

## Summary of SLURM Job Counts

| Step | Script                        | UAV jobs | Wheeled jobs |
|------|-------------------------------|---------:|-------------:|
| 1a   | step1_crossq_default.sh       |     120  |      120     |
| 1b   | step1_others_default.sh       |     600  |      600     |
| 2    | step2_tune_hyperparameters.sh |     300  |      300     |
| 3a   | step3_crossq_tuned.sh         |     120  |      120     |
| 3b   | step3_others_tuned.sh         |     600  |      600     |
| 4    | step4_5_6_ablations.sh reward |      90  |       90     |
| 5–6  | step4_5_6_ablations.sh ×2     |    2×120 |    2×120     |
| 7    | step7_dr.sh                   |     360  |      360     |
| 8a   | eval_ablations.sh reward      |      90  |       90     |
| 8b   | eval_ablations.sh uncertainty |     480  |      480     |
| 8c   | eval_ablations.sh dr          |     360  |      360     |
| 8d   | eval_wind_sweep.sh            |      60  |       60     |
| 8e   | step8_sensitivity_hp.sh       |       6  |        6     |

> **Re-submission safety:** `evaluate.py` skips a run when the matching row
> already exists in the output CSV (matched on algorithm, robot_type,
> experiment, ablation, **hp_tag**, num_robots, env_set, seed, eval_wind_min,
> eval_wind_max, eval_uncertainty_mode), so re-submitting failed array
> indices never produces duplicate rows.

---

## Dependency Graph (submit in order)

Table numbering in `writings/0_main.tex`:
  Tab. 1 = tab:obs_modes       (observation space composition)
  Tab. 2 = tab:main_results    (UAV + Wheeled side-by-side, merged)
  Tab. 3 = tab:ablations_and_dr (reward / obs / uncertainty ablations + DR, combined)
  Tab. 4 = tab:obs_gap         (sim-to-real observation gap)

```
Step 1 (default train, both robots, N=2-5)
    -> analyze_results.py --robot_type uav     -> Tab. 2 left half  (reads NPZ directly)
    -> analyze_results.py --robot_type wheeled -> Tab. 2 right half (reads NPZ directly)
    -> analyze_results.py --robot_type both    -> Tab. 2 merged (UAV + Wheeled side-by-side)
       plot_figures.py    --robot_type both    -> Fig. 1

Step 2 (tune, both robots) -> Step 3 (tuned train, N=2-5)
    -> analyze_results.py --robot_type both    -> Tab. 2 (merged, tuned half)
       plot_figures.py    --robot_type both    -> Fig. 2, 3

Steps 4-6 (ablations, both robots, N=3 fixed)
    -> eval_ablations.sh (reward, uncertainty) per robot
            -> merge CSVs -> analyze_results.py -> Tab. 3 (reward, obs, uncertainty blocks)
       (obs ablation needs no separate eval — read from NPZ)

Step 7 (DR, both robots, N=2-5)
    -> eval_ablations.sh (dr) per robot -> merge -> Tab. 3 (DR block)
    -> eval_wind_sweep.sh per robot -> plot_figures.py -> Fig. 4, 5

Step 2 -> (optional) step8_sensitivity_hp.sh (both robots) -> supplementary

sim2real.py (UAV only, after Step 1) -> Tab. 4 (tab:obs_gap)
```

---

# Simulation

## Wheeled robots (Ursina 3D)

After training (or using the bundled checkpoints under `trained_models/wheeled/`),
visualize wheeled policies in 3D:

```bash
cd simulation/wheeled
python run_simulation.py --env-key env1
python run_simulation.py --env-key env6 --random-policy   # no checkpoint needed
```

See [`simulation/wheeled/README.md`](simulation/wheeled/README.md) for all CLI
flags, asset download, and coordinate mapping.  Physics match `src/env.py`
(`MultiWheeled`); the simulator imports the production environment directly.

## CoppeliaSim (UAV)

First, download and install the CoppeliaSim robotics simulator from
<https://coppeliarobotics.com/>.  Open the scene file
`simulation/uav/coppeliasim_envs/uav_common_env.ttt` in the simulator, then
follow the instructions in `simulation/uav/uav_sim_new.ipynb`.

For the sim-to-real **observation gap** experiment (Table 4 in `0_main.tex`,
label `tab:obs_gap`) on the UAV platform, run:

```bash
python sim2real.py
```

This loads `logs/main_default/CrossQ_uav_N3_env1_seed42/best_model/best_model.zip`
(the default model path in `sim2real.py`) and runs 50 episodes per perturbation
condition.  Edit `TRAINED_MODEL_PATH` inside the script if your model lives
elsewhere.  Set `RENDER_COPPELIA = False` (the default) to run without
CoppeliaSim; set it to `True` only when the simulator is open and the scene
file is loaded.

> **IMPORTANT:** Reopen the CoppeliaSim scene each time before running the
> simulation.  Never save changes to the scene file when closing.
