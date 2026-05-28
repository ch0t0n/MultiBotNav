# SLURM Submission Order (Part B)

---

## Wave 1 — Submit Immediately (all independent)

All four groups below have no dependencies on each other and can be submitted at the same time:

```bash
# Step 1 — Default training (1,440 jobs total)
sbatch --array=0-119 slurm/step1_crossq_default.sh uav
sbatch --array=0-119 slurm/step1_crossq_default.sh wheeled
sbatch --array=0-599 slurm/step1_others_default.sh uav
sbatch --array=0-599 slurm/step1_others_default.sh wheeled

# Step 2 — Optuna tuning (600 jobs total)
sbatch --array=0-299 slurm/step2_tune_hyperparameters.sh uav
sbatch --array=0-299 slurm/step2_tune_hyperparameters.sh wheeled

# Steps 4–6 — Ablation training (900 jobs total)
sbatch --array=0-89  slurm/step4_5_6_ablations.sh ablation_reward      uav
sbatch --array=0-89  slurm/step4_5_6_ablations.sh ablation_reward      wheeled
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_obs         uav
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_obs         wheeled
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_uncertainty uav
sbatch --array=0-119 slurm/step4_5_6_ablations.sh ablation_uncertainty wheeled

# Step 7 — DR training (720 jobs total)
sbatch --array=0-359 slurm/step7_dr.sh uav
sbatch --array=0-359 slurm/step7_dr.sh wheeled
```

---

## Wave 2 — After Step 2 finishes

Step 2 produces `logs/best_hyperparams_{uav,wheeled}.json`. Both of these need it:

```bash
# Step 3 — Tuned training (1,440 jobs total)
sbatch --array=0-119 slurm/step3_crossq_tuned.sh uav
sbatch --array=0-119 slurm/step3_crossq_tuned.sh wheeled
sbatch --array=0-599 slurm/step3_others_tuned.sh uav
sbatch --array=0-599 slurm/step3_others_tuned.sh wheeled

# Step 8e — HP sensitivity (12 jobs total)
sbatch --array=0-5 slurm/step8_sensitivity_hp.sh uav
sbatch --array=0-5 slurm/step8_sensitivity_hp.sh wheeled
```

Step 3 and Step 8e are independent of each other and can be submitted simultaneously once Step 2 is done.

---

## Wave 3 — After their respective training steps finish (all independent of each other)

Three evaluation groups, each gated on a different training step:

**After Step 1 finishes:**
```bash
python sim2real.py     # UAV obs-gap study → Tab. 4
```

**After Steps 4 & 6 finish:**
```bash
# Reward ablation eval (180 jobs) → then merge CSVs → Tab. 3
sbatch --array=0-89 slurm/eval_ablations.sh ablation_reward uav
sbatch --array=0-89 slurm/eval_ablations.sh ablation_reward wheeled

# Uncertainty ablation eval (960 jobs) → then merge CSVs → Tab. 3
sbatch --array=0-479 slurm/eval_ablations.sh ablation_uncertainty uav
sbatch --array=0-479 slurm/eval_ablations.sh ablation_uncertainty wheeled
```

**After Step 7 finishes:**
```bash
# DR eval (720 jobs) → then merge CSVs → Tab. 3
sbatch --array=0-359 slurm/eval_ablations.sh dr uav
sbatch --array=0-359 slurm/eval_ablations.sh dr wheeled

# Wind sweep (120 jobs, no merge needed) → Fig. 4, 5
sbatch --array=0-59 slurm/eval_wind_sweep.sh uav
sbatch --array=0-59 slurm/eval_wind_sweep.sh wheeled
```

After each eval batch finishes, run its CSV merge commands from `INSTRUCTIONS.MD`.

---

## Wave 4 — analyze_results.py (incremental — run as data arrives)

This can be run incrementally. Each invocation is independent:

| When to run | Command | Produces |
|---|---|---|
| After **Step 1** | `python analyze_results.py --robot_type uav --log_root logs --results_dir logs/results` | Tab. 2 default-UAV rows |
| After **Step 1** | `python analyze_results.py --robot_type wheeled --log_root logs --results_dir logs/results` | Tab. 2 default-wheeled rows |
| After **Step 3** | Re-run both + `--robot_type both` | Tab. 2 tuned rows + merged both |
| After **Step 5 NPZ** + **eval merges (Steps 4, 6, 7)** | Re-run for each robot type | Tab. 3 all blocks |
| After **Step 8e** | `python sensitivity_hp.py --write_latex_only --robot_type uav --results_dir logs/results` | Supplementary HP sensitivity table |
| After **Step 8e** | `python sensitivity_hp.py --write_latex_only --robot_type wheeled --results_dir logs/results` | Supplementary HP sensitivity table |

---

## Wave 5 — plot_figures.py (after analyze_results.py is populated)

```bash
python plot_figures.py --robot_type both \
    --log_root logs --results_dir logs/results --figures_dir figures
```

---

## Visual dependency diagram

```
Wave 1 ──────────────────────────────────────────────────────────────────
  Step 1 ──────────────────┬─────────────────────────────────────────────
  Step 2 ──────────────────┼──► Wave 2: Step 3, Step 8e
  Steps 4–6 ───────────────┼──► Wave 3: eval_reward, eval_uncertainty
  Step 7 ──────────────────┼──► Wave 3: eval_dr, wind_sweep
                           │
Wave 3 ───────────────────┬┘
  After Step 1  ──────────► sim2real.py
                │          analyze_results.py (Tab. 2 default)
  After 4+6 eval merges──► analyze_results.py (Tab. 3 reward+unc blocks)
  After 7 eval merges ───► analyze_results.py (Tab. 3 DR block)

Wave 4 ──────────────────────────────────────────────────────────────────
  analyze_results.py (all robot types + both) ──► Wave 5: plot_figures.py
```

---

## Notes

- **Step 1 is the long pole.** Submit all of Wave 1 on day 1. Step 2 (Optuna tuning) typically finishes first since each trial trains for only 500k steps. Fire Wave 2 as soon as Step 2 is done without waiting for Step 1.
- **Ablation training (Steps 4–6) and DR training (Step 7) are fully independent of Steps 1–3.** They can be in the queue simultaneously.
- **obs ablation (Step 5) needs no separate eval pass.** `analyze_results.py` reads its NPZ files directly from the training callbacks.
- **Re-submission is safe.** `evaluate.py` skips any (algorithm, robot_type, experiment, ablation, hp_tag, num_robots, env_set, seed, wind, uncertainty) combination already present in the output CSV, so resubmitting failed array indices never duplicates rows.
- **Wheeled main models land in `best_model_stage2/`.** `evaluate.py` and `analyze_results.py` detect and prefer the stage-2 paths automatically.
