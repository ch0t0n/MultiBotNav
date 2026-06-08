#!/bin/bash
# ============================================================
# eval_wind_sweep.sh — Wind sensitivity sweep for Figure 4.
#
# Evaluates CrossQ under 10 wind-speed bands for both standard
# and DR-trained policies.  Run AFTER step7_dr.sh completes.
#
# Pass robot type as first argument (default: uav).
#
# Grid: 2 dr_modes × 10 wind bins × 3 seeds = 60 jobs per robot type
#
# Submit:
#   sbatch --array=0-59 eval_wind_sweep.sh uav
#   sbatch --array=0-59 eval_wind_sweep.sh wheeled
#
# Pre-submission (run once from the login node):
#   mkdir -p logs/slurm_outputs/eval_wind_sweep logs/results
# ============================================================

#SBATCH --array=0-59
#SBATCH --job-name=eval_wind_sweep
#SBATCH --output=logs/slurm_outputs/eval_wind_sweep/%x_%j.out
#SBATCH --error=logs/slurm_outputs/eval_wind_sweep/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --export=NONE

ROBOT_TYPE=${1:-uav}

seeds=(42 123 9999)
dr_modes=("none" "full")
# 10 equally-spaced wind bins spanning [0, 2.0] m/s
wind_mins=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8)
wind_maxs=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

num_seeds=${#seeds[@]}
num_dr=${#dr_modes[@]}
num_bins=${#wind_mins[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
bin_idx=$(( (index / num_seeds) % num_bins ))
dr_idx=$(( index / (num_seeds * num_bins) ))

seed=${seeds[$seed_idx]}
dr_mode=${dr_modes[$dr_idx]}
wind_min=${wind_mins[$bin_idx]}
wind_max=${wind_maxs[$bin_idx]}

# Default num_robots for wheeled is read from the .ini file by env.py
# but evaluate.py still requires a value — use 3 (a no-op for wheeled).
if [ "$ROBOT_TYPE" == "uav" ]; then
    NUM_ROBOTS=3
else
    NUM_ROBOTS=3
fi

OUT_CSV="logs/results/wind_sweep_${ROBOT_TYPE}.csv"

echo "wind_sweep | robot_type=$ROBOT_TYPE | dr_mode=$dr_mode | wind=[$wind_min,$wind_max] | seed=$seed"

/homes/choton/miniconda3/envs/robot_env/bin/python evaluate.py \
    --algorithm      CrossQ \
    --robot_type     $ROBOT_TYPE \
    --set            1 \
    --num_robots     $NUM_ROBOTS \
    --seed           $seed \
    --experiment     dr \
    --ablation       $dr_mode \
    --eval_wind_min  $wind_min \
    --eval_wind_max  $wind_max \
    --output_csv     $OUT_CSV \
    --n_eval_eps     50 \
    --device         cpu

wait
