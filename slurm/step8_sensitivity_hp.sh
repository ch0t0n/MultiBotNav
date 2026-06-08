#!/bin/bash
# ============================================================
# step8_sensitivity_hp.sh — Hyperparameter sensitivity (Table 6)
#
# One job per algorithm.
# Each job sweeps ALL hyperparameters and grid points.
#
# Pass robot type as first argument (default: uav).
#
# Total jobs per robot type: 6  →  array=0-5
#   sbatch --array=0-5 step8_sensitivity_hp.sh uav
#   sbatch --array=0-5 step8_sensitivity_hp.sh wheeled
#
# After all jobs finish:
#   python sensitivity_hp.py --robot_type uav     --write_latex_only --results_dir logs/results
#   python sensitivity_hp.py --robot_type wheeled --write_latex_only --results_dir logs/results
# ============================================================

#SBATCH --array=0-5
#SBATCH --job-name=s8_sensitivity_hp
#SBATCH --output=logs/slurm_outputs/step8_sens/%x_%j.out
#SBATCH --error=logs/slurm_errors/step8_sens/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --export=NONE

ROBOT_TYPE=${1:-uav}

algorithms=("A2C" "ARS" "PPO" "TRPO" "CrossQ" "TQC")

index=$SLURM_ARRAY_TASK_ID
algorithm=${algorithms[$index]}

device="cpu"

BEST_JSON="logs/best_hyperparams_${ROBOT_TYPE}.json"
RESULTS_DIR="logs/results"

echo "S8-sensitivity | robot=$ROBOT_TYPE | alg=$algorithm | job=$SLURM_ARRAY_TASK_ID"

/homes/choton/miniconda3/envs/robot_env/bin/python sensitivity_hp.py \
    --algorithm        $algorithm \
    --robot_type       $ROBOT_TYPE \
    --hyperparams_json $BEST_JSON \
    --results_dir      $RESULTS_DIR \
    --device           $device

wait
