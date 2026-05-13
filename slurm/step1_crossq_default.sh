#!/bin/bash
# ============================================================
# Step 1 — Main results, DEFAULT hyperparameters, CrossQ (GPU)
#
# Pass robot type as first argument (default: uav).
#
# Grid:
#   uav     → 1 alg × 10 sets × 4 robot counts × 3 seeds = 120 jobs
#   wheeled → 1 alg × 10 sets × 1 robot count  × 3 seeds =  30 jobs
#
# Index layout (innermost → outermost):
#   seed_idx  = index % 3
#   robot_idx = (index // 3) % num_robots
#   set_idx   = (index // (3 * num_robots)) % num_sets
#   alg_idx   = index // (3 * num_robots * num_sets)   (always 0)
#
# Submit:
#   sbatch --array=0-119 step1_crossq_default.sh uav
#   sbatch --array=0-29  step1_crossq_default.sh wheeled
# ============================================================

#SBATCH --array=0-119
#SBATCH --job-name=s1_crossq_default
#SBATCH --output=logs/slurm_outputs/s1_crossq_default/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s1_crossq_default/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]

ROBOT_TYPE=${1:-uav}

algorithms=("CrossQ")
seeds=(42 123 9999)

if [ "$ROBOT_TYPE" == "uav" ]; then
    sets=(1 2 3 4 5 6 7 8 9 10)
    robots=(2 3 4 5)
else
    sets=(1 2 3 4 5 6 7 8 9 10)
    robots=(3)
fi

num_sets=${#sets[@]}
num_robots=${#robots[@]}
num_seeds=${#seeds[@]}

index=$((SLURM_ARRAY_TASK_ID))
seed_idx=$(( index % num_seeds ))
robot_idx=$(( (index / num_seeds) % num_robots ))
set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
alg_idx=$(( index / (num_seeds * num_robots * num_sets) ))

algorithm=${algorithms[$alg_idx]}
set=${sets[$set_idx]}
num_robots_value=${robots[$robot_idx]}
seed=${seeds[$seed_idx]}
steps=2000000

echo "S1-CrossQ-default | robot_type=$ROBOT_TYPE | alg=$algorithm | set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

/homes/choton/miniconda3/envs/robot_env/bin/python train.py \
    --algorithm   $algorithm \
    --robot_type  $ROBOT_TYPE \
    --set         $set \
    --num_robots  $num_robots_value \
    --seed        $seed \
    --steps       $steps \
    --experiment  main \
    --verbose     1 \
    --log_steps   10000 \
    --device      cuda

wait
