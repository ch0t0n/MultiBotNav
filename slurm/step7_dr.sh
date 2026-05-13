#!/bin/bash
# ============================================================
# Step 7 — Domain Randomization (CrossQ, GPU)
#
# Three DR training conditions:
#   0 → none   (standard, no DR)
#   1 → wind   (re-sample wind each episode: U(0,1) m/s, U(0,2π))
#   2 → full   (wind + actuation noise + target radius + mass + thrust)
#
# Pass robot type as first argument (default: uav).
#
# Grid:
#   uav     → 3 DR × 10 env sets × 4 robot counts × 3 seeds = 360 jobs
#   wheeled → 3 DR ×  8 env sets × 1 robot count  × 3 seeds =  72 jobs
#
# Index layout (innermost → outermost):
#   seed_idx  = index %  num_seeds
#   robot_idx = (index / num_seeds) % num_robots
#   set_idx   = (index / (num_seeds * num_robots)) % num_sets
#   dr_idx    =  index / (num_seeds * num_robots * num_sets)
#
# Submit:
#   sbatch --array=0-359 step7_dr.sh uav
#   sbatch --array=0-71  step7_dr.sh wheeled
# ============================================================

#SBATCH --array=0-359
#SBATCH --job-name=s7_dr
#SBATCH --output=logs/slurm_outputs/s7_dr/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s7_dr/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# --- COMMAND TO EXCLUDE RTX_PRO_6000 (not supported by torch==2.4.0)
#SBATCH --exclude=warlock[41-42]

ROBOT_TYPE=${1:-uav}

dr_modes=("none" "wind" "full")
seeds=(42 123 9999)

if [ "$ROBOT_TYPE" == "uav" ]; then
    sets=(1 2 3 4 5 6 7 8 9 10)
    robots=(2 3 4 5)
else
    sets=(1 2 3 4 5 6 7 8)
    robots=(3)
fi

num_dr=${#dr_modes[@]}
num_sets=${#sets[@]}
num_robots=${#robots[@]}
num_seeds=${#seeds[@]}

index=$SLURM_ARRAY_TASK_ID
seed_idx=$(( index % num_seeds ))
robot_idx=$(( (index / num_seeds) % num_robots ))
set_idx=$(( (index / (num_seeds * num_robots)) % num_sets ))
dr_idx=$(( index / (num_seeds * num_robots * num_sets) ))

dr_mode=${dr_modes[$dr_idx]}
set=${sets[$set_idx]}
num_robots_value=${robots[$robot_idx]}
seed=${seeds[$seed_idx]}
steps=2000000

echo "S7-DR | robot_type=$ROBOT_TYPE | dr_mode=$dr_mode | set=$set | robots=$num_robots_value | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

/homes/choton/miniconda3/envs/robot_env/bin/python train.py \
    --algorithm   "CrossQ" \
    --robot_type  $ROBOT_TYPE \
    --set         $set \
    --num_robots  $num_robots_value \
    --seed        $seed \
    --steps       $steps \
    --experiment  dr \
    --ablation    $dr_mode \
    --verbose     1 \
    --log_steps   10000 \
    --device      cuda

wait
