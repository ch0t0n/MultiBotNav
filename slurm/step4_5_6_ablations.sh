#!/bin/bash
# ============================================================
# step4_5_6_ablations.sh — Merged training ablation script
#
# Covers Steps 4, 5, and 6 in a single file.
# Pass the experiment as arg1, robot type as arg2 (default: uav).
#
#   sbatch --array=0-89  step4_5_6_ablations.sh ablation_reward       uav      # 3 cond × 10 sets × 3 seeds = 90
#   sbatch --array=0-89  step4_5_6_ablations.sh ablation_reward       wheeled  # 3 cond × 10 sets × 3 seeds = 90
#   sbatch --array=0-119 step4_5_6_ablations.sh ablation_obs          uav      # 4 cond × 10 sets × 3 seeds = 120
#   sbatch --array=0-119 step4_5_6_ablations.sh ablation_obs          wheeled  # 4 cond × 10 sets × 3 seeds = 120
#   sbatch --array=0-119 step4_5_6_ablations.sh ablation_uncertainty  uav      # 4 cond × 10 sets × 3 seeds = 120
#   sbatch --array=0-119 step4_5_6_ablations.sh ablation_uncertainty  wheeled  # 4 cond × 10 sets × 3 seeds = 120
#
# Fixed: CrossQ, N=3 (or env-defined for wheeled), 2M timesteps
# Grid:
#   uav     -> 4 conditions x 10 env sets x 3 seeds = 120 jobs
#   wheeled -> 4 conditions x 10 env sets x 3 seeds = 120 jobs  (3 cond x 10 x 3 = 90 for ablation_reward)
#
# Index layout (innermost -> outermost):
#   seed_idx = index % 3
#   set_idx  = (index / 3) % num_sets
#   cond_idx = index / (3 * num_sets)
# ============================================================

#SBATCH --job-name=s4_5_6_ablations
#SBATCH --output=logs/slurm_outputs/s4_5_6_ablations/%x_%A_%a.out
#SBATCH --error=logs/slurm_errors/s4_5_6_ablations/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --export=NONE

EXPERIMENT=${1:-ablation_reward}
ROBOT_TYPE=${2:-uav}

seeds=(42 123 9999)
num_seeds=${#seeds[@]}

if [ "$ROBOT_TYPE" == "uav" ]; then
    sets=(1 2 3 4 5 6 7 8 9 10)
else
    sets=(1 2 3 4 5 6 7 8 9 10)
fi
num_sets=${#sets[@]}

index=$SLURM_ARRAY_TASK_ID

seed_idx=$(( index % num_seeds ))
set_idx=$(( (index / num_seeds) % num_sets ))
cond_idx=$(( index / (num_seeds * num_sets) ))

seed=${seeds[$seed_idx]}
set=${sets[$set_idx]}

# ============================================================
# Step 4: Reward function components
#   full     all reward terms (baseline)
#   no_term  collision penalty + success bonus disabled
#   no_path  energy, speed, path, and time penalties disabled
# ============================================================
if [ "$EXPERIMENT" == "ablation_reward" ]; then

    conditions=("full" "no_term" "no_path")
    condition=${conditions[$cond_idx]}

    echo "S4-ablation-reward | robot=$ROBOT_TYPE | cond=$condition | set=$set | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

    /homes/choton/miniconda3/envs/robot_env/bin/python train.py \
        --algorithm   "CrossQ" \
        --robot_type  $ROBOT_TYPE \
        --set         $set \
        --num_robots  3 \
        --seed        $seed \
        --steps       2000000 \
        --experiment  ablation_reward \
        --ablation    $condition \
        --verbose     1 \
        --log_steps   10000 \
        --device      cpu

# ============================================================
# Step 5: Observation space
#   full         positions + velocities + visited/goal decimal
#   no_pos       visited/goal decimal only
#   no_vis_hist  positions + velocities only
#   pos_only     robot positions only (2N)
# ============================================================
elif [ "$EXPERIMENT" == "ablation_obs" ]; then

    obs_modes=("full" "no_pos" "no_vis_hist" "pos_only")
    obs_mode=${obs_modes[$cond_idx]}

    echo "S5-ablation-obs | robot=$ROBOT_TYPE | obs=$obs_mode | set=$set | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

    /homes/choton/miniconda3/envs/robot_env/bin/python train.py \
        --algorithm   "CrossQ" \
        --robot_type  $ROBOT_TYPE \
        --set         $set \
        --num_robots  3 \
        --seed        $seed \
        --steps       2000000 \
        --experiment  ablation_obs \
        --ablation    $obs_mode \
        --verbose     1 \
        --log_steps   10000 \
        --device      cpu

# ============================================================
# Step 6: Physical uncertainty model
#   full          wind + actuation noise (default)
#   wind_only     only wind noise active
#   act_only      only actuation noise active
#   deterministic all noise sources disabled
# ============================================================
elif [ "$EXPERIMENT" == "ablation_uncertainty" ]; then

    uncertainty_modes=("full" "wind_only" "act_only" "deterministic")
    uncertainty_mode=${uncertainty_modes[$cond_idx]}

    echo "S6-ablation-uncertainty | robot=$ROBOT_TYPE | mode=$uncertainty_mode | set=$set | seed=$seed | job=$SLURM_ARRAY_TASK_ID"

    /homes/choton/miniconda3/envs/robot_env/bin/python train.py \
        --algorithm   "CrossQ" \
        --robot_type  $ROBOT_TYPE \
        --set         $set \
        --num_robots  3 \
        --seed        $seed \
        --steps       2000000 \
        --experiment  ablation_uncertainty \
        --ablation    $uncertainty_mode \
        --verbose     1 \
        --log_steps   10000 \
        --device      cpu

else
    echo "ERROR: Unknown experiment '$EXPERIMENT'."
    echo "       Valid options: ablation_reward, ablation_obs, ablation_uncertainty"
    exit 1
fi

wait
