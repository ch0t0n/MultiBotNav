#!/bin/bash

# Submit with:
# sbatch training_scripts/train_all.sh

#SBATCH --job-name=RL4PAG
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------

algorithms=("A2C" "PPO" "TRPO" "ARS" "CrossQ" "TQC")
robot_types=("uav" "wheeled_robot")                # or ("uav" "wheeled_robot")
sets=(1 2 3 4 5 6 7 8)
seed=42
num_robots=3                        # used only for UAV
steps=2000000
num_envs=4
device="cuda"

use_tuned_params=false

# ------------------------------------------------------------
# Compute job array size automatically
# ------------------------------------------------------------

num_algorithms=${#algorithms[@]}
num_robot_types=${#robot_types[@]}
num_sets=${#sets[@]}
total_jobs=$((num_algorithms * num_robot_types * num_sets - 1))

#SBATCH --array=0-$total_jobs

# ------------------------------------------------------------
# Decode SLURM task index
# ------------------------------------------------------------

index=$SLURM_ARRAY_TASK_ID
set_index=$(( index % num_sets ))
index=$(( index / num_sets ))
robot_index=$(( index % num_robot_types ))
index=$(( index / num_robot_types ))
algorithm_index=$index

algorithm=${algorithms[$algorithm_index]}
robot_type=${robot_types[$robot_index]}
set=${sets[$set_index]}

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------

echo "Running:"
echo "Algorithm: $algorithm"
echo "Robot: $robot_type"
echo "Set: $set"

conda run --no-capture-output -n rl4pag python3 train.py \
    --algorithm $algorithm \
    --robot_type $robot_type \
    --set $set \
    --num_robots $num_robots \
    --seed $seed \
    --steps $steps \
    --num_envs $num_envs \
    --device $device \
    --use_tuned_params $use_tuned_params \
    --verbose 1

wait