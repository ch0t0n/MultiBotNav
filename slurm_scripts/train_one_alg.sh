#!/bin/bash

# Run all experiments with: sbatch training_scripts/train_all.sh
# IMPORTANT: array job length = num_algorithms * num_robot_types * num_sets - 1

#SBATCH --array=0-15
#SBATCH --job-name=RL4PAG
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

# Modify these for other experiments
algorithms=("TQC")
sets=(1 2 3 4 5 6 7 8)
robot_types=("uav" "wheeled_robot")
steps=10000

num_algorithms=${#algorithms[@]}
num_robot_types=${#robot_types[@]}
num_sets=${#sets[@]}

index=$SLURM_ARRAY_TASK_ID
set_index=$(( index % num_sets ))
index=$(( index / num_sets ))
robot_index=$(( index % num_robot_types ))
index=$(( index / num_robot_types ))
algorithm_index=$index

algorithm=${algorithms[$algorithm_index]}
robot_type=${robot_types[$robot_index]}
set=${sets[$set_index]}

conda run --no-capture-output -n rl4pag python3 train.py --algorithm $algorithm --set $set --robot_type $robot_type --verbose 1 --steps $steps

wait