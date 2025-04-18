#!/bin/bash

# Run all experiments with: sbatch slurm_scripts/train_all.sh

#SBATCH --array=0-49
#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --export=NONE

# Modify these for other experiments
algorithms=("A2C" "PPO" "TRPO" "ARS" "DQN")
sets=(1 2 3 4 5 6 7 8 9 10)

# IMPORTANT: array job length = num_algorithms * num_sets - 1
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}

index=$((SLURM_ARRAY_TASK_ID))
algorithm_index=$((index / num_sets))
algorithm=${algorithms[$algorithm_index]}
set_index=$((index % num_sets))
set=${sets[$set_index]}

conda run --no-capture-output -n rl4pag python3 train.py --algorithm $algorithm --set $set --verbose 1 --steps 2000000 --log_steps 5000 --seed 33

wait