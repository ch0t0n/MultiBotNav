#!/bin/bash

# Run tuning for all algorithms with: sbatch slurm_scripts/tune_all.sh

#SBATCH --array=0-4
#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=80:00:00
#SBATCH --export=NONE

# Modify these for other experiments
algorithms=("A2C" "PPO" "TRPO" "DQN" "ARS")
sets=(1)

# IMPORTANT: array job length = num_algorithms * num_sets - 1
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}

index=$((SLURM_ARRAY_TASK_ID))
algorithm_index=$((index / num_sets))
algorithm=${algorithms[$algorithm_index]}
set_index=$((index % num_sets))
set=${sets[$set_index]}

conda run --no-capture-output -n rl4pag python3 tune.py --algorithm $algorithm --set $set --steps 1000000 --seed 33 --log_steps 5000

wait