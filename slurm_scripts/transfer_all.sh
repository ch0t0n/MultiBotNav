#!/bin/bash

# Run all experiments with: sbatch slurm_scripts/transfer_all.sh

#SBATCH --array=0-35
#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --export=NONE

# Modify these for other experiments
algorithms=("A2C" "PPO" "TRPO" "ARS")
sets=(2 3 4 5 6 7 8 9 10)

# IMPORTANT: array job length = num_algorithms * num_sets - 1
num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}

index=$((SLURM_ARRAY_TASK_ID))
algorithm_index=$((index / num_sets))
algorithm=${algorithms[$algorithm_index]}
set_index=$((index % num_sets))
set=${sets[$set_index]}

conda run --no-capture-output -n rl4pag python3 transfer.py --algorithm $algorithm --load_set 1 --train_set $set --steps 5000000 --verbose 1 --log_steps 5000 --seed 33

wait