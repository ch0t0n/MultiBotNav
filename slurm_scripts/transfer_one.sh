#!/bin/bash

# Run a single experiment with: sbatch slurm_scripts/transfer_one.sh

#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --export=NONE

algorithm="PPO"
load_set=1
train_set=2
steps=2000000

conda run --no-capture-output -n rl4pag python3 transfer.py --algorithm $algorithm --load_set $load_set --train_set $train_set --verbose 1 --steps $steps

wait