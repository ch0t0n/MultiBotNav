#!/bin/bash

# Run a single experiment with: sbatch slurm_scripts/transfer_one.sh

#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --export=NONE

algorithm="A2C"
load_set=1
train_set=3
steps=5000000

conda run --no-capture-output -n rl4pag python3 transfer.py --algorithm $algorithm --load_set $load_set --train_set $train_set --steps $steps --verbose 1 --log_steps 5000 --seed 33

wait