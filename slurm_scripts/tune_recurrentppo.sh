#!/bin/bash

# Run a single experiment with: sbatch slurm_scripts/tune_recurrentppo.sh

#SBATCH --job-name=RL4PAg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=4G
#SBATCH --time=160:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1
#SBATCH --export=NONE

algorithm="RecurrentPPO"
set=1
steps=1000000

conda run --no-capture-output -n rl4pag python3 tune.py --algorithm $algorithm --set $set --steps $steps --seed 33 --log_steps 5000 --device "cuda"

wait