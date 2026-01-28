#!/bin/bash
# Run CPU training: sbatch slurm_scripts/slurm_scripts_v2/train_from_tuned_cpu_v2.sh

#SBATCH --array=0-79           # 4 alg × 10 sets × 2 seeds = 80
#SBATCH --job-name=train_from_tuned_cpu_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm_scripts/slurm_scripts_v2/out/%x_%A_%a.out
#SBATCH --error=slurm_scripts/slurm_scripts_v2/out/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_scripts/slurm_scripts_v2/out
algorithms=("A2C" "PPO" "TRPO" "ARS")
sets=(1 2 3 4 5 6 7 8 9 10)
seeds=(0 1)

num_algorithms=${#algorithms[@]}
num_sets=${#sets[@]}
num_seeds=${#seeds[@]}

idx=$SLURM_ARRAY_TASK_ID

alg_idx=$(( idx / (num_sets * num_seeds) ))
set_idx=$(( (idx / num_seeds) % num_sets ))
seed_idx=$(( idx % num_seeds ))

algorithm=${algorithms[$alg_idx]}
set=${sets[$set_idx]}
seed=${seeds[$seed_idx]}

# Optional toggle at submit time:
#   sbatch --export=NONE,USE_TUNED_PARAMS=True slurm_scripts/slurm_scripts_v2/train_from_tuned_cpu_v2.sh
#   USE_TUNED_PARAMS=${USE_TUNED_PARAMS:-True}

echo "ALG=$algorithm SET=$set SEED=$seed USE_TUNED_PARAMS=True DEVICE=cpu"

conda run --no-capture-output -n rl4pag \
  python train_v2.py \
    --algorithm "$algorithm" \
    --set "$set" \
    --seed "$seed" \
    --version 2 \
    --steps 2000000 \
    --num_envs 4 \
    --device "cpu" \
    --use_tuned_params True \
    --use_wandb True\
    --wandb_project_name MultiBotNav_JANUARY_27


wait 
