#!/bin/bash
# Run GPU tuning (v2, all seeds)

#SBATCH --array=0-39           # 2 alg × 10 sets × 2 seeds = 40
#SBATCH --job-name=tune_gpu_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=160:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --output=slurm_scripts/slurm_scripts_v2/out/%x_%A_%a.out
#SBATCH --error=slurm_scripts/slurm_scripts_v2/out/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_scripts/slurm_scripts_v2/out

algorithms=("CrossQ" "TQC")
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

echo "ALG=$algorithm SET=$set SEED=$seed DEVICE=cuda"

conda run --no-capture-output -n rl4pag \
  python tune_v2.py \
    --algorithm "$algorithm" \
    --set "$set" \
    --seed "$seed" \
    --version 2 \
    --steps 1000000 \
    --trials 20 \
    --num_envs 4 \
    --device "cuda" \
    --use_wandb True\
    --wandb_project_name MultiBotNav_JANUARY_27

wait