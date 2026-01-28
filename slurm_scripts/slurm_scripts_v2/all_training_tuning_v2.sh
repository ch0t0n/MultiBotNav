#!/bin/bash
# Submit all v2 arrays from repo root:
#   bash slurm_scripts/slurm_scripts_v2/all_training_tuning_v2.sh

mkdir -p slurm_scripts/slurm_scripts_v2/out

sbatch slurm_scripts/slurm_scripts_v2/tuning_cpu_v2.sh
sbatch slurm_scripts/slurm_scripts_v2/tuning_gpu_v2.sh
sbatch slurm_scripts/slurm_scripts_v2/training_cpu_v2.sh
sbatch slurm_scripts/slurm_scripts_v2/training_gpu_v2.sh