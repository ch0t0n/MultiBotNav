#!/bin/bash
#SBATCH --job-name=gpu_verify
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --cpus-per-task=2        # Good practice to request a few CPUs
#SBATCH --mem=8G                 # Request 8GB of RAM
#SBATCH --time=00:10:00          # 10 minutes is plenty for this check
#SBATCH --output=logs/gpu_check_%j.log # Saves output to a file with the Job ID

# Exit immediately if a command exits with a non-zero status
set -e

# Load any necessary modules (adjust if Beocat requires specific cuda modules)
# module load cuda/12.4 

# Activate your specific conda environment
# We use 'source' to ensure the shell handles the activation correctly in Slurm
conda_env="robot_env"
source ~/miniconda3/bin/activate $conda_env

echo "Using conda env: $conda_env"

echo -e "\n--- SLURM Info ---"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"

echo -e "\n--- GPU Node Information ---"
nvidia-smi -L

python3 -c "import torch; print('Torch CUDA version:', torch.version.cuda)"

echo -e "\n--- PyTorch Architecture Support ---"
python3 -c "import torch; print(torch.cuda.get_arch_list())"

echo -e "\n--- CUDA Availability Check ---"
python3 -c "import torch; print('Is CUDA available?:', torch.cuda.is_available())"

echo -e "\n--- Current Device Info ---"
python3 -c "import torch; print('Current Device Name:', torch.cuda.get_device_name(0)); print('Compute Capability:', torch.cuda.get_device_capability(0))"

python3 - <<EOF
import torch
x = torch.rand(3,3).cuda()
print("Tensor on GPU:", x.device)
EOF

python3 - <<EOF
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
EOF

