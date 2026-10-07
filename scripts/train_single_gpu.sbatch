#!/usr/bin/env bash
#SBATCH --job-name=simpledesign
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm/train-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the simpledesign project root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

config="${1:-configs/train_config.yaml}"
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
echo "Training config: ${config}; preparing/validating data before model and W&B initialization"
srun .venv/bin/python -u scripts/train.py --config "$config" --device cuda
