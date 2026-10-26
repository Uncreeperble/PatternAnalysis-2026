#!/usr/bin/env bash
# Optional ONE-OFF training helper. Use campaign.py for the complete gated DAG.
# Create logs/ BEFORE sbatch; Slurm opens log files before executing this script.
#   mkdir -p logs
#   sbatch slurm_train.sh vqvae --epochs 100 --out checkpoints/manual_vq
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --job-name=hip_manual
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -eo pipefail
source "${CONDA_BASE:-/home/Student/s4882851/miniconda3}/etc/profile.d/conda.sh"
conda activate torch
set -u
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TORCH_HOME="${HIPMRI_TORCH_HOME:-/home/Student/s4882851/.cache/torch}"
echo "[$(date -Is)] START job=$SLURM_JOB_ID node=$(hostname)"
model="${1:?Expected model: vqvae, prior or dcgan}"
shift
python -u "${SLURM_SUBMIT_DIR}/train.py" --model "$model" --device cuda "$@"
echo "[$(date -Is)] COMPLETE job=$SLURM_JOB_ID"