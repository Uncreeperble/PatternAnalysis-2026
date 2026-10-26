#!/bin/bash
# Slurm batch script for COMP3710 Project 2.4 training/evaluation runs.
#
# Usage (from the recognition/VQVAE_PixelCNN_s4882851 directory on Rangpur):
#   sbatch slurm_train.sh vqvae --epochs 100 --out checkpoints/vqvae   # E1
#   sbatch slurm_train.sh prior  --vqvae-checkpoint checkpoints/vqvae/vqvae_best.pt --epochs 50 --out checkpoints/prior  # E4
#   sbatch slurm_train.sh dcgan  --epochs 50 --out checkpoints/dcgan   # E5
# Evaluation runs use predict.py directly, e.g. in an interactive GPU session:
#   srun --partition=comp3710 --account=comp3710 --gres=gpu:1 --mem=32G --pty bash
#   python predict.py --vqvae ... --prior ... --dcgan ... --out outputs
#
# One-off environment setup (once per account, inside an interactive CPU job):
#   pip install --no-cache-dir nibabel scikit-image torchmetrics matplotlib

#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=comp3710
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

mkdir -p logs checkpoints

# All arguments after the script name are forwarded: $1 is the train.py
# --model value, the rest are that model's flags.
MODEL=$1
shift
python train.py --model "$MODEL" "$@"
echo "finished $(date)"
