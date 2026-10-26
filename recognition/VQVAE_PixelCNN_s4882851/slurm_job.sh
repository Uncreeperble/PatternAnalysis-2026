#!/usr/bin/env bash
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

# Submission must create logs/ first. No --mem: this cluster advertises
# synthetic 1-MB Slurm memory resources rather than physical node RAM.
set -eo pipefail
echo "[$(date -Is)] START job=${SLURM_JOB_ID:-local} node=$(hostname) task=$*"
source "${CONDA_BASE:-/home/Student/s4882851/miniconda3}/etc/profile.d/conda.sh"
conda activate torch
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export TORCH_HOME="${HIPMRI_TORCH_HOME:-/home/Student/s4882851/.cache/torch}"

child=""
heartbeat=""
cleanup() {
    code=$?
    trap - EXIT
    if [[ -n "$heartbeat" ]]; then kill "$heartbeat" 2>/dev/null || true; fi
    if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child" 2>/dev/null || true
    fi
    echo "[$(date -Is)] EXIT job=${SLURM_JOB_ID:-local} code=$code"
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

python -u "${SLURM_SUBMIT_DIR}/campaign.py" run "$@" &
child=$!
(
    while kill -0 "$child" 2>/dev/null; do
        echo "[$(date -Is)] HEARTBEAT job=${SLURM_JOB_ID:-local} child=$child alive"
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
        fi
        sleep 30
    done
) &
heartbeat=$!
wait "$child"