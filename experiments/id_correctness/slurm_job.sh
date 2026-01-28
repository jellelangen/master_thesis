#!/bin/bash
#SBATCH --job-name=id_correctness
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# TODO: Add partition and GPU allocation for your HPC
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:a100:1

# ============================================================
# ID vs Correctness Experiment - SLURM Job Script
# Modify the SBATCH directives above for your HPC environment
# ============================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=========================================="

# Activate virtual environment (modify path as needed)
# source /path/to/venv/bin/activate

# Or use module system (modify for your HPC)
# module load Python/3.10-GCCcore-12.2.0
# module load CUDA/12.0.0

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)/../..}"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run experiment with config
# Override config values via command line if needed
python -m experiments.id_correctness.run_experiment \
    --config experiments/id_correctness/config.yaml \
    "$@"

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
