#!/bin/bash
#SBATCH --job-name=id_addsub_100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="$HOME/repos/master_thesis"
cd "$REPO_ROOT"
mkdir -p logs

echo "=========================================="
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "=========================================="

module purge || true
module load Python/3.9.6-GCCcore-11.2.0 || true

source "$REPO_ROOT/venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -c "import sys; print('Python:', sys.executable)"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

export HF_HOME="/scratch/$USER/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

python -m experiments.id_correctness.run_experiment \
  --config experiments/id_correctness/config.yaml \
  --dataset.name addsub \
  --dataset.split test \
  --dataset.n_samples 100 \
  --few_shot.min_shots 0 \
  --few_shot.max_shots 4 \
  --few_shot.shot_step 2 \
  --model.name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model.torch_dtype float16 \
  --output.dir outputs/addsub_medium

# Optional: run analysis on the same node
python -m experiments.id_correctness.analyze_results \
  --results_dir "$(ls -1dt outputs/addsub_medium/* | head -1)"

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
