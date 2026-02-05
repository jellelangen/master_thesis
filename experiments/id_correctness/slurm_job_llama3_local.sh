#!/bin/bash
#SBATCH --job-name=id_addsub_llama3_hf
#SBATCH --output=/home2/s4502833/repos/master_thesis/logs/%x_%j.out
#SBATCH --error=/home2/s4502833/repos/master_thesis/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=v100:1

set -euo pipefail

REPO_ROOT="$HOME/repos/master_thesis"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
LOG_DIR="$REPO_ROOT/logs"
HF_HOME="/scratch/$USER/hf"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

echo "=========================================="
echo "Job started at $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Model: $MODEL_ID"
echo "HF cache: $HF_HOME"
echo "=========================================="

module purge || true
module load Python/3.11.5-GCCcore-13.2.0 || true

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export HF_TOKEN=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


# Some stacks read this env var instead
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# Cache to scratch
export HF_HOME="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

echo "Python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Fail fast: verify token + gated access
python - <<'PY'
import os
from huggingface_hub import whoami, hf_hub_download
tok = os.environ["HF_TOKEN"]
print("HF user:", whoami(token=tok)["name"])
p = hf_hub_download("meta-llama/Meta-Llama-3-8B-Instruct", "config.json", token=tok)
print("Gated access OK (config.json):", p)
PY

python -m experiments.id_correctness.run_experiment \
  --config experiments/id_correctness/config.yaml \
  --dataset.name addsub \
  --dataset.split test \
  --dataset.n_samples 100 \
  --few_shot.min_shots 0 \
  --few_shot.max_shots 4 \
  --few_shot.shot_step 2 \
  --model.name "$MODEL_ID" \
  --model.torch_dtype float16 \
  --output.dir outputs/addsub_llama3_hf

LATEST_RUN_DIR="$(ls -1dt outputs/addsub_llama3_hf/* | head -1)"
python -m experiments.id_correctness.analyze_results --results_dir "$LATEST_RUN_DIR"

echo "=========================================="
echo "Job finished at $(date)"
echo "Results: $LATEST_RUN_DIR"
echo "=========================================="
