# ID vs Correctness Experiment

Replicates **Figure 6** from ["Reasoning in Large Language Models: A Geometric Perspective"](https://arxiv.org/abs/2407.02678) using the intrinsic dimension (ID) estimation approach from [SplineLLM](https://github.com/RandallBalestriero/SplineLLM).

## Background

The experiment investigates how the **intrinsic dimension** of attention embeddings changes with few-shot prompting and correlates with **answer correctness** on GSM8K math problems.

**Key findings from the paper:**
- Adding relevant few-shot examples increases ID in final layers
- ID increase in final layers correlates with correct answers
- Random tokens increase ID in first layers but not reasoning ability

## Quick Start

### 1. Configure the experiment

Edit `config.yaml` to set:
- Model (default: LLaMA3-8B)
- Number of samples
- Shot range (0-10)
- Output directory

### 2. Run locally (small test)

Since we updated `config.yaml` to use TinyLlama by default, you can just run:

```bash
python -m experiments.id_correctness.run_experiment
```

### 3. Run on HPC

```bash
# Edit slurm_job.sh with your partition/GPU settings
sbatch experiments/id_correctness/slurm_job.sh
```

### 4. Analyze results

```bash
python -m experiments.id_correctness.analyze_results \
    --results_dir results/id_correctness/TIMESTAMP
```

## Output Files

After running, the output directory contains:

```
results/id_correctness/TIMESTAMP/
├── config.yaml          # Copy of config used
├── results.json         # Full results per sample
├── summary.json         # Aggregated statistics
├── figure_6_left.png    # ID change by layer (correct vs incorrect)
├── figure_6_right.png   # Accuracy vs final layer ID change
├── accuracy_by_shots.png
├── id_heatmap.png
└── correlation_by_layer.png
```

## Configuration Reference

```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  torch_dtype: "float16"  # float16, bfloat16, float32
  device_map: "auto"

dataset:
  n_samples: 500          # null for all samples
  
few_shot:
  min_shots: 0
  max_shots: 10
  shot_step: 1            # 1=all, 2=0,2,4,6,8,10

intrinsic_dim:
  eps_factor: 0.1         # ID threshold: eps = max_attn * eps_factor

output:
  dir: "results/id_correctness"
```

## CLI Overrides

Override any config value via command line:

```bash
python -m experiments.id_correctness.run_experiment \
    --config config.yaml \
    --model.name "different/model" \
    --dataset.n_samples 100
```
