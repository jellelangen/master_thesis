"""
Main Experiment Script: ID Change vs Answer Correctness.

Recreates Figure 6 from "Reasoning in Large Language Models: A Geometric Perspective"
(arXiv:2407.02678) by measuring how intrinsic dimension changes with few-shot prompting
correlate with answer correctness.

USAGE:
    python -m experiments.id_correctness.run_experiment --config experiments/id_correctness/config.yaml
    
    # Override config values via CLI:
    python -m experiments.id_correctness.run_experiment --config config.yaml --model.name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset.n_samples 10
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict

from transformers import AutoTokenizer, AutoModelForCausalLM

from experiments.id_correctness.data_loader import GSM8KDataLoader


@dataclass
class ExperimentResult:
    """Results for a single sample across all shot counts."""
    sample_idx: int
    question: str
    ground_truth: float
    
    # Per-shot results: Dict[n_shots -> result]
    predictions: Dict[int, Optional[float]]
    correctness: Dict[int, bool]
    generations: Dict[int, str]
    
    # ID per layer per shot: Dict[n_shots -> List[float] per layer]
    id_per_layer: Dict[int, List[float]]
    
    # ID change relative to 0-shot baseline
    id_change_per_layer: Dict[int, List[float]]


def compute_intrinsic_dim_per_layer(
    attention_weights: Tuple[torch.Tensor, ...],
    eps_factor: float = 0.1,
) -> np.ndarray:
    """
    Compute intrinsic dimension from attention weights per layer.
    
    ID is defined as the number of tokens with attention > eps, summed across heads.
    We compute this for the LAST token position (the prediction position).
    
    Args:
        attention_weights: Tuple of [batch, n_heads, seq_len, seq_len] per layer
        eps_factor: Threshold factor: eps = max_attn * eps_factor
    
    Returns:
        id_per_layer: [n_layers] numpy array of intrinsic dimensions
    """
    id_per_layer = []
    
    for layer_attn in attention_weights:
        # layer_attn: [B, n_heads, seq_len, seq_len]
        # Get attention TO the last token FROM all other tokens
        attn_last = layer_attn[:, :, -1, :]  # [B, n_heads, seq_len]
        
        # Compute threshold per head: eps = max_attn * eps_factor
        max_attn = attn_last.max(dim=-1, keepdim=True).values  # [B, n_heads, 1]
        eps = max_attn * eps_factor
        
        # Count tokens with attention > eps, sum across heads
        id_count = (attn_last > eps).float().sum(dim=(-1, -2))  # [B]
        
        id_per_layer.append(id_count.mean().item())  # Average over batch
    
    return np.array(id_per_layer)


def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if overrides:
        for key, value in overrides.items():
            # Handle nested keys like "model.name"
            parts = key.split(".")
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            
            # Try to parse value as YAML for proper typing
            try:
                parsed_value = yaml.safe_load(value)
            except:
                parsed_value = value
            d[parts[-1]] = parsed_value
    
    return config


def run_experiment(config: Dict) -> List[ExperimentResult]:
    """
    Run the full ID vs correctness experiment.
    
    Args:
        config: Experiment configuration dictionary
    
    Returns:
        List of ExperimentResult objects
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load model and tokenizer
    model_name = config["model"]["name"]
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine torch dtype
    dtype_str = config["model"].get("torch_dtype", "float16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=config["model"].get("device_map", "auto"),
        output_attentions=True,
    )
    model.eval()
    
    # Load dataset
    print("Loading GSM8K dataset...")
    data_loader = GSM8KDataLoader(
        subset=config["dataset"].get("subset", "main"),
        n_samples=config["dataset"].get("n_samples"),
        seed=seed,
    )
    print(f"Loaded {len(data_loader)} test samples")
    
    # Shot configuration
    min_shots = config["few_shot"].get("min_shots", 0)
    max_shots = config["few_shot"].get("max_shots", 10)
    shot_step = config["few_shot"].get("shot_step", 1)
    shot_counts = list(range(min_shots, max_shots + 1, shot_step))
    
    print(f"Testing with shot counts: {shot_counts}")
    
    # ID configuration
    eps_factor = config["intrinsic_dim"].get("eps_factor", 0.1)
    
    # Run experiment
    results = []
    
    for sample_idx in tqdm(range(len(data_loader)), desc="Processing samples"):
        sample = data_loader.test_samples[sample_idx]
        
        result = ExperimentResult(
            sample_idx=sample_idx,
            question=sample.question,
            ground_truth=sample.numeric_answer,
            predictions={},
            correctness={},
            generations={},
            id_per_layer={},
            id_change_per_layer={},
        )
        
        baseline_id = None  # 0-shot baseline
        
        for n_shots in shot_counts:
            # Get prompt
            prompt, _ = data_loader.get_prompt_for_sample(sample_idx, n_shots)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate with attention outputs
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy for reproducibility
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
                
                # Get attention weights from forward pass (not generation)
                # We need to run a forward pass to get attentions
                forward_outputs = model(
                    **inputs,
                    output_attentions=True,
                )
                attention_weights = forward_outputs.attentions
            
            # Decode generation
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            generation = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract answer and check correctness
            predicted = data_loader.extract_answer_from_generation(generation)
            correct = data_loader.check_correctness(predicted, sample.numeric_answer)
            
            # Compute ID per layer
            id_per_layer = compute_intrinsic_dim_per_layer(attention_weights, eps_factor)
            
            # Store results
            result.predictions[n_shots] = predicted
            result.correctness[n_shots] = correct
            result.generations[n_shots] = generation
            result.id_per_layer[n_shots] = id_per_layer.tolist()
            
            # Compute ID change relative to baseline
            if n_shots == min_shots:
                baseline_id = id_per_layer
                result.id_change_per_layer[n_shots] = [0.0] * len(id_per_layer)
            else:
                id_change = id_per_layer - baseline_id
                result.id_change_per_layer[n_shots] = id_change.tolist()
        
        results.append(result)
    
    return results


def save_results(results: List[ExperimentResult], output_dir: Path, config: Dict):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save results as JSON
    results_data = [asdict(r) for r in results]
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Save summary statistics
    summary = compute_summary_statistics(results, config)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")


def compute_summary_statistics(results: List[ExperimentResult], config: Dict) -> Dict:
    """Compute summary statistics for the experiment."""
    min_shots = config["few_shot"].get("min_shots", 0)
    max_shots = config["few_shot"].get("max_shots", 10)
    shot_step = config["few_shot"].get("shot_step", 1)
    shot_counts = list(range(min_shots, max_shots + 1, shot_step))
    
    n_layers = len(results[0].id_per_layer[min_shots]) if results else 0
    
    summary = {
        "n_samples": len(results),
        "n_layers": n_layers,
        "shot_counts": shot_counts,
        "accuracy_by_shots": {},
        "mean_id_by_shots_and_layer": {},
        "mean_id_change_correct_vs_incorrect": {},
    }
    
    for n_shots in shot_counts:
        # Accuracy
        correct_count = sum(1 for r in results if r.correctness.get(n_shots, False))
        summary["accuracy_by_shots"][n_shots] = correct_count / len(results) if results else 0
        
        # Mean ID per layer
        id_arrays = [r.id_per_layer.get(n_shots, []) for r in results]
        if id_arrays and id_arrays[0]:
            mean_id = np.mean(id_arrays, axis=0).tolist()
            summary["mean_id_by_shots_and_layer"][n_shots] = mean_id
        
        # ID change for correct vs incorrect (for shots > 0)
        if n_shots > min_shots:
            correct_changes = [
                r.id_change_per_layer[n_shots]
                for r in results
                if r.correctness.get(n_shots, False)
            ]
            incorrect_changes = [
                r.id_change_per_layer[n_shots]
                for r in results
                if not r.correctness.get(n_shots, False)
            ]
            
            if correct_changes:
                mean_correct = np.mean(correct_changes, axis=0).tolist()
            else:
                mean_correct = [0.0] * n_layers
            
            if incorrect_changes:
                mean_incorrect = np.mean(incorrect_changes, axis=0).tolist()
            else:
                mean_incorrect = [0.0] * n_layers
            
            summary["mean_id_change_correct_vs_incorrect"][n_shots] = {
                "correct": mean_correct,
                "incorrect": mean_incorrect,
            }
    
    return summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ID vs Correctness experiment (Figure 6 replication)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/id_correctness/config.yaml",
        help="Path to config YAML file",
    )
    
    # Allow arbitrary config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse overrides from unknown args (format: --key.subkey value)
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1
    
    return args, overrides


def main():
    """Main entry point."""
    args, overrides = parse_args()
    
    # Load config
    config = load_config(args.config, overrides)
    
    print("=" * 60)
    print("ID vs Correctness Experiment")
    print("Replicating Figure 6 from arXiv:2407.02678")
    print("=" * 60)
    print(f"Config: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
    print()
    
    # Run experiment
    results = run_experiment(config)
    
    # Save results
    output_dir = Path(config["output"]["dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    
    save_results(results, output_dir, config)
    
    # Print summary
    summary = compute_summary_statistics(results, config)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples: {summary['n_samples']}")
    print(f"Layers: {summary['n_layers']}")
    print("\nAccuracy by shot count:")
    for n_shots, acc in summary["accuracy_by_shots"].items():
        print(f"  {n_shots}-shot: {acc:.2%}")
    
    print("\nRun analyze_results.py to generate plots.")


if __name__ == "__main__":
    main()
