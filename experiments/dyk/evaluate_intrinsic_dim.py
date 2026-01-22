"""
Intrinsic Dimension vs Correctness Analysis for Dyck-k Transformers.

THEORY:
    This script replicates findings from "Reasoning in Large Language Models: 
    A Geometric Perspective" (arXiv:2407.02678v1) using Dyck languages.
    
    The paper shows that intrinsic dimension (ID) at the final layer correlates
    with answer correctness:
    
    ID_ε(i) = Σ_h Σ_j 1[Attn_h(i,j) > ε]
    
    - ID is the count of tokens with attention weight above threshold ε
    - Summed across all attention heads
    - Higher ID → more expressiveness → higher probability of correct answer
    
    In Dyck-k, correctness is deterministic: predicting the right closing bracket.
    We analyze whether higher ID correlates with correct predictions.

RELEVANT FILES:
    - experiments/dyk/evaluate.py: Basic evaluation pattern
    - architectures/transformers.py: SplineTransformer with attention extraction
    - architectures/utils.py: Existing compute_intrinsic_dim function

CLI ARGUMENTS:
    --checkpoint    Path to model checkpoint
    --n_samples     Number of test samples (default: 500)
    --eps           Attention threshold for ID (default: 0.1)
    --test_k        Override k for testing (use higher k to induce OOD errors)
    --plot          Enable visualization
    --output_dir    Directory for plots (default: results/intrinsic_dim)
    --seed          Random seed (default: 42)

USAGE:
    # In-distribution testing (model achieves high accuracy)
    python -m experiments.dyk.evaluate_intrinsic_dim \
        --checkpoint=models/dyck_mixed.pt --n_samples=500 --plot
    
    # OOD testing with higher k to induce errors (for ID vs correctness analysis)
    python -m experiments.dyk.evaluate_intrinsic_dim \
        --checkpoint=models/dyck_2to6.pt --test_k=8 --n_samples=500 --plot
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy import stats

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer


def compute_intrinsic_dim_from_attention(attn_weights, eps=0.1):
    """
    Compute intrinsic dimension from attention weights per layer.
    
    ID is defined as the number of tokens with attention > eps, summed across heads.
    We compute this for the LAST token position (the prediction position).
    
    Args:
        attn_weights: list of [B, H, T, T] attention tensors per layer
        eps: threshold for counting influential tokens
        
    Returns:
        id_per_layer: [n_layers, B] numpy array of intrinsic dimensions
    """
    n_layers = len(attn_weights)
    B = attn_weights[0].shape[0]
    
    id_per_layer = np.zeros((n_layers, B))
    
    for layer_idx, attn in enumerate(attn_weights):
        # attn: [B, H, T, T] - attention from each position to all previous
        # We want attention FROM the last token TO all previous tokens
        T = attn.shape[-1]
        last_attn = attn[:, :, -1, :]  # [B, H, T] - last token's attention
        
        # Count tokens with attention > eps, sum across heads
        # ID = Σ_h Σ_j 1[Attn(h, last, j) > eps]
        above_thresh = (last_attn > eps).float()  # [B, H, T]
        id_values = above_thresh.sum(dim=(1, 2))  # [B]
        
        id_per_layer[layer_idx] = id_values.cpu().numpy()
    
    return id_per_layer


def create_diverse_test_set(pcfg, n_samples, min_len=4, max_len=64):
    """
    Generate test samples with diverse properties.
    
    Returns:
        inputs: list of token lists (prefix without last bracket)
        targets: list of target token IDs
        metadata: dict with lengths, depths, etc.
    """
    inputs = []
    targets = []
    lengths = []
    depths = []
    full_seqs = []
    
    while len(inputs) < n_samples:
        seq, depth = pcfg.sample_with_depth()
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        tokens = pcfg.tokenize(seq, add_bos=True, add_eos=False)
        
        # Input: all but last token; Target: last token
        inputs.append(tokens[:-1])
        targets.append(tokens[-1])
        lengths.append(len(seq))
        depths.append(depth)
        full_seqs.append(seq)
    
    metadata = {
        "lengths": np.array(lengths),
        "depths": np.array(depths),
        "full_seqs": full_seqs,
    }
    
    return inputs, targets, metadata


def evaluate_with_intrinsic_dim(model, inputs, targets, device, eps=0.1, batch_size=32):
    """
    Evaluate model and extract intrinsic dimension per sample.
    
    Returns:
        results: dict with predictions, correctness, ID per layer
    """
    model.eval()
    
    all_correct = []
    all_preds = []
    all_id_per_layer = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Pad batch to same length
            max_len = max(len(inp) for inp in batch_inputs)
            x = torch.zeros(len(batch_inputs), max_len, dtype=torch.long, device=device)
            
            for j, inp in enumerate(batch_inputs):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            
            # Forward pass - returns (logits, attn_weights_per_layer)
            logits, attn_all = model(x)
            
            # Get predictions and correctness for last position of each input
            batch_preds = []
            batch_correct = []
            for j, inp in enumerate(batch_inputs):
                last_pos = len(inp) - 1
                pred = logits[j, last_pos].argmax().item()
                target = batch_targets[j]
                batch_preds.append(pred)
                batch_correct.append(pred == target)
            
            all_preds.extend(batch_preds)
            all_correct.extend(batch_correct)
            
            # Compute ID from attention
            id_per_layer = compute_intrinsic_dim_from_attention(attn_all, eps=eps)
            all_id_per_layer.append(id_per_layer)
    
    # Stack ID arrays
    id_per_layer = np.concatenate(all_id_per_layer, axis=1)  # [n_layers, N]
    
    return {
        "predictions": np.array(all_preds),
        "correct": np.array(all_correct),
        "id_per_layer": id_per_layer,
    }


def analyze_id_correctness(results, metadata):
    """
    Analyze correlation between intrinsic dimension and correctness.
    
    Returns:
        analysis: dict with statistics per layer
    """
    correct = results["correct"]
    id_per_layer = results["id_per_layer"]
    n_layers = id_per_layer.shape[0]
    
    analysis = {
        "n_samples": len(correct),
        "accuracy": correct.mean(),
        "per_layer": [],
    }
    
    for layer_idx in range(n_layers):
        id_vals = id_per_layer[layer_idx]
        
        # Point-biserial correlation (continuous vs binary)
        corr, p_corr = stats.pointbiserialr(correct, id_vals)
        
        # T-test: ID for correct vs incorrect
        id_correct = id_vals[correct]
        id_incorrect = id_vals[~correct]
        
        if len(id_incorrect) > 0 and len(id_correct) > 0:
            t_stat, p_ttest = stats.ttest_ind(id_correct, id_incorrect)
        else:
            t_stat, p_ttest = np.nan, np.nan
        
        layer_stats = {
            "layer": layer_idx,
            "id_mean": id_vals.mean(),
            "id_std": id_vals.std(),
            "id_mean_correct": id_correct.mean() if len(id_correct) > 0 else np.nan,
            "id_mean_incorrect": id_incorrect.mean() if len(id_incorrect) > 0 else np.nan,
            "correlation": corr,
            "corr_pvalue": p_corr,
            "ttest_stat": t_stat,
            "ttest_pvalue": p_ttest,
        }
        analysis["per_layer"].append(layer_stats)
    
    # Correlation with sequence length
    lengths = metadata["lengths"]
    final_id = id_per_layer[-1]
    length_corr, length_p = stats.pearsonr(lengths, final_id)
    analysis["length_id_corr"] = length_corr
    analysis["length_id_pvalue"] = length_p
    
    return analysis


def print_analysis(analysis):
    """Print analysis results to console."""
    print("\n" + "=" * 70)
    print("INTRINSIC DIMENSION VS CORRECTNESS ANALYSIS")
    print("=" * 70)
    
    n_correct = int(analysis['accuracy'] * analysis['n_samples'])
    n_incorrect = analysis['n_samples'] - n_correct
    
    print(f"\nSamples: {analysis['n_samples']}")
    print(f"  Correct:   {n_correct:>5} ({analysis['accuracy']*100:.2f}%)")
    print(f"  Incorrect: {n_incorrect:>5} ({(1-analysis['accuracy'])*100:.2f}%)")
    print(f"Overall accuracy: {analysis['accuracy']:.4f}")
    
    print(f"\nSequence length vs final-layer ID correlation: r={analysis['length_id_corr']:.4f} (p={analysis['length_id_pvalue']:.4e})")
    
    print("\n" + "-" * 70)
    print("Per-Layer Analysis:")
    print("-" * 70)
    print(f"{'Layer':>6} {'ID Mean':>10} {'ID Std':>10} {'Correct':>10} {'Incorrect':>10} {'Corr':>8} {'p-value':>12}")
    print("-" * 70)
    
    for layer_stats in analysis["per_layer"]:
        print(f"{layer_stats['layer']:>6} "
              f"{layer_stats['id_mean']:>10.2f} "
              f"{layer_stats['id_std']:>10.2f} "
              f"{layer_stats['id_mean_correct']:>10.2f} "
              f"{layer_stats['id_mean_incorrect']:>10.2f} "
              f"{layer_stats['correlation']:>8.4f} "
              f"{layer_stats['corr_pvalue']:>12.4e}")
    
    print("\n" + "=" * 70)
    final_layer = analysis["per_layer"][-1]
    print(f"FINAL LAYER CORRELATION: r = {final_layer['correlation']:.4f} (p = {final_layer['corr_pvalue']:.4e})")
    print(f"T-test (correct vs incorrect ID): t = {final_layer['ttest_stat']:.4f}, p = {final_layer['ttest_pvalue']:.4e}")
    
    if final_layer['correlation'] > 0 and final_layer['corr_pvalue'] < 0.05:
        print("\n>>> CONFIRMED: Higher ID at final layer correlates with correct predictions!")
    elif final_layer['correlation'] > 0:
        print("\n>>> Positive correlation trend, but not statistically significant")
    else:
        print("\n>>> No positive correlation found (may need more samples or different model)")
    print("=" * 70)


def plot_results(results, metadata, analysis, output_dir):
    """Generate and save visualization plots."""
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    correct = results["correct"]
    id_per_layer = results["id_per_layer"]
    n_layers = id_per_layer.shape[0]
    lengths = metadata["lengths"]
    
    # 1. Box plot: ID by correctness per layer
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, ax in enumerate(axes):
        id_vals = id_per_layer[layer_idx]
        data = [id_vals[correct], id_vals[~correct]]
        bp = ax.boxplot(data, labels=["Correct", "Incorrect"], patch_artist=True)
        bp["boxes"][0].set_facecolor("lightgreen")
        bp["boxes"][1].set_facecolor("lightcoral")
        ax.set_title(f"Layer {layer_idx}")
        ax.set_ylabel("Intrinsic Dimension")
        
        # Add correlation as text
        layer_stats = analysis["per_layer"][layer_idx]
        ax.text(0.05, 0.95, f"r={layer_stats['correlation']:.3f}",
                transform=ax.transAxes, va="top", fontsize=10)
    
    plt.suptitle("Intrinsic Dimension by Prediction Correctness", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "id_by_correctness.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'id_by_correctness.png'}")
    
    # 2. Scatter: Final-layer ID vs sequence length, colored by correctness
    fig, ax = plt.subplots(figsize=(10, 6))
    final_id = id_per_layer[-1]
    
    scatter_correct = ax.scatter(lengths[correct], final_id[correct], 
                                  c="green", alpha=0.6, label="Correct", s=30)
    scatter_incorrect = ax.scatter(lengths[~correct], final_id[~correct],
                                    c="red", alpha=0.6, label="Incorrect", s=30)
    
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Intrinsic Dimension (Final Layer)")
    ax.set_title(f"ID vs Sequence Length (r={analysis['length_id_corr']:.3f})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "id_vs_length.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'id_vs_length.png'}")
    
    # 3. Layer-wise correlation bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    layer_indices = [s["layer"] for s in analysis["per_layer"]]
    correlations = [s["correlation"] for s in analysis["per_layer"]]
    
    colors = ["green" if c > 0 else "red" for c in correlations]
    ax.bar(layer_indices, correlations, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation (ID vs Correctness)")
    ax.set_title("ID-Correctness Correlation by Layer")
    ax.set_xticks(layer_indices)
    
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_by_layer.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'correlation_by_layer.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze intrinsic dimension vs correctness")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of test samples")
    parser.add_argument("--eps", type=float, default=0.1, help="Attention threshold for ID")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--test_k", type=int, default=None,
                        help="Override k for test data. Use higher k than training to induce OOD errors.")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output_dir", type=str, default="results/intrinsic_dim")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train a model first with: python -m experiments.dyk.train")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    vocab_size = ckpt["vocab_size"]
    trained_max_len = ckpt["model_state_dict"]["pos_emb.weight"].shape[0]
    
    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Model config: k={train_args['k']}, d_model={train_args['d_model']}, "
          f"n_layers={train_args['n_layers']}, n_heads={train_args['n_heads']}")
    
    # Cap max_len to model capacity
    eval_max_len = min(args.max_len, trained_max_len - 1)
    
    # Create model
    model = SplineTransformer(
        vocab_size=vocab_size,
        d_model=train_args["d_model"],
        n_heads=train_args["n_heads"],
        d_ff=train_args["d_ff"],
        n_layers=train_args["n_layers"],
        max_len=trained_max_len,
        pad_idx=0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Create test PCFG (optionally with different k for OOD testing)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    test_k = args.test_k if args.test_k is not None else train_args["k"]
    if test_k != train_args["k"]:
        print(f"\n[OOD MODE] Testing with k={test_k} (model trained on k={train_args['k']})")
        print("This introduces OOD bracket types to induce prediction errors.")
    
    pcfg = DyckPCFG(k=test_k, p_close=train_args["p_close"], seed=args.seed)
    
    print(f"\nGenerating {args.n_samples} test samples (len {args.min_len}-{eval_max_len})...")
    inputs, targets, metadata = create_diverse_test_set(
        pcfg, args.n_samples, min_len=args.min_len, max_len=eval_max_len
    )
    
    print(f"Evaluating with ID threshold eps={args.eps}...")
    results = evaluate_with_intrinsic_dim(
        model, inputs, targets, device, eps=args.eps, batch_size=args.batch_size
    )
    
    # Analyze
    analysis = analyze_id_correctness(results, metadata)
    print_analysis(analysis)
    
    # Plot
    if args.plot:
        plot_results(results, metadata, analysis, args.output_dir)


if __name__ == "__main__":
    main()
