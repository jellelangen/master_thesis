"""
Mixed Dyck evaluation: samples from Dyck-2 through Dyck-k evaluated on a single model.

THEORY:
    By sampling from multiple Dyck grammars (Dyck-2, Dyck-3, ..., Dyck-k), we get prefixes
    with varying levels of aleatoric uncertainty:
    - Dyck-2 prefix: 2-3 valid next tokens
    - Dyck-4 prefix: 4-5 valid next tokens
    - Dyck-8 prefix: 8-9 valid next tokens
    
    This allows testing whether spline features scale with the number of valid options.
    The model should be trained with --mixed flag to see all grammars during training.

RELEVANT FILES:
    - experiments/dyk/train.py: Use --mixed --k=8 for training
    - architectures/utils.py: spline_features_lasttok function
    - experiments/dyk/evaluate_uncertainty.py: Single-k version

CLI ARGUMENTS:
    --checkpoint      Path to model checkpoint (default: models/dyck8.pt)
    --max_k           Maximum k for Dyck grammars to sample (default: 8)
    --n_samples_per_k Samples per grammar (default: 200)
    --batch_size      Batch size for inference (default: 64)
    --seed            Random seed (default: 54321)
    --plot            Show plots
    --plot_path       Save plots to file
    --layer           Layer to analyze: 'all' for aggregated, or layer index (default: all)

USAGE:
    python -m experiments.dyk.evaluate_mixed --checkpoint=models/dyck_mixed.pt --max_k=8 --plot
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer
from architectures.utils import attach_gate_hooks, spline_features_lasttok


def get_valid_next_tokens_for_k(k: int, prefix_str: str, pcfg: DyckPCFG) -> set:
    """
    Compute valid next tokens for a Dyck-k prefix.
    
    Returns set of valid token IDs (using the pcfg's token encoding).
    """
    # Parse prefix to find unclosed brackets
    stack = []
    for char in prefix_str:
        if char in pcfg._open_brackets:
            stack.append(char)
        elif char in pcfg._close_brackets:
            if stack:
                stack.pop()
    
    valid_tokens = set()
    
    # All k open brackets are valid
    for i in range(k):
        valid_tokens.add(i + 3)  # Token IDs: 3, 4, ..., k+2
    
    # If stack non-empty, matching close is valid
    if stack:
        last_open = stack[-1]
        matching_close = pcfg._bracket_match[last_open]
        for i, (_, close_b) in enumerate(pcfg.config.brackets):
            if close_b == matching_close:
                valid_tokens.add(pcfg.config.k + i + 3)
                break
    
    return valid_tokens


def create_mixed_test_set(max_k: int, n_samples_per_k: int, min_len: int = 4, max_len: int = 32, seed: int = 12345):
    """
    Generate test samples from Dyck-2 through Dyck-max_k.
    
    Uses Dyck-max_k PCFG for tokenization (superset vocabulary).
    
    Returns:
        inputs: list of token lists
        n_valid: list of int
        k_values: list of int (which Dyck-k the sample is from)
        prefix_strs: list of strings
    """
    np.random.seed(seed)
    
    # Use Dyck-max_k for tokenization (largest vocabulary)
    pcfg_max = DyckPCFG(k=max_k, p_close=0.5, seed=seed)
    
    inputs = []
    n_valid = []
    k_values = []
    prefix_strs = []
    
    for k in range(2, max_k + 1):
        # Create PCFG for this k
        pcfg_k = DyckPCFG(k=k, p_close=0.5, seed=seed + k * 1000)
        
        count = 0
        while count < n_samples_per_k:
            seq = pcfg_k.sample()
            if len(seq) < min_len or len(seq) > max_len:
                continue
            
            # Random prefix
            prefix_len = np.random.randint(1, len(seq))
            prefix_str = seq[:prefix_len]
            
            # Compute valid tokens for this k
            valid = get_valid_next_tokens_for_k(k, prefix_str, pcfg_k)
            
            # Tokenize using max_k vocabulary (superset)
            tokens = pcfg_max.tokenize(prefix_str, add_bos=True, add_eos=False)
            
            inputs.append(tokens)
            n_valid.append(len(valid))
            k_values.append(k)
            prefix_strs.append(prefix_str)
            count += 1
    
    return inputs, n_valid, k_values, prefix_strs


def extract_spline_features(model, inputs, device, batch_size=64):
    """
    Extract spline features from all layers of the model.
    
    Returns:
        dict with:
            - 'per_layer': {layer_idx: {q10, softmin, sign_density arrays}}
            - 'aggregated': {q10, softmin, sign_density arrays} (mean across layers)
            - 'entropy': array (from output logits)
    """
    model.eval()
    handles, cache = attach_gate_hooks(model)
    n_layers = len(model.blocks)
    
    # Initialize per-layer storage
    per_layer = {l: {"hardmin": [], "q10": [], "sign_density": [], "lc": []} for l in range(n_layers)}
    all_entropy = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting features"):
            batch_inputs = inputs[i:i+batch_size]
            
            max_len = max(len(inp) for inp in batch_inputs)
            x = torch.zeros(len(batch_inputs), max_len, dtype=torch.long, device=device)
            
            for j, inp in enumerate(batch_inputs):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            
            logits, _ = model(x)
            
            # Extract features for each layer
            for layer_idx in range(n_layers):
                h_gate = cache[layer_idx]["h_gate"]
                gate_weight = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
                
                for j, inp in enumerate(batch_inputs):
                    last_pos = len(inp) - 1
                    h_single = h_gate[j:j+1, last_pos:last_pos+1, :]
                    feats = spline_features_lasttok(h_single, gate_weight, 0.2)

                    per_layer[layer_idx]["hardmin"].append(feats["hardmin"].item())
                    per_layer[layer_idx]["q10"].append(feats["q10"].item())
                    per_layer[layer_idx]["sign_density"].append(feats["sign_density"].item())
                    per_layer[layer_idx]["lc"].append(feats["lc"].item())
            
            # Entropy from logits (only once per sample)
            for j, inp in enumerate(batch_inputs):
                last_pos = len(inp) - 1
                last_logits = logits[j, last_pos]
                probs = torch.softmax(last_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                all_entropy.append(entropy)
    
    for h in handles:
        h.remove()
    
    # Convert to numpy arrays
    for layer_idx in range(n_layers):
        for feat_name in ["hardmin", "q10", "sign_density", "lc"]:
            per_layer[layer_idx][feat_name] = np.array(per_layer[layer_idx][feat_name])

    # Compute aggregated features (mean across layers)
    aggregated = {}
    for feat_name in ["hardmin", "q10", "sign_density", "lc"]:
        stacked = np.stack([per_layer[l][feat_name] for l in range(n_layers)], axis=0)  # [n_layers, n_samples]
        aggregated[feat_name] = stacked.mean(axis=0)  # [n_samples]
    
    return {
        "per_layer": per_layer,
        "aggregated": aggregated,
        "entropy": np.array(all_entropy),
        "n_layers": n_layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/dyck8.pt")
    parser.add_argument("--max_k", type=int, default=8, help="Max k for Dyck grammars (samples Dyck-2 to Dyck-max_k)")
    parser.add_argument("--n_samples_per_k", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_path", type=str, default=None)
    parser.add_argument("--layer", type=str, default="all", 
                        help="Layer to analyze: 'all' for aggregated, or layer index (0, 1, etc.)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    vocab_size = ckpt["vocab_size"]
    trained_max_len = ckpt["model_state_dict"]["pos_emb.weight"].shape[0]
    
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model trained on Dyck-{train_args['k']}, vocab_size={vocab_size}")
    
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
    
    # Generate mixed test set from Dyck-2 to Dyck-max_k
    print(f"\nGenerating {args.n_samples_per_k} samples each from Dyck-2 to Dyck-{args.max_k}...")
    inputs, n_valid, k_values, prefix_strs = create_mixed_test_set(
        args.max_k, args.n_samples_per_k, seed=args.seed
    )
    
    n_valid = np.array(n_valid)
    k_values = np.array(k_values)
    
    print(f"Total samples: {len(inputs)}")
    
    # Extract features
    result = extract_spline_features(model, inputs, device, args.batch_size)
    n_layers = result["n_layers"]
    
    # Select which features to use based on --layer argument
    if args.layer == "all":
        features = result["aggregated"]
        layer_desc = "aggregated (mean across all layers)"
    else:
        layer_idx = int(args.layer)
        if layer_idx >= n_layers:
            print(f"Error: layer {layer_idx} does not exist (model has {n_layers} layers)")
            return
        features = result["per_layer"][layer_idx]
        layer_desc = f"layer {layer_idx}"
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"MIXED DYCK UNCERTAINTY ANALYSIS - {layer_desc}")
    print(f"{'='*70}")
    
    print(f"\nSamples by ambiguity level:")
    unique_nvalid = sorted(set(n_valid))
    for nv in unique_nvalid:
        count = np.sum(n_valid == nv)
        print(f"  {nv} valid options: {count} samples ({100*count/len(n_valid):.1f}%)")
    
    # Correlations for selected layer/aggregated
    print(f"\nCorrelations with n_valid ({layer_desc}):")
    for feat_name in ["hardmin", "q10", "sign_density", "lc"]:
        r, p = spearmanr(n_valid, features[feat_name])
        print(f"  {feat_name:15s}: r={r:+.4f} (p={p:.2e})")
    r, p = spearmanr(n_valid, result["entropy"])
    print(f"  {'entropy':15s}: r={r:+.4f} (p={p:.2e})")
    
    # Per-layer correlation summary
    print(f"\nPer-layer correlations with n_valid:")
    print(f"  {'Layer':<8} | {'hardmin':>10} | {'q10':>10} | {'sign_density':>12} | {'lc':>10}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*10}")
    for l in range(n_layers):
        layer_feats = result["per_layer"][l]
        r_hm, _ = spearmanr(n_valid, layer_feats["hardmin"])
        r_q10, _ = spearmanr(n_valid, layer_feats["q10"])
        r_sd, _ = spearmanr(n_valid, layer_feats["sign_density"])
        r_lc, _ = spearmanr(n_valid, layer_feats["lc"])
        print(f"  Layer {l:<2} | {r_hm:+10.4f} | {r_q10:+10.4f} | {r_sd:+12.4f} | {r_lc:+10.4f}")
    # Aggregated
    r_hm, _ = spearmanr(n_valid, result["aggregated"]["hardmin"])
    r_q10, _ = spearmanr(n_valid, result["aggregated"]["q10"])
    r_sd, _ = spearmanr(n_valid, result["aggregated"]["sign_density"])
    r_lc, _ = spearmanr(n_valid, result["aggregated"]["lc"])
    print(f"  {'Agg':<8} | {r_hm:+10.4f} | {r_q10:+10.4f} | {r_sd:+12.4f} | {r_lc:+10.4f}")
    
    # Plot
    if args.plot or args.plot_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot features for selected layer
        feat_list = ["hardmin", "q10", "lc", "entropy"]
        for ax, feat_name in zip(axes.flat, feat_list):
            if feat_name == "entropy":
                feat_vals = result["entropy"]
            else:
                feat_vals = features[feat_name]

            means = []
            stds = []
            for nv in unique_nvalid:
                mask = n_valid == nv
                if mask.sum() > 0:
                    means.append(feat_vals[mask].mean())
                    stds.append(feat_vals[mask].std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            means = np.array(means)
            stds = np.array(stds)

            ax.plot(unique_nvalid, means, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel("number of valid next tokens")
            ax.set_ylabel(feat_name)
            ax.set_title(f"{feat_name} ({layer_desc})")
            ax.set_xticks(unique_nvalid)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Mixed Dyck-2 to Dyck-{args.max_k} ({layer_desc})", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if args.plot_path:
            plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved plot to {args.plot_path}")
        
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
