"""
Uncertainty evaluation for Dyck-k using spline geometry features.

THEORY:
    Aleatoric uncertainty is the inherent ambiguity in the task. For Dyck-k prefixes:
    - If stack is empty: k valid next tokens (any open bracket)
    - If stack is non-empty: k+1 valid next tokens (any open + matching close)
    
    This script extracts spline features (q10, softmin, sign_density) from the model's
    gated MLPs and correlates them with the number of valid next tokens. A positive
    correlation would suggest the features capture inherent task ambiguity.

RELEVANT FILES:
    - experiments/dyk/train.py: Training script
    - architectures/utils.py: spline_features_lasttok_softmin function
    - experiments/dyk/evaluate_mixed.py: Similar analysis across multiple k values

CLI ARGUMENTS:
    --checkpoint    Path to model checkpoint (required)
    --n_samples     Number of test samples (default: 500)
    --batch_size    Batch size for inference (default: 64)
    --seed          Random seed (default: 42)
    --plot          Show correlation plots
    --plot_path     Save plots to file

USAGE:
    python -m experiments.dyk.evaluate_uncertainty --checkpoint=models/ckpt_dyck.pt --plot
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
from architectures.utils import attach_gate_hooks, spline_features_lasttok_softmin


def get_valid_next_tokens(pcfg: DyckPCFG, prefix_str: str) -> set:
    """
    Compute the set of valid next tokens for a Dyck prefix.
    
    Valid tokens are:
    - Any open bracket (always valid if not at max depth)
    - The matching close bracket for the most recent unclosed open bracket
    
    Returns:
        Set of valid token IDs
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
    
    # All open brackets are always valid (can start new nesting)
    for i, (open_b, _) in enumerate(pcfg.config.brackets):
        valid_tokens.add(i + 3)  # Token IDs start at 3 for brackets
    
    # If there's an unclosed bracket, its matching close is valid
    if stack:
        last_open = stack[-1]
        matching_close = pcfg._bracket_match[last_open]
        for i, (_, close_b) in enumerate(pcfg.config.brackets):
            if close_b == matching_close:
                valid_tokens.add(pcfg.config.k + i + 3)
                break
    else:
        # Empty stack = balanced so far, EOS would be valid (but we're predicting brackets)
        pass
    
    return valid_tokens


def create_uncertainty_test_set(pcfg: DyckPCFG, n_samples: int, min_len: int = 4, max_len: int = 64):
    """
    Generate test set with known aleatoric uncertainty levels.
    
    For each prefix, compute:
    - The token IDs for the prefix
    - The number of valid next tokens (1 = deterministic, >1 = ambiguous)
    - The set of valid next token IDs
    
    Returns:
        inputs: list of token lists
        n_valid: list of int (number of valid next tokens)
        valid_sets: list of sets (valid token IDs)
        prefix_strs: list of string prefixes
    """
    inputs = []
    n_valid = []
    valid_sets = []
    prefix_strs = []
    
    while len(inputs) < n_samples:
        seq = pcfg.sample()
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        # Take a random prefix (not the full sequence)
        prefix_len = np.random.randint(1, len(seq))
        prefix_str = seq[:prefix_len]
        
        # Compute valid next tokens
        valid = get_valid_next_tokens(pcfg, prefix_str)
        
        # Tokenize prefix with BOS
        tokens = pcfg.tokenize(prefix_str, add_bos=True, add_eos=False)
        
        inputs.append(tokens)
        n_valid.append(len(valid))
        valid_sets.append(valid)
        prefix_strs.append(prefix_str)
    
    return inputs, n_valid, valid_sets, prefix_strs


def extract_spline_features(model, inputs, device, batch_size=64):
    """
    Extract spline features from the model's last layer.
    
    Returns:
        dict with q10, softmin, sign_density arrays of shape [N]
    """
    model.eval()
    handles, cache = attach_gate_hooks(model)
    layer_idx = len(model.blocks) - 1
    
    all_q10 = []
    all_softmin = []
    all_sign_density = []
    all_entropy = []
    all_logits = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting features"):
            batch_inputs = inputs[i:i+batch_size]
            
            # Pad to same length
            max_len = max(len(inp) for inp in batch_inputs)
            x = torch.zeros(len(batch_inputs), max_len, dtype=torch.long, device=device)
            
            for j, inp in enumerate(batch_inputs):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            
            # Forward pass
            logits, _ = model(x)
            
            # Get spline features from last layer
            h_gate = cache[layer_idx]["h_gate"]
            gate_weight = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
            
            # Extract features for each sample's last position
            for j, inp in enumerate(batch_inputs):
                last_pos = len(inp) - 1
                
                # Get single-sample h_gate slice
                h_single = h_gate[j:j+1, last_pos:last_pos+1, :]  # [1, 1, K]
                feats = spline_features_lasttok_softmin(h_single, gate_weight, tau=0.05)
                
                all_q10.append(feats["q10"].item())
                all_softmin.append(feats["softmin"].item())
                all_sign_density.append(feats["sign_density"].item())
                
                # Compute entropy at last position
                last_logits = logits[j, last_pos]
                probs = torch.softmax(last_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                all_entropy.append(entropy)
                all_logits.append(last_logits.cpu().numpy())
    
    for h in handles:
        h.remove()
    
    return {
        "q10": np.array(all_q10),
        "softmin": np.array(all_softmin),
        "sign_density": np.array(all_sign_density),
        "entropy": np.array(all_entropy),
        "logits": np.array(all_logits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/ckpt_dyck.pt")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_path", type=str, default=None)
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
    
    # Create test PCFG
    np.random.seed(args.seed)
    pcfg = DyckPCFG(k=train_args["k"], p_close=train_args["p_close"], seed=args.seed)
    
    print(f"\nGenerating {args.n_samples} test samples...")
    eval_max_len = min(args.max_len, trained_max_len - 1)
    inputs, n_valid, valid_sets, prefix_strs = create_uncertainty_test_set(
        pcfg, args.n_samples, min_len=args.min_len, max_len=eval_max_len
    )
    
    n_valid = np.array(n_valid)
    
    # Extract features
    print("Extracting spline features...")
    features = extract_spline_features(model, inputs, device, args.batch_size)
    
    # Analyze correlations
    print(f"\n{'='*60}")
    print("UNCERTAINTY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nSamples by ambiguity level:")
    for nv in sorted(set(n_valid)):
        count = np.sum(n_valid == nv)
        print(f"  {nv} valid options: {count} samples ({100*count/len(n_valid):.1f}%)")
    
    print(f"\nCorrelations with n_valid (# valid next tokens):")
    for feat_name in ["q10", "softmin", "sign_density", "entropy"]:
        r, p = spearmanr(n_valid, features[feat_name])
        print(f"  {feat_name:15s}: r={r:+.4f} (p={p:.4f})")
    
    print(f"\nCorrelations with prediction entropy:")
    for feat_name in ["q10", "softmin", "sign_density"]:
        r, p = pearsonr(features["entropy"], features[feat_name])
        print(f"  {feat_name:15s}: r={r:+.4f} (p={p:.4f})")
    
    # Compare by ambiguity levels present in data
    unique_nvalid = sorted(set(n_valid))
    
    print(f"\nFeature means by ambiguity level:")
    print(f"  {'Feature':15s} | " + " | ".join([f"{nv} valid" for nv in unique_nvalid]))
    print(f"  {'-'*15} | " + " | ".join(["-"*8 for _ in unique_nvalid]))
    
    for feat_name in ["q10", "softmin", "sign_density", "entropy"]:
        means = []
        for nv in unique_nvalid:
            mask = n_valid == nv
            if mask.sum() > 0:
                means.append(f"{features[feat_name][mask].mean():8.4f}")
            else:
                means.append(f"{'N/A':>8s}")
        print(f"  {feat_name:15s} | " + " | ".join(means))
    
    # Plot if requested
    if args.plot or args.plot_path:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        for ax, feat_name in zip(axes.flat, ["q10", "softmin", "sign_density", "entropy"]):
            # Line plot: mean feature value vs n_valid
            means = []
            stds = []
            for nv in unique_nvalid:
                mask = n_valid == nv
                if mask.sum() > 0:
                    means.append(features[feat_name][mask].mean())
                    stds.append(features[feat_name][mask].std())
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            
            means = np.array(means)
            stds = np.array(stds)
            
            ax.plot(unique_nvalid, means, marker='o', linewidth=2, markersize=8)
            ax.fill_between(unique_nvalid, means - stds, means + stds, alpha=0.2)
            ax.set_xlabel("# valid next tokens")
            ax.set_ylabel(feat_name)
            ax.set_title(f"{feat_name} vs ambiguity level")
            ax.set_xticks(unique_nvalid)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if args.plot_path:
            plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved plot to {args.plot_path}")
        
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
