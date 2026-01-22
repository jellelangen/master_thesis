"""
Epistemic uncertainty evaluation for Dyck-k using spline geometry features.

THEORY:
    Epistemic uncertainty arises from the model's lack of knowledge. We test this by:
    1. Training on Dyck-2 through Dyck-k_train (in-distribution)
    2. Evaluating on Dyck-(k_train+1) through Dyck-k_test (out-of-distribution)
    
    OOD samples contain bracket types the model has never seen during training.
    If spline features capture epistemic uncertainty, they should differ between ID and OOD.
    
    Features analyzed:
    - q10: 10th percentile distance to hyperplane (distance-based)
    - softmin: Soft minimum distance (distance-based)  
    - sign_density: Fraction of positive activations (activation-based)
    - entropy: Prediction entropy from output logits

RELEVANT FILES:
    - experiments/dyk/train.py: Use --mixed --max_k_train=6 for OOD setup
    - experiments/dyk/classify_ood.py: Binary classifier using these features
    - architectures/utils.py: Feature extraction functions

CLI ARGUMENTS:
    --checkpoint    Path to model checkpoint (required)
    --k_train       Max k seen during training (default: 6)
    --k_test        Max k for testing, OOD = k_train+1 to k_test (default: 8)
    --n_samples     Samples per distribution (default: 300)
    --batch_size    Batch size (default: 64)
    --seed          Random seed (default: 77777)
    --plot          Show histogram plots
    --plot_path     Save plots to file

USAGE:
    # First train with held-out grammars:
    python -m experiments.dyk.train --k=8 --mixed --max_k_train=6 --save_path="models/dyck_2to6.pt"
    
    # Then evaluate:
    python -m experiments.dyk.evaluate_epistemic --checkpoint=models/dyck_2to6.pt --k_train=6 --k_test=8 --plot
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib.pyplot as plt

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer
from architectures.utils import attach_gate_hooks, spline_features_lasttok_softmin


def create_test_samples(pcfg: DyckPCFG, pcfg_tokenizer: DyckPCFG, n_samples: int, 
                        min_len: int = 4, max_len: int = 32, seed: int = 12345):
    """
    Generate test samples from a specific Dyck-k grammar.
    
    Args:
        pcfg: PCFG to sample from
        pcfg_tokenizer: PCFG with full vocabulary for tokenization
        n_samples: number of samples
        min_len, max_len: sequence length constraints
        seed: random seed
        
    Returns:
        inputs: list of token lists
        prefix_strs: list of string prefixes
    """
    np.random.seed(seed)
    
    inputs = []
    prefix_strs = []
    
    while len(inputs) < n_samples:
        seq = pcfg.sample()
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        # Random prefix
        prefix_len = np.random.randint(1, len(seq))
        prefix_str = seq[:prefix_len]
        
        # Tokenize using full vocabulary
        tokens = pcfg_tokenizer.tokenize(prefix_str, add_bos=True, add_eos=False)
        
        inputs.append(tokens)
        prefix_strs.append(prefix_str)
    
    return inputs, prefix_strs


def extract_spline_features(model, inputs, device, batch_size=64):
    """Extract spline features from all layers."""
    model.eval()
    handles, cache = attach_gate_hooks(model)
    n_layers = len(model.blocks)
    
    per_layer = {l: {"q10": [], "softmin": [], "sign_density": []} for l in range(n_layers)}
    all_entropy = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting features"):
            batch_inputs = inputs[i:i+batch_size]
            
            max_len = max(len(inp) for inp in batch_inputs)
            x = torch.zeros(len(batch_inputs), max_len, dtype=torch.long, device=device)
            
            for j, inp in enumerate(batch_inputs):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            
            logits, _ = model(x)
            
            for layer_idx in range(n_layers):
                h_gate = cache[layer_idx]["h_gate"]
                gate_weight = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
                
                for j, inp in enumerate(batch_inputs):
                    last_pos = len(inp) - 1
                    h_single = h_gate[j:j+1, last_pos:last_pos+1, :]
                    feats = spline_features_lasttok_softmin(h_single, gate_weight, tau=0.05)
                    
                    per_layer[layer_idx]["q10"].append(feats["q10"].item())
                    per_layer[layer_idx]["softmin"].append(feats["softmin"].item())
                    per_layer[layer_idx]["sign_density"].append(feats["sign_density"].item())
            
            for j, inp in enumerate(batch_inputs):
                last_pos = len(inp) - 1
                last_logits = logits[j, last_pos]
                probs = torch.softmax(last_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                all_entropy.append(entropy)
    
    for h in handles:
        h.remove()
    
    # Convert to numpy
    for layer_idx in range(n_layers):
        for feat_name in ["q10", "softmin", "sign_density"]:
            per_layer[layer_idx][feat_name] = np.array(per_layer[layer_idx][feat_name])
    
    # Aggregated features
    aggregated = {}
    for feat_name in ["q10", "softmin", "sign_density"]:
        stacked = np.stack([per_layer[l][feat_name] for l in range(n_layers)], axis=0)
        aggregated[feat_name] = stacked.mean(axis=0)
    
    return {
        "per_layer": per_layer,
        "aggregated": aggregated,
        "entropy": np.array(all_entropy),
        "n_layers": n_layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k_train", type=int, default=6, help="Max k seen during training")
    parser.add_argument("--k_test", type=int, default=8, help="Max k for testing (k_train+1 to k_test are OOD)")
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=77777)
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
    print(f"Model vocab_size={vocab_size}, trained with k={train_args['k']}")
    print(f"Testing: ID = Dyck-2 to Dyck-{args.k_train}, OOD = Dyck-{args.k_train+1} to Dyck-{args.k_test}")
    
    # Check vocab compatibility
    required_vocab = 3 + 2 * args.k_test
    if vocab_size < required_vocab:
        print(f"Error: model vocab_size={vocab_size} but need {required_vocab} for Dyck-{args.k_test}")
        print("Train with --k=8 to have full vocabulary")
        return
    
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
    
    # Tokenizer uses full vocabulary
    pcfg_tokenizer = DyckPCFG(k=args.k_test, p_close=0.5, seed=args.seed)
    
    # Generate ID samples (Dyck-2 to Dyck-k_train)
    print(f"\nGenerating {args.n_samples} ID samples (Dyck-2 to Dyck-{args.k_train})...")
    id_inputs = []
    id_k_values = []
    for k in range(2, args.k_train + 1):
        pcfg = DyckPCFG(k=k, p_close=0.5, seed=args.seed + k)
        samples_per_k = args.n_samples // (args.k_train - 1)
        inputs, _ = create_test_samples(pcfg, pcfg_tokenizer, samples_per_k, seed=args.seed + k)
        id_inputs.extend(inputs)
        id_k_values.extend([k] * len(inputs))
    
    # Generate OOD samples (Dyck-k_train+1 to Dyck-k_test)
    print(f"Generating {args.n_samples} OOD samples (Dyck-{args.k_train+1} to Dyck-{args.k_test})...")
    ood_inputs = []
    ood_k_values = []
    for k in range(args.k_train + 1, args.k_test + 1):
        pcfg = DyckPCFG(k=k, p_close=0.5, seed=args.seed + k + 100)
        samples_per_k = args.n_samples // (args.k_test - args.k_train)
        inputs, _ = create_test_samples(pcfg, pcfg_tokenizer, samples_per_k, seed=args.seed + k + 100)
        ood_inputs.extend(inputs)
        ood_k_values.extend([k] * len(inputs))
    
    print(f"ID samples: {len(id_inputs)}, OOD samples: {len(ood_inputs)}")
    
    # Extract features
    print("\nExtracting features for ID samples...")
    id_features = extract_spline_features(model, id_inputs, device, args.batch_size)
    
    print("Extracting features for OOD samples...")
    ood_features = extract_spline_features(model, ood_inputs, device, args.batch_size)
    
    # Analysis
    print(f"\n{'='*70}")
    print("EPISTEMIC UNCERTAINTY ANALYSIS: ID vs OOD")
    print(f"{'='*70}")
    
    print(f"\nFeature comparison (aggregated across layers):")
    print(f"  {'Feature':15s} | {'ID mean':>10s} | {'ID std':>10s} | {'OOD mean':>10s} | {'OOD std':>10s} | {'p-value':>10s}")
    print(f"  {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    for feat_name in ["q10", "softmin", "sign_density"]:
        id_vals = id_features["aggregated"][feat_name]
        ood_vals = ood_features["aggregated"][feat_name]
        
        # Mann-Whitney U test (non-parametric)
        stat, pval = mannwhitneyu(id_vals, ood_vals, alternative='two-sided')
        
        print(f"  {feat_name:15s} | {id_vals.mean():10.4f} | {id_vals.std():10.4f} | {ood_vals.mean():10.4f} | {ood_vals.std():10.4f} | {pval:10.2e}")
    
    # Entropy
    stat, pval = mannwhitneyu(id_features["entropy"], ood_features["entropy"], alternative='two-sided')
    print(f"  {'entropy':15s} | {id_features['entropy'].mean():10.4f} | {id_features['entropy'].std():10.4f} | {ood_features['entropy'].mean():10.4f} | {ood_features['entropy'].std():10.4f} | {pval:10.2e}")
    
    # Per-layer analysis
    print(f"\nPer-layer comparison (ID mean → OOD mean):")
    print(f"  {'Layer':<8} | {'q10':>20s} | {'softmin':>20s} | {'sign_density':>20s}")
    print(f"  {'-'*8} | {'-'*20} | {'-'*20} | {'-'*20}")
    
    for l in range(id_features["n_layers"]):
        q10_diff = f"{id_features['per_layer'][l]['q10'].mean():.3f} → {ood_features['per_layer'][l]['q10'].mean():.3f}"
        sm_diff = f"{id_features['per_layer'][l]['softmin'].mean():.3f} → {ood_features['per_layer'][l]['softmin'].mean():.3f}"
        sd_diff = f"{id_features['per_layer'][l]['sign_density'].mean():.3f} → {ood_features['per_layer'][l]['sign_density'].mean():.3f}"
        print(f"  Layer {l:<2} | {q10_diff:>20s} | {sm_diff:>20s} | {sd_diff:>20s}")
    
    # Plot
    if args.plot or args.plot_path:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        
        feat_names = ["q10", "softmin", "sign_density", "entropy"]
        colors = {"ID": "tab:blue", "OOD": "tab:red"}
        
        for ax, feat_name in zip(axes.flat, feat_names):
            if feat_name == "entropy":
                id_vals = id_features["entropy"]
                ood_vals = ood_features["entropy"]
            else:
                id_vals = id_features["aggregated"][feat_name]
                ood_vals = ood_features["aggregated"][feat_name]
            
            # Histogram comparison
            ax.hist(id_vals, bins=30, alpha=0.6, label=f"ID (Dyck 2-{args.k_train})", color=colors["ID"], density=True)
            ax.hist(ood_vals, bins=30, alpha=0.6, label=f"OOD (Dyck {args.k_train+1}-{args.k_test})", color=colors["OOD"], density=True)
            ax.axvline(id_vals.mean(), color=colors["ID"], linestyle='--', linewidth=2)
            ax.axvline(ood_vals.mean(), color=colors["OOD"], linestyle='--', linewidth=2)
            ax.set_xlabel(feat_name)
            ax.set_ylabel("Density")
            ax.set_title(f"{feat_name}: ID vs OOD")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Epistemic Uncertainty: ID (Dyck 2-{args.k_train}) vs OOD (Dyck {args.k_train+1}-{args.k_test})", 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if args.plot_path:
            plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved plot to {args.plot_path}")
        
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
