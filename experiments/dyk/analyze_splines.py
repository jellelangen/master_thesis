"""
Spline feature analysis for trained Dyck-k transformers.

THEORY:
    Gated MLPs partition the input space into linear regions via ReLU activations.
    The 7 spline features from the paper characterize this geometry:
    
    Sign-based features (activation patterns):
    - feature_1: Global sign density (mean fraction of positive activations across all tokens)
    - feature_2: Minimum sign density across tokens
    - feature_3: Maximum sign density across tokens
    - feature_4: Standard deviation of sign density
    
    Distance-based features (proximity to decision boundaries):
    - feature_5: Global closest distance to any hyperplane
    - feature_6: Mean distance to hyperplanes
    - feature_7: Standard deviation of distances
    
    These features are extracted from the gate projection h_gate = gate_proj(x) in each layer.

RELEVANT FILES:
    - architectures/utils.py: spline_features_from_gate function (implements the 7 features)
    - architectures/transformers.py: SplineTransformer model with gated MLPs
    - experiments/dyk/train.py: Training script

CLI ARGUMENTS:
    --checkpoint    Path to model checkpoint (default: models/ckpt_dyck.pt)
    --n_samples     Number of sequences to analyze (default: 100)
    --seed          Random seed (default: 42)

USAGE:
    python -m experiments.dyk.analyze_splines --checkpoint=models/ckpt_dyck.pt --n_samples=200
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer
from architectures.utils import attach_gate_hooks, spline_features_from_gate


def analyze_spline_features(model, pcfg, n_samples=100, device="cpu"):
    """
    Extract spline features from the model for a set of Dyck sequences.
    
    Returns:
        features_per_layer: list of dicts, each with feature_1..7 as [N] arrays
        metadata: dict with sequence info (lengths, depths, validity)
    """
    model.eval()
    handles, cache = attach_gate_hooks(model)
    
    # Generate sequences with varying properties
    sequences = []
    depths = []
    lengths = []
    
    while len(sequences) < n_samples:
        seq, depth = pcfg.sample_with_depth()
        if 4 <= len(seq) <= 64:
            sequences.append(seq)
            depths.append(depth)
            lengths.append(len(seq))
    
    # Process in batches (sequences may have different lengths)
    all_features = [[] for _ in range(len(model.blocks))]
    
    for seq in sequences:
        tokens = pcfg.tokenize(seq)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        with torch.no_grad():
            _ = model(x)
        
        # Extract features from each layer
        for layer_idx, layer_cache in enumerate(cache):
            h_gate = layer_cache["h_gate"]
            gate_weight = model.blocks[layer_idx].mlp.gate_proj.weight
            feats = spline_features_from_gate(h_gate, gate_weight)
            all_features[layer_idx].append({k: v.cpu().numpy() for k, v in feats.items()})
    
    # Clean up hooks
    for h in handles:
        h.remove()
    
    # Aggregate features
    features_per_layer = []
    for layer_feats in all_features:
        aggregated = {}
        for key in layer_feats[0].keys():
            aggregated[key] = np.concatenate([f[key] for f in layer_feats])
        features_per_layer.append(aggregated)
    
    metadata = {
        "lengths": np.array(lengths),
        "depths": np.array(depths),
        "sequences": sequences,
    }
    
    return features_per_layer, metadata


def print_feature_stats(features_per_layer, metadata):
    """Print summary statistics of spline features."""
    print("\n" + "=" * 60)
    print("SPLINE FEATURE ANALYSIS")
    print("=" * 60)
    
    print(f"\nDataset: {len(metadata['sequences'])} sequences")
    print(f"  Length range: {metadata['lengths'].min()} - {metadata['lengths'].max()}")
    print(f"  Depth range: {metadata['depths'].min()} - {metadata['depths'].max()}")
    
    for layer_idx, feats in enumerate(features_per_layer):
        print(f"\n--- Layer {layer_idx} ---")
        for feat_name, values in feats.items():
            print(f"  {feat_name}: mean={values.mean():.4f}, std={values.std():.4f}, "
                  f"min={values.min():.4f}, max={values.max():.4f}")


def analyze_depth_correlation(features_per_layer, metadata):
    """Analyze correlation between spline features and nesting depth."""
    depths = metadata["depths"]
    
    print("\n" + "=" * 60)
    print("DEPTH CORRELATION ANALYSIS")
    print("=" * 60)
    
    for layer_idx, feats in enumerate(features_per_layer):
        print(f"\n--- Layer {layer_idx} ---")
        for feat_name, values in feats.items():
            corr = np.corrcoef(depths, values)[0, 1]
            print(f"  {feat_name} vs depth: r={corr:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/ckpt_dyck.pt")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please train the model first with: python -m experiments.dyk.train")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    vocab_size = ckpt["vocab_size"]
    
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Training config: {train_args}")
    
    # Create model
    model = SplineTransformer(
        vocab_size=vocab_size,
        d_model=train_args["d_model"],
        n_heads=train_args["n_heads"],
        d_ff=train_args["d_ff"],
        n_layers=train_args["n_layers"],
        max_len=128,
        pad_idx=0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Create PCFG with same config
    pcfg = DyckPCFG(k=train_args["k"], p_close=train_args["p_close"], seed=args.seed)
    
    # Analyze
    features, metadata = analyze_spline_features(
        model, pcfg, n_samples=args.n_samples, device=device
    )
    
    print_feature_stats(features, metadata)
    analyze_depth_correlation(features, metadata)
    
    # Test generation
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        for i in range(5):
            prompt = torch.tensor([[1]], device=device)  # Start with open bracket
            generated = model.generate(prompt, max_new_tokens=30, temperature=0.8)
            gen_seq = pcfg.detokenize(generated[0].cpu().tolist())
            is_valid = pcfg.is_valid(gen_seq)
            print(f"  {i+1}. '{gen_seq}' (len={len(gen_seq)}, valid={is_valid})")


if __name__ == "__main__":
    main()
