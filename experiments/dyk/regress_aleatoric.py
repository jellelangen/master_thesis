"""
Regression analysis for aleatoric uncertainty using 7 spline features.

THEORY:
    Tests whether spline features can predict the number of valid next tokens (n_valid),
    which is a measure of aleatoric uncertainty. This complements the classification
    approach in classify_ood.py with a regression objective.
    
    For Dyck-k prefixes:
    - n_valid = k if stack is empty (any open bracket valid)
    - n_valid = k+1 if stack is non-empty (any open + matching close)
    
    Three regressors are trained and compared:
    1. Spline features only (7 features)
    2. Entropy only (single feature baseline)
    3. Spline + Entropy combined
    
    Metrics: R², RMSE, Spearman correlation

RELEVANT FILES:
    - experiments/dyk/train.py: Use --mixed --k=8 for training
    - experiments/dyk/evaluate_mixed.py: Feature visualization by n_valid
    - architectures/utils.py: spline_features_from_gate function

CLI ARGUMENTS:
    --checkpoint      Path to model checkpoint (required)
    --max_k           Maximum k for Dyck grammars (default: 8)
    --n_samples_per_k Samples per grammar (default: 200)
    --batch_size      Batch size (default: 64)
    --seed            Random seed (default: 99999)
    --plot            Show scatter plots
    --plot_path       Save plots to file

DEPENDENCIES:
    - scikit-learn: Ridge, r2_score, train_test_split

USAGE:
    python -m experiments.dyk.regress_aleatoric --checkpoint=models/dyck_mixed.pt --max_k=8 --plot
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer
from architectures.utils import attach_gate_hooks, spline_features_from_gate


def get_valid_next_tokens_for_k(k: int, prefix_str: str, pcfg: DyckPCFG) -> int:
    """Count valid next tokens for a prefix."""
    stack = []
    for char in prefix_str:
        if char in pcfg._open_brackets:
            stack.append(char)
        elif char in pcfg._close_brackets:
            if stack:
                stack.pop()
    
    # k open brackets always valid + 1 close if stack non-empty
    return k + (1 if stack else 0)


def create_mixed_samples(max_k: int, n_samples_per_k: int, min_len: int = 4, max_len: int = 32, seed: int = 12345):
    """Generate samples from Dyck-2 to Dyck-max_k with n_valid labels."""
    np.random.seed(seed)
    
    pcfg_tokenizer = DyckPCFG(k=max_k, p_close=0.5, seed=seed)
    
    inputs = []
    n_valid_list = []
    k_values = []
    
    for k in range(2, max_k + 1):
        pcfg = DyckPCFG(k=k, p_close=0.5, seed=seed + k)
        count = 0
        while count < n_samples_per_k:
            seq = pcfg.sample()
            if len(seq) < min_len or len(seq) > max_len:
                continue
            
            prefix_len = np.random.randint(1, len(seq))
            prefix_str = seq[:prefix_len]
            
            n_valid = get_valid_next_tokens_for_k(k, prefix_str, pcfg)
            tokens = pcfg_tokenizer.tokenize(prefix_str, add_bos=True, add_eos=False)
            
            inputs.append(tokens)
            n_valid_list.append(n_valid)
            k_values.append(k)
            count += 1
    
    return inputs, np.array(n_valid_list), np.array(k_values)


def extract_features(model, inputs, device, batch_size=64):
    """Extract 7 spline features + entropy."""
    model.eval()
    handles, cache = attach_gate_hooks(model)
    n_layers = len(model.blocks)
    
    all_features = {f"feature_{i}": [] for i in range(1, 8)}
    all_features["entropy"] = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting features"):
            batch_inputs = inputs[i:i+batch_size]
            
            max_len = max(len(inp) for inp in batch_inputs)
            x = torch.zeros(len(batch_inputs), max_len, dtype=torch.long, device=device)
            
            for j, inp in enumerate(batch_inputs):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            
            logits, _ = model(x)
            
            layer_idx = n_layers - 1
            h_gate = cache[layer_idx]["h_gate"]
            gate_weight = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
            
            feats = spline_features_from_gate(h_gate, gate_weight)
            
            for j, inp in enumerate(batch_inputs):
                for k in range(1, 8):
                    all_features[f"feature_{k}"].append(feats[f"feature_{k}"][j].item())
                
                last_pos = len(inp) - 1
                last_logits = logits[j, last_pos]
                probs = torch.softmax(last_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                all_features["entropy"].append(entropy)
    
    for h in handles:
        h.remove()
    
    return {k: np.array(v) for k, v in all_features.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_k", type=int, default=8)
    parser.add_argument("--n_samples_per_k", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=99999)
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
    
    # Generate mixed samples
    print(f"\nGenerating samples from Dyck-2 to Dyck-{args.max_k}...")
    inputs, n_valid, k_values = create_mixed_samples(
        args.max_k, args.n_samples_per_k, seed=args.seed
    )
    print(f"Total samples: {len(inputs)}")
    print(f"n_valid range: {n_valid.min()} - {n_valid.max()}")
    
    # Extract features
    print("\nExtracting features...")
    features = extract_features(model, inputs, device, args.batch_size)
    
    # Build dataset
    feature_names = [f"feature_{i}" for i in range(1, 8)]
    X = np.stack([features[f] for f in feature_names], axis=1)
    y = n_valid
    X_entropy = features["entropy"].reshape(-1, 1)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    X_ent_train, X_ent_test, _, _ = train_test_split(X_entropy, y, test_size=0.3, random_state=args.seed)
    
    print("\n" + "="*70)
    print("REGRESSION RESULTS: Predicting n_valid (# valid next tokens)")
    print("="*70)
    
    # 1. Spline features only
    reg_spline = Ridge(alpha=1.0)
    reg_spline.fit(X_train, y_train)
    y_pred_spline = reg_spline.predict(X_test)
    
    r2_spline = r2_score(y_test, y_pred_spline)
    rmse_spline = np.sqrt(mean_squared_error(y_test, y_pred_spline))
    corr_spline, _ = spearmanr(y_test, y_pred_spline)
    
    print(f"\n1. SPLINE FEATURES ONLY (7 features)")
    print(f"   R²: {r2_spline:.4f}")
    print(f"   RMSE: {rmse_spline:.4f}")
    print(f"   Spearman r: {corr_spline:.4f}")
    print(f"\n   Feature coefficients:")
    for i, coef in enumerate(reg_spline.coef_):
        print(f"     feature_{i+1}: {coef:+.4f}")
    
    # 2. Entropy only
    reg_entropy = Ridge(alpha=1.0)
    reg_entropy.fit(X_ent_train, y_train)
    y_pred_ent = reg_entropy.predict(X_ent_test)
    
    r2_ent = r2_score(y_test, y_pred_ent)
    rmse_ent = np.sqrt(mean_squared_error(y_test, y_pred_ent))
    corr_ent, _ = spearmanr(y_test, y_pred_ent)
    
    print(f"\n2. ENTROPY ONLY (baseline)")
    print(f"   R²: {r2_ent:.4f}")
    print(f"   RMSE: {rmse_ent:.4f}")
    print(f"   Spearman r: {corr_ent:.4f}")
    
    # 3. Combined
    X_combined = np.hstack([X, X_entropy])
    X_comb_train, X_comb_test, _, _ = train_test_split(X_combined, y, test_size=0.3, random_state=args.seed)
    
    reg_combined = Ridge(alpha=1.0)
    reg_combined.fit(X_comb_train, y_train)
    y_pred_comb = reg_combined.predict(X_comb_test)
    
    r2_comb = r2_score(y_test, y_pred_comb)
    rmse_comb = np.sqrt(mean_squared_error(y_test, y_pred_comb))
    corr_comb, _ = spearmanr(y_test, y_pred_comb)
    
    print(f"\n3. SPLINE + ENTROPY (combined)")
    print(f"   R²: {r2_comb:.4f}")
    print(f"   RMSE: {rmse_comb:.4f}")
    print(f"   Spearman r: {corr_comb:.4f}")
    
    # Raw correlations
    print(f"\n" + "="*70)
    print("RAW FEATURE CORRELATIONS WITH n_valid")
    print("="*70)
    for fname in feature_names:
        r, p = spearmanr(n_valid, features[fname])
        print(f"  {fname}: r={r:+.4f} (p={p:.2e})")
    r, p = spearmanr(n_valid, features["entropy"])
    print(f"  {'entropy'}: r={r:+.4f} (p={p:.2e})")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':<25} | {'R²':>8} | {'RMSE':>8} | {'Spearman':>10}")
    print(f"  {'-'*25} | {'-'*8} | {'-'*8} | {'-'*10}")
    print(f"  {'Spline (7 features)':<25} | {r2_spline:8.4f} | {rmse_spline:8.4f} | {corr_spline:10.4f}")
    print(f"  {'Entropy only':<25} | {r2_ent:8.4f} | {rmse_ent:8.4f} | {corr_ent:10.4f}")
    print(f"  {'Spline + Entropy':<25} | {r2_comb:8.4f} | {rmse_comb:8.4f} | {corr_comb:10.4f}")
    
    # Plot
    if args.plot or args.plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, (name, y_pred, r2) in zip(axes, [
            ("Spline only", y_pred_spline, r2_spline),
            ("Entropy only", y_pred_ent, r2_ent),
            ("Spline + Entropy", y_pred_comb, r2_comb),
        ]):
            ax.scatter(y_test, y_pred, alpha=0.5, s=20)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
            ax.set_xlabel("True n_valid")
            ax.set_ylabel("Predicted n_valid")
            ax.set_title(f"{name} (R²={r2:.3f})")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Aleatoric Uncertainty: Predicting # Valid Next Tokens", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if args.plot_path:
            plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved plot to {args.plot_path}")
        
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
