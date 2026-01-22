"""
Binary classifier for ID vs OOD detection using 7 spline features.

THEORY:
    Tests whether spline features can discriminate between in-distribution (ID) and
    out-of-distribution (OOD) inputs. This is a direct test of epistemic uncertainty:
    if spline geometry captures model uncertainty, a classifier should achieve high AUROC.
    
    Three classifiers are trained and compared:
    1. Spline features only (7 features)
    2. Entropy only (single feature baseline)
    3. Spline + Entropy combined
    
    Feature importances reveal which spline features are most predictive of OOD status.

RELEVANT FILES:
    - experiments/dyk/train.py: Use --mixed --max_k_train=6 for OOD setup
    - experiments/dyk/evaluate_epistemic.py: Feature distribution analysis
    - architectures/utils.py: spline_features_from_gate function

CLI ARGUMENTS:
    --checkpoint    Path to model checkpoint (required)
    --k_train       Max k seen during training (default: 6)
    --k_test        Max k for testing (default: 8)
    --n_samples     Samples per distribution (default: 500)
    --batch_size    Batch size (default: 64)
    --seed          Random seed (default: 88888)
    --plot          Show ROC curves
    --plot_path     Save ROC curves to file

DEPENDENCIES:
    - scikit-learn: LogisticRegression, roc_auc_score, train_test_split

USAGE:
    python -m experiments.dyk.classify_ood --checkpoint=models/dyck_2to6.pt --k_train=6 --k_test=8 --plot
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer
from architectures.utils import attach_gate_hooks, spline_features_from_gate


def create_test_samples(pcfg: DyckPCFG, pcfg_tokenizer: DyckPCFG, n_samples: int, 
                        min_len: int = 4, max_len: int = 32, seed: int = 12345):
    """Generate test samples from a specific Dyck-k grammar."""
    np.random.seed(seed)
    
    inputs = []
    prefix_strs = []
    
    while len(inputs) < n_samples:
        seq = pcfg.sample()
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        prefix_len = np.random.randint(1, len(seq))
        prefix_str = seq[:prefix_len]
        tokens = pcfg_tokenizer.tokenize(prefix_str, add_bos=True, add_eos=False)
        
        inputs.append(tokens)
        prefix_strs.append(prefix_str)
    
    return inputs, prefix_strs


def extract_features(model, inputs, device, batch_size=64):
    """
    Extract the 7 spline features from the paper + entropy.
    
    Returns:
        features: dict with feature_1..feature_7, entropy as arrays [N]
    """
    model.eval()
    handles, cache = attach_gate_hooks(model)
    n_layers = len(model.blocks)
    
    # Storage for per-layer features
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
            
            # Extract features from last layer (can aggregate across layers if needed)
            layer_idx = n_layers - 1
            h_gate = cache[layer_idx]["h_gate"]
            gate_weight = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
            
            # Get 7 features - this gives sequence-level features
            feats = spline_features_from_gate(h_gate, gate_weight)
            
            for j, inp in enumerate(batch_inputs):
                for k in range(1, 8):
                    all_features[f"feature_{k}"].append(feats[f"feature_{k}"][j].item())
                
                # Entropy at last position
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
    parser.add_argument("--k_train", type=int, default=6, help="Max k seen during training")
    parser.add_argument("--k_test", type=int, default=8, help="Max k for testing")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=88888)
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
    
    pcfg_tokenizer = DyckPCFG(k=args.k_test, p_close=0.5, seed=args.seed)
    
    # Generate samples
    print(f"\nGenerating ID samples (Dyck-2 to Dyck-{args.k_train})...")
    id_inputs = []
    for k in range(2, args.k_train + 1):
        pcfg = DyckPCFG(k=k, p_close=0.5, seed=args.seed + k)
        n_per_k = args.n_samples // (args.k_train - 1)
        inputs, _ = create_test_samples(pcfg, pcfg_tokenizer, n_per_k, seed=args.seed + k)
        id_inputs.extend(inputs)
    
    print(f"Generating OOD samples (Dyck-{args.k_train+1} to Dyck-{args.k_test})...")
    ood_inputs = []
    for k in range(args.k_train + 1, args.k_test + 1):
        pcfg = DyckPCFG(k=k, p_close=0.5, seed=args.seed + k + 100)
        n_per_k = args.n_samples // (args.k_test - args.k_train)
        inputs, _ = create_test_samples(pcfg, pcfg_tokenizer, n_per_k, seed=args.seed + k + 100)
        ood_inputs.extend(inputs)
    
    print(f"ID: {len(id_inputs)}, OOD: {len(ood_inputs)}")
    
    # Extract features
    print("\nExtracting features...")
    id_features = extract_features(model, id_inputs, device, args.batch_size)
    ood_features = extract_features(model, ood_inputs, device, args.batch_size)
    
    # Build dataset
    feature_names = [f"feature_{i}" for i in range(1, 8)]
    
    X_id = np.stack([id_features[f] for f in feature_names], axis=1)
    X_ood = np.stack([ood_features[f] for f in feature_names], axis=1)
    
    X = np.vstack([X_id, X_ood])
    y = np.array([0] * len(X_id) + [1] * len(X_ood))  # 0=ID, 1=OOD
    
    # Also get entropy for baseline
    entropy_id = id_features["entropy"]
    entropy_ood = ood_features["entropy"]
    X_entropy = np.concatenate([entropy_id, entropy_ood]).reshape(-1, 1)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    X_ent_train, X_ent_test, _, _ = train_test_split(X_entropy, y, test_size=0.3, random_state=args.seed)
    
    # Train classifiers
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    
    # 1. Spline features only
    clf_spline = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf_spline.fit(X_train, y_train)
    y_pred_spline = clf_spline.predict(X_test)
    y_prob_spline = clf_spline.predict_proba(X_test)[:, 1]
    
    auroc_spline = roc_auc_score(y_test, y_prob_spline)
    acc_spline = accuracy_score(y_test, y_pred_spline)
    
    print(f"\n1. SPLINE FEATURES ONLY (7 features)")
    print(f"   Accuracy: {acc_spline:.4f}")
    print(f"   AUROC: {auroc_spline:.4f}")
    print(f"\n   Feature importances (coefficients):")
    for i, coef in enumerate(clf_spline.coef_[0]):
        print(f"     feature_{i+1}: {coef:+.4f}")
    
    # 2. Entropy only (baseline)
    clf_entropy = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf_entropy.fit(X_ent_train, y_train)
    y_pred_ent = clf_entropy.predict(X_ent_test)
    y_prob_ent = clf_entropy.predict_proba(X_ent_test)[:, 1]
    
    auroc_ent = roc_auc_score(y_test, y_prob_ent)
    acc_ent = accuracy_score(y_test, y_pred_ent)
    
    print(f"\n2. ENTROPY ONLY (baseline)")
    print(f"   Accuracy: {acc_ent:.4f}")
    print(f"   AUROC: {auroc_ent:.4f}")
    
    # 3. Spline + Entropy combined
    X_combined = np.hstack([X, X_entropy])
    X_comb_train, X_comb_test, _, _ = train_test_split(X_combined, y, test_size=0.3, random_state=args.seed)
    
    clf_combined = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf_combined.fit(X_comb_train, y_train)
    y_pred_comb = clf_combined.predict(X_comb_test)
    y_prob_comb = clf_combined.predict_proba(X_comb_test)[:, 1]
    
    auroc_comb = roc_auc_score(y_test, y_prob_comb)
    acc_comb = accuracy_score(y_test, y_pred_comb)
    
    print(f"\n3. SPLINE + ENTROPY (combined)")
    print(f"   Accuracy: {acc_comb:.4f}")
    print(f"   AUROC: {auroc_comb:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<25} | {'AUROC':>8} | {'Accuracy':>8}")
    print(f"  {'-'*25} | {'-'*8} | {'-'*8}")
    print(f"  {'Spline (7 features)':<25} | {auroc_spline:8.4f} | {acc_spline:8.4f}")
    print(f"  {'Entropy only':<25} | {auroc_ent:8.4f} | {acc_ent:8.4f}")
    print(f"  {'Spline + Entropy':<25} | {auroc_comb:8.4f} | {acc_comb:8.4f}")
    
    # Plot ROC curves
    if args.plot or args.plot_path:
        from sklearn.metrics import roc_curve
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # ROC curves
        for name, y_prob, color in [
            ("Spline (7 features)", y_prob_spline, "tab:blue"),
            ("Entropy only", y_prob_ent, "tab:orange"),
            ("Spline + Entropy", y_prob_comb, "tab:green"),
        ]:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auroc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})", linewidth=2, color=color)
        
        ax.plot([0, 1], [0, 1], 'k--', label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("OOD Detection: ID (Dyck 2-6) vs OOD (Dyck 7-8)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if args.plot_path:
            plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved plot to {args.plot_path}")
        
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
