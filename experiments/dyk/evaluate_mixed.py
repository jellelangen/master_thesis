"""
Mixed Dyck evaluation: samples from Dyck-2 through Dyck-k evaluated on a single model.

USAGE:
    #train mixed model
    python -m experiments.dyk.train --k=8 --mixed --save_path="models/dyck_mixed.pt"

    python -m experiments.dyk.evaluate_mixed --checkpoint=models/dyck_mixed.pt --max_k=8 --plot
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import matplotlib.pyplot as plt

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer
from architectures.utils import attach_gate_hooks, spline_features_lasttok
def kde_quantile_band_flat(xvals, yvals, quantiles=(0.25, 0.5, 0.75),
                           bw_adjust=1.0, n_sub=5000, n_grid_x=160,
                           n_grid_y=400, pad_sigma=4.0, log_space=False,
                           seed=0):
    """
    Fit one 2D Gaussian KDE to the (n_valid, value) cloud and read the
    conditional quantiles off it, column by column.

    Returns grid_x, an array of shape [len(quantiles), n_grid_x], and the
    bandwidth in data units. Use log_space=True for non-negative, right-skewed
    quantities such as hardmin, otherwise the recovered quantiles are biased
    upward by mass leaking below zero.
    """
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)

    rng = np.random.default_rng(seed)
    if len(xvals) > n_sub:
        keep = rng.choice(len(xvals), size=n_sub, replace=False)
        xvals = xvals[keep]
        yvals = yvals[keep]



    kde = gaussian_kde(np.vstack([xvals, yvals]))
    kde.set_bandwidth(bw_method=kde.factor * bw_adjust)
    sigma = np.sqrt(np.diag(kde.covariance))

    grid_x = np.linspace(xvals.min(), xvals.max(), n_grid_x)
    grid_y = np.linspace(yvals.min() - pad_sigma * sigma[1],
                         yvals.max() + pad_sigma * sigma[1], n_grid_y)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    density = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_y.shape)

    cdf = np.cumsum(density, axis=0)
    cdf = cdf / cdf[-1]
    curves = np.zeros((len(quantiles), len(grid_x)))
    for qindex in range(len(quantiles)):
        for col in range(len(grid_x)):
            curves[qindex, col] = np.interp(quantiles[qindex], cdf[:, col], grid_y)



    return grid_x, curves, sigma


def acc_band(xvals, correct, levels, z_score=1.96):
    """Per-bin proportion correct with a Wilson interval."""
    mid = np.zeros(len(levels))
    low = np.zeros(len(levels))
    high = np.zeros(len(levels))
    for col in range(len(levels)):
        selected = correct[xvals == levels[col]]
        prop = selected.mean()
        num = float(len(selected))
        zsq = z_score ** 2
        den = 1.0 + zsq / num
        centre = (prop + zsq / (2.0 * num)) / den
        spread = z_score / den * np.sqrt(prop * (1.0 - prop) / num
                                         + zsq / (4.0 * num ** 2))
        mid[col] = prop
        low[col] = centre - spread
        high[col] = centre + spread
    return mid, low, high

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
    
    Uses Dyck-max_k PCFG for tokenization .
    
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


def create_balanced_test_set(max_k: int, n_samples_per_bin: int, min_len: int = 4,
                             max_len: int = 32, seed: int = 12345):
    """Generate exactly ``n_samples_per_bin`` examples for every n_valid bin.

    A bin n can be produced by an empty-stack Dyck-n prefix or a non-empty-stack
    Dyck-(n-1) prefix.  Where both are available, the two grammar/stack-state
    sources are represented as evenly as possible. 
    """
    rng = np.random.default_rng(seed)
    pcfg_max = DyckPCFG(k=max_k, p_close=0.5, seed=seed)
    pcfgs = {
        k: DyckPCFG(k=k, p_close=0.5, seed=seed + k * 1000)
        for k in range(2, max_k + 1)
    }

    records = []
    for target_n_valid in range(2, max_k + 2):
        source_ks = []
        if target_n_valid <= max_k:
            source_ks.append(target_n_valid)      # empty stack
        if target_n_valid - 1 >= 2:
            source_ks.append(target_n_valid - 1)  # non-empty stack

        for sample_idx in range(n_samples_per_bin):
            k = source_ks[sample_idx % len(source_ks)]
            pcfg_k = pcfgs[k]

            while True:
                seq = pcfg_k.sample()
                if len(seq) < min_len or len(seq) > max_len:
                    continue
                prefix_len = int(rng.integers(1, len(seq)))
                prefix_str = seq[:prefix_len]
                valid = get_valid_next_tokens_for_k(k, prefix_str, pcfg_k)
                if len(valid) == target_n_valid:
                    break

            tokens = pcfg_max.tokenize(prefix_str, add_bos=True, add_eos=False)
            records.append((tokens, target_n_valid, k, prefix_str))
    # Shuffle samples
    order = rng.permutation(len(records))
    records = [records[i] for i in order]
    inputs, n_valid, k_values, prefix_strs = map(list, zip(*records))
    return inputs, n_valid, k_values, prefix_strs


def extract_spline_features(model, inputs, device, batch_size=64):
    """
    Extract spline features from all layers of the model.
    
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
                    feats = spline_features_lasttok(h_single, gate_weight, 0.1)

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
    parser.add_argument("--sampling", choices=["balanced", "natural"], default="balanced",
                        help="Balance n_valid bins (default) or sample each grammar naturally")
    parser.add_argument("--n_samples_per_bin", "--n_samples_per_k", dest="n_samples", type=int,
                        default=1000,
                        help="Samples per n_valid bin (or per grammar with --sampling natural)")
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
    
    # Generate mixed test set from Dyck-2 to Dyck-max_k.
    if args.sampling == "balanced":
        print(f"\nGenerating {args.n_samples} samples for each n_valid bin "
              f"from 2 to {args.max_k + 1}...")
        inputs, n_valid, k_values, prefix_strs = create_balanced_test_set(
            args.max_k, args.n_samples, seed=args.seed
        )
    else:
        print(f"\nGenerating {args.n_samples} samples each from "
              f"Dyck-2 to Dyck-{args.max_k} (natural sampling)...")
        inputs, n_valid, k_values, prefix_strs = create_mixed_test_set(
            args.max_k, args.n_samples, seed=args.seed
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


    # Also report correlations without the n_valid=2 bin because it seems like a confound.
    mask_no_two = n_valid >= 3
    print(f"\nCorrelations with n_valid, excluding n_valid=2 (n={mask_no_two.sum()}):")
    for feat_name in ["hardmin", "q10", "sign_density", "lc"]:
        r, p = spearmanr(n_valid[mask_no_two], features[feat_name][mask_no_two])
        print(f"  {feat_name:15s}: r={r:+.4f} (p={p:.2e})")
    r, p = spearmanr(n_valid[mask_no_two], result["entropy"][mask_no_two])
    print(f"  {'entropy':15s}: r={r:+.4f} (p={p:.2e})")

    print(f"\nPer-layer correlations with n_valid (n_valid >= 3):")
    for l in range(n_layers):
        layer_feats = result["per_layer"][l]
        r_q10, _ = spearmanr(n_valid[mask_no_two], layer_feats["q10"][mask_no_two])
        r_lc, _ = spearmanr(n_valid[mask_no_two], layer_feats["lc"][mask_no_two])
        print(f"  Layer {l}: q10 r={r_q10:+.4f} | lc r={r_lc:+.4f}")
    # Plot
    if args.plot or args.plot_path:
        levels = np.array(unique_nvalid, dtype=float)

        panel_list = ["hardmin", "q10", "lc", "entropy"]
        if "valid_mass" in result:
            panel_list.append("valid_mass")
        if "correct" in result:
            panel_list.append("accuracy")

        n_cols = 2
        n_rows = int(np.ceil(len(panel_list) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for position in range(len(panel_list)):
            feat_name = panel_list[position]
            axis = axes[position]

            if feat_name == "accuracy":
                mid, low, high = acc_band(n_valid, result["correct"], levels)
                axis.fill_between(levels, low, high, alpha=0.25,
                                  color="tab:blue", linewidth=0)
                axis.plot(levels, mid, marker="o", color="tab:blue")
                axis.set_ylabel("validity accuracy")
            else:
                if feat_name == "entropy":
                    feat_vals = result["entropy"]
                elif feat_name == "valid_mass":
                    feat_vals = result["valid_mass"]
                else:
                    feat_vals = features[feat_name]

                grid_x, curves, sigma = kde_quantile_band_flat(n_valid, feat_vals)
                axis.fill_between(grid_x, curves[0], curves[2], alpha=0.25,
                                  color="tab:blue", linewidth=0)

                raw_medians = np.zeros(len(levels))
                for col in range(len(levels)):
                    raw_medians[col] = np.median(feat_vals[n_valid == levels[col]])
                axis.plot(levels, raw_medians, marker="o", linestyle="none",
                          alpha=0.5, color="tab:blue")
                axis.plot(grid_x, curves[1], color="tab:blue", linewidth=1.8)
                axis.set_ylabel(feat_name)
                print("  %-12s bandwidth: %.3f bins, %.4g in value"
                      % (feat_name, sigma[0], sigma[1]))

            if levels[0] == 2:
                axis.axvspan(1.5, 2.5, color="0.92", linewidth=0, zorder=0)
            axis.set_xticks(levels)
            axis.set_xlabel("number of valid next tokens")
            axis.grid(True, alpha=0.3)

        for position in range(len(panel_list), len(axes)):
            axes[position].set_visible(False)

        plt.suptitle(f"Mixed Dyck-2 to Dyck-{args.max_k}", fontsize=12,
                     fontweight='bold')
        plt.tight_layout()
        if args.plot_path:
            plt.savefig(args.plot_path, dpi=300, bbox_inches="tight")
            print(f"\nSaved plot to {args.plot_path}")
        
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()
