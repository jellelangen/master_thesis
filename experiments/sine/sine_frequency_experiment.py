import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde


from data.builders import sample_sine_freq_batch, make_prefix
from architectures.transformers import TinySeqTransformerFreq
from architectures.utils import attach_gate_hooks, spline_features_lasttok
def kde_quantile_band(prefix_lengths, values, quantiles=(0.25, 0.5, 0.75),
                      bw_adjust=1.0, n_sub=5000, n_grid_x=160, n_grid_y=400,
                      seed=0):
    """
    Fit one 2D Gaussian KDE to the (prefix length, value) cloud and read the
    requested conditional quantiles off it, column by column.

    Returns grid_x, an array of shape [len(quantiles), n_grid_x], and the
    bandwidth in data units.
    """
    prefix_lengths = np.asarray(prefix_lengths, dtype=float)
    values = np.asarray(values, dtype=float)          # [n_lengths, batch]

    xflat = np.repeat(prefix_lengths, values.shape[1])
    yflat = values.ravel()

    rng = np.random.default_rng(seed)
    if len(xflat) > n_sub:
        keep = rng.choice(len(xflat), size=n_sub, replace=False)
        xflat = xflat[keep]
        yflat = yflat[keep]

    kde = gaussian_kde(np.vstack([xflat, yflat]))
    kde.set_bandwidth(bw_method=kde.factor * bw_adjust)

    grid_x = np.linspace(prefix_lengths.min(), prefix_lengths.max(), n_grid_x)
    grid_y = np.linspace(yflat.min(), yflat.max(), n_grid_y)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    density = kde(np.vstack([mesh_x.ravel(), mesh_y.ravel()])).reshape(mesh_y.shape)

    cdf = np.cumsum(density, axis=0)
    cdf = cdf / cdf[-1]
    curves = np.zeros((len(quantiles), len(grid_x)))
    for qindex in range(len(quantiles)):
        for col in range(len(grid_x)):
            curves[qindex, col] = np.interp(quantiles[qindex], cdf[:, col], grid_y)

    return grid_x, curves, np.sqrt(np.diag(kde.covariance))

def acc_band(correct):
    """Accuracy with a normal-approximation binomial interval."""
    correct = np.asarray(correct, dtype=float)
    mean = correct.mean(axis=1)
    stderr = np.sqrt(mean * (1.0 - mean) / correct.shape[1])
    return mean, mean - 1.96 * stderr, mean + 1.96 * stderr



@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_bins = 8

    model = TinySeqTransformerFreq(d_model=64, n_heads=4, d_ff=512, n_layers=4, max_len=128, n_bins=n_bins).to(device)
    model.load_state_dict(torch.load("models/ckpt_sine_freq.pt", map_location=device))
    model.eval()

    B = 4096//4
    T = 64
    y, y_cls, _ = sample_sine_freq_batch(batch_size=B, T=T, n_bins=n_bins, noise_std=0.00, seed=0)

    Ls = list(range(6, 50, 2))
    accs, hardmins, q10s, sds, lcs, entropies = [], [], [], [], [], []

    handles, cache = attach_gate_hooks(model)
    layer_idx = len(model.blocks) - 1

    for L in Ls:
        x = make_prefix(y, L=L)
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        yb = torch.tensor(y_cls, dtype=torch.long, device=device)

        logits, _ = model(xb)
        dist = torch.distributions.Categorical(logits=logits)
        entropies.append(dist.entropy().cpu().numpy())
        accs.append((logits.argmax(dim=-1) == yb).float().cpu().numpy())

        h_gate = cache[layer_idx]["h_gate"]
        w = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
        feats = spline_features_lasttok(h_gate, w, 0.1)

        hardmins.append(feats["hardmin"].cpu().numpy())
        q10s.append(feats["q10"].cpu().numpy())
        sds.append(feats["sign_density"].cpu().numpy())
        lcs.append(feats["lc"].cpu().numpy())
    for h in handles:
        h.remove()
    acc_mean = np.asarray(accs, dtype=float).mean(axis=1)
    hardmin_mean = np.asarray(hardmins, dtype=float).mean(axis=1)
    q10_mean = np.asarray(q10s, dtype=float).mean(axis=1)
    sd_mean = np.asarray(sds, dtype=float).mean(axis=1)
    lc_mean = np.asarray(lcs, dtype=float).mean(axis=1)
    entropy_mean = np.asarray(entropies, dtype=float).mean(axis=1)

    acc_mid, acc_low, acc_high = acc_band(accs)
    print(f"correlation between acc and hardmin: {spearmanr(acc_mean, hardmin_mean)}")
    print(f"correlation between acc and q10: {spearmanr(acc_mean, q10_mean)}")
    print(f"correlation between acc and sign density: {pearsonr(acc_mean, sd_mean)}")
    print(f"correlation between acc and lc: {spearmanr(acc_mean, lc_mean)}")
    print(f"correlation between acc and entropy: {pearsonr(acc_mean, entropy_mean)}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for axis, vals, name in zip(axes, [hardmins, q10s, lcs, entropies, accs],
                                ["hardmin", "q10", "local complexity (r=0.1)",
                                 "entropy", "accuracy"]):
        if name == "accuracy":
            mid, low, high = acc_band(vals)
            axis.fill_between(Ls, low, high, alpha=0.25, color="tab:blue", linewidth=0)
            axis.plot(Ls, mid, marker="o", color="tab:blue")
        else:
            grid_x, curves, bandwidth = kde_quantile_band(Ls, vals)
            axis.fill_between(grid_x, curves[0], curves[2], alpha=0.25,
                              color="tab:blue", linewidth=0)
            axis.plot(Ls, np.median(vals, axis=1), marker="o", linestyle="none",
                      alpha=0.5, color="tab:blue")
            axis.plot(grid_x, curves[1], color="tab:blue", linewidth=1.8)

        axis.set_xlabel("Prefix length L")
        axis.set_ylabel(name)
        axis.set_title(name)
        axis.grid(True)
    axes[-1].set_visible(False)
    fig.suptitle("Sine Frequency Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/sine_freq_features_vs_prefix_length2.pdf")
    plt.show()

if __name__ == "__main__":
    main()
