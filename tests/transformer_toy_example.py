import torch
import torch.nn as nn
import torch.optim as optim
import math
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import numpy as np
import os


from architectures.transformers import OneBlockTransformer, ThreeBlockTransformer, SinusoidalPosEncoding
from architectures.utils import compute_intrinsic_dim
from architectures.transformers import SinusoidalPosEncoding
from data.builders import build_sinusoidal_dataset as build_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"










def train_one_block(L=10, n_heads=1, head_dim=8,
                    d_hidden=128, n_steps=3000, lr=1e-3):
    d_model = n_heads * head_dim

    X_scalar, y = build_dataset(T=1000, L=L, domain=(-2*math.pi, 2*math.pi))
    N = X_scalar.shape[0]
    positional_encoder = SinusoidalPosEncoding(L, d_model).to(device)  # [L, d_model]
    pos_enc = positional_encoder.pe  # [L, d_model]
    model = OneBlockTransformer(n_heads=n_heads,
                                head_dim=head_dim,
                                d_hidden=d_hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(n_steps):
        idx = torch.randint(0, N, (64,))
        x_batch_scalar = X_scalar[idx]  # [B, L]
        y_batch = y[idx]

        x_batch = pos_enc.unsqueeze(0).repeat(64, 1, 1)  # [B, L, d_model]
        x_batch[:, :, 0] = x_batch_scalar  # put sin(t) into first dim

        y_pred, _ = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, X_scalar, y, pos_enc


def count_regions_transformer(model, X_scalar, pos_enc):
    """
    Approximate number of regions along time by sliding windows and
    watching when the MLP's hidden preactivation pattern changes.
    """
    N, L = X_scalar.shape
    with torch.no_grad():
        # build full batch
        x = pos_enc.unsqueeze(0).repeat(N, 1, 1)  # [N, L, d_model]
        x[:, :, 0] = X_scalar                     # insert the sine channel
        _, h = model(x)                           # h: [N, d_hidden], pre-ReLU

        sign_pattern = (h > 0).int()
        regions = 1
        for i in range(1, N):
            if not torch.equal(sign_pattern[i], sign_pattern[i-1]):
                regions += 1
    return regions




def run_and_plot(L, n_heads, title_prefix=""):
    model, X_scalar, y, pos_enc = train_one_block(L=L, n_heads=n_heads)
    regions = count_regions_transformer(model, X_scalar, pos_enc)
    print(f"L={L}, heads={n_heads}, approx regions: {regions}")

    # Evaluate predictions along windows, aligned by last index
    with torch.no_grad():
        N = X_scalar.shape[0]
        x = pos_enc.unsqueeze(0).repeat(N, 1, 1)  # [N, L, d_model]
        x[:, :, 0] = X_scalar
        y_pred, _ = model(x)

    y_true = y.squeeze(1).cpu().numpy()
    y_pred_np = y_pred.squeeze(1).cpu().numpy()

    # x-axis = index of last token in each window (approx time)
    t_idx = np.arange(L, L + len(y_true))

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Top: sin approximation
    axs[0].plot(t_idx, y_true, label="sin(t)")
    axs[0].plot(t_idx, y_pred_np, linestyle="--", label="model")
    axs[0].set_title(f"{title_prefix} L={L}, heads={n_heads}, regions≈{regions}")
    axs[0].legend()

    # Middle: region index (piecewise constant)
        # --- Region boundary histogram (new regions per bin) ---
    with torch.no_grad():
        _, h = model(x.to(device))
    sign_pattern = (h > 0).int().cpu()

    region_id = [0]
    cur_id = 0
    for i in range(1, len(sign_pattern)):
        if not torch.equal(sign_pattern[i - 1], sign_pattern[i]):
            cur_id += 1
        region_id.append(cur_id)
    region_id = np.array(region_id)

    boundary_idx = np.where(np.diff(region_id) > 0)[0] + 1
    boundary_positions = t_idx[boundary_idx]

    # bin boundaries
    n_bins = 50
    bins = np.linspace(t_idx.min(), t_idx.max(), n_bins + 1)
    counts, edges = np.histogram(boundary_positions, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # plot region density
    axs[1].bar(centers, counts, width=(edges[1] - edges[0]))
    axs[1].set_ylabel("New regions per bin")

    # --- Absolute error ---
    abs_err = np.abs(y_true - y_pred_np)

    # bin absolute error in same bins as region counts
    err_sums, _ = np.histogram(t_idx, bins=bins, weights=abs_err)
    err_counts, _ = np.histogram(t_idx, bins=bins)
    mean_abs_err = np.divide(err_sums, err_counts,
                             out=np.zeros_like(err_sums), where=err_counts > 0)
     # Intrinsic dimension for last token in each window
    id_values, y_pred_from_id = compute_intrinsic_dim(model, X_scalar, pos_enc, eps=0.1)
    abs_err = np.abs(y_true - y_pred_np)

    
    # --- Spearman correlation ---
    rho, p_value = spearmanr(counts, mean_abs_err)
    rho_id, p_id = spearmanr(id_values, abs_err)
    print(f"Spearman(ID, |err|): rho={rho_id:.3f}, p={p_id:.3e}")
    axs[2].plot(t_idx, abs_err)
    axs[2].set_ylabel("|error|")
    axs[2].set_xlabel("time index (last token)")

    # Update title with correlation
    axs[0].set_title(
        f"{title_prefix} L={L}, heads={n_heads}, regions≈{regions}\n"
        f"Spearman(counts, |err|) = {rho:.3f}, p={p_value:.3e}"
    )
    mask = counts > 0
    rho_act, p_act = spearmanr(counts[mask], mean_abs_err[mask])
    
    plt.tight_layout()
   
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{title_prefix}_L{L}_heads{n_heads}.png", dpi=150)



run_and_plot(L=10,  n_heads=1,  title_prefix="Toy 1D transformer")
run_and_plot(L=10,  n_heads=10, title_prefix="Toy 1D transformer")
run_and_plot(L=100, n_heads=1,  title_prefix="Toy 1D transformer")
run_and_plot(L=100, n_heads=10, title_prefix="Toy 1D transformer")