import numpy as np
import torch
import matplotlib.pyplot as plt

from data.builders import sample_sine_batch, make_prefix_targets
from architectures.transformers import TinySeqTransformer
from architectures.utils import attach_gate_hooks, spline_features_lasttok_softmin

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinySeqTransformer(d_model=64, n_heads=4, d_ff=256, n_layers=2, max_len=128).to(device)
    model.load_state_dict(torch.load("models/ckpt_sine.pt", map_location=device))
    model.eval()

    # fixed latent wave, many queries from same params
    B = 1024
    T = 64
    y, _ = sample_sine_batch(batch_size=B, T=T, noise_std=0.0, seed=123)

    Ls = list(range(6, 50, 2))
    mse_list = []
    q10_list = []
    softmin_list = []
    sd_list = []

    handles, cache = attach_gate_hooks(model)

    for L in Ls:
        x, target = make_prefix_targets(y, L=L)
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        tb = torch.tensor(target, dtype=torch.float32, device=device)

        pred, _ = model(xb)

        # extract last-layer gate features
        layer_idx = len(model.blocks) - 1
        h_gate = cache[layer_idx]["h_gate"]                       # [B,T,K]
        w = model.blocks[layer_idx].mlp.gate_proj.weight.detach() # [K,D]
        feats = spline_features_lasttok_softmin(h_gate, w, tau=0.05)

        mse = torch.mean((pred - tb) ** 2).item()
        mse_list.append(mse)
        q10_list.append(feats["q10"].mean().item())
        softmin_list.append(feats["softmin"].mean().item())
        sd_list.append(feats["sign_density"].mean().item())

    for h in handles:
        h.remove()

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ls, mse_list, label="MSE (next-step)")
    ax.set_xlabel("Prefix length L")
    ax.set_ylabel("MSE")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/sine_mse_vs_prefix_length.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ls, q10_list, label="geom q10 (last tok)")
    ax.plot(Ls, softmin_list, label="geom softmin (last tok)")
    ax.plot(Ls, sd_list, label="geom sign density (last tok)")
    ax.set_xlabel("Prefix length L")
    ax.set_ylabel("Feature mean")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/sine_features_vs_prefix_length.png")

if __name__ == "__main__":
    main()
