import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from data.builders import sample_sine_freq_batch, make_prefix
from architectures.transformers import TinySeqTransformerFreq
from architectures.utils import attach_gate_hooks, spline_features_lasttok_softmin

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
    accs, q10s, softmins, sds, entropies = [], [], [], [], []

    handles, cache = attach_gate_hooks(model)
    layer_idx = len(model.blocks) - 1

    for L in Ls:
        x = make_prefix(y, L=L)
        xb = torch.tensor(x, dtype=torch.float32, device=device)
        yb = torch.tensor(y_cls, dtype=torch.long, device=device)

        logits, _ = model(xb)
        dist = torch.distributions.Categorical(logits=logits)
        entropy = dist.entropy().mean().item()
        entropies.append(entropy)
        acc = (logits.argmax(dim=-1) == yb).float().mean().item()
        accs.append(acc)

        h_gate = cache[layer_idx]["h_gate"]
        w = model.blocks[layer_idx].mlp.gate_proj.weight.detach()
        feats = spline_features_lasttok_softmin(h_gate, w, tau=0.05)

        q10s.append(feats["q10"].mean().item())
        softmins.append(feats["softmin"].mean().item())
        sds.append(feats["sign_density"].mean().item())

    for h in handles:
        h.remove()

    plt.figure(figsize=(7,4))
    plt.plot(Ls, accs, marker="o")
    plt.xlabel("Prefix length L")
    plt.ylabel("Accuracy (freq bin)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/sine_freq_acc_vs_prefix_length2.png")
    plt.show()
    print(f"correlation between acc and q10: {spearmanr(accs, q10s)}")
    print(f"correlation between acc and softmin: {pearsonr(accs, softmins)}")
    print(f"correlation between acc and sign density: {pearsonr(accs, sds)}")
    print(f"correlation between acc and entropy: {pearsonr(accs, entropies)}")
    plt.figure(figsize=(7,4))
    # plt.plot(Ls, q10s, label="geom q10 (last tok)")
    # plt.plot(Ls, softmins, label="geom softmin (last tok)")
    # plt.plot(Ls, sds, label="geom sign density (last tok)")
    plt.plot(Ls, entropies, label="entropy (last tok)")
    plt.xlabel("Prefix length L")
    plt.ylabel("Feature mean")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/sine_freq_features_vs_prefix_length2.png")
    plt.show()

if __name__ == "__main__":
    main()