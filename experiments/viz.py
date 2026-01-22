import numpy as np
import torch
import matplotlib.pyplot as plt

from architectures.transformers import TinyTransformer2D
from architectures.utils import attach_gate_hooks, spline_features_from_gate, spline_features_cls_softmin
from data.builders import make_grid

@torch.no_grad()
def eval_on_grid(model, n=400, lim=1.0, layer_idx=-1, device="cpu", batch=8192):
    """
    Returns dict of 2D arrays on an n x n grid:
      pred, entropy, f5, f6, f7, region_id
    layer_idx selects which block's gate features to visualize.
    """
    X1, X2, _ = make_grid(n=n, lim=lim)
    pts = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)  # [N,2]
    N = pts.shape[0]

    handles, cache = attach_gate_hooks(model)

    preds = np.empty((N,), dtype=np.int64)
    ent = np.empty((N,), dtype=np.float32)
    f5 = np.empty((N,), dtype=np.float32)
    f6 = np.empty((N,), dtype=np.float32)
    f7 = np.empty((N,), dtype=np.float32)
    rid = np.empty((N,), dtype=np.uint32)

    # normalize layer index
    L = len(model.blocks)
    if layer_idx < 0:
        layer_idx = L + layer_idx
    assert 0 <= layer_idx < L

    for s in range(0, N, batch):
        e = min(N, s + batch)
        xb = torch.from_numpy(pts[s:e]).to(device)
        logits, _ = model(xb)  # fills cache via hooks

        prob = torch.softmax(logits, dim=-1)  # [B,C]
        preds[s:e] = prob.argmax(dim=-1).cpu().numpy()
        ent[s:e] = (-(prob * (prob.clamp_min(1e-12).log())).sum(dim=-1)).cpu().numpy()

        # spline features for selected layer
        blk = model.blocks[layer_idx]
        h_gate = cache[layer_idx]["h_gate"]       # [B,T,K]
        w = blk.mlp.gate_proj.weight.detach()     # [K,D]
        feats = spline_features_cls_softmin(h_gate, w, tau=0.05)

        # choose what you want to visualize
        f5[s:e] = feats["cls_softmin_dist"].cpu().numpy()
        f6[s:e] = feats["cls_q10_dist"].cpu().numpy()
        f7[s:e] = feats["cls_sign_density"].cpu().numpy()

        # region-id: hash of CLS sign pattern (token 0)
        # signbits: [B,K] bool
        signbits = (h_gate[:, 0, :] > 0).cpu().numpy()  # [B,K] bool
        K = signbits.shape[1]
        upto = min(32, K)

        packed = np.zeros((signbits.shape[0],), dtype=np.uint32)
        for i in range(upto):
            packed |= (signbits[:, i].astype(np.uint32) << i)

        rid[s:e] = packed


    for h in handles:
        h.remove()

    def reshape(v):
        return v.reshape(n, n)

    return {
        "X1": X1,
        "X2": X2,
        "pred": reshape(preds),
        "entropy": reshape(ent),
        "f5": reshape(f5),
        "f6": reshape(f6),
        "f7": reshape(f7),
        "region_id": reshape(rid),
        "layer_idx": layer_idx,
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the same architecture you trained.
    # If you saved a checkpoint, load it here. For now: quick retrain-free demo:
    model = TinyTransformer2D(d_model=64, n_heads=4, d_ff=256, n_layers=2, n_classes=3).to(device)
    ckpt_path = "ckpt.pt"
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded {ckpt_path}")
    except FileNotFoundError:
        print("No ckpt.pt found. Save one from train.py to visualize trained regions.")

    model.eval()

    out = eval_on_grid(model, n=450, lim=1.0, layer_idx=-1, device=device)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))

    ax[0, 0].set_title("Predicted class")
    im0 = ax[0, 0].imshow(out["pred"], origin="lower", extent=[-1, 1, -1, 1], aspect="equal")

    ax[0, 1].set_title("Softmax entropy")
    im1 = ax[0, 1].imshow(out["entropy"], origin="lower", extent=[-1, 1, -1, 1], aspect="equal")

    ax[0, 2].set_title(f"Region ID (CLS sign hash), layer {out['layer_idx']}")
    im2 = ax[0, 2].imshow(out["region_id"], origin="lower", extent=[-1, 1, -1, 1], aspect="equal")

    ax[1, 0].set_title("cls_softmin_dist")
    im3 = ax[1, 0].imshow(out["f5"], origin="lower", extent=[-1, 1, -1, 1], aspect="equal")

    ax[1, 1].set_title("cls_q10_dist")
    im4 = ax[1, 1].imshow(out["f6"], origin="lower", extent=[-1, 1, -1, 1], aspect="equal")

    ax[1, 2].set_title("cls_sign_density")
    im5 = ax[1, 2].imshow(out["f7"], origin="lower", extent=[-1, 1, -1, 1], aspect="equal")

    for a in ax.ravel():
        a.set_xlabel("x1")
        a.set_ylabel("x2")

    plt.tight_layout()
    plt.savefig("figures/viz_output_softmin.png")

if __name__ == "__main__":
    main()
