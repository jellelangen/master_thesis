# train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data.builders import sample_points
from architectures.transformers import TinyTransformer2D
from architectures.utils import attach_gate_hooks, spline_features_from_gate

def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    X, y = sample_points(n=50000, seed=0)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    model = TinyTransformer2D(d_model=64, n_heads=4, d_ff=256, n_layers=2, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"epoch {epoch} done")
    torch.save(model.state_dict(), "ckpt.pt")
    print("saved ckpt.pt")
    # Feature extraction demo on a batch
    model.eval()
    handles, cache = attach_gate_hooks(model)

    xb = X[:1024].to(device)
    logits, _ = model(xb)  # fills cache via hooks

    layer_feats = []
    for i, blk in enumerate(model.blocks):
        h_gate = cache[i]["h_gate"]                      # [B,T,K]
        w = blk.mlp.gate_proj.weight.detach()            # [K,D]
        feats = spline_features_from_gate(h_gate, w)     # dict -> [B]
        layer_feats.append(feats)

    for h in handles:
        h.remove()

    # Example: stack features into [B, 7*n_layers]
    F = torch.stack([
        torch.stack([layer_feats[l][f"feature_{k}"] for k in range(1, 8)], dim=1)
        for l in range(len(layer_feats))
    ], dim=1)  # [B, L, 7]
    F = F.flatten(1)  # [B, 7L]
    print("Feature matrix:", F.shape)

if __name__ == "__main__":
    main()
