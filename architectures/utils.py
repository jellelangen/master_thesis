import torch


def attach_gate_hooks(model):
    """
    Adds forward hooks to each block.mlp.gate_proj to capture its output h_gate.
    Returns: (handles, cache) where cache is list per layer.
    """
    cache = []
    handles = []

    for layer_idx, blk in enumerate(model.blocks):
        cache.append({})

        def make_hook(i):
            def hook(module, inp, out):
                # out is h_gate: [B,T,K]
                cache[i]["h_gate"] = out.detach()
            return hook

        h = blk.mlp.gate_proj.register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    return handles, cache


@torch.no_grad()
def spline_features_lasttok(
    h_gate: torch.Tensor,       # [B,T,K]
    gate_weight: torch.Tensor,  # [K,D]
    r: float = 0.005,
    eps: float = 1e-12,
):
    w_norm = gate_weight.norm(2, dim=1).clamp_min(eps)
    d = h_gate[:, -1, :].abs() / w_norm.view(1, -1)

    hardmin = d.amin(dim=1)
    q10 = torch.quantile(d, 0.10, dim=1)
    sign_density = (h_gate[:, -1, :] > 0).float().mean(dim=1)
    lc = (d < r).float().sum(dim=1)

    return {
        "hardmin": hardmin,
        "q10": q10,
        "sign_density": sign_density,
        "lc": lc,
    }