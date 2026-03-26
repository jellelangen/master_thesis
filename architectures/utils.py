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
def spline_features_lasttok_softmin(
    h_gate: torch.Tensor,       # [B,T,K]
    gate_weight: torch.Tensor,  # [K,D]
    tau: float = 0.05,
    eps: float = 1e-12,
):
    """

    Returns a dict of [B] tensors.
    """
    w_norm = gate_weight.norm(2, dim=1).clamp_min(eps)  # [K]
    d = h_gate[:, -1, :].abs() / w_norm.view(1, -1)     # [B,K]

    tau_t = max(float(tau), 1e-8)
    softmin_weights = torch.nn.Softmin(dim=1)(d / tau_t)  # [B,K]
    softmin = (softmin_weights * d).sum(dim=1)  # [B]

    q10 = torch.quantile(d, 0.10, dim=1)
    q50 = torch.quantile(d, 0.50, dim=1)
    sign_density = (h_gate[:, -1, :] > 0).float().mean(dim=1)

    return {
        "softmin": softmin,
        "q10": q10,
        "q50": q50,
        "sign_density": sign_density,
    }