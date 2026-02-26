import torch


def compute_intrinsic_dim(model, X_scalar, pos_enc, eps=0.1):
    """
    Approximate ID_ε for the last token in each window, as in Eq. (8).

    Returns: np.array of shape [N], ID per sample.
    """
     # set epsilon relative to sequence length (this is basicallly uniform dist)
    model.eval()
    N, L = X_scalar.shape
    eps = 1/L
    with torch.no_grad():
        x = pos_enc.unsqueeze(0).repeat(N, 1, 1)  # [N, L, d_model]
        x[:, :, 0] = X_scalar                     # insert sin signal in dim 0
        y_pred, h, attn_weights = model(x, return_attn=True)
    # attn_weights: [N, n_heads, L, L]
    last_idx = L - 1

    # attention to the last token: [N, n_heads, L]
    attn_last = attn_weights[:, :, last_idx, :]
    attn_mean = attn_last.mean().item()
    attn_std  = attn_last.std().item()
    # count tokens with attention > eps over all heads
    # ID = sum_{h,j} 1[Attn(h, last, j) > eps]
    id_per_sample = (attn_last > eps).sum(dim=(-1, -2))  # [N]

    return id_per_sample.cpu().numpy(), y_pred.squeeze(1).cpu().numpy()


@torch.no_grad()
def spline_features_from_gate(h_gate: torch.Tensor, gate_weight: torch.Tensor):
    """
    h_gate: [B,T,K] pre-activation from gate_proj(x)
    gate_weight: [K,D] rows w_k
    Returns: dict with feature_1..feature_7 each [B]
    Implements Listing 2. :contentReference[oaicite:5]{index=5}
    """
    # w_norm: [K]
    w_norm = gate_weight.norm(2, dim=1).clamp_min(1e-12)

    # local_closest: [B,T] = min_k |h| / ||w||
    local_closest = (h_gate.abs() / w_norm.view(1, 1, -1)).amin(dim=2)
    
    # global_closest: [B] = min_t local_closest
    global_closest = local_closest.amin(dim=1)

    # local_signs: [B,T] = mean_k 1[h>0]
    local_signs = (h_gate > 0).float().mean(dim=2)

    global_signs = local_signs.mean(dim=1)

    return {
        "feature_1": global_signs,
        "feature_2": local_signs.amin(dim=1),
        "feature_3": local_signs.amax(dim=1),
        "feature_4": local_signs.std(dim=1),
        "feature_5": global_closest,
        "feature_6": local_closest.mean(dim=1),
        "feature_7": local_closest.std(dim=1),
    }

@torch.no_grad()
def spline_features_cls_softmin(
    h_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    tau: float = 0.05,
    eps: float = 1e-12,
):
    """
    CLS-only variant + softmin over gates.

    h_gate: [B,T,K] (output of gate_proj)
    gate_weight: [K,D]
    tau: softmin temperature (smaller -> closer to hard min)

    Returns dict with:
      - cls_softmin_dist: [B]  (soft-min of |h|/||w|| over gates for CLS token)
      - cls_mean_dist:    [B]  (mean of |h|/||w|| over gates for CLS)
      - cls_q10_dist:     [B]  (10th percentile of |h|/||w|| for CLS)
      - cls_q50_dist:     [B]  (median)
      - cls_q90_dist:     [B]  (90th percentile)
      - cls_sign_density: [B]  (mean_k 1[h>0] for CLS)
    """
    # norms of hyperplane normals: [K]
    w_norm = gate_weight.norm(2, dim=1).clamp_min(eps)  # [K]

    # distances for CLS only: d = |h|/||w|| -> [B,K]
    d = h_gate[:, 0, :].abs() / w_norm.view(1, -1)

    # softmin over gates using torch.nn.Softmin (returns weights that sum to 1)
    # note: use clamp_min on tau to avoid divide-by-zero if you pass tau=0
    tau_t = max(float(tau), 1e-8)
    softmin_weights = torch.nn.Softmin(dim=1)(d / tau_t)  # [B,K]
    cls_softmin = (softmin_weights * d).sum(dim=1)  # [B]

    # a few robust summaries (often more interpretable than hard min)
    cls_mean = d.mean(dim=1)
    q10 = torch.quantile(d, 0.10, dim=1)
    q50 = torch.quantile(d, 0.50, dim=1)
    q90 = torch.quantile(d, 0.90, dim=1)

    cls_sign = (h_gate[:, 0, :] > 0).float().mean(dim=1)

    return {
        "cls_softmin_dist": cls_softmin,
        "cls_mean_dist": cls_mean,
        "cls_q10_dist": q10,
        "cls_q50_dist": q50,
        "cls_q90_dist": q90,
        "cls_sign_density": cls_sign,
    }

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

# spline_feats.py
import torch

@torch.no_grad()
def spline_features_lasttok_softmin(
    h_gate: torch.Tensor,       # [B,T,K]
    gate_weight: torch.Tensor,  # [K,D]
    tau: float = 0.05,
    eps: float = 1e-12,
):
    """
    Like your CLS-only variant, but uses the LAST token (T-1).
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