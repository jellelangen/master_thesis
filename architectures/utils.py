import torch


def compute_intrinsic_dim(model, X_scalar, pos_enc, eps=0.1):
    """
    Approximate ID_ε for the last token in each window, as in Eq. (8).

    Returns: np.array of shape [N], ID per sample.
    """
    model.eval()
    N, L = X_scalar.shape
    with torch.no_grad():
        x = pos_enc.unsqueeze(0).repeat(N, 1, 1)  # [N, L, d_model]
        x[:, :, 0] = X_scalar                     # insert sin signal in dim 0
        y_pred, h, attn_weights = model(x, return_attn=True)

    # attn_weights: [N, n_heads, L, L]
    last_idx = L - 1

    # attention to the last token: [N, n_heads, L]
    attn_last = attn_weights[:, :, last_idx, :]

    # count tokens with attention > eps over all heads
    # ID = sum_{h,j} 1[Attn(h, last, j) > eps]
    id_per_sample = (attn_last > eps).sum(dim=(-1, -2))  # [N]

    return id_per_sample.cpu().numpy(), y_pred.squeeze(1).cpu().numpy()