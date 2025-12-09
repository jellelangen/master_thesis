import torch
import torch.nn as nn
import torch.optim as optim


class OneBlockTransformer(nn.Module):
    def __init__(self, n_heads=1, head_dim=8, d_hidden=128):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.mlp_lin1 = nn.Linear(self.d_model, d_hidden)
        self.mlp_lin2 = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x, return_attn=False):
        # x: [B, L, d_model]
        attn_out, attn_weights = self.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )  # attn_weights: [B, n_heads, L, L]

        last = attn_out[:, -1, :]      # [B, d_model]
        h = self.mlp_lin1(last)        # [B, d_hidden]
        h_relu = self.relu(h)
        y = self.mlp_lin2(h_relu)      # [B, 1]

        if return_attn:
            return y, h, attn_weights
        else:
            return y, h


