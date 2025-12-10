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

class ThreeBlockTransformer(nn.Module):
    def __init__(self, n_heads=1, head_dim=8, d_hidden=128):
        super().__init__()
        self.block1 = OneBlockTransformer(n_heads, head_dim, d_hidden)
        self.block2 = OneBlockTransformer(n_heads, head_dim, d_hidden)
        self.block3 = OneBlockTransformer(n_heads, head_dim, d_hidden)

    def forward(self, x, return_attn=False):
        # Block 1: transform sequence
        x1, attn1 = self.block1.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        last1 = x1[:, -1, :]
        h1 = self.block1.mlp_lin1(last1)
        h1_relu = self.block1.relu(h1)
        y1 = self.block1.mlp_lin2(h1_relu)

        
        x2, attn2 = self.block2.attn(
            x1, x1, x1, need_weights=True, average_attn_weights=False
        )
        last2 = x2[:, -1, :]
        h2 = self.block2.mlp_lin1(last2)
        h2_relu = self.block2.relu(h2)
        y2 = self.block2.mlp_lin2(h2_relu)

        
        x3, attn3 = self.block3.attn(
            x2, x2, x2, need_weights=True, average_attn_weights=False
        )
        last3 = x3[:, -1, :]
        h3 = self.block3.mlp_lin1(last3)
        h3_relu = self.block3.relu(h3)
        y3 = self.block3.mlp_lin2(h3_relu)

        y = y3 # final output

        if return_attn:
            return y, (h1, h2, h3), (attn1, attn2, attn3)
        else:
            return y, (h1, h2, h3)


class SinusoidalPosEncoding(nn.Module):
    def __init__(self, L, d_model):
        super().__init__()
        self.L = L
        self.d_model = d_model

        pe = torch.zeros(L, d_model)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]
