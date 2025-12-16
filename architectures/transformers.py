import torch
import torch.nn as nn
import torch.nn.functional as F
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


class TinyGatedMLP(nn.Module):
    """
    Llama-style gated MLP:
      h_gate = gate_proj(x)
      h_up   = up_proj(x)
      out = down_proj( silu(h_gate) * h_up )
    We expose gate_proj for feature extraction like Listing 2. :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(F.silu(g) * u)

class TinySelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B,T,D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,dh]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B,H,T,T]
        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # [B,H,T,dh]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y), attn  # return attn for ID experiments later if desired

class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinySelfAttn(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TinyGatedMLP(d_model, d_ff)

    def forward(self, x):
        a, attn = self.attn(self.ln1(x))
        x = x + a
        m = self.mlp(self.ln2(x))
        x = x + m
        return x, attn

class TinyTransformer2D(nn.Module):
    """
    Input: 2D point -> 3-token sequence: [CLS, X(x1), Y(x2)]
    Output: class logits from CLS.
    """
    def __init__(self, d_model=64, n_heads=4, d_ff=256, n_layers=2, n_classes=3):
        super().__init__()
        self.d_model = d_model
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.x_embed = nn.Linear(1, d_model, bias=True)
        self.y_embed = nn.Linear(1, d_model, bias=True)

        self.blocks = nn.ModuleList([
            TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, xy):
        # xy: [B,2]
        B = xy.shape[0]
        x1 = xy[:, 0:1]
        x2 = xy[:, 1:2]
        tok_cls = self.cls.expand(B, 1, self.d_model)
        tok_x = self.x_embed(x1).unsqueeze(1)
        tok_y = self.y_embed(x2).unsqueeze(1)
        x = torch.cat([tok_cls, tok_x, tok_y], dim=1)  # [B,3,D]

        attn_all = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_all.append(attn)

        x = self.ln_f(x)
        logits = self.head(x[:, 0])  # CLS
        return logits, attn_all

