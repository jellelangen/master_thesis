import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://youtu.be/enPFr-WxHgQ

class TinyGatedMLP(nn.Module):
    """
    Llama-style gated MLP:
      h_gate = gate_proj(x)
      h_up   = up_proj(x)
      out = down_proj( silu(h_gate) * h_up )
    We expose gate_proj for feature extraction
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

    def forward(self, x, causal: bool = False):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T,dh]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B,H,T,T]
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask[None, None, :, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y), attn

class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinySelfAttn(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TinyGatedMLP(d_model, d_ff)

    def forward(self, x, causal: bool = False):
        a, attn = self.attn(self.ln1(x), causal=causal)
        x = x + a
        m = self.mlp(self.ln2(x))
        x = x + m
        return x, attn


class TinySeqTransformerFreq(nn.Module):
    """
    Input: y_0..y_{L-1} scalars
    Output: logits over frequency bins
    """
    def __init__(self, d_model=64, n_heads=4, d_ff=256, n_layers=2, max_len=128, n_bins=8):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_bins)

    def forward(self, x):
        # x: [B,T,1]
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.in_proj(x) + self.pos_emb(pos)[None, :, :]
        attn_all = []
        for blk in self.blocks:
            h, attn = blk(h, causal=True)
            attn_all.append(attn)
        h = self.ln_f(h)
        logits = self.head(h[:, -1, :])  # last token like "CLS"
        return logits, attn_all


class SplineTransformer(nn.Module):
    """
    Autoregressive transformer for discrete token sequences.
    Uses gated MLPs (like Llama) to enable spline feature extraction.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        max_len: int = 128,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks with gated MLP
        self.blocks = nn.ModuleList([
            TinyBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying 
        self.lm_head.weight = self.tok_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, x, return_hidden: bool = False):
        """
        Args:
            x: [B, T] token indices
            return_hidden: if True, also return hidden states before LM head
            
        Returns:
            logits: [B, T, vocab_size] next-token logits
            attn_all: list of attention weights per layer
            (hidden): [B, T, d_model] if return_hidden=True
        """
        B, T = x.shape
        device = x.device
        
        # Embeddings
        pos = torch.arange(T, device=device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)
        
        # Transformer blocks (causal)
        attn_all = []
        for blk in self.blocks:
            h, attn = blk(h, causal=True)
            attn_all.append(attn)
        
        h = self.ln_f(h)
        logits = self.lm_head(h)  # [B, T, V]
        
        if return_hidden:
            return logits, attn_all, h
        return logits, attn_all
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """
        Autoregressive generation.
        
        Args:
            prompt: [B, T] starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = normal, <1 = more deterministic)
            
        Returns:
            [B, T + max_new_tokens] generated sequence
        """
        self.eval()
        x = prompt.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(x)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_token], dim=1)
        
        return x