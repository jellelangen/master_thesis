import torch
import math
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_sinusoidal_dataset(T=1000, L=10, domain=(-4*math.pi, 4*math.pi)):
    """
    Return windows of length L and targets = sin at last position.
    """
    t = torch.linspace(*domain, steps=T)  # [T]
    s = torch.sin(t)             # [T]
    # s = torch.where(t < 0, torch.sin(t), 3 * torch.sin(5 * t))
    # s = torch.sin(t**2)
    # s = torch.sin(t) + 0.5 * torch.sin(5 * t)
    X = []
    y = []
    for start in range(0, T - L):
        window = s[start:start+L]       # [L]
        X.append(window)
        y.append(window[-1])            # predict last value

    X = torch.stack(X)                  # [N, L]
    y = torch.stack(y).unsqueeze(1)     # [N, 1]
    return X.to(device), y.to(device)




def sample_sine_batch(
    batch_size: int,
    T: int,
    A_range=(0.5, 1.5),
    w_range=(0.05, 0.35),
    phi_range=(0.0, 2*np.pi),
    b_range=(-0.5, 0.5),
    noise_std=0.0,
    seed=None,
):
    """
    Returns:
      y: [B,T] float32   where y_t = A*sin(w*t + phi) + b + noise
      params: dict of sampled latent parameters
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(*A_range, size=(batch_size, 1))
    w = rng.uniform(*w_range, size=(batch_size, 1))
    phi = rng.uniform(*phi_range, size=(batch_size, 1))
    b = rng.uniform(*b_range, size=(batch_size, 1))

    t = np.arange(T, dtype=np.float32)[None, :]  # [1,T]
    y = A * np.sin(w * t + phi) + b
    if noise_std > 0:
        y = y + rng.normal(0.0, noise_std, size=y.shape)

    y = y.astype(np.float32)
    return y, {"A": A, "w": w, "phi": phi, "b": b}

def make_prefix_targets(y: np.ndarray, L: int):
    """
    y: [B,T]
    L: prefix length (must satisfy 2 <= L < T)

    We feed tokens y[:,:L] and predict y[:,L] (next value).
    Return:
      x: [B,L,1]
      target: [B,1]
    """
    assert 2 <= L < y.shape[1]
    x = y[:, :L][:, :, None]        # [B,L,1]
    target = y[:, L][:, None]       # [B,1]
    return x, target


def sample_sine_freq_batch(
    batch_size: int,
    T: int,
    n_bins: int = 8,
    w_min: float = 0.05,
    w_max: float = 0.35,
    A_range=(0.8, 1.2),
    phi_range=(0.0, 2*np.pi),
    b_range=(-0.2, 0.2),
    noise_std: float = 0.02,
    seed=None,
):
    """
    Sample omega uniformly in [w_min, w_max], then assign it to a bin label in {0..n_bins-1}.
    Returns:
      y: [B,T] float32
      y_cls: [B] int64  (frequency bin)
      omega: [B,1] float32
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(*A_range, size=(batch_size, 1))
    w = rng.uniform(w_min, w_max, size=(batch_size, 1))
    phi = rng.uniform(*phi_range, size=(batch_size, 1))
    b = rng.uniform(*b_range, size=(batch_size, 1))

    # bin index
    edges = np.linspace(w_min, w_max, n_bins + 1)
    y_cls = np.digitize(w[:, 0], edges[1:-1], right=False).astype(np.int64)

    t = np.arange(T, dtype=np.float32)[None, :]
    y = A * np.sin(w * t + phi) + b
    if noise_std > 0:
        y = y + rng.normal(0.0, noise_std, size=y.shape)
    return y.astype(np.float32), y_cls, w.astype(np.float32)

def make_prefix(y: np.ndarray, L: int):
    """
    y: [B,T]
    returns x: [B,L,1]
    """
    assert 2 <= L <= y.shape[1]
    return y[:, :L][:, :, None].astype(np.float32)