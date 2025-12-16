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




def label_fn(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    3-class piecewise function over [-1,1]^2 with mixed regimes:
    - top-right: checkerboard
    - bottom-right: rings
    - left half: stripes + blob
    """
    y = np.zeros_like(x1, dtype=np.int64)

    # left half: vertical stripes
    left = x1 < 0
    y[left] = ((np.floor((x2[left] + 1) * 4) % 3)).astype(np.int64)

    # top-right: checkerboard
    tr = (x1 >= 0) & (x2 >= 0)
    gx = np.floor((x1[tr] + 0.0) * 6).astype(int)
    gy = np.floor((x2[tr] + 0.0) * 6).astype(int)
    y[tr] = ((gx + gy) % 3).astype(np.int64)

    # bottom-right: rings
    br = (x1 >= 0) & (x2 < 0)
    r = np.sqrt((x1[br] - 0.5) ** 2 + (x2[br] + 0.5) ** 2)
    y[br] = (np.floor(r * 8) % 3).astype(np.int64)

    return y

def make_grid(n=256, lim=1.0):
    xs = np.linspace(-lim, lim, n)
    X1, X2 = np.meshgrid(xs, xs, indexing="xy")
    Y = label_fn(X1, X2)
    return X1, X2, Y

def sample_points(n=20000, lim=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-lim, lim, size=(n,))
    x2 = rng.uniform(-lim, lim, size=(n,))
    y = label_fn(x1, x2)
    X = np.stack([x1, x2], axis=1).astype(np.float32)
    return X, y.astype(np.int64)

def visualize_label_fn(grid_n=256, lim=1.0, points_n=None, seed=0):
    import matplotlib.pyplot as plt

    X1, X2, Y = make_grid(n=grid_n, lim=lim)
    cmap = ListedColormap(["tab:blue", "tab:orange", "tab:green"])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        Y,
        origin="lower",
        extent=(-lim, lim, -lim, lim),
        cmap=cmap,
        interpolation="nearest",
        alpha=0.9,
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("label_fn regions")

    if points_n is not None and points_n > 0:
        X, y = sample_points(n=points_n, lim=lim, seed=seed)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=5, edgecolors="none", alpha=0.6)

    plt.tight_layout()
    return fig, ax

visualize_label_fn(grid_n=256, lim=1.0, points_n=5000, seed=42)
plt.savefig("figures/label_fn_regions.png")