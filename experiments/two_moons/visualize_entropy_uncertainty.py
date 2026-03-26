import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def make_two_moons(n_samples: int = 1200, noise: float = 0.10, seed: int = 42):
    """Generate a two-moons dataset without external dependencies."""
    rng = np.random.default_rng(seed)

    n0 = n_samples // 2
    n1 = n_samples - n0

    t0 = rng.uniform(0.0, np.pi, n0)
    t1 = rng.uniform(0.0, np.pi, n1)

    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    x1 = np.stack([1.0 - np.cos(t1), -np.sin(t1) + 0.5], axis=1)

    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)], axis=0)

    x += rng.normal(0.0, noise, x.shape)

    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probs, eps, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


@torch.no_grad()
def relu_region_ids(model: nn.Module, grid: np.ndarray, device: torch.device) -> np.ndarray:
    """Return integer region IDs from hidden ReLU activation patterns."""
    xg = torch.tensor(grid, dtype=torch.float32, device=device)

    l1 = model.net[0]
    l2 = model.net[2]

    z1 = l1(xg)
    h1 = torch.relu(z1)
    z2 = l2(h1)

    # Activation pattern defines a linear region for a ReLU MLP.
    pattern = torch.cat([(z1 > 0), (z2 > 0)], dim=1).cpu().numpy()
    _, inv = np.unique(pattern, axis=0, return_inverse=True)
    return inv


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = make_two_moons(n_samples=2000, noise=0.12, seed=42)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(2500):
        idx = torch.randint(0, x_tensor.shape[0], (256,), device=device)
        xb = x_tensor[idx]
        yb = y_tensor[idx]

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:
            print(f"step {step + 1:4d} | loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits_all = model(x_tensor)
        train_acc = (logits_all.argmax(dim=1) == y_tensor).float().mean().item()
    print(f"train accuracy: {100.0 * train_acc:.2f}%")

    pad = 0.8
    x_min, x_max = x[:, 0].min() - pad, x[:, 0].max() + pad
    y_min, y_max = x[:, 1].min() - pad, x[:, 1].max() + pad

    n_grid = 300
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
    )
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    with torch.no_grad():
        grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
        logits = model(grid_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    p_class1 = probs[:, 1].reshape(gx.shape)
    entropy = predictive_entropy(probs).reshape(gx.shape)

    region_ids = relu_region_ids(model, grid, device=device).reshape(gx.shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].contourf(gx, gy, p_class1, levels=100, cmap="coolwarm", vmin=0.0, vmax=1.0)
    axes[0].scatter(x[y == 0, 0], x[y == 0, 1], s=12, c="tab:blue", edgecolor="k", linewidth=0.2, alpha=0.8, label="Class 0")
    axes[0].scatter(x[y == 1, 0], x[y == 1, 1], s=12, c="tab:orange", edgecolor="k", linewidth=0.2, alpha=0.8, label="Class 1")
    axes[0].set_title("Predicted probability p(y=1|x)")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].contourf(gx, gy, entropy, levels=100, cmap="viridis")
    axes[1].scatter(x[:, 0], x[:, 1], s=8, c="white", edgecolor="k", linewidth=0.2, alpha=0.5)
    axes[1].set_title("Predictive entropy uncertainty")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Entropy (nats)")

    im2 = axes[2].imshow(
        region_ids,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="tab20",
        interpolation="nearest",
        aspect="auto",
        alpha=0.85,
    )
    axes[2].contour(gx, gy, p_class1, levels=[0.5], colors="black", linewidths=2.0)
    axes[2].scatter(x[y == 0, 0], x[y == 0, 1], s=10, c="tab:blue", edgecolor="k", linewidth=0.2, alpha=0.7)
    axes[2].scatter(x[y == 1, 0], x[y == 1, 1], s=10, c="tab:orange", edgecolor="k", linewidth=0.2, alpha=0.7)
    axes[2].set_title("SplineCAM-style regions + decision boundary")
    axes[2].set_xlabel("x1")
    axes[2].set_ylabel("x2")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Region ID")

    out_dir = os.path.join("figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "two_moons_entropy_uncertainty_regions.png")
    plt.savefig(out_path, dpi=180)
    print(f"saved figure to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
