import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# from -2π to 2π
# Generate training data
n_train = 5000

x_min = -4 * math.pi
x_max =  4 * math.pi

x_train = torch.rand(n_train, 1) * (x_max - x_min) + x_min
x_train = x_train.to(device)
y_train = torch.sin(x_train)


# Train a simple MLP with one hidden layer
def train_mlp(n_hidden, n_steps=5000, lr=1e-3):
    model = nn.Sequential(
        nn.Linear(1, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1)
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(n_steps):
        idx = torch.randint(0, n_train, (256,))
        x_batch = x_train[idx]
        y_batch = y_train[idx]

        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model

def count_regions(model, x_min=-2*math.pi, x_max=2*math.pi, n_points=5000):
    # extract layers
    lin1, relu, lin2 = model
    # sample points along x-axis
    xs = torch.linspace(x_min, x_max, n_points, device=device).unsqueeze(1)
    with torch.no_grad():
        z = lin1(xs)                     # preactivations (before ReLU), shape [N, H]
        sign_pattern = (z > 0).int()     # 0/1 pattern per hidden unit

    # Count how many times the pattern changes as we move along x
    regions = 1
    for i in range(1, n_points):
        if not torch.equal(sign_pattern[i], sign_pattern[i-1]):
            regions += 1
    return regions

def make_plot(model, title):
    xs = torch.linspace(-2*math.pi, 2*math.pi, 1000, device=device).unsqueeze(1)
    with torch.no_grad():
        y_true = torch.sin(xs)
        y_pred = model(xs)
        err = (y_pred - y_true).abs()
    
    xs_np = xs.cpu().numpy().ravel()
    y_true_np = y_true.cpu().numpy().ravel()
    y_pred_np = y_pred.cpu().numpy().ravel()
    err_np = err.cpu().numpy().ravel()

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(xs_np, y_true_np, label="Ground truth")
    axs[0].plot(xs_np, y_pred_np, linestyle="--", label="MLP")
    axs[0].set_title(title)
    axs[0].legend()

    axs[1].plot(xs_np, err_np)
    axs[1].set_ylabel("Abs error")
    axs[1].set_xlabel("x")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{title}.png", dpi=150)

# 50 hidden units
mlp_50 = train_mlp(50)
regions_50 = count_regions(mlp_50)
print("Hidden=50, approx number of regions:", regions_50)
make_plot(mlp_50, f"Hidden=50, regions≈{regions_50}")

# 500 hidden units
mlp_500 = train_mlp(500)
regions_500 = count_regions(mlp_500)
print("Hidden=500, approx number of regions:", regions_500)
make_plot(mlp_500, f"Hidden=500, regions≈{regions_500}")
