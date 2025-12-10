import torch
import math
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