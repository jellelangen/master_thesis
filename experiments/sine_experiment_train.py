# train_sine.py
import torch
import torch.nn as nn
from tqdm import tqdm

from data.builders import sample_sine_batch, make_prefix_targets
from architectures.transformers import TinySeqTransformer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinySeqTransformer(d_model=64, n_heads=4, d_ff=256, n_layers=2, max_len=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    T = 64
    steps = 4000
    batch_size = 256
    L_min, L_max = 8, 48  # vary prefix length

    model.train()
    pbar = tqdm(range(steps))
    for step in pbar:
        y, _ = sample_sine_batch(batch_size=batch_size, T=T, noise_std=0.02)
        # random prefix length per batch (global L for simplicity)
        L = int(torch.randint(low=L_min, high=L_max, size=(1,)).item())
        x, target = make_prefix_targets(y, L=L)

        xb = torch.tensor(x, dtype=torch.float32, device=device)
        tb = torch.tensor(target, dtype=torch.float32, device=device)

        pred, _ = model(xb)
        loss = loss_fn(pred, tb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            pbar.set_description(f"step {step} L={L} loss={loss.item():.6f}")

    torch.save(model.state_dict(), "models/ckpt_sine.pt")
    print("saved models/ckpt_sine.pt")

if __name__ == "__main__":
    main()
