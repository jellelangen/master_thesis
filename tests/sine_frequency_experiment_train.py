import torch
import torch.nn as nn
from tqdm import tqdm

from data.builders import sample_sine_freq_batch, make_prefix
from architectures.transformers import TinySeqTransformerFreq


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_bins = 8
    model = TinySeqTransformerFreq(d_model=64, n_heads=4, d_ff=512, n_layers=4, max_len=128, n_bins=n_bins).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    T = 64
    steps = 6000
    batch_size = 256
    L_min, L_max = 6, 48

    model.train()
    pbar = tqdm(range(steps))
    for step in pbar:
        y, y_cls, _ = sample_sine_freq_batch(batch_size=batch_size, T=T, n_bins=n_bins, noise_std=0.02)
        L = int(torch.randint(low=L_min, high=L_max, size=(1,)).item())
        x = make_prefix(y, L=L)

        xb = torch.tensor(x, dtype=torch.float32, device=device)
        yb = torch.tensor(y_cls, dtype=torch.long, device=device)

        logits, _ = model(xb)
        loss = loss_fn(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == yb).float().mean().item()
            pbar.set_description(f"step {step} L={L} loss={loss.item():.4f} acc={acc:.3f}")

    torch.save(model.state_dict(), "ckpt_sine_freq.pt")
    print("saved ckpt_sine_freq.pt")

if __name__ == "__main__":
    main()