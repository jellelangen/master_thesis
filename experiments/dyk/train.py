"""
Training script for SplineTransformer on Dyck-k sequences.

THEORY:
    Dyck-k languages consist of k types of properly matched brackets (e.g., Dyck-2 = "()" and "[]").
    The model is trained with an autoregressive language modeling objective: given a prefix,
    predict the next token. This is a structured prediction task where the grammar constrains
    valid continuations.
    
    The SplineTransformer uses gated MLPs which partition the input space into linear regions.
    This script trains the model and saves checkpoints for later analysis of the spline geometry.

RELEVANT FILES:
    - data/dyk/dyk.py: DyckPCFG class for sequence generation and tokenization
    - architectures/transformers.py: SplineTransformer model architecture
    - experiments/dyk/evaluate.py: Basic evaluation of trained models

CLI ARGUMENTS:
    --steps         Number of training steps (default: 4000)
    --batch_size    Batch size for training (default: 64)
    --d_model       Model embedding dimension (default: 64)
    --n_heads       Number of attention heads (default: 4)
    --d_ff          Feed-forward hidden dimension (default: 256)
    --n_layers      Number of transformer layers (default: 2)
    --lr            Learning rate (default: 3e-4)
    --k             Number of bracket types, e.g., k=2 for "()" and "[]" (default: 2)
    --p_close       Probability of choosing close bracket in PCFG (default: 0.5)
    --seed          Random seed (default: 42)
    --save_path     Path to save checkpoint (default: models/ckpt_dyck.pt)
    --plot          Show training curves after completion
    --plot_path     Save training curves to file
    --val_interval  Validation frequency in steps (default: 100)
    --min_len       Minimum sequence length for training (default: 4)
    --max_len       Maximum sequence length for training (default: 64)
    --model_max_len Positional embedding capacity, set higher than max_len for length generalization
    --mixed         Train on mixed Dyck-2 through Dyck-k (samples from all grammar variants)
    --max_k_train   Max k to use in mixed training (default: k). Set lower for OOD evaluation.

USAGE:
    # Basic training on Dyck-2
    python -m experiments.dyk.train --k=2 --steps=4000
    
    # Mixed training on Dyck-2 through Dyck-8
    python -m experiments.dyk.train --k=8 --mixed --steps=5000 --save_path="models/dyck_mixed.pt"
    
    # Training for epistemic uncertainty (hold out Dyck-7,8)
    python -m experiments.dyk.train --k=8 --mixed --max_k_train=6 --save_path="models/dyck_2to6.pt"
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer


def create_batch(pcfg: DyckPCFG, batch_size: int, min_len: int = 4, max_len: int = 64):
    """
    Generate a batch of Dyck-k sequences as tokenized tensors.
    
    Format with BOS/EOS:
        Input:  [BOS, t_0, t_1, ..., t_{L-1}]
        Target: [t_0, t_1, ..., t_{L-1}, EOS]
    
    Returns:
        x: [B, T] input tokens
        y: [B, T] target tokens  
        lengths: [B] original sequence lengths (including BOS/EOS)
    """
    sequences = []
    while len(sequences) < batch_size:
        seq = pcfg.sample()
        if min_len <= len(seq) <= max_len:
            # Tokenize with BOS and EOS
            tokens = pcfg.tokenize(seq, add_bos=True, add_eos=True)
            sequences.append(tokens)
    
    # Pad to max length in batch
    max_seq_len = max(len(s) for s in sequences)
    
    # Input: [BOS, t_0, ..., t_{L-1}] (drop last token = EOS)
    # Target: [t_0, ..., t_{L-1}, EOS] (drop first token = BOS)
    x = torch.zeros(batch_size, max_seq_len - 1, dtype=torch.long)
    y = torch.zeros(batch_size, max_seq_len - 1, dtype=torch.long)
    lengths = []
    
    for i, seq in enumerate(sequences):
        seq_t = torch.tensor(seq, dtype=torch.long)
        L = len(seq) - 1  # Length of input/target (without one token)
        x[i, :L] = seq_t[:-1]  # [BOS, t_0, ..., t_{L-1}]
        y[i, :L] = seq_t[1:]   # [t_0, ..., t_{L-1}, EOS]
        lengths.append(L)
    
    return x, y, lengths


def create_mixed_batch(pcfg_list: list, batch_size: int, min_len: int = 4, max_len: int = 64):
    """
    Generate a batch from mixed Dyck grammars (Dyck-2 through Dyck-k).
    
    Each sample is drawn from a randomly chosen grammar in pcfg_list.
    All tokenization uses the largest vocabulary (last pcfg in list).
    
    Args:
        pcfg_list: list of DyckPCFG objects for k=2, 3, ..., max_k
        batch_size: number of samples
        min_len, max_len: sequence length constraints
        
    Returns:
        x, y, lengths (same as create_batch)
    """
    import numpy as np
    
    # Use the max-k PCFG for tokenization (has all bracket types)
    pcfg_max = pcfg_list[-1]
    
    sequences = []
    while len(sequences) < batch_size:
        # Randomly pick a grammar
        pcfg = np.random.choice(pcfg_list)
        seq = pcfg.sample()
        if min_len <= len(seq) <= max_len:
            # Tokenize using max vocabulary
            tokens = pcfg_max.tokenize(seq, add_bos=True, add_eos=True)
            sequences.append(tokens)
    
    # Pad to max length in batch
    max_seq_len = max(len(s) for s in sequences)
    
    x = torch.zeros(batch_size, max_seq_len - 1, dtype=torch.long)
    y = torch.zeros(batch_size, max_seq_len - 1, dtype=torch.long)
    lengths = []
    
    for i, seq in enumerate(sequences):
        seq_t = torch.tensor(seq, dtype=torch.long)
        L = len(seq) - 1
        x[i, :L] = seq_t[:-1]
        y[i, :L] = seq_t[1:]
        lengths.append(L)
    
    return x, y, lengths


def compute_accuracy(logits, targets, pad_idx=0):
    """Compute accuracy ignoring padding positions."""
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx
    correct = ((preds == targets) & mask).sum()
    total = mask.sum()
    return correct.float() / total.float() if total > 0 else torch.tensor(0.0)


@torch.no_grad()
def validate(model, pcfg_or_list, vocab_size, device, n_batches=10, batch_size=64, min_len=4, max_len=64, mixed=False):
    """Run validation and return average loss and accuracy."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0.0
    total_acc = 0.0
    
    for _ in range(n_batches):
        if mixed:
            x, y, _ = create_mixed_batch(pcfg_or_list, batch_size, min_len=min_len, max_len=max_len)
        else:
            x, y, _ = create_batch(pcfg_or_list, batch_size, min_len=min_len, max_len=max_len)
        x = x.to(device)
        y = y.to(device)
        
        logits, _ = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        acc = compute_accuracy(logits, y, pad_idx=0)
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    model.train()
    return total_loss / n_batches, total_acc / n_batches


def plot_metrics(history, save_path=None):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    steps = [h["step"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    
    # Loss plot
    axes[0].plot(steps, train_loss, label="Train Loss", alpha=0.8)
    axes[0].plot(steps, val_loss, label="Val Loss", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(steps, train_acc, label="Train Acc", alpha=0.8)
    axes[1].plot(steps, val_acc, label="Val Acc", alpha=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--k", type=int, default=2, help="Number of bracket types")
    parser.add_argument("--p_close", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="models/ckpt_dyck.pt")
    parser.add_argument("--plot", action="store_true", help="Show training plot")
    parser.add_argument("--plot_path", type=str, default=None, help="Save plot to file")
    parser.add_argument("--val_interval", type=int, default=100, help="Validation frequency")
    parser.add_argument("--min_len", type=int, default=4, help="Min training sequence length")
    parser.add_argument("--max_len", type=int, default=64, help="Max training sequence length")
    parser.add_argument("--model_max_len", type=int, default=None, 
                        help="Model positional embedding capacity (default: max_len). Set higher for generalization tests.")
    parser.add_argument("--mixed", action="store_true", 
                        help="Train on mixed Dyck-2 through Dyck-k (all grammar variants)")
    parser.add_argument("--max_k_train", type=int, default=None,
                        help="Max k to use in mixed training (default: k). Set lower than k for OOD evaluation.")
    args = parser.parse_args()
    
    # Default model_max_len to max_len if not specified
    if args.model_max_len is None:
        args.model_max_len = args.max_len
    
    # Default max_k_train to k if not specified
    if args.max_k_train is None:
        args.max_k_train = args.k
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    
    # Vocab size: PAD + BOS + EOS + 2*k brackets = 3 + 2k
    vocab_size = 3 + 2 * args.k
    
    # Create PCFG generators
    if args.mixed:
        # Create PCFGs for k=2 through k=max_k_train
        pcfg_train_list = [DyckPCFG(k=kk, p_close=args.p_close, seed=args.seed + kk) 
                           for kk in range(2, args.max_k_train + 1)]
        pcfg_val_list = [DyckPCFG(k=kk, p_close=args.p_close, seed=args.seed + 1000 + kk) 
                         for kk in range(2, args.max_k_train + 1)]
        print(f"Mixed training: Dyck-2 through Dyck-{args.max_k_train}")
        print(f"Vocabulary: Dyck-{args.k} (tokens for Dyck-{args.max_k_train+1} to Dyck-{args.k} are OOD)")
        pcfg_train = pcfg_train_list[-1]  # For reference
    else:
        pcfg_train = DyckPCFG(k=args.k, p_close=args.p_close, seed=args.seed)
        pcfg_val = DyckPCFG(k=args.k, p_close=args.p_close, seed=args.seed + 1000)
        pcfg_train_list = None
        pcfg_val_list = None
        print(f"Single grammar: Dyck-{args.k}")
    
    print(f"p_close={args.p_close}")
    print(f"Vocabulary: PAD=0, BOS=1, EOS=2, brackets={list(range(3, vocab_size))}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    model = SplineTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_len=args.model_max_len + 2,  # +2 for BOS and EOS
        pad_idx=0,
    ).to(device)
    
    print(f"Training on sequences: len={args.min_len}-{args.max_len}")
    print(f"Model capacity: max_len={args.model_max_len}")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Cross-entropy loss, ignoring padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    pbar = tqdm(range(args.steps))
    
    running_loss = 0.0
    running_acc = 0.0
    log_interval = 50
    
    # History for plotting
    history = []
    
    for step in pbar:
        if args.mixed:
            x, y, lengths = create_mixed_batch(pcfg_train_list, args.batch_size, min_len=args.min_len, max_len=args.max_len)
        else:
            x, y, lengths = create_batch(pcfg_train, args.batch_size, min_len=args.min_len, max_len=args.max_len)
        x = x.to(device)
        y = y.to(device)
        
        logits, _ = model(x)
        
        # Reshape for loss: [B*T, V] vs [B*T]
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        acc = compute_accuracy(logits, y, pad_idx=0)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += acc.item()
        
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            avg_acc = running_acc / log_interval
            pbar.set_description(
                f"step {step+1} | loss: {avg_loss:.4f} | acc: {avg_acc:.3f}"
            )
            running_loss = 0.0
            running_acc = 0.0
        
        # Validation
        if (step + 1) % args.val_interval == 0:
            if args.mixed:
                val_loss, val_acc = validate(
                    model, pcfg_val_list, vocab_size, device, 
                    n_batches=5, batch_size=args.batch_size,
                    min_len=args.min_len, max_len=args.max_len, mixed=True
                )
            else:
                val_loss, val_acc = validate(
                    model, pcfg_val, vocab_size, device, 
                    n_batches=5, batch_size=args.batch_size,
                    min_len=args.min_len, max_len=args.max_len
                )
            
            # Get current train metrics (from last log_interval or current)
            train_loss = loss.item()
            train_acc = acc.item()
            
            history.append({
                "step": step + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
    
    # Save model
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "vocab_size": vocab_size,
        "history": history,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")
    
    # Plot if requested
    if (args.plot or args.plot_path) and len(history) > 0:
        plot_metrics(history, save_path=args.plot_path)
    
    # Quick validation: generate some sequences and check validity
    print("\n=== Validation ===")
    model.eval()
    n_valid = 0
    n_total = 10
    EOS_ID = DyckPCFG.EOS_ID
    BOS_ID = DyckPCFG.BOS_ID
    
    with torch.no_grad():
        for i in range(n_total):
            # Start with BOS token
            prompt = torch.tensor([[BOS_ID]], device=device)
            generated = model.generate(prompt, max_new_tokens=60, temperature=0.8)
            
            # Stop at EOS if present
            gen_tokens = generated[0].cpu().tolist()
            if EOS_ID in gen_tokens:
                gen_tokens = gen_tokens[:gen_tokens.index(EOS_ID)]
            
            gen_seq = pcfg_train.detokenize(gen_tokens)
            is_valid = pcfg_train.is_valid(gen_seq)
            if is_valid:
                n_valid += 1
            if i < 5:
                print(f"  Generated: '{gen_seq}' (len={len(gen_seq)}, valid={is_valid})")
    
    print(f"\nGeneration validity: {n_valid}/{n_total} ({100*n_valid/n_total:.0f}%)")


if __name__ == "__main__":
    main()
