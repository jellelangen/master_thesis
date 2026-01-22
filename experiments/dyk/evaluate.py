"""
Evaluation script for trained SplineTransformer on Dyck-k sequences.

THEORY:
    Tests if the model correctly predicts the final closing bracket of complete Dyck sequences.
    For valid Dyck-k sequences, the last token is always a closing bracket that matches the
    first unclosed open bracket. This is a deterministic task with exactly one correct answer.

RELEVANT FILES:
    - experiments/dyk/train.py: Training script that produces the checkpoints
    - data/dyk/dyk.py: DyckPCFG for sequence generation
    - architectures/transformers.py: SplineTransformer model

CLI ARGUMENTS:
    --checkpoint    Path to model checkpoint (default: models/ckpt_dyck.pt)
    --n_samples     Number of test samples (default: 500)
    --min_len       Minimum sequence length (default: 4)
    --max_len       Maximum sequence length (default: 64, or from checkpoint)
    --model_max_len Override model positional embedding size
    --seed          Random seed (default: 12345)

USAGE:
    python -m experiments.dyk.evaluate --checkpoint=models/ckpt_dyck.pt --n_samples=1000
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data.dyk import DyckPCFG
from architectures.transformers import SplineTransformer


def create_completion_test_set(pcfg: DyckPCFG, n_samples: int, min_len: int = 4, max_len: int = 64):
    """
    Generate test samples where the model must predict the final token.
    
    For each sample:
        - Generate a complete valid Dyck-k sequence
        - Input: [BOS, t_0, ..., t_{n-1}] (all but last bracket)
        - Target: t_n (the final closing bracket)
    
    Returns:
        inputs: list of token lists (prefix without last bracket)
        targets: list of final token IDs
        full_seqs: list of full string sequences
    """
    inputs = []
    targets = []
    full_seqs = []
    
    while len(inputs) < n_samples:
        seq = pcfg.sample()
        if len(seq) < min_len or len(seq) > max_len:
            continue
        
        # Tokenize full sequence with BOS (no EOS - we predict the last bracket)
        tokens = pcfg.tokenize(seq, add_bos=True, add_eos=False)
        
        # Input: all tokens except the last bracket
        # Target: the last bracket (which completes the sequence)
        input_tokens = tokens[:-1]  # [BOS, t_0, ..., t_{n-1}]
        target = tokens[-1]          # t_n (final closing bracket)
        
        inputs.append(input_tokens)
        targets.append(target)
        full_seqs.append(seq)
    
    return inputs, targets, full_seqs


def evaluate(model, inputs, targets, device, batch_size=64):
    """
    Evaluate final-token prediction accuracy.
    """
    model.eval()
    
    all_preds = []
    all_correct = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            # Pad to same length
            max_len = max(len(inp) for inp in batch_inputs)
            x = torch.zeros(len(batch_inputs), max_len, dtype=torch.long, device=device)
            
            for j, inp in enumerate(batch_inputs):
                x[j, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            
            # Forward pass
            logits, _ = model(x)
            
            # Get prediction for last position of each input
            for j, inp in enumerate(batch_inputs):
                last_pos = len(inp) - 1
                pred = logits[j, last_pos].argmax().item()
                target = batch_targets[j]
                
                all_preds.append(pred)
                all_correct.append(pred == target)
    
    accuracy = sum(all_correct) / len(all_correct)
    return accuracy, all_preds, all_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/ckpt_dyck.pt")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--min_len", type=int, default=4, help="Min sequence length for test")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length for test")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    vocab_size = ckpt["vocab_size"]
    
    # Detect max_len from checkpoint
    # State dict has 'pos_emb.weight' of shape [max_len, d_model]
    trained_max_len = ckpt["model_state_dict"]["pos_emb.weight"].shape[0]
    
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model trained with max_len={trained_max_len}, d_model={train_args['d_model']}")
    print(f"Evaluating samples of len={args.min_len}-{args.max_len}")
    
    if args.max_len + 1 > trained_max_len:
        print(f"WARNING: Requested max_len ({args.max_len}) is greater than model capacity ({trained_max_len-1}).")
        print(f"Sampling will be capped at length {trained_max_len-1}.")
        eval_max_len = trained_max_len - 1
    else:
        eval_max_len = args.max_len

    # Create model
    model = SplineTransformer(
        vocab_size=vocab_size,
        d_model=train_args["d_model"],
        n_heads=train_args["n_heads"],
        d_ff=train_args["d_ff"],
        n_layers=train_args["n_layers"],
        max_len=trained_max_len,
        pad_idx=0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Create test PCFG
    np.random.seed(args.seed)
    pcfg = DyckPCFG(k=train_args["k"], p_close=train_args["p_close"], seed=args.seed)
    
    print(f"\nGenerating {args.n_samples} test samples (len {args.min_len}-{eval_max_len})...")
    inputs, targets, full_seqs = create_completion_test_set(
        pcfg, args.n_samples, min_len=args.min_len, max_len=eval_max_len
    )
    
    # Evaluate
    accuracy, predictions, correct = evaluate(model, inputs, targets, device, args.batch_size)
    
    print(f"\n{'='*50}")
    print(f"FINAL TOKEN PREDICTION ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}")
    
    # Show examples
    print(f"\nExamples (predicting last bracket to complete sequence):")
    for i in range(min(10, len(inputs))):
        prefix = pcfg.detokenize(inputs[i])
        target_char = pcfg.detokenize([targets[i]])
        pred_char = pcfg.detokenize([predictions[i]])
        mark = "✓" if correct[i] else "✗"
        print(f"  '{prefix}' + ? → Target: '{target_char}', Pred: '{pred_char}' {mark}")


if __name__ == "__main__":
    main()
