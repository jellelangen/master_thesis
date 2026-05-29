"""
Dyck-k Probabilistic Context-Free Grammar (PCFG) Generator.

The Dyck-k grammar generates balanced parentheses with k bracket types.
Production rules:
    S → ε           with probability p_close
    S → (i S )i S   with probability (1 - p_close) / k   for each bracket type i ∈ {1, ..., k}

This creates hierarchically nested structures that require context-free parsing.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# Default bracket pairs for Dyck-k
BRACKET_PAIRS = [
    ('(', ')'),
    ('[', ']'),
    ('{', '}'),
    ('<', '>'),
    ('⟨', '⟩'),
    ('【', '】'),
    ('〈', '〉'),
    ('《', '》'),
]


@dataclass
class DyckConfig:
    """Configuration for Dyck-k PCFG generation."""
    k: int = 2                      # Number of bracket types
    p_close: float = 0.5            # Probability of closing (S → ε)
    max_depth: int = 100            # Maximum recursion depth
    max_length: int = 1000          # Maximum sequence length
    brackets: Optional[List[Tuple[str, str]]] = None  # Custom bracket pairs
    
    def __post_init__(self):
        if self.brackets is None:
            self.brackets = BRACKET_PAIRS[:self.k]
        if len(self.brackets) < self.k:
            raise ValueError(f"Need at least {self.k} bracket pairs, got {len(self.brackets)}")
        self.brackets = self.brackets[:self.k]


class DyckPCFG:
    """
    Dyck-k Probabilistic Context-Free Grammar generator.
    
    Generates balanced parentheses sequences with k different bracket types
    according to a probabilistic context-free grammar.
    
    Production rules:
        S → ε           with probability p_close
        S → (i S )i S   with probability (1 - p_close) / k
    
    Example:
        >>> pcfg = DyckPCFG(k=2, p_close=0.5)
        >>> sequence = pcfg.sample()
        >>> print(sequence)  # e.g., "([])[]"
        >>> assert pcfg.is_valid(sequence)
    """
    
    def __init__(
        self,
        k: int = 2,
        p_close: float = 0.5,
        max_depth: int = 100,
        max_length: int = 1000,
        brackets: Optional[List[Tuple[str, str]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Dyck-k PCFG.
        
        Args:
            k: Number of bracket types (default: 2)
            p_close: Probability of terminating recursion S → ε (default: 0.5)
            max_depth: Maximum recursion depth to prevent stack overflow
            max_length: Maximum sequence length to prevent runaway generation
            brackets: Custom list of (open, close) bracket pairs
            seed: Random seed for reproducibility
        """
        self.config = DyckConfig(
            k=k,
            p_close=p_close,
            max_depth=max_depth,
            max_length=max_length,
            brackets=brackets,
        )
        self.rng = np.random.default_rng(seed)
        
        # Pre-compute probabilities for each production rule
        # P(S → ε) = p_close
        # P(S → (i S )i S) = (1 - p_close) / k for each i
        self.p_expand = (1 - self.config.p_close) / self.config.k
        
        # Build lookup tables for validation
        self._open_brackets = {b[0] for b in self.config.brackets}
        self._close_brackets = {b[1] for b in self.config.brackets}
        self._bracket_match = {b[0]: b[1] for b in self.config.brackets}
        self._bracket_match_reverse = {b[1]: b[0] for b in self.config.brackets}
    
    def _sample_production(self) -> Optional[int]:
        """
        Sample a production rule.
        
        Returns:
            None if S → ε, or bracket index i if S → (i S )i S
        """
        r = self.rng.random()
        if r < self.config.p_close:
            return None  # S → ε
        else:
            # Choose which bracket type
            bracket_idx = int((r - self.config.p_close) / self.p_expand)
            return min(bracket_idx, self.config.k - 1)
    
    def sample(self) -> str:
        """
        Sample a string from the Dyck-k PCFG.
        
        Returns:
            A balanced parentheses string
        """
        tokens = []
        self._generate(tokens, depth=0)
        return ''.join(tokens)
    
    def _generate(self, tokens: List[str], depth: int) -> None:
        """
        Recursively generate tokens according to the PCFG.
        
        Args:
            tokens: List to append tokens to
            depth: Current recursion depth
        """
        # Safety checks
        if depth >= self.config.max_depth or len(tokens) >= self.config.max_length:
            return
        
        production = self._sample_production()
        
        if production is None:
            # S → ε
            return
        else:
            # S → (i S )i S
            open_bracket, close_bracket = self.config.brackets[production]
            tokens.append(open_bracket)
            self._generate(tokens, depth + 1)  # Inner S
            tokens.append(close_bracket)
            self._generate(tokens, depth)       # Continuation S
    
    def sample_batch(self, batch_size: int) -> List[str]:
        """
        Sample a batch of strings from the Dyck-k PCFG.
        
        Args:
            batch_size: Number of strings to generate
            
        Returns:
            List of balanced parentheses strings
        """
        return [self.sample() for _ in range(batch_size)]
    
    def sample_tokens(self) -> List[str]:
        """
        Sample a string and return as a list of tokens.
        
        Returns:
            List of bracket tokens
        """
        tokens = []
        self._generate(tokens, depth=0)
        return tokens
    
    def sample_with_depth(self) -> Tuple[str, int]:
        """
        Sample a string and return the maximum nesting depth.
        
        Returns:
            Tuple of (string, max_depth)
        """
        tokens = []
        max_depth = self._generate_with_depth(tokens, depth=0)
        return ''.join(tokens), max_depth
    
    def _generate_with_depth(self, tokens: List[str], depth: int) -> int:
        """Generate tokens and track maximum depth."""
        if depth >= self.config.max_depth or len(tokens) >= self.config.max_length:
            return depth
        
        production = self._sample_production()
        
        if production is None:
            return depth
        else:
            open_bracket, close_bracket = self.config.brackets[production]
            tokens.append(open_bracket)
            inner_max = self._generate_with_depth(tokens, depth + 1)
            tokens.append(close_bracket)
            cont_max = self._generate_with_depth(tokens, depth)
            return max(inner_max, cont_max)
    
    def is_valid(self, s: str) -> bool:
        """
        Check if a string is a valid Dyck-k sequence.
        
        Args:
            s: String to validate
            
        Returns:
            True if the string is balanced
        """
        stack = []
        for char in s:
            if char in self._open_brackets:
                stack.append(char)
            elif char in self._close_brackets:
                if not stack:
                    return False
                expected_open = self._bracket_match_reverse[char]
                if stack[-1] != expected_open:
                    return False
                stack.pop()
            # Ignore other characters
        return len(stack) == 0
    
    # Special token IDs
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    
    def tokenize(self, s: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert a Dyck-k string to integer tokens.
        
        Token encoding:
            0: PAD (padding)
            1: BOS (beginning of sequence)
            2: EOS (end of sequence)
            3 to k+2: open brackets
            k+3 to 2k+2: close brackets
        
        Args:
            s: Dyck-k string
            add_bos: if True, prepend BOS token
            add_eos: if True, append EOS token
            
        Returns:
            List of integer tokens
        """
        tokens = []
        if add_bos:
            tokens.append(self.BOS_ID)
        
        for char in s:
            for i, (open_b, close_b) in enumerate(self.config.brackets):
                if char == open_b:
                    tokens.append(i + 3)  # 3-indexed open (after PAD, BOS, EOS)
                    break
                elif char == close_b:
                    tokens.append(self.config.k + i + 3)  # k+3 indexed close
                    break
        
        if add_eos:
            tokens.append(self.EOS_ID)
        
        return tokens
    
    def detokenize(self, tokens: List[int], strip_special: bool = True) -> str:
        """
        Convert integer tokens back to a Dyck-k string.
        
        Args:
            tokens: List of integer tokens
            strip_special: if True, skip PAD/BOS/EOS tokens
            
        Returns:
            Dyck-k string
        """
        chars = []
        for token in tokens:
            if strip_special and token in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            elif 3 <= token <= self.config.k + 2:
                chars.append(self.config.brackets[token - 3][0])
            elif self.config.k + 3 <= token <= 2 * self.config.k + 2:
                chars.append(self.config.brackets[token - self.config.k - 3][1])
        return ''.join(chars)
    
    def vocab_size(self) -> int:
        """Return total vocabulary size: PAD + BOS + EOS + 2*k brackets."""
        return 3 + 2 * self.config.k
    
    def get_vocabulary(self) -> List[str]:
        """
        Get the vocabulary (all possible tokens).
        
        Returns:
            List of bracket characters
        """
        vocab = []
        for open_b, close_b in self.config.brackets:
            vocab.extend([open_b, close_b])
        return vocab


def sample_dyck_dataset(
    n_samples: int,
    k: int = 2,
    p_close: float = 0.5,
    max_length: int = 100,
    min_length: int = 2,
    seed: Optional[int] = None,
    return_tokens: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of Dyck-k sequences.
    
    Args:
        n_samples: Number of sequences to generate
        k: Number of bracket types
        p_close: Probability of closing (higher = shorter sequences)
        max_length: Maximum sequence length
        min_length: Minimum sequence length (resample if shorter)
        seed: Random seed
        return_tokens: If True, return integer tokens; else return strings
        
    Returns:
        sequences: Array of sequences (strings or padded token arrays)
        lengths: Array of sequence lengths
    """
    pcfg = DyckPCFG(k=k, p_close=p_close, max_length=max_length, seed=seed)
    
    sequences = []
    lengths = []
    
    while len(sequences) < n_samples:
        if return_tokens:
            tokens = pcfg.sample_tokens()
            if len(tokens) >= min_length:
                sequences.append(tokens)
                lengths.append(len(tokens))
        else:
            s = pcfg.sample()
            if len(s) >= min_length:
                sequences.append(s)
                lengths.append(len(s))
    
    lengths = np.array(lengths)
    
    if return_tokens:
        # Pad to max length
        max_len = max(len(s) for s in sequences)
        padded = np.zeros((n_samples, max_len), dtype=np.int64)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded, lengths
    else:
        return np.array(sequences, dtype=object), lengths


def sample_dyck_next_token_dataset(
    n_samples: int,
    k: int = 2,
    p_close: float = 0.5,
    seq_length: int = 50,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset for next-token prediction on Dyck-k sequences.
    
    For each sample:
        - Generate a Dyck-k sequence of at least seq_length + 1 tokens
        - x = first seq_length tokens
        - y = token at position seq_length (next token)
    
    Args:
        n_samples: Number of samples
        k: Number of bracket types
        p_close: Probability of closing
        seq_length: Context length
        seed: Random seed
        
    Returns:
        X: [n_samples, seq_length] input tokens
        y: [n_samples] target tokens
    """
    pcfg = DyckPCFG(k=k, p_close=p_close, max_length=seq_length * 3, seed=seed)
    
    X = np.zeros((n_samples, seq_length), dtype=np.int64)
    y = np.zeros(n_samples, dtype=np.int64)
    
    i = 0
    while i < n_samples:
        seq = pcfg.sample()
        if len(seq) > seq_length:
            int_tokens = pcfg.tokenize(seq)
            X[i] = int_tokens[:seq_length]
            y[i] = int_tokens[seq_length]
            i += 1
    
    return X, y


if __name__ == "__main__":
    # Demo usage
    print("=== Dyck-k PCFG Demo ===\n")
    
    # Create a Dyck-2 grammar
    pcfg = DyckPCFG(k=2, p_close=0.6, seed=42)
    
    print(f"Vocabulary: {pcfg.get_vocabulary()}")
    print(f"Bracket pairs: {pcfg.config.brackets}")
    print()
    
    # Generate some samples
    print("Sample sequences:")
    for i in range(5):
        seq, depth = pcfg.sample_with_depth()
        valid = pcfg.is_valid(seq)
        tokens = pcfg.tokenize(seq)
        print(f"  {i+1}. '{seq}' (len={len(seq)}, depth={depth}, valid={valid})")
        print(f"      tokens: {tokens}")
    
    print("\n--- Dataset Generation ---")
    
    # Generate a small dataset
    seqs, lens = sample_dyck_dataset(
        n_samples=10,
        k=2,
        p_close=0.6,
        min_length=4,
        seed=42,
    )
    print(f"Generated {len(seqs)} sequences")
    print(f"Length stats: min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}")
    
    # Generate tokenized dataset
    X, lens = sample_dyck_dataset(
        n_samples=5,
        k=2,
        p_close=0.6,
        min_length=4,
        return_tokens=True,
        seed=42,
    )
    print(f"\nTokenized shape: {X.shape}")
    print(f"First sequence tokens: {X[0, :lens[0]]}")
