"""
Dyck-k data module.

Provides PCFG-based generation of balanced parentheses sequences.
"""

from .dyk import (
    DyckPCFG,
    DyckConfig,
    BRACKET_PAIRS,
    sample_dyck_dataset,
    sample_dyck_next_token_dataset,
)

__all__ = [
    "DyckPCFG",
    "DyckConfig",
    "BRACKET_PAIRS",
    "sample_dyck_dataset",
    "sample_dyck_next_token_dataset",
]
