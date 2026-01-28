"""
GSM8K Data Loading and Few-Shot Prompt Formatting.

This module handles loading the GSM8K dataset and formatting prompts with
varying numbers of few-shot examples for the ID vs correctness experiment.

USAGE:
    from experiments.id_correctness.data_loader import GSM8KDataLoader
    
    loader = GSM8KDataLoader(n_samples=500, seed=42)
    prompts = loader.get_prompts_for_sample(idx=0, n_shots=5)
"""

import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class GSM8KSample:
    """A single GSM8K sample with question, answer, and numeric result."""
    question: str
    answer: str  # Full solution with steps
    numeric_answer: float  # Extracted final numeric answer


class GSM8KDataLoader:
    """
    Loads GSM8K dataset and formats prompts with few-shot examples.
    
    The GSM8K dataset contains grade school math word problems with
    step-by-step solutions ending in "#### <number>".
    """
    
    def __init__(
        self,
        subset: str = "main",
        n_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            subset: Dataset config name (default "main")
            n_samples: Number of test samples to use (None = all)
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
        # Load dataset
        self.train_data = load_dataset("gsm8k", subset, split="train")
        self.test_data = load_dataset("gsm8k", subset, split="test")
        
        # Subsample test set if requested
        if n_samples is not None and n_samples < len(self.test_data):
            indices = random.sample(range(len(self.test_data)), n_samples)
            self.test_data = self.test_data.select(indices)
        
        # Parse all samples
        self.train_samples = [self._parse_sample(s) for s in self.train_data]
        self.test_samples = [self._parse_sample(s) for s in self.test_data]
    
    def _parse_sample(self, raw: Dict) -> GSM8KSample:
        """Parse a raw GSM8K sample into structured format."""
        question = raw["question"]
        answer = raw["answer"]
        
        # Extract numeric answer after "####"
        match = re.search(r"####\s*([\d,.-]+)", answer)
        if match:
            num_str = match.group(1).replace(",", "")
            numeric_answer = float(num_str)
        else:
            numeric_answer = float("nan")
        
        return GSM8KSample(
            question=question,
            answer=answer,
            numeric_answer=numeric_answer,
        )
    
    def get_few_shot_examples(self, n_shots: int, exclude_idx: Optional[int] = None) -> List[GSM8KSample]:
        """
        Sample few-shot examples from training set.
        
        Args:
            n_shots: Number of examples to sample
            exclude_idx: Optional index to exclude (for leave-one-out)
        
        Returns:
            List of GSM8KSample objects
        """
        if n_shots == 0:
            return []
        
        indices = list(range(len(self.train_samples)))
        if exclude_idx is not None and exclude_idx < len(indices):
            indices.remove(exclude_idx)
        
        selected = random.sample(indices, min(n_shots, len(indices)))
        return [self.train_samples[i] for i in selected]
    
    def format_prompt(
        self,
        question: str,
        few_shot_examples: List[GSM8KSample],
        include_cot: bool = True,
    ) -> str:
        """
        Format a prompt with few-shot examples.
        
        Args:
            question: The question to answer
            few_shot_examples: List of few-shot examples
            include_cot: Include chain-of-thought in examples
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add few-shot examples
        for ex in few_shot_examples:
            prompt_parts.append(f"Question: {ex.question}")
            if include_cot:
                prompt_parts.append(f"Answer: {ex.answer}")
            else:
                prompt_parts.append(f"Answer: #### {ex.numeric_answer}")
            prompt_parts.append("")  # Empty line separator
        
        # Add the actual question
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")
        
        return "\n".join(prompt_parts)
    
    def get_prompt_for_sample(
        self,
        sample_idx: int,
        n_shots: int,
        include_cot: bool = True,
    ) -> Tuple[str, GSM8KSample]:
        """
        Get formatted prompt and ground truth for a test sample.
        
        Args:
            sample_idx: Index into test set
            n_shots: Number of few-shot examples
            include_cot: Include chain-of-thought in examples
        
        Returns:
            (prompt, sample) tuple
        """
        sample = self.test_samples[sample_idx]
        few_shot = self.get_few_shot_examples(n_shots)
        prompt = self.format_prompt(sample.question, few_shot, include_cot)
        return prompt, sample
    
    def extract_answer_from_generation(self, generation: str) -> Optional[float]:
        """
        Extract numeric answer from model generation.
        
        Looks for "#### <number>" pattern, falling back to last number in text.
        
        Args:
            generation: Model-generated text
        
        Returns:
            Extracted numeric answer or None if not found
        """
        # Try to find #### pattern first
        match = re.search(r"####\s*([\d,.-]+)", generation)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        
        # Fallback: find last number in the text
        numbers = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", generation)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
        
        return None
    
    def check_correctness(
        self,
        predicted: Optional[float],
        ground_truth: float,
        tolerance: float = 1e-5,
    ) -> bool:
        """
        Check if predicted answer is correct.
        
        Args:
            predicted: Predicted numeric answer
            ground_truth: Ground truth numeric answer
            tolerance: Relative tolerance for comparison
        
        Returns:
            True if correct, False otherwise
        """
        if predicted is None:
            return False
        
        if ground_truth == 0:
            return abs(predicted) < tolerance
        
        return abs(predicted - ground_truth) / abs(ground_truth) < tolerance
    
    def __len__(self) -> int:
        """Return number of test samples."""
        return len(self.test_samples)


if __name__ == "__main__":
    # Quick test
    loader = GSM8KDataLoader(n_samples=5, seed=42)
    print(f"Loaded {len(loader)} test samples")
    
    prompt, sample = loader.get_prompt_for_sample(0, n_shots=2)
    print("\n=== Example Prompt (2-shot) ===")
    print(prompt[:500] + "...")
    print(f"\nGround truth: {sample.numeric_answer}")
