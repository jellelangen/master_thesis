"""
Benchmark SplineLocalComplexitySequence uncertainty estimation on GSM8K.

Usage:
    python benchmark_gsm8k.py
    python benchmark_gsm8k.py --n_samples 200 --device cuda:0 --batch_size 4
"""
import os

import re
import argparse
import numpy as np
import pandas as pd
from typing import Optional
import json
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.estimators import (
    MaximumSequenceProbability,
    MeanTokenEntropy,
    SplineLocalComplexitySequence,
)
from lm_polygraph.generation_metrics import AggregatedMetric
from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.ue_metrics import ROCAUC, PredictionRejectionArea
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.defaults.register_default_stat_calculators import register_default_stat_calculators
from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer
from lm_polygraph.stat_calculators.spline_gate_geometry import (
    SplineGateGeometryCalculator,
)


# ---------------------------------------------------------------------------
# GSM8K exact-match generation metric
# ---------------------------------------------------------------------------

def _extract_final_number(text: str) -> Optional[str]:
    """Extracts answer after '####' (GSM8K format), or last number in string."""
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


class GSM8KExactMatch(GenerationMetric):
    """1.0 if predicted final number matches gold, else 0.0."""

    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return "GSM8K_ExactMatch"

    def __call__(self, stats, target_texts, input_texts=None):
        predictions = stats["greedy_texts"]
        scores = []
        for pred, gold in zip(predictions, target_texts):
            pred_num = _extract_final_number(pred)
            gold_num = _extract_final_number(gold)
            scores.append(1.0 if (pred_num is not None and pred_num == gold_num) else 0.0)
        return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_triviaqa(n_samples: int, seed: int = 42, tokenizer=None):
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", download_mode="reuse_cache_if_exists")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    questions = []
    for row in ds:
        messages = [{"role": "user", "content": f"{row['question']}\nAnswer in a few words."}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        questions.append(prompt)
    answers = [row["answer"]["value"] for row in ds]
    return questions, answers

class TriviaQAExactMatch(GenerationMetric):
    """Case-insensitive exact match after light normalization."""

    def __init__(self):
        super().__init__(["greedy_texts"], "sequence")

    def __str__(self):
        return "TriviaQA_ExactMatch"

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def __call__(self, stats, target_texts, input_texts=None):
        predictions = stats["greedy_texts"]
        scores = []
        for pred, gold in zip(predictions, target_texts):
            # Check if the gold answer appears anywhere in the prediction
            scores.append(1.0 if self._normalize(gold) in self._normalize(pred) else 0.0)
        return np.array(scores, dtype=np.float32)

def load_gsm8k(n_samples: int, seed: int = 42):
    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    questions = [f"Question: {row['question']}\nAnswer:" for row in ds]
    answers = [row["answer"] for row in ds]
    return questions, answers




def compute_prr(uncertainty: np.ndarray, correctness: np.ndarray) -> float:
    """
    Prediction Rejection Ratio: fraction of accuracy gain when rejecting
    the most uncertain 50% of predictions.
    """
    threshold = np.median(uncertainty)
    kept = correctness[uncertainty <= threshold]
    if len(kept) == 0:
        return float("nan")
    baseline = correctness.mean()
    return (kept.mean() - baseline) / (1.0 - baseline + 1e-12)

def build_estimators():
    estimators = []

    sweep_layers = [0, 1, 2, -3, -2, -1]
    r_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    agg_modes = ["mean", "max", "min", "std"]
    default_r = 0.05

    # Layer sweep with default r, all agg modes
    for layer in range(28):
        for agg in agg_modes:
            estimators.append(SplineLocalComplexitySequence(layer_idx=layer, r=default_r, agg=agg))

    # r sweep for first/last 3 layers, all agg modes
    for r in r_values:
        for layer in sweep_layers:
            for agg in agg_modes:
                if r == default_r and layer in [0, 1, 2, 25, 26, 27]:
                    continue
                estimators.append(SplineLocalComplexitySequence(layer_idx=layer, r=r, agg=agg))

    # Baselines
    estimators.append(MaximumSequenceProbability())
    estimators.append(MeanTokenEntropy())
    return estimators


def save_results(man, correctness, args, output_path="results.json"):
    results = {
        "metadata": {
            "model": args.model_path,
            "n_samples": len(man.stats["greedy_texts"]),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
            "accuracy": float(correctness.mean()),
        },
        "metrics": {},
        "per_sample": [],
    }


    for key, value in man.estimations.items():
        level, estimator_name = key
        ue_scores = np.array(value)
        valid = ~np.isnan(ue_scores)
        ue = ue_scores[valid]
        corr = correctness[valid]

        entry = {}
        if corr.std() > 0 and len(corr) >= 2:
            entry["roc_auc"] = float(roc_auc_score(1 - corr, ue))
            threshold = np.median(ue)
            kept = corr[ue <= threshold]
            if len(kept) > 0 and (1.0 - corr.mean()) > 1e-12:
                entry["prr"] = float((kept.mean() - corr.mean()) / (1.0 - corr.mean() + 1e-12))
        results["metrics"][estimator_name] = entry

    # Per-sample data
    greedy_texts = man.stats["greedy_texts"]
    target_texts = man.stats["target_texts"]
    input_texts = man.stats["input_texts"]

    for i in range(len(greedy_texts)):
        sample = {
            "idx": i,
            "input": input_texts[i],
            "target": target_texts[i],
            "generation": greedy_texts[i],
            "correct": int(correctness[i]),
            "scores": {},
        }
        for key, value in man.estimations.items():
            level, estimator_name = key
            v = float(np.array(value)[i])
            sample["scores"][estimator_name] = v if not np.isnan(v) else None
        results["per_sample"].append(sample)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"Loading ({args.n_samples} samples)...")
    # questions, answers = load_gsm8k(args.n_samples, seed=args.seed)
    

    print(f"Loading model: {args.model_path} on {args.device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = WhiteboxModel(base_model, tokenizer, model_path=args.model_path)
    model.generation_parameters.allow_newlines = True
    questions, answers = load_triviaqa(args.n_samples, seed=args.seed, tokenizer=tokenizer)
    dataset = Dataset(
        x=questions,
        y=answers,
        batch_size=args.batch_size,
    )

    # generation_metric = AggregatedMetric(base_metric=GSM8KExactMatch())
    generation_metric = AggregatedMetric(base_metric=TriviaQAExactMatch())
    ue_metrics = [ROCAUC(), PredictionRejectionArea()]

    estimators = build_estimators()
    all_stat_calculators = register_default_stat_calculators("Whitebox")
    builder_env_stat_calc = BuilderEnvironmentStatCalculator(model=model)
    print("Running UEManager...")
    man = UEManager(
        data=dataset,
        model=model,
        estimators=estimators,
        builder_env_stat_calc=builder_env_stat_calc,
        available_stat_calculators=all_stat_calculators,
        generation_metrics=[generation_metric],
        ue_metrics=ue_metrics,
        processors=[],
        ignore_exceptions=False,
    )
    man()




    # Compute correctness from the greedy texts directly
    greedy_texts = man.stats["greedy_texts"]
    target_texts = man.stats["target_texts"]
    metric = TriviaQAExactMatch()
    correctness = metric({"greedy_texts": greedy_texts}, target_texts)

    print(f"\nBaseline accuracy: {correctness.mean():.4f} ({int(correctness.sum())}/{len(correctness)} correct)")

    # Pull results directly from man.metrics
    results = {}
    for key, value in man.metrics.items():
        level, estimator_name, metric_name, ue_metric = key
        if estimator_name not in results:
            results[estimator_name] = {}
        results[estimator_name][ue_metric] = value

    print("\n" + "=" * 65)
    print(f"{'Method':<45} {'ROC-AUC':>8} {'PRR':>8}")
    print("-" * 65)
    for name, metrics in results.items():
        auroc = f"{metrics.get('roc-auc', float('nan')):.4f}"
        prr = f"{metrics.get('prr', float('nan')):.4f}"
        print(f"{name:<45} {auroc:>8} {prr:>8}")
    print("=" * 65)

    output_dir = os.path.join("results", "qwen_exp")
    os.makedirs(output_dir, exist_ok=True)
    save_results(man, correctness, args, output_path=os.path.join(output_dir, f"results_n{len(man.stats['greedy_texts'])}_seed{args.seed}.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--output_csv", type=str, default="results.csv")
    args = parser.parse_args()
    main(args)