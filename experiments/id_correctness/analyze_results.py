"""
Analysis and Visualization for ID vs Correctness Experiment.

Generates Figure 6-style plots showing the correlation between intrinsic dimension
changes and answer correctness across layers and shot counts.

USAGE:
    python -m experiments.id_correctness.analyze_results --results_dir results/id_correctness/20240123_120000
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List


def load_results(results_dir: Path) -> tuple:
    """Load results and summary from experiment output."""
    with open(results_dir / "results.json", "r") as f:
        results = json.load(f)
    
    with open(results_dir / "summary.json", "r") as f:
        summary = json.load(f)
    
    return results, summary


def plot_figure_6_left(summary: Dict, output_dir: Path):
    """
    Plot ID change by layer for different shot counts.
    
    This shows how ID changes relative to 0-shot baseline across layers,
    separated by correct vs incorrect answers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_layers = summary["n_layers"]
    layers = np.arange(n_layers)
    
    id_change_data = summary.get("mean_id_change_correct_vs_incorrect", {})
    
    # Colors for different shot counts
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(id_change_data)))
    
    # Left: Correct answers
    ax = axes[0]
    for (n_shots, data), color in zip(id_change_data.items(), colors):
        if "correct" in data:
            ax.plot(layers, data["correct"], 
                   label=f"{n_shots}-shot", color=color, linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("ID Change (relative to 0-shot)", fontsize=12)
    ax.set_title("Correct Answers", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Right: Incorrect answers
    ax = axes[1]
    for (n_shots, data), color in zip(id_change_data.items(), colors):
        if "incorrect" in data:
            ax.plot(layers, data["incorrect"], 
                   label=f"{n_shots}-shot", color=color, linewidth=2)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("ID Change (relative to 0-shot)", fontsize=12)
    ax.set_title("Incorrect Answers", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.suptitle("ID Change by Layer: Correct vs Incorrect Answers\n(Replication of Figure 6)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_6_left.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "figure_6_left.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'figure_6_left.png'}")


def plot_figure_6_right(results: List[Dict], summary: Dict, output_dir: Path):
    """
    Plot accuracy vs ID change in final layers.
    
    Shows the correlation between ID increase in final layers and accuracy.
    """
    n_layers = summary["n_layers"]
    shot_counts = summary["shot_counts"]
    
    # Consider "final layers" as last 25% of layers
    final_layer_start = int(n_layers * 0.75)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    shot_data = []
    for n_shots in shot_counts:
        if n_shots == 0:
            continue
        
        accuracy = summary["accuracy_by_shots"].get(str(n_shots), 0)
        
        # Mean ID change in final layers for this shot count
        id_changes = summary.get("mean_id_change_correct_vs_incorrect", {}).get(str(n_shots), {})
        
        # Combine correct and incorrect for overall mean
        correct_changes = id_changes.get("correct", [0] * n_layers)
        incorrect_changes = id_changes.get("incorrect", [0] * n_layers)
        
        # Weight by proportion correct/incorrect
        correct_prop = accuracy
        incorrect_prop = 1 - accuracy
        
        if len(correct_changes) > final_layer_start:
            mean_final_id_change = (
                np.mean(correct_changes[final_layer_start:]) * correct_prop +
                np.mean(incorrect_changes[final_layer_start:]) * incorrect_prop
            )
        else:
            mean_final_id_change = 0
        
        shot_data.append((n_shots, accuracy, mean_final_id_change))
    
    if shot_data:
        shots, accs, id_changes = zip(*shot_data)
        
        ax.scatter(id_changes, accs, c=shots, cmap='viridis', s=100, edgecolors='black')
        
        # Add labels
        for n_shots, acc, id_change in shot_data:
            ax.annotate(f"{n_shots}-shot", (id_change, acc), 
                       textcoords="offset points", xytext=(5, 5), fontsize=9)
        
        # Compute correlation
        if len(id_changes) > 2:
            rho, p_value = stats.spearmanr(id_changes, accs)
            ax.set_title(f"Accuracy vs Final Layer ID Change\n(Spearman ρ={rho:.3f}, p={p_value:.3f})", 
                        fontsize=14)
        else:
            ax.set_title("Accuracy vs Final Layer ID Change", fontsize=14)
    
    ax.set_xlabel("Mean ID Change in Final Layers", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figure_6_right.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "figure_6_right.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'figure_6_right.png'}")


def plot_accuracy_by_shots(summary: Dict, output_dir: Path):
    """Plot accuracy progression by shot count."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    shot_counts = []
    accuracies = []
    
    for n_shots, acc in summary["accuracy_by_shots"].items():
        shot_counts.append(int(n_shots))
        accuracies.append(acc)
    
    # Sort by shot count
    sorted_data = sorted(zip(shot_counts, accuracies))
    shot_counts, accuracies = zip(*sorted_data) if sorted_data else ([], [])
    
    ax.plot(shot_counts, accuracies, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel("Number of Shots", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("GSM8K Accuracy by Few-Shot Count", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_shots.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'accuracy_by_shots.png'}")


def plot_id_heatmap(summary: Dict, output_dir: Path):
    """Plot heatmap of mean ID by layer and shot count."""
    id_by_shots = summary.get("mean_id_by_shots_and_layer", {})
    
    if not id_by_shots:
        print("No ID data available for heatmap")
        return
    
    # Build matrix
    shot_counts = sorted([int(k) for k in id_by_shots.keys()])
    n_layers = len(list(id_by_shots.values())[0])
    
    matrix = np.zeros((len(shot_counts), n_layers))
    for i, n_shots in enumerate(shot_counts):
        matrix[i] = id_by_shots[str(n_shots)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Number of Shots", fontsize=12)
    ax.set_yticks(range(len(shot_counts)))
    ax.set_yticklabels(shot_counts)
    ax.set_title("Mean Intrinsic Dimension by Layer and Shot Count", fontsize=14)
    
    plt.colorbar(im, ax=ax, label="Intrinsic Dimension")
    
    plt.tight_layout()
    plt.savefig(output_dir / "id_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'id_heatmap.png'}")


def compute_layer_correlations(results: List[Dict], summary: Dict) -> Dict:
    """Compute correlation between ID change and correctness per layer."""
    n_layers = summary["n_layers"]
    shot_counts = [s for s in summary["shot_counts"] if s > 0]
    
    correlations = {}
    
    for n_shots in shot_counts:
        layer_correlations = []
        
        for layer_idx in range(n_layers):
            id_changes = []
            correctness = []
            
            for r in results:
                if str(n_shots) in r.get("id_change_per_layer", {}):
                    id_change = r["id_change_per_layer"][str(n_shots)]
                    if layer_idx < len(id_change):
                        id_changes.append(id_change[layer_idx])
                        correctness.append(1 if r["correctness"].get(str(n_shots), False) else 0)
            
            if len(set(correctness)) > 1 and len(id_changes) > 2:
                rho, p = stats.pointbiserialr(correctness, id_changes)
                layer_correlations.append({"rho": rho, "p": p})
            else:
                layer_correlations.append({"rho": 0, "p": 1})
        
        correlations[n_shots] = layer_correlations
    
    return correlations


def plot_correlation_by_layer(results: List[Dict], summary: Dict, output_dir: Path):
    """Plot point-biserial correlation between ID change and correctness by layer."""
    correlations = compute_layer_correlations(results, summary)
    
    if not correlations:
        print("No correlation data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_layers = summary["n_layers"]
    layers = np.arange(n_layers)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(correlations)))
    
    for (n_shots, layer_corrs), color in zip(correlations.items(), colors):
        rhos = [lc["rho"] for lc in layer_corrs]
        ax.plot(layers, rhos, label=f"{n_shots}-shot", color=color, linewidth=2)
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Point-Biserial Correlation (ID change ↔ Correctness)", fontsize=12)
    ax.set_title("Correlation Between ID Change and Answer Correctness by Layer", fontsize=14)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_by_layer.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "correlation_by_layer.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'correlation_by_layer.png'}")


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(description="Analyze ID vs Correctness results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory (contains results.json and summary.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results_dir)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    results, summary = load_results(results_dir)
    
    print(f"Generating plots...")
    print(f"  Samples: {summary['n_samples']}")
    print(f"  Layers: {summary['n_layers']}")
    print(f"  Shot counts: {summary['shot_counts']}")
    
    # Generate all plots
    plot_accuracy_by_shots(summary, output_dir)
    plot_id_heatmap(summary, output_dir)
    plot_figure_6_left(summary, output_dir)
    plot_figure_6_right(results, summary, output_dir)
    plot_correlation_by_layer(results, summary, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
