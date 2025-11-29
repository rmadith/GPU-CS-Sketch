#!/usr/bin/env python3
"""
Visualize the different flow distribution types used in benchmarking.

Generates histograms and CDFs for each distribution type.

Usage:
    python benchmark/plot_distributions.py [--n N] [--output OUTPUT_DIR]
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    exit(1)

# Distribution parameters (matching flow_generator.h)
ELEPHANT_WEIGHT_MIN = 10000
ELEPHANT_WEIGHT_MAX = 100000
MICE_WEIGHT_MIN = 1
MICE_WEIGHT_MAX = 100
BIMODAL_LOW_MIN = 1
BIMODAL_LOW_MAX = 10
BIMODAL_HIGH_MIN = 10000
BIMODAL_HIGH_MAX = 50000
ZIPF_ALPHA = 1.2

def generate_uniform(n: int) -> np.ndarray:
    """Generate uniform distribution weights."""
    return np.random.randint(MICE_WEIGHT_MIN, MICE_WEIGHT_MAX + 1, size=n)

def generate_zipf(n: int, alpha: float = ZIPF_ALPHA) -> np.ndarray:
    """Generate Zipf-distributed weights."""
    # Sample ranks from Zipf distribution
    ranks = np.random.zipf(alpha, size=n)
    ranks = np.clip(ranks, 1, 10000)
    
    # Convert ranks to weights (lower rank = higher weight)
    weights = 100000.0 / np.power(ranks.astype(float), alpha)
    weights = np.clip(weights, 1, 100000)
    return weights.astype(int)

def generate_heavy_elephant(n: int) -> np.ndarray:
    """Generate heavy-elephant distribution (10% elephants, 90% mice)."""
    elephant_count = max(1, n // 10)
    mice_count = n - elephant_count
    
    elephants = np.random.randint(ELEPHANT_WEIGHT_MIN, ELEPHANT_WEIGHT_MAX + 1, size=elephant_count)
    mice = np.random.randint(MICE_WEIGHT_MIN, MICE_WEIGHT_MAX + 1, size=mice_count)
    
    return np.concatenate([elephants, mice])

def generate_heavy_mice(n: int) -> np.ndarray:
    """Generate heavy-mice distribution (90% mice, 10% elephants)."""
    mice_count = (n * 9) // 10
    elephant_count = n - mice_count
    
    mice = np.random.randint(MICE_WEIGHT_MIN, MICE_WEIGHT_MAX + 1, size=mice_count)
    elephants = np.random.randint(ELEPHANT_WEIGHT_MIN, ELEPHANT_WEIGHT_MAX + 1, size=elephant_count)
    
    return np.concatenate([mice, elephants])

def generate_bimodal(n: int) -> np.ndarray:
    """Generate bimodal distribution."""
    # 50% in each cluster
    low_count = n // 2
    high_count = n - low_count
    
    low = np.random.randint(BIMODAL_LOW_MIN, BIMODAL_LOW_MAX + 1, size=low_count)
    high = np.random.randint(BIMODAL_HIGH_MIN, BIMODAL_HIGH_MAX + 1, size=high_count)
    
    return np.concatenate([low, high])

DISTRIBUTIONS = {
    "uniform": generate_uniform,
    "zipf": generate_zipf,
    "heavy_elephant": generate_heavy_elephant,
    "heavy_mice": generate_heavy_mice,
    "bimodal": generate_bimodal,
}

DIST_DESCRIPTIONS = {
    "uniform": "Uniform: All flows 1-100",
    "zipf": f"Zipf: Power-law (α={ZIPF_ALPHA})",
    "heavy_elephant": "Heavy Elephant: 10% get 90% traffic",
    "heavy_mice": "Heavy Mice: 90% small flows",
    "bimodal": "Bimodal: Two distinct clusters",
}

def plot_distributions(n: int, output_dir: Path):
    """Generate distribution visualization plots."""
    np.random.seed(42)  # Reproducible
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    
    for idx, (name, generator) in enumerate(DISTRIBUTIONS.items()):
        ax = axes[idx]
        weights = generator(n)
        
        # Histogram
        ax.hist(weights, bins=50, color=colors[idx], alpha=0.7, edgecolor="black", linewidth=0.5)
        
        # Statistics
        stats_text = (
            f"N = {n:,}\n"
            f"Mean = {np.mean(weights):,.0f}\n"
            f"Median = {np.median(weights):,.0f}\n"
            f"Std = {np.std(weights):,.0f}\n"
            f"Min = {np.min(weights):,}\n"
            f"Max = {np.max(weights):,}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Flow Weight")
        ax.set_ylabel("Count")
        ax.set_title(DIST_DESCRIPTIONS[name])
        ax.set_yscale("log")
        
        # Use log scale for x-axis if range is large
        if np.max(weights) > 1000:
            ax.set_xscale("log")
    
    # Hide extra subplot
    axes[-1].axis("off")
    
    # Add summary info
    summary_text = (
        "Flow Weight Distributions for Benchmarking\n\n"
        "• Uniform: Baseline with equal weights\n"
        "• Zipf: Realistic network traffic pattern\n"
        "• Heavy Elephant: Few large flows dominate\n"
        "• Heavy Mice: Many small flows\n"
        "• Bimodal: Two distinct flow size groups"
    )
    axes[-1].text(0.1, 0.5, summary_text, transform=axes[-1].transAxes, fontsize=11,
                  verticalalignment="center", fontfamily="monospace")
    
    plt.suptitle(f"Flow Weight Distributions (N = {n:,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved distribution plot to {output_dir / 'distributions.png'}")
    
    # Also create CDF plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (name, generator) in enumerate(DISTRIBUTIONS.items()):
        weights = generator(n)
        sorted_weights = np.sort(weights)
        cdf = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
        
        ax.plot(sorted_weights, cdf, label=name, color=colors[idx], linewidth=2)
    
    ax.set_xlabel("Flow Weight")
    ax.set_ylabel("CDF")
    ax.set_title(f"Cumulative Distribution Functions (N = {n:,})")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "distributions_cdf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CDF plot to {output_dir / 'distributions_cdf.png'}")
    
    # Traffic proportion plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.15
    x = np.arange(len(DISTRIBUTIONS))
    
    for idx, (name, generator) in enumerate(DISTRIBUTIONS.items()):
        weights = generator(n)
        total = np.sum(weights)
        
        # Sort flows by weight descending
        sorted_weights = np.sort(weights)[::-1]
        
        # Calculate traffic proportion for top 10%, 20%, 50%
        top_10_pct = np.sum(sorted_weights[:n//10]) / total * 100
        top_20_pct = np.sum(sorted_weights[:n//5]) / total * 100
        top_50_pct = np.sum(sorted_weights[:n//2]) / total * 100
        
        ax.bar(idx - bar_width, top_10_pct, bar_width, label="Top 10%" if idx == 0 else "", color="#e74c3c")
        ax.bar(idx, top_20_pct, bar_width, label="Top 20%" if idx == 0 else "", color="#f39c12")
        ax.bar(idx + bar_width, top_50_pct, bar_width, label="Top 50%" if idx == 0 else "", color="#3498db")
    
    ax.set_xlabel("Distribution")
    ax.set_ylabel("% of Total Traffic")
    ax.set_title("Traffic Concentration by Top Flows")
    ax.set_xticks(x)
    ax.set_xticklabels(DISTRIBUTIONS.keys(), rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "traffic_concentration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved traffic concentration plot to {output_dir / 'traffic_concentration.png'}")

def main():
    parser = argparse.ArgumentParser(description="Visualize flow distributions")
    parser.add_argument("--n", type=int, default=10000, help="Number of flows to generate")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent / "results"
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating distribution plots with N = {args.n:,}...")
    plot_distributions(args.n, output_dir)
    print("Done!")

if __name__ == "__main__":
    main()

