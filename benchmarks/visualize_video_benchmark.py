# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Visualization script for video benchmark results.

This script generates comparison charts from benchmark JSON files:
1. Latency comparison bar chart
2. Throughput comparison
3. Latency distribution box plot
4. Performance delta chart

Usage:
    python benchmarks/visualize_video_benchmark.py \
        --results-dir ./video_benchmark_results \
        --output-dir ./video_benchmark_results/plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "hybrid_attention": "#1a7f7a",  # Deep teal
    "standard_attention": "#e86c4a",  # Warm coral
}
CONFIG_LABELS = {
    "hybrid_attention": "Hybrid Attention",
    "standard_attention": "Standard Attention",
}


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all benchmark result JSON files from a directory."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            # Extract config from results if available
            if "results" in data and len(data["results"]) > 0:
                config_name = data["results"][0].get("config", json_file.stem)
                results[config_name] = data["results"][0]
            else:
                config_name = data.get("config", json_file.stem)
                results[config_name] = data

    return results


def plot_latency_comparison(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate latency comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ["avg", "min", "max", "p50", "p90", "p99"]
    labels = ["Average", "Minimum", "Maximum", "P50", "P90", "P99"]
    x = np.arange(len(metrics))
    width = 0.35
    offset = 0

    for config_name, data in results.items():
        values = []
        for m in metrics:
            val = data.get(f"{m}_latency_seconds")
            values.append((val * 1000) if val is not None else 0)  # Convert to ms

        if any(v > 0 for v in values):
            color = COLORS.get(config_name, "#95a5a6")
            label = CONFIG_LABELS.get(config_name, config_name)
            bars = ax.bar(
                x + offset * width - width / 2,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            offset += 1

    ax.set_xlabel("Latency Metric", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Comparison: Hybrid vs Standard Attention", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved latency comparison chart to: {output_path}")


def plot_throughput_comparison(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate throughput comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Total Throughput\n(tokens/sec)", "Generation Speed\n(tokens/sec)"]
    x = np.arange(len(categories))
    width = 0.35
    offset = 0

    for config_name, data in results.items():
        values = [
            data.get("throughput_tokens_per_second", 0) or 0,
            data.get("generation_tokens_per_second", 0) or 0,
        ]

        if any(v > 0 for v in values):
            color = COLORS.get(config_name, "#95a5a6")
            label = CONFIG_LABELS.get(config_name, config_name)
            bars = ax.bar(
                x + offset * width - width / 2,
                values,
                width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )

            offset += 1

    ax.set_ylabel("Tokens per Second", fontsize=12)
    ax.set_title("Throughput Comparison: Hybrid vs Standard Attention", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved throughput comparison chart to: {output_path}")


def plot_latency_distribution(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate latency distribution box plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = []
    all_latencies = []
    colors = []

    for config_name, data in results.items():
        latencies = data.get("all_latencies", [])
        if latencies:
            configs.append(config_name)
            all_latencies.append([l * 1000 for l in latencies])  # Convert to ms
            colors.append(COLORS.get(config_name, "#95a5a6"))

    if not configs:
        ax.text(
            0.5,
            0.5,
            "No latency distribution data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    positions = np.arange(1, len(configs) + 1)
    bp = ax.boxplot(
        all_latencies,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(linewidth=2, color="#2c3e50"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add scatter points for individual data points
    for i, (latencies, color) in enumerate(zip(all_latencies, colors)):
        jitter = np.random.uniform(-0.1, 0.1, len(latencies))
        ax.scatter(
            [positions[i] + j for j in jitter],
            latencies,
            color=color,
            s=50,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )

    labels = [CONFIG_LABELS.get(c, c) for c in configs]
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Distribution", fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved latency distribution chart to: {output_path}")


def plot_performance_delta(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate performance delta chart showing % difference between configs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Need exactly two configs for delta comparison
    if "hybrid_attention" not in results or "standard_attention" not in results:
        ax.text(
            0.5,
            0.5,
            "Need both hybrid and standard attention results\nfor delta comparison",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    hybrid = results["hybrid_attention"]
    standard = results["standard_attention"]

    metrics = {
        "Avg Latency": (
            hybrid.get("avg_latency_seconds", 0),
            standard.get("avg_latency_seconds", 0),
            True,
        ),
        "P99 Latency": (
            hybrid.get("p99_latency_seconds", 0),
            standard.get("p99_latency_seconds", 0),
            True,
        ),
        "Total Throughput": (
            hybrid.get("throughput_tokens_per_second", 0),
            standard.get("throughput_tokens_per_second", 0),
            False,
        ),
        "Gen Speed": (
            hybrid.get("generation_tokens_per_second", 0),
            standard.get("generation_tokens_per_second", 0),
            False,
        ),
    }

    labels = list(metrics.keys())
    deltas = []
    colors = []

    for name, (h_val, s_val, lower_is_better) in metrics.items():
        if s_val and s_val != 0:
            pct_change = ((h_val - s_val) / s_val) * 100
        else:
            pct_change = 0

        if lower_is_better:
            # For latency, negative is good (hybrid is faster)
            is_better = pct_change < 0
        else:
            # For throughput, positive is good (hybrid is faster)
            is_better = pct_change > 0

        deltas.append(pct_change)
        colors.append("#27ae60" if is_better else "#e74c3c")

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, deltas, color=colors, edgecolor="white", linewidth=0.5, height=0.6
    )

    ax.axvline(x=0, color="#2c3e50", linewidth=1.5, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("% Difference (Hybrid vs Standard)", fontsize=12)
    ax.set_title("Performance Delta (Green = Hybrid is Better)", fontsize=14)

    # Add value labels
    for bar, delta in zip(bars, deltas):
        width = bar.get_width()
        label_x = width + 0.1 if width >= 0 else width - 0.1
        ha = "left" if width >= 0 else "right"
        ax.annotate(
            f"{delta:+.2f}%",
            xy=(label_x, bar.get_y() + bar.get_height() / 2),
            ha=ha,
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved performance delta chart to: {output_path}")


def create_summary_table(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Create a summary table in Markdown format."""
    # Get metadata from any result
    sample_result = next(iter(results.values()), {})
    model = sample_result.get("model", "Unknown")
    num_frames = sample_result.get("num_frames", "Unknown")
    total_input_tokens = sample_result.get("total_input_tokens", "Unknown")

    lines = [
        "# Video Benchmark Summary\n",
        f"**Model:** {model}  ",
        f"**Frames:** {num_frames}  ",
        f"**Input Tokens:** {total_input_tokens}\n",
        "## Configuration Comparison\n",
        "| Configuration | Avg Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (tok/s) | Gen Speed (tok/s) |",
        "|--------------|------------------|----------|----------|----------|-------------------|-------------------|",
    ]

    for config_name, data in results.items():
        label = CONFIG_LABELS.get(config_name, config_name)
        avg_latency = (data.get("avg_latency_seconds", 0) or 0) * 1000
        p50_latency = (data.get("p50_latency_seconds", 0) or 0) * 1000
        p90_latency = (data.get("p90_latency_seconds", 0) or 0) * 1000
        p99_latency = (data.get("p99_latency_seconds", 0) or 0) * 1000
        throughput = data.get("throughput_tokens_per_second", 0) or 0
        gen_speed = data.get("generation_tokens_per_second", 0) or 0

        lines.append(
            f"| {label} | {avg_latency:.2f} | {p50_latency:.2f} | {p90_latency:.2f} | {p99_latency:.2f} | {throughput:.1f} | {gen_speed:.1f} |"
        )

    # Add delta section if both configs present
    if "hybrid_attention" in results and "standard_attention" in results:
        hybrid = results["hybrid_attention"]
        standard = results["standard_attention"]

        lines.append("\n## Performance Delta (Hybrid vs Standard)\n")
        lines.append("| Metric | Hybrid | Standard | Δ (%) | Better |")
        lines.append("|--------|--------|----------|-------|--------|")

        comparisons = [
            (
                "Avg Latency (ms)",
                (hybrid.get("avg_latency_seconds", 0) or 0) * 1000,
                (standard.get("avg_latency_seconds", 0) or 0) * 1000,
                True,
            ),
            (
                "Throughput (tok/s)",
                hybrid.get("throughput_tokens_per_second", 0) or 0,
                standard.get("throughput_tokens_per_second", 0) or 0,
                False,
            ),
        ]

        for name, h_val, s_val, lower_is_better in comparisons:
            if s_val != 0:
                delta = ((h_val - s_val) / s_val) * 100
            else:
                delta = 0

            if lower_is_better:
                better = "Hybrid ✓" if delta < 0 else "Standard ✓"
            else:
                better = "Hybrid ✓" if delta > 0 else "Standard ✓"

            lines.append(
                f"| {name} | {h_val:.2f} | {s_val:.2f} | {delta:+.2f}% | {better} |"
            )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved summary table to: {output_path}")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the visualization script."""
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing benchmark result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output plots (defaults to results-dir/plots)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots",
    )


def main(args: argparse.Namespace) -> None:
    """Main visualization entry point."""
    # Set up output directory
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"Found {len(results)} result files: {list(results.keys())}")

    # Generate all plots
    plot_latency_comparison(
        results,
        os.path.join(output_dir, f"latency_comparison.{args.format}"),
    )

    plot_throughput_comparison(
        results,
        os.path.join(output_dir, f"throughput_comparison.{args.format}"),
    )

    plot_latency_distribution(
        results,
        os.path.join(output_dir, f"latency_distribution.{args.format}"),
    )

    plot_performance_delta(
        results,
        os.path.join(output_dir, f"performance_delta.{args.format}"),
    )

    create_summary_table(
        results,
        os.path.join(output_dir, "summary.md"),
    )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize video benchmark results"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

