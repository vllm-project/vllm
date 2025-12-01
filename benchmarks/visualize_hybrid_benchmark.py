# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Visualization script for hybrid attention benchmark results.

This script generates comparison charts from benchmark JSON files:
1. Memory comparison bar chart
2. Throughput comparison across input lengths
3. Latency percentile comparison
4. Scaling analysis

Usage:
    python benchmarks/visualize_hybrid_benchmark.py \
        --results-dir ./hybrid_benchmark_results \
        --output-dir ./hybrid_benchmark_results/plots
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
    "full_attention": "#2ecc71",  # Green
    "sliding_window_only": "#3498db",  # Blue
    "hybrid_ssm_sliding": "#9b59b6",  # Purple
}
CONFIG_LABELS = {
    "full_attention": "Full Attention",
    "sliding_window_only": "Sliding Window",
    "hybrid_ssm_sliding": "Hybrid (SSM + SW)",
}


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all benchmark result JSON files from a directory."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            config_name = data.get("config", json_file.stem)
            results[config_name] = data

    return results


def plot_memory_comparison(
    results: dict[str, dict],
    output_path: str,
    input_length: str = "1024",
) -> None:
    """Generate memory comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = []
    kv_cache_memory = []
    model_memory = []

    for config_name, data in results.items():
        if "by_input_length" not in data:
            continue

        length_data = data["by_input_length"].get(input_length, {})
        if "error" in length_data:
            continue

        memory_info = length_data.get("memory", {})
        kv_mem = memory_info.get("kv_cache_memory_gib")
        model_mem = memory_info.get("model_memory_gib")

        if kv_mem is not None or model_mem is not None:
            configs.append(config_name)
            kv_cache_memory.append(kv_mem or 0)
            model_memory.append(model_mem or 0)

    if not configs:
        print("No memory data available for plotting")
        return

    x = np.arange(len(configs))
    width = 0.35

    colors = [COLORS.get(c, "#95a5a6") for c in configs]
    labels = [CONFIG_LABELS.get(c, c) for c in configs]

    bars1 = ax.bar(
        x - width / 2,
        model_memory,
        width,
        label="Model Memory",
        color=colors,
        alpha=0.7,
    )
    bars2 = ax.bar(
        x + width / 2,
        kv_cache_memory,
        width,
        label="KV Cache Memory",
        color=colors,
        alpha=1.0,
    )

    ax.set_ylabel("Memory (GiB)", fontsize=12)
    ax.set_title(
        f"Memory Footprint Comparison (Input Length: {input_length})", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved memory comparison chart to: {output_path}")


def plot_throughput_comparison(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate throughput comparison line chart across input lengths."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for config_name, data in results.items():
        if "by_input_length" not in data:
            continue

        lengths = []
        throughputs = []

        for length, length_data in sorted(
            data["by_input_length"].items(), key=lambda x: int(x[0])
        ):
            if "error" in length_data:
                continue
            throughput = length_data.get("throughput", {}).get("tokens_per_second")
            if throughput is not None:
                lengths.append(int(length))
                throughputs.append(throughput)

        if lengths:
            color = COLORS.get(config_name, "#95a5a6")
            label = CONFIG_LABELS.get(config_name, config_name)
            ax.plot(
                lengths,
                throughputs,
                marker="o",
                linewidth=2,
                markersize=8,
                color=color,
                label=label,
            )

    ax.set_xlabel("Input Length (tokens)", fontsize=12)
    ax.set_ylabel("Throughput (tokens/second)", fontsize=12)
    ax.set_title("Throughput vs Input Length", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis if range is large
    if ax.get_xlim()[1] / ax.get_xlim()[0] > 10:
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved throughput comparison chart to: {output_path}")


def plot_latency_percentiles(
    results: dict[str, dict],
    output_path: str,
    input_length: str = "1024",
) -> None:
    """Generate latency percentile comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    percentiles = ["avg", "p50", "p90", "p99"]
    x = np.arange(len(percentiles))
    width = 0.25
    offset = 0

    for config_name, data in results.items():
        if "by_input_length" not in data:
            continue

        length_data = data["by_input_length"].get(input_length, {})
        if "error" in length_data:
            continue

        latency_data = length_data.get("latency", {})
        values = []
        for p in percentiles:
            key = f"{p}_seconds"
            val = latency_data.get(key)
            values.append((val * 1000) if val is not None else 0)  # Convert to ms

        if any(v > 0 for v in values):
            color = COLORS.get(config_name, "#95a5a6")
            label = CONFIG_LABELS.get(config_name, config_name)
            ax.bar(x + offset * width, values, width, label=label, color=color)
            offset += 1

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Latency Distribution (Input Length: {input_length})", fontsize=14)
    ax.set_xticks(x + width * (offset - 1) / 2)
    ax.set_xticklabels(["Average", "P50", "P90", "P99"], fontsize=11)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved latency percentiles chart to: {output_path}")


def plot_latency_vs_input_length(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate latency vs input length comparison chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for config_name, data in results.items():
        if "by_input_length" not in data:
            continue

        lengths = []
        latencies = []

        for length, length_data in sorted(
            data["by_input_length"].items(), key=lambda x: int(x[0])
        ):
            if "error" in length_data:
                continue
            latency = length_data.get("latency", {}).get("avg_seconds")
            if latency is not None:
                lengths.append(int(length))
                latencies.append(latency * 1000)  # Convert to ms

        if lengths:
            color = COLORS.get(config_name, "#95a5a6")
            label = CONFIG_LABELS.get(config_name, config_name)
            ax.plot(
                lengths,
                latencies,
                marker="s",
                linewidth=2,
                markersize=8,
                color=color,
                label=label,
            )

    ax.set_xlabel("Input Length (tokens)", fontsize=12)
    ax.set_ylabel("Average Latency (ms)", fontsize=12)
    ax.set_title("Latency vs Input Length", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if ax.get_xlim()[1] / ax.get_xlim()[0] > 10:
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved latency vs input length chart to: {output_path}")


def plot_memory_efficiency(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Generate memory efficiency chart (throughput per GiB of memory)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = []
    efficiency_values = []

    for config_name, data in results.items():
        if "by_input_length" not in data:
            continue

        # Use the median input length for comparison
        lengths = sorted(data["by_input_length"].keys(), key=int)
        if not lengths:
            continue
        mid_length = lengths[len(lengths) // 2]

        length_data = data["by_input_length"][mid_length]
        if "error" in length_data:
            continue

        throughput = length_data.get("throughput", {}).get("tokens_per_second", 0)
        memory = length_data.get("memory", {})
        total_memory = (
            memory.get("kv_cache_memory_gib", 0) + memory.get("model_memory_gib", 0)
        )

        if throughput > 0 and total_memory > 0:
            efficiency = throughput / total_memory
            configs.append(config_name)
            efficiency_values.append(efficiency)

    if not configs:
        print("No data available for memory efficiency plot")
        return

    colors = [COLORS.get(c, "#95a5a6") for c in configs]
    labels = [CONFIG_LABELS.get(c, c) for c in configs]

    bars = ax.bar(labels, efficiency_values, color=colors)

    ax.set_ylabel("Throughput per GiB (tokens/s/GiB)", fontsize=12)
    ax.set_title("Memory Efficiency Comparison", fontsize=14)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved memory efficiency chart to: {output_path}")


def create_summary_table(
    results: dict[str, dict],
    output_path: str,
) -> None:
    """Create a summary table in Markdown format."""
    lines = [
        "# Hybrid Attention Benchmark Summary\n",
        "## Configuration Comparison\n",
        "| Configuration | Input Length | Throughput (tok/s) | Avg Latency (ms) | P99 Latency (ms) | KV Cache (GiB) |",
        "|--------------|--------------|-------------------|------------------|------------------|----------------|",
    ]

    for config_name, data in results.items():
        if "by_input_length" not in data:
            continue

        label = CONFIG_LABELS.get(config_name, config_name)

        for length, length_data in sorted(
            data["by_input_length"].items(), key=lambda x: int(x[0])
        ):
            if "error" in length_data:
                continue

            throughput = length_data.get("throughput", {}).get("tokens_per_second", 0)
            avg_latency = length_data.get("latency", {}).get("avg_seconds", 0) * 1000
            p99_latency = length_data.get("latency", {}).get("p99_seconds", 0) * 1000
            kv_memory = length_data.get("memory", {}).get("kv_cache_memory_gib", "-")

            if isinstance(kv_memory, float):
                kv_memory = f"{kv_memory:.2f}"

            lines.append(
                f"| {label} | {length} | {throughput:.1f} | {avg_latency:.1f} | {p99_latency:.1f} | {kv_memory} |"
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
        "--input-length",
        type=str,
        default="1024",
        help="Input length to use for single-length comparison charts",
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
    plot_memory_comparison(
        results,
        os.path.join(output_dir, f"memory_comparison.{args.format}"),
        input_length=args.input_length,
    )

    plot_throughput_comparison(
        results,
        os.path.join(output_dir, f"throughput_comparison.{args.format}"),
    )

    plot_latency_percentiles(
        results,
        os.path.join(output_dir, f"latency_percentiles.{args.format}"),
        input_length=args.input_length,
    )

    plot_latency_vs_input_length(
        results,
        os.path.join(output_dir, f"latency_vs_input_length.{args.format}"),
    )

    plot_memory_efficiency(
        results,
        os.path.join(output_dir, f"memory_efficiency.{args.format}"),
    )

    create_summary_table(
        results,
        os.path.join(output_dir, "summary.md"),
    )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize hybrid attention benchmark results"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)

