# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Visualization script for video benchmark results.

This script generates comparison charts from benchmark JSON files:
1. Latency comparison bar chart
2. Throughput comparison
3. Latency distribution box plot
4. Performance delta chart

For streaming benchmarks (--streaming-mode), additional charts:
5. Memory scaling plot (O(1) vs O(n) comparison)
6. Frame processing time plot
7. Concurrent query throughput
8. Memory growth rate comparison

Usage:
    # Standard video benchmark
    python benchmarks/visualize_video_benchmark.py \
        --results-dir ./video_benchmark_results \
        --output-dir ./video_benchmark_results/plots

    # Streaming video benchmark
    python benchmarks/visualize_video_benchmark.py \
        --results-dir ./streaming_benchmark_results \
        --output-dir ./streaming_benchmark_results/plots \
        --streaming-mode
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
    "hybrid_attention": "Hybrid SSM + SW",
    "standard_attention": "Standard Full Attention",
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
                better = "Hybrid" if delta < 0 else "Standard"
            else:
                better = "Hybrid" if delta > 0 else "Standard"

            lines.append(
                f"| {name} | {h_val:.2f} | {s_val:.2f} | {delta:+.2f}% | {better} |"
            )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved summary table to: {output_path}")


# =============================================================================
# Streaming Video Benchmark Visualizations
# =============================================================================


def load_streaming_results(results_dir: str) -> dict[str, list[dict]]:
    """Load streaming benchmark results from comparison JSON files.

    Returns:
        Dictionary mapping scenario names to list of results.
    """
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("*_comparison.json"):
        with open(json_file) as f:
            data = json.load(f)
            scenario = data.get("scenario", json_file.stem.replace("_comparison", ""))
            results[scenario] = data.get("results", [])

    # Also try individual result files
    for json_file in results_path.glob("*.json"):
        if "_comparison" not in json_file.name:
            with open(json_file) as f:
                data = json.load(f)
                if "results" in data and len(data["results"]) > 0:
                    scenario = data.get("scenario", json_file.stem)
                    if scenario not in results:
                        results[scenario] = data["results"]

    return results


def plot_memory_scaling(
    results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Plot memory usage over frames to show O(1) vs O(n) scaling.

    This is the most important visualization for demonstrating SSM efficiency.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Try to find a scenario with memory samples
    scenario_data = None
    for scenario_name, scenario_results in results.items():
        for result in scenario_results:
            if result.get("memory_samples"):
                scenario_data = scenario_results
                break
        if scenario_data:
            break

    if not scenario_data:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No memory scaling data available",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved memory scaling chart to: {output_path}")
        return

    # Left plot: Memory vs Frame Index
    ax1 = axes[0]
    for result in scenario_data:
        config = result.get("config", "unknown")
        samples = result.get("memory_samples", [])
        if samples:
            frames = [s.get("frame_idx", i) for i, s in enumerate(samples)]
            memory = [s.get("used_memory_gib", 0) for s in samples]

            color = COLORS.get(config, "#95a5a6")
            label = CONFIG_LABELS.get(config, config)
            ax1.plot(frames, memory, "o-", color=color, label=label, linewidth=2, markersize=4)

            # Add trend line
            if len(frames) >= 2:
                z = np.polyfit(frames, memory, 1)
                p = np.poly1d(z)
                ax1.plot(
                    frames,
                    p(frames),
                    "--",
                    color=color,
                    alpha=0.5,
                    linewidth=1.5,
                    label=f"{label} trend",
                )

    ax1.set_xlabel("Frame Index", fontsize=12)
    ax1.set_ylabel("GPU Memory Used (GiB)", fontsize=12)
    ax1.set_title("Memory Usage vs Video Length", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: Memory Growth Rate Comparison
    ax2 = axes[1]
    configs = []
    growth_rates = []
    colors = []

    for result in scenario_data:
        config = result.get("config", "unknown")
        growth_rate = result.get("memory_growth_rate_gib_per_frame", 0) * 1000  # MiB/frame
        configs.append(CONFIG_LABELS.get(config, config))
        growth_rates.append(growth_rate)
        colors.append(COLORS.get(config, "#95a5a6"))

    if configs:
        bars = ax2.bar(configs, growth_rates, color=colors, edgecolor="white", linewidth=0.5)

        # Add value labels
        for bar, rate in zip(bars, growth_rates):
            height = bar.get_height()
            ax2.annotate(
                f"{rate:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax2.set_ylabel("Memory Growth Rate (MiB/frame)", fontsize=12)
        ax2.set_title("Memory Scaling Behavior", fontsize=14)

        # Add annotation explaining the meaning
        if len(growth_rates) == 2:
            if growth_rates[0] > 0 and growth_rates[1] > 0:
                ratio = max(growth_rates) / min(growth_rates) if min(growth_rates) > 0 else 0
                if ratio > 1.5:
                    ax2.annotate(
                        f"SSM scales {ratio:.1f}x better",
                        xy=(0.5, 0.95),
                        xycoords="axes fraction",
                        ha="center",
                        fontsize=12,
                        fontweight="bold",
                        color="#27ae60",
                    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved memory scaling chart to: {output_path}")


def plot_frame_processing_time(
    results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Plot frame processing time distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))

    has_data = False
    for scenario_name, scenario_results in results.items():
        for result in scenario_results:
            frame_times = result.get("frame_times_ms", [])
            if frame_times:
                has_data = True
                config = result.get("config", "unknown")
                color = COLORS.get(config, "#95a5a6")
                label = f"{CONFIG_LABELS.get(config, config)} ({scenario_name})"

                frames = np.arange(len(frame_times))
                ax.plot(frames, frame_times, "-", color=color, label=label, linewidth=1.5, alpha=0.8)

                # Add rolling average
                if len(frame_times) > 5:
                    window = 5
                    rolling_avg = np.convolve(frame_times, np.ones(window) / window, mode="valid")
                    ax.plot(
                        frames[window - 1 :],
                        rolling_avg,
                        "-",
                        color=color,
                        linewidth=2.5,
                        alpha=1.0,
                    )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No frame processing time data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
    else:
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Processing Time (ms)", fontsize=12)
        ax.set_title("Frame Processing Time Over Video", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved frame processing time chart to: {output_path}")


def plot_concurrent_query_throughput(
    results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Plot concurrent query throughput comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect data across scenarios
    scenario_names = []
    hybrid_qps = []
    standard_qps = []
    hybrid_latencies = []
    standard_latencies = []

    for scenario_name, scenario_results in results.items():
        hybrid = next((r for r in scenario_results if r.get("config") == "hybrid_attention"), None)
        standard = next((r for r in scenario_results if r.get("config") == "standard_attention"), None)

        if hybrid and standard:
            scenario_names.append(scenario_name.replace("-", "\n"))
            hybrid_qps.append(hybrid.get("queries_per_second", 0) or 0)
            standard_qps.append(standard.get("queries_per_second", 0) or 0)
            hybrid_latencies.append((hybrid.get("avg_query_latency_seconds", 0) or 0) * 1000)
            standard_latencies.append((standard.get("avg_query_latency_seconds", 0) or 0) * 1000)

    if not scenario_names:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No concurrent query data available",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved concurrent query chart to: {output_path}")
        return

    x = np.arange(len(scenario_names))
    width = 0.35

    # Left plot: Queries per second
    ax1 = axes[0]
    bars1 = ax1.bar(
        x - width / 2,
        standard_qps,
        width,
        label="Standard Attention",
        color=COLORS["standard_attention"],
        edgecolor="white",
    )
    bars2 = ax1.bar(
        x + width / 2,
        hybrid_qps,
        width,
        label="Hybrid SSM + SW",
        color=COLORS["hybrid_attention"],
        edgecolor="white",
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax1.set_xlabel("Scenario", fontsize=12)
    ax1.set_ylabel("Queries per Second", fontsize=12)
    ax1.set_title("Query Throughput Comparison", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, fontsize=10)
    ax1.legend(fontsize=10)

    # Right plot: Query latency
    ax2 = axes[1]
    bars1 = ax2.bar(
        x - width / 2,
        standard_latencies,
        width,
        label="Standard Attention",
        color=COLORS["standard_attention"],
        edgecolor="white",
    )
    bars2 = ax2.bar(
        x + width / 2,
        hybrid_latencies,
        width,
        label="Hybrid SSM + SW",
        color=COLORS["hybrid_attention"],
        edgecolor="white",
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax2.set_xlabel("Scenario", fontsize=12)
    ax2.set_ylabel("Avg Query Latency (ms)", fontsize=12)
    ax2.set_title("Query Latency Comparison", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_names, fontsize=10)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved concurrent query chart to: {output_path}")


def plot_streaming_memory_comparison(
    results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Plot memory comparison bar chart for streaming benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect data
    scenarios = []
    hybrid_memory = []
    standard_memory = []
    savings_pct = []

    for scenario_name, scenario_results in results.items():
        hybrid = next((r for r in scenario_results if r.get("config") == "hybrid_attention"), None)
        standard = next((r for r in scenario_results if r.get("config") == "standard_attention"), None)

        if hybrid and standard:
            h_mem = hybrid.get("peak_memory_gib", 0) or 0
            s_mem = standard.get("peak_memory_gib", 0) or 0

            if s_mem > 0 and h_mem > 0:
                scenarios.append(scenario_name.replace("-", "\n"))
                hybrid_memory.append(h_mem)
                standard_memory.append(s_mem)
                savings_pct.append((s_mem - h_mem) / s_mem * 100)

    if not scenarios:
        ax.text(
            0.5,
            0.5,
            "No memory comparison data available",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved memory comparison chart to: {output_path}")
        return

    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        standard_memory,
        width,
        label="Standard Attention",
        color=COLORS["standard_attention"],
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        hybrid_memory,
        width,
        label="Hybrid SSM + SW",
        color=COLORS["hybrid_attention"],
        edgecolor="white",
    )

    # Add savings annotations
    for i, (std, hyb, save) in enumerate(zip(standard_memory, hybrid_memory, savings_pct)):
        ax.annotate(
            f"{save:.1f}% saved",
            xy=(x[i], max(std, hyb)),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#27ae60" if save > 0 else "#e74c3c",
        )

    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Peak Memory (GiB)", fontsize=12)
    ax.set_title("Peak Memory Usage Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved memory comparison chart to: {output_path}")


def create_streaming_summary_table(
    results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Create a summary table for streaming benchmark results."""
    lines = [
        "# Streaming Video Benchmark Summary\n",
        "## SSM + Sliding Window vs Standard Attention\n",
        "### Key Insight: SSM provides O(1) memory scaling vs O(n) for full attention\n",
    ]

    for scenario_name, scenario_results in results.items():
        hybrid = next((r for r in scenario_results if r.get("config") == "hybrid_attention"), None)
        standard = next((r for r in scenario_results if r.get("config") == "standard_attention"), None)

        if not hybrid or not standard:
            continue

        lines.append(f"\n## Scenario: {scenario_name}\n")
        lines.append(f"**Frames:** {hybrid.get('num_frames', 'N/A')}  ")
        lines.append(f"**Concurrent Queries:** {hybrid.get('concurrent_queries', 'N/A')}\n")

        # Memory comparison
        lines.append("\n### Memory Performance\n")
        lines.append("| Metric | Standard | Hybrid | Improvement |")
        lines.append("|--------|----------|--------|-------------|")

        std_peak = standard.get("peak_memory_gib", 0) or 0
        hyb_peak = hybrid.get("peak_memory_gib", 0) or 0
        mem_save = ((std_peak - hyb_peak) / std_peak * 100) if std_peak > 0 else 0
        lines.append(f"| Peak Memory (GiB) | {std_peak:.2f} | {hyb_peak:.2f} | {mem_save:.1f}% less |")

        std_growth = (standard.get("memory_growth_rate_gib_per_frame", 0) or 0) * 1000
        hyb_growth = (hybrid.get("memory_growth_rate_gib_per_frame", 0) or 0) * 1000
        growth_ratio = std_growth / hyb_growth if hyb_growth > 0 else 0
        lines.append(f"| Growth Rate (MiB/frame) | {std_growth:.3f} | {hyb_growth:.3f} | {growth_ratio:.1f}x better |")

        # Query performance
        lines.append("\n### Query Performance\n")
        lines.append("| Metric | Standard | Hybrid | Delta |")
        lines.append("|--------|----------|--------|-------|")

        std_lat = (standard.get("avg_query_latency_seconds", 0) or 0) * 1000
        hyb_lat = (hybrid.get("avg_query_latency_seconds", 0) or 0) * 1000
        lat_delta = ((hyb_lat - std_lat) / std_lat * 100) if std_lat > 0 else 0
        lines.append(f"| Avg Query Latency (ms) | {std_lat:.1f} | {hyb_lat:.1f} | {lat_delta:+.1f}% |")

        std_qps = standard.get("queries_per_second", 0) or 0
        hyb_qps = hybrid.get("queries_per_second", 0) or 0
        qps_delta = ((hyb_qps - std_qps) / std_qps * 100) if std_qps > 0 else 0
        lines.append(f"| Queries/Second | {std_qps:.2f} | {hyb_qps:.2f} | {qps_delta:+.1f}% |")

    lines.append("\n---\n")
    lines.append("*Generated by vLLM Streaming Video Benchmark*\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved streaming summary to: {output_path}")


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
    parser.add_argument(
        "--streaming-mode",
        action="store_true",
        help="Generate visualizations for streaming video benchmarks",
    )


def main(args: argparse.Namespace) -> None:
    """Main visualization entry point."""
    # Set up output directory
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    if args.streaming_mode:
        # Load and visualize streaming benchmark results
        print(f"Loading streaming results from: {args.results_dir}")
        streaming_results = load_streaming_results(args.results_dir)

        if not streaming_results:
            print(f"No streaming result files found in {args.results_dir}")
            return

        print(f"Found {len(streaming_results)} scenarios: {list(streaming_results.keys())}")

        # Generate streaming-specific plots
        plot_memory_scaling(
            streaming_results,
            os.path.join(output_dir, f"memory_scaling.{args.format}"),
        )

        plot_frame_processing_time(
            streaming_results,
            os.path.join(output_dir, f"frame_processing_time.{args.format}"),
        )

        plot_concurrent_query_throughput(
            streaming_results,
            os.path.join(output_dir, f"concurrent_query_throughput.{args.format}"),
        )

        plot_streaming_memory_comparison(
            streaming_results,
            os.path.join(output_dir, f"memory_comparison.{args.format}"),
        )

        create_streaming_summary_table(
            streaming_results,
            os.path.join(output_dir, "streaming_summary.md"),
        )

        print(f"\nAll streaming visualizations saved to: {output_dir}")
        return

    # Standard video benchmark visualization
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

