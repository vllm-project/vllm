#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Visualize CUTLASS MLA num_kv_splits benchmark results.

Usage:
    python visualize_numsplits.py cutlass_numsplits_results.json
"""

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import regex as re
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def parse_batch_spec(spec: str) -> tuple[int, int]:
    """Parse batch spec like '32q1s16k' into (batch_size, seq_length_k)."""
    match = re.match(r"(\d+)q1s(\d+)k", spec)
    if not match:
        raise ValueError(f"Cannot parse batch spec: {spec}")
    batch_size = int(match.group(1))
    seq_length_k = int(match.group(2))
    return batch_size, seq_length_k


def load_results(json_path: str) -> list:
    """Load benchmark results from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def extract_optimal_splits(
    results: list, exclude_auto: bool = False
) -> dict[tuple[int, int], int]:
    """
    Extract optimal num_kv_splits for each (batch_size, seq_length) pair.

    Args:
        results: List of benchmark results
        exclude_auto: If True, exclude "auto" backend from consideration

    Returns:
        Dict mapping (batch_size, seq_length_k) -> optimal_num_kv_splits
    """
    # Group results by batch_spec
    by_batch_spec = {}
    for result in results:
        batch_spec = result["config"]["batch_spec"]
        backend_name = result["config"]["backend"]

        # Skip auto if requested
        if exclude_auto and "auto" in backend_name:
            continue

        if batch_spec not in by_batch_spec:
            by_batch_spec[batch_spec] = []
        by_batch_spec[batch_spec].append(result)

    optimal_splits = {}

    for batch_spec, batch_results in by_batch_spec.items():
        batch_size, seq_length_k = parse_batch_spec(batch_spec)

        # Find the configuration with minimum time
        min_time = float("inf")
        optimal_split = 1

        for result in batch_results:
            if result["error"] is None and "mean_time" in result:
                time = result["mean_time"]
                if time < min_time:
                    min_time = time
                    # Extract num_kv_splits from backend name
                    backend_name = result["config"]["backend"]
                    match = re.search(r"numsplits_(\d+)", backend_name)
                    if match:
                        optimal_split = int(match.group(1))

        optimal_splits[(batch_size, seq_length_k)] = optimal_split

    return optimal_splits


def _get_axes_from_splits_dict(
    splits_dict: Mapping[tuple[int, int], int | float],
) -> tuple[list[int], list[int]]:
    """Extract sorted batch sizes and sequence lengths from splits dictionary."""
    batch_sizes = sorted(set(b for b, _ in splits_dict))
    seq_lengths = sorted(set(s for _, s in splits_dict), reverse=True)
    return batch_sizes, seq_lengths


def _create_splits_matrix(
    splits_dict: Mapping[tuple[int, int], int | float],
    batch_sizes: list[int],
    seq_lengths: list[int],
) -> np.ndarray:
    """Create matrix from splits dictionary."""
    matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
    for i, seq_len in enumerate(seq_lengths):
        for j, batch_size in enumerate(batch_sizes):
            matrix[i, j] = splits_dict.get((batch_size, seq_len), np.nan)
    return matrix


def _setup_heatmap_axes(ax, batch_sizes: list[int], seq_lengths: list[int], title: str):
    """Setup common axes properties for heatmaps."""
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(seq_lengths)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels([f"{s}k" for s in seq_lengths])
    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)


def _create_log2_colormap(min_log2: float, max_log2: float) -> tuple:
    """Create discrete log2 colormap and bounds."""
    n_colors = int(max_log2 - min_log2 + 1)
    viridis = plt.cm.viridis
    indices = np.linspace(0, 1, n_colors)
    colors = [viridis(i) for i in indices]
    cmap = ListedColormap(colors)
    vmin = min_log2 - 0.5
    vmax = max_log2 + 0.5
    return cmap, vmin, vmax


def _add_log2_colorbar(im, ax, label: str, min_log2: float, max_log2: float):
    """Add colorbar with power-of-2 labels."""
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, rotation=270, labelpad=20, fontsize=12)
    tick_positions = np.arange(min_log2, max_log2 + 1)
    tick_labels = [str(int(2**i)) for i in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)


def _annotate_splits_matrix(
    ax, matrix: np.ndarray, batch_sizes: list[int], seq_lengths: list[int]
):
    """Add text annotations showing split values."""
    for i in range(len(seq_lengths)):
        for j in range(len(batch_sizes)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    int(value),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                    fontweight="bold",
                )


def create_heatmap(optimal_splits: dict[tuple[int, int], int], output_path: str):
    """Create heatmap showing optimal num_kv_splits."""
    batch_sizes, seq_lengths = _get_axes_from_splits_dict(optimal_splits)
    matrix = _create_splits_matrix(optimal_splits, batch_sizes, seq_lengths)

    _fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to log2 scale for coloring
    matrix_log2 = np.log2(matrix)
    valid_values = matrix_log2[~np.isnan(matrix_log2)]
    min_log2 = np.floor(valid_values.min())
    max_log2 = np.ceil(valid_values.max())

    cmap, vmin, vmax = _create_log2_colormap(min_log2, max_log2)
    im = ax.imshow(matrix_log2, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    _setup_heatmap_axes(
        ax,
        batch_sizes,
        seq_lengths,
        "Optimal num_kv_splits for CUTLASS MLA\n"
        "(Lower is simpler, higher is more parallelism)",
    )
    _annotate_splits_matrix(ax, matrix, batch_sizes, seq_lengths)
    _add_log2_colorbar(im, ax, "Optimal num_kv_splits", min_log2, max_log2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    plt.close()


def _create_speedup_colormap():
    """Create colormap for speedup: 1.0 = white, higher = green."""
    colors_dict = {
        "red": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        "green": [(0.0, 1.0, 1.0), (1.0, 0.5, 0.5)],
        "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
    }
    return LinearSegmentedColormap("Speedup", colors_dict)


def _annotate_speedup_matrix(
    ax, matrix: np.ndarray, batch_sizes: list[int], seq_lengths: list[int]
):
    """Add text annotations showing speedup values."""
    for i in range(len(seq_lengths)):
        for j in range(len(batch_sizes)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}x",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                )


def _compute_speedup_matrix(
    results: list, exclude_auto: bool = False
) -> dict[tuple[int, int], float]:
    """Compute speedup matrix from results (optimal vs splits=1)."""
    by_batch_spec = {}
    for result in results:
        batch_spec = result["config"]["batch_spec"]
        backend_name = result["config"]["backend"]

        if exclude_auto and "auto" in backend_name:
            continue

        if batch_spec not in by_batch_spec:
            by_batch_spec[batch_spec] = []
        by_batch_spec[batch_spec].append(result)

    speedup_matrix = {}
    for batch_spec, batch_results in by_batch_spec.items():
        batch_size, seq_length_k = parse_batch_spec(batch_spec)

        baseline_time = None
        min_time = float("inf")

        for result in batch_results:
            if result["error"] is None and "mean_time" in result:
                time = result["mean_time"]
                backend_name = result["config"]["backend"]
                if backend_name.endswith("numsplits_1"):
                    baseline_time = time
                if time < min_time:
                    min_time = time

        if baseline_time:
            speedup = baseline_time / min_time
            speedup_matrix[(batch_size, seq_length_k)] = speedup

    return speedup_matrix


def create_performance_heatmap(
    results: list, output_path: str, exclude_auto: bool = False
):
    """Create heatmap showing speedup from optimal splits vs splits=1."""
    speedup_dict = _compute_speedup_matrix(results, exclude_auto)
    batch_sizes, seq_lengths = _get_axes_from_splits_dict(speedup_dict)
    matrix = _create_splits_matrix(speedup_dict, batch_sizes, seq_lengths)

    _fig, ax = plt.subplots(figsize=(12, 8))

    max_speedup = np.nanmax(matrix)
    speedup_cmap = _create_speedup_colormap()
    im = ax.imshow(matrix, cmap=speedup_cmap, aspect="auto", vmin=1.0, vmax=max_speedup)

    _setup_heatmap_axes(
        ax,
        batch_sizes,
        seq_lengths,
        "Speedup from Optimal num_kv_splits vs. splits=1\n"
        "(Green = better with splits, White = same)",
    )
    _annotate_speedup_matrix(ax, matrix, batch_sizes, seq_lengths)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup Factor", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved speedup heatmap to {output_path}")
    plt.close()


def heuristic_ratio_based(batch_size: int, seq_length_k: int) -> int:
    """Original ratio-based heuristic (from visualize_numsplits.py)."""
    ratio = seq_length_k / batch_size
    if ratio >= 2.5:
        return 8
    elif ratio >= 1.2:
        return 4
    elif ratio >= 0.5:
        return 2
    else:
        return 1


def heuristic_constant(batch_size: int, seq_length_k: int) -> int:
    """Ultra-simple constant heuristic: always use 2 for small batches."""
    if batch_size <= 32:
        return 2
    else:
        return 1


def heuristic_batch_based(batch_size: int, seq_length_k: int) -> int:
    """
    Simple batch-based heuristic with zero slowdowns.
    """
    if batch_size <= 4 and seq_length_k >= 8:
        return 16
    elif batch_size <= 8 and seq_length_k >= 2:
        return 8
    elif (batch_size <= 16 and seq_length_k >= 4) or (
        batch_size == 48 and seq_length_k >= 32
    ):
        return 4
    elif (batch_size <= 32 and seq_length_k >= 8) or (
        batch_size == 96 and seq_length_k >= 16
    ):
        return 2
    else:
        return 1


def _annotate_heuristic_matrix(
    ax,
    matrix: np.ndarray,
    batch_sizes: list[int],
    seq_lengths: list[int],
    optimal_splits: dict[tuple[int, int], int],
):
    """Add text annotations showing heuristic values and mismatches."""
    for i in range(len(seq_lengths)):
        for j in range(len(batch_sizes)):
            value = matrix[i, j]
            seq_len = seq_lengths[i]
            batch_size = batch_sizes[j]
            optimal = optimal_splits.get((batch_size, seq_len), None)

            if not np.isnan(value):
                # Mark mismatches with red text
                if optimal is not None and int(value) != optimal:
                    color = "red"
                    text = f"{int(value)}\n✗"
                else:
                    color = "black"
                    text = str(int(value))

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    fontweight="bold",
                )


def create_heuristic_policy_heatmaps(
    optimal_splits: dict[tuple[int, int], int], output_dir: Path
):
    """Create heatmaps showing num_splits chosen by each heuristic policy."""
    heuristics = {
        "Ratio-based": heuristic_ratio_based,
        "Batch-based (improved)": heuristic_batch_based,
        "Constant (batch<=32)": heuristic_constant,
    }

    batch_sizes, seq_lengths = _get_axes_from_splits_dict(optimal_splits)

    for heuristic_name, heuristic_func in heuristics.items():
        # Build matrix of chosen num_splits
        matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
        for i, seq_len in enumerate(seq_lengths):
            for j, batch_size in enumerate(batch_sizes):
                matrix[i, j] = heuristic_func(batch_size, seq_len)

        _fig, ax = plt.subplots(figsize=(12, 8))

        # Convert to log2 scale for coloring
        matrix_log2 = np.log2(matrix)
        valid_values = matrix_log2[~np.isnan(matrix_log2)]
        min_log2 = np.floor(valid_values.min())
        max_log2 = np.ceil(valid_values.max())

        cmap, vmin, vmax = _create_log2_colormap(min_log2, max_log2)
        im = ax.imshow(matrix_log2, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        _setup_heatmap_axes(
            ax,
            batch_sizes,
            seq_lengths,
            f"num_kv_splits Chosen by {heuristic_name} Policy",
        )
        _annotate_heuristic_matrix(ax, matrix, batch_sizes, seq_lengths, optimal_splits)
        _add_log2_colorbar(im, ax, "num_kv_splits", min_log2, max_log2)

        plt.tight_layout()

        safe_name = (
            heuristic_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        output_path = output_dir / f"numsplits_policy_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {heuristic_name} policy heatmap to {output_path}")
        plt.close()


def _build_timings_lookup(results: list) -> dict[str, dict[int, float]]:
    """Build lookup table of timings by batch_spec and num_splits."""
    by_batch_spec = {}
    for result in results:
        batch_spec = result["config"]["batch_spec"]
        if batch_spec not in by_batch_spec:
            by_batch_spec[batch_spec] = {}

        if result["error"] is None and "mean_time" in result:
            backend_name = result["config"]["backend"]
            match = re.search(r"numsplits_(\d+)", backend_name)
            if match:
                num_splits = int(match.group(1))
                by_batch_spec[batch_spec][num_splits] = result["mean_time"]
    return by_batch_spec


def create_heuristic_speedup_heatmaps(
    results: list, optimal_splits: dict[tuple[int, int], int], output_dir: Path
):
    """Create speedup heatmaps for each heuristic policy."""
    heuristics = {
        "Ratio-based (Original)": heuristic_ratio_based,
        "Batch-based (improved)": heuristic_batch_based,
        "Constant (batch<=32)": heuristic_constant,
    }

    by_batch_spec = _build_timings_lookup(results)
    batch_sizes, seq_lengths = _get_axes_from_splits_dict(optimal_splits)

    for heuristic_name, heuristic_func in heuristics.items():
        speedup_matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
        total_speedup = 0.0
        count = 0

        for i, seq_len in enumerate(seq_lengths):
            for j, batch_size in enumerate(batch_sizes):
                batch_spec = f"{batch_size}q1s{seq_len}k"
                timings = by_batch_spec.get(batch_spec, {})
                baseline_time = timings.get(1)

                if baseline_time:
                    predicted_splits = heuristic_func(batch_size, seq_len)
                    predicted_time = timings.get(predicted_splits, baseline_time)
                    speedup = baseline_time / predicted_time
                    speedup_matrix[i, j] = speedup
                    total_speedup += speedup
                    count += 1
                else:
                    speedup_matrix[i, j] = np.nan

        avg_speedup = total_speedup / count if count > 0 else 1.0

        _fig, ax = plt.subplots(figsize=(12, 8))

        max_speedup = np.nanmax(speedup_matrix)
        speedup_cmap = _create_speedup_colormap()
        im = ax.imshow(
            speedup_matrix,
            cmap=speedup_cmap,
            aspect="auto",
            vmin=1.0,
            vmax=max_speedup,
        )

        _setup_heatmap_axes(
            ax,
            batch_sizes,
            seq_lengths,
            f"Speedup with {heuristic_name} Policy\n"
            f"(Average speedup: {avg_speedup:.3f}x vs. splits=1)",
        )
        _annotate_speedup_matrix(ax, speedup_matrix, batch_sizes, seq_lengths)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Speedup Factor", rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()

        safe_name = (
            heuristic_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        output_path = output_dir / f"numsplits_speedup_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {heuristic_name} speedup heatmap to {output_path}")
        plt.close()


def create_auto_heatmap(results: list, output_path: str):
    """Create heatmap showing num_kv_splits chosen by auto policy."""
    # Find all configs with auto results
    auto_configs = set()
    for result in results:
        if "auto" in result["config"]["backend"] and result["error"] is None:
            batch_spec = result["config"]["batch_spec"]
            batch_size, seq_length_k = parse_batch_spec(batch_spec)
            auto_configs.add((batch_size, seq_length_k))

    if not auto_configs:
        print("Skipping auto heatmap (no auto results found)")
        return

    batch_sizes, seq_lengths = _get_axes_from_splits_dict({k: 1 for k in auto_configs})
    matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
    for i, seq_len in enumerate(seq_lengths):
        for j, batch_size in enumerate(batch_sizes):
            matrix[i, j] = 1 if (batch_size, seq_len) in auto_configs else np.nan

    _fig, ax = plt.subplots(figsize=(12, 8))

    cmap = ListedColormap(["#2ca02c"])  # Green for auto
    _im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    _setup_heatmap_axes(
        ax, batch_sizes, seq_lengths, "Auto num_kv_splits Policy Coverage"
    )

    # Add "AUTO" text annotations
    for i in range(len(seq_lengths)):
        for j in range(len(batch_sizes)):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    "AUTO",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved auto heatmap to {output_path}")
    plt.close()


def create_auto_speedup_heatmap(results: list, output_path: str):
    """Create heatmap showing speedup from auto vs splits=1."""
    # Build speedup dictionary
    speedup_dict = {}
    timings_by_spec = {}

    for result in results:
        if result["error"] is not None or "mean_time" not in result:
            continue

        batch_spec = result["config"]["batch_spec"]
        backend_name = result["config"]["backend"]

        if batch_spec not in timings_by_spec:
            timings_by_spec[batch_spec] = {}

        if "auto" in backend_name:
            timings_by_spec[batch_spec]["auto"] = result["mean_time"]
        elif backend_name.endswith("numsplits_1"):
            timings_by_spec[batch_spec]["baseline"] = result["mean_time"]

    for batch_spec, timings in timings_by_spec.items():
        if "baseline" in timings and "auto" in timings:
            batch_size, seq_length_k = parse_batch_spec(batch_spec)
            speedup = timings["baseline"] / timings["auto"]
            speedup_dict[(batch_size, seq_length_k)] = speedup

    if not speedup_dict:
        print("Skipping auto speedup heatmap (no auto results found)")
        return

    batch_sizes, seq_lengths = _get_axes_from_splits_dict(speedup_dict)
    matrix = _create_splits_matrix(speedup_dict, batch_sizes, seq_lengths)

    _fig, ax = plt.subplots(figsize=(12, 8))

    max_speedup = np.nanmax(matrix)
    speedup_cmap = _create_speedup_colormap()
    im = ax.imshow(matrix, cmap=speedup_cmap, aspect="auto", vmin=1.0, vmax=max_speedup)

    _setup_heatmap_axes(
        ax,
        batch_sizes,
        seq_lengths,
        "Speedup from Auto Policy vs. splits=1\n(Green = better with auto)",
    )
    _annotate_speedup_matrix(ax, matrix, batch_sizes, seq_lengths)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup Factor", rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved auto speedup heatmap to {output_path}")
    plt.close()


def analyze_pattern(optimal_splits: dict[tuple[int, int], int]):
    """Analyze the pattern and suggest a formula."""
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    # Group by optimal split value
    by_split_value = {}
    for (batch, seq), split in optimal_splits.items():
        if split not in by_split_value:
            by_split_value[split] = []
        by_split_value[split].append((batch, seq))

    print("\nConfigurations grouped by optimal num_kv_splits:")
    for split in sorted(by_split_value.keys()):
        configs = by_split_value[split]
        print(f"\n  num_kv_splits = {split} ({len(configs)} configs):")
        for batch, seq in sorted(configs)[:5]:  # Show first 5
            print(f"    - batch={batch:3d}, seq={seq:3d}k")
        if len(configs) > 5:
            print(f"    ... and {len(configs) - 5} more")

    # Analyze ratio: seq_length / batch_size
    print("\n" + "-" * 80)
    print("Analysis of seq_length/batch_size ratio:")
    print("-" * 80)

    ratio_by_split = {split: [] for split in by_split_value}
    for (batch, seq), split in optimal_splits.items():
        ratio = seq / batch
        ratio_by_split[split].append(ratio)

    print(f"\n{'Split':<8} {'Min Ratio':<12} {'Max Ratio':<12} {'Avg Ratio':<12}")
    print("-" * 50)
    for split in sorted(ratio_by_split.keys()):
        ratios = ratio_by_split[split]
        if ratios:
            print(
                f"{split:<8} {min(ratios):<12.1f} {max(ratios):<12.1f} "
                f"{np.mean(ratios):<12.1f}"
            )

    # Test heuristics
    print("\n" + "=" * 80)
    print("HEURISTIC COMPARISON")
    print("=" * 80)

    heuristics = {
        "Ratio-based": heuristic_ratio_based,
        "Batch-based (improved)": heuristic_batch_based,
        "Constant (batch<=32)": heuristic_constant,
    }

    for name, heuristic_func in heuristics.items():
        correct = 0
        total = 0
        mismatches = []

        for (batch, seq), actual_split in optimal_splits.items():
            predicted_split = heuristic_func(batch, seq)
            total += 1
            if predicted_split == actual_split:
                correct += 1
            else:
                mismatches.append((batch, seq, predicted_split, actual_split))

        accuracy = 100 * correct / total
        print(f"\n{name}:")
        print(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")

        if mismatches and len(mismatches) <= 10:
            print("  Mismatches:")
            for batch, seq, pred, actual in mismatches:
                print(
                    f"    batch={batch:3d}, seq={seq:3d}k -> "
                    f"predicted={pred}, actual={actual}"
                )
        elif mismatches:
            print(f"  {len(mismatches)} mismatches (showing first 5):")
            for batch, seq, pred, actual in mismatches[:5]:
                print(
                    f"    batch={batch:3d}, seq={seq:3d}k -> "
                    f"predicted={pred}, actual={actual}"
                )

    print("\n" + "=" * 80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_numsplits.py <results.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = Path(json_path).parent

    print(f"Loading results from {json_path}...")
    results = load_results(json_path)

    print("Extracting optimal splits (excluding auto)...")
    optimal_splits = extract_optimal_splits(results, exclude_auto=True)

    print(f"Found {len(optimal_splits)} configurations")

    # Create visualizations
    print("\nGenerating visualizations...")

    print("\n--- Manual Configuration Plots (excluding auto) ---")
    create_heatmap(optimal_splits, str(output_dir / "numsplits_heatmap.png"))
    create_performance_heatmap(
        results, str(output_dir / "numsplits_speedup.png"), exclude_auto=True
    )
    create_heuristic_policy_heatmaps(optimal_splits, output_dir)
    create_heuristic_speedup_heatmaps(results, optimal_splits, output_dir)

    print("\n--- Auto Policy Plots ---")
    create_auto_heatmap(results, str(output_dir / "numsplits_heatmap_auto.png"))
    create_auto_speedup_heatmap(results, str(output_dir / "numsplits_speedup_auto.png"))

    # Analyze pattern
    analyze_pattern(optimal_splits)

    print("\nDone! Check the output directory for visualization files.")


if __name__ == "__main__":
    main()
