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


def extract_optimal_splits(results: list) -> dict[tuple[int, int], int]:
    """
    Extract optimal num_kv_splits for each (batch_size, seq_length) pair.

    Returns:
        Dict mapping (batch_size, seq_length_k) -> optimal_num_kv_splits
    """
    # Group results by batch_spec
    by_batch_spec = {}
    for result in results:
        batch_spec = result["config"]["batch_spec"]
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


def create_heatmap(optimal_splits: dict[tuple[int, int], int], output_path: str):
    """Create heatmap showing optimal num_kv_splits."""
    # Extract unique batch sizes and sequence lengths
    batch_sizes = sorted(set(b for b, _ in optimal_splits))
    seq_lengths = sorted(
        set(s for _, s in optimal_splits), reverse=True
    )  # Reverse for bottom-to-top

    # Create matrix
    matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
    for i, seq_len in enumerate(seq_lengths):
        for j, batch_size in enumerate(batch_sizes):
            matrix[i, j] = optimal_splits.get((batch_size, seq_len), np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to log2 scale for coloring
    matrix_log2 = np.log2(matrix)

    # Get min/max values from actual data
    valid_values = matrix_log2[~np.isnan(matrix_log2)]
    min_log2 = np.floor(valid_values.min())
    max_log2 = np.ceil(valid_values.max())

    # Extend bounds by 0.5 on each side to center the discrete colors
    # e.g., value 1 (log2=0) spans -0.5 to 0.5
    #       value 2 (log2=1) spans 0.5 to 1.5, etc.
    vmin = min_log2 - 0.5
    vmax = max_log2 + 0.5

    # Create discrete colormap - one color per power of 2
    # Powers of 2: 1, 2, 4, 8, 16, 32 -> log2: 0, 1, 2, 3, 4, 5
    n_colors = int(max_log2 - min_log2 + 1)

    # Use viridis colormap (no value judgment on num_kv_splits)
    # Sample evenly across the colormap
    viridis = plt.cm.viridis
    indices = np.linspace(0, 1, n_colors)
    colors_to_use = [viridis(i) for i in indices]

    cmap = ListedColormap(colors_to_use)

    # Create heatmap with log2 scaled data
    im = ax.imshow(matrix_log2, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Set ticks
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(seq_lengths)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels([f"{s}k" for s in seq_lengths])

    # Labels
    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_title(
        "Optimal num_kv_splits for CUTLASS MLA\n(Lower is simpler, higher is more"
        " parallelism)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add text annotations
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

    # Colorbar with power-of-2 labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Optimal num_kv_splits", rotation=270, labelpad=20, fontsize=12)

    # Set colorbar ticks at the center of each discrete segment
    # Ticks should be at integer log2 values (0, 1, 2, 3...) which are centered in each
    # color band
    tick_positions = np.arange(min_log2, max_log2 + 1)
    tick_labels = [str(int(2**i)) for i in tick_positions]
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")
    plt.close()


def create_performance_heatmap(results: list, output_path: str):
    """Create heatmap showing speedup from optimal splits vs splits=1."""
    # Group results by batch_spec
    by_batch_spec = {}
    for result in results:
        batch_spec = result["config"]["batch_spec"]
        if batch_spec not in by_batch_spec:
            by_batch_spec[batch_spec] = []
        by_batch_spec[batch_spec].append(result)

    speedup_matrix = {}

    for batch_spec, batch_results in by_batch_spec.items():
        batch_size, seq_length_k = parse_batch_spec(batch_spec)

        # Get time for splits=1
        baseline_time = None
        min_time = float("inf")

        for result in batch_results:
            if result["error"] is None and "mean_time" in result:
                time = result["mean_time"]
                backend_name = result["config"]["backend"]
                # Match exactly numsplits_1 (not numsplits_16, etc.)
                if backend_name.endswith("numsplits_1"):
                    baseline_time = time
                if time < min_time:
                    min_time = time

        if baseline_time:
            speedup = baseline_time / min_time
            speedup_matrix[(batch_size, seq_length_k)] = speedup

    # Extract unique batch sizes and sequence lengths
    batch_sizes = sorted(set(b for b, _ in speedup_matrix))
    seq_lengths = sorted(
        set(s for _, s in speedup_matrix), reverse=True
    )  # Reverse for bottom-to-top

    # Create matrix
    matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
    for i, seq_len in enumerate(seq_lengths):
        for j, batch_size in enumerate(batch_sizes):
            matrix[i, j] = speedup_matrix.get((batch_size, seq_len), np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap with colormap: 1.0x = white (neutral), higher = green (good)

    # Create colormap: 1.0 = white, higher = green
    max_speedup = np.nanmax(matrix)
    colors_dict = {
        "red": [
            (0.0, 1.0, 1.0),  # At 1.0x (vmin): white
            (1.0, 0.0, 0.0),
        ],  # At max speedup: green
        "green": [(0.0, 1.0, 1.0), (1.0, 0.5, 0.5)],
        "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
    }
    speedup_cmap = LinearSegmentedColormap("Speedup", colors_dict)

    im = ax.imshow(matrix, cmap=speedup_cmap, aspect="auto", vmin=1.0, vmax=max_speedup)

    # Set ticks
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(seq_lengths)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels([f"{s}k" for s in seq_lengths])

    # Labels
    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_title(
        "Speedup from Optimal num_kv_splits vs. splits=1\n(Green = better with splits, "
        "Red = same)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add text annotations
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

    # Colorbar
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


def create_heuristic_policy_heatmaps(
    optimal_splits: dict[tuple[int, int], int], output_dir: Path
):
    """Create heatmaps showing num_splits chosen by each heuristic policy."""
    # Define heuristics to compare
    heuristics = {
        "Ratio-based": heuristic_ratio_based,
        "Constant (batch<=32)": heuristic_constant,
    }

    # Extract unique batch sizes and sequence lengths
    batch_sizes = sorted(set(b for b, _ in optimal_splits))
    seq_lengths = sorted(set(s for _, s in optimal_splits), reverse=True)

    # Create a separate heatmap for each heuristic
    for heuristic_name, heuristic_func in heuristics.items():
        # Build matrix of chosen num_splits
        matrix = np.zeros((len(seq_lengths), len(batch_sizes)))

        for i, seq_len in enumerate(seq_lengths):
            for j, batch_size in enumerate(batch_sizes):
                predicted_splits = heuristic_func(batch_size, seq_len)
                matrix[i, j] = predicted_splits

        # Create heatmap
        _fig, ax = plt.subplots(figsize=(12, 8))

        # Convert to log2 scale for coloring (same as optimal heatmap)
        matrix_log2 = np.log2(matrix)

        # Get min/max values
        valid_values = matrix_log2[~np.isnan(matrix_log2)]
        min_log2 = np.floor(valid_values.min())
        max_log2 = np.ceil(valid_values.max())

        vmin = min_log2 - 0.5
        vmax = max_log2 + 0.5

        # Create discrete colormap
        n_colors = int(max_log2 - min_log2 + 1)
        from matplotlib import cm

        viridis = cm.viridis
        indices = np.linspace(0, 1, n_colors)
        colors_to_use = [viridis(i) for i in indices]
        cmap = ListedColormap(colors_to_use)

        # Create heatmap with log2 scaled data
        im = ax.imshow(matrix_log2, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Set ticks
        ax.set_xticks(np.arange(len(batch_sizes)))
        ax.set_yticks(np.arange(len(seq_lengths)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticklabels([f"{s}k" for s in seq_lengths])

        # Labels
        ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
        ax.set_title(
            f"num_kv_splits Chosen by {heuristic_name} Policy",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add text annotations (show actual value and mark mismatches)
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

        # Colorbar with power-of-2 labels
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("num_kv_splits", rotation=270, labelpad=20, fontsize=12)
        tick_positions = np.arange(min_log2, max_log2 + 1)
        tick_labels = [str(int(2**i)) for i in tick_positions]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)

        plt.tight_layout()

        # Save with sanitized filename
        safe_name = (
            heuristic_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        output_path = output_dir / f"numsplits_policy_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {heuristic_name} policy heatmap to {output_path}")
        plt.close()


def create_heuristic_speedup_heatmaps(
    results: list, optimal_splits: dict[tuple[int, int], int], output_dir: Path
):
    """Create speedup heatmaps for each heuristic policy."""
    # Define heuristics to compare
    heuristics = {
        "Ratio-based (Original)": heuristic_ratio_based,
        "Constant (batch<=32)": heuristic_constant,
    }

    # Group results by batch_spec for performance lookup
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

    # Extract unique batch sizes and sequence lengths
    batch_sizes = sorted(set(b for b, _ in optimal_splits))
    seq_lengths = sorted(set(s for _, s in optimal_splits), reverse=True)

    # Create a separate heatmap for each heuristic
    for heuristic_name, heuristic_func in heuristics.items():
        # Build speedup matrix for this heuristic
        speedup_matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
        total_speedup = 0.0
        count = 0

        for i, seq_len in enumerate(seq_lengths):
            for j, batch_size in enumerate(batch_sizes):
                batch_spec = f"{batch_size}q1s{seq_len}k"
                if batch_spec not in by_batch_spec:
                    speedup_matrix[i, j] = np.nan
                    continue

                timings = by_batch_spec[batch_spec]
                baseline_time = timings.get(1, None)

                if not baseline_time:
                    speedup_matrix[i, j] = np.nan
                    continue

                # Get the num_splits predicted by this heuristic
                predicted_splits = heuristic_func(batch_size, seq_len)
                predicted_time = timings.get(predicted_splits, baseline_time)
                speedup = baseline_time / predicted_time

                speedup_matrix[i, j] = speedup
                total_speedup += speedup
                count += 1

        avg_speedup = total_speedup / count if count > 0 else 1.0

        # Create heatmap
        _fig, ax = plt.subplots(figsize=(12, 8))

        # Colormap: 1.0 = white (neutral), higher = green (good)
        max_speedup = np.nanmax(speedup_matrix)
        colors_dict = {
            "red": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            "green": [(0.0, 1.0, 1.0), (1.0, 0.5, 0.5)],
            "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        }
        speedup_cmap = LinearSegmentedColormap("Speedup", colors_dict)

        im = ax.imshow(
            speedup_matrix,
            cmap=speedup_cmap,
            aspect="auto",
            vmin=1.0,
            vmax=max_speedup,
        )

        # Set ticks
        ax.set_xticks(np.arange(len(batch_sizes)))
        ax.set_yticks(np.arange(len(seq_lengths)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticklabels([f"{s}k" for s in seq_lengths])

        # Labels
        ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Speedup with {heuristic_name} Policy\n"
            f"(Average speedup: {avg_speedup:.3f}x vs. splits=1)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add text annotations
        for i in range(len(seq_lengths)):
            for j in range(len(batch_sizes)):
                value = speedup_matrix[i, j]
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

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Speedup Factor", rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()

        # Save with sanitized filename
        safe_name = (
            heuristic_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        output_path = output_dir / f"numsplits_speedup_{safe_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {heuristic_name} speedup heatmap to {output_path}")
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

    print("Extracting optimal splits...")
    optimal_splits = extract_optimal_splits(results)

    print(f"Found {len(optimal_splits)} configurations")

    # Create visualizations
    print("\nGenerating visualizations...")

    create_heatmap(optimal_splits, output_dir / "numsplits_heatmap.png")
    create_performance_heatmap(results, output_dir / "numsplits_speedup.png")
    create_heuristic_policy_heatmaps(optimal_splits, output_dir)
    create_heuristic_speedup_heatmaps(results, optimal_splits, output_dir)

    # Analyze pattern
    analyze_pattern(optimal_splits)

    print("\nDone! Check the output directory for visualization files.")


if __name__ == "__main__":
    main()
