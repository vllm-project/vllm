#!/usr/bin/env python3
"""
Analyze KV cache rank structure via SVD decomposition.

This script analyzes the intrinsic dimensionality of KV cache tensors
by computing effective rank at different context lengths and window sizes.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def compute_effective_rank(
    matrix: torch.Tensor, energy_threshold: float = 0.99
) -> dict:
    """
    Compute effective rank of a matrix using SVD.

    Args:
        matrix: 2D tensor [S, d] where S is sequence length, d is head dimension
        energy_threshold: Fraction of total energy to capture (default 0.99)

    Returns:
        dict with rank statistics
    """
    if matrix.shape[0] < 2:
        return {
            "effective_rank": 1,
            "full_rank": min(matrix.shape),
            "singular_values": [1.0],
            "energy_ratio": 1.0,
        }

    # Compute SVD
    U, s, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)

    # Compute energy (squared singular values)
    s_squared = s**2
    total_energy = s_squared.sum()

    if total_energy < 1e-10:
        return {
            "effective_rank": 0,
            "full_rank": min(matrix.shape),
            "singular_values": s.tolist(),
            "energy_ratio": 0.0,
        }

    # Cumulative energy ratio
    cumsum = s_squared.cumsum(0) / total_energy

    # Effective rank: minimum k such that sum(s[0:k]^2) / sum(s^2) >= threshold
    effective_rank = int((cumsum < energy_threshold).sum().item()) + 1
    effective_rank = min(effective_rank, len(s))

    # Also compute rank at other thresholds
    rank_90 = int((cumsum < 0.90).sum().item()) + 1
    rank_95 = int((cumsum < 0.95).sum().item()) + 1
    rank_99 = int((cumsum < 0.99).sum().item()) + 1
    rank_999 = int((cumsum < 0.999).sum().item()) + 1

    return {
        "effective_rank": effective_rank,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "rank_99": rank_99,
        "rank_999": rank_999,
        "full_rank": min(matrix.shape),
        "max_possible_rank": min(matrix.shape),
        "singular_values": s[:10].tolist(),  # Top 10 for brevity
        "energy_ratio": cumsum[effective_rank - 1].item()
        if effective_rank > 0
        else 0.0,
        "condition_number": (s[0] / s[-1]).item() if s[-1] > 1e-10 else float("inf"),
    }


def analyze_at_context_lengths(
    kv_tensor: torch.Tensor,
    context_lengths: list[int],
) -> dict:
    """
    Analyze rank at different context lengths (prefix of sequence).

    Args:
        kv_tensor: [S, H_kv, d] tensor
        context_lengths: List of context lengths to analyze

    Returns:
        dict mapping context_length -> head -> rank_stats
    """
    S, H_kv, d = kv_tensor.shape
    results = {}

    for ctx_len in context_lengths:
        if ctx_len > S:
            continue

        # Take first ctx_len tokens
        prefix = kv_tensor[:ctx_len]  # [ctx_len, H_kv, d]

        head_results = {}
        for h in range(H_kv):
            head_matrix = prefix[:, h, :]  # [ctx_len, d]
            head_results[f"head_{h}"] = compute_effective_rank(head_matrix)

        # Also compute aggregate stats
        all_ranks = [head_results[f"head_{h}"]["effective_rank"] for h in range(H_kv)]
        head_results["aggregate"] = {
            "mean_rank": np.mean(all_ranks),
            "min_rank": min(all_ranks),
            "max_rank": max(all_ranks),
            "std_rank": np.std(all_ranks),
        }

        results[ctx_len] = head_results

    return results


def analyze_sliding_window(
    kv_tensor: torch.Tensor,
    window_sizes: list[int],
    stride: int = None,
) -> dict:
    """
    Analyze rank within sliding windows of different sizes.

    Args:
        kv_tensor: [S, H_kv, d] tensor
        window_sizes: List of window sizes to analyze
        stride: Stride for sliding window (default: window_size // 2)

    Returns:
        dict mapping window_size -> statistics across all windows
    """
    S, H_kv, d = kv_tensor.shape
    results = {}

    for window_size in window_sizes:
        if window_size > S:
            continue

        ws_stride = stride if stride else max(1, window_size // 4)

        # Collect ranks from all windows and all heads
        all_window_ranks = []
        window_details = []

        for start in range(0, S - window_size + 1, ws_stride):
            end = start + window_size
            window = kv_tensor[start:end]  # [window_size, H_kv, d]

            window_head_ranks = []
            for h in range(H_kv):
                head_matrix = window[:, h, :]  # [window_size, d]
                rank_stats = compute_effective_rank(head_matrix)
                window_head_ranks.append(rank_stats["effective_rank"])
                all_window_ranks.append(rank_stats["effective_rank"])

            window_details.append(
                {
                    "start": start,
                    "end": end,
                    "mean_rank": np.mean(window_head_ranks),
                    "ranks_by_head": window_head_ranks,
                }
            )

        results[window_size] = {
            "num_windows": len(window_details),
            "mean_rank_across_windows": np.mean(all_window_ranks),
            "std_rank_across_windows": np.std(all_window_ranks),
            "min_rank": min(all_window_ranks),
            "max_rank": max(all_window_ranks),
            "max_possible_rank": min(window_size, d),
            "rank_ratio": np.mean(all_window_ranks) / min(window_size, d),
            # Per-window breakdown (first 5 and last 5)
            "window_samples": window_details[:3] + window_details[-3:]
            if len(window_details) > 6
            else window_details,
        }

    return results


def analyze_dump_file(filepath: str, verbose: bool = True) -> dict:
    """
    Analyze a single KV dump file.

    Args:
        filepath: Path to safetensors file
        verbose: Print progress

    Returns:
        Complete analysis results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {Path(filepath).name}")
        print(f"{'=' * 60}")

    # Load tensors and metadata
    tensors = load_file(filepath)
    with safe_open(filepath, framework="pt") as f:
        metadata = f.metadata()

    # Extract info
    S = tensors["token_ids"].shape[0]
    d = int(metadata.get("d", 64))
    H_kv = int(metadata.get("H_kv", 1))

    if verbose:
        print(f"Sequence length (S): {S}")
        print(f"Head dimension (d): {d}")
        print(f"KV heads (H_kv): {H_kv}")
        print(f"Layers: {metadata.get('layer_ids', 'unknown')}")

    # Define analysis parameters
    context_lengths = [8, 16, 32, 64, 128, 256, 512, 1024]
    context_lengths = [c for c in context_lengths if c <= S]
    context_lengths.append(S)  # Always include full sequence
    context_lengths = sorted(set(context_lengths))

    window_sizes = [64, 128, 256, 512]
    window_sizes = [w for w in window_sizes if w <= S]

    results = {
        "filepath": filepath,
        "metadata": metadata,
        "sequence_length": S,
        "head_dim": d,
        "num_kv_heads": H_kv,
        "layers": {},
    }

    # Analyze each layer
    for key in sorted(tensors.keys()):
        if not key.startswith("K_layer"):
            continue

        layer_idx = key.replace("K_layer", "")
        K = tensors[f"K_layer{layer_idx}"]  # [S, H_kv, d]
        V = tensors[f"V_layer{layer_idx}"]  # [S, H_kv, d]

        if verbose:
            print(f"\n--- Layer {layer_idx} ---")
            print(f"K shape: {tuple(K.shape)}, V shape: {tuple(V.shape)}")

        layer_results = {
            "K": {
                "context_length_analysis": analyze_at_context_lengths(
                    K, context_lengths
                ),
                "sliding_window_analysis": analyze_sliding_window(K, window_sizes),
            },
            "V": {
                "context_length_analysis": analyze_at_context_lengths(
                    V, context_lengths
                ),
                "sliding_window_analysis": analyze_sliding_window(V, window_sizes),
            },
        }

        results["layers"][f"layer_{layer_idx}"] = layer_results

        # Print summary for this layer
        if verbose:
            print(f"\n  Key (K) - Full sequence (S={S}):")
            k_full = layer_results["K"]["context_length_analysis"][S]
            for h in range(H_kv):
                h_stats = k_full[f"head_{h}"]
                print(
                    f"    Head {h}: effective_rank={h_stats['effective_rank']}/{h_stats['max_possible_rank']} "
                    f"(99% energy), rank_90={h_stats['rank_90']}, rank_95={h_stats['rank_95']}"
                )

            print(f"\n  Value (V) - Full sequence (S={S}):")
            v_full = layer_results["V"]["context_length_analysis"][S]
            for h in range(H_kv):
                h_stats = v_full[f"head_{h}"]
                print(
                    f"    Head {h}: effective_rank={h_stats['effective_rank']}/{h_stats['max_possible_rank']} "
                    f"(99% energy), rank_90={h_stats['rank_90']}, rank_95={h_stats['rank_95']}"
                )

            # Sliding window summary
            if window_sizes:
                print(f"\n  Sliding Window Analysis (K):")
                for ws in window_sizes:
                    if ws in layer_results["K"]["sliding_window_analysis"]:
                        ws_stats = layer_results["K"]["sliding_window_analysis"][ws]
                        print(
                            f"    Window={ws}: mean_rank={ws_stats['mean_rank_across_windows']:.1f} "
                            f"(ratio={ws_stats['rank_ratio']:.2%}), "
                            f"range=[{ws_stats['min_rank']}, {ws_stats['max_rank']}]"
                        )

    return results


def print_summary_table(all_results: list[dict]):
    """Print a summary table across all analyzed files."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    for result in all_results:
        print(f"\nFile: {Path(result['filepath']).name}")
        print(
            f"S={result['sequence_length']}, d={result['head_dim']}, H_kv={result['num_kv_heads']}"
        )

        # Table header
        print(
            f"\n{'Layer':<10} {'Type':<5} {'Head':<6} {'Rank@99%':<10} {'Rank@95%':<10} {'Rank@90%':<10} {'Max':<6}"
        )
        print("-" * 60)

        for layer_name, layer_data in result["layers"].items():
            layer_idx = layer_name.replace("layer_", "")
            S = result["sequence_length"]

            for kv_type in ["K", "V"]:
                full_stats = layer_data[kv_type]["context_length_analysis"][S]
                for h in range(result["num_kv_heads"]):
                    h_stats = full_stats[f"head_{h}"]
                    print(
                        f"{layer_idx:<10} {kv_type:<5} {h:<6} {h_stats['rank_99']:<10} "
                        f"{h_stats['rank_95']:<10} {h_stats['rank_90']:<10} {h_stats['max_possible_rank']:<6}"
                    )


def plot_rank_vs_context(results: dict, output_path: str = None):
    """Generate plots of rank vs context length."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    for layer_name, layer_data in results["layers"].items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, kv_type in enumerate(["K", "V"]):
            ax = axes[idx]
            ctx_analysis = layer_data[kv_type]["context_length_analysis"]

            ctx_lengths = sorted([c for c in ctx_analysis.keys() if isinstance(c, int)])

            for h in range(results["num_kv_heads"]):
                ranks = [
                    ctx_analysis[c][f"head_{h}"]["effective_rank"] for c in ctx_lengths
                ]
                ax.plot(ctx_lengths, ranks, marker="o", label=f"Head {h}")

            # Plot max possible rank
            max_ranks = [min(c, results["head_dim"]) for c in ctx_lengths]
            ax.plot(ctx_lengths, max_ranks, "k--", alpha=0.5, label="Max rank")

            ax.set_xlabel("Context Length")
            ax.set_ylabel("Effective Rank (99% energy)")
            ax.set_title(f"{layer_name} - {kv_type}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)

        plt.tight_layout()

        if output_path:
            plot_file = Path(output_path) / f"{layer_name}_rank_vs_context.png"
            plt.savefig(plot_file, dpi=150)
            print(f"Saved plot: {plot_file}")
        else:
            plt.show()

        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze KV cache rank structure")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to safetensors file or directory containing them",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for results JSON and plots",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    # Default to the test dump directory
    input_path = args.input or str(Path.home() / "temp/vllm_kv_dump_test")

    # Find all safetensors files
    input_path = Path(input_path)
    if input_path.is_file():
        files = [str(input_path)]
    else:
        files = sorted(glob.glob(str(input_path / "*.safetensors")))

    if not files:
        print(f"No safetensors files found in {input_path}")
        return

    print(f"Found {len(files)} dump file(s)")

    # Analyze each file
    all_results = []
    for filepath in files:
        result = analyze_dump_file(filepath, verbose=not args.quiet)
        all_results.append(result)

        if args.plot:
            plot_rank_vs_context(result, args.output)

    # Print summary
    print_summary_table(all_results)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types for JSON
        def convert_to_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json(v) for v in obj]
            return obj

        results_file = output_dir / "rank_analysis.json"
        with open(results_file, "w") as f:
            json.dump(convert_to_json(all_results), f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
