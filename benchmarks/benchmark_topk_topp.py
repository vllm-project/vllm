#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark comparing Triton vs PyTorch sort-based top-k/top-p implementations.

Compares:
- apply_top_k_top_p_triton (Triton binary search)
- apply_top_k_top_p (PyTorch sort-based)

Scenarios:
- top_k only (whole batch, partial batch)
- top_p only (whole batch, partial batch)
- mix of top_k and top_p
"""

import argparse
import gc
from dataclasses import dataclass

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton,
    reset_buffer_cache,
)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    batch_size: int
    vocab_size: int
    # k and p can be tensors or None
    k_values: torch.Tensor | None  # [batch_size] or None
    p_values: torch.Tensor | None  # [batch_size] or None
    description: str
    ops_pct: float = 0.0  # Percentage of ops relative to batch size


def calculate_ops_pct(
    k_values: torch.Tensor | None,
    p_values: torch.Tensor | None,
    vocab_size: int,
    batch_size: int,
) -> float:
    """
    Calculate the percentage of active top-k and top-p operations.

    Returns percentage where 100% = batch_size ops.
    E.g., if all rows have both top-k and top-p active, returns 200%.
    """
    active_ops = 0

    if k_values is not None:
        # Count rows where k < vocab_size (active top-k filtering)
        active_ops += (k_values < vocab_size).sum().item()

    if p_values is not None:
        # Count rows where p < 1.0 (active top-p filtering)
        active_ops += (p_values < 1.0).sum().item()

    return (active_ops / batch_size) * 100 if batch_size > 0 else 0.0


def create_logits(
    batch_size: int, vocab_size: int, device: str = "cuda"
) -> torch.Tensor:
    """Create random logits tensor."""
    return torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device)


def measure_memory() -> tuple[int, int]:
    """Return (allocated, reserved) memory in bytes."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()


def reset_memory_stats():
    """Reset peak memory statistics."""
    reset_buffer_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def benchmark_function(
    func,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> tuple[float, int]:
    """
    Benchmark a function and return (avg_time_ms, peak_memory_bytes).

    Returns average time in milliseconds and peak memory usage.
    """
    # Warmup
    for _ in range(warmup_iters):
        logits_copy = logits.clone()
        func(logits_copy, k, p)
    torch.cuda.synchronize()

    # Reset memory stats before benchmark
    reset_memory_stats()

    # Benchmark
    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)
    ]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]

    for i in range(benchmark_iters):
        logits_copy = logits.clone()
        start_events[i].record()
        func(logits_copy, k, p)
        end_events[i].record()

    torch.cuda.synchronize()

    # Calculate timing
    times = [
        start_events[i].elapsed_time(end_events[i]) for i in range(benchmark_iters)
    ]
    avg_time = sum(times) / len(times)

    # Get peak memory
    _, peak_memory = measure_memory()

    return avg_time, peak_memory


def create_benchmark_configs(
    batch_sizes: list[int],
    vocab_sizes: list[int],
    device: str = "cuda",
) -> list[BenchmarkConfig]:
    """Create all benchmark configurations."""
    configs = []

    for vocab_size in vocab_sizes:
        for batch_size in batch_sizes:
            # 1. Top-k only - whole batch (all rows have k < vocab_size)
            k_all = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topk_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_all,
                    p_values=None,
                    description=f"Top-k only (whole batch, k=50), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_all, None, vocab_size, batch_size),
                )
            )

            # 2. Top-k only - partial batch (half have k=50, half have k=vocab_size)
            k_partial = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
            k_partial[batch_size // 2 :] = vocab_size  # No filtering for second half
            configs.append(
                BenchmarkConfig(
                    name=f"topk_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_partial,
                    p_values=None,
                    description=f"Top-k only (partial batch, 50% k=50, 50% k=vocab), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_partial, None, vocab_size, batch_size),
                )
            )

            # 3. Top-p only - whole batch (all rows have p < 1.0)
            p_all = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topp_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=p_all,
                    description=f"Top-p only (whole batch, p=0.9), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(None, p_all, vocab_size, batch_size),
                )
            )

            # 4. Top-p only - partial batch (half have p=0.9, half have p=1.0)
            p_partial = torch.full(
                (batch_size,), 0.9, dtype=torch.float32, device=device
            )
            p_partial[batch_size // 2 :] = 1.0  # No filtering for second half
            configs.append(
                BenchmarkConfig(
                    name=f"topp_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=None,
                    p_values=p_partial,
                    description=f"Top-p only (partial batch, 50% p=0.9, 50% p=1.0), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(None, p_partial, vocab_size, batch_size),
                )
            )

            # 5. Mix of top-k and top-p (both applied to whole batch)
            k_mix = torch.full((batch_size,), 100, dtype=torch.int32, device=device)
            p_mix = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
            configs.append(
                BenchmarkConfig(
                    name=f"topk_topp_whole_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_mix,
                    p_values=p_mix,
                    description=f"Top-k + Top-p (whole batch, k=100, p=0.9), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_mix, p_mix, vocab_size, batch_size),
                )
            )

            # 6. Mix with partial application (some rows k only, some p only, some both)
            k_mixed = torch.full(
                (batch_size,), vocab_size, dtype=torch.int32, device=device
            )
            p_mixed = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)
            # First third: k only
            third = batch_size // 3
            k_mixed[:third] = 50
            # Second third: p only
            p_mixed[third : 2 * third] = 0.9
            # Last third: both k and p
            k_mixed[2 * third :] = 100
            p_mixed[2 * third :] = 0.9
            configs.append(
                BenchmarkConfig(
                    name=f"mixed_partial_b{batch_size}_v{vocab_size // 1000}k",
                    batch_size=batch_size,
                    vocab_size=vocab_size,
                    k_values=k_mixed,
                    p_values=p_mixed,
                    description=f"Mixed partial (1/3 k=50, 1/3 p=0.9, 1/3 both), "
                    f"batch={batch_size}, vocab={vocab_size}",
                    ops_pct=calculate_ops_pct(k_mixed, p_mixed, vocab_size, batch_size),
                )
            )

    return configs


def format_memory(bytes_val: int) -> str:
    """Format memory in human-readable form."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"


def run_benchmark(
    configs: list[BenchmarkConfig],
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    verbose: bool = True,
):
    """Run all benchmarks and print results."""
    results = []

    print("=" * 100)
    print("Top-k/Top-p Benchmark: Triton vs PyTorch Sort-based")
    print("=" * 100)
    print()

    for config in configs:
        if verbose:
            print(f"Running: {config.description}")

        # Create fresh logits for this config
        logits = create_logits(config.batch_size, config.vocab_size)

        # Benchmark Triton
        reset_memory_stats()
        triton_time, triton_mem = benchmark_function(
            apply_top_k_top_p_triton,
            logits,
            config.k_values,
            config.p_values,
            warmup_iters,
            benchmark_iters,
        )

        # Benchmark PyTorch
        reset_memory_stats()
        pytorch_time, pytorch_mem = benchmark_function(
            apply_top_k_top_p_pytorch,
            logits,
            config.k_values,
            config.p_values,
            warmup_iters,
            benchmark_iters,
        )

        speedup = pytorch_time / triton_time if triton_time > 0 else float("inf")
        mem_ratio = pytorch_mem / triton_mem if triton_mem > 0 else float("inf")

        result = {
            "config": config,
            "triton_time_ms": triton_time,
            "pytorch_time_ms": pytorch_time,
            "triton_mem": triton_mem,
            "pytorch_mem": pytorch_mem,
            "speedup": speedup,
            "mem_ratio": mem_ratio,
        }
        results.append(result)

        if verbose:
            print(f"  Triton:  {triton_time:.3f} ms, {format_memory(triton_mem)}")
            print(f"  PyTorch: {pytorch_time:.3f} ms, {format_memory(pytorch_mem)}")
            print(f"  Speedup: {speedup:.2f}x, Memory ratio: {mem_ratio:.2f}x")
            print()

        # Clean up
        del logits
        reset_memory_stats()

    return results


def print_summary_table(results: list[dict]):
    """Print a summary table of results."""
    print()
    print("=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    print()

    # Header
    header = (
        f"{'Scenario':<40} {'Batch':>6} {'Vocab':>7} {'Ops%':>6} "
        f"{'Triton (ms)':>12} {'PyTorch (ms)':>13} {'Speedup':>8} "
        f"{'Tri Mem':>10} {'Pyt Mem':>10}"
    )
    print(header)
    print("-" * 130)

    # Group by scenario type
    current_vocab = None
    for result in results:
        config = result["config"]

        # Add separator between vocab sizes
        if current_vocab != config.vocab_size:
            if current_vocab is not None:
                print("-" * 130)
            current_vocab = config.vocab_size

        scenario = config.name.split("_b")[0]  # Extract scenario name
        print(
            f"{scenario:<40} {config.batch_size:>6} {config.vocab_size:>7} "
            f"{config.ops_pct:>5.0f}% "
            f"{result['triton_time_ms']:>12.3f} {result['pytorch_time_ms']:>13.3f} "
            f"{result['speedup']:>7.2f}x "
            f"{format_memory(result['triton_mem']):>10} "
            f"{format_memory(result['pytorch_mem']):>10}"
        )

    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton vs PyTorch sort-based top-k/top-p implementations"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16, 24, 32, 48, 56, 64, 96, 128, 192, 256, 512, 1024],
        help="Batch sizes to test (default: 1 4 16 64)",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        default=[32768, 131072],  # 32k, 128k
        help="Vocabulary sizes to test (default: 32768 131072)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
        help="Number of benchmark iterations (default: 20)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary table",
    )

    args = parser.parse_args()

    # Print configuration
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Vocab sizes: {args.vocab_sizes}")
    print(f"Warmup iterations: {args.warmup_iters}")
    print(f"Benchmark iterations: {args.benchmark_iters}")
    print()

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print()

    # Create configs
    configs = create_benchmark_configs(
        args.batch_sizes,
        args.vocab_sizes,
    )

    # Run benchmarks
    results = run_benchmark(
        configs,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary_table(results)


if __name__ == "__main__":
    main()
