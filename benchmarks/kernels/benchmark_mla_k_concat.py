# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script comparing torch.cat vs direct copy for k_nope/k_pe concatenation
in MLA (Multi-head Latent Attention) prefill.

This validates that the optimization from commit 8d4142bd is beneficial across
various batch sizes, not just the originally tested batch size of 32768.
"""

import time
from collections.abc import Callable

import torch

# DeepSeek-V3 MLA dimensions
NUM_HEADS = 128
QK_NOPE_HEAD_DIM = 128
PE_DIM = 64


def cat_method(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """Original torch.cat approach with expand."""
    return torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)


def direct_copy_method(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    """Optimized direct copy approach (avoids expand + cat overhead)."""
    k = torch.empty(
        (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
        dtype=k_nope.dtype,
        device=k_nope.device,
    )
    k[..., : k_nope.shape[-1]] = k_nope
    k[..., k_nope.shape[-1] :] = k_pe
    return k


def benchmark_method(
    method: Callable,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> float:
    """Benchmark a concatenation method and return mean latency in ms."""
    # Warmup
    for _ in range(num_warmup):
        _ = method(k_nope, k_pe)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = method(k_nope, k_pe)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_iters * 1000  # Convert to ms


@torch.inference_mode()
def run_benchmark(dtype: torch.dtype, dtype_name: str):
    """Run benchmark for a specific dtype."""
    torch.set_default_device("cuda")

    # Batch sizes to test (powers of 2 from 32 to 65536)
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print("=" * 80)
    print("Benchmark: torch.cat vs direct copy for MLA k_nope/k_pe concatenation")
    print("=" * 80)
    print(
        f"Tensor shapes: k_nope=[B, {NUM_HEADS}, {QK_NOPE_HEAD_DIM}], "
        f"k_pe=[B, 1, {PE_DIM}]"
    )
    print(f"dtype: {dtype_name}")
    print()
    print(
        f"{'Batch Size':>12} | {'cat (ms)':>10} | {'direct (ms)':>12} | "
        f"{'Speedup':>8} | {'Reduction':>10}"
    )
    print("-" * 70)

    results = []
    for batch_size in batch_sizes:
        # Create input tensors (generate in float32 then convert for FP8 compatibility)
        k_nope = torch.randn(
            batch_size, NUM_HEADS, QK_NOPE_HEAD_DIM, dtype=torch.float32, device="cuda"
        ).to(dtype)
        k_pe = torch.randn(
            batch_size, 1, PE_DIM, dtype=torch.float32, device="cuda"
        ).to(dtype)

        # Benchmark both methods
        cat_time = benchmark_method(cat_method, k_nope, k_pe)
        direct_time = benchmark_method(direct_copy_method, k_nope, k_pe)

        speedup = cat_time / direct_time
        reduction = (1 - direct_time / cat_time) * 100

        results.append((batch_size, cat_time, direct_time, speedup, reduction))

        print(
            f"{batch_size:>12} | {cat_time:>10.3f} | {direct_time:>12.3f} | "
            f"{speedup:>7.2f}x | {reduction:>9.1f}%"
        )

    print("=" * 80)

    # Summary statistics
    speedups = [r[3] for r in results]
    print("\nSpeedup summary:")
    print(f"  Min:  {min(speedups):.2f}x")
    print(f"  Max:  {max(speedups):.2f}x")
    print(f"  Mean: {sum(speedups) / len(speedups):.2f}x")

    # Find crossover point
    crossover_batch = None
    for batch_size, _, _, speedup, _ in results:
        if speedup >= 1.0:
            crossover_batch = batch_size
            break

    print("\nConclusion:")
    if crossover_batch:
        print(f"  - Direct copy becomes beneficial at batch size >= {crossover_batch}")
    # Filter for large batches (>= 512 which is typical for prefill)
    large_batch_speedups = [r[3] for r in results if r[0] >= 512]
    if large_batch_speedups:
        avg_large = sum(large_batch_speedups) / len(large_batch_speedups)
        print(f"  - For batch sizes >= 512: avg speedup = {avg_large:.2f}x")
    print("  - MLA prefill typically uses large batches, so optimization is effective")

    return results


@torch.inference_mode()
def main():
    # Test bfloat16
    print("\n")
    run_benchmark(torch.bfloat16, "bfloat16")

    # Test float8_e4m3fn
    print("\n")
    run_benchmark(torch.float8_e4m3fn, "float8_e4m3fn")


if __name__ == "__main__":
    main()
