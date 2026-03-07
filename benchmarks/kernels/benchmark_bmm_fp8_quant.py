# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: Fused BMM+FP8 quant vs separate BMM then FP8 quant.

Measures the latency of the MLA _v_up_proj operation for decode:
  - Baseline: torch.bmm → transpose+reshape → static_scaled_fp8_quant
  - Fused:    bmm_fp8_quant (single Triton kernel)

Usage:
    python benchmarks/kernels/benchmark_bmm_fp8_quant.py
"""

import argparse

import torch

from vllm._custom_ops import static_scaled_fp8_quant
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_bmm_fp8 import bmm_fp8_quant


def benchmark_fn(fn, warmup=100, iters=500):
    """Benchmark a function using CUDA events for accurate timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    return elapsed_ms


def run_benchmark(N, B, L, V, dtype=torch.bfloat16):
    """Run baseline vs fused benchmark for given dimensions."""
    device = torch.device("cuda:0")
    fp8_dtype = current_platform.fp8_dtype()

    inp = torch.randn(N, B, L, dtype=dtype, device=device)
    weight = torch.randn(N, L, V, dtype=dtype, device=device)
    scale = torch.tensor([0.01], dtype=torch.float32, device=device)

    # Pre-allocate outputs
    out_bf16 = torch.empty(B, N * V, dtype=dtype, device=device)
    out_fp8_baseline = torch.empty(B, N * V, dtype=fp8_dtype, device=device)
    out_fp8_fused = torch.empty(B, N * V, dtype=fp8_dtype, device=device)

    # Baseline: torch.bmm + transpose + static_scaled_fp8_quant
    def baseline():
        bmm_result = torch.bmm(inp, weight)  # (N, B, V)
        out_bf16[:] = bmm_result.transpose(0, 1).reshape(B, N * V)
        static_scaled_fp8_quant(out_fp8_baseline, out_bf16, scale)

    # Fused: single Triton kernel
    def fused():
        bmm_fp8_quant(inp, weight, scale, out_fp8_fused)

    baseline_ms = benchmark_fn(baseline)
    fused_ms = benchmark_fn(fused)
    speedup = baseline_ms / fused_ms

    return baseline_ms, fused_ms, speedup


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused BMM+FP8 quant kernel")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256],
        help="Batch sizes (num tokens) to benchmark",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[16, 64, 128],
        help="Number of attention heads",
    )
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--v-head-dim", type=int, default=128)
    args = parser.parse_args()

    L = args.kv_lora_rank
    V = args.v_head_dim

    print(
        f"{'N':>6} {'B':>6} {'L':>6} {'V':>6} | "
        f"{'Baseline (ms)':>14} {'Fused (ms)':>12} {'Speedup':>8}"
    )
    print("-" * 75)

    for N in args.num_heads:
        for B in args.batch_sizes:
            baseline_ms, fused_ms, speedup = run_benchmark(N, B, L, V)
            print(
                f"{N:>6} {B:>6} {L:>6} {V:>6} | "
                f"{baseline_ms:>14.4f} {fused_ms:>12.4f} {speedup:>7.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
