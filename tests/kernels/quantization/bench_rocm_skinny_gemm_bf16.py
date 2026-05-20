#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the bf16/fp16 wvSplitK skinny GEMM kernel.

Measures throughput and weight bandwidth for representative model shapes
at batch sizes 1-4. Validates accuracy against torch.mm. Dynamically
determines iteration count per shape based on IQR convergence.

Usage:
    python tests/kernels/quantization/bench_rocm_skinny_gemm_bf16.py
    python tests/kernels/quantization/bench_rocm_skinny_gemm_bf16.py --dtype bf16
    python tests/kernels/quantization/bench_rocm_skinny_gemm_bf16.py --batch-sizes 1 4
    python tests/kernels/quantization/bench_rocm_skinny_gemm_bf16.py --shapes 4096x4096
"""

import argparse
import math
import time

import torch

import vllm._custom_ops as ops
from vllm.utils.platform_utils import num_compute_units as get_cu_count

# Infinity Cache on Strix Halo is 32-64MB depending on SKU.
# Use a conservative estimate to ensure we bust L3.
CACHE_SIZE_BYTES = 64 * 1024 * 1024

SHAPES = [
    # Qwen3-4B / Qwen3-VL-4B (identical backbone)
    (6144, 2560, "Qwen3-4B qkv"),
    (2560, 4096, "Qwen3-4B o_proj"),
    (19456, 2560, "Qwen3-4B gate_up"),
    (2560, 9728, "Qwen3-4B down"),
    (151936, 2560, "Qwen3-4B lm_head"),
    # Qwen2.5-VL-7B
    (4608, 3584, "Qwen2.5VL-7B qkv"),
    (3584, 3584, "Qwen2.5VL-7B o_proj"),
    (37888, 3584, "Qwen2.5VL-7B gate_up"),
    (3584, 18944, "Qwen2.5VL-7B down"),
    (152064, 3584, "Qwen2.5VL-7B lm_head"),
    # Qwen3.5-35B-A3B (vocab=248320, hidden=2048)
    (248320, 2048, "Qwen3.5-35B-A3B lm_head"),
    (1024, 2048, "Qwen3.5-35B-A3B 1024 proj"),
    # Llama-3.1-8B (hidden=4096, intermediate=14336, vocab=128256)
    (4096, 4096, "Llama-8B q/o_proj"),
    (6144, 4096, "Llama-8B qkv"),
    (28672, 4096, "Llama-8B gate_up"),
    (4096, 14336, "Llama-8B down"),
    (128256, 4096, "Llama-8B lm_head"),
]


def _median_se(times_sorted):
    """Standard error of the median as % of median, using MAD estimator."""
    n = len(times_sorted)
    med = times_sorted[n // 2]
    if med == 0 or n < 3:
        return med, 0.0
    mad = sorted(abs(t - med) for t in times_sorted)[n // 2]
    # SE_median ≈ 1.253 * σ / √n, with σ ≈ 1.4826 * MAD
    se = 1.253 * 1.4826 * mad / math.sqrt(n)
    return med, se / med * 100


def bench_dynamic(fn, target_se_pct=0.1, min_iters=20, max_iters=5000, max_time_s=2.0):
    """Benchmark fn with adaptive iteration count.

    Collects per-iteration GPU times via CUDA events. Stops when the
    standard error of the median drops below target_se_pct of the median,
    or when max_iters / max_time_s is reached.

    fn is called as fn(iteration_index) so callers can rotate buffers
    to avoid cache hits.

    Returns (median_ms, num_iters, se_pct).
    """
    for i in range(10):
        fn(i)
    torch.accelerator.synchronize()

    times = []
    start_ev = torch.Event(enable_timing=True)
    end_ev = torch.Event(enable_timing=True)
    wall_start = time.monotonic()

    for i in range(max_iters):
        start_ev.record()
        fn(i)
        end_ev.record()
        torch.accelerator.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

        if len(times) >= min_iters and len(times) % 10 == 0:
            med, se_pct = _median_se(sorted(times))
            if se_pct < target_se_pct:
                return med, len(times), se_pct
            if time.monotonic() - wall_start > max_time_s:
                return med, len(times), se_pct

    med, se_pct = _median_se(sorted(times))
    return med, len(times), se_pct


def parse_shape(s):
    parts = s.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Shape must be MxK, got '{s}'")
    return (int(parts[0]), int(parts[1]), s)


def run_bench(shapes, batch_sizes, dtype, target_se_pct):
    cu_count = get_cu_count()
    gpu_name = torch.cuda.get_device_name(0)
    dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"

    print(f"GPU: {gpu_name}, CU count: {cu_count}")
    print(f"dtype: {dtype_name}, target SE: {target_se_pct}%")
    print(f"Shapes: {len(shapes)}, Batch sizes: {batch_sizes}")
    print()

    print(
        f"{'N':>2} {'M':>6}x{'K':<6} {'Label':<22} "
        f"{'time_us':>9} {'BW GiB/s':>9} {'bufs':>5} {'iters':>6} {'SE%':>5}"
    )
    print("-" * 80)

    t0 = time.time()
    for M, K, label in shapes:
        for N in batch_sizes:
            xavier = math.sqrt(2 / K)
            weight = (torch.rand(M, K, dtype=dtype, device="cuda") * 2 - 1) * xavier
            activation = (torch.rand(N, K, dtype=dtype, device="cuda") * 2 - 1) * xavier

            ref_out = torch.mm(activation, weight.t())
            out = ops.wvSplitK(weight, activation, cu_count)
            atol = max(1e-3, torch.finfo(dtype).eps * math.sqrt(K))
            torch.testing.assert_close(out, ref_out, atol=atol, rtol=1e-2)

            weight_bytes = M * K * dtype.itemsize
            n_bufs = max(1, CACHE_SIZE_BYTES // weight_bytes + 1)
            weights = [
                (torch.rand(M, K, dtype=dtype, device="cuda") * 2 - 1) * xavier
                for _ in range(n_bufs)
            ]

            fn = lambda i, ws=weights, a=activation: ops.wvSplitK(
                ws[i % len(ws)], a, cu_count
            )
            med_ms, iters, se_pct = bench_dynamic(
                fn,
                target_se_pct=target_se_pct,
            )
            time_us = med_ms * 1000
            bw_gibs = weight_bytes / (med_ms * 1e-3) / (1 << 30)

            print(
                f"{N:>2} {M:>6}x{K:<6} {label:<22} "
                f"{time_us:>8.1f} {bw_gibs:>8.1f} {n_bufs:>5} {iters:>6} {se_pct:>5.2f}"
            )

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bf16/fp16 wvSplitK skinny GEMM"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Batch sizes (N) to test (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=None,
        help="Shapes as MxK (default: all representative shapes)",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="Data type (default: bf16)",
    )
    parser.add_argument(
        "--target-se",
        type=float,
        default=0.1,
        help="Stop when SE of median < this %% of median (default: 0.1)",
    )
    args = parser.parse_args()

    shapes = args.shapes if args.shapes else SHAPES
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    run_bench(shapes, args.batch_sizes, dtype, args.target_se)


if __name__ == "__main__":
    main()
