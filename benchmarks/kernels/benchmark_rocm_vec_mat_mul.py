# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark for the ROCm vecMatMul (skinny GEMM) kernel.

Usage:
    python benchmarks/kernels/benchmark_rocm_vec_mat_mul.py
    python benchmarks/kernels/benchmark_rocm_vec_mat_mul.py --dtype float16
    python benchmarks/kernels/benchmark_rocm_vec_mat_mul.py --m 4096 --k 8192
"""

import argparse
import itertools
import time

import torch

import vllm._custom_ops as ops

# (M, K) pairs representative of real inference shapes:
#   M = hidden / output dim, K = input / embedding dim
DEFAULT_SHAPES = [
    (256, 512),
    (256, 4096),
    (256, 8192),
    (1024, 512),
    (1024, 4096),
    (1024, 8192),
    (4096, 512),
    (4096, 4096),
    (4096, 8192),
]

ROWS_PER_BLOCK_CHOICES = [2, 4, 8, 16]
WARMUP = 50
ITERS = 500


def time_kernel(fn, warmup: int, iters: int) -> float:
    """Returns mean latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.accelerator.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--m", type=int, default=None, help="Override M dimension")
    parser.add_argument("--k", type=int, default=None, help="Override K dimension")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    shapes = DEFAULT_SHAPES
    if args.m is not None and args.k is not None:
        shapes = [(args.m, args.k)]

    row = "{:>6}  {:>6}  {:>4}  {:>14}  {:>11}  {:>9}  {:>8}"
    print(f"dtype={args.dtype}  warmup={WARMUP}  iters={ITERS}")
    print(row.format("M", "K", "rpb", "vecMatMul us", "matmul us", "speedup", "GB/s"))
    print("-" * 68)

    for (m, k), rows_per_block in itertools.product(shapes, ROWS_PER_BLOCK_CHOICES):
        if m % rows_per_block != 0:
            continue

        device = torch.accelerator.current_accelerator()
        mat = torch.randn(m, k, dtype=dtype, device=device)
        vec = torch.randn(1, k, dtype=dtype, device=device)

        vec_us = time_kernel(
            lambda _m=mat, _v=vec, _r=rows_per_block: ops.vecMatMul(_m, _v, _r),
            WARMUP,
            ITERS,
        )
        mat_us = time_kernel(
            lambda _v=vec, _m=mat: torch.matmul(_v, _m.t()), WARMUP, ITERS
        )
        speedup = mat_us / vec_us

        # bytes read: mat (M*K) + vec (K); bytes written: out (M)
        bytes_accessed = (m * k + k + m) * torch.finfo(dtype).bits // 8
        bandwidth = bytes_accessed / (vec_us * 1e-6) / 1e9  # GB/s

        print(
            row.format(
                m,
                k,
                rows_per_block,
                f"{vec_us:.2f}",
                f"{mat_us:.2f}",
                f"{speedup:.2f}x",
                f"{bandwidth:.1f}",
            )
        )


if __name__ == "__main__":
    main()
