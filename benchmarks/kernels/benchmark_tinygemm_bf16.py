# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark for tinygemm_bf16 optimization in small-batch BF16 GEMMs."""

import argparse
import time

import torch

from vllm.model_executor.layers.utils import default_unquantized_gemm


def benchmark_tinygemm_bf16(
    m: int, n: int, k: int, num_warmup: int = 10, num_iterations: int = 100
):
    """Benchmark tinygemm_bf16 vs torch.nn.functional.linear."""
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn(m, k, dtype=dtype, device=device)
    weight = torch.randn(n, k, dtype=dtype, device=device)

    layer = torch.nn.Linear(k, n, bias=False, dtype=dtype, device=device)
    layer.weight.data = weight

    for _ in range(num_warmup):
        _ = default_unquantized_gemm(layer, x, weight, None)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        out_tinygemm = default_unquantized_gemm(layer, x, weight, None)
    torch.cuda.synchronize()
    tinygemm_time = (time.perf_counter() - start) / num_iterations * 1000

    start = time.perf_counter()
    for _ in range(num_iterations):
        out_linear = torch.nn.functional.linear(x, weight, None)
    torch.cuda.synchronize()
    linear_time = (time.perf_counter() - start) / num_iterations * 1000

    speedup = linear_time / tinygemm_time if tinygemm_time > 0 else 0

    print(f"Shape: M={m}, N={n}, K={k}")
    print(f"  tinygemm_bf16: {tinygemm_time:.4f} ms")
    print(f"  torch.linear:  {linear_time:.4f} ms")
    print(f"  Speedup:       {speedup:.2f}x")

    torch.testing.assert_close(out_tinygemm, out_linear, rtol=1e-2, atol=1e-2)
    print("  ✓ Correctness verified")


def main():
    parser = argparse.ArgumentParser(description="Benchmark tinygemm_bf16 optimization")
    parser.add_argument(
        "--m", type=int, default=1, help="Number of tokens (batch size)"
    )
    parser.add_argument("--n", type=int, default=4096, help="Output dimension")
    parser.add_argument("--k", type=int, default=4096, help="Input dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print(f"Benchmarking tinygemm_bf16 optimization")
    print(f"=" * 50)
    benchmark_tinygemm_bf16(args.m, args.n, args.k, args.warmup, args.iterations)


if __name__ == "__main__":
    main()
