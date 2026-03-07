# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for triton kernel launch caching.

Measures the Python-side dispatch overhead of triton kernel launches
with and without CachedKernel, using a near-no-op kernel to isolate
dispatch cost from GPU compute time.

Usage:
    python benchmarks/kernels/benchmark_triton_cache.py
"""

import time

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _noop_kernel(
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Minimal kernel — single store per program."""
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, pid)


def bench_triton_jit(out: torch.Tensor, n: int, warmup: int, iters: int) -> float:
    """Benchmark standard triton JIT dispatch."""
    for _ in range(warmup):
        _noop_kernel[(1,)](out, n, BLOCK_SIZE=64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _noop_kernel[(1,)](out, n, BLOCK_SIZE=64)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_cached(out: torch.Tensor, n: int, warmup: int, iters: int) -> float:
    """Benchmark CachedKernel dispatch (via global patch)."""
    from vllm.triton_utils.cache import patch_triton_kernel_launches

    patch_triton_kernel_launches()

    for _ in range(warmup):
        _noop_kernel[(1,)](out, n, BLOCK_SIZE=64)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _noop_kernel[(1,)](out, n, BLOCK_SIZE=64)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main():
    device = torch.device("cuda")
    out = torch.zeros(1, dtype=torch.int32, device=device)
    n = 1

    warmup = 200
    iters = 5000

    # Must run JIT first (before patching).
    jit_us = bench_triton_jit(out, n, warmup, iters) * 1e6

    # Now patch and benchmark cached dispatch.
    cached_us = bench_cached(out, n, warmup, iters) * 1e6

    print(f"Triton JIT dispatch:    {jit_us:8.2f} us/call")
    print(f"CachedKernel dispatch:  {cached_us:8.2f} us/call")
    print(f"Speedup:                {jit_us / cached_us:8.2f}x")


if __name__ == "__main__":
    main()
