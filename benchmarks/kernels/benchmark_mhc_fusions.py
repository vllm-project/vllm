#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark optimized MHC kernel fusions.

This script measures the performance improvement of fused kernels
compared to separate kernel calls.
"""

import argparse
import time

import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if not has_tilelang() or not current_platform.is_cuda_alike():
    raise RuntimeError("TileLang and CUDA required for MHC benchmarks")

from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.optimized_wrappers import (
    mhc_post_hc_head_fused,
    mhc_post_hc_head_norm_fused,
    mhc_post_mean_fused,
)


def create_test_inputs(num_tokens, hc_mult, hidden_size, device="cuda"):
    """Create test inputs."""
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    post_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    return x, residual, post_mix, comb_mix


def create_hc_head_params(hc_mult, hidden_size, device="cuda"):
    """Create HC head parameters."""
    hc_dim = hc_mult * hidden_size
    fn = torch.randn(hc_mult, hc_dim, device=device, dtype=torch.float32)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    base = torch.randn(hc_mult, device=device, dtype=torch.float32)
    return fn, scale, base


def benchmark_kernel(func, inputs, num_warmup=10, num_iters=100):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(num_warmup):
        output = func(*inputs)
        if isinstance(output, tuple):
            for o in output:
                _ = o

    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        output = func(*inputs)
        if isinstance(output, tuple):
            for o in output:
                _ = o
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / num_iters

    return avg_ms


def benchmark_mhc_post_hc_head(num_tokens, hc_mult, hidden_size, num_iters=100):
    """Benchmark mhc_post + hc_head fusion."""
    print(f"\n{'='*80}")
    print(f"Benchmark: MHC Post + HC Head Fusion")
    print(f"Config: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    print(f"{'='*80}")

    device = "cuda"
    x, residual, post_mix, comb_mix = create_test_inputs(num_tokens, hc_mult, hidden_size, device)
    fn, scale, base = create_hc_head_params(hc_mult, hidden_size, device)

    rms_eps = 1e-6
    hc_eps = 1e-3

    # Benchmark original separate calls
    def original_ops():
        residual_out = mhc_post_tilelang(x, residual, post_mix, comb_mix)
        return hc_head_fused_kernel_tilelang(residual_out, fn, scale, base, rms_eps, hc_eps)

    orig_time = benchmark_kernel(original_ops, [], num_iters=num_iters)

    # Benchmark fused kernel
    def fused_ops():
        return mhc_post_hc_head_fused(
            x, residual, post_mix, comb_mix, fn, scale, base, rms_eps, hc_eps
        )

    fused_time = benchmark_kernel(fused_ops, [], num_iters=num_iters)

    speedup = orig_time / fused_time
    saved_ms = orig_time - fused_time

    print(f"Original (separate):  {orig_time:.3f} ms")
    print(f"Fused:                {fused_time:.3f} ms")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Time saved:           {saved_ms:.3f} ms ({saved_ms/orig_time*100:.1f}%)")

    return {
        "num_tokens": num_tokens,
        "original_ms": orig_time,
        "fused_ms": fused_time,
        "speedup": speedup,
    }


def benchmark_mhc_post_hc_head_norm(num_tokens, hc_mult, hidden_size, num_iters=100):
    """Benchmark mhc_post + hc_head + norm fusion."""
    print(f"\n{'='*80}")
    print(f"Benchmark: MHC Post + HC Head + RMSNorm Fusion")
    print(f"Config: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    print(f"{'='*80}")

    device = "cuda"
    x, residual, post_mix, comb_mix = create_test_inputs(num_tokens, hc_mult, hidden_size, device)
    fn, scale, base = create_hc_head_params(hc_mult, hidden_size, device)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)

    rms_eps = 1e-6
    hc_eps = 1e-3
    norm_eps = 1e-5

    # Benchmark original separate calls
    def original_ops():
        residual_out = mhc_post_tilelang(x, residual, post_mix, comb_mix)
        hc_out = hc_head_fused_kernel_tilelang(residual_out, fn, scale, base, rms_eps, hc_eps)
        # RMSNorm
        variance = hc_out.float().pow(2).mean(-1, keepdim=True)
        return hc_out * torch.rsqrt(variance + norm_eps) * norm_weight

    orig_time = benchmark_kernel(original_ops, [], num_iters=num_iters)

    # Benchmark fused kernel
    def fused_ops():
        return mhc_post_hc_head_norm_fused(
            x, residual, post_mix, comb_mix, fn, scale, base,
            norm_weight, rms_eps, hc_eps, norm_eps
        )

    fused_time = benchmark_kernel(fused_ops, [], num_iters=num_iters)

    speedup = orig_time / fused_time
    saved_ms = orig_time - fused_time

    print(f"Original (separate):  {orig_time:.3f} ms")
    print(f"Fused:                {fused_time:.3f} ms")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Time saved:           {saved_ms:.3f} ms ({saved_ms/orig_time*100:.1f}%)")

    return {
        "num_tokens": num_tokens,
        "original_ms": orig_time,
        "fused_ms": fused_time,
        "speedup": speedup,
    }


def benchmark_mhc_post_mean(num_tokens, hc_mult, hidden_size, num_iters=100):
    """Benchmark mhc_post + mean fusion."""
    print(f"\n{'='*80}")
    print(f"Benchmark: MHC Post + Mean Fusion")
    print(f"Config: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    print(f"{'='*80}")

    device = "cuda"
    x, residual, post_mix, comb_mix = create_test_inputs(num_tokens, hc_mult, hidden_size, device)

    # Benchmark original separate calls
    def original_ops():
        residual_out = mhc_post_tilelang(x, residual, post_mix, comb_mix)
        mean_out = residual_out.mean(dim=1)
        return residual_out, mean_out

    orig_time = benchmark_kernel(original_ops, [], num_iters=num_iters)

    # Benchmark fused kernel
    def fused_ops():
        return mhc_post_mean_fused(x, residual, post_mix, comb_mix)

    fused_time = benchmark_kernel(fused_ops, [], num_iters=num_iters)

    speedup = orig_time / fused_time
    saved_ms = orig_time - fused_time

    print(f"Original (separate):  {orig_time:.3f} ms")
    print(f"Fused:                {fused_time:.3f} ms")
    print(f"Speedup:              {speedup:.2f}x")
    print(f"Time saved:           {saved_ms:.3f} ms ({saved_ms/orig_time*100:.1f}%)")

    return {
        "num_tokens": num_tokens,
        "original_ms": orig_time,
        "fused_ms": fused_time,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MHC kernel fusions")
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024],
                        help="Token counts to benchmark")
    parser.add_argument("--hc-mult", type=int, default=4, help="HC multiplier")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden size")
    parser.add_argument("--num-iters", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    print(f"MHC Kernel Fusion Benchmarks")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"HC multiplier: {args.hc_mult}")

    all_results = []

    for num_tokens in args.num_tokens:
        result = benchmark_mhc_post_hc_head(
            num_tokens, args.hc_mult, args.hidden_size, args.num_iters
        )
        all_results.append(("mhc_post_hc_head", result))

        result = benchmark_mhc_post_hc_head_norm(
            num_tokens, args.hc_mult, args.hidden_size, args.num_iters
        )
        all_results.append(("mhc_post_hc_head_norm", result))

        result = benchmark_mhc_post_mean(
            num_tokens, args.hc_mult, args.hidden_size, args.num_iters
        )
        all_results.append(("mhc_post_mean", result))

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Kernel':<30} {'Tokens':<10} {'Speedup':<10} {'Time Saved (ms)'}")
    print(f"{'-'*80}")

    for kernel_name, result in all_results:
        saved_ms = result['original_ms'] - result['fused_ms']
        print(f"{kernel_name:<30} {result['num_tokens']:<10} "
              f"{result['speedup']:<10.2f} {saved_ms:.3f}")


if __name__ == "__main__":
    main()
