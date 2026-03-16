#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: NVFP4 Marlin latency/throughput with and without the FP16 overflow fix.

Tests the two distinct patches:
  Bug 1: input_global_scale forwarding (Python-only, already active)
  Bug 2: epilogue scale-before-cast + SM75 FP32 accum (kernel, requires rebuild)

Usage:
    python benchmarks/benchmark_nvfp4_marlin.py
    python benchmarks/benchmark_nvfp4_marlin.py --sizes large   # LLM-scale shapes
    python benchmarks/benchmark_nvfp4_marlin.py --iters 200     # more iterations
"""
import argparse
import time
from dataclasses import dataclass

import torch

from tests.quantization.utils import is_quant_method_supported
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    rand_marlin_weight_nvfp4_like,
)

WARMUP = 30
ITERS = 100


@dataclass
class BenchResult:
    label: str
    shape: str
    ms_mean: float
    ms_std: float
    tflops: float

    def __str__(self) -> str:
        return (
            f"{self.label:50s}  shape={self.shape:25s}  "
            f"{self.ms_mean:.4f} ± {self.ms_std:.4f} ms  "
            f"{self.tflops:.2f} TFLOP/s"
        )


def bench(fn, iters: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = torch.tensor(times)
    return float(t.mean()), float(t.std())


def setup_layer(m: int, k: int, n: int, dtype: torch.dtype):
    torch.manual_seed(0)
    dense_weight = torch.randn((k, n), device="cuda", dtype=dtype)
    weight_ref, marlin_qw, marlin_scales, marlin_gs = rand_marlin_weight_nvfp4_like(
        dense_weight.T, group_size=16
    )
    workspace = marlin_make_workspace_new(torch.device("cuda"))
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    gs = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    return x, marlin_qw, marlin_scales, marlin_gs, workspace, gs, n, k


def run_benchmark(
    label: str,
    m: int,
    k: int,
    n: int,
    dtype: torch.dtype,
    input_global_scale,
    iters: int,
) -> BenchResult:
    x, w, ws, wgs, workspace, gs, size_n, size_k = setup_layer(m, k, n, dtype)
    scale_arg = gs if input_global_scale else None

    def fn():
        return apply_fp4_marlin_linear(
            input=x,
            weight=w,
            weight_scale=ws,
            weight_global_scale=wgs,
            workspace=workspace,
            size_n=size_n,
            size_k=size_k,
            input_global_scale=scale_arg,
            bias=None,
        )

    ms_mean, ms_std = bench(fn, iters, WARMUP)
    # 2 * m * k * n FLOPs for a matmul
    tflops = (2 * m * k * n) / (ms_mean * 1e-3) / 1e12
    shape = f"({m},{k},{n})"
    return BenchResult(label, shape, ms_mean, ms_std, tflops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        choices=["small", "large", "all"],
        default="all",
        help="Shape set to benchmark",
    )
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if not is_quant_method_supported("gptq_marlin"):
        raise RuntimeError("Marlin not supported on this GPU (need SM >= 7.5)")

    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"\nDevice : {dev}  (SM {cap[0]}{cap[1]})")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"dtype  : {dtype}")
    print(f"iters  : {args.iters}\n")

    # Representative shapes:
    # small = decode (small m), large = prefill / LLM linear layers
    small_shapes = [
        (1,   4096, 4096),
        (4,   4096, 4096),
        (16,  4096, 4096),
        (32,  4096, 4096),
    ]
    large_shapes = [
        (64,  4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (64,  4096, 14336),   # Qwen3 MLP gate
        (64,  14336, 4096),   # Qwen3 MLP down
    ]

    if args.sizes == "small":
        shapes = small_shapes
    elif args.sizes == "large":
        shapes = large_shapes
    else:
        shapes = small_shapes + large_shapes

    results = []
    for m, k, n in shapes:
        # Bug 1: no input_global_scale (old, broken)
        r_no_scale = run_benchmark(
            "NVFP4-Marlin  no-input-scale  (OLD/broken)",
            m, k, n, dtype, input_global_scale=False, iters=args.iters,
        )
        # Bug 1 fix: with input_global_scale (correct, Python multiply)
        r_with_scale = run_benchmark(
            "NVFP4-Marlin  with-input-scale (FIXED)",
            m, k, n, dtype, input_global_scale=True, iters=args.iters,
        )
        results.append((r_no_scale, r_with_scale))

    print(f"{'Label':50s}  {'Shape':25s}  {'Mean ms':>13s}  {'TFLOP/s':>10s}")
    print("-" * 112)
    for no_s, with_s in results:
        print(no_s)
        print(with_s)
        overhead_pct = (with_s.ms_mean - no_s.ms_mean) / no_s.ms_mean * 100
        print(f"  → overhead of input-scale multiply: {overhead_pct:+.2f}%\n")

    print("\nNote: Bug 2 (kernel epilogue ordering / SM75 FP16 accum) requires")
    print("      recompiling from source (VLLM_USE_PRECOMPILED=0 pip install -e .)")
    print("      to measure. The kernel change adds 2 FP32 muls per output element")
    print("      in the epilogue — expected < 1% overhead vs full GEMM compute.")


if __name__ == "__main__":
    main()
