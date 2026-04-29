# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.router.ll_router_gemm import (
    ll_router_gemm,
)
from vllm.triton_utils import triton

_HAS_DSV3 = hasattr(ops, "dsv3_router_gemm")

_providers = ["ll-router-bf16", "ll-router-fp8", "cublas-bf16"]
if _HAS_DSV3:
    _providers.append("dsv3-trtllm")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[1, 2, 4, 8, 16],
        x_log=False,
        line_arg="provider",
        line_vals=_providers,
        line_names=_providers,
        ylabel="Latency (us, lower is better)",
        plot_name="LL Router GEMM",
        args={},
    )
)
def benchmark(M, provider, N, K):
    device = "cuda"
    quantiles = [0.5, 0.2, 0.8]

    if provider == "ll-router-bf16":
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: ll_router_gemm(a, b), quantiles=quantiles
        )

    elif provider == "ll-router-fp8":
        a = torch.randn(M, K, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device=device).to(torch.float8_e4m3fn)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: ll_router_gemm(a, b), quantiles=quantiles
        )

    elif provider == "cublas-bf16":
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.mm(a, b.T, out=out), quantiles=quantiles
        )

    elif provider == "dsv3-trtllm":
        # DSV3 only supports N∈{256,384}, K=7168
        if N not in (256, 384) or K != 7168:
            return float("nan"), float("nan"), float("nan")
        from vllm import _custom_ops as ops

        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: ops.dsv3_router_gemm(a, b, torch.float32), quantiles=quantiles
        )

    # Return latency in us
    return ms * 1000, min_ms * 1000, max_ms * 1000


SHAPES = [
    (256, 7168, "DSV3 router"),
    (256, 2048, "Small K"),
    (128, 5120, "DeepSeek V2"),
    (8, 4096, "Mixtral-8x7B"),
    (64, 2880, "Non-aligned K"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    for N, K, desc in SHAPES:
        print(f"{desc}, N={N} K={K}:")
        save_dir = args.save_path or f"bench_ll_router_n{N}_k{K}"
        os.makedirs(save_dir, exist_ok=True)
        benchmark.run(
            print_data=True,
            show_plots=False,
            save_path=save_dir,
            N=N,
            K=K,
        )
        print()
