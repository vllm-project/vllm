# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TritonW4A16LinearKernel's GEMM against a BF16 GEMM.

Usage:
    python benchmarks/kernels/benchmark_triton_w4a16_gemm.py
    python benchmarks/kernels/benchmark_triton_w4a16_gemm.py \
        --models meta-llama/Llama-3.1-70B-Instruct --tp-sizes 8
"""

import argparse
import copy
import itertools

import torch

from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)
from vllm.triton_utils import triton

# [K, N], TP split dim. Merged projections as vLLM builds them.
WEIGHT_SHAPES = {
    "Qwen/Qwen3.8-27B": [
        ([5120, 16384], 1),  # linear_attn.in_proj_qkvz
        ([6144, 5120], 0),  # linear_attn.out_proj / self_attn.o_proj
        ([5120, 14336], 1),  # self_attn.qkv_proj (gated q)
        ([5120, 34816], 1),  # mlp.gate_up_proj
        ([17408, 5120], 0),  # mlp.down_proj
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        ([4096, 6144], 1),
        ([4096, 4096], 0),
        ([4096, 28672], 1),
        ([14336, 4096], 0),
    ],
    "meta-llama/Llama-3.1-70B-Instruct": [
        ([8192, 10240], 1),
        ([8192, 8192], 0),
        ([8192, 57344], 1),
        ([28672, 8192], 0),
    ],
}

PROVIDERS = ["torch-bf16", "triton-w4a16"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 32, 64, 128, 512, 2048],
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="us (lower is better)",
        plot_name="BF16 vs Triton W4A16",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, group_size, weights):
    """``do_bench`` flushes the cache between reps, so weights come from HBM."""
    a = torch.randn((batch_size, K), device="cuda", dtype=torch.bfloat16)
    if provider == "torch-bf16":
        fn = lambda: torch.nn.functional.linear(a, weights["w_bf16"])
    else:
        fn = lambda: triton_w4a16_gemm(
            a, weights["b_q"], weights["scales"], None, group_size
        )
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms * 1e3, max_ms * 1e3, min_ms * 1e3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3.8-27B"],
        choices=list(WEIGHT_SHAPES),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            K, N = KN
            print(f"\n{model} TP={tp_size}: K={K} N={N} g={args.group_size}")
            weights = {
                "b_q": torch.randint(
                    -(2**31), 2**31 - 1, (K, N // 8), dtype=torch.int32, device="cuda"
                ),
                "scales": torch.rand(
                    (K // args.group_size, N), dtype=torch.bfloat16, device="cuda"
                )
                * 0.01,
                "w_bf16": torch.randn((N, K), dtype=torch.bfloat16, device="cuda"),
            }
            benchmark.run(
                print_data=True, N=N, K=K, group_size=args.group_size, weights=weights
            )
