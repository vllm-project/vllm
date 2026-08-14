# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the RDNAHybridW4A16LinearKernel across decode and prefill shapes.

Usage:
    python benchmark_int4_gemm.py
    python benchmark_int4_gemm.py --models Qwen/Qwen3-4B
    python benchmark_int4_gemm.py --group-size 128
"""

import argparse
import copy
import itertools
import os

import torch

from vllm.triton_utils import triton

# ---------------------------------------------------------------------------
# Weight shapes: [K, N], TP_SPLIT_DIM
# ---------------------------------------------------------------------------
WEIGHT_SHAPES = {
    "Qwen/Qwen3-4B": [
        ([2560, 3840], 1),  # qkv_proj
        ([2560, 2560], 0),  # o_proj
        ([2560, 19456], 1),  # gate_up_proj
        ([9728, 2560], 0),  # down_proj
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        ([3584, 4608], 1),
        ([3584, 3584], 0),
        ([3584, 37888], 1),
        ([18944, 3584], 0),
    ],
    "trymirai/SmolLM2-1.7B-Instruct-AWQ": [
        ([2048, 6144], 1),  # qkv_proj
        ([2048, 2048], 0),  # o_proj
        ([2048, 16384], 1),  # gate_up_proj
        ([8192, 2048], 0),  # down_proj
    ],
    "RedHatAI/Qwen3-8B-quantized.w4a16": [
        ([4096, 6144], 1),  # qkv_proj
        ([4096, 4096], 0),  # o_proj
        ([4096, 24576], 1),  # gate_up_proj
        ([12288, 4096], 0),  # down_proj
    ],
}


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------
def prepare_hybrid_weights(K, N, group_size, device="cuda"):
    """Create random weights for benchmarking.

    Returns (w_q_skinny, w_s_skinny, w_fp16, w_zp). The triton path derives
    its int32 view from w_q_skinny, so no separate int32 buffer is returned.
    """
    num_groups = K // group_size

    # Random packed weights — actual values don't matter for throughput
    w_q_skinny_i32 = torch.randint(
        0, 2**31, (N, K // 8), dtype=torch.int32, device=device
    )
    w_q_skinny = w_q_skinny_i32.view(torch.int8).contiguous()
    w_s_skinny = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.01

    # Raw per-group zero-points for asymmetric benchmarks
    w_zp = torch.randint(0, 16, (N, num_groups), dtype=torch.int32, device=device).to(
        torch.float16
    )

    # FP16 baseline for F.linear
    w_fp16 = torch.randn(N, K, dtype=torch.float16, device=device) * 0.01

    return w_q_skinny, w_s_skinny, w_fp16, w_zp


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
PROVIDERS = ["torch-fp16", "hybrid-w4a16", "hybrid-w4a16-zp"]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        x_log=False,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="TFLOP/s (larger is better)",
        plot_name="FP16 vs Hybrid W4A16",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, group_size, weights):
    M = batch_size
    device = "cuda"
    dtype = torch.float16
    a = torch.randn((M, K), device=device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-fp16":
        w_fp16 = weights["w_fp16"]
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a, w_fp16),
            quantiles=quantiles,
        )
    elif provider in ("hybrid-w4a16", "hybrid-w4a16-zp"):
        from vllm.model_executor.kernels.linear.mixed_precision import (
            rdna_hybrid_w4a16 as _k,
        )

        _rdna_hybrid_w4a16_apply_impl = _k._rdna_hybrid_w4a16_apply_impl
        from vllm.utils.platform_utils import num_compute_units

        w = weights
        cu_count = num_compute_units()
        use_zp = provider == "hybrid-w4a16-zp"

        def run():
            return _rdna_hybrid_w4a16_apply_impl(
                a,
                w["w_q_skinny"],
                w["w_s_skinny"],
                w["w_zp"] if use_zp else None,
                None,  # bias
                cu_count,
                group_size,
            )

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            run,
            quantiles=quantiles,
        )
    else:
        return 0.0, 0.0, 0.0

    to_tflops = lambda t_ms: (2 * M * N * K) * 1e-12 / (t_ms * 1e-3)
    return to_tflops(ms), to_tflops(max_ms), to_tflops(min_ms)


def prepare_shapes(args):
    KN_model_names = []
    for model, tp_size in itertools.product(args.models, args.tp_sizes):
        for KN, tp_dim in copy.deepcopy(WEIGHT_SHAPES[model]):
            KN[tp_dim] //= tp_size
            KN.append(model)
            KN_model_names.append(KN)
    return KN_model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RDNAHybridW4A16LinearKernel"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["Qwen/Qwen3-4B"],
        choices=list(WEIGHT_SHAPES.keys()),
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    for K, N, model in prepare_shapes(args):
        group_size = args.group_size
        print(f"\n{'=' * 70}")
        print(f"{model}, N={N} K={K}, group_size={group_size}")
        print(f"{'=' * 70}")

        w_q_skinny, w_s_skinny, w_fp16, w_zp = prepare_hybrid_weights(K, N, group_size)

        weights = {
            "w_q_skinny": w_q_skinny,
            "w_s_skinny": w_s_skinny,
            "w_fp16": w_fp16,
            "w_zp": w_zp,
        }

        save_path = args.save_path or f"bench_int4_res_n{N}_k{K}"
        os.makedirs(save_path, exist_ok=True)
        benchmark.run(
            print_data=True,
            show_plots=False,
            save_path=save_path,
            N=N,
            K=K,
            group_size=group_size,
            weights=weights,
        )

    print("\nBenchmark finished!")
