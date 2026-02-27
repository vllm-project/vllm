#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark comparing old vs new default fused MoE configs.

Runs the triton fused_moe kernel with three configurations for each scenario:
  1. Tuned config (from JSON file, if available) — the target to match
  2. Old default (the hardcoded defaults before this change)
  3. New default (the improved defaults)

Usage:
    python benchmarks/kernels/benchmark_moe_defaults.py

Produces a table showing kernel time (us) and speedup of new vs old defaults.
"""

import torch

from vllm.model_executor.layers.fused_moe import fused_topk, override_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    get_default_config,
    get_moe_configs,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

FP8_DTYPE = current_platform.fp8_dtype()


def old_default_config(M, E, N, K, topk, dtype=None, block_shape=None):
    """The original defaults before https://github.com/vllm-project/vllm/pull/34846,
    for comparison."""
    if dtype == "fp8_w8a8" and block_shape is not None:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "SPLIT_K": 1,
            "num_warps": 4,
            "num_stages": 3 if not current_platform.is_rocm() else 2,
        }
    elif M <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        }
    else:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1,
        }


def benchmark_config(
    config,
    M,
    E,
    N,
    K,
    topk,
    dtype,
    use_fp8=False,
    block_shape=None,
    num_iters=100,
):
    """Time a single kernel config. Returns kernel time in microseconds."""
    init_dtype = torch.float16 if use_fp8 else dtype

    a = torch.randn(M, K, device="cuda", dtype=init_dtype) / 10
    w1 = torch.randn(E, 2 * N, K, device="cuda", dtype=init_dtype) / 10
    w2 = torch.randn(E, K, N, device="cuda", dtype=init_dtype) / 10

    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    if use_fp8:
        if block_shape is not None:
            bsn, bsk = block_shape
            n_tiles_w1 = triton.cdiv(2 * N, bsn)
            k_tiles_w1 = triton.cdiv(K, bsk)
            n_tiles_w2 = triton.cdiv(K, bsn)
            k_tiles_w2 = triton.cdiv(N, bsk)
            w1_scale = torch.rand(
                E, n_tiles_w1, k_tiles_w1, device="cuda", dtype=torch.float32
            )
            w2_scale = torch.rand(
                E, n_tiles_w2, k_tiles_w2, device="cuda", dtype=torch.float32
            )
        else:
            w1_scale = torch.rand(E, device="cuda", dtype=torch.float32)
            w2_scale = torch.rand(E, device="cuda", dtype=torch.float32)
        a1_scale = torch.rand(1, device="cuda", dtype=torch.float32)
        a2_scale = torch.rand(1, device="cuda", dtype=torch.float32)
        # Only weights are stored in fp8; activations stay in bf16/fp16
        # and get dynamically quantized inside the kernel.
        w1 = w1.to(FP8_DTYPE)
        w2 = w2.to(FP8_DTYPE)

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn if use_fp8 else None,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
    )

    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)

    # Warmup
    for _ in range(20):
        with override_config(config):
            topk_weights, topk_ids, _ = fused_topk(a, gating, topk, renormalize=True)
            fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        with override_config(config):
            topk_weights, topk_ids, _ = fused_topk(a, gating, topk, renormalize=True)
            fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iters * 1000  # ms -> us


# Model configurations: (name, E, N, K, topk, dtype_str, use_fp8, block_shape)
# N = moe_intermediate_size // tp_size (the value used in config file lookup)
MODELS = [
    # --- Few experts ---
    ("Mixtral bf16", 8, 7168, 4096, 2, None, False, None),
    ("Mixtral fp8", 8, 7168, 4096, 2, "fp8_w8a8", True, None),
    # --- Many experts: real model shapes at tp=1 ---
    # Qwen2-MoE-57B: E=60, topk=4, N=1408, K=2048
    ("Qwen2-MoE bf16", 60, 1408, 2048, 4, None, False, None),
    # DeepSeek-V2: E=64, topk=6, N=1407, K=4096
    # (use 1408 to avoid odd alignment; real model is 1407)
    ("DeepSeek-V2 bf16", 64, 1408, 4096, 6, None, False, None),
    # OLMoE-7B: E=64, topk=8, N=2048, K=2048
    ("OLMoE bf16", 64, 2048, 2048, 8, None, False, None),
    # GLM-4-100B-A10B: E=128, topk=8, N=1408, K=4096
    ("GLM-4-MoE bf16", 128, 1408, 4096, 8, None, False, None),
    # Qwen3-30B-A3B: E=128, topk=8, N=768, K=2048
    ("Qwen3-MoE bf16", 128, 768, 2048, 8, None, False, None),
    # DeepSeek-V3 / MiMo-V2-Flash: E=256, topk=8, N=2048, K=7168
    ("DeepSeek-V3 bf16", 256, 2048, 7168, 8, None, False, None),
    # Qwen3.5-70B-A22B (Qwen3-Next): E=512, topk=10, N=512, K=2048
    ("Qwen3-Next bf16", 512, 512, 2048, 10, None, False, None),
    # E=128 N=1856 bf16
    ("E128 N1856 bf16", 128, 1856, 4096, 8, None, False, None),
    # E=256 N=512 bf16 (DS-V3 tp=4)
    ("DS-V3 tp4 bf16", 256, 512, 7168, 8, None, False, None),
    # E=512 N=512 bf16 (Qwen3-Next tp=1)
    ("Qwen3-Next bf16", 512, 512, 2048, 10, None, False, None),
    # E=512 N=256 bf16 (Qwen3-Next tp=2)
    ("Qwen3-Next tp2", 512, 256, 2048, 10, None, False, None),
    # --- FP8 block quant (many experts) ---
    # DS-V3 tp=4: E=256, N=512, fp8 block
    ("DS-V3 tp4 fp8blk", 256, 512, 7168, 8, "fp8_w8a8", True, [128, 128]),
    # DS-V3 tp=8: E=256, N=256, fp8 block
    ("DS-V3 tp8 fp8blk", 256, 256, 7168, 8, "fp8_w8a8", True, [128, 128]),
    # Qwen3-Next tp=2 fp8 block
    ("Qwen3-Next tp2 fp8blk", 512, 256, 2048, 10, "fp8_w8a8", True, [128, 128]),
]

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def main():
    set_random_seed(0)
    torch.set_default_device("cuda")
    dtype = torch.bfloat16

    for name, E, N, K, topk, dtype_str, use_fp8, block_shape in MODELS:
        print(f"\n{'=' * 90}")
        print(f"  {name}  (E={E}, N={N}, K={K}, topk={topk})")
        print(f"{'=' * 90}")

        # Try to load tuned config
        block_n = block_shape[0] if block_shape else None
        block_k = block_shape[1] if block_shape else None
        tuned = get_moe_configs(E, N, dtype_str, block_n, block_k)
        has_tuned = tuned is not None
        print(f"  Tuned config available: {has_tuned}")

        hdr = (
            f"{'Batch':>6} | {'Tuned (us)':>11} | {'Old (us)':>11} | "
            f"{'New (us)':>11} | {'New/Old':>8} | {'New/Tuned':>10}"
        )
        print(f"  {hdr}")
        print(f"  {'-' * len(hdr)}")

        for M in BATCH_SIZES:
            old_cfg = old_default_config(M, E, N, K, topk, dtype_str, block_shape)
            new_cfg = get_default_config(M, E, N, K, topk, dtype_str, block_shape)

            if has_tuned:
                tuned_cfg = tuned[min(tuned.keys(), key=lambda x: abs(x - M))]
                t_tuned = benchmark_config(
                    tuned_cfg,
                    M,
                    E,
                    N,
                    K,
                    topk,
                    dtype,
                    use_fp8=use_fp8,
                    block_shape=block_shape,
                )
            else:
                t_tuned = None

            t_old = benchmark_config(
                old_cfg,
                M,
                E,
                N,
                K,
                topk,
                dtype,
                use_fp8=use_fp8,
                block_shape=block_shape,
            )
            t_new = benchmark_config(
                new_cfg,
                M,
                E,
                N,
                K,
                topk,
                dtype,
                use_fp8=use_fp8,
                block_shape=block_shape,
            )

            ratio_new_old = t_new / t_old
            tuned_str = f"{t_tuned:11.2f}" if t_tuned else f"{'N/A':>11}"
            ratio_tuned = f"{t_new / t_tuned:10.2f}x" if t_tuned else f"{'N/A':>10}"
            # flag regressions where new default is >5% slower than old
            marker = " <--" if ratio_new_old > 1.05 else ""

            print(
                f"  {M:>6} | {tuned_str} | {t_old:11.2f} | {t_new:11.2f} "
                f"| {ratio_new_old:7.2f}x | {ratio_tuned}{marker}"
            )


if __name__ == "__main__":
    main()
