# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the FP4 fused_indexer_q_rope_quant path."""

import argparse

import torch

from indexer_q_mxfp4 import fused_indexer_q_rope_quant as cuda_cpp_impl
from indexer_q_mxfp4_cutedsl import fused_indexer_q_rope_quant as cutedsl_impl
from vllm.triton_utils import triton
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    fused_indexer_q_rope_quant as prod_impl,
)

NUM_HEADS = 64
HEAD_DIM = 128
ROPE_DIM = 64
MAX_POS = 100_000
TOKENS = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192]
QUANTILES = [0.5, 0.2, 0.8]
PROVIDERS = {
    "cuda_cpp": cuda_cpp_impl,
    "cutedsl": cutedsl_impl,
    "prod": prod_impl,
}


def measure(num_tokens, provider, check_provider, cache_dtype, skip_check):
    torch.set_default_device("cuda")
    positions = torch.randint(MAX_POS, (num_tokens,), dtype=torch.int64)
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=cache_dtype)
    weights = torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16)
    kernel_args = (
        positions,
        query,
        cos_sin_cache,
        weights,
        HEAD_DIM**-0.5,
        NUM_HEADS**-0.5,
        True,
    )

    selected_impl = PROVIDERS[provider]
    selected_q, selected_weights = selected_impl(*kernel_args)
    torch.accelerator.synchronize()

    if not skip_check and provider != check_provider:
        ref_q, ref_weights = PROVIDERS[check_provider](*kernel_args)
        ref_q_packed, ref_q_scale = ref_q
        selected_q_packed, selected_q_scale = selected_q

        assert torch.equal(ref_q_packed, selected_q_packed), (
            f"q packed mismatch for provider={provider} "
            f"num_tokens={num_tokens} cache_dtype={cache_dtype}"
        )
        assert torch.equal(ref_q_scale, selected_q_scale), (
            f"q scale mismatch for provider={provider} "
            f"num_tokens={num_tokens} cache_dtype={cache_dtype}"
        )
        assert torch.equal(ref_weights, selected_weights), (
            f"weights mismatch for provider={provider} "
            f"num_tokens={num_tokens} cache_dtype={cache_dtype}"
        )
        torch.accelerator.synchronize()

    benchmark_fn = lambda: selected_impl(*kernel_args)
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        benchmark_fn,
        quantiles=QUANTILES,
    )

    bytes_per_token = 8  # position int64
    bytes_per_token += NUM_HEADS * HEAD_DIM * 2  # q in bf16
    bytes_per_token += ROPE_DIM * torch.empty((), dtype=cache_dtype).element_size()
    bytes_per_token += NUM_HEADS * 2  # weights in bf16
    bytes_per_token += NUM_HEADS * HEAD_DIM // 2  # q out fp4
    bytes_per_token += NUM_HEADS * HEAD_DIM // 32  # q_scale uint8
    bytes_per_token += NUM_HEADS * 4  # weights out fp32
    total_bytes = bytes_per_token * num_tokens

    return median_ms, p20_ms, p80_ms, total_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    parser.add_argument(
        "--providers",
        choices=PROVIDERS,
        nargs="+",
        default=PROVIDERS.keys(),
    )
    parser.add_argument("--check-provider", choices=PROVIDERS, default="prod")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--cache-dtype",
        choices=["float32", "bfloat16"],
        default="float32",
    )
    args = parser.parse_args()

    cache_dtype = getattr(torch, args.cache_dtype)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(
        f"H={NUM_HEADS} D={HEAD_DIM} rope_dim={ROPE_DIM} "
        f"use_fp4=True cache_dtype={args.cache_dtype}\n"
    )

    for num_tokens in args.tokens:
        for provider in args.providers:
            median_ms, p20_ms, p80_ms, moved_bytes = measure(
                num_tokens,
                provider,
                args.check_provider,
                cache_dtype,
                args.skip_check,
            )
            bandwidth_gb_s = moved_bytes / (median_ms * 1e-3) * 1e-9
            print(
                f"[{provider:8s}] T={num_tokens:6d}  "
                f"{median_ms * 1e3:7.2f} us  "
                f"BW {bandwidth_gb_s:7.1f} GB/s  "
                f"(p20={p20_ms * 1e3:.2f} p80={p80_ms * 1e3:.2f} us)"
            )
