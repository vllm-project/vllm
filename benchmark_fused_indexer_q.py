# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the FP4 fused_indexer_q_rope_quant path."""

import argparse

import torch

from indexer_q_mxfp4 import fused_indexer_q_rope_quant as dev_impl
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
PROVIDERS = {"production": prod_impl, "dev": dev_impl}


def measure(num_tokens, provider):
    torch.set_default_device("cuda")
    positions = torch.randint(MAX_POS, (num_tokens,), dtype=torch.int64)
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=torch.float32)
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
    if provider != "production":
        prod_q, prod_weights = prod_impl(*kernel_args)
        dev_q, dev_weights = selected_impl(*kernel_args)
        prod_q_packed, prod_q_scale = prod_q
        dev_q_packed, dev_q_scale = dev_q

        assert torch.equal(prod_q_packed, dev_q_packed), (
            f"q packed mismatch for num_tokens={num_tokens}"
        )
        assert torch.equal(prod_q_scale, dev_q_scale), (
            f"q scale mismatch for num_tokens={num_tokens}"
        )
        assert torch.equal(prod_weights, dev_weights), (
            f"weights mismatch for num_tokens={num_tokens}"
        )

    benchmark_fn = lambda: selected_impl(*kernel_args)
    median_ms, p20_ms, p80_ms = triton.testing.do_bench(
        benchmark_fn,
        quantiles=QUANTILES,
    )

    bytes_per_token = 8  # position int64
    bytes_per_token += NUM_HEADS * HEAD_DIM * 2  # q in bf16
    bytes_per_token += ROPE_DIM * 4  # rope fp32
    bytes_per_token += NUM_HEADS * 2  # weights in bf16
    bytes_per_token += NUM_HEADS * HEAD_DIM // 2  # q out fp4
    bytes_per_token += NUM_HEADS * HEAD_DIM // 32  # q_scale uint8
    bytes_per_token += NUM_HEADS * 4  # weights out fp32
    total_bytes = bytes_per_token * num_tokens

    return median_ms, p20_ms, p80_ms, total_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=TOKENS)
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"H={NUM_HEADS} D={HEAD_DIM} rope_dim={ROPE_DIM} use_fp4=True\n")

    for num_tokens in args.tokens:
        for provider in PROVIDERS:
            median_ms, p20_ms, p80_ms, moved_bytes = measure(
                num_tokens,
                provider,
            )
            bandwidth_gb_s = moved_bytes / (median_ms * 1e-3) * 1e-9
            print(
                f"[{provider:10s}] T={num_tokens:6d}  "
                f"{median_ms * 1e3:7.2f} us  "
                f"BW {bandwidth_gb_s:7.1f} GB/s  "
                f"(p20={p20_ms * 1e3:.2f} p80={p80_ms * 1e3:.2f} us)"
            )
