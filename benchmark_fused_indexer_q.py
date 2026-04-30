# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)

NUM_HEADS = 64
HEAD_DIM = 128
ROPE_DIM = 64
MAX_POS = 100_000
TOKENS = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192]
ROPE_DTYPE = torch.float32


def make_inputs(num_tokens: int):
    positions = torch.randint(MAX_POS, (num_tokens,), dtype=torch.int64)
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=ROPE_DTYPE)
    weights = torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16)
    return (
        positions,
        query,
        cos_sin_cache,
        weights,
        HEAD_DIM**-0.5,
        NUM_HEADS**-0.5,
        True,
    )


def benchmark(num_tokens: int):
    torch.set_default_device("cuda")

    # run multiple times per measurement for more reliable results
    # separate sets of inputs to avoid L2 cache
    N = 10
    inputs_list = [make_inputs(num_tokens) for _ in range(N)]

    def f():
        for kernel_args in inputs_list:
            fused_indexer_q_rope_quant(*kernel_args)

    median_ms = triton.testing.do_bench(f) / N

    bytes_per_token = 8  # position int64
    bytes_per_token += NUM_HEADS * HEAD_DIM * 2  # q in bf16
    bytes_per_token += ROPE_DIM * torch.empty((), dtype=ROPE_DTYPE).element_size()
    bytes_per_token += NUM_HEADS * 2  # weights in bf16
    bytes_per_token += NUM_HEADS * HEAD_DIM // 2  # q out fp4
    bytes_per_token += NUM_HEADS * HEAD_DIM // 32  # q_scale uint8
    bytes_per_token += NUM_HEADS * 4  # weights out fp32
    total_bytes = bytes_per_token * num_tokens

    return median_ms, total_bytes


if __name__ == "__main__":
    for num_tokens in TOKENS:
        median_ms, moved_bytes = benchmark(num_tokens)
        bandwidth_gb_s = moved_bytes / (median_ms * 1e-3) * 1e-9
        print(
            f"T={num_tokens:6d}  "
            f"{median_ms * 1e3:7.2f} us  "
            f"BW {bandwidth_gb_s:7.1f} GB/s  "
        )
