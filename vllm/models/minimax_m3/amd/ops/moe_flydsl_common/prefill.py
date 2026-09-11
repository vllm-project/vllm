# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE prefill helpers."""

from __future__ import annotations

import functools

MIN_PREFILL_TOKENS = 3072


MAX_PREFILL_TOKENS = 65536


BM256_FROM_TOKENS = 16384  # sort / gemm1 block of 256 rows from here up


def block_m_for(n_tokens: int) -> int:
    return 256 if n_tokens >= BM256_FROM_TOKENS else 128


@functools.cache
def _get_sort(num_experts: int, topk: int, block_m: int):
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import compile_moe_sort

    return compile_moe_sort(E=num_experts, topk=topk, block_m=block_m)


@functools.cache
def _get_tile_map(intermediate_size: int, block_m: int):
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import (
        compile_tile_map,
    )

    return compile_tile_map(I=intermediate_size, BM=block_m)


@functools.cache
def _get_reduce_bf16(hidden_size: int, topk: int):
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.reduce_bf16 import (
        compile_moe_reduce_bf16,
    )

    return compile_moe_reduce_bf16(H=hidden_size, topk=topk)
