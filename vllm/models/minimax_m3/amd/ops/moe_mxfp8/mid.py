# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-batch chain (fp8 x fp8, ``MIN_MID_TOKENS <= M <= MAX_MID_TOKENS``):
``moe_flydsl_common.sort`` in ``block_m_for(M)``-row blocks, aiter's fused per-token
fp8 quant, ``gemm1_mid`` (gate/up + swiglu-OAI + MXFP8 quant, zeroes the output),
``gemm2_mid`` (down GEMM, routing-weighted bf16 atomic accumulation). No tile map,
no partials, no reduction.
"""

from __future__ import annotations

import functools

import torch

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import (
    _get_sort,
    _run_compiled,
    _u8_flat,
)

MIN_MID_TOKENS = 257
MAX_MID_TOKENS = 3071
# sort block by batch size (a routed expert holds 16..64 rows here: one block each)
BM64_FROM_TOKENS = 768
BM128_FROM_TOKENS = 1536
GEMM2_MAX_TILE_M = 128  # gemm2 row tile = the sort block


def block_m_for(n_tokens: int) -> int:
    if n_tokens >= BM128_FROM_TOKENS:
        return 128
    if n_tokens >= BM64_FROM_TOKENS:
        return 64
    return 32


@functools.cache
def _get_gemm1(
    hidden_size: int, intermediate_size: int, num_experts: int, block_m: int
):
    from .gemm1_mid import compile_moe_gemm1_mid

    return compile_moe_gemm1_mid(
        H=hidden_size, I=intermediate_size, E=num_experts, BM=block_m
    )


@functools.cache
def _get_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    tile_m: int,
    block_m: int,
):
    from .gemm2_mid import compile_moe_gemm2_mid

    return compile_moe_gemm2_mid(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        BM=tile_m,
        sort_block_m=block_m,
    )


def a8w8_mid_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One MoE layer for ``MIN_MID_TOKENS <= M <= MAX_MID_TOKENS``, returned as
    ``[M, hidden_size]`` bf16 (``out`` when given: contiguous, zeroed and
    accumulated into here)."""
    from aiter import dtypes
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import SortBuffers

    n_tokens, hidden = x.shape
    assert hidden == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    inter = intermediate_size

    bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, device)
    _get_sort(num_experts, topk, bm)(
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)
    )
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(
        x,
        bufs.sorted_ids,
        bufs.num_valid_ids,
        token_num=n_tokens,
        topk=topk,
        block_size=bm,
        quant_dtype=dtypes.fp8,
    )
    num_m_blocks = bufs.max_sorted // bm
    rows = num_m_blocks * bm
    h_q = torch.empty((rows, inter), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    assert out.shape == (n_tokens, hidden_size) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    gemm1 = _get_gemm1(hidden_size, inter, num_experts, bm)
    _run_compiled(
        gemm1,
        _u8_flat(a_q),
        _u8_flat(w13),
        h_q.view(-1),
        _u8_flat(a_s),
        _u8_flat(w13_scale),
        h_s,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        out.view(-1),
        n_tokens,
        num_m_blocks,
        int(a_s.numel() * a_s.element_size()),
        num_m_blocks * gemm1.n_tiles,
        stream,
    )
    gemm2 = _get_gemm2(hidden_size, inter, num_experts, min(bm, GEMM2_MAX_TILE_M), bm)
    num_tiles = num_m_blocks * gemm2.tiles_per_block
    _run_compiled(
        gemm2,
        h_q.view(-1),
        _u8_flat(w2),
        out.data_ptr(),
        h_s,
        _u8_flat(w2_scale),
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_tiles,
        num_tiles * gemm2.n_tiles,
        stream,
    )
    return out


__all__ = [
    "MAX_MID_TOKENS",
    "MIN_MID_TOKENS",
    "a8w8_mid_moe",
    "block_m_for",
]
