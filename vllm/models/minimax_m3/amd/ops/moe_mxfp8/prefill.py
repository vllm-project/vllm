# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefill chain (fp8 x fp8, ``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``):
``moe_flydsl_common`` sort + tile map, aiter's fused per-token fp8 quant,
``gemm1_prefill`` (gate/up + swiglu-OAI + MXFP8 quant of the intermediate),
``gemm2_prefill`` (down GEMM writing ``[M, topk, H]`` partials) and the top-k
reduction (``reduce_bf16``, or ``reduce_fp8`` when ``AITER_FLYDSL_STAGE2_FP8=1``).
"""

from __future__ import annotations

import functools
import os

import torch

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import (
    _get_reduce_bf16,
    _get_sort,
    _get_tile_map,
    _run_compiled,
    _u8_flat,
)

MIN_PREFILL_TOKENS = 3072
MAX_PREFILL_TOKENS = 65536
BM256_FROM_TOKENS = 16384  # sort / gemm1 block of 256 rows from here up
GEMM1_SWIGLU_ALPHA = 1.702
GEMM1_SWIGLU_LIMIT = 7.0
# CTAs sharing one m-block's 24 n-tiles in gemm2 (gemm2_prefill.compile_moe_gemm2)
GEMM2_N_SPLIT = 6
GEMM2_N_SPLIT_LARGE = 4
GEMM2_N_SPLIT_LARGE_FROM_TOKENS = 32768


def block_m_for(n_tokens: int) -> int:
    return 256 if n_tokens >= BM256_FROM_TOKENS else 128


def gemm2_n_split_for(n_tokens: int) -> int:
    return (
        GEMM2_N_SPLIT_LARGE
        if n_tokens >= GEMM2_N_SPLIT_LARGE_FROM_TOKENS
        else GEMM2_N_SPLIT
    )


def default_out_mode() -> str:
    """gemm2 output mode (see ``gemm2.compile_moe_gemm2``): "bf16" = token-major
    bf16 partials + ``moe_flydsl_common.reduce_bf16`` (deterministic); "fp8" = MXFP8
    partials + ``reduce_fp8`` (deterministic, faster, one more quantization).
    Follows aiter's fp8 route-out switch, ``AITER_FLYDSL_STAGE2_FP8=1``
    (read per call, like aiter); a caller may also pass ``out_mode`` explicitly."""
    return "fp8" if os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") == "1" else "bf16"


_GEMM1_BLOCK_K = 128
_GEMM2_INTERMEDIATE = 768


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 unrolls the K loop by 4 steps of 128 after 4 peeled ones; gemm2's
    pipeline is written for K = 768."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % 256 == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


@functools.cache
def _get_gemm1(
    hidden_size: int, intermediate_size: int, num_experts: int, block_m: int
):
    from .gemm1_prefill import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=block_m
    )


@functools.cache
def _get_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    block_m: int,
    n_split: int,
    out_mode: str,
):
    from .gemm2_prefill import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=n_split,
        sort_block_m=block_m,
        out_mode=out_mode,
    )


@functools.cache
def _get_reduce_fp8(hidden_size: int, topk: int):
    from .reduce_fp8 import compile_moe_reduce_fp8

    return compile_moe_reduce_fp8(H=hidden_size, topk=topk)


def a8w8_prefill_moe(
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
    out_mode: str | None = None,
) -> torch.Tensor:
    """One MoE layer for ``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``:
    stage 1 (sort, aiter fp8 quant, tile map, gemm1), then gemm2 + the reduction
    of ``out_mode`` (default ``default_out_mode()``): "bf16" partials +
    ``moe_flydsl_common.reduce_bf16``, "fp8" partials + ``reduce_fp8``. Returns
    ``[M, hidden_size]`` bf16."""
    from .gemm2_prefill import OUT_MODES, gemm2_grid

    out_mode = default_out_mode() if out_mode is None else out_mode
    assert out_mode in OUT_MODES, out_mode
    n_tokens = x.shape[0]
    topk = topk_ids.shape[1]
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    bufs, _a_q, _a_s, h_q, h_s, num_m_blocks = a8w8_prefill_stage1(
        x,
        w13,
        w13_scale,
        topk_weights,
        topk_ids,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )
    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    if out_mode == "fp8":
        gemm2_out = torch.empty(
            (n_tokens * topk, hidden_size), dtype=torch.uint8, device=device
        )
        partial_scale = torch.empty(
            (n_tokens * topk * (hidden_size // 32),), dtype=torch.uint8, device=device
        )
    else:
        gemm2_out = torch.empty(
            (n_tokens * topk, hidden_size), dtype=torch.bfloat16, device=device
        )
        partial_scale = torch.empty((16,), dtype=torch.uint8, device=device)
    num_m_blocks2 = (num_m_blocks * bm) // 128
    n_split = gemm2_n_split_for(n_tokens)
    grid2 = gemm2_grid(num_m_blocks2, n_split)
    # the bf16 partials are 4.03 GB at 65536 tokens: pass the element view (a byte
    # view exceeds 2^31 elements); the kernels' i32 byte offsets are read as u32 by
    # the buffer instructions, so 4 GB is the limit
    _run_compiled(
        _get_gemm2(
            hidden_size, intermediate_size, num_experts, topk, bm, n_split, out_mode
        ),
        h_q.view(-1),
        _u8_flat(w2),
        gemm2_out.view(-1),
        h_s,
        _u8_flat(w2_scale),
        partial_scale,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        grid2,
        stream,
    )
    if out_mode == "fp8":
        _run_compiled(
            _get_reduce_fp8(hidden_size, topk),
            gemm2_out.view(-1),
            partial_scale,
            topk_weights.to(torch.float32).contiguous().view(-1),
            out.view(-1),
            n_tokens,
            stream,
        )
        return out
    _run_compiled(
        _get_reduce_bf16(hidden_size, topk),
        gemm2_out.view(-1),
        out.view(-1),
        n_tokens,
        stream,
    )
    return out


def a8w8_prefill_stage1(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
):
    """Sort + aiter's fp8 quant + tile map + gemm1. Returns ``(bufs, a_q, a_s,
    h_q, h_s, num_m_blocks)``: ``h_q`` fp8 ``[rows, I]`` and its e8m0 scales in
    sorted-row order (the layout gemm2 reads)."""
    from aiter import dtypes
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import SortBuffers
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import tile_map_grid

    n_tokens, hidden = x.shape
    assert MIN_PREFILL_TOKENS <= n_tokens <= MAX_PREFILL_TOKENS, n_tokens
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
    grid1 = tile_map_grid(num_m_blocks, inter)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
    _run_compiled(
        _get_tile_map(inter, bm),
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        tile_map,
        grid1,
        stream,
    )
    h_q = torch.empty((rows, inter), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    _run_compiled(
        _get_gemm1(hidden_size, inter, num_experts, bm),
        _u8_flat(a_q),
        _u8_flat(w13),
        h_q.view(-1),
        _u8_flat(a_s),
        _u8_flat(w13_scale),
        h_s,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        n_tokens,
        num_m_blocks,
        int(a_s.numel() * a_s.element_size()),
        tile_map,
        grid1,
        stream,
    )
    return bufs, a_q, a_s, h_q, h_s, num_m_blocks
