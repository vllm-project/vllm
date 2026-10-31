# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL MXFP8 MoE prefill chain on AITER's MXFP8 weights: sort, gemm1 (gate/up +
SwiGLU-OAI + MXFP8 quant), gemm2 (routing-weighted partials), fixed-order reduce."""

import functools

import torch

from vllm.models.minimax_m3.amd.ops.flydsl_utils import run_compiled
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import (
    _get_reduce_bf16,
    _get_sort,
    _get_tile_map,
    _u8_flat,
)

MIN_PREFILL_TOKENS = 3072
MAX_PREFILL_TOKENS = 65536
BM256_FROM_TOKENS = 16384

GEMM1_SWIGLU_ALPHA = 1.702
GEMM1_SWIGLU_LIMIT = 7.0
# CTAs sharing one m-block's n-tiles in gemm2
GEMM2_N_SPLIT = 6
GEMM2_N_SPLIT_LARGE = 4
GEMM2_N_SPLIT_LARGE_FROM_TOKENS = 32768
GEMM2_BM256_MIN_INTERMEDIATE = 768
GEMM2_BM256_FROM_TOKENS = 16384

_GEMM1_BLOCK_K = 128
_GEMM2_INTERMEDIATES = (768, 1536, 3072)


def block_m_for(
    n_tokens: int, hidden_size: int, intermediate_size: int, num_experts: int, topk: int
) -> int:
    if (hidden_size, intermediate_size, num_experts, topk) == (6144, 3072, 128, 4):
        threshold = 8192
    elif (hidden_size, intermediate_size) == (6144, 768):
        threshold = 4096
    else:
        threshold = BM256_FROM_TOKENS
    return 256 if n_tokens >= threshold else 128


def gemm2_n_split_for(n_tokens: int) -> int:
    if n_tokens >= GEMM2_N_SPLIT_LARGE_FROM_TOKENS:
        return GEMM2_N_SPLIT_LARGE
    return GEMM2_N_SPLIT


def gemm2_block_m_for(sort_block_m: int, n_tokens: int, intermediate_size: int) -> int:
    if (
        sort_block_m == 256
        and intermediate_size >= GEMM2_BM256_MIN_INTERMEDIATE
        and n_tokens >= GEMM2_BM256_FROM_TOKENS
    ):
        return 256
    return 128


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 peels 4 K-steps of 128 and unrolls the rest by 4."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % 256 == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size in _GEMM2_INTERMEDIATES
    )


@functools.cache
def _get_gemm1(hidden_size: int, intermediate_size: int, num_experts: int, bm: int):
    from .gemm1_prefill import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=bm
    )


@functools.cache
def _get_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    bm: int,
    n_split: int,
    gemm2_bm: int,
):
    if gemm2_bm == 256:
        from .gemm2_prefill_bm256 import compile_moe_gemm2_bm256 as compile_moe_gemm2
    else:
        from .gemm2_prefill import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=n_split,
        sort_block_m=bm,
    )


def a8w8_prefill_moe(
    x_quant: tuple[torch.Tensor, torch.Tensor],
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
    """``x_quant`` = (fp8 ``[M, H]``, e8m0 ``[M, H/32]``) -> ``[M, H]`` bf16, on
    ``AiterMxfp8Experts``' weights."""
    from aiter import dtypes
    from aiter.ops.quant import mxfp4_moe_sort_hip

    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import SortBuffers
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import (
        tile_map_grid,
    )

    from .gemm2_prefill import gemm2_grid

    a_q, x_s = x_quant
    n_tokens, hidden = a_q.shape
    assert MIN_PREFILL_TOKENS <= n_tokens <= MAX_PREFILL_TOKENS, n_tokens
    assert hidden == hidden_size and a_q.is_contiguous() and x_s.is_contiguous()
    assert x_s.shape == (n_tokens, hidden // 32), x_s.shape
    inter = intermediate_size
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = topk_ids.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens, hidden_size, inter, num_experts, topk)

    bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, device)
    _get_sort(num_experts, topk, bm)(
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)
    )
    a_s = torch.empty(
        ((bufs.sorted_ids.shape[0] + 31) // 32 * 32, (hidden // 32 + 7) // 8 * 8),
        dtype=dtypes.fp8_e8m0,
        device=device,
    )
    mxfp4_moe_sort_hip(
        a_s,
        x_s.view(dtypes.fp8_e8m0),
        bufs.sorted_ids,
        bufs.num_valid_ids,
        n_tokens,
        hidden,
    )
    num_m_blocks = bufs.max_sorted // bm
    rows = num_m_blocks * bm
    grid1 = tile_map_grid(num_m_blocks, inter)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
    run_compiled(
        _get_tile_map(inter, bm),
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        tile_map,
        grid1,
        stream,
    )
    h_q = torch.empty((rows, inter), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    run_compiled(
        _get_gemm1(hidden_size, inter, num_experts, bm),
        _u8_flat(a_q),
        w13.view(torch.int64).view(-1),
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

    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    gemm2_out = torch.empty(
        (n_tokens * topk, hidden_size), dtype=torch.bfloat16, device=device
    )
    n_split = gemm2_n_split_for(n_tokens)
    gemm2_bm = gemm2_block_m_for(bm, n_tokens, inter)
    num_m_blocks2 = rows // gemm2_bm
    # 4 GB of partials at 65536 tokens: a byte view would exceed 2^31 elements
    run_compiled(
        _get_gemm2(hidden_size, inter, num_experts, topk, bm, n_split, gemm2_bm),
        h_q.view(-1),
        w2.view(torch.int64).view(-1),
        gemm2_out.view(-1),
        h_s,
        _u8_flat(w2_scale),
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        gemm2_grid(num_m_blocks2, n_split),
        stream,
    )
    run_compiled(
        _get_reduce_bf16(hidden_size, topk),
        gemm2_out.view(-1),
        out.view(-1),
        n_tokens,
        stream,
    )
    return out
