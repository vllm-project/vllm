# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL W4A4 MoE prefill chain at TP4 on the ``AITER_MXFP4_MXFP4`` weights: sort,
aiter's scale sort, gemm1 (gate/up + SwiGLU-OAI + MXFP4 quant), gemm2
(routing-weighted partials), fixed-order reduce.
"""

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
GEMM2_N_SPLIT = 2
GEMM1_SWIGLU_ALPHA = 1.702
GEMM1_SWIGLU_LIMIT = 7.0
_GEMM1_BLOCK_K = 256
# gemm2's K pipeline is written for 3 K-steps of 256
_GEMM2_INTERMEDIATE = 768


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 peels 4 K-steps of 256 and unrolls the rest by 4."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % _GEMM1_BLOCK_K == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


def block_m_for(n_tokens: int) -> int:
    return 256 if n_tokens >= BM256_FROM_TOKENS else 128


@functools.cache
def _get_gemm1(hidden_size: int, intermediate_size: int, num_experts: int, bm: int):
    from .gemm1_prefill import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=bm
    )


@functools.cache
def _get_gemm2(
    hidden_size: int, intermediate_size: int, num_experts: int, topk: int, bm: int
):
    from .gemm2_prefill import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=GEMM2_N_SPLIT,
        out_dtype="bf16",
        sort_block_m=bm,
    )


def a4w4_prefill_moe(
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
    """``x_quant`` = (fp4 ``[M, H/2]``, e8m0 ``[M, H/32]``) of aiter's per-token
    MXFP4 quant -> ``[M, H]`` bf16, on ``AiterMxfp4Experts``' weights."""
    from aiter import dtypes
    from aiter.ops.quant import mxfp4_moe_sort_hip

    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import SortBuffers
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.tile_map import (
        tile_map_grid,
    )

    from .gemm2_prefill import gemm2_grid

    a_q, x_s = x_quant
    n_tokens = a_q.shape[0]
    hidden = hidden_size
    assert MIN_PREFILL_TOKENS <= n_tokens <= MAX_PREFILL_TOKENS, n_tokens
    assert a_q.shape == (n_tokens, hidden // 2) and a_q.is_contiguous()
    assert x_s.shape == (n_tokens, hidden // 32) and x_s.is_contiguous()
    inter = intermediate_size
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = topk_ids.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)

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
    h_q = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    run_compiled(
        _get_gemm1(hidden, inter, num_experts, bm),
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

    partial = torch.empty(
        (n_tokens * topk, hidden), dtype=torch.bfloat16, device=device
    )
    # read only by gemm2's fp8-partial mode
    partial_scale = torch.empty((16,), dtype=torch.uint8, device=device)
    num_m_blocks2 = rows // 128
    run_compiled(
        _get_gemm2(hidden, inter, num_experts, topk, bm),
        h_q.view(-1),
        _u8_flat(w2),
        partial.view(-1),
        h_s,
        _u8_flat(w2_scale),
        partial_scale,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        gemm2_grid(num_m_blocks2, GEMM2_N_SPLIT),
        stream,
    )
    if out is None:
        out = torch.empty((n_tokens, hidden), dtype=torch.bfloat16, device=device)
    run_compiled(
        _get_reduce_bf16(hidden, topk),
        partial.view(-1),
        out.view(-1),
        n_tokens,
        stream,
    )
    return out


def mxfp4_quant(
    x: torch.Tensor, q_out: torch.Tensor | None, scale_out: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token MXFP4 quant of bf16 ``[M, H]`` by aiter's kernel (the regular
    path's from 512 tokens) into ``q_out`` ``[M, H/2]`` / ``scale_out``
    ``[M, H/32]`` uint8."""
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    m, h = x.shape
    if q_out is None:
        q_out = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    if scale_out is None:
        scale_out = torch.empty((m, h // 32), dtype=torch.uint8, device=x.device)
    dynamic_per_group_scaled_quant(
        q_out.view(dtypes.fp4x2),
        x,
        scale_out.view(dtypes.fp8_e8m0),
        32,
        shuffle_scale=False,
    )
    return q_out, scale_out
