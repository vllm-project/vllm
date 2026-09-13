# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode chain (bf16 activations x MXFP8 weights, M <= 256): inline routing up to 16
tokens, ``sort_decode`` above; gate/up GEMM + swiglu-OAI (``gemm1_decode``), down
GEMM with routing-weighted bf16 atomic accumulation (``gemm2_decode``). Weights:
the gate/up-interleaved ``shuffle_weight`` / ``shuffle_scale`` tensors of AITER_MXFP8.
"""

import functools

import torch

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import _get, _run_compiled
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort import max_sorted_rows
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.sort_decode import (
    moe_sort_decode,
    wide_layout_rows,
)

from .gemm1_decode import BM, WIDE_BM, compile_gemm1
from .gemm2_decode import compile_gemm2

MAX_DECODE_TOKENS = 256
# sorted mode with a fused shared expert: its rows first in WIDE_BM-row blocks run
# by the wide kernel bodies, so its weights stream once per WIDE_BM rows (not BM)
WIDE_MIN_TOKENS = 40
_workspaces: dict = {}


def wide_for(n_tokens: int) -> bool:
    return n_tokens >= WIDE_MIN_TOKENS


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """Dimensions must fit the 256-wide K tiles and the three-way split-K."""
    return hidden_size % 256 == 0 and intermediate_size % (256 * 3) == 0


def _intermediate_workspace(
    device: torch.device, topk: int, intermediate_size: int, num_experts: int
) -> torch.Tensor:
    """gemm1 output, bf16 ``[rows, I]``: by pair (``pair * BM + row``) on the
    sort-free path, by sorted row on the sorted one; sized for the largest layout
    at ``MAX_DECODE_TOKENS`` and allocated once per (device, topk, I, E) so
    HIP-graph capture records no allocation."""
    key = (
        device.index if device.index is not None else -1,
        topk,
        intermediate_size,
        num_experts,
    )
    ws = _workspaces.get(key)
    if ws is None:
        rows = max(
            32 * topk * 32,
            max_sorted_rows(MAX_DECODE_TOKENS, num_experts, topk, BM),
            sum(wide_layout_rows(MAX_DECODE_TOKENS, num_experts, topk, BM, WIDE_BM)),
        )
        ws = torch.empty((rows, intermediate_size), dtype=torch.bfloat16, device=device)
        _workspaces[key] = ws
    return ws


def a16w8_decode_moe(
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
    swiglu_alpha: float,
    swiglu_limit: float,
    fused_shared_expert: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One MoE layer for ``M <= 256`` tokens -> ``[M, hidden_size]`` bf16 (``out`` when
    given: contiguous, zeroed and accumulated into). ``fused_shared_expert``: the
    last expert is routed by every token; only then may the wide-first sort layout
    (``WIDE_MIN_TOKENS``) be used, as it budgets that expert's blocks from ``M``.
    """
    n_tokens = x.shape[0]
    assert n_tokens <= MAX_DECODE_TOKENS, n_tokens
    assert x.shape[1] == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    inter = _intermediate_workspace(x.device, topk, intermediate_size, num_experts)
    if out is None:
        out = torch.empty(
            (n_tokens, hidden_size), dtype=torch.bfloat16, device=x.device
        )
    assert out.shape == (n_tokens, hidden_size) and out.dtype == torch.bfloat16
    assert out.is_contiguous()
    bm = BM
    inline = n_tokens <= bm
    wide = fused_shared_expert and wide_for(n_tokens)
    if inline:
        sorted_ids = sorted_w = sorted_eids = num_valid = None
    else:
        sorted_ids, sorted_w, sorted_eids, num_valid = moe_sort_decode(
            topk_ids,
            topk_weights,
            num_experts,
            hidden_size,
            bm,
            out,
            wide_first=(num_experts - 1, WIDE_BM) if wide else None,
        )
    a16w8_gemm1(
        x_bf16=x,
        w1_fp8=w13,
        w1_scale_u8=w13_scale,
        inter_sorted_bf16=inter,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        topk=topk,
        alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        inline_sort=inline,
        topk_ids=topk_ids,
        zero_out=out,
        sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid,
        sorted_token_ids=sorted_ids,
        BM=bm,
        wide=wide,
    )
    a16w8_gemm2(
        inter_sorted_bf16=inter,
        w2_fp8=w2,
        w2_scale_u8=w2_scale,
        out_bf16=out,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        inline_sort=inline,
        topk=topk,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_w,
        BM=bm,
        wide=wide,
    )
    return out


@functools.cache
def get_gemm1(**kw):
    return _get(compile_gemm1, **kw)


@functools.cache
def get_gemm2(**kw):
    return _get(compile_gemm2, **kw)


def a16w8_gemm1(
    *,
    x_bf16,
    w1_fp8,
    w1_scale_u8,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    alpha=1.702,
    swiglu_limit=7.0,
    sorted_expert_ids=None,
    num_valid_ids=None,
    sorted_token_ids=None,
    inline_sort=False,
    topk_ids=None,
    zero_out=None,
    BM=BM,
    wide=False,
):
    """Stage 1: gate/up GEMM + swiglu-OAI -> bf16 ``[sorted rows, D_INTER]``;
    ``BM`` is the sort's row block (16 or 32), ``wide`` the sort's wide-first
    layout of the shared expert (sorted mode)."""
    launch = get_gemm1(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        n_tokens=int(n_tokens),
        inline_sort=inline_sort,
        BM=BM,
        wide=wide,
    )
    if inline_sort:
        assert int(n_tokens) <= BM and topk_ids is not None and zero_out is not None
        max_m_blocks = int(n_tokens) * int(topk)
        eids_ptr, cumsum_ptr, mind_ptr = 0, 0, topk_ids.data_ptr()
        zero_ptr = zero_out.data_ptr()
        zero_dw = (zero_out.numel() * zero_out.element_size()) // 4
    else:
        max_m_blocks = int(sorted_expert_ids.numel())
        eids_ptr, cumsum_ptr, mind_ptr = (
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
        )
        zero_ptr, zero_dw = 0, 0
    if wide:
        n_wide = (int(n_tokens) + launch.wide_bm - 1) // launch.wide_bm
        grid = n_wide * launch.wide_n_blocks + (max_m_blocks - n_wide) * (
            D_INTER // launch.tile_n
        )
    else:
        grid = max_m_blocks * (D_INTER // launch.tile_n)
    _run_compiled(
        launch,
        x_bf16.data_ptr(),
        w1_fp8.data_ptr(),
        w1_scale_u8.data_ptr(),
        eids_ptr,
        cumsum_ptr,
        mind_ptr,
        int(n_tokens),
        int(grid),
        float(alpha),
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        int(zero_ptr),
        int(zero_dw),
        torch.cuda.current_stream(),
    )
    return inter_sorted_bf16


def a16w8_gemm2(
    *,
    inter_sorted_bf16,
    w2_fp8,
    w2_scale_u8,
    out_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    sorted_expert_ids=None,
    num_valid_ids=None,
    sorted_token_ids=None,
    sorted_weights=None,
    inline_sort=False,
    topk=None,
    topk_ids=None,
    topk_weights=None,
    BM=BM,
    wide=False,
):
    """Stage 2: down GEMM, routing-weighted bf16 atomic add into ``out_bf16``
    ``[n_tokens, D_HIDDEN]`` (zeroed beforehand); ``BM`` / ``wide`` as for gemm1."""
    launch = get_gemm2(
        NE=NE,
        N_OUT=D_HIDDEN,
        D_INTER=D_INTER,
        n_tokens=int(n_tokens),
        inline_sort=inline_sort,
        TOPK=topk if inline_sort else None,
        BM=BM,
        wide=wide,
    )
    if inline_sort:
        assert int(n_tokens) <= BM and topk_ids is not None and topk_weights is not None
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        max_m_blocks = int(n_tokens) * int(topk)
        eids_ptr, cumsum_ptr = 0, 0
        stids_ptr, sw_ptr = topk_ids.data_ptr(), topk_weights.data_ptr()
    else:
        max_m_blocks = int(sorted_expert_ids.numel())
        eids_ptr, cumsum_ptr = sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr()
        stids_ptr, sw_ptr = sorted_token_ids.data_ptr(), sorted_weights.data_ptr()
    nnb = D_HIDDEN // launch.tile_n
    if wide:
        n_sort_wide = (int(n_tokens) + launch.wide_sort_bm - 1) // launch.wide_sort_bm
        n_wide = n_sort_wide * (launch.wide_sort_bm // launch.wide_bm)
        grid = (n_wide + (max_m_blocks - n_sort_wide)) * nnb * launch.ksplit
    else:
        grid = max_m_blocks * nnb * launch.ksplit
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_fp8.data_ptr(),
        w2_scale_u8.data_ptr(),
        eids_ptr,
        cumsum_ptr,
        stids_ptr,
        sw_ptr,
        int(n_tokens),
        int(grid),
        out_bf16.data_ptr(),
        torch.cuda.current_stream(),
    )
    return out_bf16
