# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrappers for the a16w8 decode gemm1 / gemm2 kernels.

Uses aiter ``moe_sorting`` buffers on the sorted path and
``topk_ids`` / ``topk_weights`` on the inline-sort path; the
weights are the MXFP8 tensors ``shuffle_mxfp8_moe_weights`` stores on the layer.
"""

import functools

import torch

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import _get, _run_compiled

from .gemm1 import BM, compile_gemm1
from .gemm2 import compile_gemm2


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
