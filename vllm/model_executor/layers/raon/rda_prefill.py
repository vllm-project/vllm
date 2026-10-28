# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled Raon prefill convolution with fused post-convolution value scaling."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


@triton.jit
def _kda_front_conv_kernel(
    x_ptr,  # [T, dim] row-strided, feature stride 1
    w_ptr,  # [dim, W]
    conv_states_ptr,  # [N, dim, >= W-1]
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # [T, dim] feature stride 1
    beta_ptr,
    dim,
    num_cache_lines,
    stride_x_token: tl.int64,
    stride_w_dim,
    stride_w_width,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    stride_cache_indices,
    stride_o_token: tl.int64,
    stride_b_t,
    stride_b_h,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    RDA: tl.constexpr,
    HEAD_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SUB_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    KW: tl.constexpr = 4
    state_len: tl.constexpr = KW - 1
    NH: tl.constexpr = BLOCK_N // HEAD_D
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    offs_h = tl.program_id(1) * NH + tl.arange(0, NH)
    offs_d = tl.arange(0, HEAD_D)
    idx_feats = (offs_h[:, None] * HEAD_D + offs_d[None, :])[None, :, :]  # [1, NH, D]
    if idx_seq == pad_slot_id:
        return
    seq_start = tl.load(query_start_loc_ptr + idx_seq)
    seq_end = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = seq_end - seq_start
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    coord = tl.load(cache_indices_ptr + idx_seq * stride_cache_indices).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa: SIM102 - null_block_id may be None at compile time.
        if coord == null_block_id:
            return
    mask_f = idx_feats < dim
    load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
    cs_base = conv_states_ptr + coord * stride_cs_seq + idx_feats * stride_cs_dim
    x_seq = x_ptr + seq_start.to(tl.int64) * stride_x_token + idx_feats

    w_base = w_ptr + idx_feats * stride_w_dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_f, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_f, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask=mask_f, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask=mask_f, other=0.0)
    for s0 in tl.static_range(0, BLOCK_M, SUB_M):
        rows = s0 + tl.arange(0, SUB_M)
        rows3 = rows[:, None, None]
        row_ok = rows3 < segment_len
        acc = tl.zeros((SUB_M, NH, HEAD_D), dtype=tl.float32)
        for j in tl.static_range(KW):
            u = token_offset + rows3 - (KW - 1 - j)
            mx = (u >= 0) & row_ok & mask_f
            xv = tl.load(x_seq + u.to(tl.int64) * stride_x_token, mask=mx, other=0.0)
            if j < KW - 1:
                ms = (u < 0) & mask_f & load_init
                sv = tl.load(
                    cs_base + (state_len + u) * stride_cs_tok, mask=ms, other=0.0
                )
                xv = tl.where(u < 0, sv, xv)
            if j == 0:
                acc += xv * w0
            elif j == 1:
                acc += xv * w1
            elif j == 2:
                acc += xv * w2
            else:
                acc += xv * w3
        acc = acc / (1 + tl.exp(-acc))
        y = acc.to(o_ptr.dtype.element_ty)
        tok = seq_start + token_offset + rows
        mo = row_ok & mask_f
        if RDA:
            mh = (rows < segment_len)[:, None] & (offs_h < dim // HEAD_D)[None, :]
            b = tl.load(
                beta_ptr + (tok * stride_b_t)[:, None] + (offs_h * stride_b_h)[None, :],
                mask=mh,
                other=0.0,
            ).to(tl.float32)
            ratio = b
            y = (y.to(tl.float32) * ratio[:, :, None]).to(o_ptr.dtype.element_ty)
        tl.store(
            o_ptr + (tok.to(tl.int64) * stride_o_token)[:, None, None] + idx_feats,
            y,
            mask=mo,
        )

    if chunk_offset == 0:
        idx_c = tl.arange(0, 4)[:, None, None]  # NP2_STATELEN for state_len 3
        tl.debug_barrier()
        if state_len <= seqlen:
            it = (seqlen - state_len) + idx_c
            cmx = (it >= 0) & (it < seqlen) & mask_f
            cnv = tl.load(x_seq + it.to(tl.int64) * stride_x_token, mask=cmx, other=0.0)
        else:
            VAL = state_len - seqlen
            cmx = ((idx_c - VAL) >= 0) & ((idx_c - VAL) < seqlen) & mask_f
            cnv = tl.load(
                x_seq + (idx_c - VAL).to(tl.int64) * stride_x_token, mask=cmx, other=0.0
            )
            if load_init:
                cms = (
                    (coord < num_cache_lines) & ((idx_c + seqlen) < state_len) & mask_f
                )
                csv = tl.load(
                    cs_base + (idx_c + seqlen) * stride_cs_tok, mask=cms, other=0.0
                )
                tl.debug_barrier()
                cnv = tl.where(cms, csv, cnv)
        tl.debug_barrier()
        mw = (idx_c < state_len) & mask_f
        tl.store(cs_base + idx_c * stride_cs_tok, cnv, mask=mw)


def prefill_conv(x, state, weight, metadata, scale=None):
    count, dim = x.shape
    nd = metadata.nums_dict[8]
    batch = nd.get("batch_ptr")
    chunks = nd.get("token_chunk_offset_ptr")
    if batch is None:
        batch = metadata.batch_ptr
    if chunks is None:
        chunks = metadata.token_chunk_offset_ptr
    result = torch.empty_like(x, memory_format=torch.contiguous_format)
    indices = metadata.non_spec_state_indices_tensor
    ratio = scale if scale is not None else x
    _kda_front_conv_kernel[(nd["tot"], triton.cdiv(dim, 256))](
        x,
        weight,
        state,
        indices,
        metadata.has_initial_state,
        metadata.non_spec_query_start_loc,
        batch,
        chunks,
        result,
        ratio,
        dim,
        state.size(0),
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        indices.stride(0),
        result.stride(0),
        ratio.stride(0),
        ratio.stride(1),
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        HAS_NULL_BLOCK=NULL_BLOCK_ID is not None,
        RDA=scale is not None,
        HEAD_D=128,
        BLOCK_M=8,
        SUB_M=8,
        BLOCK_N=256,
        num_warps=4,
    )
    return result
