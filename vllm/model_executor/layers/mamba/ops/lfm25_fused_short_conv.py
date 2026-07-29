# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Guarded LFM2.5 width-three ShortConv decode fusion."""

import logging
import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

FUSED_DECODE_ENABLED = os.getenv("VLLM_LFM25_FUSED_SHORTCONV", "0") == "1"
if FUSED_DECODE_ENABLED:
    logging.getLogger(__name__).info("[LFM2.5] ShortConv decode fusion ENABLED")


@triton.jit()
def _lfm25_fused_short_conv_decode_kernel(
    b_ptr,
    c_ptr,
    x_ptr,
    state_ptr,
    weight_ptr,
    state_indices_ptr,
    out_ptr,
    stride_b_token,
    stride_b_dim,
    stride_c_token,
    stride_c_dim,
    stride_x_token,
    stride_x_dim,
    stride_state_block,
    stride_state_dim,
    stride_state_token,
    stride_weight_dim,
    stride_weight_token,
    stride_indices,
    stride_out_token,
    stride_out_dim,
    dim: tl.constexpr,
    null_block_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = offs < dim

    b = tl.load(b_ptr + pid * stride_b_token + offs * stride_b_dim, mask=m, other=0.0)
    c = tl.load(c_ptr + pid * stride_c_token + offs * stride_c_dim, mask=m, other=0.0)
    x = tl.load(x_ptr + pid * stride_x_token + offs * stride_x_dim, mask=m, other=0.0)

    gate = (b.to(tl.float32) * x.to(tl.float32)).to(b_ptr.dtype.element_ty)
    out_p = out_ptr + pid * stride_out_token + offs * stride_out_dim

    sidx = tl.load(state_indices_ptr + pid * stride_indices).to(tl.int64)
    if sidx == null_block_id:
        p = c.to(tl.float32) * gate.to(tl.float32)
        tl.store(out_p, p, mask=m)
        return

    sb = state_ptr + sidx * stride_state_block + offs * stride_state_dim
    s0 = tl.load(sb, mask=m, other=0.0)
    s1 = tl.load(sb + stride_state_token, mask=m, other=0.0)

    wb = weight_ptr + offs * stride_weight_dim
    w0 = tl.load(wb, mask=m, other=0.0)
    w1 = tl.load(wb + stride_weight_token, mask=m, other=0.0)
    w2 = tl.load(wb + 2 * stride_weight_token, mask=m, other=0.0)

    gs = gate.to(state_ptr.dtype.element_ty)
    cv = (
        (s0 * w0 + s1 * w1 + gs * w2)
        .to(state_ptr.dtype.element_ty)
        .to(b_ptr.dtype.element_ty)
    )
    tl.store(out_p, c.to(tl.float32) * cv.to(tl.float32), mask=m)
    tl.store(sb, s1, mask=m)
    tl.store(sb + stride_state_token, gs, mask=m)


def fused_lfm25_short_conv_decode(
    b,
    c,
    x,
    conv_state,
    weight,
    state_indices,
):
    if not all(
        t.is_cuda and t.device == b.device
        for t in (b, c, x, conv_state, weight, state_indices)
    ):
        raise ValueError("all tensors must be on the same CUDA device")
    if b.shape != c.shape or b.shape != x.shape or weight.shape != (b.shape[1], 3):
        raise ValueError("shape mismatch: need B, C, X [n, d] and weight [d, 3]")
    if b.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("B, C, X must be BF16/FP16")
    if (
        weight.stride(1) != 1
        or conv_state.ndim != 3
        or conv_state.shape[1] != b.shape[1]
    ):
        raise ValueError(
            "weight must be contiguous [d, 3]; state must be [blocks, d, >=2]"
        )
    if state_indices.shape[0] < b.shape[0]:
        raise ValueError("need one state index per token")

    out = torch.empty_like(b)
    from vllm.platforms import current_platform

    bn = (
        512
        if getattr(current_platform, "has_device_capability", lambda x: False)(90)
        else 256
    )
    _lfm25_fused_short_conv_decode_kernel[(b.shape[0], triton.cdiv(b.shape[1], bn))](
        b,
        c,
        x,
        conv_state,
        weight,
        state_indices,
        out,
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        x.stride(0),
        x.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight.stride(0),
        weight.stride(1),
        state_indices.stride(0),
        out.stride(0),
        out.stride(1),
        b.shape[1],
        NULL_BLOCK_ID,
        BLOCK_N=bn,
        num_warps=8,
    )
    return out
