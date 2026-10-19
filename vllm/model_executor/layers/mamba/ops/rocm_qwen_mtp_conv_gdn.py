# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Width-1..4 conv plus gated delta for Qwen GDN MTP verify.

Tap products and SiLU are rounded to the input dtype, matching
``causal_conv1d_update``. History is read at ``num_accepted_tokens - 1``.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _conv_channels(
    x_ptr,
    w_ptr,
    conv_ptr,
    chan,
    mask,
    stride_x_tok,
    stride_w_dim,
    stride_w_w,
    stride_conv_slot,
    stride_conv_dim,
    stride_conv_tok,
    slot,
    offset,
    seq_len,
    SIZE: tl.constexpr,
    T: tl.constexpr,
):
    dtype = x_ptr.dtype.element_ty
    w_base = w_ptr + chan * stride_w_dim
    w0 = tl.load(w_base + 0 * stride_w_w, mask=mask, other=0).to(tl.float32)
    w1 = tl.load(w_base + 1 * stride_w_w, mask=mask, other=0).to(tl.float32)
    w2 = tl.load(w_base + 2 * stride_w_w, mask=mask, other=0).to(tl.float32)
    w3 = tl.load(w_base + 3 * stride_w_w, mask=mask, other=0).to(tl.float32)
    base = (
        conv_ptr
        + slot.to(tl.int64) * stride_conv_slot.to(tl.int64)
        + chan.to(tl.int64) * stride_conv_dim
    )
    c0 = tl.load(base + offset * stride_conv_tok, mask=mask, other=0).to(tl.float32)
    c1 = tl.load(base + (offset + 1) * stride_conv_tok, mask=mask, other=0).to(
        tl.float32
    )
    c2 = tl.load(base + (offset + 2) * stride_conv_tok, mask=mask, other=0).to(
        tl.float32
    )
    y0 = tl.zeros([SIZE], dtype=tl.float32)
    y1 = tl.zeros([SIZE], dtype=tl.float32)
    y2 = tl.zeros([SIZE], dtype=tl.float32)
    y3 = tl.zeros([SIZE], dtype=tl.float32)
    for t in tl.static_range(T):
        if t < seq_len:
            raw = tl.load(x_ptr + t * stride_x_tok + chan, mask=mask, other=0)
            acc = (
                (c0 * w0).to(dtype).to(tl.float32)
                + (c1 * w1).to(dtype).to(tl.float32)
                + (c2 * w2).to(dtype).to(tl.float32)
                + (raw.to(tl.float32) * w3).to(dtype).to(tl.float32)
            )
            y = acc / (1.0 + tl.exp(-acc))
            y = y.to(dtype).to(tl.float32)
            if t == 0:
                y0 = y
            elif t == 1:
                y1 = y
            elif t == 2:
                y2 = y
            else:
                y3 = y
            c0 = c1
            c1 = c2
            c2 = raw.to(tl.float32)
    return y0, y1, y2, y3


@triton.jit
def _mtp_conv_gdn(
    x_ptr,
    w_ptr,
    conv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    indices_ptr,
    accepted_ptr,
    cu_seqlens_ptr,
    state_ptr,
    out_ptr,
    stride_x_tok,
    stride_w_dim,
    stride_w_w,
    stride_conv_slot,
    stride_conv_dim,
    stride_conv_tok,
    stride_a,
    stride_b,
    stride_indices,
    stride_state_slot,
    stride_state_hv,
    stride_state_v,
    stride_state_k,
    stride_out_tok,
    stride_out_hv,
    stride_out_v,
    scale,
    stride_indices_batch,
    stride_accepted,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    T: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """``T`` is the max tokens per sequence. With ``IS_VARLEN``, sequence
    ``i`` is rows ``cu_seqlens[i]:cu_seqlens[i + 1]``; otherwise ``i * T``
    onward, ``T`` rows."""
    i_tile = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_tile * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    seq = tl.program_id(2).to(tl.int64)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + seq).to(tl.int64)
        seq_len = tl.load(cu_seqlens_ptr + seq + 1).to(tl.int32) - bos.to(tl.int32)
    else:
        bos = seq * T
        seq_len = T
    x_ptr = x_ptr + bos * stride_x_tok
    a_ptr = a_ptr + bos * stride_a
    b_ptr = b_ptr + bos * stride_b
    out_ptr = out_ptr + bos * stride_out_tok
    indices_ptr = indices_ptr + seq * stride_indices_batch
    accepted_ptr = accepted_ptr + seq * stride_accepted

    accepted = tl.load(accepted_ptr).to(tl.int32)
    offset = accepted - 1
    # slot * stride_state_slot overflows int32 on large pools.
    conv_slot = tl.load(indices_ptr).to(tl.int64)
    src_slot = tl.load(indices_ptr + offset * stride_indices).to(tl.int64)
    if (conv_slot <= 0) | (src_slot <= 0):
        return

    chan_q = i_h * K + o_k
    chan_k = H * K + i_h * K + o_k
    chan_v = 2 * H * K + i_hv * V + o_v
    q0, q1, q2, q3 = _conv_channels(
        x_ptr,
        w_ptr,
        conv_ptr,
        chan_q,
        mask_k,
        stride_x_tok,
        stride_w_dim,
        stride_w_w,
        stride_conv_slot,
        stride_conv_dim,
        stride_conv_tok,
        conv_slot,
        offset,
        seq_len,
        BK,
        T,
    )
    k0, k1, k2, k3 = _conv_channels(
        x_ptr,
        w_ptr,
        conv_ptr,
        chan_k,
        mask_k,
        stride_x_tok,
        stride_w_dim,
        stride_w_w,
        stride_conv_slot,
        stride_conv_dim,
        stride_conv_tok,
        conv_slot,
        offset,
        seq_len,
        BK,
        T,
    )
    v0, v1, v2, v3 = _conv_channels(
        x_ptr,
        w_ptr,
        conv_ptr,
        chan_v,
        mask_v,
        stride_x_tok,
        stride_w_dim,
        stride_w_w,
        stride_conv_slot,
        stride_conv_dim,
        stride_conv_tok,
        conv_slot,
        offset,
        seq_len,
        BV,
        T,
    )

    neg_a = -tl.exp(tl.load(A_log_ptr + i_hv).to(tl.float32))
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    p_src = (
        state_ptr
        + src_slot * stride_state_slot.to(tl.int64)
        + i_hv * stride_state_hv
        + o_v[:, None] * stride_state_v
        + o_k[None, :] * stride_state_k
    )
    b_h = tl.load(p_src, mask=mask_h, other=0).to(tl.float32)
    for t in tl.static_range(T):
        if t < seq_len:
            if t == 0:
                b_q, b_k, b_v = q0, k0, v0
            elif t == 1:
                b_q, b_k, b_v = q1, k1, v1
            elif t == 2:
                b_q, b_k, b_v = q2, k2, v2
            else:
                b_q, b_k, b_v = q3, k3, v3
            gate_x = tl.load(a_ptr + t * stride_a + i_hv).to(tl.float32)
            gate_x = gate_x + dt_bias_val
            softplus_x = tl.where(gate_x <= 20.0, tl.log(1.0 + tl.exp(gate_x)), gate_x)
            b_g = neg_a * softplus_x
            b_beta = tl.sigmoid(tl.load(b_ptr + t * stride_b + i_hv).to(tl.float32))
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale
            b_h = b_h * tl.exp(b_g)
            b_v = b_v - tl.sum(b_h * b_k[None, :], 1)
            b_v = b_v * b_beta
            b_h = b_h + b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(
                out_ptr
                + t * stride_out_tok
                + i_hv * stride_out_hv
                + o_v * stride_out_v,
                b_o.to(out_ptr.dtype.element_ty),
                mask=mask_v,
            )
            dest = tl.load(indices_ptr + t * stride_indices).to(tl.int64)
            if dest > 0:
                p_ht = (
                    state_ptr
                    + dest * stride_state_slot.to(tl.int64)
                    + i_hv * stride_state_hv
                    + o_v[:, None] * stride_state_v
                    + o_k[None, :] * stride_state_k
                )
                tl.store(p_ht, b_h, mask=mask_h)
