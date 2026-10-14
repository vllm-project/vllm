# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Width-1..4 conv plus gated delta for Qwen3.8-27B MTP verify.

The conv-state row is a second launch. Tap products and SiLU are
rounded to bf16, matching ``causal_conv1d_update``. History is read at
``num_accepted_tokens - 1``. The stored row is
``[history[offset+1], history[offset+2], x0, ..., x_{T-1}]``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_H = 16
_HV = 48
_K = 128
_V = 128
_BK = 128
_BV = 32


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
    store,
    conv_out_ptr,
    SIZE: tl.constexpr,
    T: tl.constexpr,
    WRITE_Y: tl.constexpr,
):
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
    h0 = tl.load(base + offset * stride_conv_tok, mask=mask, other=0)
    h1 = tl.load(base + (offset + 1) * stride_conv_tok, mask=mask, other=0)
    h2 = tl.load(base + (offset + 2) * stride_conv_tok, mask=mask, other=0)
    c0 = h0.to(tl.float32)
    c1 = h1.to(tl.float32)
    c2 = h2.to(tl.float32)
    y0 = tl.zeros([SIZE], dtype=tl.float32)
    y1 = tl.zeros([SIZE], dtype=tl.float32)
    y2 = tl.zeros([SIZE], dtype=tl.float32)
    y3 = tl.zeros([SIZE], dtype=tl.float32)
    x0 = tl.zeros([SIZE], dtype=tl.bfloat16)
    x1 = tl.zeros([SIZE], dtype=tl.bfloat16)
    x2 = tl.zeros([SIZE], dtype=tl.bfloat16)
    x3 = tl.zeros([SIZE], dtype=tl.bfloat16)
    for t in tl.static_range(T):
        raw = tl.load(x_ptr + t * stride_x_tok + chan, mask=mask, other=0)
        acc = (
            (c0 * w0).to(tl.bfloat16).to(tl.float32)
            + (c1 * w1).to(tl.bfloat16).to(tl.float32)
            + (c2 * w2).to(tl.bfloat16).to(tl.float32)
            + (raw.to(tl.float32) * w3).to(tl.bfloat16).to(tl.float32)
        )
        y = acc / (1.0 + tl.exp(-acc))
        y = y.to(tl.bfloat16).to(tl.float32)
        if t == 0:
            y0 = y
            x0 = raw
        elif t == 1:
            y1 = y
            x1 = raw
        elif t == 2:
            y2 = y
            x2 = raw
        else:
            y3 = y
            x3 = raw
        if WRITE_Y:
            tl.store(
                conv_out_ptr + t * stride_x_tok + chan, y.to(tl.bfloat16), mask=mask
            )
        c0 = c1
        c1 = c2
        c2 = raw.to(tl.float32)
    if store:
        tl.store(base + 0 * stride_conv_tok, h1, mask=mask)
        tl.store(base + 1 * stride_conv_tok, h2, mask=mask)
        tl.store(base + 2 * stride_conv_tok, x0, mask=mask)
        if T > 1:
            tl.store(base + 3 * stride_conv_tok, x1, mask=mask)
        if T > 2:
            tl.store(base + 4 * stride_conv_tok, x2, mask=mask)
        if T > 3:
            tl.store(base + 5 * stride_conv_tok, x3, mask=mask)
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
    state_ptr,
    out_ptr,
    conv_out_ptr,
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
    stride_x_batch,
    stride_a_batch,
    stride_b_batch,
    stride_out_batch,
    stride_indices_batch,
    stride_accepted,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    T: tl.constexpr,
    WRITE_Y: tl.constexpr,
):
    i_tile = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_tile * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    seq = tl.program_id(2).to(tl.int64)
    x_ptr = x_ptr + seq * stride_x_batch
    a_ptr = a_ptr + seq * stride_a_batch
    b_ptr = b_ptr + seq * stride_b_batch
    out_ptr = out_ptr + seq * stride_out_batch
    conv_out_ptr = conv_out_ptr + seq * stride_x_batch
    indices_ptr = indices_ptr + seq * stride_indices_batch
    accepted_ptr = accepted_ptr + seq * stride_accepted

    accepted = tl.load(accepted_ptr).to(tl.int32)
    offset = accepted - 1
    # 48*128*128 * slot overflows int32 at slot 2731. This pool has ~2970 blocks.
    conv_slot = tl.load(indices_ptr).to(tl.int64)
    src_slot = tl.load(indices_ptr + offset * stride_indices).to(tl.int64)
    if (conv_slot <= 0) | (src_slot <= 0):
        return

    chan_q = i_h * K + o_k
    chan_k = H * K + i_h * K + o_k
    chan_v = 2 * H * K + i_hv * V + o_v
    store_qk = False
    q0, q1, q2, q3 = _conv_channels(
        x_ptr, w_ptr, conv_ptr, chan_q, mask_k,
        stride_x_tok, stride_w_dim, stride_w_w,
        stride_conv_slot, stride_conv_dim, stride_conv_tok,
        conv_slot, offset, store_qk, conv_out_ptr, BK, T, WRITE_Y,
    )
    k0, k1, k2, k3 = _conv_channels(
        x_ptr, w_ptr, conv_ptr, chan_k, mask_k,
        stride_x_tok, stride_w_dim, stride_w_w,
        stride_conv_slot, stride_conv_dim, stride_conv_tok,
        conv_slot, offset, store_qk, conv_out_ptr, BK, T, WRITE_Y,
    )
    v0, v1, v2, v3 = _conv_channels(
        x_ptr, w_ptr, conv_ptr, chan_v, mask_v,
        stride_x_tok, stride_w_dim, stride_w_w,
        stride_conv_slot, stride_conv_dim, stride_conv_tok,
        conv_slot, offset, False, conv_out_ptr, BV, T, WRITE_Y,
    )

    a_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
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
        if t == 0:
            b_q, b_k, b_v = q0, k0, v0
        elif t == 1:
            b_q, b_k, b_v = q1, k1, v1
        elif t == 2:
            b_q, b_k, b_v = q2, k2, v2
        else:
            b_q, b_k, b_v = q3, k3, v3
        gate_x = tl.load(a_ptr + t * stride_a + i_hv).to(tl.float32) + dt_bias_val
        softplus_x = tl.where(gate_x <= 20.0, tl.log(1.0 + tl.exp(gate_x)), gate_x)
        b_g = -tl.exp(a_log_val) * softplus_x
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
            out_ptr + t * stride_out_tok + i_hv * stride_out_hv + o_v * stride_out_v,
            b_o.to(tl.bfloat16),
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


@triton.jit
def _shift_conv_state(
    x_ptr,
    conv_ptr,
    indices_ptr,
    accepted_ptr,
    stride_x_tok,
    stride_conv_slot,
    stride_conv_dim,
    stride_conv_tok,
    stride_x_batch,
    stride_indices_batch,
    stride_accepted,
    DIM,
    BLOCK: tl.constexpr,
    T: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.program_id(1).to(tl.int64)
    x_ptr = x_ptr + seq * stride_x_batch
    indices_ptr = indices_ptr + seq * stride_indices_batch
    accepted_ptr = accepted_ptr + seq * stride_accepted
    chan = pid * BLOCK + tl.arange(0, BLOCK)
    mask = chan < DIM
    slot = tl.load(indices_ptr).to(tl.int64)
    if slot <= 0:
        return
    offset = tl.load(accepted_ptr).to(tl.int32) - 1
    base = (
        conv_ptr
        + slot * stride_conv_slot.to(tl.int64)
        + chan.to(tl.int64) * stride_conv_dim
    )
    h1 = tl.load(base + (offset + 1) * stride_conv_tok, mask=mask, other=0)
    h2 = tl.load(base + (offset + 2) * stride_conv_tok, mask=mask, other=0)
    tl.store(base + 0 * stride_conv_tok, h1, mask=mask)
    tl.store(base + 1 * stride_conv_tok, h2, mask=mask)
    for t in tl.static_range(T):
        raw = tl.load(x_ptr + t * stride_x_tok + chan, mask=mask, other=0)
        tl.store(base + (t + 2) * stride_conv_tok, raw, mask=mask)


def launch_conv_gdn(
    mixed, weight, conv_state, a, b, A_log, dt_bias, indices, num_accepted,
    ssm_state, out, conv_out=None, num_warps: int = 4, num_stages: int = 1,
    bv: int = _BV, n_seq: int = 1,
) -> None:
    tokens = mixed.shape[0]
    if n_seq < 1 or tokens % n_seq != 0:
        raise ValueError(f"tokens={tokens} is not divisible by n_seq={n_seq}")
    seq_tokens = tokens // n_seq
    if seq_tokens not in (1, 2, 3, 4):
        raise ValueError(f"T must be 1..4, got {seq_tokens}")
    write_y = conv_out is not None
    if conv_out is None:
        conv_out = mixed
    _mtp_conv_gdn[(triton.cdiv(_V, bv), _HV, n_seq)](
        mixed, weight, conv_state, a, b, A_log, dt_bias, indices, num_accepted,
        ssm_state, out, conv_out,
        mixed.stride(0), weight.stride(0), weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        a.stride(0), b.stride(0), indices.stride(1),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        _K**-0.5,
        seq_tokens * mixed.stride(0),
        seq_tokens * a.stride(0),
        seq_tokens * b.stride(0),
        seq_tokens * out.stride(0),
        indices.stride(0),
        num_accepted.stride(0),
        H=_H, HV=_HV, K=_K, V=_V, BK=_BK, BV=bv, T=seq_tokens, WRITE_Y=write_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    _shift_conv_state[(triton.cdiv(mixed.shape[1], 256), n_seq)](
        mixed, conv_state, indices, num_accepted,
        mixed.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        seq_tokens * mixed.stride(0),
        indices.stride(0),
        num_accepted.stride(0),
        mixed.shape[1], BLOCK=256, T=seq_tokens, num_warps=4,
    )


def try_qwen_mtp_conv_gdn(
    mixed_qkv, conv_state, weight, a, b, A_log, dt_bias, ssm_state,
    indices, num_accepted, out,
) -> bool:
    tokens = mixed_qkv.shape[0]
    if tokens not in (1, 2, 3, 4):
        return False
    if indices is None or num_accepted is None:
        return False
    if indices.ndim != 2 or indices.shape[0] != 1 or indices.shape[1] < tokens:
        return False
    if num_accepted.numel() != 1:
        return False
    if out.shape[0] < tokens:
        return False
    launch_conv_gdn(
        mixed_qkv, weight, conv_state, a, b, A_log, dt_bias,
        indices[:, :tokens], num_accepted, ssm_state, out[:tokens],
        num_warps=2,
    )
    return True
