# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv, gated delta, and gated RMSNorm for Qwen GDN MTP verify.

Conv state is shifted in a second launch. The stored row is
``[history[offset+1], history[offset+2], x0, ..., x_{T-1}]``. Mixed
widths read per-sequence rows from ``cu_seqlens`` on device.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.mamba.ops.rocm_qwen_mtp_conv_gdn import (
    _mtp_conv_gdn,
)

_BV = 32
_CONV_BLOCK = 256
_CONV_WIDTH = 4


@triton.jit
def _shift_and_norm(
    x_ptr,
    conv_ptr,
    indices_ptr,
    accepted_ptr,
    cu_seqlens_ptr,
    o_ptr,
    z_ptr,
    norm_w_ptr,
    y_ptr,
    stride_x_tok,
    stride_conv_slot,
    stride_conv_dim,
    stride_conv_tok,
    stride_o_row,
    stride_z_tok,
    stride_z_head,
    stride_y_row,
    stride_indices_batch,
    stride_accepted,
    eps,
    DIM,
    N_CONV,
    HV: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK: tl.constexpr,
    T: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    GATE_SIGMOID: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + seq).to(tl.int64)
        seq_len = tl.load(cu_seqlens_ptr + seq + 1).to(tl.int32) - bos.to(tl.int32)
    else:
        bos = seq * T
        seq_len = T
    if pid < N_CONV:
        chan = pid * BLOCK + tl.arange(0, BLOCK)
        mask = chan < DIM
        slot = tl.load(indices_ptr + seq * stride_indices_batch).to(tl.int64)
        if slot > 0:
            offset = tl.load(accepted_ptr + seq * stride_accepted).to(tl.int32) - 1
            base = (
                conv_ptr
                + slot * stride_conv_slot.to(tl.int64)
                + chan.to(tl.int64) * stride_conv_dim
            )
            h1 = tl.load(base + (offset + 1) * stride_conv_tok, mask=mask, other=0)
            h2 = tl.load(base + (offset + 2) * stride_conv_tok, mask=mask, other=0)
            tl.store(base + 0 * stride_conv_tok, h1, mask=mask)
            tl.store(base + 1 * stride_conv_tok, h2, mask=mask)
            x_seq = x_ptr + bos * stride_x_tok
            for t in tl.static_range(T):
                if t < seq_len:
                    raw = tl.load(x_seq + t * stride_x_tok + chan, mask=mask, other=0)
                    tl.store(base + (t + 2) * stride_conv_tok, raw, mask=mask)
    else:
        row = pid - N_CONV
        t_local = row // HV
        # The uniform grid has exactly T * HV norm rows per sequence.
        valid = t_local < seq_len if IS_VARLEN else True
        if valid:
            tok = bos + t_local
            head = row % HV
            cols = tl.arange(0, BLOCK_V)
            mask_v = cols < V
            x = tl.load(
                o_ptr + (tok * HV + head) * stride_o_row + cols, mask=mask_v, other=0
            ).to(tl.float32)
            z = tl.load(
                z_ptr + tok * stride_z_tok + head * stride_z_head + cols,
                mask=mask_v,
                other=0,
            ).to(tl.float32)
            w = tl.load(norm_w_ptr + cols, mask=mask_v, other=0).to(tl.float32)
            # Same op order as RMSNormGated.forward_static (norm_before_gate).
            rstd = tl.rsqrt(tl.sum(x * x) / V + eps)
            y = (x * rstd) * w
            y = y * tl.sigmoid(z) if GATE_SIGMOID else y * (z * tl.sigmoid(z))
            tl.store(
                y_ptr + (tok * HV + head) * stride_y_row + cols,
                y.to(y_ptr.dtype.element_ty),
                mask=mask_v,
            )


def launch_conv_gdn_norm(
    mixed,
    weight,
    conv_state,
    a,
    b,
    A_log,
    dt_bias,
    indices,
    num_accepted,
    ssm_state,
    core_out,
    z,
    norm_weight,
    eps,
    normed_out,
    gate_sigmoid: bool = False,
    n_seq: int = 1,
    cu_seqlens: torch.Tensor | None = None,
) -> None:
    """``core_out``, ``z``, ``normed_out`` are ``[tokens, HV, V]``;
    ``normed_out`` may alias ``core_out``.

    ``ssm_state`` is ``[slots, HV, V, K]``. ``mixed`` is
    ``[tokens, 2 * H * K + HV * V]``. ``indices`` is ``[n_seq, >= T]``;
    column ``t`` is the slot written after token ``t``, and the incoming
    state is read at column ``num_accepted - 1`` (``num_accepted`` is
    ``[n_seq]``, each in ``1..indices.shape[1]``). Without ``cu_seqlens`` every sequence
    has ``T = tokens // n_seq`` rows. With ``cu_seqlens`` (``[n_seq + 1]``,
    on device), sequence ``i`` is rows ``cu_seqlens[i]:cu_seqlens[i + 1]``,
    at most ``indices.shape[1]``; other rows are left untouched.
    """
    tokens, dim = mixed.shape
    seq_tokens = tokens // n_seq if cu_seqlens is None else indices.shape[1]
    _, num_v_heads, head_v, head_k = ssm_state.shape
    qk_dim = dim - num_v_heads * head_v
    num_k_heads = max(qk_dim // (2 * head_k), 1)
    if cu_seqlens is None and tokens != n_seq * seq_tokens:
        raise ValueError(f"tokens={tokens} is not divisible by n_seq={n_seq}")
    if not 1 <= seq_tokens <= 4:
        raise ValueError(f"T={seq_tokens} must be 1..4")
    if indices.shape[0] < n_seq or indices.shape[1] < seq_tokens:
        raise ValueError(
            f"indices {tuple(indices.shape)} for n_seq={n_seq} T={seq_tokens}"
        )
    if conv_state.shape[-1] < indices.shape[1] + 2:
        raise ValueError(
            f"conv state length {conv_state.shape[-1]} < indices width "
            f"{indices.shape[1]} + 2"
        )
    if num_accepted.shape[0] < n_seq:
        raise ValueError(f"num_accepted {tuple(num_accepted.shape)} for n_seq={n_seq}")
    if (
        mixed.dtype not in (torch.bfloat16, torch.float16)
        or ssm_state.dtype != torch.float32
    ):
        raise ValueError(
            f"mixed {mixed.dtype} needs bf16/fp16, ssm_state {ssm_state.dtype} fp32"
        )
    if weight.shape[1] != _CONV_WIDTH:
        raise ValueError(f"conv width {weight.shape[1]}, expected {_CONV_WIDTH}")
    if qk_dim != 2 * num_k_heads * head_k or num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"mixed width {dim} does not split into q, k, v for "
            f"HV={num_v_heads} V={head_v} K={head_k}"
        )
    if z.shape != (tokens, num_v_heads, head_v) or any(
        t.stride(-1) != 1 for t in (mixed, a, b, z)
    ):
        raise ValueError("z must be [tokens, HV, V]; mixed, a, b, z need unit stride")
    is_varlen = cu_seqlens is not None
    bv = min(_BV, triton.next_power_of_2(head_v))
    _mtp_conv_gdn[(triton.cdiv(head_v, bv), num_v_heads, n_seq)](
        mixed,
        weight,
        conv_state,
        a,
        b,
        A_log,
        dt_bias,
        indices,
        num_accepted,
        cu_seqlens,
        ssm_state,
        core_out,
        mixed.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        a.stride(0),
        b.stride(0),
        indices.stride(1),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        core_out.stride(0),
        core_out.stride(1),
        core_out.stride(2),
        head_k**-0.5,
        indices.stride(0),
        num_accepted.stride(0),
        H=num_k_heads,
        HV=num_v_heads,
        K=head_k,
        V=head_v,
        BK=triton.next_power_of_2(head_k),
        BV=bv,
        T=seq_tokens,
        IS_VARLEN=is_varlen,
        num_warps=2,
        num_stages=1,
    )
    n_conv = triton.cdiv(dim, _CONV_BLOCK)
    o2 = core_out.view(tokens * num_v_heads, head_v)
    y2 = normed_out.view(tokens * num_v_heads, head_v)
    _shift_and_norm[(n_conv + seq_tokens * num_v_heads, n_seq)](
        mixed,
        conv_state,
        indices,
        num_accepted,
        cu_seqlens,
        o2,
        z,
        norm_weight,
        y2,
        mixed.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        o2.stride(0),
        z.stride(0),
        z.stride(1),
        y2.stride(0),
        indices.stride(0),
        num_accepted.stride(0),
        eps,
        dim,
        n_conv,
        HV=num_v_heads,
        V=head_v,
        BLOCK_V=triton.next_power_of_2(head_v),
        BLOCK=_CONV_BLOCK,
        T=seq_tokens,
        IS_VARLEN=is_varlen,
        GATE_SIGMOID=gate_sigmoid,
        num_warps=4,
    )
