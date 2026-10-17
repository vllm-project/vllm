# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv, gated delta, and gated RMSNorm for Qwen3.8-27B MTP verify.

Conv state is shifted in a second launch. Mixed widths are one launch
per width.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.mamba.ops.rocm_qwen_mtp_conv_gdn import (
    _BK,
    _BV,
    _H,
    _HV,
    _K,
    _V,
    _mtp_conv_gdn,
)

_CONV_BLOCK = 256


@triton.jit
def _shift_and_norm(
    x_ptr,
    conv_ptr,
    indices_ptr,
    accepted_ptr,
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
    stride_x_batch,
    stride_indices_batch,
    stride_accepted,
    eps,
    DIM,
    N_CONV,
    HV: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
    T: tl.constexpr,
    GATE_SIGMOID: tl.constexpr,
):
    pid = tl.program_id(0)
    seq = tl.program_id(1).to(tl.int64)
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
            x_seq = x_ptr + seq * stride_x_batch
            for t in tl.static_range(T):
                raw = tl.load(x_seq + t * stride_x_tok + chan, mask=mask, other=0)
                tl.store(base + (t + 2) * stride_conv_tok, raw, mask=mask)
    else:
        row = pid - N_CONV
        tok = seq * T + row // HV
        head = row % HV
        cols = tl.arange(0, V)
        x = tl.load(o_ptr + (tok * HV + head) * stride_o_row + cols).to(tl.float32)
        z = tl.load(
            z_ptr + tok * stride_z_tok + head * stride_z_head + cols
        ).to(tl.float32)
        w = tl.load(norm_w_ptr + cols).to(tl.float32)
        # Same op order as RMSNormGated.forward_static (norm_before_gate).
        rstd = tl.rsqrt(tl.sum(x * x) / V + eps)
        y = (x * rstd) * w
        if GATE_SIGMOID:
            y = y * tl.sigmoid(z)
        else:
            y = y * (z * tl.sigmoid(z))
        tl.store(
            y_ptr + (tok * HV + head) * stride_y_row + cols, y.to(tl.bfloat16)
        )


def launch_conv_gdn_norm(
    mixed, weight, conv_state, a, b, A_log, dt_bias, indices, num_accepted,
    ssm_state, core_out, z, norm_weight, eps, normed_out,
    gate_sigmoid: bool = False, num_warps: int = 2, bv: int = _BV,
    n_seq: int = 1,
) -> None:
    """``core_out``, ``z``, ``normed_out`` are ``[n_seq * T, HV, V]``.

    ``indices`` is ``[n_seq, T]``. ``num_accepted`` is ``[n_seq]``.
    """
    tokens = mixed.shape[0]
    if n_seq < 1 or tokens % n_seq != 0:
        raise ValueError(f"tokens={tokens} is not divisible by n_seq={n_seq}")
    seq_tokens = tokens // n_seq
    if seq_tokens not in (1, 2, 3, 4):
        raise ValueError(f"T must be 1..4, got {seq_tokens}")
    if indices.shape[0] < n_seq or indices.shape[1] < seq_tokens:
        raise ValueError(
            f"indices {tuple(indices.shape)} for n_seq={n_seq} T={seq_tokens}"
        )
    if num_accepted.shape[0] < n_seq:
        raise ValueError(
            f"num_accepted {tuple(num_accepted.shape)} for n_seq={n_seq}"
        )
    _mtp_conv_gdn[(triton.cdiv(_V, bv), _HV, n_seq)](
        mixed, weight, conv_state, a, b, A_log, dt_bias, indices, num_accepted,
        ssm_state, core_out, mixed,
        mixed.stride(0), weight.stride(0), weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        a.stride(0), b.stride(0), indices.stride(1),
        ssm_state.stride(0), ssm_state.stride(1), ssm_state.stride(2),
        ssm_state.stride(3),
        core_out.stride(0), core_out.stride(1), core_out.stride(2),
        _K**-0.5,
        seq_tokens * mixed.stride(0),
        seq_tokens * a.stride(0),
        seq_tokens * b.stride(0),
        seq_tokens * core_out.stride(0),
        indices.stride(0),
        num_accepted.stride(0),
        H=_H, HV=_HV, K=_K, V=_V, BK=_BK, BV=bv, T=seq_tokens, WRITE_Y=False,
        num_warps=num_warps, num_stages=1,
    )
    dim = mixed.shape[1]
    n_conv = triton.cdiv(dim, _CONV_BLOCK)
    o2 = core_out.reshape(tokens * _HV, _V)
    y2 = normed_out.reshape(tokens * _HV, _V)
    if z.stride(-1) != 1 or z.shape != (tokens, _HV, _V):
        raise ValueError("z must be [n_seq * T, HV, V] with unit stride along V")
    _shift_and_norm[(n_conv + seq_tokens * _HV, n_seq)](
        mixed, conv_state, indices, num_accepted, o2, z, norm_weight, y2,
        mixed.stride(0), conv_state.stride(0), conv_state.stride(1),
        conv_state.stride(2),
        o2.stride(0), z.stride(0), z.stride(1), y2.stride(0),
        seq_tokens * mixed.stride(0),
        indices.stride(0),
        num_accepted.stride(0),
        eps, dim, n_conv,
        HV=_HV, V=_V, BLOCK=_CONV_BLOCK, T=seq_tokens, GATE_SIGMOID=gate_sigmoid,
        num_warps=4,
    )


def launch_conv_gdn_norm_varlen(
    mixed, weight, conv_state, a, b, A_log, dt_bias, indices, num_accepted,
    ssm_state, z, norm_weight, eps, normed_out, lengths: list[int],
    gate_sigmoid: bool = False, num_warps: int = 2,
) -> None:
    """Packed sequences of widths 1..4. One launch per distinct width."""
    if not lengths or sum(lengths) > normed_out.shape[0]:
        raise ValueError(f"lengths {lengths} do not fit {tuple(normed_out.shape)}")
    total = sum(lengths)
    normed_out[:total].zero_()
    if all(length == lengths[0] for length in lengths):
        core = torch.empty_like(normed_out[:total])
        launch_conv_gdn_norm(
            mixed[:total], weight, conv_state, a[:total], b[:total], A_log,
            dt_bias, indices[: len(lengths), : lengths[0]],
            num_accepted[: len(lengths)], ssm_state, core, z[:total],
            norm_weight, eps, normed_out[:total], gate_sigmoid=gate_sigmoid,
            num_warps=num_warps, n_seq=len(lengths),
        )
        return

    groups: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append((index, offset))
        offset += length
    for length, members in groups.items():
        n = len(members)
        mixed_g = torch.cat(
            [mixed[start : start + length] for _, start in members], 0
        )
        a_g = torch.cat([a[start : start + length] for _, start in members], 0)
        b_g = torch.cat([b[start : start + length] for _, start in members], 0)
        z_g = torch.cat([z[start : start + length] for _, start in members], 0)
        rows = torch.tensor(
            [index for index, _ in members],
            device=indices.device,
            dtype=torch.long,
        )
        core_g = torch.empty(
            (n * length, z.shape[1], z.shape[2]),
            dtype=normed_out.dtype,
            device=normed_out.device,
        )
        out_g = torch.empty_like(core_g)
        launch_conv_gdn_norm(
            mixed_g, weight, conv_state, a_g, b_g, A_log, dt_bias,
            indices[rows, :length], num_accepted[rows], ssm_state, core_g, z_g,
            norm_weight, eps, out_g, gate_sigmoid=gate_sigmoid,
            num_warps=num_warps, n_seq=n,
        )
        for piece, (_, start) in enumerate(members):
            normed_out[start : start + length].copy_(
                out_g[piece * length : (piece + 1) * length]
            )
