# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton port of the fused GDN MTP decode kernel (``fused_gdn_decode_post_conv_mtp``).

The CUDA kernel in csrc only builds for CUDA 13, so on ROCm every Qwen3.5 GDN layer's
speculative-decode step runs the unfused path: ~8 small copy/cat/fill kernels around the
recurrent kernel plus the gated RMSNorm. This kernel does the post-conv work in one launch.
One program per request and value head reads q/k/v, the output gate and a/b in place from the
projections, runs the gated delta-rule update over the request's tokens (same state reads and
writes as ``fused_sigmoid_gating_delta_rule_update``: start from the state of the last accepted
token, store the state after every token, slot index <= 0 means no state), applies the gated
RMSNorm and writes the out_proj input.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_gdn_decode_post_conv_mtp_kernel(
    mixed_qkv,
    b,
    a,
    z,
    A_log,
    dt_bias,
    norm_weight,
    state,
    state_indices,
    cu_seqlens,
    num_accepted_tokens,
    out,
    stride_qkv_row,
    stride_b_row,
    stride_a_row,
    stride_z_row,
    stride_state_slot,
    stride_indices_row,
    num_indices_per_row,
    scale,
    eps,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
):
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    num_tokens = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if num_tokens <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    p_out = out + (bos * HV + i_hv) * V + o_v
    accepted = tl.load(num_accepted_tokens + i_n).to(tl.int64)
    p_src = state_indices + i_n * stride_indices_row + accepted - 1
    has_src = (accepted > 0) & (accepted <= num_indices_per_row)
    src = tl.load(p_src, mask=has_src, other=0).to(tl.int64)
    if src <= 0:
        for t in range(num_tokens):
            tl.store(p_out + t * HV * V, tl.zeros([V], dtype=out.dtype.element_ty))
        return

    p_state = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(state + src * stride_state_slot + p_state).to(tl.float32)
    neg_A = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    bias = tl.load(dt_bias + i_hv).to(tl.float32)
    w = tl.load(norm_weight + o_v).to(tl.float32)
    for t in range(num_tokens):
        row = bos + t
        p_qkv = mixed_qkv + row * stride_qkv_row
        b_q = tl.load(p_qkv + i_h * K + o_k).to(tl.float32)
        b_k = tl.load(p_qkv + (H + i_h) * K + o_k).to(tl.float32)
        b_v = tl.load(p_qkv + 2 * H * K + i_hv * V + o_v).to(tl.float32)
        x = tl.load(a + row * stride_a_row + i_hv).to(tl.float32) + bias
        decay = tl.exp(neg_A * tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x))
        beta = tl.sigmoid(tl.load(b + row * stride_b_row + i_hv).to(tl.float32))

        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= decay
        b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * beta
        b_h += b_v[:, None] * b_k[None, :]
        # Rounded like the unfused path, whose recurrent kernel stores it first.
        b_o = tl.sum(b_h * b_q[None, :], 1).to(out.dtype.element_ty).to(tl.float32)

        b_z = tl.load(z + row * stride_z_row + i_hv * V + o_v).to(tl.float32)
        if SIGMOID_GATE:
            gate = tl.sigmoid(b_z)
        else:
            gate = b_z * tl.sigmoid(b_z)
        rstd = tl.rsqrt(tl.sum(b_o * b_o) / V + eps)
        y = b_o * rstd * w * gate
        tl.store(p_out + t * HV * V, y.to(out.dtype.element_ty))

        dst = tl.load(state_indices + i_n * stride_indices_row + t).to(tl.int64)
        if dst > 0:
            p_dst = state + dst * stride_state_slot + p_state
            tl.store(p_dst, b_h.to(state.dtype.element_ty))


def fused_gdn_decode_post_conv_mtp_triton(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    norm_eps: float,
    output_gate_activation: str,
) -> torch.Tensor:
    """Same arguments and result as ``ops.fused_gdn_decode_post_conv_mtp``.

    ``out`` and ``output_gate`` are [tokens, HV, V]; ``state`` is [slots, HV, V, K].
    """
    HV, V = out.shape[-2:]
    K = state.shape[-1]
    H = (mixed_qkv.shape[-1] - HV * V) // (2 * K)
    _fused_gdn_decode_post_conv_mtp_kernel[(cu_seqlens.numel() - 1, HV)](
        mixed_qkv,
        b,
        a,
        output_gate,
        A_log,
        dt_bias,
        norm_weight,
        state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        out,
        mixed_qkv.stride(0),
        b.stride(0),
        a.stride(0),
        output_gate.stride(0),
        state.stride(0),
        state_indices.stride(0),
        state_indices.shape[1],
        scale,
        norm_eps,
        H=H,
        HV=HV,
        K=K,
        V=V,
        SIGMOID_GATE=output_gate_activation == "sigmoid",
        num_warps=4,
    )
    return out
