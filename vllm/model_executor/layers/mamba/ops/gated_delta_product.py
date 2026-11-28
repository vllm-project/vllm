# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Single-token decode kernel for gated delta product layers."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def fused_recurrent_gated_delta_product_kernel(
    conv_out,
    b,
    a,
    A_log,
    dt_bias,
    o,
    state,
    ssm_state_indices,
    stride_conv_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_o_tok: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    M: tl.constexpr,  # Householder sub-steps per real token
    H: tl.constexpr,  # key/query groups
    HV: tl.constexpr,  # value heads
    K: tl.constexpr,  # state (key) dim
    V: tl.constexpr,  # head (value) dim
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    tok = i_n.to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    VDIM: tl.constexpr = HV * V
    KDIM: tl.constexpr = H * K

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)

    if state_idx <= 0:
        # NULL slot: the kernel skips this sequence; keep its output rows defined.
        zero = tl.zeros([BV], dtype=tl.float32).to(o.dtype.element_ty)
        p_o = o + tok * stride_o_tok + i_hv * V + o_v
        tl.store(p_o, zero, mask=mask_v)
        return

    p_state = (
        state
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_state, mask=mask_h, other=0).to(tl.float32)

    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    neg_A = -tl.exp(A_log_val)

    p_conv = conv_out + tok * stride_conv_tok
    # gate on the first sub-step only (eager path: virtual_g[:, 0] = g)
    a_val = tl.load(a + tok * stride_a_tok + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g_val = neg_A * softplus_x
    b_h *= tl.exp(g_val)

    for s in tl.static_range(M):
        b_v = tl.load(p_conv + s * VDIM + i_hv * V + o_v, mask=mask_v, other=0).to(
            tl.float32
        )
        b_k = tl.load(
            p_conv + M * VDIM + s * KDIM + i_h * K + o_k, mask=mask_k, other=0
        ).to(tl.float32)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        beta_val = tl.sigmoid(
            tl.load(b + tok * stride_b_tok + s * HV + i_hv).to(tl.float32)
        )
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]

    # output of the last sub-step with the (normalised, scaled) query
    b_q = tl.load(
        p_conv + M * VDIM + M * KDIM + i_h * K + o_k, mask=mask_k, other=0
    ).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_q = b_q * K**-0.5
    b_o = tl.sum(b_h * b_q[None, :], 1)
    p_o = o + tok * stride_o_tok + i_hv * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    tl.store(p_state, b_h.to(p_state.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_product(
    conv_out: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor,
    *,
    num_householder: int,
    num_groups: int,
    num_heads: int,
    state_dim: int,
    head_dim: int,
    ssm_state_indices: torch.Tensor,
) -> None:
    """Update one state slot per request and write the next-token outputs."""
    M, H, HV, K, V = num_householder, num_groups, num_heads, state_dim, head_dim
    T = conv_out.shape[0]
    assert conv_out.shape[1] == M * HV * V + (M + 1) * H * K, conv_out.shape
    assert conv_out.stride(1) == 1 and b.stride(1) == 1 and a.stride(1) == 1
    assert out.shape[0] >= T and out.shape[1:] == (HV, V) and out.is_contiguous()
    assert ssm_state.stride(-1) == 1 and ssm_state.shape[1:] == (HV, V, K)
    stride_seq = ssm_state_indices.stride(0)
    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    if T == 0:
        return
    # Sequences x heads on grid axis 0: axis 1 is limited to 65535 blocks.
    grid = (T * HV, NV)
    fused_recurrent_gated_delta_product_kernel[grid](
        conv_out,
        b,
        a,
        A_log,
        dt_bias,
        out,
        ssm_state,
        ssm_state_indices,
        stride_conv_tok=conv_out.stride(0),
        stride_b_tok=b.stride(0),
        stride_a_tok=a.stride(0),
        stride_o_tok=out.stride(0),
        stride_state_slot=ssm_state.stride(0),
        stride_indices_seq=stride_seq,
        M=M,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=1,
        num_stages=3,
    )
