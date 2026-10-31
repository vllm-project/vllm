# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton port of the fused GDN MTP decode kernel (``fused_gdn_decode_post_conv_mtp``).

The CUDA kernel in csrc only builds for CUDA 13, so on ROCm every Qwen3.5 GDN layer's
speculative-decode step runs the unfused path: ~8 small copy/cat/fill kernels around the
recurrent kernel plus the gated RMSNorm. This does the post-conv work in two launches.
The update kernel reads q/k/v and a/b in place from the projections and runs the gated
delta-rule update over each request's tokens (same state reads and writes as
``fused_sigmoid_gating_delta_rule_update``: start from the state of the last accepted token,
store the state after every token, slot index <= 0 means no state). Like that kernel it tiles
the value dimension, so a small batch still fills the GPU; every tile writes its outputs and
their sum of squares. The epilogue then applies the gated RMSNorm, the only step that needs
the whole value dimension, and writes the out_proj input.

With ``defer_spec_state`` the update kernel stores the full state only after a request's
first token (slot 0 of its ``state_indices`` row). For the other tokens it records the
normalized k, the v tile, the decay and beta in slot 1; the next step replays the
``num_accepted_tokens - 1`` accepted records onto slot 0 before its own tokens. That is one
state store per step instead of one per token. Every spec step of a request must use this
mode, and nothing else may read the per-token slots.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units


# The value tile size of the deferred state mode; the records depend on it.
DEFER_SPEC_STATE_BV = 16


def deferred_spec_state_fits(num_spec: int, K: int, V: int) -> bool:
    """Whether the records of ``num_spec`` draft tokens fit in one state slot."""
    BV = DEFER_SPEC_STATE_BV
    return num_spec * (V // BV) * (K + BV + 2) <= V * K


@triton.jit
def _gdn_mtp_update_kernel(
    mixed_qkv,
    b,
    a,
    A_log,
    dt_bias,
    state,
    state_indices,
    cu_seqlens,
    num_accepted_tokens,
    out,
    sum_sq,
    stride_qkv_row,
    stride_b_row,
    stride_a_row,
    stride_state_slot,
    stride_indices_row,
    num_indices_per_row,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    DEFER_SPEC_STATE: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV: tl.constexpr = V // BV
    # One deferred-state record: normalized k, the v tile, decay and beta.
    RECORD: tl.constexpr = K + BV + 2
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    num_tokens = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    if num_tokens <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_out = out + (bos * HV + i_hv) * V + o_v
    p_sum_sq = sum_sq + (bos * HV + i_hv) * NV + i_v
    accepted = tl.load(num_accepted_tokens + i_n).to(tl.int64)
    if DEFER_SPEC_STATE:
        src = tl.load(state_indices + i_n * stride_indices_row).to(tl.int64)
    else:
        p_src = state_indices + i_n * stride_indices_row + accepted - 1
        has_src = (accepted > 0) & (accepted <= num_indices_per_row)
        src = tl.load(p_src, mask=has_src, other=0).to(tl.int64)
    if src <= 0:
        for t in range(num_tokens):
            tl.store(p_out + t * HV * V, tl.zeros([BV], dtype=out.dtype.element_ty))
            tl.store(p_sum_sq + t * HV * NV, 0.0)
        return

    p_state = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(state + src * stride_state_slot + p_state).to(tl.float32)
    neg_A = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    bias = tl.load(dt_bias + i_hv).to(tl.float32)
    if DEFER_SPEC_STATE:
        # Each program owns its records, so BV must be the same in every step.
        record_slot = tl.load(state_indices + i_n * stride_indices_row + 1).to(tl.int64)
        p_records = state + record_slot * stride_state_slot + (i_hv * NV + i_v) * RECORD
        o_bv = tl.arange(0, BV)
        for r in range(tl.minimum(accepted, num_indices_per_row) - 1):
            p_record = p_records + r * HV * NV * RECORD
            b_k = tl.load(p_record + o_k)
            b_v = tl.load(p_record + K + o_bv)
            b_h *= tl.load(p_record + K + BV)
            b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * tl.load(p_record + K + BV + 1)
            b_h += b_v[:, None] * b_k[None, :]
        # The records are read before this step overwrites them.
        tl.debug_barrier()
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
        if DEFER_SPEC_STATE and t > 0:
            p_record = p_records + (t - 1) * HV * NV * RECORD
            tl.store(p_record + o_k, b_k)
            tl.store(p_record + K + o_bv, b_v)
            tl.store(p_record + K + BV, decay)
            tl.store(p_record + K + BV + 1, beta)
        b_h *= decay
        b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * beta
        b_h += b_v[:, None] * b_k[None, :]
        # Rounded like the unfused path, whose recurrent kernel stores it first.
        b_o = tl.sum(b_h * b_q[None, :], 1).to(out.dtype.element_ty)
        tl.store(p_out + t * HV * V, b_o)
        b_o = b_o.to(tl.float32)
        tl.store(p_sum_sq + t * HV * NV, tl.sum(b_o * b_o))

        if DEFER_SPEC_STATE:
            if t == 0:
                tl.store(state + src * stride_state_slot + p_state, b_h)
        else:
            dst = tl.load(state_indices + i_n * stride_indices_row + t).to(tl.int64)
            if dst > 0:
                p_dst = state + dst * stride_state_slot + p_state
                tl.store(p_dst, b_h.to(state.dtype.element_ty))


@triton.jit
def _gdn_mtp_gated_rmsnorm_kernel(
    out,
    sum_sq,
    z,
    norm_weight,
    cu_seqlens,
    stride_z_row,
    eps,
    HV: tl.constexpr,
    V: tl.constexpr,
    NV: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
):
    i_n, i_hv, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    if t >= tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos:
        return

    row = bos + t
    o_v = tl.arange(0, V)
    p_out = out + (row * HV + i_hv) * V + o_v
    b_o = tl.load(p_out).to(tl.float32)
    p_sum_sq = sum_sq + (row * HV + i_hv) * NV + tl.arange(0, NV)
    rstd = tl.rsqrt(tl.sum(tl.load(p_sum_sq)) / V + eps)
    b_z = tl.load(z + row * stride_z_row + i_hv * V + o_v).to(tl.float32)
    if SIGMOID_GATE:
        gate = tl.sigmoid(b_z)
    else:
        gate = b_z * tl.sigmoid(b_z)
    w = tl.load(norm_weight + o_v).to(tl.float32)
    tl.store(p_out, (b_o * rstd * w * gate).to(out.dtype.element_ty))


def gdn_decode_mtp_update_triton(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    defer_spec_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The update kernel alone: writes the outputs before the gated RMSNorm to ``out``.

    Returns ``out`` and the per-tile sums of squares of the outputs, [tokens, HV, NV].
    """
    HV, V = out.shape[-2:]
    K = state.shape[-1]
    H = (mixed_qkv.shape[-1] - HV * V) // (2 * K)
    num_tokens = mixed_qkv.shape[0]
    num_requests = cu_seqlens.numel() - 1
    # Layouts the kernels index with (as the CUDA op checks).
    assert K == triton.next_power_of_2(K) and V == triton.next_power_of_2(V) >= 64
    assert H > 0 and HV % H == 0 and mixed_qkv.shape[-1] == 2 * H * K + HV * V
    assert mixed_qkv.dim() == 2 and mixed_qkv.stride(1) == 1
    assert a.shape == b.shape == (num_tokens, HV) and a.stride(1) == b.stride(1) == 1
    assert A_log.is_contiguous() and A_log.numel() == HV
    assert dt_bias.is_contiguous() and dt_bias.numel() == HV
    assert state.dim() == 4 and state.shape[1:3] == (HV, V)
    assert state.stride(1) == V * K and state.stride(2) == K and state.stride(3) == 1
    assert state_indices.dim() == 2 and state_indices.shape[0] == num_requests
    assert state_indices.stride(1) == 1 and cu_seqlens.is_contiguous()
    assert num_accepted_tokens.is_contiguous()
    assert num_accepted_tokens.numel() == num_requests
    assert out.shape == (num_tokens, HV, V) and out.is_contiguous()
    if defer_spec_state:
        # The replay matches the per-token states only if no state is rounded. The
        # records fix the tile layout, so BV must not depend on the batch; one warp per
        # program is fastest except for a single request.
        assert state.dtype == torch.float32
        assert deferred_spec_state_fits(state_indices.shape[1] - 1, K, V)
        BV, num_warps = DEFER_SPEC_STATE_BV, 4 if num_requests == 1 else 1
    else:
        # Tile V only while the GPU needs more programs: 32 rows per program while there
        # are fewer (request, value head) pairs than CUs, 64 up to 4x the CUs, else one
        # tile per value head (tuned on MI300X and MI350X).
        heads, num_cus = num_requests * HV, num_compute_units(out.device.index)
        BV = 32 if heads < num_cus else 64 if heads <= 4 * num_cus else V
        num_warps = 4
    sum_sq = torch.empty(
        (out.shape[0], HV, V // BV), dtype=torch.float32, device=out.device
    )
    _gdn_mtp_update_kernel[(V // BV, num_requests, HV)](
        mixed_qkv,
        b,
        a,
        A_log,
        dt_bias,
        state,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        out,
        sum_sq,
        mixed_qkv.stride(0),
        b.stride(0),
        a.stride(0),
        state.stride(0),
        state_indices.stride(0),
        state_indices.shape[1],
        scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BV,
        DEFER_SPEC_STATE=defer_spec_state,
        num_warps=num_warps,
    )
    return out, sum_sq


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
    defer_spec_state: bool = False,
) -> torch.Tensor:
    """Same arguments and result as ``ops.fused_gdn_decode_post_conv_mtp``.

    ``out`` and ``output_gate`` are [tokens, HV, V]; ``state`` is [slots, HV, V, K].
    ``defer_spec_state`` selects the deferred state layout (see the module docstring).
    """
    HV, V = out.shape[-2:]
    num_requests = cu_seqlens.numel() - 1
    assert output_gate.shape == out.shape
    assert output_gate.stride(2) == 1 and output_gate.stride(1) == V
    assert norm_weight.is_contiguous() and norm_weight.numel() == V
    assert output_gate_activation in ("silu", "sigmoid")
    out, sum_sq = gdn_decode_mtp_update_triton(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state_indices,
        cu_seqlens,
        num_accepted_tokens,
        state,
        out,
        scale,
        defer_spec_state,
    )
    _gdn_mtp_gated_rmsnorm_kernel[(num_requests, HV, state_indices.shape[1])](
        out,
        sum_sq,
        output_gate,
        norm_weight,
        cu_seqlens,
        output_gate.stride(0),
        norm_eps,
        HV=HV,
        V=V,
        NV=sum_sq.shape[-1],
        SIGMOID_GATE=output_gate_activation == "sigmoid",
        num_warps=2,
    )
    return out
