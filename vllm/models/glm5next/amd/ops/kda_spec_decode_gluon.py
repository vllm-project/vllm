# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA spec decode with deferred state stores, in Gluon.

Same contract as the Triton kernel's ``DEFER_SPEC_STATE`` mode
(third_party/kda/fused_recurrent.py): slot 0 of a sequence's state indices
holds the state after the first token of the step, slot t > 0 a record of
token t, and the next step replays its ``num_accepted - 1`` accepted records
onto slot 0 before running its own tokens.

Two things make it faster than the Triton kernel (MI355X, 64 sequences x 32
heads, 8 tokens, 4 accepted: 165 -> 107 us):

* The layout is explicit. A program owns a [BV, K] slab of one head's
  fp32 state with 8 contiguous K columns per lane and K spread over 16 lanes
  of one warp, so every K reduction stays in the warp and no vector changes
  layout inside the token loop. The gate and the normalizations use one
  rcp/exp instruction per element instead of IEEE divisions and range-fixed
  exponentials.
* A record holds the normalized k, the decay a = exp(gate) and the
  delta-rule error err = sigmoid(beta) * (v - (a * S)^T k) rather than the
  raw inputs, so the replay is S = a * S + err (x) k: no reduction and no
  gate math. The token loop and the replay apply the update with the same
  ``_decay_update``, so the replayed state is bit-identical to the one the
  token loop computed.

The records are not interchangeable with the Triton kernel's, so a process
must use one kernel for all its steps.
"""

import torch

from vllm.triton_utils import gl, gluon

# Layout: 8 K columns per lane (one 16-byte bf16 load), BV = 32 state rows per
# one-warp program. Measured best of SPT_K in {4, 8, 16, 32}, BV in
# {16, 32, 64} and 1, 2 or 4 warps.
SPT_K = 8
BLOCK_V = 32
NUM_WARPS = 1


LOG2E: gl.constexpr = gl.constexpr(1.4426950408889634)


@gluon.jit
def _rcp(x):
    # v_rcp_f32 (1 ulp) instead of a ~10-instruction IEEE division.
    return gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _exp2(x):
    # Bare v_exp_f32, without the denormal range fixup (tiny results flush
    # to 0, which the gate and the sigmoid absorb).
    return gl.inline_asm_elementwise(
        "v_exp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _decay_update(b_h, b_a, err, b_k):
    # S = a * S + err (x) k, the one update both the token loop and the
    # replay apply.
    return gl.fma(err[:, None], b_k[None, :], b_h * b_a[None, :])


@gluon.jit
def kda_spec_decode_defer_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    a_log,
    g_bias,
    scale,
    stride_state_slot,
    stride_indices_seq,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    stride_beta_t,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    SPT_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
):
    TPW_K: gl.constexpr = K // SPT_K
    L: gl.constexpr = gl.BlockedLayout(
        [1, SPT_K], [64 // TPW_K, TPW_K], [NUM_WARPS, 1], [1, 0]
    )
    LK: gl.constexpr = gl.SliceLayout(0, L)
    LV: gl.constexpr = gl.SliceLayout(1, L)

    i_v = gl.program_id(0)
    i_nh = gl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = gl.load(cu_seqlens + i_n).to(gl.int64)
    T = gl.load(cu_seqlens + i_n + 1).to(gl.int64) - bos
    p_idx = ssm_state_indices + i_n * stride_indices_seq
    state_idx = gl.load(p_idx).to(gl.int64)
    if (T == 0) | (state_idx <= 0):
        return

    o_k = gl.arange(0, K, layout=LK)
    o_r = gl.arange(0, BV, layout=LV)
    o_v = i_v * BV + o_r
    p_h = (
        state
        + state_idx * stride_state_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = gl.load(p_h)
    # gate = LOWER_BOUND * sigmoid(exp(a_log) * (g + g_bias)), in base 2.
    neg_a_exp_log2e = -LOG2E * gl.exp(gl.load(a_log + i_h).to(gl.float32))
    b_bias = gl.load(g_bias + i_h * K + o_k).to(gl.float32)

    # Record: k [K] | a [K] | err [BV]. Each program has its own, so programs
    # never share a record and the replay reads only what this program wrote.
    REC_W: gl.constexpr = 2 * K + BV
    rec_off = (i_hv * gl.num_programs(0) + i_v) * REC_W
    num_replay = gl.load(num_accepted_tokens + i_n).to(gl.int32) - 1
    for i_r in range(1, num_replay + 1):
        rec_idx = gl.load(p_idx + i_r).to(gl.int64)
        p_r = state + rec_idx * stride_state_slot + rec_off
        b_h = _decay_update(
            b_h,
            gl.load(p_r + K + o_k),
            gl.load(p_r + 2 * K + o_r),
            gl.load(p_r + o_k),
        )
    # The replay must finish reading the records before this step's tokens
    # overwrite them.
    gl.barrier()

    p_q = q + bos * stride_q_t + i_h * K + o_k
    p_k = k + bos * stride_k_t + i_h * K + o_k
    p_g = g + (bos * HV + i_hv) * K + o_k
    p_v = v + bos * stride_v_t + i_hv * V + o_v
    p_beta = beta + bos * stride_beta_t + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    for i_t in range(0, T):
        b_q = gl.load(p_q).to(gl.float32)
        b_k = gl.load(p_k).to(gl.float32)
        b_g = gl.load(p_g).to(gl.float32)
        b_v = gl.load(p_v).to(gl.float32)
        b_beta = gl.load(p_beta).to(gl.float32)
        # One IEEE division per token-wide scalar, none per element.
        b_q = b_q * (scale / gl.sqrt(gl.sum(b_q * b_q, 0) + 1e-6))
        b_k = b_k * (1.0 / gl.sqrt(gl.sum(b_k * b_k, 0) + 1e-6))
        b_a = _exp2(
            (LOWER_BOUND * LOG2E) * _rcp(1.0 + _exp2(neg_a_exp_log2e * (b_g + b_bias)))
        )
        b_sk = gl.sum(b_h * b_a[None, :] * b_k[None, :], 1)
        err = (b_v - b_sk) * (1.0 / (1.0 + _exp2(-LOG2E * b_beta)))
        b_h = _decay_update(b_h, b_a, err, b_k)
        b_o = gl.sum(b_h * b_q[None, :], 1)
        gl.store(p_o, b_o.to(o.dtype.element_ty))
        if i_t == 0:
            gl.store(p_h, b_h)
        else:
            rec_idx = gl.load(p_idx + i_t).to(gl.int64)
            if rec_idx > 0:
                p_r = state + rec_idx * stride_state_slot + rec_off
                gl.store(p_r + o_k, b_k)
                gl.store(p_r + K + o_k, b_a)
                gl.store(p_r + 2 * K + o_r, err)
        p_q += stride_q_t
        p_k += stride_k_t
        p_g += HV * K
        p_v += stride_v_t
        p_beta += stride_beta_t
        p_o += HV * V


def supports_kda_spec_decode_gluon(K: int, V: int, headwise_beta: bool) -> bool:
    """Shapes the kernel handles. It has no masks, so K must be a power of 2
    filling whole lanes of a warp and V whole programs; beta must be one value
    per head. Only static properties decide, so a process never mixes the two
    kernels' records."""
    return (
        K & (K - 1) == 0
        and K % SPT_K == 0
        and 64 % (K // SPT_K) == 0
        and V % BLOCK_V == 0
        and not headwise_beta
    )


def kda_spec_decode_defer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    lower_bound: float,
    out: torch.Tensor,
    stride_q_t: int,
    stride_k_t: int,
    stride_v_t: int,
    stride_beta_t: int,
) -> None:
    """Writes the outputs to ``out`` and updates ``state`` in place.
    ``a_log`` and ``g_bias`` are flat; ``g`` is contiguous [.., HV, K]."""
    H, K = k.shape[-2:]
    HV, V = v.shape[-2:]
    assert supports_kda_spec_decode_gluon(K, V, beta.ndim == v.ndim)
    assert g.is_contiguous() and out.is_contiguous()
    assert state.dtype == torch.float32
    # Each slot holds one record per program.
    NV = V // BLOCK_V
    assert NV * (2 * K + BLOCK_V) <= V * K
    grid = (NV, (len(cu_seqlens) - 1) * HV)
    kda_spec_decode_defer_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        out,
        state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        a_log,
        g_bias,
        scale,
        state.stride(0),
        ssm_state_indices.stride(0),
        stride_q_t,
        stride_k_t,
        stride_v_t,
        stride_beta_t,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BV=BLOCK_V,
        SPT_K=SPT_K,
        NUM_WARPS=NUM_WARPS,
        LOWER_BOUND=lower_bound,
        num_warps=NUM_WARPS,
    )
