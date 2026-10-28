# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gated_delta_rule_step(
    b_h,
    b_k,
    b_v,
    b_a,
    b_b,
    b_A_log,
    b_dt_bias,
    beta,
    threshold,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = b_a + b_dt_bias
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    b_g = -tl.exp(b_A_log) * softplus_x

    # compute beta_output = sigmoid(b)
    b_beta = tl.sigmoid(b_b)

    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
    # [BV, BK]
    if not IS_KDA:
        b_h *= tl.exp(b_g)
    else:
        b_h *= tl.exp(b_g[None, :])
    # [BV]
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    # [BV, BK]
    b_h += b_v[:, None] * b_k[None, :]
    return b_h


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    DEFER_SPEC_STATE: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    # With DEFER_SPEC_STATE, a spec-decode step stores the full state only after
    # its first token (slot 0); the raw inputs of the later tokens are recorded
    # in slot 1, which holds HV * V * K elements, far more than they need. The
    # next step replays the accepted ones, rebuilding the same state that the
    # per-token stores would have kept, for one state store per step instead
    # of one per token.
    STASH_W: tl.constexpr = BK + BV + 2
    o_r = tl.arange(0, BV)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if DEFER_SPEC_STATE:
        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
        if state_idx <= 0:
            return
        p_h0 = h0 + state_idx * stride_init_state_token
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        stash_idx = tl.load(
            ssm_state_indices + i_n * stride_indices_seq + stride_indices_tok
        ).to(tl.int64)
        # Each program records its own copy of k, so programs never share a
        # record and replay reads only what this program wrote.
        p_stash = (
            h0
            + stash_idx * stride_init_state_token
            + (i_hv * tl.num_programs(1) + i_v) * STASH_W
        )
        stash_stride = HV * tl.num_programs(1) * STASH_W
        num_replay = tl.load(num_accepted_tokens + i_n).to(tl.int32) - 1
        for i_r in range(0, num_replay):
            p_r = p_stash + i_r * stash_stride
            b_h = _gated_delta_rule_step(
                b_h,
                tl.load(p_r + o_k, mask=mask_k, other=0).to(tl.float32),
                tl.load(p_r + BK + o_r, mask=mask_v, other=0).to(tl.float32),
                tl.load(p_r + BK + BV).to(tl.float32),
                tl.load(p_r + BK + BV + 1).to(tl.float32),
                b_A_log,
                b_dt_bias,
                beta,
                threshold,
                USE_QK_L2NORM_IN_KERNEL,
                IS_KDA,
            )
        # Replay must finish reading the records before this step's tokens
        # overwrite them.
        tl.debug_barrier()
    elif USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for invalid entries
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                tl.int64
            )
            # Skip if state index is invalid (NULL_BLOCK_ID=0)
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        if DEFER_SPEC_STATE:
            if i_t > 0:
                p_r = p_stash + (i_t - 1) * stash_stride
                tl.store(p_r + o_k, b_k.to(p_r.dtype.element_ty), mask=mask_k)
                tl.store(p_r + BK + o_r, b_v.to(p_r.dtype.element_ty), mask=mask_v)
                tl.store(p_r + BK + BV, b_a.to(p_r.dtype.element_ty))
                tl.store(p_r + BK + BV + 1, b_b.to(p_r.dtype.element_ty))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_q = b_q * scale
        b_h = _gated_delta_rule_step(
            b_h,
            b_k,
            b_v,
            b_a,
            b_b,
            b_A_log,
            b_dt_bias,
            beta,
            threshold,
            USE_QK_L2NORM_IN_KERNEL,
            IS_KDA,
        )
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if DEFER_SPEC_STATE:
            if i_t == 0:
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
        elif INPLACE_FINAL_STATE:
            # Load state index and check for invalid entries
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int64)
            # Only store if state index is valid (not NULL_BLOCK_ID=0)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    defer_spec_state: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating
    computation and the recurrent delta rule update for better performance.

    With defer_spec_state, a spec-decode step keeps the state after its first
    token in slot 0 of each row of ssm_state_indices and records the inputs of
    the other tokens in slot 1, rather than storing a state per token. The
    next step replays num_accepted_tokens - 1 records from slot 1 onto slot 0.
    Every spec-decode step of a request must then use this mode, and nothing
    else may read the per-token slots.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    # The deferred mode does less memory traffic per token, so it is bound by
    # the serial token loop and gains from more, smaller programs. BV fixes the
    # record layout, so it must not change between steps; num_warps may, and a
    # single sequence is too few programs for one warp each.
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 16 if defer_spec_state else 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1 if defer_spec_state and N > 1 else 4

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    if defer_spec_state:
        assert (
            inplace_final_state
            and not is_kda
            and num_accepted_tokens is not None
            and ssm_state_indices is not None
            and ssm_state_indices.ndim == 2
        )
        max_query_len = ssm_state_indices.size(1)
        # Replay matches the per-token states exactly only if the state is
        # never rounded between tokens.
        assert initial_state.dtype == torch.float32
        assert (max_query_len - 1) * NV * (BK + BV + 2) <= V * K, (
            "too many speculative tokens to record in one state slot"
        )

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        DEFER_SPEC_STATE=defer_spec_state,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
