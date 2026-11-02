# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import num_compute_units


def _tree_tile(V: int, heads: int, sms: int) -> tuple[int, int]:
    """(BV, num_warps) of a tree verify over ``heads`` (sequence, value head)
    pairs on ``sms`` SMs.

    A verify walks its rows one after another, so a program's latency per row
    sets the launch time: narrow V tiles on one warp give the shortest rows.
    4-wide tiles while the grid has at most 12 programs per SM; past that the
    q/k work every program repeats costs more than the shorter rows save. The
    tile sets the order of the fp32 sums, so a tree's bits depend on which side
    of that line its batch falls; a path has the bits of the same tile run on
    it alone.
    """
    if heads * triton.cdiv(V, 4) <= 12 * sms:
        return min(triton.next_power_of_2(V), 4), 1
    return min(triton.next_power_of_2(V), 8), 1


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

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
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

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        # compute beta_output = sigmoid(b)
        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
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
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
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


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_tree_kernel(
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
    state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    parents,
    scale,
    T: tl.int64,  # num of tokens
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """The tree verify of fused_sigmoid_gating_delta_rule_update_kernel, for a
    spec-decode batch that updates ``state`` in place. A program walks one
    (sequence, value head, V tile)'s rows in order; a row starts from its
    parent's state."""
    i_v, i_nh = tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    T = eos - bos
    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_idx = ssm_state_indices + i_n * stride_indices_seq

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    state_idx = tl.load(p_idx + i_t).to(tl.int64)
    # Skip if state index is invalid (NULL_BLOCK_ID=0)
    if state_idx <= 0:
        return
    p_h0 = state + state_idx * stride_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if launch_pdl:
        # Everything above (the sequence bounds, slots and initial state)
        # loads while the kernel before this one finishes: no kernel of the
        # layer's forward writes them. q/k/v/a/b load below the wait.
        tl.extra.cuda.gdc_wait()

    # The rows run one after another, so whatever a row computes off the
    # state chain is latency the row before can hide. Row t + 1's q/k norms,
    # gate and beta are computed in row t, next to row t's state math, from
    # inputs loaded in row t - 1; its parent and state slot load in row t.
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    c_q, c_k, c_v, c_eg, c_beta = _prepare_row(
        tl.load(p_q, mask=mask_k, other=0),
        tl.load(p_k, mask=mask_k, other=0),
        tl.load(p_v, mask=mask_v, other=0),
        tl.load(p_a),
        tl.load(p_b),
        b_A_log,
        b_dt_bias,
        beta,
        threshold,
        scale,
        USE_QK_L2NORM_IN_KERNEL,
    )
    n_q = tl.load(p_q + H * K, mask=mask_k & (T > 1), other=0)
    n_k = tl.load(p_k + H * K, mask=mask_k & (T > 1), other=0)
    n_v = tl.load(p_v + HV * V, mask=mask_v & (T > 1), other=0)
    n_a = tl.load(p_a + HV, mask=T > 1, other=0)
    n_b = tl.load(p_b + HV, mask=T > 1, other=0)
    n_parent = tl.load(parents + bos).to(tl.int64)
    n_state_idx = tl.load(p_idx).to(tl.int64)

    for i_t in range(0, T):
        b_q, b_k, b_v, b_eg, b_beta = c_q, c_k, c_v, c_eg, c_beta
        r_q, r_k, r_v, r_a, r_b = n_q, n_k, n_v, n_a, n_b
        # row t + 2's inputs
        has_next2 = i_t + 2 < T
        n_q = tl.load(p_q + 2 * H * K, mask=mask_k & has_next2, other=0)
        n_k = tl.load(p_k + 2 * H * K, mask=mask_k & has_next2, other=0)
        n_v = tl.load(p_v + 2 * HV * V, mask=mask_v & has_next2, other=0)
        n_a = tl.load(p_a + 2 * HV, mask=has_next2, other=0)
        n_b = tl.load(p_b + 2 * HV, mask=has_next2, other=0)
        has_next = i_t + 1 < T
        final_state_idx = n_state_idx
        n_state_idx = tl.load(p_idx + i_t + 1, mask=has_next, other=0).to(tl.int64)
        # A row whose parent is not the row before it starts from the
        # parent's state, which this program stored at the parent's step.
        i_p = n_parent
        n_parent = tl.load(parents + bos + i_t + 1, mask=has_next, other=0).to(
            tl.int64
        )
        if i_p != i_t - 1:  # the first row's parent is -1
            tl.debug_barrier()
            parent_state_idx = tl.load(p_idx + i_p).to(tl.int64)
            p_hp = state + parent_state_idx * stride_state_token
            p_hp = p_hp + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            b_h = tl.load(p_hp, mask=mask_h, other=0).to(tl.float32)

        # [BV, BK]
        b_h *= b_eg
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        # row t + 1, off the state chain (zeros past the last row)
        c_q, c_k, c_v, c_eg, c_beta = _prepare_row(
            r_q,
            r_k,
            r_v,
            r_a,
            r_b,
            b_A_log,
            b_dt_bias,
            beta,
            threshold,
            scale,
            USE_QK_L2NORM_IN_KERNEL,
        )
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Only store if state index is valid (not NULL_BLOCK_ID=0)
        if final_state_idx > 0:
            p_ht = state + final_state_idx * stride_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


@triton.jit
def _prepare_row(
    b_q,
    b_k,
    b_v,
    b_a,
    b_b,
    b_A_log,
    b_dt_bias,
    beta,
    threshold,
    scale,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """A row's state-independent terms from its loaded inputs: q (normed and
    scaled), k (normed), v, exp(gate) and beta, in fp32."""
    b_q = b_q.to(tl.float32)
    b_k = b_k.to(tl.float32)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = b_a.to(tl.float32) + b_dt_bias
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    # compute beta_output = sigmoid(b)
    b_beta = tl.sigmoid(b_b.to(tl.float32))
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
    b_q = b_q * scale
    return b_q, b_k, b_v.to(tl.float32), tl.exp(b_g), b_beta


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
    parents: torch.Tensor | None = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating
    computation and the recurrent delta rule update for better performance.

    ``parents`` (``[T]`` int32) makes it a tree verify of spec-decode rows:
    each row's parent as a row index within its sequence, -1 for the first row,
    every other row's parent an earlier row. A row starts from its parent's
    state instead of the previous row's. Each row's state is stored in
    ``ssm_state_indices[seq, row]`` as for a chain; a row whose parent is not
    the row before it reloads that stored state, so it runs at the state
    dtype's precision. None is a chain.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

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

    if parents is not None:
        if (
            cu_seqlens is None
            or ssm_state_indices is None
            or num_accepted_tokens is None
            or not inplace_final_state
            or is_kda
        ):
            raise ValueError(
                "a tree update (`parents`) verifies spec-decode rows: it needs "
                "`cu_seqlens`, `ssm_state_indices` and `num_accepted_tokens`, "
                "updates the state in place and takes GDN gates"
            )
        o = q.new_empty(v.shape)
        BV, num_warps = _tree_tile(
            V, N * HV, num_compute_units(q.device.index or 0) if q.is_cuda else 0
        )
        fused_sigmoid_gating_delta_rule_tree_kernel[(1, triton.cdiv(V, BV), N * HV)](
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
            state=initial_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            parents=parents,
            scale=scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            stride_state_token=initial_state.stride(0),
            stride_indices_seq=ssm_state_indices.stride(0),
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            # PDL: the initial state loads under the tail of the conv
            launch_pdl=current_platform.is_arch_support_pdl(),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return o, initial_state

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
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
