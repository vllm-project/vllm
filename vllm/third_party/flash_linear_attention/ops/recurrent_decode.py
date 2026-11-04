# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
"""One recurrent GDN decode kernel for every decode path.

Speculative verification (:func:`fused_sigmoid_gating_delta_rule_update`) and
the non-speculative decode fast path
(:func:`fused_recurrent_gated_delta_rule_packed_decode`) advance the same
recurrent state over the same token, so they must produce bitwise identical
results. Both now run the same kernel body and the same tile configuration;
the remaining per-path differences are base pointers, their strides and
control-flow predicates such as ``IS_VARLEN``, not arithmetic.

That is deliberate. Two kernels with identical source-level math can still pick
a different reduction order or contract different FMAs once their tile shape or
warp count changes; that is the few-ULP residual the sm_110 review on #65338
reported. Splitting normalization and gating out of the packed kernel, and
keeping the tile shape out of each path's control, removes the two ways the
recurrence used to drift: a different formula, and a different tiling. For the
single-token specializations the two entry points emit the same floating-point
operation sequence on sm_110 in both TTGIR and PTX.
"""

import functools

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .recurrent_common import l2norm_qk, recurrent_step, sigmoid_gating


@triton.jit(
    do_not_specialize=[
        "stride_indices_seq",
        "stride_indices_tok",
    ]
)
def gated_delta_rule_decode_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
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
    stride_q_tok: tl.constexpr,
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    stride_o_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq,
    stride_indices_tok,
    T: tl.constexpr,  # num of tokens per sequence (unused when IS_VARLEN)
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    SPLIT_BATCH_HEAD_GRID: tl.constexpr,
):
    if SPLIT_BATCH_HEAD_GRID:
        i_v, i_hv, i_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    else:
        i_v, i_nh = tl.program_id(0), tl.program_id(1)
        i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        num_steps = tl.load(cu_seqlens + i_n + 1).to(tl.int64) - bos
    else:
        bos = i_n * T
        num_steps = T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_o = o + bos * stride_o_tok + i_hv * V + o_v

    if num_steps == 0:
        # no tokens to process for this sequence
        return

    p_q = q + bos * stride_q_tok + i_h * K + o_k
    p_k = k + bos * stride_k_tok + i_h * K + o_k
    p_v = v + bos * stride_v_tok + i_hv * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * stride_a_tok + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + bos * stride_a_tok + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    p_b = b + bos * stride_b_tok + i_hv

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
            # Skip if state index is invalid (NULL_BLOCK_ID=0)
            state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if state_idx <= 0:
                # Padded slots are never read back; keep the output finite so a
                # stray consumer cannot see uninitialized memory.
                tl.store(
                    p_o,
                    tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty),
                    mask=mask_v,
                )
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, num_steps):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        b_g, b_beta = sigmoid_gating(
            tl.load(p_a).to(tl.float32),
            tl.load(p_b).to(tl.float32),
            tl.load(p_A_log).to(tl.float32),
            tl.load(p_dt_bias).to(tl.float32),
            beta,
            threshold,
        )

        if USE_QK_L2NORM_IN_KERNEL:
            b_q, b_k = l2norm_qk(b_q, b_k, 0)
        b_q = b_q * scale
        b_h, b_o = recurrent_step(b_h, b_q, b_k, b_v, b_g, b_beta, IS_KDA)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
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
        p_q += stride_q_tok
        p_k += stride_k_tok
        p_v += stride_v_tok
        p_o += stride_o_tok
        p_b += stride_b_tok
        p_a += stride_a_tok


@functools.lru_cache(maxsize=128)
def _select_decode_config_cached(
    K: int, V: int, state_dtype: torch.dtype
) -> tuple[int, int, int, int]:
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    if K == V == 128 and state_dtype == torch.float32 and current_platform.is_cuda():
        # The earlier BV choices were tuned for the one-warp packed kernel.
        # This kernel uses four warps, so the same per-thread tile (and
        # register pressure) needs 4x the BV: the tuned BV=4 becomes 16.
        # Measured on H100, BV=16 is also neutral for nine-token verification,
        # whereas BV=4 costs ~1.5x there. BF16 keeps 32.
        BV = 16
    return BK, BV, 4, 3


def select_decode_config(
    K: int,
    V: int,
    state_dtype: torch.dtype,
) -> tuple[int, int, int, int]:
    """Return ``(BK, BV, num_warps, num_stages)`` for the decode recurrence.

    Every decode path must call this, so a request cannot be advanced by two
    different tile shapes depending on whether the current step is speculative
    or not. The result deliberately does not depend on the batch size: a
    request can be advanced alone in a one-request verification batch and then
    with many requests in a plain decode batch, and BV changes the FP32
    reduction order. ``test_decode_paths_share_one_kernel_config`` enforces the
    shared choice.
    """
    return _select_decode_config_cached(K, V, state_dtype)


def gated_delta_rule_decode(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    scale: float,
    num_seqs: int,
    num_tokens: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    beta: float = 1.0,
    threshold: float = 20.0,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    inplace_final_state: bool = True,
    stride_q_tok: int | None = None,
    stride_k_tok: int | None = None,
    stride_v_tok: int | None = None,
    stride_o_tok: int | None = None,
) -> None:
    """Launch the shared decode recurrence.

    ``q``, ``k`` and ``v`` are base pointers with their token stride given
    explicitly, so the packed fast path can hand over three views of one
    ``mixed_qkv`` row without materializing separate tensors.
    """
    BK, BV, num_warps, num_stages = select_decode_config(K, V, initial_state.dtype)
    # ``(B, T, H, D)`` and ``(NK, B, T, HV, V)`` keep the timestep three
    # dimensions from the back; a packed qkv row is two-dimensional.
    if stride_q_tok is None:
        stride_q_tok = q.stride(-3) if q.dim() >= 3 else q.stride(0)
    if stride_k_tok is None:
        stride_k_tok = k.stride(-3) if k.dim() >= 3 else k.stride(0)
    if stride_v_tok is None:
        stride_v_tok = v.stride(-3) if v.dim() >= 3 else v.stride(0)
    if stride_o_tok is None:
        stride_o_tok = o.stride(-3) if o.dim() >= 3 else o.stride(0)

    # a/b are indexed by the flattened timestep, so their row stride is dim 1
    # for (B, T, HV) / (B, T, HV, K) inputs and dim 0 for (tokens, HV).
    stride_a_tok = a.stride(1) if a.dim() >= 3 else a.stride(0)
    stride_b_tok = b.stride(1) if b.dim() >= 3 else b.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    # ``T`` is a constexpr specialization constant and is only read by the
    # non-varlen path, so pin it under varlen to keep every sequence length on
    # the same compiled kernel.
    if cu_seqlens is not None:
        num_tokens = 1
    stride_init_state_token = initial_state.stride(0)
    if final_state is initial_state:
        stride_final_state_token = stride_init_state_token
    else:
        stride_final_state_token = final_state.stride(0)

    # CUDA limits grid Y/Z dimensions to 65535.
    split_batch_head_grid = num_seqs * HV > 65535
    NV = (V + BV - 1) // BV
    grid = (NV, HV, num_seqs) if split_batch_head_grid else (NV, num_seqs * HV)
    # Positional launch: this runs once per GDN layer per decode step, and
    # Triton binds keyword arguments far more slowly than positional ones.
    gated_delta_rule_decode_kernel[grid](
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
        initial_state,
        final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        stride_q_tok,
        stride_k_tok,
        stride_v_tok,
        stride_o_tok,
        stride_a_tok,
        stride_b_tok,
        stride_init_state_token,
        stride_final_state_token,
        stride_indices_seq,
        stride_indices_tok,
        num_tokens,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        initial_state is not None,
        inplace_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens is not None,
        ssm_state_indices is not None,
        num_accepted_tokens is not None,
        is_kda,
        split_batch_head_grid,
        num_warps=num_warps,
        num_stages=num_stages,
    )
