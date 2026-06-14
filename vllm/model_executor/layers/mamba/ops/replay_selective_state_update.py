# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from TensorRT-LLM's replay_selective_state_update.py and the
# state-spaces/mamba selective-state-update Triton kernel.

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    convert_rs_fp16x2,
    softplus,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)}
)
@triton.jit()
def _replay_precompute_kernel(
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    cache_buf_idx_ptr,
    state_batch_indices_ptr,
    null_block_id,
    T: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_dt_batch,
    stride_dt_T,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_T,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    HEADS_PER_BLOCK: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_hg = tl.program_id(axis=1)
    first_head = pid_hg * HEADS_PER_BLOCK

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == null_block_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()

    buf_read = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    buf_write = 1 - buf_read

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    t_mask = offs_t < T
    n_mask = offs_n < dstate
    causal_mask = offs_t[:, None] >= offs_t[None, :]
    valid_mask = causal_mask & t_mask[:, None] & t_mask[None, :]

    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local
        dt_base = dt_ptr + pid_b * stride_dt_batch + head_idx * stride_dt_head
        dt = tl.load(
            dt_base + offs_t * stride_dt_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + head_idx * stride_dt_bias_head).to(
                tl.float32
            )
            dt = dt + dt_bias
        if DT_SOFTPLUS:
            dt = softplus(dt)

        A = tl.load(A_ptr + head_idx * stride_A_head).to(tl.float32)
        dA_cumsum = tl.cumsum(A * dt, axis=0)
        decay_vec = tl.exp(dA_cumsum)

        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + buf_write * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        tl.store(old_dt_base + offs_t * stride_old_dt_T, dt, mask=t_mask)

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_write * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        tl.store(
            old_dA_cumsum_base + offs_t * stride_old_dA_cumsum_T,
            dA_cumsum,
            mask=t_mask,
        )

        decay_vec_base = (
            decay_vec_ptr + pid_b * stride_dv_batch + head_idx * stride_dv_head
        )
        tl.store(decay_vec_base + offs_t * stride_dv_t, decay_vec, mask=t_mask)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    group_idx = first_head // nheads_ngroups_ratio
    C_base = C_ptr + pid_b * stride_C_batch + group_idx * stride_C_group
    B_base = B_ptr + pid_b * stride_B_batch + group_idx * stride_B_group

    C_all = tl.load(
        C_base + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    B_all = tl.load(
        B_base + offs_t[:, None] * stride_B_T + offs_n[None, :] * stride_B_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )

    raw_CB = tl.dot(
        C_all.to(tl.float32),
        tl.trans(B_all.to(tl.float32)),
        input_precision="ieee",
    )

    if first_head % nheads_ngroups_ratio == 0:
        old_B_base = (
            old_B_ptr
            + cache_batch_idx * stride_old_B_cache
            + buf_write * stride_old_B_dbuf
            + group_idx * stride_old_B_group
        )
        tl.store(
            old_B_base
            + offs_t[:, None] * stride_old_B_T
            + offs_n[None, :] * stride_old_B_dstate,
            B_all,
            mask=t_mask[:, None] & n_mask[None, :],
        )

    for h_local in range(HEADS_PER_BLOCK):
        head_idx = first_head + h_local
        old_dt_base = (
            old_dt_ptr
            + cache_batch_idx * stride_old_dt_cache
            + buf_write * stride_old_dt_dbuf
            + head_idx * stride_old_dt_head
        )
        dt = tl.load(
            old_dt_base + offs_t * stride_old_dt_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

        old_dA_cumsum_base = (
            old_dA_cumsum_ptr
            + cache_batch_idx * stride_old_dA_cumsum_cache
            + buf_write * stride_old_dA_cumsum_dbuf
            + head_idx * stride_old_dA_cumsum_head
        )
        dA_cumsum = tl.load(
            old_dA_cumsum_base + offs_t * stride_old_dA_cumsum_T,
            mask=t_mask,
            other=0.0,
        ).to(tl.float32)

        decay_matrix = tl.exp(dA_cumsum[:, None] - dA_cumsum[None, :])
        CB_scaled = tl.where(valid_mask, raw_CB * decay_matrix * dt[None, :], 0.0)

        cb_scaled_base = (
            cb_scaled_ptr + pid_b * stride_cb_batch + head_idx * stride_cb_head
        )
        tl.store(
            cb_scaled_base
            + offs_t[:, None] * stride_cb_t
            + offs_t[None, :] * stride_cb_j,
            CB_scaled,
            mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        )


@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_CACHE_BATCH_INDICES": lambda args: (
            args["state_batch_indices_ptr"] is not None
        )
    }
)
@triton.heuristics({"USE_RS_ROUNDING": lambda args: args["rand_seed_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.heuristics(
    {"BLOCK_SIZE_T": lambda args: max(triton.next_power_of_2(args["T"]), 16)}
)
@triton.jit()
def _replay_state_update_kernel(
    state_ptr,
    old_x_ptr,
    old_B_ptr,
    old_dt_ptr,
    old_dA_cumsum_ptr,
    num_accepted_tokens_ptr,
    cache_buf_idx_ptr,
    replay_valid_ptr,
    x_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    cb_scaled_ptr,
    decay_vec_ptr,
    state_batch_indices_ptr,
    rand_seed_ptr,
    null_block_id,
    T: tl.constexpr,
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_old_x_cache,
    stride_old_x_T,
    stride_old_x_head,
    stride_old_x_dim,
    stride_old_B_cache,
    stride_old_B_dbuf,
    stride_old_B_T,
    stride_old_B_group,
    stride_old_B_dstate,
    stride_old_dt_cache,
    stride_old_dt_dbuf,
    stride_old_dt_head,
    stride_old_dt_T,
    stride_old_dA_cumsum_cache,
    stride_old_dA_cumsum_dbuf,
    stride_old_dA_cumsum_head,
    stride_old_dA_cumsum_T,
    stride_x_batch,
    stride_x_T,
    stride_x_head,
    stride_x_dim,
    stride_C_batch,
    stride_C_T,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_out_batch,
    stride_out_T,
    stride_out_head,
    stride_out_dim,
    stride_cb_batch,
    stride_cb_head,
    stride_cb_t,
    stride_cb_j,
    stride_dv_batch,
    stride_dv_head,
    stride_dv_t,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_CACHE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_RS_ROUNDING: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if HAS_CACHE_BATCH_INDICES:
        cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        if cache_batch_idx == null_block_id:
            return
    else:
        cache_batch_idx = pid_b.to(tl.int64)

    buf_read = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    offs_t = tl.arange(0, BLOCK_SIZE_T)
    m_mask = offs_m < dim
    n_mask = offs_n < dstate
    t_mask = offs_t < T

    state_head_ptr = (
        state_ptr + cache_batch_idx * stride_state_batch + pid_h * stride_state_head
    )
    state_ptrs = (
        state_head_ptr
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state_mask = m_mask[:, None] & n_mask[None, :]
    state = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)

    replay_valid = tl.load(replay_valid_ptr + cache_batch_idx).to(tl.int32)
    prev_num_accepted_tokens = tl.load(num_accepted_tokens_ptr + pid_b)
    prev_num_accepted_tokens = tl.where(
        replay_valid != 0,
        prev_num_accepted_tokens,
        0,
    )

    group_idx = pid_h // nheads_ngroups_ratio
    old_dt_base = (
        old_dt_ptr
        + cache_batch_idx * stride_old_dt_cache
        + buf_read * stride_old_dt_dbuf
        + pid_h * stride_old_dt_head
    )
    old_dt_all = tl.load(
        old_dt_base + offs_t * stride_old_dt_T,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)

    old_dA_cumsum_base = (
        old_dA_cumsum_ptr
        + cache_batch_idx * stride_old_dA_cumsum_cache
        + buf_read * stride_old_dA_cumsum_dbuf
        + pid_h * stride_old_dA_cumsum_head
    )
    old_dA_cumsum_all = tl.load(
        old_dA_cumsum_base + offs_t * stride_old_dA_cumsum_T,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)

    prev_k_idx = tl.minimum(tl.maximum(prev_num_accepted_tokens - 1, 0), T - 1)
    total_dA_cumsum = tl.load(
        old_dA_cumsum_base + prev_k_idx * stride_old_dA_cumsum_T
    ).to(tl.float32)

    coeff = tl.exp(total_dA_cumsum - old_dA_cumsum_all) * old_dt_all
    accepted_mask = t_mask & (offs_t < prev_num_accepted_tokens)
    coeff = tl.where(accepted_mask, coeff, 0.0)

    old_x_base = (
        old_x_ptr + cache_batch_idx * stride_old_x_cache + pid_h * stride_old_x_head
    )
    old_x_all = tl.load(
        old_x_base
        + offs_t[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        mask=accepted_mask[:, None] & m_mask[None, :],
        other=0.0,
    )

    old_B_base = (
        old_B_ptr
        + cache_batch_idx * stride_old_B_cache
        + buf_read * stride_old_B_dbuf
        + group_idx * stride_old_B_group
    )
    old_B_all = tl.load(
        old_B_base
        + offs_t[:, None] * stride_old_B_T
        + offs_n[None, :] * stride_old_B_dstate,
        mask=accepted_mask[:, None] & n_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    dB_scaled = coeff[:, None] * old_B_all
    total_decay = tl.where(
        prev_num_accepted_tokens > 0,
        tl.exp(total_dA_cumsum),
        1.0,
    )
    state *= total_decay
    state += tl.dot(
        tl.trans(old_x_all.to(tl.float32)),
        dB_scaled.to(tl.float32),
        input_precision="ieee",
    )

    if USE_RS_ROUNDING:
        rand_seed = tl.load(rand_seed_ptr + cache_batch_idx)
        rand_offsets = (
            cache_batch_idx * stride_state_batch
            + pid_h * stride_state_head
            + offs_m[:, None] * stride_state_dim
            + offs_n[None, :] * stride_state_dstate
        )
        if PHILOX_ROUNDS > 0:
            rand = tl.randint(rand_seed, rand_offsets, PHILOX_ROUNDS)
        else:
            rand = tl.randint(rand_seed, rand_offsets)
        state = convert_rs_fp16x2(state, rand)
        tl.store(state_ptrs, state, mask=state_mask)
        state = state.to(tl.float32)
    else:
        state = state.to(state_ptrs.dtype.element_ty)
        tl.store(state_ptrs, state, mask=state_mask)
        state = state.to(tl.float32)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    x_base = x_ptr + pid_b * stride_x_batch + pid_h * stride_x_head
    C_base = C_ptr + pid_b * stride_C_batch + group_idx * stride_C_group
    out_base = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head

    C_all = tl.load(
        C_base + offs_t[:, None] * stride_C_T + offs_n[None, :] * stride_C_dstate,
        mask=t_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    x_all = tl.load(
        x_base + offs_t[:, None] * stride_x_T + offs_m[None, :] * stride_x_dim,
        mask=t_mask[:, None] & m_mask[None, :],
        other=0.0,
    )
    tl.store(
        old_x_base
        + offs_t[:, None] * stride_old_x_T
        + offs_m[None, :] * stride_old_x_dim,
        x_all,
        mask=t_mask[:, None] & m_mask[None, :],
    )
    x_all = x_all.to(tl.float32)

    cb_scaled_base = cb_scaled_ptr + pid_b * stride_cb_batch + pid_h * stride_cb_head
    CB_scaled = tl.load(
        cb_scaled_base + offs_t[:, None] * stride_cb_t + offs_t[None, :] * stride_cb_j,
        mask=(offs_t[:, None] < BLOCK_SIZE_T) & (offs_t[None, :] < BLOCK_SIZE_T),
        other=0.0,
    ).to(tl.float32)

    decay_vec_base = decay_vec_ptr + pid_b * stride_dv_batch + pid_h * stride_dv_head
    decay_vec = tl.load(
        decay_vec_base + offs_t * stride_dv_t,
        mask=t_mask,
        other=0.0,
    ).to(tl.float32)

    init_out = (
        tl.dot(
            C_all.to(tl.float32),
            tl.trans(state.to(tl.float32)),
            input_precision="ieee",
        )
        * decay_vec[:, None]
    )
    cb_out = tl.dot(
        CB_scaled.to(tl.float32),
        x_all.to(tl.float32),
        input_precision="ieee",
    )
    out_all = init_out + cb_out

    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim,
            mask=m_mask,
            other=0.0,
        ).to(tl.float32)
        out_all = out_all + x_all * D[None, :]

    out_ptrs = (
        out_base + offs_t[:, None] * stride_out_T + offs_m[None, :] * stride_out_dim
    )
    tl.store(out_ptrs, out_all, mask=t_mask[:, None] & m_mask[None, :])


@triton.jit()
def _commit_replay_cache_kernel(
    cache_buf_idx_ptr,
    replay_valid_ptr,
    state_batch_indices_ptr,
    null_block_id,
):
    pid_b = tl.program_id(axis=0)
    cache_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
    if cache_batch_idx == null_block_id:
        return
    buf_read = tl.load(cache_buf_idx_ptr + cache_batch_idx).to(tl.int32)
    tl.store(cache_buf_idx_ptr + cache_batch_idx, 1 - buf_read)
    tl.store(replay_valid_ptr + cache_batch_idx, 1)


def _get_replay_launch_config(
    *,
    batch: int,
    nheads: int,
    ngroups: int,
    block_size_t: int,
    state_dtype: torch.dtype,
    use_philox: bool,
) -> tuple[int, int, int, int]:
    total_heads = batch * nheads
    heads_per_group = nheads // ngroups
    state_is_16bit = state_dtype in (torch.float16, torch.bfloat16)

    if block_size_t <= 16:
        if use_philox and state_is_16bit:
            if total_heads <= 16:
                return 4, 4, 4, 1
            if total_heads <= 512:
                return 8, 1, 2, 1
            return 32, 4, 2, 1
        if state_is_16bit:
            if total_heads <= 16:
                return 32, 4, 4, 1
            if total_heads <= 64:
                return 8, 1, 2, 1
            if total_heads <= 256:
                return 16, 2, 2, 1
            if total_heads <= 512:
                return 32, 1, 1, min(2, heads_per_group)
            return 32, 4, 2, 1
        if total_heads <= 32:
            return 8, 1, 4, 1
        if total_heads <= 64:
            return 8, 1, 2, 1
        if total_heads <= 128:
            return 8, 2, 2, 1
        if total_heads <= 256:
            return 16, 1, 2, 1
        if total_heads <= 512:
            return 64, 2, 2, min(2, heads_per_group)
        return 32, 4, 2, 1

    if state_is_16bit:
        if total_heads <= 128:
            return 16, 2, 4, 1
        if total_heads <= 256:
            return 16, 1, 4, min(2, heads_per_group)
        if total_heads <= 512:
            return 32, 1, 1, min(4, heads_per_group)
        return 32, 1, 4, min(2, heads_per_group)
    if total_heads <= 128:
        return 16, 2, 4, 1
    if total_heads <= 256:
        return 32, 2, 4, min(2, heads_per_group)
    if total_heads <= 512:
        return 64, 2, 2, min(4, heads_per_group)
    return 64, 2, 4, min(2, heads_per_group)


def replay_selective_state_update(
    state: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt: torch.Tensor,
    old_dA_cumsum: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    replay_valid: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    enable_stochastic_rounding: bool = False,
    cache_philox_rounds: int = 0,
    cb_scaled: torch.Tensor | None = None,
    decay_vec: torch.Tensor | None = None,
    launch_with_pdl: bool = False,
    use_internal_pdl: bool = True,
) -> None:
    """Replay-based MTP SSM update.

    ``state`` is advanced only to the end of the previous accepted step. The
    current step's token inputs are stored in compact replay buffers, then the
    next invocation uses ``num_accepted_tokens`` to replay only the accepted
    prefix. This avoids writing one full SSM checkpoint per speculative token.
    """
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    cache_size, nheads, dim, dstate = state.shape
    batch, T, _, _ = x.shape
    ngroups = B.shape[2]
    assert nheads % ngroups == 0

    assert x.shape == (batch, T, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    assert B.shape == (batch, T, ngroups, dstate)
    assert C.shape == B.shape
    assert old_x.shape == (cache_size, T, nheads, dim)
    assert old_B.shape == (cache_size, 2, T, ngroups, dstate)
    assert old_dt.shape == (cache_size, 2, nheads, T)
    assert old_dA_cumsum.shape == (cache_size, 2, nheads, T)
    assert cache_buf_idx.shape == (cache_size,)
    assert replay_valid.shape == (cache_size,)
    assert num_accepted_tokens.shape == (batch,)
    assert out.shape == x.shape
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    if D is not None:
        assert D.shape == (nheads, dim)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is None or dt_bias.stride(-1) == 0)
    )
    assert tie_hdim

    pdl_supported = (
        current_platform.is_cuda() and current_platform.has_device_capability(90)
    )
    launch_with_pdl = launch_with_pdl and pdl_supported
    use_internal_pdl = use_internal_pdl and pdl_supported

    block_size_t = max(triton.next_power_of_2(T), 16)
    rand_seed = None
    if enable_stochastic_rounding:
        assert state.dtype == torch.float16
        rand_seed = torch.randint(
            0,
            2**32,
            (cache_size,),
            dtype=torch.int64,
            device=state.device,
        )

    (
        block_size_m,
        num_warps,
        precompute_num_warps,
        heads_per_block,
    ) = _get_replay_launch_config(
        batch=batch,
        nheads=nheads,
        ngroups=ngroups,
        block_size_t=block_size_t,
        state_dtype=state.dtype,
        use_philox=rand_seed is not None,
    )

    heads_per_group = nheads // ngroups
    while heads_per_group % heads_per_block != 0:
        heads_per_block -= 1
    assert nheads % heads_per_block == 0
    assert heads_per_block <= heads_per_group

    if cb_scaled is None:
        cb_scaled = torch.empty(
            batch,
            nheads,
            block_size_t,
            block_size_t,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        assert cb_scaled.shape[0] >= batch
        assert cb_scaled.shape[1:] == (nheads, block_size_t, block_size_t)
        assert cb_scaled.dtype == torch.float32
        assert cb_scaled.device == x.device
        cb_scaled = cb_scaled[:batch]
    if decay_vec is None:
        decay_vec = torch.empty(
            batch,
            nheads,
            block_size_t,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        assert decay_vec.shape[0] >= batch
        assert decay_vec.shape[1:] == (nheads, block_size_t)
        assert decay_vec.dtype == torch.float32
        assert decay_vec.device == x.device
        decay_vec = decay_vec[:batch]

    has_state_indices = state_batch_indices is not None
    D_strides = (D.stride(0), D.stride(1)) if D is not None else (0, 0)
    with torch.accelerator.device_index(x.device.index):
        _replay_precompute_kernel[(batch, nheads // heads_per_block)](
            dt,
            dt_bias,
            A,
            B,
            C,
            cb_scaled,
            decay_vec,
            old_B,
            old_dt,
            old_dA_cumsum,
            cache_buf_idx,
            state_batch_indices,
            null_block_id,
            T,
            dstate,
            nheads // ngroups,
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt.stride(3),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_bias.stride(1) if dt_bias is not None else 0,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            cb_scaled.stride(0),
            cb_scaled.stride(1),
            cb_scaled.stride(2),
            cb_scaled.stride(3),
            decay_vec.stride(0),
            decay_vec.stride(1),
            decay_vec.stride(2),
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            old_B.stride(4),
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_dt.stride(3),
            old_dA_cumsum.stride(0),
            old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2),
            old_dA_cumsum.stride(3),
            dt_softplus,
            HAS_CACHE_BATCH_INDICES=has_state_indices,
            HEADS_PER_BLOCK=heads_per_block,
            LAUNCH_WITH_PDL=launch_with_pdl,
            LAUNCH_DEPENDENT_KERNELS=use_internal_pdl,
            launch_pdl=launch_with_pdl or use_internal_pdl,
            num_warps=precompute_num_warps,
        )

        def grid(meta):
            return (triton.cdiv(dim, meta["BLOCK_SIZE_M"]), batch, nheads)

        _replay_state_update_kernel[grid](
            state,
            old_x,
            old_B,
            old_dt,
            old_dA_cumsum,
            num_accepted_tokens,
            cache_buf_idx,
            replay_valid,
            x,
            C,
            D,
            out,
            cb_scaled,
            decay_vec,
            state_batch_indices,
            rand_seed,
            null_block_id,
            T,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            old_x.stride(0),
            old_x.stride(1),
            old_x.stride(2),
            old_x.stride(3),
            old_B.stride(0),
            old_B.stride(1),
            old_B.stride(2),
            old_B.stride(3),
            old_B.stride(4),
            old_dt.stride(0),
            old_dt.stride(1),
            old_dt.stride(2),
            old_dt.stride(3),
            old_dA_cumsum.stride(0),
            old_dA_cumsum.stride(1),
            old_dA_cumsum.stride(2),
            old_dA_cumsum.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            C.stride(3),
            *D_strides,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            cb_scaled.stride(0),
            cb_scaled.stride(1),
            cb_scaled.stride(2),
            cb_scaled.stride(3),
            decay_vec.stride(0),
            decay_vec.stride(1),
            decay_vec.stride(2),
            block_size_m,
            PHILOX_ROUNDS=cache_philox_rounds,
            LAUNCH_WITH_PDL=use_internal_pdl,
            launch_pdl=use_internal_pdl,
            num_warps=num_warps,
        )

        if state_batch_indices is not None:
            _commit_replay_cache_kernel[(batch,)](
                cache_buf_idx,
                replay_valid,
                state_batch_indices,
                null_block_id,
            )
        else:
            cache_buf_idx[:batch] = 1 - cache_buf_idx[:batch]
            replay_valid[:batch] = 1
