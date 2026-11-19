# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton Mamba2 ReplaySSM kernels for speculative decoding.

These follow the ring contract of FlashInfer's `checkpointing_ssu` and
`replayssm_materialize`, so the ReplaySSM trackers in
`vllm.v1.worker.replayssm_utils` drive both backends.
"""

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _replay_ring(
    S,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    ring_start,
    num_tokens,
    A,
    offs_m,
    offs_n,
    dim,
    dstate,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_pos,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    RING_BUFFER_LEN: tl.constexpr,
):
    for j in range(num_tokens):
        row = (ring_start + j) % RING_BUFFER_LEN
        x = tl.load(
            x_cache_ptr + row * stride_x_cache_pos + offs_m * stride_x_cache_dim,
            mask=offs_m < dim,
            other=0.0,
        ).to(tl.float32)
        B = tl.load(
            B_cache_ptr + row * stride_B_cache_pos + offs_n * stride_B_cache_dstate,
            mask=offs_n < dstate,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(dt_cache_ptr + row * stride_dt_cache_pos)
        S = S * tl.exp(A * dt) + (dt * x)[:, None] * B[None, :]
    return S


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["batch"])
def _replayssm_spec_kernel(
    state_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    ring_start_ptr,
    prev_num_accepted_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    cu_seqlens_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    batch,
    dim,
    dstate,
    nheads_ngroups_ratio,
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_cache_batch,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_batch,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_batch,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_x_token,
    stride_x_head,
    stride_x_dim,
    stride_dt_token,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_token,
    stride_B_group,
    stride_B_dstate,
    stride_C_token,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    DT_SOFTPLUS: tl.constexpr,
    MAX_WINDOW: tl.constexpr,
    RING_BUFFER_LEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    slot = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
    if slot == pad_slot_id:
        return
    bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
    seqlen = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64) - bos
    ring_start = tl.load(ring_start_ptr + slot)
    num_replay = tl.load(prev_num_accepted_ptr + slot)

    group = pid_h // nheads_ngroups_ratio
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    mask_m = offs_m < dim
    mask_n = offs_n < dstate

    state_ptr += slot * stride_state_batch + pid_h * stride_state_head
    x_cache_ptr += slot * stride_x_cache_batch + pid_h * stride_x_cache_head
    dt_cache_ptr += slot * stride_dt_cache_batch + pid_h * stride_dt_cache_head
    B_cache_ptr += slot * stride_B_cache_batch + group * stride_B_cache_group
    x_ptr += pid_h * stride_x_head
    dt_ptr += pid_h * stride_dt_head
    B_ptr += group * stride_B_group
    C_ptr += group * stride_C_group
    out_ptr += pid_h * stride_out_head

    state_ptrs = (
        state_ptr
        + offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    state_mask = mask_m[:, None] & mask_n[None, :]
    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    S = tl.load(state_ptrs, mask=state_mask, other=0.0).to(tl.float32)

    # Fast-forward the checkpoint over the tokens accepted since it was taken.
    S = _replay_ring(
        S,
        x_cache_ptr,
        dt_cache_ptr,
        B_cache_ptr,
        ring_start,
        num_replay,
        A,
        offs_m,
        offs_n,
        dim,
        dstate,
        stride_x_cache_pos,
        stride_x_cache_dim,
        stride_dt_cache_pos,
        stride_B_cache_pos,
        stride_B_cache_dstate,
        RING_BUFFER_LEN,
    )
    # Checkpoint when this step's tokens would overflow the replay window; the
    # trackers then restart the ring after the replayed tokens.
    if num_replay + seqlen > MAX_WINDOW:
        tl.store(state_ptrs, S.to(state_ptr.dtype.element_ty), mask=state_mask)

    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    if HAS_D:
        D = tl.load(
            D_ptr + pid_h * stride_D_head + offs_m * stride_D_dim,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
    writes_B = (pid_m == 0) & (pid_h % nheads_ngroups_ratio == 0)
    for t in range(seqlen):
        token = bos + t
        row = (ring_start + num_replay + t) % RING_BUFFER_LEN
        x = tl.load(x_ptr + token * stride_x_token + offs_m * stride_x_dim, mask=mask_m)
        B = tl.load(
            B_ptr + token * stride_B_token + offs_n * stride_B_dstate,
            mask=mask_n,
            other=0.0,
        )
        C = tl.load(
            C_ptr + token * stride_C_token + offs_n * stride_C_dstate,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        dt = tl.load(dt_ptr + token * stride_dt_token).to(tl.float32)
        if HAS_DT_BIAS:
            dt += dt_bias
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        x_f = x.to(tl.float32)
        S = S * tl.exp(A * dt) + (dt * x_f)[:, None] * B.to(tl.float32)[None, :]
        out = tl.sum(S * C[None, :], axis=1)
        if HAS_D:
            out += x_f * D
        tl.store(
            out_ptr + token * stride_out_token + offs_m * stride_out_dim,
            out,
            mask=mask_m,
        )

        # Append every verified token; the trackers commit only accepted ones.
        tl.store(
            x_cache_ptr + row * stride_x_cache_pos + offs_m * stride_x_cache_dim,
            x,
            mask=mask_m,
        )
        if pid_m == 0:
            tl.store(dt_cache_ptr + row * stride_dt_cache_pos, dt)
        if writes_B:
            tl.store(
                B_cache_ptr + row * stride_B_cache_pos + offs_n * stride_B_cache_dstate,
                B,
                mask=mask_n,
            )


def selective_state_update_replayssm_spec(
    state: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    state_batch_indices: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> torch.Tensor:
    """Verify a packed batch of speculative windows against ReplaySSM rings.

    Each sequence first replays the `prev_num_accepted_tokens[slot]` ring
    entries starting at `ring_start[slot]` onto its checkpoint `state`,
    stores that state as the new checkpoint when `prev + seqlen` exceeds
    `max_window = ring_len - max_seqlen`, then scans its own tokens and
    appends them to the ring after the replayed entries.

    Args:
        state: (cache, nheads, dim, dstate) checkpoints, updated in place.
        x_cache: (cache, nheads, ring_len, dim) ring of x.
        B_cache: (cache, ngroups, ring_len, dstate) ring of B.
        dt_cache: (cache, nheads, ring_len) fp32 ring of processed dt.
        ring_start: (cache,) int32 oldest live ring row.
        prev_num_accepted_tokens: (cache,) int32 ring entries to replay.
        x: (total_tokens, nheads, dim) packed tokens.
        dt: (total_tokens, nheads, dim) with stride(-1) == 0.
        A: (nheads, dim, dstate) with stride(-1) == stride(-2) == 0.
        B: (total_tokens, ngroups, dstate).
        C: (total_tokens, ngroups, dstate).
        out: (total_tokens, nheads, dim) output.
        cu_seqlens: (batch + 1,) sequence boundaries in the packed tokens.
        max_seqlen: Upper bound on every sequence length.
        state_batch_indices: (batch,) cache slot per sequence.
        D: Optional (nheads, dim) skip connection.
        dt_bias: Optional (nheads, dim) with stride(-1) == 0.
        dt_softplus: Apply softplus to `dt + dt_bias`.
        pad_slot_id: Slot index of padded sequences, which are skipped.

    Returns:
        `out`.

    """
    _, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ring_len = x_cache.shape[2]
    max_window = ring_len - max_seqlen
    batch = cu_seqlens.numel() - 1
    assert max_window >= max_seqlen
    assert x.shape[1:] == (nheads, dim) and out.shape == x.shape
    assert nheads % ngroups == 0
    assert B.shape[1:] == (ngroups, dstate) and C.shape == B.shape
    assert x_cache.shape[1:] == (nheads, ring_len, dim)
    assert B_cache.shape[1:] == (ngroups, ring_len, dstate)
    assert dt_cache.shape[1:] == (nheads, ring_len)
    assert dt_cache.dtype == torch.float32
    assert dt.stride(-1) == 0 and A.stride(-1) == 0 and A.stride(-2) == 0
    assert dt_bias is None or dt_bias.stride(-1) == 0
    assert state_batch_indices.dim() == 1
    assert state_batch_indices.shape[0] >= batch
    if batch == 0:
        return out

    block_size_m = min(triton.next_power_of_2(dim), 16)
    grid = (triton.cdiv(dim, block_size_m), batch, nheads)
    D_strides = (D.stride(0), D.stride(1)) if D is not None else (0, 0)
    with torch.accelerator.device_index(x.device.index):
        _replayssm_spec_kernel[grid](
            state,
            x_cache,
            dt_cache,
            B_cache,
            ring_start,
            prev_num_accepted_tokens,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            out,
            cu_seqlens,
            state_batch_indices,
            pad_slot_id,
            batch,
            dim,
            dstate,
            nheads // ngroups,
            *state.stride(),
            x_cache.stride(0),
            x_cache.stride(1),
            x_cache.stride(2),
            x_cache.stride(3),
            *dt_cache.stride(),
            *B_cache.stride(),
            *x.stride(),
            dt.stride(0),
            dt.stride(1),
            dt_bias.stride(0) if dt_bias is not None else 0,
            A.stride(0),
            *B.stride(),
            *C.stride(),
            *D_strides,
            *out.stride(),
            DT_SOFTPLUS=dt_softplus,
            MAX_WINDOW=max_window,
            RING_BUFFER_LEN=ring_len,
            BLOCK_SIZE_M=block_size_m,
            num_warps=4,
        )
    return out


@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit(do_not_specialize=["num_reqs"])
def _replayssm_materialize_kernel(
    state_ptrs,
    state_slot_strides,
    x_cache_ptrs,
    x_cache_slot_strides,
    B_cache_ptrs,
    B_cache_slot_strides,
    dt_cache_ptrs,
    dt_cache_slot_strides,
    A_ptrs,
    src_slots_ptr,
    dst_slots_ptr,
    ring_start_ptr,
    replay_prefix_len_ptr,
    active_request_indices_ptr,
    pad_slot_id,
    num_reqs,
    dim,
    dstate,
    nheads_ngroups_ratio,
    stride_state_head,
    stride_state_dim,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_dt_cache_head,
    stride_B_cache_group,
    stride_B_cache_pos,
    STATE_DTYPE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    A_DTYPE: tl.constexpr,
    MAX_NUM_REQS: tl.constexpr,
    RING_BUFFER_LEN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_r = tl.program_id(axis=0)
    pid_l = tl.program_id(axis=1)
    pid_hm = tl.program_id(axis=2)
    req = tl.load(active_request_indices_ptr + pid_r)
    if req < 0:
        return
    src = tl.load(src_slots_ptr + pid_l * MAX_NUM_REQS + req).to(tl.int64)
    dst = tl.load(dst_slots_ptr + pid_l * MAX_NUM_REQS + req).to(tl.int64)
    if (src == pad_slot_id) | (dst == pad_slot_id):
        return

    num_m_blocks = tl.cdiv(dim, BLOCK_SIZE_M)
    pid_h = pid_hm // num_m_blocks
    pid_m = pid_hm % num_m_blocks
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)

    state = tl.load(state_ptrs + pid_l).to(tl.pointer_type(STATE_DTYPE))
    state_slot_stride = tl.load(state_slot_strides + pid_l)
    x_cache = tl.load(x_cache_ptrs + pid_l).to(tl.pointer_type(INPUT_DTYPE))
    B_cache = tl.load(B_cache_ptrs + pid_l).to(tl.pointer_type(INPUT_DTYPE))
    dt_cache = tl.load(dt_cache_ptrs + pid_l).to(tl.pointer_type(tl.float32))
    A_ptr = tl.load(A_ptrs + pid_l).to(tl.pointer_type(A_DTYPE))

    offs_state = (
        pid_h * stride_state_head + offs_m[:, None] * stride_state_dim + offs_n[None, :]
    )
    S = tl.load(state + src * state_slot_stride + offs_state, mask=mask, other=0.0)
    S = _replay_ring(
        S.to(tl.float32),
        x_cache
        + src * tl.load(x_cache_slot_strides + pid_l)
        + pid_h * stride_x_cache_head,
        dt_cache
        + src * tl.load(dt_cache_slot_strides + pid_l)
        + pid_h * stride_dt_cache_head,
        B_cache
        + src * tl.load(B_cache_slot_strides + pid_l)
        + (pid_h // nheads_ngroups_ratio) * stride_B_cache_group,
        tl.load(ring_start_ptr + req),
        tl.load(replay_prefix_len_ptr + req),
        tl.load(A_ptr + pid_h).to(tl.float32),
        offs_m,
        offs_n,
        dim,
        dstate,
        stride_x_cache_pos,
        1,
        1,
        stride_B_cache_pos,
        1,
        RING_BUFFER_LEN,
    )
    tl.store(state + dst * state_slot_stride + offs_state, S.to(STATE_DTYPE), mask=mask)


def _tl_dtype(dtype: torch.dtype) -> tl.dtype:
    return getattr(tl, str(dtype).removeprefix("torch."))


def replayssm_materialize(
    state_ptrs: torch.Tensor,
    state_slot_strides: torch.Tensor,
    x_cache_ptrs: torch.Tensor,
    x_cache_slot_strides: torch.Tensor,
    B_cache_ptrs: torch.Tensor,
    B_cache_slot_strides: torch.Tensor,
    dt_cache_ptrs: torch.Tensor,
    dt_cache_slot_strides: torch.Tensor,
    A_ptrs: torch.Tensor,
    src_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    ring_start: torch.Tensor,
    replay_prefix_len: torch.Tensor,
    active_request_indices: torch.Tensor,
    *,
    num_reqs: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    num_heads: int,
    dim: int,
    dstate: int,
    heads_per_group: int,
    ring_buffer_len: int,
    pad_slot_id: int = NULL_BLOCK_ID,
) -> None:
    """Write `src` advanced by a ring prefix into `dst` for every layer.

    Triton counterpart of FlashInfer's `replayssm_materialize`: the tables
    hold one base pointer and slot stride per layer, `src_slots`/
    `dst_slots` are `(layers, max_num_reqs)` and the first `num_reqs`
    entries of `active_request_indices` select requests (`-1` skips).
    Per-slot state and ring layouts must be contiguous.
    """
    num_layers, max_num_reqs = src_slots.shape
    if num_reqs == 0 or num_layers == 0:
        return
    block_size_m = min(triton.next_power_of_2(dim), 16)
    grid = (num_reqs, num_layers, num_heads * triton.cdiv(dim, block_size_m))
    _replayssm_materialize_kernel[grid](
        state_ptrs,
        state_slot_strides,
        x_cache_ptrs,
        x_cache_slot_strides,
        B_cache_ptrs,
        B_cache_slot_strides,
        dt_cache_ptrs,
        dt_cache_slot_strides,
        A_ptrs,
        src_slots,
        dst_slots,
        ring_start,
        replay_prefix_len,
        active_request_indices,
        pad_slot_id,
        num_reqs,
        dim,
        dstate,
        heads_per_group,
        dim * dstate,
        dstate,
        ring_buffer_len * dim,
        dim,
        ring_buffer_len,
        ring_buffer_len * dstate,
        dstate,
        STATE_DTYPE=_tl_dtype(state_dtype),
        INPUT_DTYPE=_tl_dtype(input_dtype),
        A_DTYPE=_tl_dtype(matrixA_dtype),
        MAX_NUM_REQS=max_num_reqs,
        RING_BUFFER_LEN=ring_buffer_len,
        BLOCK_SIZE_M=block_size_m,
        num_warps=4,
    )
