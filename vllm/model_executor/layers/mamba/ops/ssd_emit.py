# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-gated Mamba2 SSD kernels for batch-invariant decode.

Given the buffered inputs of a sequence's active (partial) chunk and the fp32
state at the last chunk boundary, these kernels compute the output row of one
token without materializing the chunk states: the online counterpart of SSD
stages 1 (dt cumsum), 4 (C.B^T row) and 5 (chunk scan) for a single row. The
chunk state (stages 2 and 3) is only computed when a chunk completes, by the
fold kernel, which runs for every row and returns early for rows whose token
does not complete a chunk.

The buffers are addressed as ``buf[slot, pos]`` through explicit strides, so
views into the paged Mamba KV cache work directly. Every kernel skips rows
whose slot is negative and otherwise processes each row independently, so a
launch over a padded batch computes the same bits for the real rows.
"""

import torch

from vllm.model_executor.layers.mamba.ops import ssd_bmm, ssd_chunk_scan
from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _CHUNK_STATE_BATCH_INVARIANT_CONFIG,
    _CUMSUM_BATCH_INVARIANT_CONFIG,
)
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.triton_utils import tl, triton


@triton.jit
def _workspace_chunk_cumsum_fwd_kernel(
    dt_ptr,
    current_dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    slot_indices_ptr,
    chunk_offsets_ptr,
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    stride_dt_slot: tl.int64,
    stride_dt_token: tl.int64,
    stride_dt_head: tl.constexpr,
    stride_current_dt_row: tl.int64,
    stride_current_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    HAS_CURRENT_DT: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)

    slot = tl.load(slot_indices_ptr + row)
    chunk_offset = tl.load(chunk_offsets_ptr + row)
    valid_row = slot >= 0
    chunk_size_limit = chunk_offset + 1

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = (
        dt_ptr
        + slot * stride_dt_slot
        + offs_c[None, :] * stride_dt_token
        + offs_h[:, None] * stride_dt_head
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = (
        dt_out_ptr
        + offs_h[:, None] * stride_dt_out_head
        + row * stride_dt_out_chunk
        + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = (
        dA_cumsum_ptr
        + offs_h[:, None] * stride_dA_cs_head
        + row * stride_dA_cs_chunk
        + offs_c[None, :] * stride_dA_cs_csize
    )

    active = (
        valid_row & (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit)
    )
    page_mask = active
    if HAS_CURRENT_DT:
        page_mask &= offs_c[None, :] != chunk_offset
    dt = tl.load(dt_ptrs, mask=page_mask, other=0.0).to(tl.float32)
    if HAS_CURRENT_DT:
        current_dt = tl.load(
            current_dt_ptr
            + row * stride_current_dt_row
            + offs_h * stride_current_dt_head,
            mask=valid_row & (offs_h < nheads),
            other=0.0,
        )
        current_mask = active & (offs_c[None, :] == chunk_offset)
        dt = tl.where(current_mask, current_dt[:, None].to(tl.float32), dt)
        tl.store(dt_ptrs, current_dt[:, None], mask=current_mask)
    dt_bias = tl.load(
        dt_bias_ptr + offs_h * stride_dt_bias_head,
        mask=offs_h < nheads,
        other=0.0,
    ).to(tl.float32)
    dt += dt_bias[:, None]
    dt = tl.where(dt <= 20.0, softplus(dt), dt)
    dt = tl.where(active, dt, 0.0)
    store_mask = valid_row & (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size)
    tl.store(dt_out_ptrs, dt, mask=store_mask)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=store_mask)


def _workspace_chunk_cumsum_fwd(
    dt,
    A,
    chunk_size,
    slot_indices,
    chunk_offsets,
    dt_bias,
    *,
    current_dt=None,
    dt_out,
    dA_cumsum,
):
    _, dt_chunk_size, nheads = dt.shape
    assert dt_chunk_size == chunk_size
    assert A.shape == (nheads,)
    num_rows = slot_indices.shape[0]
    assert chunk_offsets.shape[0] == num_rows
    assert dt_bias.shape == (nheads,)
    if current_dt is not None:
        assert current_dt.shape == (num_rows, nheads)
        assert current_dt.dtype == dt.dtype
        assert current_dt.device == dt.device
    assert dt_out.ndim == 3
    assert dt_out.shape[0] == nheads
    assert dt_out.shape[1] >= num_rows
    assert dt_out.shape[2] == chunk_size
    assert dt_out.dtype == torch.float32
    assert dt_out.device == dt.device
    assert dA_cumsum.shape == dt_out.shape
    assert dA_cumsum.dtype == torch.float32
    assert dA_cumsum.device == dt.device
    # Same head tile as the single-shot cumsum kernel's pinned configuration,
    # whatever the head count: tl.cumsum's summation order depends on the
    # tile shape, so the prefix sums only match bit for bit with the same tile.
    block_h = _CUMSUM_BATCH_INVARIANT_CONFIG.kwargs["BLOCK_SIZE_H"]
    grid_chunk_cs = (num_rows, triton.cdiv(nheads, block_h))
    with torch.accelerator.device_index(dt.device.index):
        _workspace_chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt_ptr=dt,
            current_dt_ptr=current_dt if current_dt is not None else dt,
            A_ptr=A,
            dt_bias_ptr=dt_bias,
            dt_out_ptr=dt_out,
            dA_cumsum_ptr=dA_cumsum,
            slot_indices_ptr=slot_indices,
            chunk_offsets_ptr=chunk_offsets,
            nheads=nheads,
            chunk_size=chunk_size,
            stride_dt_slot=dt.stride(0),
            stride_dt_token=dt.stride(1),
            stride_dt_head=dt.stride(2),
            stride_current_dt_row=current_dt.stride(0) if current_dt is not None else 0,
            stride_current_dt_head=current_dt.stride(1)
            if current_dt is not None
            else 0,
            stride_A_head=A.stride(0),
            stride_dt_bias_head=dt_bias.stride(0),
            stride_dt_out_head=dt_out.stride(0),
            stride_dt_out_chunk=dt_out.stride(1),
            stride_dt_out_csize=dt_out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            HAS_CURRENT_DT=current_dt is not None,
            BLOCK_SIZE_H=block_h,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


@triton.jit
def _bmm_chunk_workspace_range_fwd_kernel(
    emit_a_ptr,
    b_ptr,
    current_b_ptr,
    out_ptr,
    slot_indices_ptr,
    replay_offsets_ptr,
    emit_row_indices_ptr,
    emit_chunk_positions_ptr,
    chunk_size: tl.constexpr,
    K: tl.constexpr,
    stride_emit_a_row: tl.int64,
    stride_emit_a_head: tl.int64,
    stride_emit_ak: tl.constexpr,
    stride_b_slot: tl.int64,
    stride_b_token: tl.int64,
    stride_b_head: tl.int64,
    stride_bk: tl.constexpr,
    stride_current_b_row: tl.int64,
    stride_current_b_head: tl.int64,
    stride_current_bk: tl.constexpr,
    stride_out_emit: tl.int64,
    stride_out_head: tl.int64,
    stride_outn: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_CURRENT_B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    emit_idx = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    row_idx = tl.load(emit_row_indices_ptr + emit_idx)
    slot = tl.load(slot_indices_ptr + row_idx)
    replay_offset = tl.load(replay_offsets_ptr + row_idx)
    emit_pos = tl.load(emit_chunk_positions_ptr + emit_idx)
    valid_emit = (
        (slot >= 0)
        & (emit_pos >= 0)
        & (emit_pos <= replay_offset)
        & (replay_offset < chunk_size)
    )
    chunk_size_limit = replay_offset + 1
    current_emit = valid_emit & (emit_pos == replay_offset)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        emit_a_ptr
        + emit_idx * stride_emit_a_row
        + pid_h * stride_emit_a_head
        + offs_m[:, None] * 0
        + offs_k[None, :] * stride_emit_ak
    )
    b_ptrs = (
        b_ptr
        + slot * stride_b_slot
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_b_token
        + pid_h * stride_b_head
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (
            valid_emit
            & (offs_m[:, None] == 0)
            & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        )
        b_mask = (
            valid_emit
            & (offs_k[:, None] < K - k * BLOCK_SIZE_K)
            & (offs_n[None, :] < chunk_size_limit)
        )
        if HAS_CURRENT_B:
            b_mask &= offs_n[None, :] != replay_offset
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        ).to(dot_dtype)
        b = tl.load(
            b_ptrs,
            mask=b_mask,
            other=0.0,
        ).to(dot_dtype)
        if HAS_CURRENT_B:
            current_b = tl.load(
                current_b_ptr
                + row_idx * stride_current_b_row
                + pid_h * stride_current_b_head
                + (k * BLOCK_SIZE_K + offs_k) * stride_current_bk,
                mask=current_emit & (offs_k < K - k * BLOCK_SIZE_K),
                other=0.0,
            ).to(dot_dtype)
            b = tl.where(
                current_emit & (offs_n[None, :] == replay_offset),
                current_b[:, None],
                b,
            )
            persist_mask = current_emit & (pid_n == 0) & (offs_k < K - k * BLOCK_SIZE_K)
            tl.store(
                b_ptr
                + slot * stride_b_slot
                + replay_offset * stride_b_token
                + pid_h * stride_b_head
                + (k * BLOCK_SIZE_K + offs_k) * stride_bk,
                current_b,
                mask=persist_mask,
            )
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_emit_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    out_ptr += emit_idx * stride_out_emit + pid_h * stride_out_head
    out = tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offs_n * stride_outn, out, mask=valid_emit & (offs_n < chunk_size)
    )


def _bmm_chunk_workspace_range_fwd(
    emit_a,
    b,
    chunk_size,
    slot_indices,
    replay_offsets,
    emit_row_indices,
    emit_chunk_positions,
    out,
    current_b=None,
):
    _, b_chunk_size, ngroups, k = b.shape
    num_emit_tokens = emit_a.shape[0]
    assert b_chunk_size == chunk_size
    assert emit_a.shape == (num_emit_tokens, ngroups, k)
    assert slot_indices.shape[0] == replay_offsets.shape[0]
    assert emit_row_indices.shape[0] == num_emit_tokens
    assert emit_chunk_positions.shape[0] == num_emit_tokens
    if current_b is not None:
        assert current_b.shape == (replay_offsets.shape[0], ngroups, k)
        assert current_b.dtype == b.dtype
        assert current_b.device == b.device
    if emit_a.stride(-1) != 1 and emit_a.stride(0) != 1:
        emit_a = emit_a.contiguous()
    if b.stride(-1) != 1 and b.stride(0) != 1:
        b = b.contiguous()

    dot_dtype = (
        tl.bfloat16
        if emit_a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16
        else (
            tl.float16
            if emit_a.dtype == torch.float16 or b.dtype == torch.float16
            else tl.float32
        )
    )
    assert out.shape == (num_emit_tokens, ngroups, chunk_size)
    assert out.device == emit_a.device
    assert out.dtype == torch.float32
    if num_emit_tokens == 0:
        return out

    grid = (
        num_emit_tokens,
        ngroups,
        triton.cdiv(chunk_size, 64),
    )
    with torch.accelerator.device_index(emit_a.device.index):
        _bmm_chunk_workspace_range_fwd_kernel[grid](
            emit_a_ptr=emit_a,
            b_ptr=b,
            current_b_ptr=current_b if current_b is not None else b,
            out_ptr=out,
            slot_indices_ptr=slot_indices,
            replay_offsets_ptr=replay_offsets,
            emit_row_indices_ptr=emit_row_indices,
            emit_chunk_positions_ptr=emit_chunk_positions,
            chunk_size=chunk_size,
            K=k,
            stride_emit_a_row=emit_a.stride(0),
            stride_emit_a_head=emit_a.stride(1),
            stride_emit_ak=emit_a.stride(2),
            stride_b_slot=b.stride(0),
            stride_b_token=b.stride(1),
            stride_b_head=b.stride(2),
            stride_bk=b.stride(3),
            stride_current_b_row=current_b.stride(0) if current_b is not None else 0,
            stride_current_b_head=current_b.stride(1) if current_b is not None else 0,
            stride_current_bk=current_b.stride(2) if current_b is not None else 0,
            stride_out_emit=out.stride(0),
            stride_out_head=out.stride(1),
            stride_outn=out.stride(2),
            dot_dtype=dot_dtype,
            HAS_CURRENT_B=current_b is not None,
            # The row's dot products reduce over dstate in the same K blocks
            # as the pinned prefill bmm kernel; M and N tile the single output
            # row and the chunk positions and do not enter the reduction.
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=ssd_bmm._BATCH_INVARIANT_CONFIG.kwargs["BLOCK_SIZE_K"],
            num_warps=4,
            num_stages=4,
        )
    return out


@triton.jit
def _chunk_scan_workspace_range_fwd_kernel(
    cb_emit_ptr,
    x_ptr,
    current_x_ptr,
    out_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    C_ptr,
    initstates_ptr,
    D_ptr,
    slot_indices_ptr,
    initial_state_indices_ptr,
    replay_offsets_ptr,
    emit_row_indices_ptr,
    emit_chunk_positions_ptr,
    emit_token_indices_ptr,
    num_output_tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_cb_emit: tl.int64,
    stride_cb_head: tl.int64,
    stride_cb_csize_k: tl.constexpr,
    stride_x_slot: tl.int64,
    stride_x_token: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_current_x_row: tl.int64,
    stride_current_x_head: tl.int64,
    stride_current_x_hdim: tl.constexpr,
    stride_out_row: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_C_emit: tl.int64,
    stride_C_head: tl.int64,
    stride_C_dstate: tl.constexpr,
    stride_init_states_slot: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    stride_D_head: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_CURRENT_X: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    emit_idx = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    row_idx = tl.load(emit_row_indices_ptr + emit_idx)
    slot = tl.load(slot_indices_ptr + row_idx)
    initial_state_index = tl.load(initial_state_indices_ptr + row_idx)
    replay_offset = tl.load(replay_offsets_ptr + row_idx)
    emit_pos = tl.load(emit_chunk_positions_ptr + emit_idx)
    out_token = tl.load(emit_token_indices_ptr + emit_idx)
    valid_emit = (
        (slot >= 0)
        & (initial_state_index >= 0)
        & (emit_pos >= 0)
        & (emit_pos <= replay_offset)
        & (replay_offset < chunk_size)
        & (out_token >= 0)
        & (out_token < num_output_tokens)
    )
    current_emit = valid_emit & (emit_pos == replay_offset)
    chunk_size_limit = emit_pos + 1
    group = pid_h // nheads_ngroups_ratio
    pid_m = emit_pos // BLOCK_SIZE_M

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    emit_row_mask = valid_emit & (offs_m == emit_pos)
    dt_row = row_idx
    dA_cumsum_ptr += pid_h * stride_dA_cs_head + dt_row * stride_dA_cs_chunk
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize,
        mask=emit_row_mask,
        other=0.0,
    ).to(tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    row_load_mask = emit_row_mask

    offs_k_dstate = tl.arange(
        0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K
    )
    C_ptrs = (
        C_ptr
        + emit_idx * stride_C_emit
        + offs_m[:, None] * 0
        + group * stride_C_head
        + offs_k_dstate[None, :] * stride_C_dstate
    )
    scale_m = fast_exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(
            C_ptrs,
            mask=row_load_mask[:, None] & (offs_k_dstate[None, :] < dstate),
            other=0.0,
        )
        prev_states_ptrs = (
            initstates_ptr
            + initial_state_index * stride_init_states_slot
            + pid_h * stride_init_states_head
            + offs_n[None, :] * stride_init_states_hdim
            + offs_k_dstate[:, None] * stride_init_states_dstate
        )
        prev_states_mask = (
            valid_emit & (offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim)
        )
        prev_states = tl.load(prev_states_ptrs, mask=prev_states_mask, other=0.0)
        prev_states = prev_states.to(C_ptr.dtype.element_ty)
        acc = tl.dot(C, prev_states) * scale_m[:, None]
    else:
        prev_states_ptrs = (
            initstates_ptr
            + initial_state_index * stride_init_states_slot
            + pid_h * stride_init_states_head
            + offs_n[None, :] * stride_init_states_hdim
            + offs_k_dstate[:, None] * stride_init_states_dstate
        )
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(
                C_ptrs,
                mask=row_load_mask[:, None] & (offs_k_dstate[None, :] < dstate - k),
                other=0.0,
            )
            prev_states_mask = (
                valid_emit
                & (offs_k_dstate[:, None] < dstate - k)
                & (offs_n[None, :] < hdim)
            )
            prev_states = tl.load(prev_states_ptrs, mask=prev_states_mask, other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K * stride_C_dstate
            prev_states_ptrs += BLOCK_SIZE_K * stride_init_states_dstate
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = (
        cb_emit_ptr
        + emit_idx * stride_cb_emit
        + group * stride_cb_head
        + offs_k * stride_cb_csize_k
    )
    x_ptrs = (
        x_ptr
        + slot * stride_x_slot
        + offs_k[:, None] * stride_x_token
        + pid_h * stride_x_head
        + offs_n[None, :] * stride_x_hdim
    )
    dt_ptrs = (
        dt_ptr
        + pid_h * stride_dt_head
        + dt_row * stride_dt_chunk
        + offs_k * stride_dt_csize
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_CURRENT_X:
        current_x = tl.load(
            current_x_ptr
            + row_idx * stride_current_x_row
            + pid_h * stride_current_x_head
            + offs_n * stride_current_x_hdim,
            mask=current_emit & (offs_n < hdim),
            other=0.0,
        )
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        token_mask = valid_emit & (offs_k < chunk_size_limit - k)
        cb_emit = tl.load(cb_ptrs, mask=token_mask, other=0.0).to(tl.float32)
        cb = tl.where(offs_m[:, None] == emit_pos, cb_emit[None, :], 0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=token_mask, other=0.0).to(tl.float32)
        cb *= fast_exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
        dt_k = tl.load(dt_ptrs, mask=token_mask, other=0.0).to(tl.float32)
        cb *= dt_k[None, :]
        causal_mask = offs_m[:, None] >= k + offs_k[None, :]
        cb = tl.where(causal_mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x_mask = (
            valid_emit
            & (offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < hdim)
        )
        if HAS_CURRENT_X:
            x_mask &= (k + offs_k[:, None]) != replay_offset
        x = tl.load(
            x_ptrs,
            mask=x_mask,
            other=0.0,
        )
        if HAS_CURRENT_X:
            x = tl.where(
                current_emit & ((k + offs_k[:, None]) == replay_offset),
                current_x[None, :],
                x,
            )
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_token
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(
                D_ptr + pid_h * stride_D_head + offs_n,
                mask=offs_n < hdim,
                other=0.0,
            ).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        residual_mask = row_load_mask[:, None] & (offs_n[None, :] < hdim)
        if HAS_CURRENT_X:
            residual_mask &= offs_m[:, None] != replay_offset
        x_residual = tl.load(
            x_ptr
            + slot * stride_x_slot
            + offs_m[:, None] * stride_x_token
            + pid_h * stride_x_head
            + offs_n[None, :] * stride_x_hdim,
            mask=residual_mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_CURRENT_X:
            x_residual = tl.where(
                current_emit & (offs_m[:, None] == replay_offset),
                current_x[None, :].to(tl.float32),
                x_residual,
            )
        acc += x_residual * D

    if HAS_CURRENT_X:
        current_x_dst = (
            x_ptr
            + slot * stride_x_slot
            + replay_offset * stride_x_token
            + pid_h * stride_x_head
            + offs_n * stride_x_hdim
        )
        tl.store(
            current_x_dst,
            current_x,
            mask=current_emit & (offs_n < hdim),
        )

    out_ptr += out_token * stride_out_row + pid_h * stride_out_head
    out_ptrs = out_ptr + offs_m[:, None] * 0 + offs_n[None, :] * stride_out_hdim
    tl.store(
        out_ptrs,
        acc,
        mask=valid_emit & (offs_m[:, None] == emit_pos) & (offs_n[None, :] < hdim),
    )


def _chunk_scan_workspace_range_fwd(
    cb_emit,
    x,
    dt,
    dA_cumsum,
    C,
    initial_states,
    initial_state_indices,
    slot_indices,
    replay_offsets,
    emit_row_indices,
    emit_chunk_positions,
    emit_token_indices,
    out,
    D=None,
    current_x=None,
):
    num_slots, chunk_size, nheads, headdim = x.shape
    num_emit_tokens, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (num_emit_tokens, ngroups, dstate)
    assert cb_emit.shape == (num_emit_tokens, ngroups, chunk_size)
    num_rows = replay_offsets.shape[0]
    assert dt.ndim == 3
    assert dt.shape[0] == nheads
    assert dt.shape[1] >= num_rows
    assert dt.shape[2] == chunk_size
    assert dA_cumsum.shape == dt.shape
    assert initial_states.shape[1:] == (nheads, headdim, dstate)
    assert initial_state_indices.shape[0] == replay_offsets.shape[0]
    assert slot_indices.shape[0] == replay_offsets.shape[0]
    assert emit_row_indices.shape[0] == num_emit_tokens
    assert emit_chunk_positions.shape[0] == num_emit_tokens
    assert emit_token_indices.shape[0] == num_emit_tokens
    assert out.shape[1:] == (nheads, headdim)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    has_current_x = current_x is not None
    if has_current_x:
        assert current_x.shape == (num_rows, nheads, headdim)
        assert current_x.dtype == x.dtype
    if num_emit_tokens == 0:
        return

    grid = lambda META: (
        num_emit_tokens,
        nheads,
        triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
    )
    with torch.accelerator.device_index(x.device.index):
        _chunk_scan_workspace_range_fwd_kernel[grid](
            cb_emit_ptr=cb_emit,
            x_ptr=x,
            current_x_ptr=current_x if has_current_x else x,
            out_ptr=out,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            C_ptr=C,
            initstates_ptr=initial_states,
            D_ptr=D,
            slot_indices_ptr=slot_indices,
            initial_state_indices_ptr=initial_state_indices,
            replay_offsets_ptr=replay_offsets,
            emit_row_indices_ptr=emit_row_indices,
            emit_chunk_positions_ptr=emit_chunk_positions,
            emit_token_indices_ptr=emit_token_indices,
            num_output_tokens=out.shape[0],
            chunk_size=chunk_size,
            hdim=headdim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_cb_emit=cb_emit.stride(0),
            stride_cb_head=cb_emit.stride(1),
            stride_cb_csize_k=cb_emit.stride(2),
            stride_x_slot=x.stride(0),
            stride_x_token=x.stride(1),
            stride_x_head=x.stride(2),
            stride_x_hdim=x.stride(3),
            stride_current_x_row=current_x.stride(0) if has_current_x else 0,
            stride_current_x_head=current_x.stride(1) if has_current_x else 0,
            stride_current_x_hdim=current_x.stride(2) if has_current_x else 0,
            stride_out_row=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_hdim=out.stride(2),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_C_emit=C.stride(0),
            stride_C_head=C.stride(1),
            stride_C_dstate=C.stride(2),
            stride_init_states_slot=initial_states.stride(0),
            stride_init_states_head=initial_states.stride(1),
            stride_init_states_hdim=initial_states.stride(2),
            stride_init_states_dstate=initial_states.stride(3),
            stride_D_head=D.stride(0) if D is not None else 0,
            HAS_D=D is not None,
            D_HAS_HDIM=D.dim() == 2 if D is not None else True,
            HAS_CURRENT_X=has_current_x,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            # Same tiles as the pinned prefill chunk-scan kernel so the dot
            # products accumulate in the same order (smaller M tiles change
            # the bits, measured); warps and stages only affect scheduling.
            num_warps=4,
            num_stages=2,
            **ssd_chunk_scan._BATCH_INVARIANT_CONFIG.kwargs,
        )
    return


@triton.jit
def _fold_chunk_fwd_kernel(
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    states_ptr,
    slot_indices_ptr,
    chunk_offsets_ptr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    stride_x_slot: tl.int64,
    stride_x_token: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_slot: tl.int64,
    stride_b_token: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_states_slot: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    slot = tl.load(slot_indices_ptr + row).to(tl.int64)
    chunk_offset = tl.load(chunk_offsets_ptr + row)
    if (slot < 0) | (chunk_offset != chunk_size - 1):
        return

    x_ptr += slot * stride_x_slot + pid_h * stride_x_head
    b_ptr += slot * stride_b_slot + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dt_ptr += pid_h * stride_dt_head + row * stride_dt_chunk
    dA_cumsum_ptr += pid_h * stride_dA_cs_head + row * stride_dA_cs_chunk

    # Same tiling, loop order and arithmetic as _chunk_state_fwd_kernel over a
    # full chunk, so the fp32 chunk state matches the prefill kernels bit for
    # bit; the state-passing update (_state_passing_fwd_kernel) is fused at
    # the end.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_token
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_token
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(
            tl.float32
        )
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        scale = fast_exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        x_ptrs += BLOCK_SIZE_K * stride_x_token
        b_ptrs += BLOCK_SIZE_K * stride_b_token
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    states_ptrs = (
        states_ptr
        + slot * stride_states_slot
        + pid_h * stride_states_head
        + offs_m[:, None] * stride_states_hdim
        + offs_n[None, :] * stride_states_dstate
    )
    states_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    prev_states = tl.load(states_ptrs, mask=states_mask, other=0.0).to(tl.float32)
    # Same explicit fused multiply-add as _state_passing_fwd_kernel; a plain
    # `a * b + c` after the dot loop is not contracted the same way.
    states = tl.fma(fast_exp(dA_cs_last), prev_states, acc)
    tl.store(states_ptrs, states.to(states_ptr.dtype.element_ty), mask=states_mask)


def _fold_chunk_fwd(x, b, dt, dA_cumsum, states, slot_indices, chunk_offsets):
    """Fold every completed chunk into its slot's boundary state, in place.

    A row completes a chunk when its offset is the chunk's last position; all
    other rows and rows with a negative slot return immediately. ``dt`` and
    ``dA_cumsum`` are the ``(nheads, num_rows, chunk_size)`` outputs of
    :func:`_workspace_chunk_cumsum_fwd` for the same rows.
    """
    num_slots, chunk_size, nheads, headdim = x.shape
    _, _, ngroups, dstate = b.shape
    num_rows = slot_indices.shape[0]
    assert nheads % ngroups == 0
    assert b.shape[:2] == (num_slots, chunk_size)
    assert chunk_offsets.shape[0] == num_rows
    assert dt.shape[0] == nheads
    assert dt.shape[1] >= num_rows
    assert dt.shape[2] == chunk_size
    assert dA_cumsum.shape == dt.shape
    assert states.shape[1:] == (nheads, headdim, dstate)
    if num_rows == 0:
        return

    config = _CHUNK_STATE_BATCH_INVARIANT_CONFIG
    grid = (
        triton.cdiv(headdim, config.kwargs["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, config.kwargs["BLOCK_SIZE_N"]),
        num_rows,
        nheads,
    )
    with torch.accelerator.device_index(x.device.index):
        _fold_chunk_fwd_kernel[grid](
            x_ptr=x,
            b_ptr=b,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            states_ptr=states,
            slot_indices_ptr=slot_indices,
            chunk_offsets_ptr=chunk_offsets,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_slot=x.stride(0),
            stride_x_token=x.stride(1),
            stride_x_head=x.stride(2),
            stride_x_hdim=x.stride(3),
            stride_b_slot=b.stride(0),
            stride_b_token=b.stride(1),
            stride_b_head=b.stride(2),
            stride_b_dstate=b.stride(3),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_states_slot=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            **config.kwargs,
        )
