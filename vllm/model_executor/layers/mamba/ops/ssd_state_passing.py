# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    dim,
    nchunks,
    seqlen,
    chunk_size,
    # Strides
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_dim,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_out_dim,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_initstates_batch,
    stride_initstates_head,
    stride_initstates_dim,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_CONT_BATCHED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head + (
        chunk_size - 1) * stride_dA_cs_csize
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    if HAS_INITSTATES:
        initstates_ptrs = initstates_ptr + 0 * stride_initstates_batch \
            + pid_h * stride_initstates_head \
            + offs_m * stride_initstates_dim

        states = tl.load(initstates_ptrs,
                         mask=offs_m < dim,
                         other=0.0).to(tl.float32)
    else:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

    prev_seq_idx = 0
    for c in range(nchunks):

        chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + c)

        new_states = tl.load(states_ptrs, mask=offs_m < dim,
                             other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)

        seq_idx = tl.load(seq_idx_ptr + chunk_seqlen_start * stride_seq_idx_seqlen)

        # we are started a new sequence
        if prev_seq_idx != seq_idx:
            if HAS_INITSTATES:
                initstates_ptrs = initstates_ptr + seq_idx * stride_initstates_batch \
                    + pid_h * stride_initstates_head \
                    + offs_m * stride_initstates_dim
                states = tl.load(initstates_ptrs,
                                 mask=offs_m < dim,
                                 other=0.0).to(tl.float32)
            else:
                states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)

        prev_seq_idx = seq_idx

        states = tl.exp(dA_cs) * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
    states,
    dA_cumsum,
    cu_chunk_seqlens,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
    chunk_offsets=None,
):
    batch, nchunks, nheads, dim = states.shape
    assert batch == 1
    if chunk_size is None:
        chunk_size = dA_cumsum.shape[-1]
    else:
        assert chunk_size == dA_cumsum.shape[-1]
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if initial_states is not None:
        if is_cont_batched:
            # - if cu_seqlens is provided, then the initial states
            #   are used for continuous batching. In which case we
            #   require seq_idx to be provided
            assert seq_idx is not None, "seq_idx must be provided for continuous batching"
            # - we also need chunk_offsets to be provided, to account
            #   for computation of dA_cumsum from the start of the
            #   sequence
            assert chunk_offsets is not None, "chunk_offsets must be provided for continuous batching"
        else:
            # - this is the regular batching case, where initial
            #   states are used are for each example of the batch.
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim),
                      device=states.device,
                      dtype=out_dtype)

    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            dA_cumsum,
            initial_states,
            seq_idx,
            chunk_offsets,
            len(chunk_offsets) if chunk_offsets is not None else 0,
            cu_chunk_seqlens,
            dim,
            nchunks,
            seqlen if seq_idx is not None else 0,
            chunk_size,
            states.stride(0),
            states.stride(1),
            states.stride(2),
            states.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *((initial_states.stride(0), initial_states.stride(1),
               initial_states.stride(2)) if initial_states is not None else
              (0, 0, 0)),
            *((seq_idx.stride(0),
               seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_CONT_BATCHED=is_cont_batched,
        )
    return out
