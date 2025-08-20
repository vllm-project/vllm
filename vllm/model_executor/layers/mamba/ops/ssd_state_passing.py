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
    final_states_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
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
    stride_final_states_batch,
    stride_final_states_head,
    stride_final_states_dim,
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
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_initstates_head
        if not IS_CONT_BATCHED:
            initstates_ptr += pid_b * stride_initstates_batch

    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    final_states_ptrs = final_states_ptr + offs_m * stride_final_states_dim

    # - states will be the past state of the sequence that continues on the current check
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptr += offs_m * stride_initstates_dim
        initstates_ptrs = initstates_ptr
        # - for cont batches, for the first chunk mean it will be the first batch's
        #   init state
        states = tl.load(initstates_ptrs, mask=offs_m < dim,
                         other=0.0).to(tl.float32)

    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    prev_seq_idx_chunk_end = 0
    logical_chunk_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim,
                             other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale_mask = True
        if HAS_SEQ_IDX:
            # - the seq to pass forward is the one that is flushed to the right
            #   boundary.
            # - that is given by seq_idx_chunk_end below: the sequence index at the end of the chunk.
            seq_idx_chunk_end = tl.load(seq_idx_ptr + (min(
                (c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            if HAS_INITSTATES:
                if IS_CONT_BATCHED and prev_seq_idx_chunk_end != seq_idx_chunk_end:
                    # this means in the current chunk the rightmost flushed seq
                    # has changed.
                    # - so we do not propagate the state from previous chunk
                    # - but rather we load that sequence's init state
                    initstates_ptrs = initstates_ptr + seq_idx_chunk_end * stride_initstates_batch

                    # - update state with seq_idx_new's init state
                    states = tl.load(initstates_ptrs,
                                     mask=offs_m < dim,
                                     other=0.0).to(tl.float32)

                    # - we need to consider the cumsum only of the last sequence in the chunk
                    # - find its starting position (given by c_off of the logical chunk index)
                    # - and subtract the cumsum just before that position from the total cumsum
                    # - first, update the logical chunk index (add the number of sequences in the current physical chunk):
                    # sequence index at the start of the current chunk
                    seq_idx_chunk_start = tl.load(seq_idx_ptr +
                                                  min(c * chunk_size, seqlen) *
                                                  stride_seq_idx_seqlen)
                    logical_chunk_idx += seq_idx_chunk_end - seq_idx_chunk_start
                    # - load the chunk offset:
                    c_off = tl.load(chunk_offsets_ptr + logical_chunk_idx,
                                    mask=logical_chunk_idx < chunk_meta_num,
                                    other=0)
                    # - if offset is 0, then the sequence starts at the beginning of the chunk, and we don't need to subtract anything
                    if c_off > 0:
                        # - dA_cs_ptr currently points to the cumsum at the end of the chunk - subtract the chunk size and add the offset
                        dA_cs_boundary = tl.load(
                            dA_cs_ptr - (chunk_size - 1) * stride_dA_cs_csize +
                            (c_off - 1) * stride_dA_cs_csize,
                            mask=(c_off - 1) > -1 and c_off < chunk_size,
                            other=0.0)
                        dA_cs -= dA_cs_boundary

                # - increment logical chunk index for every physical chunk
                logical_chunk_idx += 1
            else:
                scale_mask = seq_idx_chunk_end == prev_seq_idx_chunk_end
            prev_seq_idx_chunk_end = seq_idx_chunk_end

        scale = tl.where(scale_mask, tl.exp(dA_cs), 0.0)
        states = scale * states + new_states
        if c < nchunks - 1:
            tl.store(out_ptrs, states, mask=offs_m < dim)
        else:
            tl.store(final_states_ptrs, states, mask=offs_m < dim)
        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
    states,
    dA_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
    chunk_offsets=None,
):
    batch, nchunks, nheads, dim = states.shape
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
    final_states = torch.empty((batch, nheads, dim),
                               device=states.device,
                               dtype=torch.float32)
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states,
            out,
            final_states,
            dA_cumsum,
            initial_states,
            seq_idx,
            chunk_offsets,
            len(chunk_offsets) if chunk_offsets is not None else 0,
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
            final_states.stride(0),
            final_states.stride(1),
            final_states.stride(2),
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
    return out, final_states
