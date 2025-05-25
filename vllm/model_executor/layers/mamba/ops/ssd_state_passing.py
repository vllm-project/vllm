# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils.jit_cache import jitcache


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
@jitcache(
    # should include constexpr args that may change across invocations
    check_keys=[
        "HAS_INITSTATES", "HAS_SEQ_IDX", "IS_CONT_BATCHED", "BLOCK_SIZE"
    ],
    # for variables that cannot be labeled constexpr because range > 32 bit
    assume_const=[
        "dim",
        "chunk_size",
        "stride_states_batch",
        "stride_states_chunk",
        "stride_states_head",
        "stride_states_dim",
        "stride_out_batch",
        "stride_out_chunk",
        "stride_out_head",
        "stride_out_dim",
        "stride_final_states_batch",
        "stride_final_states_head",
        "stride_final_states_dim",
        "stride_dA_cs_chunk",
    ],
    # cache_launch_grid=True,
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
    # Matrix dimensions
    dim: tl.int64,
    nchunks: tl.int64,
    seqlen: tl.int64,
    chunk_size: tl.int64,
    # Strides
    stride_states_batch: tl.int64,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_dim: tl.int64,
    stride_out_batch: tl.int64,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.int64,
    stride_final_states_batch: tl.int64,
    stride_final_states_head: tl.int64,
    stride_final_states_dim: tl.int64,
    stride_dA_cs_batch: tl.int64,  # not const, nchunk is dynamic
    stride_dA_cs_head: tl.int64,  # not const, nchunk is dynamic
    stride_dA_cs_chunk: tl.int64,
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_dim: tl.int64,
    stride_seq_idx_batch: tl.int64,
    stride_seq_idx_seqlen: tl.int64,
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
    dA_cs_ptr += pid_b * stride_dA_cs_batch + pid_h * stride_dA_cs_head
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
    seq_idx = 0
    for c in range(nchunks):
        new_states = tl.load(states_ptrs, mask=offs_m < dim,
                             other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            # - the seq to pass forward is the one that is flushed to the right
            #   boundary.
            # - that is given by seq_idx_new below.
            seq_idx_new = tl.load(seq_idx_ptr +
                                  (min((c + 1) * chunk_size, seqlen) - 1) *
                                  stride_seq_idx_seqlen)
            if HAS_INITSTATES:
                if IS_CONT_BATCHED and seq_idx != seq_idx_new:
                    # this means in the current chunk the rightmost flushed seq
                    # has changed.
                    # - so we do not propagate the state from previous chunk
                    # - but rather we load that sequence's init state
                    initstates_ptrs = initstates_ptr + seq_idx_new * stride_initstates_batch

                    # - update state with seq_idx_new's init state
                    states = tl.load(initstates_ptrs,
                                     mask=offs_m < dim,
                                     other=0.0).to(tl.float32)
            else:
                scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)

            seq_idx = seq_idx_new
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
    dA_chunk_cumsum,
    initial_states=None,
    seq_idx=None,
    chunk_size=None,
    out_dtype=None,
    is_cont_batched=False,
):
    batch, nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        if is_cont_batched:
            # - if cu_seqlens is provided, then the initial states
            #   are used for continuous batching. In which case we
            #   require seq_idx to be provided
            assert seq_idx is not None, ""
        else:
            # - this is the regular batching case, where initial
            #   states are used are for each example of the batch.
            assert initial_states.shape == (batch, nheads, dim)

    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
        assert seq_idx.shape == (batch, seqlen)
    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((batch, nchunks, nheads, dim),
                      device=states.device,
                      dtype=out_dtype)
    final_states = torch.empty((batch, nheads, dim),
                               device=states.device,
                               dtype=torch.float32)
    init_states_strides = ((initial_states.stride(0), initial_states.stride(1),
                            initial_states.stride(2))
                           if initial_states is not None else (0, 0, 0))
    seq_idx_strides = ((seq_idx.stride(0),
                        seq_idx.stride(1)) if seq_idx is not None else (0, 0))
    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), batch, nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            final_states_ptr=final_states,
            dA_cs_ptr=dA_chunk_cumsum,
            initstates_ptr=initial_states,
            seq_idx_ptr=seq_idx,
            dim=dim,
            nchunks=nchunks,
            seqlen=seqlen if seq_idx is not None else 0,
            chunk_size=chunk_size if seq_idx is not None else 0,
            stride_states_batch=states.stride(0),
            stride_states_chunk=states.stride(1),
            stride_states_head=states.stride(2),
            stride_states_dim=states.stride(3),
            stride_out_batch=out.stride(0),
            stride_out_chunk=out.stride(1),
            stride_out_head=out.stride(2),
            stride_out_dim=out.stride(3),
            stride_final_states_batch=final_states.stride(0),
            stride_final_states_head=final_states.stride(1),
            stride_final_states_dim=final_states.stride(2),
            stride_dA_cs_batch=dA_chunk_cumsum.stride(0),
            stride_dA_cs_head=dA_chunk_cumsum.stride(1),
            stride_dA_cs_chunk=dA_chunk_cumsum.stride(2),
            stride_initstates_batch=init_states_strides[0],
            stride_initstates_head=init_states_strides[1],
            stride_initstates_dim=init_states_strides[2],
            stride_seq_idx_batch=seq_idx_strides[0],
            stride_seq_idx_seqlen=seq_idx_strides[1],
            HAS_INITSTATES=initial_states is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            IS_CONT_BATCHED=is_cont_batched,
        )
    return out, final_states
