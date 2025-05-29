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
    # Matrix dimensions
    dim: tl.constexpr,
    nchunks,
    seqlen,
    chunk_size: tl.constexpr,
    # Strides
    stride_states_chunk: tl.constexpr,
    stride_states_head: tl.constexpr,
    stride_states_dim: tl.constexpr,
    stride_out_chunk: tl.constexpr,
    stride_out_head: tl.constexpr,
    stride_out_dim: tl.constexpr,
    stride_dA_cs_head,
    stride_dA_cs_chunk: tl.constexpr,
    stride_initstates_batch: tl.constexpr,
    stride_initstates_head: tl.constexpr,
    stride_initstates_dim: tl.constexpr,
    stride_seq_idx_seqlen: tl.constexpr,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_h * stride_states_head
    dA_cs_ptr += pid_h * stride_dA_cs_head
    out_ptr += pid_h * stride_out_head
    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_initstates_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

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
    for c in range(nchunks - 1):
        new_states = tl.load(states_ptrs, mask=offs_m < dim,
                             other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)

        # - the seq to pass forward is the one that is flushed to the right
        #   boundary.
        # - that is given by seq_idx_new below.
        seq_idx_new = tl.load(seq_idx_ptr +
                              (min((c + 1) * chunk_size, seqlen) - 1) *
                              stride_seq_idx_seqlen)
        if HAS_INITSTATES:
            if seq_idx != seq_idx_new:
                # this means in the current chunk the rightmost flushed seq
                # has changed.
                # - so we do not propagate the state from previous chunk
                # - but rather we load that sequence's init state
                initstates_ptrs = initstates_ptr + seq_idx_new * stride_initstates_batch

                # - update state with seq_idx_new's init state
                states = tl.load(initstates_ptrs, mask=offs_m < dim,
                                 other=0.0).to(tl.float32)
        else:
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        seq_idx = seq_idx_new

        states = scale * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
    states,
    dA_chunk_cumsum,
    seq_idx,
    chunk_size,
    initial_states=None,
    out_dtype=None,
):
    nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (nheads, nchunks)
    seqlen = seq_idx.shape[-1]

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim),
                      device=states.device,
                      dtype=out_dtype)

    initial_states_strides = ((initial_states.stride(0),
                               initial_states.stride(1),
                               initial_states.stride(2))
                              if initial_states is not None else (0, 0, 0))

    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_chunk_cumsum,
            initstates_ptr=initial_states,
            seq_idx_ptr=seq_idx,
            dim=dim,
            nchunks=nchunks,
            seqlen=seqlen if seq_idx is not None else 0,
            chunk_size=chunk_size if seq_idx is not None else 0,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_chunk_cumsum.stride(0),
            stride_dA_cs_chunk=dA_chunk_cumsum.stride(1),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            stride_seq_idx_seqlen=seq_idx.stride(0),
            HAS_INITSTATES=initial_states is not None,
        )
    return out
