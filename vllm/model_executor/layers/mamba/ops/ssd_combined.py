# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

# ruff: noqa: E501

import torch
from einops import rearrange
from packaging import version

from vllm.triton_utils import triton

from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import (_chunk_cumsum_fwd, _chunk_state_fwd,
                              chunk_state_varlen)
from .ssd_state_passing import _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def _mamba_chunk_scan_combined_fwd(x,
                                   dt,
                                   A,
                                   B,
                                   C,
                                   chunk_size,
                                   D=None,
                                   z=None,
                                   dt_bias=None,
                                   initial_states=None,
                                   seq_idx=None,
                                   chunk_indices=None,
                                   chunk_offsets=None,
                                   cu_seqlens=None,
                                   dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   out=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads, )
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads, )
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(
            1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(
            1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        if cu_seqlens is None:
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        else:
            assert initial_states.shape == (len(cu_seqlens) - 1, nheads,
                                            headdim, dstate)

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    dA_cumsum, dt = _chunk_cumsum_fwd(dt,
                                      A,
                                      chunk_size,
                                      dt_bias=dt_bias,
                                      dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(B,
                              x,
                              dt,
                              dA_cumsum,
                              seq_idx=seq_idx,
                              states_in_fp32=True)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states
    #   ii) seq_idx and iii) is_cont_batched to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    # - this will ensure that states will be updated with the rightmost flushed seq_idx
    #   of the previous chunk. This implies that the first chunk of states is either 0
    #   or equal to init_states of the first example.
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=rearrange(initial_states, "... p n -> ... (p n)")
        if initial_states is not None else None,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=C.dtype,
        is_cont_batched=cu_seqlens is not None)
    states, final_states = (rearrange(t, "... (p n) -> ... p n", n=dstate)
                            for t in [states, final_states])

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C,
                        B,
                        chunk_size,
                        seq_idx=seq_idx,
                        output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    out_x = _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        D=D,
        z=z,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_states=initial_states,
        out=out,
    )
    if cu_seqlens is None:
        return out_x, dt, dA_cumsum, states, final_states
    else:
        assert batch == 1, "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(
            B.squeeze(0),
            x.squeeze(0),
            dt.squeeze(0),
            dA_cumsum.squeeze(0),
            cu_seqlens,
            states.squeeze(0),
            initial_states=initial_states,
        )
        return out_x, dt, dA_cumsum, states, final_states, varlen_states


def mamba_chunk_scan_combined(x,
                              dt,
                              A,
                              B,
                              C,
                              chunk_size,
                              D=None,
                              z=None,
                              dt_bias=None,
                              initial_states=None,
                              seq_idx=None,
                              chunk_indices=None,
                              chunk_offsets=None,
                              cu_seqlens=None,
                              dt_softplus=False,
                              dt_limit=(0.0, float("inf")),
                              out=None,
                              return_final_states=False,
                              return_varlen_states=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
    """

    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert cu_seqlens is not None, "cu_seqlens must be provided if return_varlen_states is True"
    out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=cu_seqlens,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        out=out)
    if not return_varlen_states:
        if not return_final_states:
            return
        else:
            return final_states
    else:
        varlen_states = rest[0]
        return (varlen_states) if not return_final_states else (final_states,
                                                                varlen_states)
