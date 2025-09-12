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


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


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
                                   cu_chunk_seqlens=None,
                                   last_chunk=None,
                                   dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   state_dtype=None,
                                   out=None,
                                   layer=None):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
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
                                      cu_chunk_seqlens,
                                      dt_bias=dt_bias,
                                      dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)

    '''
    print("layer: ", layer)
    has_init = initial_states is not None
    print("has_init: ", has_init)

    dA_cumsum_ref = torch.load("dump/dA_cumsum_%s_main_%d" % (layer, has_init))
    torch.cuda.synchronize()
    torch.testing.assert_close(dA_cumsum, dA_cumsum_ref, atol=0.0, rtol=0.0)

    dt_ref = torch.load("dump/dt_%s_main_%d" % (layer, has_init))
    torch.cuda.synchronize()
    torch.testing.assert_close(dt, dt_ref, atol=0.0, rtol=0.0)
    '''


    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(B,
                              x,
                              dt,
                              dA_cumsum,
                              cu_chunk_seqlens,
                              seq_idx=seq_idx,
                              states_in_fp32=True)

    '''
    states_ref = torch.load("dump/states_%s_main_%d" % (layer, has_init))
    torch.cuda.synchronize()
    torch.testing.assert_close(states, states_ref, atol=0.0, rtol=0.0)
    '''

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states
    #   ii) seq_idx iii) is_cont_batched and (iv) chunk_offsets to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    # - We will also make sure that the dA_cumsum is taken only from the start of the
    #   sequence (hence we need the full dA_cumsum tensor and not just the values at chunk boundaries)
    # - this will ensure that states will be updated with the rightmost flushed seq_idx
    #   of the previous chunk. This implies that the first chunk of states is either 0
    #   or equal to init_states of the first example.
    states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        cu_chunk_seqlens,
        initial_states=rearrange(initial_states, "... p n -> ... (p n)")
        if initial_states is not None else None,
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
        is_cont_batched=cu_seqlens is not None,
        chunk_offsets=chunk_offsets)

    states = rearrange(states, "... (p n) -> ... p n", n=dstate)

    '''
    print("after state passing: ")
    states_ref = torch.load("dump/final_states_%s_main_%d" % (layer, has_init)).unsqueeze(0)
    print("states.shape: ", states.shape)
    print("states_ref.shape: ", states_ref.shape)
    torch.testing.assert_close(states, states_ref, atol=0.0, rtol=0.0)
    '''

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C,
                        B,
                        chunk_size,
                        cu_chunk_seqlens,
                        seq_idx=seq_idx,
                        output_dtype=torch.float32)

    '''
    CB_ref = torch.load("dump/CB_%s_main_%d" % (layer, has_init))
    torch.testing.assert_close(CB, CB_ref, atol=0.0, rtol=0.0)
    '''

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
        cu_chunk_seqlens,
        D=D,
        z=z,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_states=initial_states,
        out=out,
    )
    '''
    out_x_ref = torch.load("dump/out_x_%s_main_%d" % (layer, has_init))
    torch.testing.assert_close(out_x, out_x_ref, atol=0.0, rtol=0.0)

    out_ref = torch.load("dump/out_%s_main_%d" % (layer, has_init))
    torch.testing.assert_close(out, out_ref, atol=0.0, rtol=0.0)
    '''

    if cu_seqlens is None:
        return out_x, dt, dA_cumsum, states, final_states
    else:
        assert batch == 1, "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        #print("last_chunk: ", last_chunk)
        varlen_states = states[:, last_chunk, ...].clone().squeeze(0)
        #print("varlen_states: ", varlen_states[0,0,0,:10])
        final_states = states[:, -1, ...]
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
                              cu_chunk_seqlens=None,
                              last_chunk=None,
                              dt_softplus=False,
                              dt_limit=(0.0, float("inf")),
                              out=None,
                              return_final_states=False,
                              return_varlen_states=False,
                              state_dtype=None,
                              layer=None):
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
        cu_chunk_seqlens: (num_chunks + 1)
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
        state_dtype: The data type of the ssm state
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
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk=last_chunk,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        out=out,
        state_dtype=state_dtype,
        layer=layer)
    if not return_varlen_states:
        if not return_final_states:
            return
        else:
            return final_states
    else:
        varlen_states = rest[0]
        return (varlen_states) if not return_final_states else (final_states,
                                                                varlen_states)
