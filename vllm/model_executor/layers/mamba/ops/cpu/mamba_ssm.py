# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._custom_ops as ops
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _mamba_chunk_scan_combined_fwd_cpu(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    return_intermediate_states=False,
    seq_idx=None,
    cu_seqlens=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
    **kwargs,
):
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape

    assert cu_seqlens is not None
    batch = cu_seqlens.size(0) - 1

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float().unsqueeze(0)
    if dt_softplus:
        dt_f = torch.nn.functional.softplus(dt_f)
    if dt_limit[0] > 0.0 or dt_limit[1] < float("inf"):
        dt_f = dt_f.clamp(min=dt_limit[0], max=dt_limit[1])

    all_states = torch.zeros(
        batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device
    )
    if initial_states is not None:
        all_states.copy_(initial_states.float())

    assert out.is_contiguous(), (
        "_mamba_chunk_scan_combined_fwd_cpu: `out` must be "
        "pre-allocated as a contiguous tensor"
    )

    D_1d = None
    if D is not None:
        d = D.float()
        while d.dim() > 1 and d.stride(-1) == 0:
            d = d.squeeze(-1)
        D_1d = d.contiguous()

    ops.mamba_chunk_scan_fwd_cpu(
        out,
        all_states,
        x,
        dt_f,
        A,
        B,
        C,
        D_1d,
        z,
        cu_seqlens.to(torch.int32),
    )

    out_dtype = state_dtype if state_dtype is not None else x.dtype
    all_states = all_states.to(out_dtype)

    return all_states


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    null_block_id=NULL_BLOCK_ID,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    is_blackwell=False,
    enable_stochastic_rounding=False,
    cache_philox_rounds=0,
):
    """CPU implementation for selective_state_update."""
    # Ensure out tensor exists
    if out is None:
        out = torch.empty_like(x if x.dim() == 2 else x)

    _state = state.unsqueeze(1) if state.dim() == 3 else state
    _x = x.unsqueeze(1) if x.dim() == 2 else x
    _dt = dt.unsqueeze(1) if dt.dim() == 2 else dt
    _A = A.unsqueeze(0) if A.dim() == 2 else A
    _B = B.unsqueeze(1) if B.dim() == 2 else B
    _C = C.unsqueeze(1) if C.dim() == 2 else C
    _D = D.unsqueeze(0) if (D is not None and D.dim() == 1) else D
    _z = z.unsqueeze(1) if (z is not None and z.dim() == 2) else z
    _dt_bias = (
        dt_bias.unsqueeze(0)
        if (dt_bias is not None and dt_bias.dim() == 1)
        else dt_bias
    )
    _out = out.unsqueeze(1) if out.dim() == 2 else out

    _sbi = state_batch_indices
    _dsbi = dst_state_batch_indices
    ops.selective_state_update_cpu(
        _state,
        _x,
        _dt,
        _A,
        _B,
        _C,
        _D,
        _z,
        _dt_bias,
        dt_softplus,
        _sbi,
        _dsbi,
        null_block_id,
        _out,
        num_accepted_tokens,
        cu_seqlens,
    )
    return _out.squeeze(1) if out.dim() == 2 else _out
