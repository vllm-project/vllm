# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm._cpu_ops as ops


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
