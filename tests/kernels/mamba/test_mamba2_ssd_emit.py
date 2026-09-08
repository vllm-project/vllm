# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-gated emit reproduces the single-shot chunked scan bit for bit.

A sequence is scanned once with the full SSD pipeline (reference). Then every
token is produced again the way batch-invariant decode would: the fp32 state at
the last chunk boundary plus the buffered inputs of the active chunk, held in
paged ``buf[slot, pos]`` views, feed the row-gated emit kernels for that one
position. Every emitted row must equal the reference row.
"""

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssd_emit import (
    _bmm_chunk_workspace_range_fwd,
    _chunk_scan_workspace_range_fwd,
    _workspace_chunk_cumsum_fwd,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike device.",
)


def _reference(x, dt, A, B, C, D, dt_bias, chunk_size):
    """Single-shot scan: output rows and the fp32 state after every chunk."""
    seqlen = x.shape[0]
    cu_chunk = list(range(0, seqlen, chunk_size)) + [seqlen]
    n_chunks = len(cu_chunk) - 1
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=x.device)  # noqa: E731
    out = torch.empty_like(x)
    states = mamba_chunk_scan_combined_varlen(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=chunk_size,
        cu_seqlens=i32([0, seqlen]),
        cu_chunk_seqlens=i32(cu_chunk),
        last_chunk_indices=i32([n_chunks - 1]),
        seq_idx=i32([0] * n_chunks),
        out=out,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=None,
        return_intermediate_states=True,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        state_dtype=torch.float32,
    )
    return out, states  # states[c] = state after chunk c


def _paged(num_slots, shape, dtype, device, pad_bytes=4096):
    """One buffer per slot carved out of a padded page, as the KV cache does."""
    nbytes = int(torch.empty(shape, dtype=dtype).numel() * dtype.itemsize)
    pages = torch.zeros(num_slots, nbytes + pad_bytes, dtype=torch.uint8, device=device)
    return pages[:, :nbytes].view(dtype).view(-1, *shape)


@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("seed", [0, 1])
def test_emit_matches_single_shot_prefill_at_every_position(chunk_size, seed):
    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    nheads, head_dim, ngroups, dstate = 8, 64, 1, 64
    dtype = torch.bfloat16
    seqlen = 3 * chunk_size + 17

    A = -(torch.rand(nheads, device=device) * 15 + 1)
    dt_target = torch.exp(
        torch.rand(nheads, device=device)
        * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
        + torch.log(torch.tensor(1e-3))
    )
    dt_bias = dt_target + torch.log(-torch.expm1(-dt_target))
    D = torch.ones(nheads, device=device)
    x = torch.randn(seqlen, nheads, head_dim, device=device).to(dtype)
    dt = (0.5 * torch.randn(seqlen, nheads, device=device)).to(dtype)
    B = torch.randn(seqlen, ngroups, dstate, device=device).to(dtype)
    C = torch.randn(seqlen, ngroups, dstate, device=device).to(dtype)
    ref_out, ref_states = _reference(x, dt, A, B, C, D, dt_bias, chunk_size)

    # sequence lives in slot 1 (slot 0 is the null block)
    num_slots, slot = 3, 1
    buf_x = _paged(num_slots, (chunk_size, nheads, head_dim), dtype, device)
    buf_dt = _paged(num_slots, (chunk_size, nheads), dtype, device)
    buf_B = _paged(num_slots, (chunk_size, ngroups, dstate), dtype, device)
    ssm_state = torch.zeros(num_slots, nheads, head_dim, dstate, device=device)

    # scratch for one row
    dt_out = torch.empty(nheads, 1, chunk_size, dtype=torch.float32, device=device)
    dA_cumsum = torch.empty_like(dt_out)
    cb_emit = torch.empty(1, ngroups, chunk_size, dtype=torch.float32, device=device)
    row = torch.tensor([0], dtype=torch.int32, device=device)
    slots = torch.tensor([slot], dtype=torch.int32, device=device)

    for pos in range(seqlen):
        chunk, t = divmod(pos, chunk_size)
        if t == 0 and chunk > 0:
            # a chunk completed: the boundary state is the reference state
            ssm_state[slot] = ref_states[chunk - 1]
        # the step's own inputs go into the buffer at position t
        buf_x[slot, t] = x[pos]
        buf_dt[slot, t] = dt[pos]
        buf_B[slot, t] = B[pos]
        offsets = torch.tensor([t], dtype=torch.int32, device=device)
        out = torch.empty(1, nheads, head_dim, dtype=dtype, device=device)
        _workspace_chunk_cumsum_fwd(
            buf_dt,
            A,
            chunk_size,
            slots,
            offsets,
            dt_bias,
            dt_out=dt_out,
            dA_cumsum=dA_cumsum,
        )
        _bmm_chunk_workspace_range_fwd(
            C[pos : pos + 1],
            buf_B,
            chunk_size,
            slots,
            offsets,
            row,
            offsets,
            out=cb_emit,
        )
        _chunk_scan_workspace_range_fwd(
            cb_emit,
            buf_x,
            dt_out,
            dA_cumsum,
            C[pos : pos + 1],
            ssm_state,
            slots,
            slots,
            offsets,
            row,
            offsets,
            row,
            out,
            D=D,
        )
        assert torch.equal(out[0].view(torch.int16), ref_out[pos].view(torch.int16)), (
            f"position {pos} (chunk {chunk}, offset {t}) differs"
        )
