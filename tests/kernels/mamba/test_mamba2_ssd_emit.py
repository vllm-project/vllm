# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row-gated emit and fold reproduce the single-shot chunked scan bit for bit.

A sequence is scanned once with the full SSD pipeline (reference). Then every
token is produced again the way batch-invariant decode would: the fp32 state at
the last chunk boundary plus the buffered inputs of the active chunk, held in
paged ``buf[slot, pos]`` views, feed the row-gated emit kernels for that one
position, and the fold kernel advances the boundary state when a chunk
completes. Every emitted row and every boundary state must equal the reference.
"""

import pytest
import torch

from tests.kernels.mamba.utils import carve_paged_states, single_shot_scan
from vllm.model_executor.layers.mamba.ops.ssd_emit import (
    _bmm_chunk_workspace_range_fwd,
    _chunk_scan_workspace_range_fwd,
    _fold_chunk_fwd,
    _workspace_chunk_cumsum_fwd,
)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Mamba2 SSD Triton kernels require a CUDA-alike device.",
)


def _paged(num_slots, shape, dtype, device):
    return carve_paged_states(num_slots, [shape], [dtype], device)[0]


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
    ref_out, ref_states = single_shot_scan(x, dt, A, B, C, D, dt_bias, chunk_size)

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
    slots = torch.tensor([slot], dtype=torch.int32, device=device)

    for pos in range(seqlen):
        chunk, t = divmod(pos, chunk_size)
        if t == 0 and chunk > 0:
            # a chunk completed: the boundary state is the reference state
            ssm_state[slot] = ref_states[chunk - 1]
        # the kernels store the step's own inputs into the buffer at position t
        offsets = torch.tensor([t], dtype=torch.int32, device=device)
        out = torch.empty(1, nheads, head_dim, dtype=dtype, device=device)
        _workspace_chunk_cumsum_fwd(
            buf_dt,
            A,
            chunk_size,
            slots,
            offsets,
            dt_bias,
            current_dt=dt[pos : pos + 1],
            dt_out=dt_out,
            dA_cumsum=dA_cumsum,
        )
        _bmm_chunk_workspace_range_fwd(
            C[pos : pos + 1],
            buf_B,
            chunk_size,
            slots,
            offsets,
            cb_emit,
            B[pos : pos + 1],
        )
        _chunk_scan_workspace_range_fwd(
            cb_emit,
            buf_x,
            dt_out,
            dA_cumsum,
            C[pos : pos + 1],
            ssm_state,
            slots,
            offsets,
            out,
            D=D,
            current_x=x[pos : pos + 1],
        )
        assert torch.equal(out[0].view(torch.int16), ref_out[pos].view(torch.int16)), (
            f"position {pos} (chunk {chunk}, offset {t}) differs"
        )
        assert torch.equal(buf_x[slot, t], x[pos])
        assert torch.equal(buf_dt[slot, t], dt[pos])
        assert torch.equal(buf_B[slot, t], B[pos])


@pytest.mark.parametrize("chunk_size", [64])
def test_fold_matches_single_shot_chunk_states(chunk_size):
    """Folding a completed chunk gives the reference boundary state bit for bit.

    Rows whose token is not the chunk's last position and rows with a negative
    slot must leave the state untouched: that is what lets the fold run over
    every row of a padded decode batch.
    """
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    nheads, head_dim, ngroups, dstate = 8, 64, 1, 64
    dtype = torch.bfloat16
    n_chunks = 3
    seqlen = n_chunks * chunk_size

    # Realistic decay: exp(dA over a chunk) must stay well above zero, or the
    # previous state drops out of the update and its rounding goes untested.
    A = -torch.exp(torch.randn(nheads, device=device))
    dt_bias = torch.log(torch.expm1(torch.rand(nheads, device=device) * 0.09 + 0.01))
    D = torch.ones(nheads, device=device)
    x = torch.randn(seqlen, nheads, head_dim, device=device).to(dtype)
    dt = (0.3 * torch.randn(seqlen, nheads, device=device)).to(dtype)
    B = torch.randn(seqlen, ngroups, dstate, device=device).to(dtype)
    C = torch.randn(seqlen, ngroups, dstate, device=device).to(dtype)
    _, ref_states = single_shot_scan(x, dt, A, B, C, D, dt_bias, chunk_size)
    dA_chunk = (torch.nn.functional.softplus(dt.float() + dt_bias) * A).view(
        n_chunks, chunk_size, nheads
    )
    assert dA_chunk.sum(1).exp().max() > 0.1

    # rows: 0 folds chunk c, 1 has the same inputs but is mid-chunk, 2 is padding
    num_slots = 3
    buf_x = _paged(num_slots, (chunk_size, nheads, head_dim), dtype, device)
    buf_dt = _paged(num_slots, (chunk_size, nheads), dtype, device)
    buf_B = _paged(num_slots, (chunk_size, ngroups, dstate), dtype, device)
    ssm_state = torch.zeros(num_slots, nheads, head_dim, dstate, device=device)
    slots = torch.tensor([1, 2, -1], dtype=torch.int32, device=device)
    offsets = torch.tensor(
        [chunk_size - 1, chunk_size - 2, chunk_size - 1],
        dtype=torch.int32,
        device=device,
    )
    dt_out = torch.empty(nheads, 3, chunk_size, dtype=torch.float32, device=device)
    dA_cumsum = torch.empty_like(dt_out)
    for c in range(n_chunks):
        lo = c * chunk_size
        for slot in (1, 2):
            buf_x[slot] = x[lo : lo + chunk_size]
            buf_dt[slot] = dt[lo : lo + chunk_size]
            buf_B[slot] = B[lo : lo + chunk_size]
        untouched = torch.randn_like(ssm_state[2])
        ssm_state[2] = untouched
        # each row's own dt is the buffered value at its position
        current_dt = torch.stack(
            [
                buf_dt[1, chunk_size - 1],
                buf_dt[2, chunk_size - 2],
                buf_dt[2, chunk_size - 1],
            ]
        )
        _workspace_chunk_cumsum_fwd(
            buf_dt,
            A,
            chunk_size,
            slots,
            offsets,
            dt_bias,
            current_dt=current_dt,
            dt_out=dt_out,
            dA_cumsum=dA_cumsum,
        )
        _fold_chunk_fwd(buf_x, buf_B, dt_out, dA_cumsum, ssm_state, slots, offsets)
        assert torch.equal(ssm_state[1], ref_states[c]), f"chunk {c} state differs"
        assert torch.equal(ssm_state[2], untouched)
        assert torch.equal(ssm_state[0], torch.zeros_like(ssm_state[0]))


@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("seed", [0, 1])
def test_emit_decode_schedule_matches_single_shot_prefill(chunk_size, seed):
    """Prefill through the replayed scan, then decode through emit, batched.

    Four sequences with prompts at different chunk offsets are prefilled with
    ``exact_replay_ssd`` (which also zeroes the state of sequences still in
    their first chunk) and then decoded one token per step with
    ``exact_replay_emit`` while the batch shrinks as sequences finish and a
    padding row pointing at the null block rides along. Every produced row must
    equal the single-shot scan of the whole sequence.
    """
    from vllm.model_executor.layers.mamba.exact_replay import (
        ExactReplayBuffers,
        exact_replay_emit,
        exact_replay_ssd,
    )
    from vllm.v1.attention.backends.mamba2_attn import build_exact_replay_metadata

    torch.manual_seed(seed)
    device = torch.device(DEVICE)
    nheads, head_dim, ngroups, dstate = 8, 64, 1, 64
    dtype = torch.bfloat16
    A = -(torch.rand(nheads, device=device) * 15 + 1)
    dt_target = torch.exp(
        torch.rand(nheads, device=device)
        * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(1e-3)))
        + torch.log(torch.tensor(1e-3))
    )
    dt_bias = dt_target + torch.log(-torch.expm1(-dt_target))
    D = torch.ones(nheads, device=device)

    total_lens = [150, 210, 30, 130]
    prompt_lens = [100, 133, 20, 64]
    n = len(total_lens)
    xs = [torch.randn(L, nheads, head_dim, device=device).to(dtype) for L in total_lens]
    dts = [(0.5 * torch.randn(L, nheads, device=device)).to(dtype) for L in total_lens]
    Bs = [torch.randn(L, ngroups, dstate, device=device).to(dtype) for L in total_lens]
    Cs = [torch.randn(L, ngroups, dstate, device=device).to(dtype) for L in total_lens]
    refs = [
        single_shot_scan(xs[i], dts[i], A, Bs[i], Cs[i], D, dt_bias, chunk_size)[0]
        for i in range(n)
    ]

    num_slots = n + 1  # slot 0 is the null block
    buffers = ExactReplayBuffers(
        _paged(num_slots, (chunk_size, nheads, head_dim), dtype, device),
        _paged(num_slots, (chunk_size, nheads), dtype, device),
        _paged(num_slots, (chunk_size, ngroups, dstate), dtype, device),
    )
    ssm_state = torch.full(
        (num_slots, nheads, head_dim, dstate), float("nan"), device=device
    )
    all_slots = torch.arange(1, n + 1, dtype=torch.int32, device=device)
    outs = [torch.empty_like(x) for x in xs]
    computed = [0] * n

    def gather(seqs, active, lens):
        return torch.cat(
            [seqs[i][computed[i] : computed[i] + k] for i, k in zip(active, lens)]
        )

    def prefill(active, lens):
        x, dt, B, C = (gather(s, active, lens) for s in (xs, dts, Bs, Cs))
        slots = all_slots[torch.tensor(active, device=device)]
        meta = build_exact_replay_metadata(
            [computed[i] for i in active], lens, chunk_size, device
        )
        out = torch.empty_like(x)
        exact_replay_ssd(
            x,
            dt,
            B,
            C,
            A=A,
            D=D,
            dt_bias=dt_bias,
            out=out,
            ssm_state=ssm_state,
            slots=slots,
            meta=meta,
            chunk_size=chunk_size,
            buffers=buffers,
        )
        return out

    def decode(active):
        # one padding row at the end, as a CUDA graph batch would have
        lens = [1] * len(active)
        x, dt, B, C = (gather(s, active, lens) for s in (xs, dts, Bs, Cs))
        pad = lambda t: torch.cat([t, torch.randn_like(t[:1])])  # noqa: E731
        x, dt, B, C = pad(x), pad(dt), pad(B), pad(C)
        slots = torch.cat(
            [all_slots[torch.tensor(active, device=device)], all_slots.new_zeros(1)]
        )
        pos = torch.tensor(
            [computed[i] % chunk_size for i in active] + [0],
            dtype=torch.int32,
            device=device,
        )
        out = torch.empty_like(x)
        exact_replay_emit(
            x,
            dt,
            B,
            C,
            A=A,
            D=D,
            dt_bias=dt_bias,
            out=out,
            ssm_state=ssm_state,
            slots=slots,
            pos=pos,
            chunk_size=chunk_size,
            buffers=buffers,
        )
        return out[:-1]

    def record(active, lens, out):
        off = 0
        for i, k in zip(active, lens):
            outs[i][computed[i] : computed[i] + k] = out[off : off + k]
            off += k
            computed[i] += k

    record(list(range(n)), prompt_lens, prefill(list(range(n)), prompt_lens))
    while any(computed[i] < total_lens[i] for i in range(n)):
        active = [i for i in range(n) if computed[i] < total_lens[i]]
        record(active, [1] * len(active), decode(active))

    for i in range(n):
        assert torch.equal(outs[i].view(torch.int16), refs[i].view(torch.int16)), (
            f"sequence {i}: emit decode diverged from single-shot prefill"
        )
