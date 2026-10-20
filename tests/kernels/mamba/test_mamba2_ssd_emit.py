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

import vllm.envs as envs
from tests.kernels.mamba.utils import carve_paged_states, single_shot_scan
from vllm.model_executor.layers.mamba.ops import (
    ssd_bmm,
    ssd_chunk_scan,
    ssd_chunk_state,
    ssd_state_passing,
)
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


@pytest.fixture(autouse=True, scope="module")
def _reference_on_batch_invariant_tiles():
    """The emit kernels always run the batch-invariant tiles and state update,
    while the reference scan autotunes unless VLLM_BATCH_INVARIANT is set at
    import and only uses the fused state update under it. Both change the bits,
    so run the reference the batch-invariant way."""
    pinned = [
        (ssd_bmm._bmm_chunk_fwd_kernel, ssd_bmm._BATCH_INVARIANT_CONFIG),
        (
            ssd_chunk_state._chunk_cumsum_fwd_kernel,
            ssd_chunk_state._CUMSUM_BATCH_INVARIANT_CONFIG,
        ),
        (
            ssd_chunk_state._chunk_state_fwd_kernel,
            ssd_chunk_state._CHUNK_STATE_BATCH_INVARIANT_CONFIG,
        ),
        (ssd_chunk_scan._chunk_scan_fwd_kernel, ssd_chunk_scan._BATCH_INVARIANT_CONFIG),
        (
            ssd_state_passing._state_passing_fwd_kernel,
            ssd_state_passing._BATCH_INVARIANT_CONFIG,
        ),
    ]
    saved = [(kernel, kernel.configs) for kernel, _ in pinned]
    for kernel, config in pinned:
        kernel.configs = [config]
        kernel.cache.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        yield
    for kernel, configs in saved:
        kernel.configs = configs
        kernel.cache.clear()


def _paged(num_slots, shape, dtype, device):
    return carve_paged_states(num_slots, [shape], [dtype], device)[0]


@pytest.mark.parametrize("chunk_size", [64, 128, 256])
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


@pytest.mark.parametrize("chunk_size", [64, 128, 256])
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


@pytest.mark.parametrize("chunk_size", [64, 128, 256])
@pytest.mark.parametrize("seed", [0, 1])
def test_emit_decode_schedule_matches_single_shot_prefill(chunk_size, seed):
    """Prefill through the replayed scan, then decode through emit, batched.

    Four sequences with prompts at different chunk offsets are prefilled with
    ``exact_replay_ssd`` (which also zeroes the state of sequences still in
    their first chunk) and then decoded one token per step with
    ``exact_replay_emit`` while the batch shrinks as sequences finish and a
    padding row (null block, position -1) rides along. Every produced row must
    equal the single-shot scan of the whole sequence, and the padding row must
    leave the null block untouched and output zeros.
    """
    from vllm.model_executor.layers.mamba.exact_replay import (
        ExactReplayBuffers,
        build_exact_replay_metadata,
        exact_replay_emit,
        exact_replay_ssd,
    )

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
            [computed[i] % chunk_size for i in active] + [-1],
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
        assert not out[-1].any()
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
    assert not any(buf[0].any() for buf in buffers)
    assert ssm_state[0].isnan().all()


@pytest.mark.parametrize("chunk_size", [64, 128])
@pytest.mark.parametrize("num_spec", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_emit_spec_decode_matches_single_shot_and_plain_decode(
    chunk_size, num_spec, seed
):
    """Speculative decode through emit keeps outputs and state exact.

    Each step a row carries its last sampled token and ``num_spec`` drafts;
    a draft is the true next token or a garbage one, and acceptance stops at
    the first garbage draft or at a draft on a chunk's last position, as the
    rejection sampler does in batch-invariant mode. The windows cross chunk
    boundaries at every offset. Every accepted row must equal the single-shot
    scan, and after every step the boundary state and the valid buffer prefix
    must equal those of plain one-token decode up to the same length.
    """
    from vllm.model_executor.layers.mamba.exact_replay import (
        ExactReplayBuffers,
        build_exact_replay_metadata,
        exact_replay_emit,
        exact_replay_emit_spec,
        exact_replay_spec_decode_layout,
        exact_replay_ssd,
    )

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
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
    kw = dict(A=A, D=D, dt_bias=dt_bias, chunk_size=chunk_size)

    total_lens = [2 * chunk_size + 41, 3 * chunk_size + 2, chunk_size - 3]
    prompt_lens = [chunk_size - 5, chunk_size + 1, 7]
    n = len(total_lens)

    def rand_inputs(L):
        return (
            torch.randn(L, nheads, head_dim, device=device).to(dtype),
            (0.5 * torch.randn(L, nheads, device=device)).to(dtype),
            torch.randn(L, ngroups, dstate, device=device).to(dtype),
            torch.randn(L, ngroups, dstate, device=device).to(dtype),
        )

    seqs = [rand_inputs(L) for L in total_lens]
    refs = [
        single_shot_scan(x, dt, A, B, C, D, dt_bias, chunk_size)[0]
        for x, dt, B, C in seqs
    ]

    # slot 0 is the null block; spec run in slots 1..n, plain run in n+1..2n
    num_slots = 2 * n + 1
    buffers = ExactReplayBuffers(
        _paged(num_slots, (chunk_size, nheads, head_dim), dtype, device),
        _paged(num_slots, (chunk_size, nheads), dtype, device),
        _paged(num_slots, (chunk_size, ngroups, dstate), dtype, device),
    )
    ssm_state = torch.full(
        (num_slots, nheads, head_dim, dstate), float("nan"), device=device
    )
    spec_slots = torch.arange(1, n + 1, dtype=torch.int32, device=device)
    plain_slots = spec_slots + n
    outs = [torch.empty_like(s[0]) for s in seqs]
    computed = [0] * n
    plain_computed = [0] * n

    def take(i, start, k):
        return [t[start : start + k] for t in seqs[i]]

    # prefill both runs
    for slots in (spec_slots, plain_slots):
        x, dt, B, C = (
            torch.cat(parts)
            for parts in zip(*(take(i, 0, prompt_lens[i]) for i in range(n)))
        )
        out = torch.empty_like(x)
        exact_replay_ssd(
            x,
            dt,
            B,
            C,
            out=out,
            ssm_state=ssm_state,
            slots=slots,
            meta=build_exact_replay_metadata([0] * n, prompt_lens, chunk_size, device),
            buffers=buffers,
            **kw,
        )
    for i in range(n):
        computed[i] = plain_computed[i] = prompt_lens[i]
        outs[i][: prompt_lens[i]] = refs[i][: prompt_lens[i]]

    def plain_step(i):
        x, dt, B, C = take(i, plain_computed[i], 1)
        out = torch.empty_like(x)
        exact_replay_emit(
            x,
            dt,
            B,
            C,
            out=out,
            ssm_state=ssm_state,
            slots=plain_slots[i : i + 1],
            pos=torch.tensor(
                [plain_computed[i] % chunk_size], dtype=torch.int32, device=device
            ),
            buffers=buffers,
            **kw,
        )
        plain_computed[i] += 1

    crossed_with_drafts = 0
    while any(computed[i] < total_lens[i] for i in range(n)):
        active = [i for i in range(n) if computed[i] < total_lens[i]]
        lens = [min(1 + num_spec, total_lens[i] - computed[i]) for i in active]
        good = []
        parts = []
        for i, q in zip(active, lens):
            true = take(i, computed[i], q)
            is_good = torch.rand(q, generator=gen) < 0.7
            is_good[0] = True
            garbage = rand_inputs(q)
            parts.append(
                [torch.where(_bcast(is_good, t), t, g) for t, g in zip(true, garbage)]
            )
            good.append(is_good)
        # one CUDA graph padding row with a full window of junk tokens
        parts.append(list(rand_inputs(1 + num_spec)))
        x, dt, B, C = (torch.cat(p) for p in zip(*parts))
        qsl = torch.tensor([0, *lens, 1 + num_spec], device=device).cumsum(0)
        seq_lens = torch.tensor(
            [computed[i] + q for i, q in zip(active, lens)] + [0], device=device
        )
        pos, token_idx = exact_replay_spec_decode_layout(
            seq_lens, qsl, num_spec, chunk_size
        )
        out = torch.empty_like(x)
        exact_replay_emit_spec(
            x,
            dt,
            B,
            C,
            out=out,
            ssm_state=ssm_state,
            slots=torch.cat([spec_slots[active], spec_slots.new_zeros(1)]),
            pos=pos,
            token_idx=token_idx,
            buffers=buffers,
            **kw,
        )
        assert not out[int(qsl[-2]) :].any()
        for row, (i, q) in enumerate(zip(active, lens)):
            accepted = 1
            while (
                accepted < q
                and good[row][accepted]
                and (computed[i] + accepted) % chunk_size != chunk_size - 1
            ):
                accepted += 1
            start = int(qsl[row])
            if (computed[i] + accepted - 1) // chunk_size != computed[i] // chunk_size:
                crossed_with_drafts += accepted > 1
            outs[i][computed[i] : computed[i] + accepted] = out[
                start : start + accepted
            ]
            computed[i] += accepted
            while plain_computed[i] < computed[i]:
                plain_step(i)
            s, p = int(spec_slots[i]), int(plain_slots[i])
            assert torch.equal(ssm_state[s], ssm_state[p]), (
                f"sequence {i}: boundary state differs at {computed[i]}"
            )
            tail = computed[i] % chunk_size
            for buf in buffers:
                assert torch.equal(buf[s, :tail], buf[p, :tail]), (
                    f"sequence {i}: partial chunk differs at {computed[i]}"
                )

    assert crossed_with_drafts > 0
    for i in range(n):
        assert torch.equal(outs[i].view(torch.int16), refs[i].view(torch.int16)), (
            f"sequence {i}: speculative emit decode diverged from single-shot"
        )
    assert not any(buf[0].any() for buf in buffers)
    assert ssm_state[0].isnan().all()


def _bcast(mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return mask.to(t.device).view(-1, *([1] * (t.dim() - 1)))
