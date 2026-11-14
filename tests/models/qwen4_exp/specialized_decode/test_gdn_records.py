# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN record/commit state protocol (`gdn/commit.py`,
`runtime/gdn_records.py`) against vLLM's protocol on the GPU.

The fused core's `records` verify against the stock core: outputs within one
BF16 unit, equal convolution state, no recurrent state written, equal copies
of the records of each request's rows in the record bands of its two copy
slots (in `align` mode avoiding the slot of a row that can be the boundary
row), the header naming them, and nothing else written; then the commit, for
every commit schedule, with and without the prefix-caching `align` boundary
row, with sampled counts 0 to `q`, short requests and FULL-graph padding: the
slot of the last accepted row and the boundary row's slot hold bit for bit
what the stock core wrote there, every other slot keeps its value outside the
record bands.

Then real decode steps of the executor's GDN layer (`GdnLayers` bound to a
`GdnRecords`, replayed from a FULL CUDA graph capture, committed after each
step) against the stock layer, with stock steps interleaved: outputs within
BF16 rounding step after step (each step reads the state the previous one
committed), and equal slots wherever the next reader looks. And the capture
check: a capture in which only some GDN layers launch the records core raises.
"""

from typing import TYPE_CHECKING, cast

import pytest
import torch

from vllm.platforms import current_platform

from .test_gdn import (
    HIDDEN,
    Caches,
    GdnModule,
    Q,
    _assert_outputs_match,
    _relative_error,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(100),
    reason="the specialized decode's GDN core requires SM100+",
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

if current_platform.is_cuda():
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn import (
        CommitSchedule,
        RecordLayout,
        Schedule,
        commit_tables,
        default_schedule,
        gdn_decode_core,
        gdn_state_commit,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gdn import GdnLayers
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gdn_records import (
        GdnRecords,
        boundary_row,
        boundary_rows,
        commit_rows,
        copy_slots,
    )

BLOCK = 16  # the mamba block size of align mode
CAPACITY = 16  # requests of the records buffers (one compiled layout per key heads)
COMMIT_SCHEDULES = [None, (4, 4), (2, 8, 2), (1, 8), (8, 4, 4), (2, 4, 4), (4, 8, 2)]


def _as_layers(*attns: GdnModule) -> "list[QwenGatedDeltaNetAttention]":
    """The stand-in GDN modules as the layers `GdnLayers` and `GdnRecords`
    take (they use only what `GdnModule` has)."""
    return cast("list[QwenGatedDeltaNetAttention]", list(attns))


def _records_core(
    caches: Caches, schedule, records, layout, seq_lens=None
) -> torch.Tensor:
    """The fused core's `records` verify of `caches`' step (in `align`
    mode with `seq_lens`)."""
    out = torch.zeros(
        caches.qkvz.shape[0], caches.hv * 128, dtype=torch.bfloat16, device="cuda"
    )
    return gdn_decode_core(
        caches.qkvz,
        caches.ba,
        caches.conv_weight,
        caches.conv_view(),
        caches.state,
        caches.a_log,
        caches.dt_bias,
        caches.norm_weight,
        caches.slots,
        caches.accepted,
        caches.cu,
        caches.requests,
        out,
        1e-6,
        None if schedule is None else Schedule(*schedule),
        records=records,
        layout=layout,
        seq_lens=seq_lens,
        block_size=BLOCK if seq_lens is not None else 0,
    )


def _align_inputs(
    counts: list[int], lengths: list[int], gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """`idx_mapping` (a permutation), the computed tokens after the step,
    `seq_lens` (the computed tokens before it plus the request's rows,
    `lengths`) and each request row's boundary row: request row `b`
    crosses a block boundary after folded row `b % (count + 1) - 1` (none
    for -1)."""
    states = 2 * len(counts)
    idx_mapping = torch.randperm(states, generator=gen)[: len(counts)]  # int64
    computed = torch.zeros(states, dtype=torch.int32)
    seq_lens = torch.zeros(len(counts), dtype=torch.int32)
    rows = []
    for b, count in enumerate(counts):
        bias = b % (count + 1) - 1
        after = 2 * BLOCK - 1 - bias + count if bias >= 0 else 2 * BLOCK + 8
        computed[idx_mapping[b]] = after
        seq_lens[b] = after - count + lengths[b]
        rows.append(boundary_row(count, after, BLOCK) if count else -1)
    return idx_mapping.cuda(), computed.cuda(), seq_lens.cuda(), rows


def _copies(caches: Caches, seq_lens: torch.Tensor | None) -> list[list[int]]:
    """Each request's copy slots (`copy_slots`; none for a request without
    a source slot), avoiding in `align` mode (with `seq_lens`) the slot of
    a row that can be the boundary row."""
    lengths = caches.cu.diff().tolist()
    copies: list[list[int]] = []
    for b in range(caches.requests):
        slots = caches.slots[b].tolist()
        source = slots[int(caches.accepted[b]) - 1]
        avoid: list[int] = []
        if seq_lens is not None:
            rows = lengths[b]
            avoid = boundary_rows(int(seq_lens[b]) - rows, rows, BLOCK)
        copies.append(copy_slots(slots, source, avoid) if source > 0 else [])
    return copies


def _recorded_words(layout: "RecordLayout", rows: int) -> torch.Tensor:
    """The words of a record band that the verify writes for a request of
    `rows` rows: the keys, decays and deltas of those rows."""
    mask = torch.zeros(layout.band_words, dtype=torch.bool, device="cuda")
    mask[layout.keys : layout.keys + rows * 128] = True
    mask[layout.decays : layout.decays + rows * 3] = True  # three value heads
    mask[layout.deltas : layout.deltas + rows * 3 * 128] = True
    return mask


def _check_slots(
    caches: Caches,
    state0: torch.Tensor,
    layout: "RecordLayout",
    written: set[int],
    all_copies: list[list[int]],
    headers: torch.Tensor,
) -> None:
    """Every slot not in `written` keeps its value of `state0`, except the
    record bands of each request's two copy slots (`all_copies`), which hold
    equal records of the request's rows and which its header names."""
    bands = caches.state.view(caches.state.shape[0], caches.kh, layout.band_stride)
    bands0 = state0.view_as(bands)
    words = layout.band_words
    lengths = caches.cu.diff().tolist()
    copied: set[int] = set()
    for b in range(caches.requests):
        slots = caches.slots[b].tolist()
        copies = all_copies[b]
        if not copies:
            continue
        assert len(copies) == 2, (b, copies)
        marked = sum(1 << t for t, slot in enumerate(slots) if slot in copies)
        header = headers[b * layout.header_stride : (b + 1) * layout.header_stride]
        assert int(header[layout.copies_word]) == marked, (b, header.tolist())
        copied.update(copies)
        kept = [slot for slot in copies if slot not in written]
        mask = _recorded_words(layout, lengths[b])
        for slot in kept:
            assert torch.equal(bands[slot, :, words:], bands0[slot, :, words:]), slot
            records, records0 = bands[slot, :, :words], bands0[slot, :, :words]
            assert torch.equal(records[:, ~mask], records0[:, ~mask]), slot
            assert not torch.equal(records[:, mask], records0[:, mask]), slot
            assert torch.equal(records[:, mask], bands[kept[0], :, :words][:, mask])
    for slot in range(caches.state.shape[0]):
        if slot not in written and slot not in copied:
            assert torch.equal(caches.state[slot], state0[slot]), slot


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("commit_schedule", COMMIT_SCHEDULES)
@pytest.mark.parametrize(
    "key_heads,lengths,empty,schedule",
    [
        (4, [4], 0, None),
        (4, [4, 2, 4, 1, 3], 3, (1, 1, 8, 2)),
        (16, [4] * 3, 1, None),
        (16, [4] * 9, 0, (3, 4, 12)),
    ],
)
def test_records_and_commit_match_the_stock_core(
    key_heads, lengths, empty, schedule, commit_schedule, align
):
    caches = Caches(
        key_heads, len(lengths), seed=len(lengths), lengths=lengths, empty=empty
    )
    conv0, state0 = caches.conv.clone(), caches.state.clone()
    expected = caches.stock()
    expected_conv, expected_state = caches.conv.clone(), caches.state.clone()
    caches.conv.copy_(conv0)
    caches.state.copy_(state0)

    gen = torch.Generator().manual_seed(len(lengths) + 100 * align)
    sampled = torch.randint(0, Q + 1, (caches.requests,), generator=gen).int()
    rows = lengths + [0] * empty
    counts = [commit_rows(int(s), r) for s, r in zip(sampled, rows)]
    idx_mapping = computed = seq_lens = None
    boundaries = [-1] * caches.requests
    if align:
        idx_mapping, computed, seq_lens, boundaries = _align_inputs(counts, rows, gen)

    layout = RecordLayout(CAPACITY, Q, key_heads)
    headers = layout.allocate(torch.device("cuda"))
    headers.fill_(-1)  # stale words: the verify rewrites every word the commit reads
    actual = _records_core(caches, schedule, headers[0], layout, seq_lens)
    _assert_outputs_match(actual, expected)
    assert torch.equal(caches.conv, expected_conv)
    copies = _copies(caches, seq_lens)
    _check_slots(caches, state0, layout, set(), copies, headers[0])

    gdn_state_commit(
        commit_tables([caches.state], headers),
        caches.requests,
        sampled.cuda(),
        layout,
        None if commit_schedule is None else CommitSchedule(*commit_schedule),
        idx_mapping,
        computed,
        BLOCK if align else 0,
    )
    torch.accelerator.synchronize()

    written = set()
    for b, count in enumerate(counts):
        for row in (count - 1, boundaries[b]):
            slot = int(caches.slots[b, row]) if row >= 0 else 0
            if slot > 0:
                written.add(slot)
                assert torch.equal(caches.state[slot], expected_state[slot]), (b, row)
    _check_slots(caches, state0, layout, written, copies, headers[0])


def _step_inputs(requests: int, gen: torch.Generator):
    """A step's sampled counts and, for `align` mode, its request state
    indices and computed tokens."""
    sampled = torch.randint(0, Q + 1, (requests,), generator=gen).int()
    computed = torch.randint(Q, 4 * BLOCK, (requests,), generator=gen).int()
    return sampled.cuda(), torch.arange(requests).cuda(), computed.cuda()


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("key_heads", [4, 16])
def test_records_steps_of_the_gdn_layers_match_the_stock_layer(key_heads, align):
    requests, empty = 3, 1
    lengths = [Q] * requests
    ref = GdnModule(Caches(key_heads, requests, seed=9, lengths=lengths, empty=empty))
    cand = GdnModule(Caches(key_heads, requests, seed=9, lengths=lengths, empty=empty))
    gdn = GdnLayers(
        _as_layers(cand),
        Q,
        CAPACITY,
        gdn_decode_core,
        default_schedule,
        torch.cuda.Stream(),
    )
    records = GdnRecords(_as_layers(cand), Q, CAPACITY, key_heads, align)
    gdn.bind_records(records)
    gdn.warm_up()
    records.warm_up()
    records.bind_caches(BLOCK)
    # The runner's seq_lens buffer, which the align-mode verify reads.
    seq_lens = torch.zeros(CAPACITY, dtype=torch.int32, device="cuda")
    records.bind_seq_lens(seq_lens)

    metadata = cand.caches.metadata()
    rows = Q * (requests + empty)
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    # One eager call first, so that cuBLAS initializes outside the capture. The
    # records core writes no recurrent state; the convolution state is restored.
    conv0 = cand.caches.conv.clone()
    gdn(cand, hidden, metadata)
    cand.caches.conv.copy_(conv0)
    graph = torch.cuda.CUDAGraph()
    records.start_capture()
    with torch.cuda.graph(graph):
        out = gdn(cand, hidden, metadata)
    records.finish_capture(rows)
    assert records.records_step(rows)

    gen = torch.Generator().manual_seed(key_heads + align)
    real = Q * requests
    total = requests + empty
    for step in range(10):
        hidden.copy_(torch.randn(rows, HIDDEN, generator=gen).bfloat16())
        expected = ref(hidden)
        sampled, idx_mapping, computed = _step_inputs(total, gen)
        counts = [commit_rows(int(s), Q) for s in sampled[:requests]]
        # The computed tokens before the step plus its rows.
        seq_lens[:requests] = (
            computed[:requests]
            - torch.tensor(counts, dtype=torch.int32, device="cuda")
            + Q
        )
        if step in (3, 4, 7):  # a stock step (mixed batch or eager) in between
            actual = cand(hidden)
        else:
            graph.replay()
            records.commit(total, sampled, idx_mapping, computed)
            actual = out
        torch.accelerator.synchronize()
        assert _relative_error(actual[:real], expected[:real]) < 2**-7, step
        assert torch.equal(cand.caches.conv, ref.caches.conv), step
        for b, count in enumerate(counts):
            boundary = boundary_row(count, int(computed[b]), BLOCK) if align else -1
            for row in {count - 1, boundary} - {-1}:
                slot = int(cand.caches.slots[b, row])
                assert torch.equal(cand.caches.state[slot], ref.caches.state[slot]), (
                    step,
                    b,
                    row,
                )
        # The stock postprocess: the next step reads slot max(sampled, 1) - 1.
        accepted = torch.ones(total, dtype=torch.int32, device="cuda")
        accepted[:requests] = sampled[:requests].clamp(min=1)
        ref.caches.accepted.copy_(accepted)
        cand.caches.accepted.copy_(accepted)


def test_a_capture_with_the_records_core_in_some_gdn_layers_raises():
    layers = [GdnModule(Caches(4, 2, seed=s, lengths=[Q, Q])) for s in (1, 2)]
    gdn = GdnLayers(
        _as_layers(*layers), Q, CAPACITY, gdn_decode_core, default_schedule, None
    )
    records = GdnRecords(_as_layers(*layers), Q, CAPACITY, 4, align=False)
    gdn.bind_records(records)
    gdn.warm_up()
    records.bind_caches(0)
    hidden = torch.zeros(2 * Q, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def capture(attns) -> None:
        graph = torch.cuda.CUDAGraph()
        records.start_capture()
        with torch.cuda.graph(graph):
            for attn in attns:
                gdn(attn, hidden, attn.caches.metadata())

    for attn in layers:  # eagerly first: cuBLAS initializes outside the capture
        gdn(attn, hidden, attn.caches.metadata())
    capture(layers)
    records.finish_capture(2 * Q)
    assert records.records_step(2 * Q) and not records.records_step(Q)
    capture(layers[:1])
    with pytest.raises(RuntimeError, match="1 of 2 GDN layers"):
        records.finish_capture(Q)
    # A FULL graph of a recorded size without the records core.
    with pytest.raises(RuntimeError, match="with and without"):
        records.finish_capture(2 * Q)
