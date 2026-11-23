# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning of the overlapped PCP+DCP KV gather: a wrong split reads rows the chunk
is still writing, and ranks that disagree wait on each other's barriers forever."""

import dataclasses
import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.v1.attention import kv_ce_triton_launch as launch
from vllm.config import ParallelConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops import kv_ce_overlap
from vllm.v1.attention.ops.kv_ce_overlap import (
    KVCEOverlapPlan,
    LayerOrder,
    build_kv_ce_overlap_plan,
    dcp_min_shard_rows,
    dcp_shard_rows,
    predict_rows,
    shard_lens,
)


def _shard_rows(seq_len: int, dcp_size: int, interleave: int) -> list[int]:
    lens = get_dcp_local_seq_lens(torch.tensor([seq_len]), dcp_size, None, interleave)
    return lens.flatten().tolist()


def _plan(**overrides):
    kwargs = {
        "num_chunks": 1,
        "num_entries": 1,
        "num_decode_tokens": 0,
        "shard_rows": 256_000,
        "context_start": 1_007_616,
        "dcp_size": 4,
        "interleave": 1,
        "shard_capacity": 524_288,
        "gathered_capacity": 2_097_152,
    }
    return build_kv_ce_overlap_plan(**{**kwargs, **overrides})


def _parallel_config(pcp_size: int, dcp_size: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=2,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        pcp_dcp_kv_prefetch=True,
        distributed_executor_backend="mp",
    )


@pytest.mark.parametrize("pcp_size, dcp_size", [(1, 1), (1, 2), (2, 1)])
def test_prefetch_requires_pcp_and_dcp(pcp_size, dcp_size):
    with pytest.raises(ValueError, match="pcp_dcp_kv_prefetch requires"):
        _parallel_config(pcp_size, dcp_size)


@pytest.mark.parametrize("dcp_size", [2, 3, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 64, 256])
def test_min_shard_rows_is_the_shortest_rank(dcp_size, interleave):
    for seq_len in [*range(0, 3 * dcp_size * interleave), 1_024_000, 1_007_617]:
        assert dcp_min_shard_rows(seq_len, dcp_size, interleave) == min(
            _shard_rows(seq_len, dcp_size, interleave)
        )


def _token_rows(
    tokens: np.ndarray, dcp_size: int, interleave: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """(DCP rank, shard row) of each token, by the slot mapping of the KV cache."""
    virtual = block_size * dcp_size
    offsets = tokens % virtual
    ranks = offsets // interleave % dcp_size
    rows = (
        tokens // virtual * block_size
        + offsets // (interleave * dcp_size) * interleave
        + offsets % interleave
    )
    return ranks, rows


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 16, 64])
@pytest.mark.parametrize(
    "start, chunk",
    [(1_007_616, 16_384), (1_019_904, 4_096), (65_537, 4_097), (1_000, 3)],
)
def test_split_keeps_every_chunk_row_out_of_the_head(
    dcp_size, interleave, start, chunk
):
    """The head is read while the chunk writes its rows: it must hold only rows
    of tokens before the chunk, on every rank."""
    block_size = 64
    end = start + chunk
    ranks, rows = _token_rows(np.arange(end), dcp_size, interleave, block_size)
    shard_rows = int(np.bincount(ranks, minlength=dcp_size).max())
    plan, reason = _plan(
        shard_rows=shard_rows,
        context_start=start,
        dcp_size=dcp_size,
        interleave=interleave,
        shard_capacity=shard_rows,
        gathered_capacity=2 * dcp_size * shard_rows,
    )
    assert reason == "planned"
    assert rows[start:].min() >= plan.head_rows
    old_lens = []
    for rank in range(dcp_size):
        old_rows = np.sort(rows[:start][ranks[:start] == rank])
        assert np.array_equal(old_rows, np.arange(old_rows.size))
        old_lens.append(old_rows.size)
    # As long as possible: on the shortest rank, the next row is the chunk's.
    assert plan.head_rows == min(old_lens)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"num_chunks": 2}, "multiple_chunks_or_requests"),
        ({"num_entries": 2}, "multiple_chunks_or_requests"),
        ({"num_decode_tokens": 1}, "decode_rows"),
        ({"context_start": None}, "no_context"),
        ({"context_start": 0}, "no_context"),
        ({"context_start": 3}, "no_context"),
        ({"shard_capacity": 255_999}, "shard_capacity"),
        ({"gathered_capacity": 2_047_999}, "two_layers_do_not_fit"),
    ],
)
def test_ineligible_steps_stay_serial(overrides, reason):
    assert _plan(**overrides) == (None, reason)


def test_two_layers_of_a_full_context_fit_exactly():
    plan, reason = _plan(gathered_capacity=2_048_000)
    assert reason == "planned"
    assert plan == KVCEOverlapPlan(256_000, 251_904, 1_007_616)


def _forward(order: LayerOrder, names: str) -> list[tuple[int | None, str | None]]:
    """Position of each layer in one forward, and the layer the order expects
    next."""
    out = []
    for i, name in enumerate(names):
        position = order.advance(name, new_forward=i == 0)
        following = None
        if order.in_order and order.cursor < len(order.names):
            following = order.names[order.cursor]
        out.append((position, following))
    return out


ORDERED = [(0, "b"), (1, "c"), (2, "d"), (3, "e"), (4, None)]
UNKNOWN = [(None, None)] * 5


def test_layer_order_is_learned_from_a_whole_forward():
    order = LayerOrder()
    assert _forward(order, "abcde") == UNKNOWN
    assert _forward(order, "abcde") == ORDERED
    assert _forward(order, "abcde") == ORDERED


def test_after_an_out_of_order_forward_position_0_is_the_first_layer_again():
    """A forward missing a layer once left the last layer at position 0, so it
    prefetched the next forward's first layer with this forward's plan."""
    order = LayerOrder()
    _forward(order, "abcde")
    assert _forward(order, "abde") == [(0, "b"), (1, "c"), (None, None), (None, None)]
    assert _forward(order, "abcde") == UNKNOWN
    assert _forward(order, "abcde") == ORDERED
    assert _forward(order, "abcde") == ORDERED


@pytest.mark.parametrize("odd", ["abc", "bcde", "abcdef", "abcdea"])
def test_a_forward_that_leaves_the_order_is_learned_again(odd):
    """Stopping early, starting elsewhere or running past the end."""
    order = LayerOrder()
    _forward(order, "abcde")
    _forward(order, odd)
    assert _forward(order, "abcde") == UNKNOWN
    assert _forward(order, "abcde") == ORDERED


def test_a_layer_twice_in_one_forward_is_never_followed():
    order = LayerOrder()
    for _ in range(3):
        assert _forward(order, "abca") == [(None, None)] * 4


def _chunk(shard_rows: int, context_start: int | None):
    chunk_type = FlashMLASparseMetadata.FP8SeparatePrefillDecode.Prefill.Chunk
    return chunk_type(
        tokens_slice=slice(0, 1),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        req_start_idx=0,
        workspace_starts=torch.zeros(1, dtype=torch.int32),
        chunk_tot_seqlen=shard_rows,
        context_start=context_start,
    )


def test_a_plan_serves_only_chunks_with_its_layout():
    plan = KVCEOverlapPlan(256_000, 251_904, 1_007_616)
    assert plan.serves(_chunk(256_000, 1_007_616))
    # The next chunk of the same request: one more row per rank.
    assert not plan.serves(_chunk(260_096, 1_024_000))
    assert not plan.serves(_chunk(256_000, 1_007_615))
    assert not plan.serves(_chunk(256_000, None))


@pytest.mark.parametrize("dcp_size", [2, 3, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 64])
@pytest.mark.parametrize(
    "start, chunk", [(1_007_616, 16_384), (65_537, 4_097), (1_000, 3), (5, 64)]
)
def test_shard_lens_are_every_ranks_shard_before_and_after(
    dcp_size, interleave, start, chunk
):
    block_size = 64
    end = start + chunk
    ranks, _ = _token_rows(np.arange(end), dcp_size, interleave, block_size)
    old = np.bincount(ranks[:start], minlength=dcp_size)
    new = np.bincount(ranks, minlength=dcp_size)
    lens = shard_lens(start, end, dcp_size, interleave)
    assert lens == (tuple(old.tolist()), tuple(new.tolist()))
    assert max(lens[1]) == dcp_shard_rows(end, dcp_size, 0, interleave)


def _pcp_restore(chunk: int, pcp: int) -> tuple[torch.Tensor, int]:
    """Gathered row of each chunk token under the load-balanced PCP split: rank
    `r` holds pieces `r` and `2 * pcp - 1 - r`, each rank padded to the largest
    share."""
    bounds = np.linspace(0, chunk, 2 * pcp + 1).astype(int)
    shares = [
        list(range(bounds[r], bounds[r + 1]))
        + list(range(bounds[2 * pcp - 1 - r], bounds[2 * pcp - r]))
        for r in range(pcp)
    ]
    padded = max(len(share) for share in shares)
    restore = torch.empty(chunk, dtype=torch.int64)
    for rank, share in enumerate(shares):
        restore[share] = rank * padded + torch.arange(len(share))
    return restore, pcp * padded


def stage_slot_map(
    context_start: int,
    seq_end: int,
    restore_idx: torch.Tensor,
    rows: int,
    dcp_size: int,
    interleave: int,
    per_rank: int,
) -> torch.Tensor:
    """The scratch slot of each PCP-gathered new row: the reference of
    `kv_ce_rows.stage_slots`.

    Token `context_start + g` is gathered row `restore_idx[g]`. DCP rank `p`'s
    new rows go in token order to slots `[p * per_rank, p * per_rank + new_p - old_p)`
    (the cache slot mapping's shard row minus `old_p`); rows no token maps to (PCP
    padding) get -1, which the cache-write kernel skips.
    """
    device = restore_idx.device
    tokens = torch.arange(context_start, seq_end, dtype=torch.int64, device=device)
    owner = tokens // interleave % dcp_size
    shard_row = tokens // (interleave * dcp_size) * interleave + tokens % interleave
    base = context_start // interleave // dcp_size * interleave
    remainder = context_start - base * dcp_size
    ranks = torch.arange(dcp_size, dtype=torch.int64, device=device)
    old = base + (remainder - ranks * interleave).clamp(0, interleave)
    slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    slots[restore_idx] = owner * per_rank + shard_row - old[owner]
    return slots


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 16, 64])
@pytest.mark.parametrize(
    "start, chunk", [(1_007_616, 16_384), (1_019_904, 4_096), (65_537, 4_097)]
)
def test_staged_rows_land_where_the_serial_gather_puts_them(
    dcp_size, interleave, start, chunk
):
    """Quantize into the scratch by slot, upconvert each rank's range into its slot
    range: every new row ends where the rank-major gather of the shards has it."""
    block_size = 64
    end = start + chunk
    ranks, rows = _token_rows(np.arange(end), dcp_size, interleave, block_size)
    old, new = shard_lens(start, end, dcp_size, interleave)
    shard = max(new)
    restore, gathered_rows = _pcp_restore(chunk, dcp_size)
    per_rank = -(-max(n - o for n, o in zip(new, old)) // block_size) * block_size
    slots = stage_slot_map(
        start, end, restore, gathered_rows, dcp_size, interleave, per_rank
    )
    assert int((slots >= 0).sum()) == chunk
    # A gathered row holds its token id; padding rows hold -2.
    gathered = torch.full((gathered_rows,), -2, dtype=torch.int64)
    gathered[restore] = torch.arange(start, end)
    scratch = torch.full((dcp_size * per_rank,), -1, dtype=torch.int64)
    keep = slots >= 0
    scratch[slots[keep]] = gathered[keep]
    region = torch.full((dcp_size * shard,), -1, dtype=torch.int64)
    for rank in range(dcp_size):
        count = new[rank] - old[rank]
        start_row = rank * shard + old[rank]
        region[start_row : start_row + count] = scratch[
            rank * per_rank : rank * per_rank + count
        ]
    expected = torch.full((dcp_size * shard,), -1, dtype=torch.int64)
    tokens = np.arange(start, end)
    expected[torch.from_numpy(ranks[start:] * shard + rows[start:])] = torch.from_numpy(
        tokens
    )
    assert torch.equal(region, expected)


def test_ranks_that_pull_the_union_elsewhere_do_not_share_a_plan():
    """A union build adds a SEL_MARKS barrier and a compact head pairs its copies
    with its peers' packs: the digest must tell where both happen."""
    plan = KVCEOverlapPlan(12, 10, 40)
    compact = dataclasses.replace(
        plan, sel_positions=frozenset({3, 4}), sel_builds=frozenset({3})
    )
    built_later = dataclasses.replace(compact, sel_builds=frozenset({4}))
    digests = [plan.digest(), compact.digest(), built_later.digest()]
    assert len({tuple(digest) for digest in digests}) == 3


def test_a_prediction_adds_a_margin_to_the_last_union_sizes():
    # 10% and 64 rows are added to the last sizes seen, at most the head.
    assert predict_rows([100, 0, 1000, 3800], 4096, 4) == [174, 64, 1164, 4096]
    # A new request starts from the sizes of the last one, capped at its head.
    assert predict_rows([100, 0, 1000, 3800], 512, 4) == [174, 64, 512, 512]


def test_without_history_the_whole_head_is_copied():
    """The whole head needs no top-up: every union row lies below it."""
    assert predict_rows(None, 4096, 4) == [4096] * 4


def test_chunks_cut_only_at_rows_and_may_span_copies():
    """No chunk exceeds the chunk size; the chunks still move every byte of every
    copy exactly once, in order."""
    row = 16
    copies = [(0x1000, 0x9000, 5 * row), (0x2000, 0xA000, 2 * row)]
    copies.append((0x3000, 0xB000, 7 * row))
    chunks = kv_ce_overlap.chunk_copies(copies, 4 * row + 3, row)
    assert all(sum(size for *_, size in chunk) <= 4 * row for chunk in chunks)
    pieces = [piece for chunk in chunks for piece in chunk]
    assert all(size % row == 0 and size > 0 for *_, size in pieces)
    moved = []
    for destination, source, size in copies:
        offset = 0
        while offset < size:
            piece = pieces.pop(0)
            assert piece[:2] == (destination + offset, source + offset)
            offset += piece[2]
        moved.append(offset)
    assert moved == [size for *_, size in copies] and not pieces
    assert any(len(chunk) > 1 for chunk in chunks)
    assert kv_ce_overlap.chunk_copies(copies, row // 2, row)[0] == [
        (0x1000, 0x9000, row)
    ]


class _CopyDriver:
    """Records the copies `KVCEOverlap._copy` enqueues, as the driver sees them."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def memcpy_list(self, copies, stream: int) -> None:
        self.calls.append((tuple(copies.copies), stream))


def test_a_head_goes_out_one_call_per_chunk_on_the_current_stream(monkeypatch):
    """Every chunk is one call on the stream the head is issued on (the auxiliary
    stream), in order, so the work queued behind the head waits for all of it."""
    after = 5
    monkeypatch.setattr(
        kv_ce_overlap, "current_stream", lambda: SimpleNamespace(cuda_stream=after)
    )
    overlap = kv_ce_overlap.KVCEOverlap.__new__(kv_ce_overlap.KVCEOverlap)
    driver = _CopyDriver()
    overlap.driver = driver
    row = kv_ce_overlap.KV_ENTRY_BYTES
    per_chunk = kv_ce_overlap.COPY_CHUNK_BYTES // row
    copies = [(0x100, 1 << 40, (2 * per_chunk + 7) * row), (1 << 41, 1 << 42, row)]
    chunks = kv_ce_overlap.head_chunks(copies, row)
    assert len(chunks) == 3
    overlap._copy(chunks)
    assert driver.calls == [(tuple(c.copies), after) for c in chunks]
    # The last chunk holds the end of the first copy and the whole second one.
    assert len(chunks[-1].copies) == 2
    del driver.calls[:]
    overlap._copy(kv_ce_overlap.head_chunks([(0, 8, 0)], row))
    assert driver.calls == []


def test_chunks_keep_the_chunk_size_bounds():
    """However long the head, no chunk exceeds COPY_CHUNK_BYTES, and every chunk
    but the last is less than a row short of it (a chunk is whole rows)."""
    row = kv_ce_overlap.KV_ENTRY_BYTES
    small = kv_ce_overlap.head_chunks([(0, 1 << 30, 100 * row)], row)
    assert [chunk.nbytes for chunk in small] == [100 * row]
    # Over 1 GiB: still chunks of at most COPY_CHUNK_BYTES, as many as it takes.
    rows = 80 * (kv_ce_overlap.COPY_CHUNK_BYTES // row)
    big = kv_ce_overlap.head_chunks([(0, 1 << 40, rows * row)], row)
    assert rows * row > 1 << 30 and len(big) == 80
    assert all(chunk.nbytes <= kv_ce_overlap.COPY_CHUNK_BYTES for chunk in big)
    assert all(
        chunk.nbytes > kv_ce_overlap.COPY_CHUNK_BYTES - row for chunk in big[:-1]
    )
    assert sum(chunk.nbytes for chunk in big) == rows * row
    assert kv_ce_overlap.head_chunks([(0, 8, 0)], row) == []


def _layer(name: str = "a") -> SimpleNamespace:
    """A layer whose cache holds raw `fp8_ds_mla` rows."""
    cache = torch.zeros((1, 2, kv_ce_overlap.KV_ENTRY_BYTES), dtype=torch.uint8)
    return SimpleNamespace(layer_name=name, kv_cache=cache)


def _planner(world: int, names: tuple[str, ...] = ("a",)):
    """A `KVCEOverlap` that can plan steps of layers `names` without a device; the
    batches of stream memory operations it issues are recorded."""
    overlap = kv_ce_overlap.KVCEOverlap.__new__(kv_ce_overlap.KVCEOverlap)
    overlap.world_size = world
    overlap.shard_capacity = 1 << 20
    overlap.gathered_capacity = 4 * world * (1 << 20)
    overlap.wire_rows = 1 << 20
    overlap.order = LayerOrder()
    for _ in range(2):
        for name in names:
            overlap.order.advance(name, name == names[0])
    overlap.layers = {name: _layer(name) for name in names}
    overlap.sel, overlap.programs = None, {}
    overlap.plan = overlap.step = overlap.pending = overlap.forward = None
    overlap.position = None
    overlap.plan_count = overlap.plan_seq = 0
    overlap.interleave, overlap.restore_idx = 1, None
    overlap.gathered_ptr, overlap.row_bytes = 1 << 30, 1152
    writes: list = []

    def writer(slot: int) -> SimpleNamespace:
        def write(stream, start, values):
            writes.append((slot, stream, start, list(values)))

        return SimpleNamespace(write=write)

    writers: list = [writer(s) for s in range(kv_ce_overlap.PLAN_SLOTS)]
    overlap.table_writers, overlap.table_writer = writers, writers[0]
    return overlap, writes


def _scheduled(monkeypatch, manager, scheduled: bool = True) -> None:
    from vllm.v1.worker.gpu import pcp_manager

    monkeypatch.setattr(pcp_manager, "_ACTIVE_PCP_MANAGER", manager)
    monkeypatch.setattr(kv_ce_overlap, "scheduled_step", lambda: scheduled)
    stream = SimpleNamespace(cuda_stream=7)
    monkeypatch.setattr(kv_ce_overlap, "current_stream", lambda: stream)


def _request(start: int, end: int) -> SimpleNamespace:
    """The PCP manager of a step that prefills `[start, end)` of its only request,
    split across the ranks."""
    return SimpleNamespace(
        split_prefill_request=(start, end),
        hidden_restore_idx=torch.zeros(end - start, dtype=torch.int64),
    )


def _prefill_of(chunk):
    prefill = SimpleNamespace(chunks=[chunk])
    return lambda name: prefill


def test_a_step_is_planned_from_the_schedule_without_asking_a_rank(monkeypatch):
    """The plan of a split prefill follows from its context start and end. It is
    taken at once: its tables go out in one batch with the plan word the ranks
    compare, and consecutive plans alternate between the two words."""
    world, start, end = 4, 1_007_616, 1_024_000
    overlap, writes = _planner(world)
    _scheduled(monkeypatch, _request(start, end))
    shard = dcp_shard_rows(end, world, 0, 1)
    chunk = _chunk(shard, start)
    overlap._derive_plan("a", _prefill_of(chunk), 0, 1, lambda name: object())
    plan = overlap.plan
    assert plan is not None
    assert (plan.shard_rows, plan.context_start, plan.seq_end) == (shard, start, end)
    assert plan.head_rows == dcp_min_shard_rows(start, world, 1)
    assert plan.old_lens == tuple(dcp_shard_rows(start, world, r, 1) for r in range(4))
    step = overlap.step
    assert step.plan is plan and step.chunks == [chunk] and not step.checked
    low, high = kv_ce_overlap.plan_hash(plan.digest())
    tables = [plan.head_rows, *(p * shard for p in range(world)), *plan.new_lens]
    assert writes == [(1, 7, -2, [low, high, *tables])] and step.slot == 1
    del writes[:]
    overlap._derive_plan(
        "a", _prefill_of(_chunk(shard, start)), 0, 1, lambda name: object()
    )
    assert overlap.step.slot == 0 and writes[0][0] == 0


@pytest.mark.parametrize("why", ["dummy_batch", "no_manager", "other_batch", "unused"])
def test_a_step_the_schedule_does_not_plan_stays_serial(monkeypatch, why):
    """A capture or warmup forward, a batch that is not one split prefill and
    metadata the prefetch cannot use are the same on every rank: no plan, and
    nothing of the rank's own metadata is asked."""
    overlap, writes = _planner(4)
    manager = _request(4096, 8192)
    asked: list = []

    def prefill_of(name):
        asked.append(name)
        return None

    if why == "other_batch":
        manager.split_prefill_request = None
    _scheduled(
        monkeypatch, None if why == "no_manager" else manager, why != "dummy_batch"
    )
    overlap._derive_plan("a", prefill_of, 0, 1, asked.append)
    assert overlap.plan is None and overlap.step is None and writes == []
    assert asked == (["a"] if why == "unused" else [])


@pytest.mark.parametrize(
    "field", ["chunks", "entries", "decode_tokens", "shard_rows", "context_start"]
)
def test_a_rank_whose_metadata_contradicts_the_schedule_raises(monkeypatch, field):
    """Staying serial alone would leave the peers waiting at their barriers."""
    world, start, end = 4, 4096, 8192
    overlap, writes = _planner(world)
    _scheduled(monkeypatch, _request(start, end))
    shard = dcp_shard_rows(end, world, 0, 1)
    chunk = _chunk(shard, start)
    prefill = SimpleNamespace(chunks=[chunk])
    decode_tokens = 0
    if field == "chunks":
        prefill.chunks.append(chunk)
    elif field == "entries":
        chunk.block_table = torch.zeros(2, 1, dtype=torch.int32)
    elif field == "decode_tokens":
        decode_tokens = 1
    elif field == "shard_rows":
        chunk.chunk_tot_seqlen = shard + 1
    else:
        chunk.context_start = start - 1
    with pytest.raises(RuntimeError, match="implies"):
        overlap._derive_plan(
            "a", lambda name: prefill, decode_tokens, 1, lambda name: None
        )
    assert overlap.plan is None and writes == []


def test_a_layer_with_another_layout_keeps_the_whole_step_serial(monkeypatch):
    world, start, end = 4, 4096, 8192
    overlap, writes = _planner(world, ("a", "b"))
    _scheduled(monkeypatch, _request(start, end))
    shard = dcp_shard_rows(end, world, 0, 1)
    prefills = {
        "a": SimpleNamespace(chunks=[_chunk(shard, start)]),
        "b": SimpleNamespace(chunks=[_chunk(shard, start + 64)]),
    }
    overlap._derive_plan("a", prefills.get, 0, 1, prefills.get)
    assert overlap.plan is None and writes == []
    prefills["b"] = SimpleNamespace(chunks=[_chunk(shard, start)])
    overlap._derive_plan("a", prefills.get, 0, 1, prefills.get)
    assert overlap.plan is not None and len(overlap.step.chunks) == 2


def test_a_forward_is_planned_at_its_first_layer(monkeypatch):
    """A new forward drops the last one's plan and prefetch; the layer the
    learned order starts with plans it, a layer out of order drops it."""
    overlap, _ = _planner(4, ("a", "b"))
    events: list = []
    vars(overlap)["_derive_plan"] = lambda name, *a: events.append(("plan", name))
    vars(overlap)["_abandon"] = lambda: events.append("abandon")
    layers = overlap.layers

    class Forward:
        pass

    first = Forward()
    overlap.begin_layer(layers["a"], first, None, 0, 1, None)
    assert events == ["abandon", ("plan", "a")] and overlap.position == 0
    overlap.begin_layer(layers["b"], first, None, 0, 1, None)
    assert events == ["abandon", ("plan", "a")] and overlap.position == 1
    del events[:]
    second = Forward()
    overlap.begin_layer(layers["b"], second, None, 0, 1, None)
    assert events == ["abandon", "abandon"] and overlap.position is None


def test_a_dummy_batch_pays_no_barrier(monkeypatch):
    """A capture or warmup forward runs on every rank alike with a batch the
    scheduler did not make: an owed HEAD_DONE stays owed until the next scheduled
    step, and nothing is packed before it."""
    monkeypatch.setattr(kv_ce_overlap, "scheduled_step", lambda: False)
    overlap = kv_ce_overlap.KVCEOverlap.__new__(kv_ce_overlap.KVCEOverlap)
    barriers: list = []
    overlap.handle = SimpleNamespace(
        barrier=lambda channel, timeout_ms: barriers.append(channel)
    )
    vars(overlap)["_wait"] = lambda waiter, signaler: barriers.append("wait")
    overlap.head_done_owed = True
    overlap._serial()
    assert barriers == [] and overlap.head_done_owed
    monkeypatch.setattr(kv_ce_overlap, "scheduled_step", lambda: True)
    monkeypatch.setattr(kv_ce_overlap, "current_stream", lambda: None)
    overlap.side = None
    overlap._serial()
    assert barriers == ["wait", kv_ce_overlap.CHANNEL_HEAD_DONE]
    assert not overlap.head_done_owed


def test_the_plan_word_is_a_fixed_hash_of_everything_a_rank_acts_on():
    """Every rank hashes its own digest: the function must not depend on the
    process, and any field a rank acts on must change it."""
    plan = KVCEOverlapPlan(12, 10, 40, (10,) * 4, (12,) * 4, 48)
    word = kv_ce_overlap.plan_hash(plan.digest())
    assert word == (4059130925, 623991693)
    variants = [
        dataclasses.replace(plan, shard_rows=13),
        dataclasses.replace(plan, head_rows=9),
        dataclasses.replace(plan, context_start=41),
        dataclasses.replace(plan, seq_end=49),
        dataclasses.replace(plan, sel_positions=frozenset({3})),
        dataclasses.replace(
            plan, sel_positions=frozenset({3}), sel_builds=frozenset({3})
        ),
    ]
    words = {kv_ce_overlap.plan_hash(variant.digest()) for variant in variants}
    assert len(words) == len(variants) and word not in words
    assert all(0 <= half < 2**32 for half in word)


def test_the_ranks_plan_words_are_compared_once_behind_the_first_head(monkeypatch):
    """Every rank wrote its plan word before its HEAD_READY signal, so behind the
    barrier a kernel on the auxiliary stream can compare them; later heads of the
    step do not repeat it."""
    overlap, _ = _planner(4)
    calls: list = []
    overlap.handle = SimpleNamespace(
        barrier=lambda channel, timeout_ms: calls.append(("barrier", channel))
    )
    stream = SimpleNamespace(cuda_stream=9)
    monkeypatch.setattr(kv_ce_overlap, "current_stream", lambda: stream)
    launch = SimpleNamespace(set=lambda name, value: calls.append((name, value)))
    overlap.plan_check = lambda stream, grid: calls.append(("check", stream, grid))
    vars(overlap.plan_check).update(vars(launch))
    plan = KVCEOverlapPlan(12, 10, 40)
    overlap.step = kv_ce_overlap._Step(plan, [], (0, 0), 1)
    ready = ("barrier", kv_ce_overlap.CHANNEL_HEAD_READY)
    overlap._head_ready()
    assert calls == [ready, ("slot", 1), ("check", 9, 1)]
    overlap._head_ready()
    assert calls[3:] == [ready]
    # A head of a forward that left its step: only the barrier.
    overlap.step = None
    overlap._head_ready()
    assert calls[4:] == [ready]


def _rank_metadata(
    manager,
    rank: int,
    dcp: int,
    interleave: int,
    scheduled: np.ndarray,
    computed: np.ndarray,
    prefilling: np.ndarray,
    threshold: int,
    uniform: bool,
    workspace_rows: int,
):
    """What rank `rank` of a global batch hands to the KV prefetch: its prefill
    metadata (None without prefill rows) and its decode tokens. The rows are
    PCP's (`_get_rank_segments`), the decode split, the request shards and the
    chunks are the functions the model runner and the sparse MLA builder call."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        gathered_prefill_shards,
    )
    from vllm.v1.attention.backends.utils import (
        split_decodes_and_prefills,
        split_prefill_chunks,
    )

    starts = np.concatenate([[0], np.cumsum(scheduled)]).astype(np.int32)
    segments = manager._get_rank_segments(rank, scheduled, prefilling, starts)
    if not segments:
        return None, 0
    rows = np.array([segment.global_batch_req_idx for segment in segments])
    lens = np.array([segment.num_tokens for segment in segments])
    query_start_loc = torch.from_numpy(np.concatenate([[0], np.cumsum(lens)]))
    common = SimpleNamespace(
        max_query_len=int(lens.max()),
        num_reqs=len(rows),
        num_actual_tokens=int(lens.sum()),
        query_start_loc_cpu=query_start_loc,
        is_prefilling=torch.from_numpy(prefilling[rows]),
    )
    num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
        common,
        decode_threshold=threshold,
        require_uniform=uniform,
        treat_short_extends_as_decodes=False,
    )
    if num_prefills == 0:
        return None, num_decode_tokens
    ends = torch.from_numpy((computed + scheduled)[rows].astype(np.int32))
    shard_rows = get_dcp_local_seq_lens(ends, dcp, 0, interleave).numpy()
    row_bounds, rows_per_rank = gathered_prefill_shards(
        rows[num_decodes:], shard_rows[num_decodes:]
    )
    workspace = torch.from_numpy(rows_per_rank.astype(np.int32))
    chunks = []
    for first, last in split_prefill_chunks(workspace, workspace_rows // dcp):
        context_start = None
        if last - first == 1:
            context_start = int(computed[rows[num_decodes + int(row_bounds[first])]])
        chunk = _chunk(int(workspace[first:last].sum()), context_start)
        chunk.block_table = torch.zeros(last - first, 1, dtype=torch.int32)
        chunks.append(chunk)
    return SimpleNamespace(chunks=chunks), num_decode_tokens


def _agreed_plan(prefill, decode_tokens, overlap, interleave, end, new_rows):
    """The plan one rank proposed while the ranks still compared digests."""
    if prefill is None:
        return None
    chunk = prefill.chunks[0]
    plan, _ = build_kv_ce_overlap_plan(
        num_chunks=len(prefill.chunks),
        num_entries=len(chunk.block_table),
        num_decode_tokens=decode_tokens,
        shard_rows=int(chunk.chunk_tot_seqlen),
        context_start=chunk.context_start,
        dcp_size=overlap.world_size,
        interleave=interleave,
        shard_capacity=overlap.shard_capacity,
        gathered_capacity=overlap.gathered_capacity,
    )
    if plan is None:
        return None
    if end is None or new_rows is None or new_rows != end - plan.context_start:
        # The ranks then agreed on a plan that exchanged the new rows.
        return plan
    old, new = shard_lens(plan.context_start, end, overlap.world_size, interleave)
    if new_rows <= 0 or max(new) != plan.shard_rows:
        return plan
    return dataclasses.replace(plan, old_lens=old, new_lens=new, seq_end=end)


def _random_batch(rng: np.random.Generator, pcp: int):
    kind = rng.choice(["long", "long", "short", "decode", "mixed", "many"])
    if kind == "long":
        scheduled = [int(rng.integers(2 * pcp, 40_000))]
        computed = [int(rng.integers(0, 1_000_000))]
        prefilling = [True]
    elif kind == "short":
        scheduled = [int(rng.integers(1, 6 * pcp * pcp))]
        computed = [int(rng.integers(0, 70_000))]
        prefilling = [True]
    elif kind == "decode":
        scheduled, computed, prefilling = [1], [int(rng.integers(1, 9000))], [False]
    elif kind == "mixed":
        scheduled = [int(rng.integers(64, 9000)), 1]
        computed = [int(rng.integers(0, 90_000)), int(rng.integers(1, 9000))]
        prefilling = [True, False]
    else:
        count = int(rng.integers(2, 4))
        scheduled = [int(rng.integers(1, 5000)) for _ in range(count)]
        computed = [int(rng.integers(0, 90_000)) for _ in range(count)]
        prefilling = [True] * count
    if rng.random() < 0.15:
        computed[0] = 0
    as_i32 = lambda values: np.array(values, dtype=np.int32)  # noqa: E731
    return as_i32(scheduled), as_i32(computed), np.array(prefilling, dtype=np.bool_)


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("seed", range(4))
def test_every_rank_derives_the_plan_the_ranks_used_to_agree_on(monkeypatch, pcp, seed):
    """The property the derivation rests on, over random batches laid out by the
    real PCP partition and chunked by the builder's functions: no rank's own
    metadata contradicts the schedule, all ranks derive the same plan, and it is
    the plan every rank proposed when they still compared digests. Only a
    request too short to split, which every rank holds whole, was planned before
    and is serial now."""
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    rng = np.random.default_rng(1000 * pcp + seed)
    planned = serial_now = 0
    for _ in range(150):
        dcp = pcp * int(rng.choice([1, 1, 2]))
        interleave = int(rng.choice([1, 16, 64]))
        threshold, uniform = int(rng.choice([1, 1, 4])), bool(rng.random() < 0.5)
        scheduled, computed, prefilling = _random_batch(rng, pcp)
        end = int(computed[0] + scheduled[0]) if len(scheduled) == 1 else None
        derived, agreed = [], []
        for rank in range(pcp):
            manager = PCPManager(
                pcp_world_size=pcp,
                pcp_rank=rank,
                device=torch.device("cpu"),
                shard_decode_requests=False,
                dcp_world_size=dcp,
                cp_interleave=interleave,
            )
            manager._scheduled_batch = True
            manager._global_batch = SimpleNamespace(
                num_reqs=len(scheduled),
                num_scheduled_tokens=scheduled,
                num_computed_tokens_np=computed,
                is_prefilling_np=prefilling,
            )
            manager._hidden_restore_idx = torch.zeros(
                int(scheduled.sum()), dtype=torch.int64
            )
            prefill, decode_tokens = _rank_metadata(
                manager,
                rank,
                dcp,
                interleave,
                scheduled,
                computed,
                prefilling,
                threshold,
                uniform,
                40 << 20,
            )
            overlap, _ = _planner(dcp)
            _scheduled(monkeypatch, manager)
            if prefill is not None:
                # Layers without prefill rows never reach the prefetch.
                overlap._derive_plan(
                    "a", {"a": prefill}.get, decode_tokens, interleave, id
                )
            derived.append(overlap.plan)
            agreed.append(
                _agreed_plan(
                    prefill,
                    decode_tokens,
                    overlap,
                    interleave,
                    end,
                    int(scheduled.sum()),
                )
            )
        assert all(plan == derived[0] for plan in derived)
        was_planned = agreed[0] is not None and all(p == agreed[0] for p in agreed)
        split = (
            len(scheduled) == 1
            and bool(prefilling[0])
            and not manager.replicated_requests(scheduled, prefilling)[0]
        )
        if derived[0] is not None:
            assert was_planned and split and derived[0] == agreed[0]
            planned += 1
        elif was_planned:
            assert not split
            serial_now += 1
    assert planned > 20


STAGE_SCRIPT = r"""
import torch
from vllm.v1.attention.ops import kv_ce_rows
from vllm.v1.attention.ops.kv_ce_overlap import dcp_shard_rows

STAGE_SLOT_MAP

# The slots of every gathered row: the kernel against the tensor expression.
g = torch.Generator().manual_seed(0)
for start, new, dcp, interleave, pad in (
    (1_007_616, 16_384, 4, 1, 0),
    (4_096, 4_096, 8, 1, 16),
    (130_001, 4_099, 4, 64, 8),
    (7, 1, 4, 16, 0),
    (2**33 + 5, 2_050, 8, 256, 4),
):
    rows = new + pad
    restore = torch.randperm(rows, generator=g)[:new]
    counts = [
        dcp_shard_rows(start + new, dcp, r, interleave)
        - dcp_shard_rows(start, dcp, r, interleave)
        for r in range(dcp)
    ]
    per_rank = -(-max(counts) // 64) * 64
    want = stage_slot_map(start, start + new, restore, rows, dcp, interleave, per_rank)
    slots = torch.full((rows,), -1, dtype=torch.int64)
    stage_slots(restore, slots, start, interleave, dcp, per_rank)
    assert torch.equal(slots, want), (start, new, dcp, interleave)

# Every rank's staged entries, upconverted into its rows of the region. The scales
# are powers of two, so every upconverted value is exact in BF16.
ENTRY, ROW = kv_ce_rows.FP8_ENTRY, kv_ce_rows.ROW_DIM
entries = 96
raw = torch.empty((entries, ENTRY), dtype=torch.uint8)
values = (torch.randn(entries, 512, generator=g) * 50).clamp(-448, 448)
raw[:, :512] = values.to(torch.float8_e4m3fn).view(torch.uint8)
scales = 2.0 ** torch.randint(-3, 4, (entries, 4), generator=g).float()
raw[:, 512:528] = scales.view(torch.uint8).view(entries, 16)
rope = torch.randn(entries, 64, generator=g).to(torch.bfloat16)
raw[:, 528:] = rope.view(torch.uint8).view(entries, 128)
nope = raw[:, :512].view(torch.float8_e4m3fn).float()
full = torch.cat(
    [(nope * scales.repeat_interleave(128, dim=1)).to(torch.bfloat16), rope], dim=1
)
i32 = torch.int32
sources = torch.tensor([0, 32, 64, 80], dtype=i32)
targets = torch.tensor([5, 60, 100, 170], dtype=i32)
counts = torch.tensor([17, 0, 16, 9], dtype=i32)
region = torch.full((200, ROW), 3.0, dtype=torch.bfloat16)
cache = raw.view(-1, 16, ENTRY)
stage_rows(cache, sources, targets, counts, region, 17)
want = torch.full((200, ROW), 3.0, dtype=torch.bfloat16)
for s, t, c in zip(sources.tolist(), targets.tolist(), counts.tolist()):
    want[t : t + c] = full[s : s + c]
assert torch.equal(region.view(torch.int16), want.view(torch.int16))
print("ok")
"""


def test_the_staging_kernels_place_every_ranks_new_rows():
    """Run under the Triton interpreter: no GPU needed."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            inspect.getsource(launch)
            + STAGE_SCRIPT.replace("STAGE_SLOT_MAP", inspect.getsource(stage_slot_map)),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-3000:]


PLAN_CHECK_SCRIPT = r"""
import torch
from vllm.v1.attention.ops import kv_ce_rows

# Four ranks' plan words in one buffer, 64 bytes apart. The interpreter does not
# evaluate device assertions: this only shows that the kernel addresses every
# rank's word of either slot (a load outside the buffer would fail).
words = torch.zeros(4 * 8, dtype=torch.int64)
deltas = torch.arange(4, dtype=torch.int64) * 64
for slot in (0, 1):
    words[slot::8] = 0x1234_5678_9ABC_DEF0 + slot
    plan_check(words, deltas, slot)
print("ok")
"""


def test_the_plan_check_kernel_reads_every_ranks_plan_word():
    """Run under the Triton interpreter: no GPU needed. That differing words trap
    is a device test (`test_a_rank_whose_peer_planned_another_step_traps`)."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", inspect.getsource(launch) + PLAN_CHECK_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-3000:]


PLAN_TRAP_SCRIPT = r"""
import sys
import torch
from vllm.v1.attention.ops import kv_ce_rows

# The kernel as a step launches it: by its handle, on a stream of its own.
device = torch.device("cuda", 0)
words = torch.zeros(4 * 8, dtype=torch.int64, device=device)
deltas = torch.arange(4, dtype=torch.int64, device=device) * 64
words[1::8] = 77
kernel = kv_ce_rows.plan_check_kernel(4)
assert kernel is not None, "the plan check cannot be launched directly"
launch = kernel.bind(words_ptr=words.data_ptr(), deltas_ptr=deltas.data_ptr(), slot=1)
side = torch.cuda.Stream(device=device)
torch.accelerator.synchronize()
launch(side.cuda_stream, 1)
torch.accelerator.synchronize()
print("equal words pass", flush=True)
if sys.argv[1] == "differ":
    words[8 * 3 + 1] = 78
    torch.accelerator.synchronize()
    launch(side.cuda_stream, 1)
    try:
        torch.accelerator.synchronize()
    except RuntimeError as error:
        print("trapped:", str(error).splitlines()[0], flush=True)
        raise SystemExit(3)
    raise SystemExit("differing plan words passed")
"""


@pytest.mark.skipif(
    not (current_platform.is_cuda() and torch.cuda.is_available()),
    reason="needs a CUDA GPU",
)
@pytest.mark.parametrize("how", ["equal", "differ"])
def test_a_rank_whose_peer_planned_another_step_traps(how):
    """On the device: a peer's differing plan word is a device-side assertion,
    which the rank's next host call reports, instead of barriers that never pair
    up. In a process of its own: the assertion ends the CUDA context."""
    result = subprocess.run(
        [sys.executable, "-c", PLAN_TRAP_SCRIPT, how],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "equal words pass" in result.stdout, result.stderr[-3000:]
    if how == "equal":
        assert result.returncode == 0, result.stderr[-3000:]
    else:
        assert result.returncode == 3 and "trapped" in result.stdout, (
            result.stdout[-2000:] + result.stderr[-3000:]
        )
