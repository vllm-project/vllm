# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning and kernels of the overlapped DCP KV gather: a wrong split reads rows
the chunk is still writing, and ranks that disagree wait on each other's barriers
forever."""

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
    DCPPrefillStep,
    KVCEOverlapPlan,
    LayerOrder,
    build_kv_ce_overlap_plan,
    dcp_min_shard_rows,
    dcp_shard_rows,
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


def _parallel_config(tp_size: int, pcp_size: int, dcp_size: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=tp_size,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        dcp_kv_prefetch=True,
        distributed_executor_backend="mp",
    )


@pytest.mark.parametrize("tp_size, pcp_size, dcp_size", [(2, 1, 1), (2, 2, 1)])
def test_prefetch_requires_dcp(tp_size, pcp_size, dcp_size):
    with pytest.raises(ValueError, match="dcp_kv_prefetch requires"):
        _parallel_config(tp_size, pcp_size, dcp_size)


@pytest.mark.parametrize(
    "tp_size, pcp_size, dcp_size", [(2, 1, 2), (1, 2, 2), (2, 2, 4)]
)
def test_prefetch_allows_tp_dcp_and_pcp_dcp(tp_size, pcp_size, dcp_size):
    cfg = _parallel_config(tp_size, pcp_size, dcp_size)
    assert cfg.dcp_kv_prefetch is True


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
    restore_idx: torch.Tensor | None,
    rows: int,
    dcp_size: int,
    interleave: int,
    per_rank: int,
) -> torch.Tensor:
    """The scratch slot of each new row: the reference of `kv_ce_rows.stage_slots`.

    Token `context_start + g` is row `restore_idx[g]` (or `g` when `restore_idx`
    is None). DCP rank `p`'s new rows go in token order to slots
    `[p * per_rank, p * per_rank + new_p - old_p)`; unmapped rows get -1.
    """
    device = restore_idx.device if restore_idx is not None else torch.device("cpu")
    tokens = torch.arange(context_start, seq_end, dtype=torch.int64, device=device)
    owner = tokens // interleave % dcp_size
    shard_row = tokens // (interleave * dcp_size) * interleave + tokens % interleave
    base = context_start // interleave // dcp_size * interleave
    remainder = context_start - base * dcp_size
    ranks = torch.arange(dcp_size, dtype=torch.int64, device=device)
    old = base + (remainder - ranks * interleave).clamp(0, interleave)
    slots = torch.full((rows,), -1, dtype=torch.int64, device=device)
    target = (
        restore_idx
        if restore_idx is not None
        else torch.arange(seq_end - context_start, dtype=torch.int64, device=device)
    )
    slots[target] = owner * per_rank + shard_row - old[owner]
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


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pull_order_is_conflict_free_across_all_ranks(world_size: int):
    """At every ring step, no two ranks pull from the same peer simultaneously."""
    orders = []
    for rank in range(world_size):
        ov = kv_ce_overlap.KVCEOverlap.__new__(kv_ce_overlap.KVCEOverlap)
        ov.world_size = world_size
        ov.rank = rank
        order = ov._pull_order()
        assert len(order) == world_size - 1
        assert rank not in order
        assert set(order) == set(range(world_size)) - {rank}
        orders.append(order)
    for step in range(world_size - 1):
        peers_at_step = [orders[r][step] for r in range(world_size)]
        assert len(set(peers_at_step)) == world_size


@pytest.mark.parametrize(
    "head_rows, expect_pipelined",
    [(1024, False), (kv_ce_overlap.PIPELINE_MIN_ROWS, True)],
)
def test_fp8_head_uses_multistream_copy_engine_and_staggered_pull_order(
    monkeypatch, head_rows: int, expect_pipelined: bool
):
    """Verify HEAD_READY precedes copy_ready and own upconvert, copies run on
    copy_stream in staggered pull_order, and long heads pipeline per-peer
    upconverts behind per-peer copy events."""
    world, rank = 4, 1
    shard_rows = head_rows + 64
    ov = kv_ce_overlap.KVCEOverlap.__new__(kv_ce_overlap.KVCEOverlap)
    ov.world_size = world
    ov.rank = rank
    ov.wire_rows = shard_rows
    ov.wire_peers = [p for p in range(world) if p != rank]
    ov.pull_order = ov._pull_order()
    ov.wire_index = {p: i for i, p in enumerate(ov.wire_peers)}
    ov.stage_ptr = 0x10_0000_0000
    ov.remote_ptrs = [0x20_0000_0000 + p * 0x1_0000_0000 for p in range(world)]
    ov.own_wire = torch.zeros(
        (shard_rows, kv_ce_overlap.KV_ENTRY_BYTES), dtype=torch.uint8
    )
    ov.wire_cache = torch.zeros((1, 1, 1), dtype=torch.uint8)
    ov.wire_table = torch.zeros((len(ov.wire_peers), 1), dtype=torch.int32)
    ov.wire_part_table = [ov.wire_table[i : i + 1] for i in range(len(ov.wire_peers))]
    ov.wire_zero = torch.zeros(1, dtype=torch.int32)
    ov.wire_bounds = torch.tensor([0, 2**31 - 1], dtype=torch.int32)
    ov.slot_starts = torch.zeros(world, dtype=torch.int32)
    ov.peer_new = torch.full((world,), shard_rows, dtype=torch.int32)
    ov.gathered = torch.zeros((2 * world * shard_rows, 4), dtype=torch.bfloat16)
    ov.gathered_ptr = ov.gathered.data_ptr()
    ov.row_bytes = 4 * 2
    ov.side_handle = 101
    ov.copy_handle = 202
    ov.copy_ready = 900
    ov.peer_events = [901, 902, 903]

    events: list[tuple] = []

    class _FakeDriver:
        def record(self, ev, stream):
            events.append(("record", ev, stream))

        def wait_event(self, stream, ev):
            events.append(("wait", stream, ev))

        def memcpy_list(self, copies, stream):
            # Record source peer id inferred from remote_ptrs base.
            peers = [
                (src - 0x20_0000_0000) // 0x1_0000_0000 for _, src, _ in copies.copies
            ]
            events.append(("copy", stream, tuple(peers)))

    ov.driver = _FakeDriver()
    ov._head_ready = lambda: events.append(("head_ready",))  # type: ignore[method-assign]

    monkeypatch.setattr(
        kv_ce_overlap.ops,
        "cp_gather_cache",
        lambda *a, **kw: events.append(("pack_own",)),
    )
    monkeypatch.setattr(
        kv_ce_overlap.ops,
        "cp_gather_and_upconvert_fp8_kv_cache",
        lambda src, dst, *a, **kw: events.append(
            (
                "upconvert_own" if src is not ov.wire_cache else "upconvert_peer",
                dst.shape[0],
            )
        ),
    )

    plan = KVCEOverlapPlan(
        shard_rows=shard_rows,
        head_rows=head_rows,
        context_start=head_rows * world,
        old_lens=(head_rows,) * world,
        new_lens=(shard_rows,) * world,
        seq_end=shard_rows * world,
    )
    chunk = _chunk(shard_rows, plan.context_start)
    ov.step = kv_ce_overlap._Step(plan, [chunk], (ov.gathered_ptr, ov.gathered_ptr))
    ov._fp8_head(torch.zeros((1, 1, 1), dtype=torch.uint8), chunk, 0, plan)

    assert ov.step.head is not None
    assert ov.step.head.pipelined is expect_pipelined
    # Order invariant: pack -> head_ready -> copy_ready fork -> copy_stream pulls
    # -> own upconvert -> wait peer event(s) -> peer upconvert(s).
    idx_pack = events.index(("pack_own",))
    idx_ready = events.index(("head_ready",))
    idx_fork = events.index(("record", 900, 101))
    idx_own_up = events.index(("upconvert_own", head_rows))
    assert idx_pack < idx_ready < idx_fork < idx_own_up

    copy_events = [e for e in events if e[0] == "copy"]
    assert all(e[1] == 202 for e in copy_events)
    copied_peers = [p for e in copy_events for p in e[2]]
    # Deduplicate consecutive chunks of the same peer while preserving order.
    dedup_peers = [
        p for i, p in enumerate(copied_peers) if i == 0 or p != copied_peers[i - 1]
    ]
    assert dedup_peers == [2, 3, 0]  # staggered pull_order for rank=1, world=4

    if expect_pipelined:
        peer_upconverts = [e for e in events if e[0] == "upconvert_peer"]
        assert len(peer_upconverts) == 3
        for ev_id in ov.peer_events:
            assert ("record", ev_id, 202) in events
            assert ("wait", 101, ev_id) in events
    else:
        assert ("record", 901, 202) in events
        assert ("wait", 101, 901) in events
        # 2 contiguous runs for rank=1 in world=4: [0] and [2, 3]
        assert len([e for e in events if e[0] == "upconvert_peer"]) == 2


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


def _mock_stream(monkeypatch) -> None:
    stream = SimpleNamespace(cuda_stream=7)
    monkeypatch.setattr(kv_ce_overlap, "current_stream", lambda: stream)


def _step_info(start: int, end: int, with_restore: bool = True) -> DCPPrefillStep:
    restore = torch.zeros(end - start, dtype=torch.int64) if with_restore else None
    return DCPPrefillStep(start, end, restore)


def _prefill_of(chunk):
    prefill = SimpleNamespace(chunks=[chunk])
    return lambda name: prefill


@pytest.mark.parametrize("with_restore", [True, False])
def test_a_step_is_planned_from_the_schedule_without_asking_a_rank(
    monkeypatch, with_restore
):
    """The plan follows from `DCPPrefillStep` (with PCP `restore_idx` or without
    it for TP+DCP identity order). Its tables go out in one batch with the plan
    word, and consecutive plans alternate between the two words."""
    world, start, end = 4, 1_007_616, 1_024_000
    overlap, writes = _planner(world)
    _mock_stream(monkeypatch)
    shard = dcp_shard_rows(end, world, 0, 1)
    chunk = _chunk(shard, start)
    overlap._derive_plan(
        "a", _prefill_of(chunk), 0, 1, _step_info(start, end, with_restore)
    )
    plan = overlap.plan
    assert plan is not None
    assert (plan.shard_rows, plan.context_start, plan.seq_end) == (shard, start, end)
    assert plan.head_rows == dcp_min_shard_rows(start, world, 1)
    assert plan.old_lens == tuple(dcp_shard_rows(start, world, r, 1) for r in range(4))
    assert (overlap.restore_idx is not None) is with_restore
    step = overlap.step
    assert step.plan is plan and step.chunks == [chunk] and not step.checked
    low, high = kv_ce_overlap.plan_hash(plan.digest())
    tables = [plan.head_rows, *(p * shard for p in range(world)), *plan.new_lens]
    assert writes == [(1, 7, -2, [low, high, *tables])] and step.slot == 1
    del writes[:]
    overlap._derive_plan(
        "a",
        _prefill_of(_chunk(shard, start)),
        0,
        1,
        _step_info(start, end, with_restore),
    )
    assert overlap.step.slot == 0 and writes[0][0] == 0


@pytest.mark.parametrize("why", ["no_step_info", "unused"])
def test_a_step_the_schedule_does_not_plan_stays_serial(monkeypatch, why):
    """A capture, warmup, or non-eligible batch supplies `step_info=None` (or
    unused metadata): no plan, and nothing of the rank's own metadata is asked
    when `step_info` is None."""
    overlap, writes = _planner(4)
    _mock_stream(monkeypatch)
    asked: list = []

    def prefill_of(name):
        asked.append(name)
        return None

    step_info = None if why == "no_step_info" else _step_info(4096, 8192)
    overlap._derive_plan("a", prefill_of, 0, 1, step_info)
    assert overlap.plan is None and overlap.step is None and writes == []
    assert asked == (["a"] if why == "unused" else [])


@pytest.mark.parametrize(
    "field", ["chunks", "entries", "decode_tokens", "shard_rows", "context_start"]
)
def test_a_rank_whose_metadata_contradicts_the_schedule_raises(monkeypatch, field):
    """Staying serial alone would leave the peers waiting at their barriers."""
    world, start, end = 4, 4096, 8192
    overlap, writes = _planner(world)
    _mock_stream(monkeypatch)
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
            "a", lambda name: prefill, decode_tokens, 1, _step_info(start, end)
        )
    assert overlap.plan is None and writes == []


def test_a_layer_with_another_layout_keeps_the_whole_step_serial(monkeypatch):
    world, start, end = 4, 4096, 8192
    overlap, writes = _planner(world, ("a", "b"))
    _mock_stream(monkeypatch)
    shard = dcp_shard_rows(end, world, 0, 1)
    prefills = {
        "a": SimpleNamespace(chunks=[_chunk(shard, start)]),
        "b": SimpleNamespace(chunks=[_chunk(shard, start + 64)]),
    }
    overlap._derive_plan("a", prefills.get, 0, 1, _step_info(start, end))
    assert overlap.plan is None and writes == []
    prefills["b"] = SimpleNamespace(chunks=[_chunk(shard, start)])
    overlap._derive_plan("a", prefills.get, 0, 1, _step_info(start, end))
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
    overlap = kv_ce_overlap.KVCEOverlap.__new__(kv_ce_overlap.KVCEOverlap)
    barriers: list = []
    overlap.handle = SimpleNamespace(
        barrier=lambda channel, timeout_ms: barriers.append(channel)
    )
    vars(overlap)["_wait"] = lambda waiter, signaler: barriers.append("wait")
    overlap.head_done_owed = True
    overlap._serial(scheduled=False)
    assert barriers == [] and overlap.head_done_owed
    monkeypatch.setattr(kv_ce_overlap, "current_stream", lambda: None)
    overlap.side = None
    overlap._serial(scheduled=True)
    assert barriers == ["wait", kv_ce_overlap.CHANNEL_HEAD_DONE]
    assert not overlap.head_done_owed


def test_the_plan_word_is_a_fixed_hash_of_everything_a_rank_acts_on():
    """Every rank hashes its own digest: the function must not depend on the
    process, and any field a rank acts on must change it."""
    plan = KVCEOverlapPlan(12, 10, 40, (10,) * 4, (12,) * 4, 48)
    word = kv_ce_overlap.plan_hash(plan.digest())
    variants = [
        dataclasses.replace(plan, shard_rows=13),
        dataclasses.replace(plan, head_rows=9),
        dataclasses.replace(plan, context_start=41),
        dataclasses.replace(plan, seq_end=49),
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
    launch_fn = SimpleNamespace(set=lambda name, value: calls.append((name, value)))
    overlap.plan_check = lambda stream, grid: calls.append(("check", stream, grid))
    vars(overlap.plan_check).update(vars(launch_fn))
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


STAGE_SCRIPT = r"""
import torch
from vllm.v1.attention.ops import kv_ce_rows
from vllm.v1.attention.ops.kv_ce_overlap import dcp_shard_rows

STAGE_SLOT_MAP

# The slots of every gathered row (both PCP restore_idx and TP+DCP identity order).
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
    # Identity order (restore_idx=None, e.g. TP + DCP without PCP reordering).
    want_id = stage_slot_map(start, start + new, None, rows, dcp, interleave, per_rank)
    slots_id = torch.full((rows,), -1, dtype=torch.int64)
    stage_slots(None, slots_id, start, interleave, dcp, per_rank, new_rows=new)
    assert torch.equal(slots_id, want_id), ("identity", start, new, dcp, interleave)

# Every rank's staged entries, upconverted into its rows of the region.
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

words = torch.zeros(4 * 8, dtype=torch.int64)
deltas = torch.arange(4, dtype=torch.int64) * 64
for slot in (0, 1):
    words[slot::8] = 0x1234_5678_9ABC_DEF0 + slot
    plan_check(words, deltas, slot)
print("ok")
"""


def test_the_plan_check_kernel_reads_every_ranks_plan_word():
    """Run under the Triton interpreter: no GPU needed."""
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
    """On the device: a peer's differing plan word triggers a device assertion."""
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
