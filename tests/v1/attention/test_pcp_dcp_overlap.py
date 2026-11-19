# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the decisions behind the PCP + DCP gather overlap: a step that is
wrongly taken as eligible reads another layout's rows, and ranks that decide
differently wait on each other's barriers."""

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention import kv_ce_triton_launch as launch
from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
    DeepseekV32IndexerPrefillMetadata,
)
from vllm.v1.attention.ops import cp_common, pcp_dcp_k_replica, pcp_dcp_overlap
from vllm.v1.attention.ops.cuda_direct import CopyList
from vllm.v1.attention.ops.pcp_dcp_k_replica import (
    KReplica,
    LayerReplica,
    k_replica_chunk,
    replica_appends,
    replica_bytes,
)
from vllm.v1.attention.ops.pcp_dcp_overlap import (
    K_CAPACITY_SLACK_ROWS,
    IndexerOverlap,
    NewRows,
    PrefillOverlap,
    RowOutputs,
    SymmExchange,
    new_rows_capacity,
    new_rows_eligible,
    pop_indexer_overlap,
    prefill_overlap_for_layer,
    staggered_peers,
)

# GLM-5.2 at 1M context on PCP4 x DCP4, chunk 16384: each rank holds two eighths of
# the chunk (4096 rows) as two local requests.
LOCAL_ROWS = 4096
NEW_ROWS = 16_384


def _chunk(**overrides) -> DeepseekV32IndexerPrefillChunkMetadata:
    fields = dict(
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        cu_seqlen_ks=torch.zeros(LOCAL_ROWS, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(LOCAL_ROWS, dtype=torch.int32),
        cu_seq_lens=torch.zeros(3, dtype=torch.int32),
        token_to_seq=torch.zeros(1, dtype=torch.int32),
        total_seq_lens=1_024_000,
        max_seqlen_k=1_024_000,
        token_start=0,
        token_end=LOCAL_ROWS,
        num_reqs=2,
        local_cu_seq_lens=torch.zeros(3, dtype=torch.int32),
        local_total_seq_lens=256_000,
        max_local_total_seq_lens=256_000,
        pcp_deinterleave_idx=torch.zeros(1, dtype=torch.int64),
    )
    return DeepseekV32IndexerPrefillChunkMetadata(**{**fields, **overrides})


def _prefill(**overrides) -> DeepseekV32IndexerPrefillMetadata:
    fields = dict(chunks=[_chunk()], max_prefill_seq_len=1_024_000)
    return DeepseekV32IndexerPrefillMetadata(**{**fields, **overrides})


def _eligibility(prefill, **overrides):
    args = dict(
        num_decode_tokens=0,
        local_rows=LOCAL_ROWS,
        pcp_world_size=4,
        dcp_world_size=4,
        capacity_rows=1_052_672,
        global_new_rows=NEW_ROWS,
        global_end=1_024_000,
    )
    return k_replica_chunk(prefill, **{**args, **overrides})


def test_off_by_default():
    assert not ParallelConfig().pcp_dcp_gather_overlap
    assert pop_indexer_overlap() is None
    assert prefill_overlap_for_layer(False, 1, object(), True, 4, 4, 1 << 20) is None


def test_needs_pcp_and_dcp():
    assert prefill_overlap_for_layer(True, 1, object(), False, 1, 4, 1 << 20) is None
    assert prefill_overlap_for_layer(True, 1, object(), True, 4, 1, 1 << 20) is None


@pytest.mark.parametrize(("pcp_size", "dcp_size"), [(1, 1), (1, 2), (2, 1)])
def test_option_requires_pcp_and_dcp(pcp_size, dcp_size):
    with pytest.raises(ValueError, match="pcp_dcp_gather_overlap requires"):
        ParallelConfig(
            tensor_parallel_size=2,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            pcp_dcp_gather_overlap=True,
            distributed_executor_backend="mp",
        )


class _Driver:
    """Records the driver calls of an exchange instead of issuing them."""

    def __init__(self, calls: list) -> None:
        self.calls = calls
        self.events = 0

    def event(self) -> int:
        self.events += 1
        return 100 + self.events

    def record(self, event: int, stream: int) -> None:
        self.calls.append(("record", event, stream))

    def wait_event(self, stream: int, event: int) -> None:
        self.calls.append(("wait", stream, event))

    def memcpy_list(self, copies: CopyList, stream: int) -> None:
        self.calls.append(("copy", tuple(copies.copies), stream))


def _exchange(calls: list, nbytes: int, rank: int, world: int) -> SymmExchange:
    """An exchange over host memory: peer `p`'s buffer is at `1000 * (p + 1)`."""
    exchange = SymmExchange.__new__(SymmExchange)
    exchange.buffer = torch.zeros(nbytes, dtype=torch.uint8)
    exchange.nbytes, exchange.rank, exchange.world_size = nbytes, rank, world
    exchange.ptr = 0
    exchange.remote_ptrs = [1000 * (peer + 1) for peer in range(world)]
    exchange.peer_bases = torch.tensor(exchange.remote_ptrs, dtype=torch.int64)
    exchange.handle = SimpleNamespace(
        barrier=lambda channel, timeout_ms: calls.append(("barrier", channel))
    )
    return exchange


MAIN, AUX = 7, 9


def _step(rows: int = 2, world: int = 4, decode_tokens: int = 0):
    """The attention metadata and slot mapping of a prefill step with `rows` rows
    on every rank."""
    return SimpleNamespace(num_decode_tokens=decode_tokens), torch.zeros(world * rows)


class _PullKernel:
    """Stands in for the loaded pull kernel: records the parameters each layer
    shape binds and the launches."""

    def __init__(self, calls: list) -> None:
        self.calls = calls
        self.bound: list[dict] = []

    def bind(self, **values: int):
        self.bound.append(values)

        def launch(stream: int, grid_x: int, grid_y: int = 1) -> None:
            self.calls.append(("pull_kernel", stream, grid_x, grid_y))

        return launch


def _overlap_on_host(monkeypatch, rank: int = 1, world: int = 4, rows: int = 2):
    """A `PrefillOverlap` whose exchange, streams, driver and pull kernel are host
    stand-ins: `kv_c` is 6 wide, `k_pe` 2 and the indexer K 4, all int16."""
    calls: list = []
    current = [SimpleNamespace(cuda_stream=MAIN)]
    monkeypatch.setattr(pcp_dcp_overlap, "current_stream", lambda: current[0])
    monkeypatch.setattr(
        torch.cuda, "set_stream", lambda stream: current.__setitem__(0, stream)
    )
    overlap = PrefillOverlap(world, world, 1_024_000, 2 * world * rows)
    assert overlap.new_rows_capacity is not None
    capacity = world * overlap.new_rows_capacity
    overlap._new_exchange = _exchange(calls, capacity * (8 + 4) * 2, rank, world)
    overlap._stream = SimpleNamespace(cuda_stream=AUX)
    overlap._stream_handle = AUX
    overlap._driver = driver = _Driver(calls)
    overlap._ready, overlap._rows_done = driver.event(), driver.event()
    overlap._row_layout = (8, 4, torch.int16)
    overlap._pull = _PullKernel(calls)  # type: ignore[assignment]
    monkeypatch.setattr(overlap, "_groups_usable", lambda: True)
    # An eager forward of a scheduled prefill, unless a test says otherwise.
    state = SimpleNamespace(mode=CUDAGraphMode.NONE, segment=False, scheduled=True)
    setattr(overlap, "test_state", state)  # noqa: B010
    monkeypatch.setattr(
        pcp_dcp_overlap,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=state.mode),
    )
    monkeypatch.setattr(
        pcp_dcp_overlap, "capturing_breakable_segment", lambda: state.segment
    )
    monkeypatch.setattr(pcp_dcp_overlap, "capturing_run", lambda: state.segment)
    monkeypatch.setattr(
        pcp_dcp_overlap, "scheduled_step", lambda: state.scheduled and not state.segment
    )
    i16 = torch.int16
    kv_c = torch.zeros((rows, 6), dtype=i16)
    k_pe = torch.zeros((rows, 2), dtype=i16)
    index_k = torch.zeros((rows, 4), dtype=i16)
    return overlap, calls, (kv_c, k_pe, index_k)


def test_new_rows_are_written_into_this_ranks_segments(monkeypatch):
    """The fused norm/rope writes where the peers read: this rank's rows of the
    rank-major gather, `kv_c` and `k_pe` side by side and the K rows contiguous."""
    overlap, _, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch, rank=1)
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    assert outputs is not None
    assert outputs.kv_c.shape == kv_c.shape and outputs.k_pe.shape == k_pe.shape
    assert outputs.index_k.shape == index_k.shape and outputs.index_k.is_contiguous()
    outputs.kv_c.fill_(1)
    outputs.k_pe.fill_(2)
    outputs.index_k.fill_(3)
    rows = kv_c.shape[0]
    mine = slice(rows, 2 * rows)
    assert outputs.rows == 4 * rows
    assert (outputs.gathered_kv[mine] == 1).all()
    assert (outputs.gathered_pe[mine] == 2).all()
    assert outputs.gathered_pe.shape == (4 * rows, 1, 2)
    assert (outputs.gathered_k[mine] == 3).all()
    for gathered in (outputs.gathered_kv, outputs.gathered_pe, outputs.gathered_k):
        assert int(gathered[:rows].abs().sum()) == 0
        assert int(gathered[2 * rows :].abs().sum()) == 0
    # The same views serve the next layer of this shape.
    assert overlap.new_row_outputs(kv_c, k_pe, index_k, *_step()) is outputs
    # A layer without an indexer exchanges `kv_c` and `k_pe` only.
    plain = overlap.new_row_outputs(kv_c, k_pe, None, *_step())
    assert plain.index_k is None and plain.gathered_k is None
    assert plain.copies.count == 3 and outputs.copies.count == 6


def test_layers_that_split_a_row_differently_do_not_share_views(monkeypatch):
    """The views are kept per layer shape. A layer with the same rows and K width
    but another `kv_c`/`k_pe` split must write through views of its own."""
    overlap, _, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch)
    first = overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    rows, i16 = kv_c.shape[0], torch.int16
    narrow, wide = torch.zeros((rows, 5), dtype=i16), torch.zeros((rows, 3), dtype=i16)
    other = overlap.new_row_outputs(narrow, wide, index_k, *_step())
    assert other is not None and other is not first
    assert other.kv_c.shape == narrow.shape and other.k_pe.shape == wide.shape
    assert other.copies.copies == first.copies.copies


def test_every_peer_segment_is_pulled_once_in_staggered_order(monkeypatch):
    overlap, _, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch, rank=1)
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    rows = kv_c.shape[0]
    a_segment, b_segment = rows * 8 * 2, rows * 4 * 2
    b_start = 4 * overlap.new_rows_capacity * 8 * 2
    expected = []
    for peer in staggered_peers(1, 4):
        remote = 1000 * (peer + 1)
        expected.append((peer * a_segment, remote + peer * a_segment, a_segment))
        at = b_start + peer * b_segment
        expected.append((at, remote + at, b_segment))
    assert outputs.copies.copies == expected


def test_an_exchange_is_a_few_calls_issued_by_the_models_thread(monkeypatch):
    """Nothing on the auxiliary stream waits for work that is not enqueued yet:
    the event it waits for is recorded first, by the same thread, and the
    consumer waits for an event recorded behind the exchange."""
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch)
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    new_rows = overlap.start_new_rows(outputs)
    ready, done = overlap._ready, overlap._rows_done
    assert calls == [
        ("record", ready, MAIN),
        ("wait", AUX, ready),
        ("barrier", pcp_dcp_overlap.CHANNEL_READY),
        ("copy", tuple(outputs.copies.copies), AUX),
        ("barrier", pcp_dcp_overlap.CHANNEL_DONE),
        ("record", done, AUX),
    ]
    assert pcp_dcp_overlap.current_stream().cuda_stream == MAIN
    del calls[:]
    slots = torch.arange(64)
    kv, pe, kv_slots = new_rows.kv(slots)
    assert calls == [("wait", MAIN, done)]
    assert kv is outputs.gathered_kv and pe is outputs.gathered_pe
    assert torch.equal(kv_slots, slots[: outputs.rows])
    # One wait serves every consumer of the layer.
    k, k_slots = new_rows.index_k(slots)
    assert k is outputs.gathered_k and calls == [("wait", MAIN, done)]


def test_a_segment_is_not_rewritten_before_its_exchange_was_joined(monkeypatch):
    """The next layer's rows go into the same segment. The stream that writes
    them first waits for the last exchange, which ends once every peer read it."""
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch)
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    overlap.start_new_rows(outputs)
    del calls[:]
    overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    assert calls == [("wait", MAIN, overlap._rows_done)]
    del calls[:]
    # Joined already: nothing to wait for.
    overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    assert calls == []


def test_rows_that_do_not_fit_the_buffer_gather_serially(monkeypatch):
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch, rows=2)
    big = overlap.new_rows_capacity + 2
    i16 = torch.int16
    wide = torch.zeros((big, 6), dtype=i16), torch.zeros((big, 2), dtype=i16)
    assert overlap.new_row_outputs(*wide, None, *_step(big)) is None
    other = index_k.to(torch.int32)
    assert overlap.new_row_outputs(kv_c, k_pe, other, *_step()) is None
    assert calls == []


def test_the_buffer_is_allocated_once_while_the_model_is_built(monkeypatch):
    """Captured graphs record addresses inside the buffer: it is sized for a full
    batch when the first layer registers and never allocated again."""
    asked: list = []
    exchange = _exchange([], 0, 0, 4)

    def allocate(nbytes, device):
        asked.append(nbytes)
        return exchange

    monkeypatch.setattr(pcp_dcp_overlap, "try_symm_exchange", allocate)
    overlap = PrefillOverlap(4, 4, 1_024_000, 16_384)
    monkeypatch.setattr(overlap, "_groups_usable", lambda: True)
    monkeypatch.setattr(overlap, "_auxiliary", lambda device: asked.append("aux"))
    bf16 = torch.bfloat16
    overlap.allocate_new_rows(512, 64, 128, bf16, torch.device("cpu"))
    # Every rank's 4096 rows: `kv_c` | `k_pe` in region A, the indexer K in B.
    assert asked == [4 * 4096 * (512 + 64 + 128) * 2, "aux"]
    assert overlap._new_exchange is exchange
    assert overlap._row_layout == (576, 128, bf16)
    overlap.allocate_new_rows(512, 64, 128, bf16, torch.device("cpu"))
    assert len(asked) == 2
    # Where a rank cannot have it, every rank keeps the serial gather for good.
    monkeypatch.setattr(pcp_dcp_overlap, "try_symm_exchange", lambda *a: None)
    serial = PrefillOverlap(4, 4, 1_024_000, 16_384)
    monkeypatch.setattr(serial, "_groups_usable", lambda: True)
    serial.allocate_new_rows(512, 64, 128, bf16, torch.device("cpu"))
    assert serial._new_exchange is False
    rows = torch.zeros((4096, 512), dtype=bf16), torch.zeros((4096, 64), dtype=bf16)
    assert serial._outputs_for(*rows, None) is None


def _recorded_calls(overlap: PrefillOverlap, outputs: RowOutputs) -> list:
    """The calls a segment records for the exchange of `outputs`: its branch, with
    one launch of the pull kernel over every peer's segments, and the join before
    the eager region."""
    ready, done = overlap._ready, overlap._rows_done

    def chunks(*tensors: torch.Tensor | None) -> int:
        words = sum(t.numel() * t.element_size() // 8 for t in tensors if t is not None)
        return -(-words // pcp_dcp_overlap.PULL_BLOCK_WORDS)

    grid = chunks(outputs.kv_c, outputs.k_pe) + chunks(outputs.index_k)
    exchange = overlap._new_exchange
    assert isinstance(exchange, SymmExchange)
    peers = exchange.world_size - 1
    return [
        ("record", ready, MAIN),
        ("wait", AUX, ready),
        ("barrier", pcp_dcp_overlap.CHANNEL_READY),
        ("pull_kernel", AUX, grid, peers),
        ("barrier", pcp_dcp_overlap.CHANNEL_DONE),
        ("record", done, AUX),
        ("wait", MAIN, done),
    ]


def test_a_captured_segment_writes_into_the_buffer_for_every_later_step(monkeypatch):
    """Where the fused norm/rope is recorded into a breakable graph no step is
    known: the dummy batch's metadata is not asked, the rows go into this rank's
    segments when the graph's size fits, and their exchange is recorded behind
    them as a branch that rejoins before the eager region."""
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch)
    overlap.test_state.segment = True
    overlap.test_state.mode = CUDAGraphMode.PIECEWISE
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, None, None)
    assert outputs is not None and outputs.kv_c.shape == kv_c.shape
    assert calls == [] and not outputs.captured
    assert overlap.issue_new_rows(outputs) is None and outputs.captured
    overlap.join_issued_rows(None)
    assert calls == _recorded_calls(overlap, outputs)
    assert not overlap._rows_unjoined
    del calls[:]
    # A graph whose rows do not fit records plain outputs.
    big = overlap.new_rows_capacity + 2
    i16 = torch.int16
    wide = torch.zeros((big, 6), dtype=i16), torch.zeros((big, 2), dtype=i16)
    assert overlap.new_row_outputs(*wide, None, None, None) is None
    # Where the pull kernel cannot be launched directly no segment records the
    # exchange: the rows go where the serial gather reads them.
    serial, calls, tensors = _overlap_on_host(monkeypatch)
    serial._pull = None
    serial.test_state.segment = True
    serial.test_state.mode = CUDAGraphMode.PIECEWISE
    assert serial.new_row_outputs(*tensors, None, None) is None and calls == []
    # So do segments the kernel cannot copy as whole 16-byte words: one row's
    # indexer K is 8 bytes here.
    odd, calls, tensors = _overlap_on_host(monkeypatch, rows=1)
    odd.test_state.segment = True
    odd.test_state.mode = CUDAGraphMode.PIECEWISE
    assert odd.new_row_outputs(*tensors, None, None) is None and calls == []
    # The decision passes the function that decides every step.
    monkeypatch.setattr(pcp_dcp_overlap, "new_rows_eligible", lambda *a: False)
    assert overlap.new_row_outputs(kv_c, k_pe, index_k, None, None) is None


@pytest.mark.parametrize(
    ("mode", "scheduled"),
    [
        # A FULL graph is recorded and replayed without eager regions.
        (CUDAGraphMode.FULL, True),
        (CUDAGraphMode.PIECEWISE, True),
        # A warmup forward: eager, on a dummy batch.
        (CUDAGraphMode.NONE, False),
    ],
)
def test_other_forwards_keep_plain_outputs(monkeypatch, mode, scheduled):
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch)
    overlap.test_state.mode, overlap.test_state.scheduled = mode, scheduled
    assert overlap.new_row_outputs(kv_c, k_pe, index_k, *_step()) is None
    assert overlap.issue_new_rows(None) is None and calls == []


def _record_segment(overlap, kv_c, k_pe, index_k) -> RowOutputs:
    """What a breakable capture of one layer's segment does with its rows: the
    fused norm/rope's outputs, the exchange's branch and the join behind it."""
    overlap.test_state.segment = True
    overlap.test_state.mode = CUDAGraphMode.PIECEWISE
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, None, None)
    assert outputs is not None
    assert overlap.issue_new_rows(outputs) is None
    overlap.join_issued_rows(None)
    overlap.test_state.segment = False
    return outputs


def _baked(monkeypatch, rank: int = 1, rows: int = 2):
    """The outputs a captured segment of `rows` rows per rank writes and
    exchanges, and the overlap back in the eager region of a replay."""
    overlap, calls, tensors = _overlap_on_host(monkeypatch, rank=rank, rows=rows)
    outputs = _record_segment(overlap, *tensors)
    del calls[:]
    return overlap, calls, outputs


def test_the_eager_region_takes_the_rows_a_replay_exchanged(monkeypatch):
    """Per step, from the step's own metadata: an eligible step takes the rows
    the segment exchanged and joined, and issues nothing, neither here nor where
    it consumes them."""
    overlap, calls, outputs = _baked(monkeypatch)
    rows = (outputs.kv_c, outputs.k_pe, outputs.index_k)
    new_rows = overlap.baked_new_rows(*rows, *_step())
    assert isinstance(new_rows, NewRows) and new_rows.outputs is outputs
    assert new_rows.joined and not overlap._rows_unjoined
    slots = torch.arange(64)
    kv, pe, kv_slots = new_rows.kv(slots)
    k = new_rows.index_k(slots)
    assert kv is outputs.gathered_kv and pe is outputs.gathered_pe
    assert k is not None and k[0] is outputs.gathered_k
    assert torch.equal(kv_slots, slots[: outputs.rows])
    assert calls == []


@pytest.mark.parametrize(
    "why",
    [
        "decode_tokens",
        "replicated",
        "dummy_batch",
        "plain_rows",
        "no_rows",
        "not_recorded",
    ],
)
def test_a_step_that_cannot_exchange_gathers_the_baked_rows_serially(monkeypatch, why):
    """The segment wrote and exchanged the rows whatever the step. A step with
    decode tokens, an unexpanded slot mapping or a dummy batch issues nothing and
    returns None: the caller gathers the rows serially, from where they are."""
    overlap, calls, outputs = _baked(monkeypatch)
    rows = [outputs.kv_c, outputs.k_pe, outputs.index_k]
    metadata, slots = _step()
    if why == "decode_tokens":
        metadata.num_decode_tokens = 1
    elif why == "replicated":
        slots = slots[:2]
    elif why == "dummy_batch":
        overlap.test_state.scheduled = False
    elif why == "plain_rows":
        # A segment recorded without the buffer: the same shapes elsewhere.
        rows = [torch.zeros_like(row) for row in rows]
    elif why == "not_recorded":
        # Rows in the buffer whose exchange no segment recorded.
        outputs.captured = False
    else:
        rows[0] = None
    assert overlap.baked_new_rows(*rows, metadata, slots) is None
    assert calls == [] and not overlap._rows_unjoined


def test_rank_0_segments_of_two_graph_sizes_start_at_the_same_address(monkeypatch):
    """Rank 0's segment starts at the buffer's first byte for every graph size,
    so the views are found by shape and only then checked by address."""
    overlap, calls, small = _baked(monkeypatch, rank=0, rows=2)
    i16 = torch.int16
    shapes = [torch.zeros((4, width), dtype=i16) for width in (6, 2, 4)]
    large = _record_segment(overlap, *shapes)
    assert large is not small and large.kv_c.data_ptr() == small.kv_c.data_ptr()
    del calls[:]
    for outputs, rows in ((small, 2), (large, 4)):
        new_rows = overlap.baked_new_rows(
            outputs.kv_c, outputs.k_pe, outputs.index_k, *_step(rows)
        )
        assert new_rows is not None and new_rows.outputs is outputs
    assert calls == []


def test_every_size_of_a_layer_shape_records_its_own_pulls(monkeypatch):
    """Each graph size records copies of its own segment size: a replay of
    another size never pulls ranges recorded for this one."""
    overlap, calls, tensors = _overlap_on_host(monkeypatch, rows=2)
    small = _record_segment(overlap, *tensors)
    i16 = torch.int16
    shapes = [torch.zeros((4, width), dtype=i16) for width in (6, 2, 4)]
    del calls[:]
    large = _record_segment(overlap, *shapes)
    assert calls == _recorded_calls(overlap, large)
    assert {size for _, _, size in small.copies.copies} == {2 * 8 * 2, 2 * 4 * 2}
    assert {size for _, _, size in large.copies.copies} == {4 * 8 * 2, 4 * 4 * 2}


def test_an_eager_forward_joins_its_exchange_before_the_eager_region(monkeypatch):
    """Without a graph the same two calls start the exchange behind the fused
    norm/rope and join it before the eager region, once: the region's consumers
    find it joined."""
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch)
    outputs = overlap.new_row_outputs(kv_c, k_pe, index_k, *_step())
    new_rows = overlap.issue_new_rows(outputs)
    assert isinstance(new_rows, NewRows) and not new_rows.joined
    assert [call[0] for call in calls] == [
        "record",
        "wait",
        "barrier",
        "copy",
        "barrier",
        "record",
    ]
    del calls[:]
    overlap.join_issued_rows(new_rows)
    assert calls == [("wait", MAIN, overlap._rows_done)] and new_rows.joined
    new_rows.kv(torch.arange(64))
    new_rows.index_k(torch.arange(64))
    overlap.join_issued_rows(None)
    assert calls == [("wait", MAIN, overlap._rows_done)]
    assert not overlap._rows_unjoined


def test_a_recorded_exchange_pulls_with_one_kernel_between_the_barriers(monkeypatch):
    """Copies recorded into a graph run one at a time: a segment records one
    launch of the pull kernel in their place, bound to this layer shape's
    segments."""
    overlap, calls, (kv_c, k_pe, index_k) = _overlap_on_host(monkeypatch, rows=2)
    kernel = _PullKernel(calls)
    overlap._pull = kernel  # type: ignore[assignment]
    outputs = _record_segment(overlap, kv_c, k_pe, index_k)
    ready, done = overlap._ready, overlap._rows_done
    # One rank's rows: `kv_c` | `k_pe` are 8 int16 wide, the indexer K 4.
    a_words, b_words = 2 * 8 * 2 // 8, 2 * 4 * 2 // 8
    assert calls == [
        ("record", ready, MAIN),
        ("wait", AUX, ready),
        ("barrier", pcp_dcp_overlap.CHANNEL_READY),
        ("pull_kernel", AUX, 2, 3),
        ("barrier", pcp_dcp_overlap.CHANNEL_DONE),
        ("record", done, AUX),
        ("wait", MAIN, done),
    ]
    capacity = 4 * overlap.new_rows_capacity
    assert kernel.bound[-1] == dict(
        buffer_ptr=0,
        peer_bases_ptr=overlap._new_exchange.peer_bases.data_ptr(),
        rank=1,
        world=4,
        a_words=a_words,
        b_words=b_words,
        b_start=capacity * 8 * 2 // 8,
        a_chunks=1,
    )
    # A layer without the indexer K pulls region A only.
    del calls[:]
    plain = _record_segment(overlap, kv_c, k_pe, None)
    assert kernel.bound[-1]["b_words"] == 0 and plain.pull_kernel is not None
    assert ("pull_kernel", AUX, 1, 3) in calls
    # An eager exchange keeps its batched copy.
    del calls[:]
    overlap.start_new_rows(outputs)
    assert ("copy", tuple(outputs.copies.copies), AUX) in calls
    assert not any(call[0] == "pull_kernel" for call in calls)


PULL_SCRIPT = """
import torch
from vllm.v1.attention.ops import pcp_dcp_overlap as po

world, rank, block = 4, 1, 64
a_words, b_words, b_start = 150, 40, 1000
bufs = [torch.full((1400,), -(p + 1), dtype=torch.int64) for p in range(world)]
for p in range(world):
    bufs[p][: world * a_words] = torch.arange(world * a_words) + 10_000 * (p + 1)
    region_b = torch.arange(world * b_words) - 10_000 * (p + 1)
    bufs[p][b_start : b_start + world * b_words] = region_b
want = bufs[rank].clone()
for p in range(world):
    if p != rank:
        want[p * a_words : (p + 1) * a_words] = bufs[p][p * a_words : (p + 1) * a_words]
        at = b_start + p * b_words
        want[at : at + b_words] = bufs[p][at : at + b_words]
bases = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64)
a_chunks, b_chunks = -(-a_words // block), -(-b_words // block)
po._pull_segments_kernel[(a_chunks + b_chunks, world - 1)](
    bufs[rank], bases, rank, world, a_words, b_words, b_start, a_chunks, BLOCK=block
)
assert torch.equal(bufs[rank], want), (bufs[rank] != want).nonzero()[:8]
print("ok")
"""


def test_the_pull_kernel_copies_every_peers_segments_and_nothing_else():
    """Run under the Triton interpreter: no GPU needed. This rank's own segments
    and every byte outside the segments keep their values."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", PULL_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_a_capture_that_records_no_exchange_joins_nothing(monkeypatch):
    """A graph size whose rows do not fit, a FULL graph or a vetoed capture
    records plain outputs: no branch, nothing to join."""
    overlap, calls, _ = _overlap_on_host(monkeypatch)
    overlap.test_state.segment = True
    overlap.test_state.mode = CUDAGraphMode.PIECEWISE
    assert overlap.issue_new_rows(None) is None
    overlap.join_issued_rows(None)
    assert calls == []


@pytest.mark.parametrize(
    ("decode_tokens", "slots", "same_dtype", "scheduled", "expected"),
    [
        (0, 16384, True, True, True),
        (1, 16384, True, True, False),
        (None, 16384, True, True, False),
        # Replicated draft decodes do not use the expanded slot mapping.
        (0, 4096, True, True, False),
        (0, 16384, False, True, False),
        (0, 16384, True, False, False),
    ],
)
def test_new_rows_eligibility(decode_tokens, slots, same_dtype, scheduled, expected):
    eligible = new_rows_eligible(decode_tokens, slots, 4096, 4, same_dtype, scheduled)
    assert eligible is expected


def test_single_request_chunk_is_eligible():
    prefill = _prefill()
    chunk, reason = _eligibility(prefill)
    assert chunk is prefill.chunks[0] and reason == "eligible"


@pytest.mark.parametrize(
    ("prefill", "overrides", "reason"),
    [
        (None, {}, "no_prefill"),
        (_prefill(chunks=[]), {}, "no_prefill"),
        (_prefill(), {"num_decode_tokens": 1}, "decode_tokens"),
        (_prefill(chunks=[_chunk(), _chunk()]), {}, "many_chunks"),
        (_prefill(chunks=[_chunk(pcp_deinterleave_idx=None)]), {}, "no_gather"),
        # A query sub-chunk does not read its shard.
        (_prefill(chunks=[_chunk(skip_kv_gather=True)]), {}, "no_gather"),
        (_prefill(), {"global_end": None}, "many_requests"),
        # The dedicated buffers cannot hold the context: serial path.
        (_prefill(), {"capacity_rows": 1_000_000}, "over_capacity"),
        (_prefill(), {"global_new_rows": None}, "uneven_partition"),
        (_prefill(), {"global_new_rows": NEW_ROWS + 4}, "uneven_partition"),
        # A request too short to split is replicated on every rank.
        (_prefill(), {"global_new_rows": 2 * LOCAL_ROWS}, "replicated_rows"),
        (_prefill(chunks=[_chunk(token_end=LOCAL_ROWS // 2)]), {}, "partial_query"),
        (_prefill(), {"global_end": 8192}, "short_context"),
        (_prefill(chunks=[_chunk(total_seq_lens=1_000_000)]), {}, "short_context"),
        # Compressed K rows are not one per token.
        (_prefill(max_prefill_seq_len=4_096_000), {}, "compressed_k"),
    ],
)
def test_ineligible_chunks_use_the_serial_path(prefill, overrides, reason):
    assert _eligibility(prefill, **overrides) == (None, reason)


def test_indexer_with_its_own_decode_split_gathers_serially():
    """The indexer and the main MLA classify short extends independently: with
    decode tokens of its own the indexer does not take the gathered rows."""
    state = PrefillOverlap(4, 4, 64)
    index_k = torch.ones(4, 2)
    outputs = RowOutputs(
        4,
        torch.zeros(1, 4),
        torch.zeros(1, 2),
        index_k[:1],
        torch.zeros(4, 4),
        torch.zeros(4, 1, 2),
        index_k,
        CopyList([]),
    )
    rows = NewRows(state, outputs)
    # Already joined: no exchange to wait for.
    rows.joined = True
    slots = torch.arange(8)
    overlap = IndexerOverlap(state, 2, rows)
    assert overlap.gathered_index_k(slots, 1) is None
    gathered = overlap.gathered_index_k(slots, 0)
    assert gathered is not None
    assert gathered[0] is index_k and torch.equal(gathered[1], slots[:4])
    assert IndexerOverlap(state, 2, None).gathered_index_k(slots, 0) is None


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_staggered_pulls_read_each_owner_once_per_step(world_size):
    orders = [staggered_peers(rank, world_size) for rank in range(world_size)]
    for rank, order in enumerate(orders):
        assert sorted(order + [rank]) == list(range(world_size))
    for step in range(world_size - 1):
        owners = [order[step] for order in orders]
        assert sorted(owners) == list(range(world_size))


@pytest.mark.parametrize(
    ("batched", "pcp", "rows"), [(16384, 4, 4096), (16380, 4, 4096), (4096, 8, 512)]
)
def test_new_rows_capacity_holds_the_padded_partition(batched, pcp, rows):
    assert new_rows_capacity(batched, pcp) == rows


TOKEN_SLOTS_SCRIPT = """
import torch
from vllm.v1.attention.ops import pcp_dcp_k_replica as kr

g = torch.Generator().manual_seed(0)
for rows, extra, start in ((1500, 36, 1_020_000), (1, 0, 0), (1024, 8, 2**33)):
    restore = torch.randperm(rows + extra, generator=g)[:rows]
    slots = torch.full((rows + extra,), -1, dtype=torch.int64)
    token_slots(restore, slots, start)
    positions = torch.arange(start, start + rows, dtype=torch.int64)
    expected = torch.full_like(slots, -1).index_copy_(0, restore, positions)
    assert torch.equal(slots, expected), rows
print("ok")
"""


def test_token_slots_kernel_matches_the_index_copy():
    """Run under the Triton interpreter: no GPU needed."""
    env = {**os.environ, "TRITON_INTERPRET": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        [sys.executable, "-c", inspect.getsource(launch) + TOKEN_SLOTS_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-2000:]


def test_replica_memory_per_rank():
    rows = 1_048_576 + K_CAPACITY_SLACK_ROWS
    assert replica_bytes(rows, 128, 1) == 138_952_704
    assert replica_bytes(rows, 128, 21) == 2_918_006_784


@pytest.mark.parametrize(
    "req_id, start, held_req_id, held_rows, appends",
    [
        ("a", 0, None, -1, True),
        ("a", 0, "b", 4096, True),
        ("a", 4096, "a", 4096, True),
        ("a", 4096, None, -1, False),
        ("a", 4096, "b", 4096, False),
        ("a", 4096, "a", 8192, False),
        ("a", 8192, "a", 4096, False),
    ],
)
def test_replica_appends_only_the_next_rows_of_the_request_it_holds(
    req_id, start, held_req_id, held_rows, appends
):
    assert replica_appends(req_id, start, held_req_id, held_rows) is appends


def test_replica_values_and_scales_are_the_rows_of_a_one_block_cache():
    """The cache-write kernel puts row `i`'s values at `[i * D, (i + 1) * D)` of a
    block and its scale at `capacity * D + 4 * i`: the replica's views must read
    exactly those bytes."""
    capacity, head_dim = 16, 8
    buffer = torch.arange(capacity * (head_dim + 4), dtype=torch.int64).to(torch.uint8)
    replica = LayerReplica(buffer, capacity, head_dim)
    assert replica.cache.shape == (1, capacity, head_dim + 4)
    assert replica.values.shape == (capacity, head_dim)
    assert replica.scales.shape == (capacity, 4)
    assert replica.values.is_contiguous() and replica.scales.is_contiguous()
    values = buffer[: capacity * head_dim].view(capacity, head_dim)
    assert torch.equal(replica.values.view(torch.uint8), values)
    assert torch.equal(replica.scales, buffer[capacity * head_dim :].view(capacity, 4))
    replica.req_id, replica.rows = "a", 8
    replica.invalidate()
    assert replica.req_id is None and replica.rows == -1


@pytest.mark.parametrize("max_model_len", [1_048_576, 1_024_000, 163_840, 65_537])
def test_replica_k_starts_where_the_top_k_kernels_can_read_it(max_model_len):
    """The top-k kernels take the K values and the scales as contiguous rows at
    16-byte multiples: both regions start at one whatever the capacity."""
    capacity, head_dim = max_model_len + K_CAPACITY_SLACK_ROWS, 128
    buffer = torch.empty(0, dtype=torch.uint8)
    replica = LayerReplica.__new__(LayerReplica)
    replica.buffer, replica.capacity, replica.head_dim = buffer, capacity, head_dim
    values_bytes = capacity * head_dim
    assert values_bytes % 16 == 0
    assert replica_bytes(capacity, head_dim, 1) == values_bytes + 4 * capacity


def _scheduled_manager(monkeypatch, scheduled: bool = True, **fields):
    """The process's PCP manager, as the overlaps see it."""
    from vllm.v1.worker.gpu import pcp_manager

    manager = SimpleNamespace(scheduled_batch=scheduled, **fields)
    monkeypatch.setattr(pcp_manager, "_ACTIVE_PCP_MANAGER", manager)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return manager


def test_a_step_is_scheduled_only_outside_captures_and_dummy_batches(monkeypatch):
    """The overlaps act on batches `partition_batch` laid out. A warmup or profile
    batch bypasses it, and a capture runs its eager regions for real: none of them
    may reach the replicas or issue a barrier."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.v1.worker.gpu import pcp_manager

    monkeypatch.setattr(pcp_manager, "_ACTIVE_PCP_MANAGER", None)
    assert not cp_common.scheduled_step()
    manager = _scheduled_manager(monkeypatch)
    assert cp_common.scheduled_step() and not cp_common.capturing_run()
    manager.scheduled_batch = False
    assert not cp_common.scheduled_step()
    manager.scheduled_batch = True
    # The eager region of a breakable capture: the stream is not capturing.
    monkeypatch.setattr(BreakableCUDAGraphCapture._tls, "active", object(), False)
    assert cp_common.capturing_run() and not cp_common.scheduled_step()
    monkeypatch.setattr(BreakableCUDAGraphCapture._tls, "active", None, False)
    assert cp_common.scheduled_step()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert cp_common.capturing_run() and not cp_common.scheduled_step()


def _replica_step(monkeypatch, scheduled: bool):
    """A `KReplica` with one layer at the step that prefills rows
    `[1_007_616, 1_024_000)` of request `r`, under a piecewise CUDA graph."""
    restore_idx = torch.zeros(NEW_ROWS, dtype=torch.int64)
    _scheduled_manager(
        monkeypatch,
        scheduled,
        hidden_restore_idx=restore_idx if scheduled else None,
        single_request_end=1_024_000 if scheduled else None,
        global_batch=SimpleNamespace(req_ids=["r"]),
    )
    prefill = _prefill()
    context = SimpleNamespace(
        attn_metadata={"k": SimpleNamespace(prefill=prefill, num_decode_tokens=0)},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
    )
    monkeypatch.setattr(pcp_dcp_k_replica, "get_forward_context", lambda: context)
    monkeypatch.setattr(pcp_dcp_overlap, "get_forward_context", lambda: context)
    replica = KReplica(4, 4, 1_052_672)
    replica._indexers[3] = SimpleNamespace(k_cache=SimpleNamespace(prefix="k"))
    layer = replica.layers[3] = LayerReplica(torch.zeros(48, dtype=torch.uint8), 4, 8)
    layer.req_id, layer.rows = "r", 1_007_616
    return replica, layer, prefill.chunks[0]


def test_the_replica_serves_a_scheduled_step_under_a_piecewise_graph(monkeypatch):
    """Its entry points run in the eager regions of a replayed graph: the mode of
    the forward does not matter, only that the batch was scheduled."""
    replica, layer, chunk = _replica_step(monkeypatch, scheduled=True)
    decision = replica.decide(3, LOCAL_ROWS)
    assert decision is not None and decision.chunk is chunk
    assert (decision.start, decision.end) == (1_007_616, 1_024_000)
    # Decided once per step.
    assert replica.decide(3, LOCAL_ROWS) is decision


def test_rows_the_replica_does_not_hold_gather_the_k_serially(monkeypatch):
    """A chunk that does not continue the replica's rows (a prefix cache hit, a
    resumed request, a step the replica missed) takes the serial gather, and the
    replica stays invalid for the rest of the request."""
    replica, layer, _ = _replica_step(monkeypatch, scheduled=True)
    layer.rows = 1_003_520
    assert replica.decide(3, LOCAL_ROWS) is None
    assert layer.req_id is None and layer.rows == -1


def test_a_dummy_batch_invalidates_the_replica(monkeypatch):
    """A capture or warmup forward after a real step would otherwise append the
    dummy batch's rows to what the replica holds of the last request."""
    replica, layer, _ = _replica_step(monkeypatch, scheduled=False)
    assert replica.decide(3, LOCAL_ROWS) is None
    assert layer.req_id is None and layer.rows == -1


@pytest.mark.parametrize("pcp", [2, 4, 8])
@pytest.mark.parametrize("fused_mqa_topk_eligible", [False, True])
def test_every_rank_splits_a_prefill_into_the_same_indexer_chunks(
    pcp, fused_mqa_topk_eligible
):
    """No rank asks another whether a layer's replica serves a step: the decision
    rests on the indexer chunks, among other things, and PCP gives the ranks
    different rows. The builder plans the chunks from the request's largest shard
    and from rows x longest row, which every rank computes alike: for random
    split prefills laid out by the real PCP partition, every rank gets the same
    chunks, whether or not the plan is the one for the fused top-k kernel."""
    import numpy as np

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    rng = np.random.default_rng(7 * pcp + int(fused_mqa_topk_eligible))
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.dcp_world_size = pcp
    builder.fused_mqa_topk_eligible = fused_mqa_topk_eligible
    builder.max_prefill_buffer_size = 40 * 2**20
    several = 0
    for _ in range(200):
        scheduled = np.array([int(rng.integers(2 * pcp, 40_000))], dtype=np.int32)
        computed = np.array([int(rng.integers(0, 1_040_000))], dtype=np.int32)
        prefilling = np.ones(1, dtype=np.bool_)
        budget = int(rng.choice([64, 512, 2048])) * 2**20
        starts = np.concatenate([[0], np.cumsum(scheduled)]).astype(np.int32)
        plans = []
        for rank in range(pcp):
            manager = PCPManager(
                pcp_world_size=pcp,
                pcp_rank=rank,
                device=torch.device("cpu"),
                shard_decode_requests=False,
                dcp_world_size=pcp,
            )
            if manager.replicated_requests(scheduled, prefilling)[0]:
                break
            segments = manager._get_rank_segments(rank, scheduled, prefilling, starts)
            rows = np.array([segment.global_batch_req_idx for segment in segments])
            lens = torch.tensor([segment.num_tokens for segment in segments])
            ends = torch.from_numpy((computed + scheduled)[rows])
            shard_rows = get_dcp_local_seq_lens(ends, pcp, 0, 1).numpy()
            plans.append(
                builder._split_pcp_dcp_prefill_chunks(
                    rows, shard_rows, lens, budget, request_offset=0
                )
            )
        if not plans:
            continue
        assert all(plan == plans[0] for plan in plans), (scheduled, computed, budget)
        several += len(plans[0]) > 1
    # Both outcomes occur: one chunk (the replica can serve) and several.
    assert several > 0
