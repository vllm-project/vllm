# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather a PCP prefill layer's newly computed MLA KV and indexer K rows off
the main stream.

Under PCP a DeepSeek V3.2-style prefill layer runs two all-gathers of newly
computed rows on the main stream before its cache writes: its new `kv_c`/`k_pe`
rows and its new indexer K rows across the PCP ranks. Enabled by
`ParallelConfig.pcp_new_row_overlap`, the fused norm/rope writes each rank's
`kv_c`/`k_pe` rows, and its indexer K rows, directly into that rank's segments
of a peer-mapped buffer across the PCP group, and every rank pulls the other
PCP ranks' segments in staggered order on an auxiliary stream: with the copy
engines in an eager forward, and with one copy kernel recorded into the graph
under a breakable CUDA graph. The indexer and the cache update find the rows
gathered where they consume them.

The thread that runs the model issues all of it, in program order, and is the
only one that enqueues on the auxiliary stream. Each piece of auxiliary work
first waits for an event recorded on the model's stream where it is issued, and
a consumer makes the model's stream wait for an event recorded behind that
work. Per layer the exchange of the new rows is a handful of driver calls: the
fused norm/rope writes straight into this rank's segment (`new_row_outputs`),
and one batched copy between the two barriers pulls the other ranks' segments
(`start_new_rows`).

Under a breakable CUDA graph. A replayed graph runs the captured segments
without their Python, and each layer's indexer and attention in an eager region
between them. The fused norm/rope is captured, so where it writes is decided at
capture, once per graph size, from what that size fixes: a graph whose rows fit
writes them into this rank's segments of the buffer, which is allocated while
the model is built and never moves. The exchange is recorded into the same
segment, as a branch that forks behind the fused norm/rope, runs beside the
projections that follow it and rejoins before the eager region
(`issue_new_rows`, `join_issued_rows`): a replay issues it with no host call,
and its pull kernel and barriers depend only on the graph's size, which every
PCP rank replays alike. Copies recorded into a graph run one at a time, so the
pulls are recorded as one kernel (`_pull_segments_kernel`); a graph size whose
segments that kernel cannot copy keeps the serial gather. Whether a step uses
the exchanged rows is decided in the eager region, from the step's own metadata
(`baked_new_rows`): any other step gathers the same rows serially from this
rank's segments.

Every eligibility decision depends only on the global schedule and on the graph
a step replays, the same on all PCP ranks, so they take the same path and issue
the same barriers. Anything not eligible runs the serial path. A capture,
warmup or profile forward runs a dummy batch: nothing here acts on it
(`cp_common.scheduled_step`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.ops.cp_common import (
    capturing_breakable_segment,
    capturing_run,
    scheduled_step,
)
from vllm.v1.attention.ops.cuda_direct import (
    CopyList,
    CudaDriver,
    DirectKernel,
    cuda_driver,
    direct_launch,
)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

BARRIER_TIMEOUT_MS = 60_000
# Signal-pad channels of the copy-engine exchange.
CHANNEL_READY = 1
CHANNEL_DONE = 2


def new_rows_eligible(
    num_decode_tokens: int | None,
    slot_mapping_rows: int,
    local_rows: int,
    pcp_world_size: int,
    same_dtype: bool,
    scheduled: bool,
) -> bool:
    """Whether a layer's new rows are the plain rank-major gather of every PCP
    rank.

    `scheduled` is False for a batch the scheduler did not make. Where the
    fused norm/rope is recorded into a CUDA graph no step is known yet: that
    caller passes what a scheduled prefill of the graph's size has, and the
    eager region behind the recorded segment asks again with each step's own
    values.
    """
    return (
        scheduled
        and same_dtype
        and num_decode_tokens == 0
        and local_rows > 0
        and slot_mapping_rows == pcp_world_size * local_rows
    )


# Words of 8 bytes one program of the recorded pull copies.
PULL_BLOCK_WORDS = 4096


@triton.jit(
    do_not_specialize=["rank", "world", "a_words", "b_words", "b_start", "a_chunks"]
)
def _pull_segments_kernel(
    buffer_ptr,
    peer_bases_ptr,
    rank: tl.int32,
    world: tl.int32,
    a_words: tl.int64,
    b_words: tl.int64,
    b_start: tl.int64,
    a_chunks: tl.int32,
    BLOCK: tl.constexpr,
):
    """Copy every peer's segments of the new-row buffer into the same offsets of
    this rank's buffer: program (c, j) copies chunk c of peer j (staggered order)
    of region A, or of region B from chunk `a_chunks` on."""
    chunk = tl.program_id(0)
    peer = (rank + world - 1 - tl.program_id(1)) % world
    in_b = chunk >= a_chunks
    words = tl.where(in_b, b_words, a_words)
    start = tl.where(in_b, b_start + peer * b_words, peer * a_words)
    chunk = tl.where(in_b, chunk - a_chunks, chunk)
    idx = chunk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < words
    remote = tl.load(peer_bases_ptr + peer).to(tl.pointer_type(tl.int64))
    values = tl.load(remote + start + idx, mask=mask)
    tl.store(buffer_ptr + start + idx, values, mask=mask)


def staggered_peers(rank: int, world_size: int) -> list[int]:
    """Peers in the order `rank` pulls from them: no two ranks read one owner at
    once, and a rank's own segment is already in place."""
    return [(rank - step) % world_size for step in range(1, world_size)]


def new_rows_capacity(max_num_batched_tokens: int, pcp_world_size: int) -> int:
    """Rows of one rank's new-row segment: PCP pads the batch to 2 * PCP ranks."""
    pairs = -(-max_num_batched_tokens // (2 * pcp_world_size))
    return 2 * pairs


def _exchange_group() -> GroupCoordinator:
    """The PCP group for new-row exchange, reusing the DCP group when both
    span the exact same ranks in the same order so PyTorch symmetric memory
    shares one rendezvous context across KV prefetch and new-row overlap."""
    pcp = get_pcp_group()
    dcp = get_dcp_group()
    return dcp if dcp.ranks == pcp.ranks else pcp


class SymmExchange:
    """A peer-mapped byte buffer across the PCP group whose rank-major segments
    the copy engines gather.

    A rank writes its own segment, then every PCP rank pulls the others'
    segments into the same offsets of its own buffer: afterwards each buffer
    holds the gather. Two barriers bracket the pulls: no rank reads a segment
    before its owner wrote it, and no owner overwrites a segment, in its next
    exchange, before every peer read it.
    """

    def __init__(self, buffer: torch.Tensor) -> None:
        from vllm.v1.attention.ops.cp_common import symm_mem

        group = _exchange_group()
        self.buffer = buffer
        self.nbytes = buffer.numel()
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        buffer.zero_()
        torch.accelerator.synchronize()
        self.handle = symm_mem.rendezvous(buffer, group.device_group.group_name)
        self.handle.barrier()
        self.remote = [
            self.handle.get_buffer(peer, [self.nbytes], torch.uint8, 0)
            for peer in range(self.world_size)
        ]
        self.ptr = buffer.data_ptr()
        self.remote_ptrs = [remote.data_ptr() for remote in self.remote]
        # Every rank's buffer address here, for a kernel that pulls the peers.
        self.peer_bases = torch.tensor(
            self.remote_ptrs, dtype=torch.int64, device=buffer.device
        )

    def view(
        self, offset: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        size = dtype.itemsize
        for dim in shape:
            size *= dim
        return self.buffer[offset : offset + size].view(dtype).view(shape)

    def copies(self, spans: Callable[[int], list[tuple[int, int]]]) -> CopyList:
        """The pulls of every peer's byte ranges `spans(peer)`, in staggered
        order."""
        return CopyList(
            [
                (self.ptr + begin, self.remote_ptrs[peer] + begin, end - begin)
                for peer in staggered_peers(self.rank, self.world_size)
                for begin, end in spans(peer)
            ]
        )

    def pull(
        self,
        copies: CopyList,
        driver: CudaDriver,
        stream: int,
        kernel: Callable[[int], None] | None = None,
    ) -> None:
        """Pull the peers' ranges on the current stream, whose handle is
        `stream`, between the two barriers: all of them with one driver call,
        or with `kernel(stream)`, a kernel that copies the same ranges."""
        self.handle.barrier(channel=CHANNEL_READY, timeout_ms=BARRIER_TIMEOUT_MS)
        if kernel is None:
            driver.memcpy_list(copies, stream)
        else:
            kernel(stream)
        self.handle.barrier(channel=CHANNEL_DONE, timeout_ms=BARRIER_TIMEOUT_MS)


def try_symm_exchange(nbytes: int, device: torch.device) -> SymmExchange | None:
    """Allocate an exchange on every PCP rank, or on none of them."""
    from vllm.v1.attention.ops.cp_common import direct_cp_enabled, symm_mem

    group = _exchange_group()
    buffer = None
    if direct_cp_enabled(group, torch.uint8, None):
        try:
            buffer = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
        except (RuntimeError, ValueError) as error:
            logger.warning("PCP copy-engine gather unavailable: %s", error)
    local = torch.tensor([int(buffer is not None)], dtype=torch.int64)
    flags = [torch.empty_like(local) for _ in range(group.world_size)]
    # A rendezvous with a rank that could not allocate would never return.
    dist.all_gather(flags, local, group=group.cpu_group)
    if buffer is None or not all(int(f[0]) == 1 for f in flags):
        return None
    return SymmExchange(buffer)


@dataclass
class RowOutputs:
    """Where one layer shape's new rows live in the peer-mapped buffer.

    `kv_c`, `k_pe` and `index_k` are this rank's segments, which the fused
    norm/rope writes; the `gathered_*` views hold every PCP rank's rows once
    the exchange completed. `copies` pulls the other ranks' segments with the
    copy engines (an eager exchange), and `pull_kernel` does the same as one
    kernel (a graph segment's exchange; None where it cannot be launched
    directly, and then no segment records the exchange). `captured` is set once
    a graph segment recorded the exchange behind a fused norm/rope that writes
    them.
    """

    rows: int
    kv_c: torch.Tensor
    k_pe: torch.Tensor
    index_k: torch.Tensor | None
    gathered_kv: torch.Tensor
    gathered_pe: torch.Tensor
    gathered_k: torch.Tensor | None
    copies: CopyList
    pull_kernel: Callable[[int], None] | None = None
    captured: bool = False


class NewRows:
    """A layer's new rows, gathered on the auxiliary stream."""

    __slots__ = ("overlap", "outputs", "joined")

    def __init__(
        self, overlap: PrefillOverlap, outputs: RowOutputs, joined: bool = False
    ) -> None:
        self.overlap = overlap
        self.outputs = outputs
        self.joined = joined

    def _join(self) -> None:
        if not self.joined:
            self.overlap.join_new_rows()
            self.joined = True

    def index_k(
        self, slot_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the gathered indexer K and its slots."""
        outputs = self.outputs
        if outputs.gathered_k is None:
            return None
        self._join()
        return outputs.gathered_k, slot_mapping[: outputs.rows]

    def kv(
        self, slot_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the gathered `kv_c`, `k_pe` and their slots."""
        self._join()
        outputs = self.outputs
        return outputs.gathered_kv, outputs.gathered_pe, slot_mapping[: outputs.rows]


class PrefillOverlap:
    """Per-process state of the overlapped PCP new-row gather; one per worker
    process."""

    def __init__(
        self,
        pcp_world_size: int,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        self.pcp_world_size = pcp_world_size
        self.new_rows_capacity = (
            None
            if max_num_batched_tokens is None
            else new_rows_capacity(max_num_batched_tokens, pcp_world_size)
        )
        # None until allocated, False where it cannot be allocated on every rank.
        # It is allocated while the model is built (`allocate_new_rows`):
        # captured graphs record addresses inside it.
        self._new_exchange: SymmExchange | bool | None = None
        # The auxiliary stream, made with the exchange, and the events that
        # order it with the stream that runs the model.
        self._stream: torch.cuda.Stream | None = None
        self._driver: CudaDriver | None = None
        self._outputs: dict[tuple[Any, ...], RowOutputs] = {}
        # The widths of region A (`kv_c` + `k_pe`) and of region B (indexer K).
        self._row_layout: tuple[int, int, torch.dtype] | None = None
        # The main stream has not waited for the last exchange of new rows yet.
        self._rows_unjoined = False
        # The kernel of the pulls a graph segment records (`allocate_new_rows`).
        self._pull: DirectKernel | None = None

    def _auxiliary(self, device: torch.device) -> torch.cuda.Stream:
        """The auxiliary stream. Only the thread that runs the model enqueues
        on it, in program order, like on its own stream."""
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=device)
            self._stream_handle = self._stream.cuda_stream
            self._driver = driver = cuda_driver()
            self._ready = driver.event()
            self._rows_done = driver.event()
        return self._stream

    def allocate_new_rows(
        self,
        kv_dim: int,
        pe_dim: int,
        k_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate the buffer of the new-row exchange, with the auxiliary
        stream and its events, for rows of `kv_dim` + `pe_dim` elements of
        `dtype` and indexer K rows of up to `k_dim`.

        Called once, while the model is built: a graph captured later records
        addresses inside the buffer, so it is never allocated again, and no step
        pays for any of it. Collective over the PCP group, whose ranks build
        the same layers in the same order; where any rank cannot have it, every
        rank gathers the new rows serially.
        """
        if self._new_exchange is not None:
            return
        self._new_exchange = False
        if self.new_rows_capacity is None:
            return
        capacity = self.pcp_world_size * self.new_rows_capacity
        row_bytes = (kv_dim + pe_dim + k_dim) * dtype.itemsize
        exchange = try_symm_exchange(capacity * row_bytes, device)
        if exchange is None:
            logger.warning_once(
                "pcp_new_row_overlap: the copy-engine gather of the new rows is "
                "unavailable; using the serial gather."
            )
            return
        self._new_exchange = exchange
        self._row_layout = (kv_dim + pe_dim, k_dim, dtype)
        self._auxiliary(device)
        # The pull a graph segment records: one kernel, compiled and loaded now.
        self._pull = direct_launch(
            _pull_segments_kernel,
            {"buffer_ptr": torch.int64, "peer_bases_ptr": torch.int64},
            BLOCK=PULL_BLOCK_WORDS,
        )
        if self._pull is None:
            logger.warning_once(
                "pcp_new_row_overlap: the pull kernel of the new rows cannot be "
                "launched directly; CUDA graph replays gather the new rows serially."
            )

    def _new_rows_eligible(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: Any,
        slot_mapping: torch.Tensor | None,
    ) -> bool:
        """Whether the step in progress exchanges a layer's new rows."""
        if attn_metadata is None or slot_mapping is None:
            return False
        return new_rows_eligible(
            getattr(attn_metadata, "num_decode_tokens", None),
            slot_mapping.shape[0],
            kv_c.shape[0],
            self.pcp_world_size,
            kv_c.dtype == k_pe.dtype,
            scheduled_step(),
        )

    def new_row_outputs(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
        attn_metadata: Any,
        slot_mapping: torch.Tensor | None,
    ) -> RowOutputs | None:
        """Where the fused norm/rope should write a layer's new rows so that the
        copy engines can gather them: this rank's segments of the peer-mapped
        buffer. None when the layer gathers them serially."""
        if capturing_breakable_segment():
            local_rows = kv_c.shape[0]
            fits = new_rows_eligible(
                0,
                self.pcp_world_size * local_rows,
                local_rows,
                self.pcp_world_size,
                kv_c.dtype == k_pe.dtype,
                True,
            )
            if not fits:
                return None
            outputs = self._outputs_for(kv_c, k_pe, index_k)
            if outputs is None or outputs.pull_kernel is None:
                return None
            return outputs
        if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE:
            return None
        if not self._new_rows_eligible(kv_c, k_pe, attn_metadata, slot_mapping):
            return None
        outputs = self._outputs_for(kv_c, k_pe, index_k)
        if outputs is not None and self._rows_unjoined:
            # Peers and this rank's consumers finished with the last exchange
            # before its segment is written again.
            self.join_new_rows()
        return outputs

    @staticmethod
    def _outputs_key(
        kv_c: torch.Tensor, k_pe: torch.Tensor, index_k: torch.Tensor | None
    ) -> tuple[Any, ...]:
        """Everything the views depend on: layers that differ in any of it do
        not share them."""
        return (
            kv_c.shape,
            k_pe.shape,
            kv_c.dtype,
            None if index_k is None else (index_k.shape, index_k.dtype),
        )

    def _outputs_for(
        self, kv_c: torch.Tensor, k_pe: torch.Tensor, index_k: torch.Tensor | None
    ) -> RowOutputs | None:
        key = self._outputs_key(kv_c, k_pe, index_k)
        outputs = self._outputs.get(key)
        if outputs is None:
            outputs = self._make_outputs(kv_c, k_pe, index_k)
            if outputs is not None:
                self._outputs[key] = outputs
        return outputs

    def _make_outputs(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        index_k: torch.Tensor | None,
    ) -> RowOutputs | None:
        """The views and pulls of layers with `kv_c.shape[0]` rows per rank,
        with or without the indexer K.

        The buffer has two rank-major regions: A holds `kv_c` and `k_pe` of one
        token side by side, B the indexer K, whose cache-write kernel takes
        contiguous rows.
        """
        local_rows = kv_c.shape[0]
        kv_dim, pe_dim = kv_c.shape[1], k_pe.shape[1]
        dtype = kv_c.dtype
        k_dim = 0 if index_k is None else index_k.shape[1]
        exchange, layout = self._new_exchange, self._row_layout
        if (
            not isinstance(exchange, SymmExchange)
            or layout is None
            or self.new_rows_capacity is None
        ):
            return None
        a_width, b_width, layout_dtype = layout
        if (
            local_rows > self.new_rows_capacity
            or (kv_dim + pe_dim, dtype) != (a_width, layout_dtype)
            or k_dim > b_width
            or (index_k is not None and index_k.dtype != dtype)
        ):
            return None
        capacity = self.pcp_world_size * self.new_rows_capacity
        a_bytes = capacity * a_width * dtype.itemsize
        rank, world = exchange.rank, self.pcp_world_size
        rows = world * local_rows
        mine = slice(rank * local_rows, (rank + 1) * local_rows)
        region_a = exchange.view(0, (rows, a_width), dtype)
        region_b = (
            exchange.view(a_bytes, (rows, k_dim), dtype)
            if index_k is not None
            else None
        )
        a_segment = local_rows * a_width * dtype.itemsize
        b_segment = local_rows * k_dim * dtype.itemsize
        pull_kernel = self._pull_kernel_for(exchange, a_segment, b_segment, a_bytes)

        def spans(peer: int) -> list[tuple[int, int]]:
            ranges = [(peer * a_segment, (peer + 1) * a_segment)]
            if b_segment:
                start = a_bytes + peer * b_segment
                ranges.append((start, start + b_segment))
            return ranges

        return RowOutputs(
            rows,
            region_a[mine, :kv_dim],
            region_a[mine, kv_dim:],
            None if region_b is None else region_b[mine],
            region_a[:, :kv_dim],
            region_a[:, kv_dim:].unsqueeze(1),
            region_b,
            exchange.copies(spans),
            pull_kernel,
        )

    def _pull_kernel_for(
        self, exchange: SymmExchange, a_segment: int, b_segment: int, b_start: int
    ) -> Callable[[int], None] | None:
        """The pull of `_make_outputs`' ranges as one kernel launch on a stream:
        every peer's region A segment of `a_segment` bytes at `peer * a_segment`
        and region B segment of `b_segment` bytes at `b_start + peer * b_segment`.
        None where the kernel cannot be launched directly or a range is not made
        of whole words."""
        kernel = self._pull
        if kernel is None or (a_segment | b_segment | b_start) % 16:
            return None
        a_words, b_words = a_segment // 8, b_segment // 8
        a_chunks = -(-a_words // PULL_BLOCK_WORDS)
        b_chunks = -(-b_words // PULL_BLOCK_WORDS)
        launch = kernel.bind(
            buffer_ptr=exchange.ptr,
            peer_bases_ptr=exchange.peer_bases.data_ptr(),
            rank=exchange.rank,
            world=exchange.world_size,
            a_words=a_words,
            b_words=b_words,
            b_start=b_start // 8,
            a_chunks=a_chunks,
        )
        grid = (a_chunks + b_chunks, exchange.world_size - 1)

        def pull(stream: int) -> None:
            launch(stream, *grid)

        return pull

    def issue_new_rows(self, outputs: RowOutputs | None) -> NewRows | None:
        """Right behind the fused norm/rope that wrote `outputs`: start their
        exchange, which `join_issued_rows` joins before the layer's eager
        region."""
        if outputs is None:
            return None
        if capturing_breakable_segment():
            self.start_new_rows(outputs, recorded=True)
            outputs.captured = True
            return None
        if capturing_run():
            return None
        return self.start_new_rows(outputs)

    def join_issued_rows(self, new_rows: NewRows | None) -> None:
        """Before a layer's eager region: make the current stream wait for the
        exchange `issue_new_rows` started, so that a branch recorded into a
        graph segment rejoins that segment. `new_rows` is what it returned."""
        if self._rows_unjoined:
            self.join_new_rows()
        if new_rows is not None:
            new_rows.joined = True

    def baked_new_rows(
        self,
        kv_c: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        index_k: torch.Tensor | None,
        attn_metadata: Any,
        slot_mapping: torch.Tensor | None,
    ) -> NewRows | None:
        """In a layer's eager region, for new rows a replayed graph segment
        wrote and exchanged: the gathered rows, when the step in progress uses
        them."""
        if kv_c is None or k_pe is None:
            return None
        outputs = self._outputs.get(self._outputs_key(kv_c, k_pe, index_k))
        if (
            outputs is None
            or not outputs.captured
            or outputs.kv_c.data_ptr() != kv_c.data_ptr()
        ):
            return None
        if not self._new_rows_eligible(kv_c, k_pe, attn_metadata, slot_mapping):
            return None
        return NewRows(self, outputs, joined=True)

    def start_new_rows(self, outputs: RowOutputs, recorded: bool = False) -> NewRows:
        """Gather the new rows a layer wrote to `outputs` on the current
        stream: behind them, on the auxiliary stream, every PCP rank pulls the
        other ranks' segments between the exchange's two barriers."""
        exchange, driver = self._new_exchange, self._driver
        assert isinstance(exchange, SymmExchange) and driver is not None
        assert self._stream is not None
        assert outputs.pull_kernel is not None or not recorded
        main = current_stream()
        aux = self._stream_handle
        driver.record(self._ready, main.cuda_stream)
        driver.wait_event(aux, self._ready)
        torch.cuda.set_stream(self._stream)
        try:
            exchange.pull(
                outputs.copies,
                driver,
                aux,
                outputs.pull_kernel if recorded else None,
            )
        finally:
            torch.cuda.set_stream(main)
        driver.record(self._rows_done, aux)
        self._rows_unjoined = True
        return NewRows(self, outputs)

    def join_new_rows(self) -> None:
        """Make the current stream wait for the last exchange of new rows."""
        assert self._driver is not None
        self._driver.wait_event(current_stream().cuda_stream, self._rows_done)
        self._rows_unjoined = False


@dataclass
class IndexerNewRows:
    """Per-call handle handing gathered new indexer K rows to
    `sparse_attn_indexer`."""

    new_rows: NewRows

    def gathered_index_k(
        self, slot_mapping: torch.Tensor, num_decode_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return the new K rows, unless the indexer splits them differently."""
        if num_decode_tokens != 0:
            return None
        return self.new_rows.index_k(slot_mapping)


_PREFILL_OVERLAP: PrefillOverlap | None = None
# sparse_attn_indexer is a registered custom op: non-tensor state cannot be an argument.
_INDEXER_NEW_ROWS: IndexerNewRows | None = None


def set_indexer_new_rows(rows: IndexerNewRows) -> None:
    """Hand `rows` to the next `sparse_attn_indexer` call on this thread."""
    global _INDEXER_NEW_ROWS
    _INDEXER_NEW_ROWS = rows


def pop_indexer_new_rows() -> IndexerNewRows | None:
    """Take what `set_indexer_new_rows` handed over, if anything."""
    global _INDEXER_NEW_ROWS
    rows, _INDEXER_NEW_ROWS = _INDEXER_NEW_ROWS, None
    return rows


def prefill_overlap_for_layer(
    enabled: bool,
    use_pcp: bool,
    pcp_world_size: int,
    max_num_batched_tokens: int | None = None,
    new_row_dims: tuple[int, int, int] | None = None,
    dtype: torch.dtype | None = None,
) -> PrefillOverlap | None:
    """Register an attention layer with the process's new-row overlap state,
    when enabled.

    `new_row_dims` are the model's widths of `kv_c`, `k_pe` and the indexer K,
    and `dtype` their element type: the first layer allocates the buffer of the
    new-row exchange for them.
    """
    global _PREFILL_OVERLAP
    if not enabled or not use_pcp or pcp_world_size <= 1:
        return None
    first = _PREFILL_OVERLAP is None
    if _PREFILL_OVERLAP is None:
        _PREFILL_OVERLAP = PrefillOverlap(pcp_world_size, max_num_batched_tokens)
    if first and new_row_dims is not None and dtype is not None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
        _PREFILL_OVERLAP.allocate_new_rows(*new_row_dims, dtype, device)
    return _PREFILL_OVERLAP
