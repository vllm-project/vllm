# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hide the PCP+DCP prefill KV all-gather behind the previous layer's attention.

Each layer attends its whole context, so it upconverts its KV shard and all-gathers
the shards right before its attention kernel. While layer `l` attends, an auxiliary
stream prepares layer `l + 1`'s gathered context in a region of its own, and the
hardware copy engines read every rank's rows through peer-mapped symmetric memory.
Adjacent layers alternate between two regions, so one is read while the next
arrives. Scratch from `WorkspaceManager.get_simultaneous` is reused by whatever runs
between two layers, so both regions live in their own allocation.

Rows on the wire. A rank packs the raw `fp8_ds_mla` cache rows (656 bytes) into a
symmetric buffer. Every rank copies each peer's rows into a local staging area and
upconverts them into the region with the kernel the serial path uses; its own rows
are upconverted straight from its cache. The region is the serial result byte for
byte.

The chunk's own rows. The chunk's new `kv_c`/`k_pe` rows are all-gathered across
PCP before the cache write, so every rank already holds every rank's new rows. Each
rank quantizes all of them with the cache-write kernel into a local scratch block
cache and upconverts them into every peer's slot range
`[p * shard + old_p, p * shard + new_p)` of the region, on the auxiliary stream as
soon as the rows exist (`stage_new_rows`). The head then moves each peer's rows from
before the chunk, `[0, old_p)`, and its padding rows `[new_p, shard)`; cache rows
`[old_p, new_p)` never travel.

Ordering the copies. Peers read rank `r`'s packed rows only in a head's copies and
in the top-up kernel (see "Selected rows only" below), both on their auxiliary
stream. Rank `r` must not pack rows over them before every peer finished reading.
A barrier after the head copies would order that, but a barrier is a kernel that
waits for SMs, so a long main-stream kernel delays it and with it the join. The
head therefore issues no such barrier; rank `r` owes it (`head_done_owed`) and pays
it as a HEAD_DONE barrier before it next packs: on the auxiliary stream at the
start of the next `start_next`, or on the main stream when a layer gathers serially
(`join` returns None), after the main stream waited for the auxiliary one so the
barrier follows this rank's own copies.

Every rank makes the same `start_next` and `join` calls in the same order (their
barriers already require it) and the debt changes only in them, so all ranks pay
the same barriers. No barrier channel is signalled again before every rank consumed
its previous signal: between two HEAD_READY of a rank lies a HEAD_DONE that every
rank passed after consuming the first.

Copies in chunks. The head copies run in the background of the main stream's
compute, but they share the links with its collectives, the MoE all-to-all above
all, which every rank waits for. A head's copies therefore go out in chunks of at
most `COPY_CHUNK_BYTES` (`head_chunks`), one copy call per chunk on the auxiliary
stream: issued as one copy per peer, a long-context head overlapped the all-to-all
for longer and made it slower on every rank.

The plan. Every rank issues the same barriers or none, so every rank must split a
step the same way. No rank asks another: a step is planned only when the
scheduler's batch is one prefill request that PCP splits across the ranks, and the
plan is then a function of that request's context start and end, of the
configuration and of the layers (`_derive_plan`). A rank whose own metadata does
not show what the schedule implies raises instead of planning alone. As a net
under that argument each rank writes a hash of its plan to peer-mapped memory with
the step's tables, and behind the step's first HEAD_READY a kernel on the
auxiliary stream traps the rank if a peer's hash differs (`_head_ready`); a rank
that planned where a peer did not traps at that barrier's timeout. Neither is on
the path of the step.

Host work. Short contexts leave the host little time per layer: there the step
is as long as the thread that runs the model needs to issue it. Every layer
therefore issues as few calls as it can, each on arguments that were prepared
before: every layer's chunk is checked once when the step is planned, and the
copies of all peers of a head go out together, one driver call per chunk on arrays
built once per step. A whole head is upconverted one launch per run of
adjacent peer slots; the new rows of every rank are upconverted into their slot
ranges by one launch (`kv_ce_rows`); the step's small tables are written with one
batch of stream memory operations; and the Triton kernels are compiled and loaded
when the state is built and launched by their handles (`cuda_direct`), so no step
pays a compilation or Triton's launch path. Where they cannot be launched that way
the KV prefetch is unavailable and every step takes the serial all-gather.

Selected rows only. A layer that reuses the top-k of the layer before it (a later
layer of the same index group, reading the same rows of the same top-k buffer)
attends only the rows that top-k selects, and the top-k is final before the layer
before it attends. Its head therefore carries, below `head_rows`, only the rows
some query of the chunk selects (`kv_ce_sel_union`). The group's first such layer
builds the union on the auxiliary stream: every rank marks its queries' rows in
peer-mapped memory, copies every peer's marks with the copy engines after a
SEL_MARKS barrier, ORs them, numbers each rank's union rows with one scan and
remaps the converted top-k onto the numbering; its later layers reuse all of it.
Each rank packs the raw rows of its union, in row order, to the front of the
buffer peers read and upconverts them into its own slot, then passes HEAD_READY;
its rows past `head_rows` move as in any head. Rows copied past a union are never
addressed by the remapped top-k. Peers read a rank's marks only between its
SEL_MARKS and HEAD_READY of the same head, and every rank passes that HEAD_READY
before it writes its marks again. These positions are part of the plan, so the
barriers pair up.

The host never waits for the union sizes `U`, which only exist on the device. A
compact head copies a predicted number of rows of every peer: the group's latest
sizes whose readback reached the host, plus `PRED_MARGIN_PERCENT` and
`PRED_SLACK_ROWS`, at most the head; the whole head while the group has no history.
A new request starts from the sizes of the last one. Right after the copies, on the
auxiliary stream, one launch of the kernel of `kv_ce_topup` reads `U` on the device
and upconverts every peer's rows `[0, U)`: those the copy engines brought from the
staging, and rows `[predicted, U)` of a peer whose prediction was short from the
buffer the copy engines read. It runs where the copies run, before this rank's
next HEAD_DONE, and issues no barrier, so what each rank copies is its own choice.
The sizes reach the host later through a non-blocking readback and become the next
prediction.

Enabled by `ParallelConfig.pcp_dcp_kv_prefetch`. Any step this cannot serve takes
the serial all-gather.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import struct
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.ops import kv_ce_rows, kv_ce_topup
from vllm.v1.attention.ops.cp_common import (
    direct_cp_enabled,
    scheduled_step,
    symm_mem,
    symm_mem_available,
)
from vllm.v1.attention.ops.cuda_direct import (
    CopyList,
    DirectKernel,
    Int32Table,
    KernelLaunch,
    OnStream,
    cuda_driver,
)
from vllm.v1.attention.ops.kv_ce_sel_union import (
    PACK_ROWS,
    GroupState,
    SelUnion,
    sel_digest,
    sel_union_positions,
    topk_use,
)

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

# A rank that never reaches a barrier traps its peers instead of hanging them.
BARRIER_TIMEOUT_MS = 60_000
CHANNEL_HEAD_READY = 1
CHANNEL_HEAD_DONE = 2
CHANNEL_SEL_MARKS = 3
# A predicted union pull copies the last sizes plus this margin and these rows.
PRED_MARGIN_PERCENT = 10.0
PRED_SLACK_ROWS = 64
# Bytes of one fp8_ds_mla cache row: 512 FP8 values, 4 FP32 scales, 64 BF16 rope.
KV_ENTRY_BYTES = 656
# Rows per block when an FP8 staging buffer is read as a paged cache.
STAGE_BLOCK = 64
# A head's copies go out in chunks of at most this many bytes; see "Copies in
# chunks" in the module doc.
COPY_CHUNK_BYTES = 16 << 20
# Plan words of a rank, in peer-mapped memory. Consecutive plans alternate: a peer
# may compare a step's word while this rank already writes the next step's.
PLAN_SLOTS = 2


def plan_hash(digest: list[int]) -> tuple[int, int]:
    """The halves of a 64-bit hash of a plan's digest: the plan word the ranks
    compare on the device."""
    packed = struct.pack(f"<{len(digest)}q", *digest)
    low, high = struct.unpack("<II", hashlib.blake2b(packed, digest_size=8).digest())
    return low, high


def tables_readable(chunk: Any) -> bool:
    """Whether the union's kernels can read a chunk's block table and workspace
    start as one request's."""
    table, starts = chunk.block_table, chunk.workspace_starts
    return (
        table.dim() == 2
        and table.shape[0] == 1
        and table.dtype == starts.dtype == torch.int32
    )


def predict_rows(last: list[int] | None, head_rows: int, world_size: int) -> list[int]:
    """Rows of every slot a predicted union pull copies: the union sizes `last` of
    the slots in the last step whose readback reached the host, plus
    `PRED_MARGIN_PERCENT` and `PRED_SLACK_ROWS`, at most the head. Without them,
    the whole head, which needs no top-up."""
    if last is None:
        return [head_rows] * world_size
    grown = 100.0 + PRED_MARGIN_PERCENT
    return [
        min(head_rows, max(0, math.ceil(int(u) * grown / 100.0) + PRED_SLACK_ROWS))
        for u in last
    ]


def chunk_copies(
    copies: list[tuple[int, int, int]], chunk_bytes: int, row_bytes: int
) -> list[list[tuple[int, int, int]]]:
    """Split `(destination, source, bytes)` copies into chunks of at most
    `chunk_bytes` (at least one row), in order, cutting copies only at rows of
    `row_bytes`. A chunk may hold the end of one copy and the start of the next."""
    step = max(chunk_bytes // row_bytes, 1) * row_bytes
    chunks: list[list[tuple[int, int, int]]] = []
    chunk: list[tuple[int, int, int]] = []
    room = step
    for destination, source, size in copies:
        offset = 0
        while offset < size:
            take = min(size - offset, room)
            chunk.append((destination + offset, source + offset, take))
            offset += take
            room -= take
            if room == 0:
                chunks.append(chunk)
                chunk, room = [], step
    if chunk:
        chunks.append(chunk)
    return chunks


def head_chunks(copies: list[tuple[int, int, int]], row_bytes: int) -> list[CopyList]:
    """A head's `(destination, source, bytes)` copies as the chunks it issues: of
    at most `COPY_CHUNK_BYTES`, cut at rows of `row_bytes` (`chunk_copies`)."""
    chunks = chunk_copies(copies, COPY_CHUNK_BYTES, row_bytes)
    return [CopyList(chunk) for chunk in chunks]


def dcp_min_shard_rows(seq_len: int, dcp_size: int, interleave: int) -> int:
    """Rows of a `seq_len` context on the DCP rank holding the fewest of them."""
    base = seq_len // interleave // dcp_size * interleave
    remainder = seq_len - base * dcp_size
    return base + min(max(remainder - (dcp_size - 1) * interleave, 0), interleave)


def dcp_shard_rows(seq_len: int, dcp_size: int, rank: int, interleave: int) -> int:
    """Rows of a `seq_len` context on DCP rank `rank`."""
    base = seq_len // interleave // dcp_size * interleave
    remainder = seq_len - base * dcp_size
    return base + min(max(remainder - rank * interleave, 0), interleave)


def shard_lens(
    context_start: int, seq_end: int, dcp_size: int, interleave: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Every DCP rank's shard rows before and after a step that computes the
    request's tokens `[context_start, seq_end)`."""
    ranks = range(dcp_size)
    old = tuple(dcp_shard_rows(context_start, dcp_size, r, interleave) for r in ranks)
    new = tuple(dcp_shard_rows(seq_end, dcp_size, r, interleave) for r in ranks)
    return old, new


@dataclass(frozen=True)
class KVCEOverlapPlan:
    """How one step splits every layer's shard, identical on all DCP ranks."""

    shard_rows: int
    head_rows: int
    context_start: int
    # Every rank's shard rows before and after the chunk, whose new rows are
    # staged locally.
    old_lens: tuple[int, ...] = ()
    new_lens: tuple[int, ...] = ()
    seq_end: int = 0
    # Positions whose head carries only the selected rows, and those of them that
    # build their index group's union.
    sel_positions: frozenset[int] = frozenset()
    sel_builds: frozenset[int] = frozenset()

    def digest(self) -> list[int]:
        """Everything of the plan a rank acts on, as integers: the DCP ranks
        compare a hash of it (`plan_hash`)."""
        return [
            1,
            self.shard_rows,
            self.head_rows,
            self.context_start,
            self.seq_end,
            *sel_digest(self.sel_positions, self.sel_builds),
        ]

    def serves(self, chunk: Any) -> bool:
        """Whether a chunk has the layout this plan split. Both values belong to
        the whole request, so every DCP rank gets the same answer."""
        return (
            int(chunk.chunk_tot_seqlen) == self.shard_rows
            and chunk.context_start == self.context_start
        )


def build_kv_ce_overlap_plan(
    *,
    num_chunks: int,
    num_entries: int,
    num_decode_tokens: int,
    shard_rows: int,
    context_start: int | None,
    dcp_size: int,
    interleave: int,
    shard_capacity: int,
    gathered_capacity: int,
) -> tuple[KVCEOverlapPlan | None, str]:
    """Return the step's plan, or None and why the step stays serial."""
    if num_chunks != 1 or num_entries != 1:
        return None, "multiple_chunks_or_requests"
    if num_decode_tokens != 0:
        return None, "decode_rows"
    if context_start is None or context_start <= 0:
        return None, "no_context"
    head_rows = min(dcp_min_shard_rows(context_start, dcp_size, interleave), shard_rows)
    if head_rows <= 0:
        return None, "no_context"
    if shard_rows > shard_capacity:
        return None, "shard_capacity"
    if 2 * dcp_size * shard_rows > gathered_capacity:
        return None, "two_layers_do_not_fit"
    return KVCEOverlapPlan(shard_rows, head_rows, context_start), "planned"


class LayerOrder:
    """The order layers attend in, learned from one whole forward.

    Learning starts only at a forward's first layer, so position 0 is always the
    first layer of a forward and the last learned layer never has a next one. A
    forward that leaves the order, or stops before its end, is serial from there
    and the next forward learns the order again.
    """

    def __init__(self) -> None:
        self.names: list[str] = []
        self.learning = False
        self.frozen = False
        self.in_order = False
        self.cursor = 0

    def advance(self, name: str, new_forward: bool) -> int | None:
        """Return the layer's position in its forward, None when it is not known."""
        if new_forward:
            if self.learning and self.names:
                self.frozen = True
            elif self.in_order and self.cursor != len(self.names):
                self.frozen = False
            self.learning = not self.frozen
            if self.learning:
                self.names = []
            self.in_order = self.frozen
            self.cursor = 0
        if self.learning:
            if name in self.names:
                # A layer seen twice in one forward gives no order to follow.
                self.learning = False
                self.names = []
            else:
                self.names.append(name)
            return None
        if (
            not self.in_order
            or self.cursor >= len(self.names)
            or name != self.names[self.cursor]
        ):
            self.in_order = self.frozen = False
            return None
        position = self.cursor
        self.cursor += 1
        return position


@dataclass
class _Pending:
    """The layer whose head rows the auxiliary stream is gathering."""

    name: str
    position: int
    # The layer's new rows are in its region.
    staged: bool = False
    # The head carries only the rows of its index group's union.
    compact: bool = False


@dataclass
class _PredWatch:
    """A predicted union pull whose sizes the host has not seen yet."""

    state: GroupState
    # The build whose readback the pull belongs to (`UnionSizes.builds`).
    build: int


Span = tuple[int, int, "torch.Tensor | None"]


@dataclass
class _WholeHead:
    """A whole head's calls for one step: the same for every layer but for its
    cache and its region."""

    # This rank's rows: where they go in the buffer peers read, and in its slot
    # of either region; each with the tensor holding its first row.
    wire: list[tuple[torch.Tensor, torch.Tensor | None]]
    slot: tuple[list[torch.Tensor], list[torch.Tensor]]
    # Every peer's rows, in chunks (`head_chunks`).
    chunks: list[CopyList]
    # The whole-slot upconverts into either region.
    runs: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]


@dataclass
class _CompactCopies:
    """A compact head's copies for one set of row counts."""

    chunks: list[CopyList]
    # Every peer's whole head is copied: its slots are upconverted whole.
    whole: bool
    # Per peer in pull order: its staging part, its slot and its spans.
    spans: list[tuple[int, int, list[Span]]]
    # The destination rows of the spans past `head_rows` in either region, made
    # on first use; the packed rows go through the receive kernel.
    views: list[list[torch.Tensor] | None]


class _Step:
    """What the layers of one step share, made once when its plan is taken."""

    __slots__ = (
        "plan",
        "chunks",
        "regions",
        "slot",
        "checked",
        "rest",
        "head",
        "compact",
        "kept",
        "tables",
    )

    def __init__(
        self,
        plan: KVCEOverlapPlan,
        chunks: list[Any],
        regions: tuple[int, int],
        slot: int = 0,
    ) -> None:
        self.plan = plan
        # Every position's chunk, checked when planned.
        self.chunks = chunks
        # The address of either region's first row.
        self.regions = regions
        # The plan word of the step, and whether the ranks' words were compared.
        self.slot = slot
        self.checked = False
        self.rest: list[list[Span]] | None = None
        self.head: _WholeHead | None = None
        self.compact: dict[tuple[int, ...], _CompactCopies] = {}
        # Chunks whose tensors the auxiliary stream already keeps alive.
        self.kept: set[int] = set()
        # Per chunk: the addresses of its block table and workspace start.
        self.tables: dict[int, tuple[int, int]] = {}


@dataclass
class _LayerProgram:
    """What one layer's heads reuse in every step."""

    kv_cache: torch.Tensor
    group_id: int
    # `_pack_kernel` bound to the layer's cache; None where it cannot read it.
    pack: KernelLaunch | None


class KVCEOverlap:
    """Buffers, auxiliary stream and per-step state of the overlapped KV gather."""

    def __init__(
        self,
        group: GroupCoordinator,
        device: torch.device,
        shard_capacity: int,
        gathered_capacity: int,
        row_dim: int,
        block_size: int | None = None,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        self.device_group = group.device_group
        self.cpu_group = group.cpu_group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.device = torch.device(device)
        self.shard_capacity = shard_capacity
        self.gathered_capacity = gathered_capacity
        self.row_dim = row_dim
        # Set while peers may still read this rank's head rows; see the module doc.
        self.head_done_owed = False
        # The PCP gather order and the step's staging slots of the new rows.
        self.interleave = 1
        self.restore_idx: torch.Tensor | None = None
        self.plan_seq = 0
        self.stage_key: tuple[int, int] | None = None
        self.stage_slots: torch.Tensor | None = None
        self.stage_slot_buffer: torch.Tensor | None = None
        self.stage_cache: torch.Tensor | None = None
        self.stage_blocks = 0
        self.stage_launch: KernelLaunch | None = None
        self.slots_launch: KernelLaunch | None = None
        self.receive_launch: KernelLaunch | None = None

        self.order = LayerOrder()
        self.plan: KVCEOverlapPlan | None = None
        self.step: _Step | None = None
        self.pending: _Pending | None = None
        self.position: int | None = None
        self.forward: weakref.ref | None = None
        self.layers: dict[str, Any] = {}
        self.programs: dict[str, _LayerProgram] = {}
        # Layers whose cache the union's pack kernel cannot read.
        self.unpackable: set[str] = set()
        # The union pull: its buffers, the remapped top-k of the layer `join`
        # served, each group's history (by the position that builds it), and the
        # predicted pulls whose sizes the host has not seen yet.
        self.sel: SelUnion | None = None
        self.sel_requested = False
        # Plans taken so far, and the kernel that compares the ranks' plan words.
        self.plan_count = 0
        self.plan_check: KernelLaunch | None = None
        self.joined_topk: tuple[torch.Tensor, torch.Tensor] | None = None
        self.pred_last: dict[int, list[int]] = {}
        self.pred_watch: list[_PredWatch] = []

        self.ready = self._allocate()
        if self.ready and block_size and max_num_batched_tokens:
            # The staging of a full batch, so no step allocates it.
            per_rank = -(-max_num_batched_tokens // self.world_size) + block_size
            self._ensure_stage_cache(
                self.world_size * per_rank, block_size, KV_ENTRY_BYTES, torch.uint8
            )
            torch.accelerator.synchronize()

    def _agree(self, values: list[int]) -> bool:
        """Whether every DCP rank passed the same values. Collective and blocking:
        only for what is allocated while the state is built."""
        local = torch.tensor(values, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.cpu_group)
        return all(torch.equal(other, local) for other in gathered)

    def _allocate(self) -> bool:
        shape = (self.shard_capacity, self.row_dim)
        try:
            # Peers read the packed cache rows; the shard only serves serial layers.
            self.shard = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            self.gathered = torch.empty(
                (self.gathered_capacity, self.row_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            # The step's int32 tables, written with stream memory operations:
            # `head_rows`, every slot's first row, every rank's new shard rows
            # (where its padding rows start), and for the staging of every rank's
            # new rows their first scratch entry, first region row and count.
            world = self.world_size
            self.tables = torch.zeros(
                1 + 5 * world, dtype=torch.int32, device=self.device
            )
            self.seq_starts = self.tables[:1]
            self.slot_starts = self.tables[1 : 1 + world]
            self.peer_new = self.tables[1 + world : 1 + 2 * world]
            self.stage_src = self.tables[1 + 2 * world : 1 + 3 * world]
            self.stage_dst = self.tables[1 + 3 * world : 1 + 4 * world]
            self.stage_count = self.tables[1 + 4 * world :]
            self._allocate_wire()
            self.driver = cuda_driver()
            # This rank's plan words, which the peers read. A step's word goes
            # out with its tables in one batch: one writer per word.
            self.plan_words = symm_mem.empty(
                PLAN_SLOTS, dtype=torch.int64, device=self.device
            )
            words = self.plan_words.data_ptr()
            self.table_writers = [
                Int32Table(
                    self.driver,
                    self.tables,
                    before=(words + 8 * slot, words + 8 * slot + 4),
                )
                for slot in range(PLAN_SLOTS)
            ]
            self.table_writer = self.table_writers[0]
            self.fence = self.driver.event()
            self.kernels = self._compile_kernels()
            self.side = torch.cuda.Stream(device=self.device)
            self.side_handle = self.side.cuda_stream
            allocated = 1
        except (RuntimeError, ValueError, OSError, AttributeError) as error:
            logger.warning("KV copy-engine overlap is unavailable: %s", error)
            allocated = 0
        # A rendezvous with a rank that failed to allocate would never return.
        all_allocated = self._agree([allocated])
        if not allocated or not all_allocated:
            self.shard = self.gathered = None  # type: ignore[assignment]
            return False
        peer_read = self.own_wire
        peer_read.zero_()
        torch.accelerator.synchronize()
        self.handle = symm_mem.rendezvous(peer_read, self.device_group.group_name)
        self.handle.barrier()
        self.remote = [
            self.handle.get_buffer(peer, list(peer_read.shape), peer_read.dtype, 0)
            for peer in range(self.world_size)
        ]
        self.remote_ptrs = [buffer.data_ptr() for buffer in self.remote]
        self.plan_words.zero_()
        torch.accelerator.synchronize()
        self.plan_handle = symm_mem.rendezvous(
            self.plan_words, self.device_group.group_name
        )
        self.plan_handle.barrier()
        self.plan_remote = [
            self.plan_handle.get_buffer(peer, [PLAN_SLOTS], torch.int64, 0)
            for peer in range(self.world_size)
        ]
        # Where every rank's plan words lie from this rank's, in bytes.
        self.plan_deltas = torch.tensor(
            [
                words.data_ptr() - self.plan_words.data_ptr()
                for words in self.plan_remote
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self.gathered_ptr = self.gathered.data_ptr()
        self.row_bytes = self.row_dim * self.gathered.element_size()
        self.pull_order = self._pull_order()
        self._load_kernels()
        return True

    def _compile_kernels(self) -> tuple[DirectKernel, DirectKernel, DirectKernel]:
        """The staging and plan-check kernels, compiled and loaded for direct
        launches; raises where they cannot be launched that way, and the KV
        prefetch is then unavailable."""
        slots = kv_ce_rows.stage_slots_kernel()
        rows = kv_ce_rows.stage_rows_kernel()
        check = kv_ce_rows.plan_check_kernel(self.world_size)
        if slots is None or rows is None or check is None:
            raise RuntimeError("its Triton kernels cannot be launched directly")
        return slots, rows, check

    def _load_kernels(self) -> None:
        """Bind the staging and plan-check kernels and run every device operation
        of a head and of the staging once on the auxiliary stream, with nothing
        to do, so the first step with history pays no kernel load or first
        launch."""
        slots, rows, check = self.kernels
        handle = self.side_handle
        # A launch of zero tokens reads and writes nothing.
        anchor = self.gathered_ptr
        self.slots_launch = slots.bind(
            restore_ptr=anchor,
            slots_ptr=anchor,
            interleave=1,
            dcp_size=1,
            per_rank=1,
        )
        self.slots_launch(handle, 1)
        self.stage_launch = rows.bind(
            cache_ptr=self.stage_ptr,
            sources_ptr=self.stage_src.data_ptr(),
            targets_ptr=self.stage_dst.data_ptr(),
            counts_ptr=self.stage_count.data_ptr(),
            region_ptr=self.gathered_ptr,
        )
        # The counts are zero.
        self.stage_launch(handle, 1, self.world_size)
        block = torch.zeros(
            (1, STAGE_BLOCK, KV_ENTRY_BYTES), dtype=torch.uint8, device=self.device
        )
        table = torch.zeros((1, 1), dtype=torch.int32, device=self.device)
        bf16 = {"dtype": torch.bfloat16, "device": self.device}
        # The auxiliary stream reads what the current stream just filled.
        torch.accelerator.synchronize()
        self.plan_check = check.bind(
            words_ptr=self.plan_words.data_ptr(),
            deltas_ptr=self.plan_deltas.data_ptr(),
            slot=0,
        )
        # Every rank's words are still zero.
        self.plan_check(handle, 1)
        with OnStream(self.side):
            ops.cp_gather_cache(
                block,
                self.own_wire[:STAGE_BLOCK],
                table,
                self.wire_bounds,
                1,
                self.wire_zero,
            )
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                block, self.gathered[:STAGE_BLOCK], table, self.wire_zero, 1
            )
            ops.concat_and_cache_mla(
                torch.zeros((1, kv_ce_rows.FP8_NOPE), **bf16),
                torch.zeros((1, kv_ce_rows.ROPE_DIM), **bf16),
                block,
                torch.zeros(1, dtype=torch.int64, device=self.device),
                kv_cache_dtype="fp8_ds_mla",
                scale=torch.ones(1, dtype=torch.float32, device=self.device),
            )
        torch.accelerator.synchronize()

    def enable_union(self, topk_shape: tuple[int, int]) -> None:
        """Allocate the union pull's buffers for a top-k buffer of `topk_shape`,
        once. Called while the first layer that reuses an index group's top-k is
        built; collective over the DCP group, whose ranks build the same layers in
        the same order. If any rank cannot have them, every layer prefetches its
        whole head."""
        if self.sel_requested:
            return
        self.sel_requested = True
        sel, receive, ok = None, None, 1
        try:
            sel = SelUnion(self, *topk_shape)
            sel.compile_kernels()
            receive = kv_ce_topup.receive_kernel()
            if receive is None:
                raise RuntimeError("the receive kernel cannot be launched directly")
        except (RuntimeError, ValueError, AssertionError) as error:
            logger.warning("The KV prefetch copies whole heads: %s", error)
            ok = 0
        # The rendezvous below is collective: every rank or none.
        if not self._agree([ok]) or sel is None or receive is None:
            return
        sel.connect()
        sel.load_kernels(self.side)
        self.sel = sel
        self._receive_setup(receive)

    def _receive_setup(self, kernel: DirectKernel) -> None:
        """What the receive kernel of a compact head needs: the peers in pull
        order, the staging part of each, where each peer's buffer lies from the
        staging, and the kernel bound and launched once (with zero counts, which
        moves nothing), so no step pays its first launch."""
        assert self.sel is not None
        order = self.pull_order
        i32, device = torch.int32, self.device
        self.receive_peers = torch.tensor(order, dtype=i32, device=device)
        self.receive_parts = torch.tensor(
            [self.wire_peers.index(p) for p in order], dtype=i32, device=device
        )
        self.receive_deltas = kv_ce_topup.peer_deltas(
            self.wire_stage, [self.remote[p] for p in order]
        )
        zero_deltas = torch.zeros_like(self.receive_deltas)
        zero_ends = torch.zeros(self.world_size + 1, dtype=i32, device=device)
        # The launch below reads them on the auxiliary stream.
        torch.accelerator.synchronize()
        launch = kernel.bind(
            anchor_ptr=self.stage_ptr,
            deltas_ptr=zero_deltas.data_ptr(),
            ends_ptr=zero_ends.data_ptr(),
            slots_ptr=self.receive_peers.data_ptr(),
            parts_ptr=self.receive_parts.data_ptr(),
            stage_rows=self.wire_rows,
        )
        kv_ce_topup.launch_receive(
            launch, self.side_handle, [0] * len(order), 1, self.gathered_ptr, 1
        )
        torch.accelerator.synchronize()
        launch.set("deltas_ptr", self.receive_deltas.data_ptr())
        launch.set("ends_ptr", self.sel.ends.data_ptr())
        self.receive_launch = launch

    def _pull_order(self) -> list[int]:
        """The peers in staggered order: no two ranks read one peer at once."""
        world, rank = self.world_size, self.rank
        return [(rank + step) % world for step in range(1, world)]

    def _allocate_wire(self) -> None:
        """The wire buffers: this rank's raw cache rows that peers read (symmetric),
        and the rows received from each peer, read back as a paged cache. A shard
        holds at most half a gathered region's rows: two regions must fit."""
        rows = -(-self.gathered_capacity // (2 * self.world_size))
        rows = -(-rows // STAGE_BLOCK) * STAGE_BLOCK
        self.wire_rows = rows
        self.wire_peers = [p for p in range(self.world_size) if p != self.rank]
        uint8 = {"dtype": torch.uint8, "device": self.device}
        self.own_wire = symm_mem.empty((rows, KV_ENTRY_BYTES), **uint8)
        self.wire_stage = torch.empty(
            (len(self.wire_peers) * rows, KV_ENTRY_BYTES), **uint8
        )
        blocks = rows // STAGE_BLOCK
        self.wire_cache = self.wire_stage.view(-1, STAGE_BLOCK, KV_ENTRY_BYTES)
        # Peer p's rows are part i of the staging, i its index in wire_peers.
        table = torch.arange(len(self.wire_peers), dtype=torch.int32)[:, None] * blocks
        self.wire_table = (table + torch.arange(blocks, dtype=torch.int32)).to(
            self.device
        )
        self.wire_index = {peer: index for index, peer in enumerate(self.wire_peers)}
        self.wire_part_table = [
            self.wire_table[index : index + 1] for index in range(len(self.wire_peers))
        ]
        self.stage_ptr = self.wire_stage.data_ptr()
        self.wire_zero = torch.zeros(1, dtype=torch.int32, device=self.device)
        # cp_gather_cache bounds: copy as many rows as the destination holds.
        self.wire_bounds = torch.tensor(
            [0, torch.iinfo(torch.int32).max], dtype=torch.int32, device=self.device
        )

    def shard_view(self, shard_rows: int) -> torch.Tensor:
        return self.shard[:shard_rows]

    def _wait(self, waiter: torch.cuda.Stream, signaler: torch.cuda.Stream) -> None:
        """`waiter.wait_stream(signaler)` with one reused event: two driver calls."""
        self.driver.record(self.fence, signaler.cuda_stream)
        self.driver.wait_event(waiter.cuda_stream, self.fence)

    def _region(self, position: int, shard_rows: int) -> torch.Tensor:
        """Adjacent layers alternate so one can be read while the next arrives."""
        rows = self.world_size * shard_rows
        start = (position % 2) * rows
        return self.gathered[start : start + rows]

    def serial_region(self, chunk: Any) -> torch.Tensor:
        """The gathered view of a layer that takes the serial all-gather."""
        shard_rows = int(chunk.chunk_tot_seqlen)
        # `join` dropped a plan that does not serve the chunk.
        if self.plan is None or self.position is None:
            return self.gathered[: self.world_size * shard_rows]
        return self._region(self.position, shard_rows)

    def _abandon(self) -> None:
        """Forget the step; the auxiliary stream must finish before reuse."""
        if self.pending is not None:
            self._wait(current_stream(), self.side)
            self.pending = None
        self.plan = self.step = None
        self.joined_topk = None
        if self.sel is not None:
            self.sel.reset()

    def begin_layer(
        self,
        layer: Any,
        forward: object,
        prefill_of: Callable[[str], Any],
        num_decode_tokens: int,
        interleave: int,
        metadata_of: Callable[[str], Any],
    ) -> None:
        """Track the layer's position in `forward`; its first layer plans it.

        A plan and a prefetch belong to the forward that made them: both are
        dropped when another forward starts, so no layer joins rows that were
        laid out for another step's shard length. `metadata_of` gives another
        layer's whole metadata, for the union positions.
        """
        name = layer.layer_name
        self.layers[name] = layer
        if self.sel is not None and name not in self.programs:
            # Bound when the layer is first seen (a forward that learns the
            # order), so no head pays for it.
            self._program(name)
        seen = self.forward() if self.forward is not None else None
        new_forward = seen is not forward
        if new_forward:
            self._abandon()
            self.forward = weakref.ref(forward)
        self.position = self.order.advance(name, new_forward)
        if self.position is None:
            self._abandon()
        elif self.position == 0:
            self._derive_plan(
                name, prefill_of, num_decode_tokens, interleave, metadata_of
            )

    def _derive_plan(
        self,
        name: str,
        prefill_of: Callable[[str], Any],
        num_decode_tokens: int,
        interleave: int,
        metadata_of: Callable[[str], Any],
    ) -> None:
        """Derive the step's plan and take it, or leave the step serial.

        Whether the step is planned and every number of the plan follow from the
        global schedule (`PCPManager.split_prefill_request`), the configuration
        and the layers, which are the same on every rank: no rank waits for
        another. This rank's own metadata is only checked against them. A
        mismatch means this rank would issue barriers its peers do not, so it is
        an error and not a reason to gather serially.
        """
        from vllm.v1.worker.gpu.pcp_manager import active_pcp_manager

        manager = active_pcp_manager()
        if manager is None or not scheduled_step():
            # A capture or warmup forward, on every rank alike: its dummy batch
            # is never planned.
            return
        request = manager.split_prefill_request
        prefill = None if request is None else prefill_of(name)
        if request is None or prefill is None:
            return
        start, end = request
        world = self.world_size
        shard_rows = dcp_shard_rows(end, world, 0, interleave)
        chunk = prefill.chunks[0]
        local = (
            len(prefill.chunks),
            len(chunk.block_table),
            num_decode_tokens,
            int(chunk.chunk_tot_seqlen),
            chunk.context_start,
        )
        if local != (1, 1, 0, shard_rows, start):
            raise RuntimeError(
                "PCP + DCP KV prefetch: this rank's prefill metadata (chunks, "
                f"entries, decode tokens, shard rows, context start) is {local}, "
                f"but the scheduled request [{start}, {end}) implies "
                f"{(1, 1, 0, shard_rows, start)}."
            )
        plan, _ = build_kv_ce_overlap_plan(
            num_chunks=1,
            num_entries=1,
            num_decode_tokens=0,
            shard_rows=shard_rows,
            context_start=start,
            dcp_size=world,
            interleave=interleave,
            shard_capacity=self.shard_capacity,
            gathered_capacity=self.gathered_capacity,
        )
        if plan is None or not self._wire_serves(name, plan):
            return
        union = self.sel is not None
        if union and self.unpackable:
            return
        names = self.order.names
        chunks: list[Any] = []
        metadata: list[Any] = []
        # Layers that share a metadata object share the answer.
        served: dict[int, Any] = {}
        for other in names:
            layer_metadata = metadata_of(other)
            key = id(layer_metadata)
            if key not in served:
                other_prefill = prefill_of(other)
                served[key] = (
                    other_prefill.chunks[0]
                    if self._same_plan(other_prefill, plan)
                    and (not union or tables_readable(other_prefill.chunks[0]))
                    else None
                )
            if served[key] is None:
                return
            chunks.append(served[key])
            metadata.append(layer_metadata)
        restore = manager.hidden_restore_idx
        if restore is None or restore.shape[0] != end - start:
            raise RuntimeError(
                "PCP + DCP KV prefetch: the PCP layout restores "
                f"{None if restore is None else restore.shape[0]} rows of a "
                f"request scheduled for {end - start}."
            )
        self.interleave = interleave
        self.restore_idx = restore
        old_lens, new_lens = shard_lens(start, end, world, interleave)
        plan = dataclasses.replace(
            plan, old_lens=old_lens, new_lens=new_lens, seq_end=end
        )
        if self.sel is not None:
            positions, builds = self._sel_positions(names, metadata)
            plan = dataclasses.replace(plan, sel_positions=positions, sel_builds=builds)
        self._take_plan(plan, chunks)

    def _sel_positions(
        self, names: list[str], metadata: list[Any]
    ) -> tuple[frozenset[int], frozenset[int]]:
        """`sel_union_positions` of the step's layers."""
        return sel_union_positions(
            [
                topk_use(self.layers.get(other), layer_metadata)
                for other, layer_metadata in zip(names, metadata)
            ]
        )

    def _take_plan(self, plan: KVCEOverlapPlan, chunks: list[Any]) -> None:
        """Make `plan` the step's: what its layers share, and its tables, written
        on the current stream together with the plan word the ranks compare."""
        self.plan = plan
        world, shard_rows = self.world_size, plan.shard_rows
        region_bytes = world * shard_rows * self.row_bytes
        self.plan_count += 1
        slot = self.plan_count % PLAN_SLOTS
        self.step = _Step(
            plan,
            chunks,
            (self.gathered_ptr, self.gathered_ptr + region_bytes),
            slot,
        )
        values = [
            *plan_hash(plan.digest()),
            plan.head_rows,
            *(p * shard_rows for p in range(world)),
            *plan.new_lens,
        ]
        self.plan_seq += 1
        self.table_writer = self.table_writers[slot]
        self.table_writer.write(current_stream().cuda_stream, -2, values)

    def _head_ready(self) -> None:
        """Pass HEAD_READY on the current (auxiliary) stream. Behind a step's
        first one every rank has written the step's plan word (each on its main
        stream, before its head): a kernel compares them and traps this rank if
        they differ."""
        self.handle.barrier(channel=CHANNEL_HEAD_READY, timeout_ms=BARRIER_TIMEOUT_MS)
        step = self.step
        if step is None or step.checked:
            return
        step.checked = True
        assert self.plan_check is not None
        self.plan_check.set("slot", step.slot)
        self.plan_check(current_stream().cuda_stream, 1)

    def _wire_serves(self, name: str, plan: KVCEOverlapPlan) -> bool:
        """Raw `fp8_ds_mla` rows travel: the step stays serial when its cache rows
        are not 656 contiguous bytes or its shard exceeds the staging."""
        cache = self.layers[name].kv_cache
        return (
            cache.dtype == torch.uint8
            and cache.dim() == 3
            and cache.shape[2] == KV_ENTRY_BYTES
            and cache.stride(1) == KV_ENTRY_BYTES
            and plan.shard_rows <= self.wire_rows
        )

    @staticmethod
    def _same_plan(prefill: Any, plan: KVCEOverlapPlan) -> bool:
        return (
            prefill is not None
            and len(prefill.chunks) == 1
            and plan.serves(prefill.chunks[0])
        )

    def _pay_head_done(self, where: str) -> None:
        """Issue an owed HEAD_DONE on the current stream before this rank packs
        rows again (see the module doc). `where` names the stream."""
        if not self.head_done_owed or not scheduled_step():
            # A capture or warmup forward issues no barrier; the debt stays for
            # the next scheduled step, and nothing is packed before it.
            return
        if where == "main":
            # The barrier must follow this rank's own head copies.
            self._wait(current_stream(), self.side)
        self.handle.barrier(channel=CHANNEL_HEAD_DONE, timeout_ms=BARRIER_TIMEOUT_MS)
        self.head_done_owed = False

    def _serial(self) -> None:
        """A layer that gathers serially settles the owed barrier first."""
        self._pay_head_done("main")

    def start_next(
        self,
        prefill_of: Callable[[str], Any],
        topk: torch.Tensor,
        convert: Callable[[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Start the next layer's head rows; call right before attention launches.

        `topk` is this layer's chunk of the shared top-k buffer, before conversion;
        `convert(name, topk)` converts it as layer `name` would. A next layer that
        reuses this top-k prefetches only the rows it selects.

        The next layer is always in this forward: the learned order ends at a
        forward's last layer, and every layer's chunk was checked against the
        plan when the forward was planned.
        """
        if self.pred_watch:
            self._pred_poll()
        step, position = self.step, self.position
        if step is None or position is None:
            return
        position += 1
        if position >= len(step.chunks):
            # The forward's last layer. Nothing of the step is used again, and it
            # holds the step's chunks: drop it here, not when the next forward
            # starts.
            self.step = None
            return
        plan = step.plan
        name = self.order.names[position]
        chunk = step.chunks[position]
        compact = position in plan.sel_positions
        converted = None
        if position in plan.sel_builds:
            # The next layer's own conversion of the top-k, which the shared
            # buffer already holds in full: on this stream, whose allocator the
            # auxiliary stream then never needs.
            converted = convert(name, topk)
        main = current_stream()
        driver = self.driver
        driver.record(self.fence, main.cuda_stream)
        driver.wait_event(self.side_handle, self.fence)
        if id(chunk) not in step.kept:
            # The step's metadata may be freed before the main stream joins.
            step.kept.add(id(chunk))
            chunk.block_table.record_stream(self.side)
            chunk.workspace_starts.record_stream(self.side)
        torch.cuda.set_stream(self.side)
        try:
            # Peers may still read the rows this head packs over.
            if self.head_done_owed:
                self.handle.barrier(
                    channel=CHANNEL_HEAD_DONE, timeout_ms=BARRIER_TIMEOUT_MS
                )
                self.head_done_owed = False
            if compact:
                self._compact_head(name, position, chunk, converted)
            else:
                self._fp8_head(self.layers[name].kv_cache, chunk, position, plan)
        finally:
            torch.cuda.set_stream(main)
        # No barrier follows the head copies; see the module doc.
        self.head_done_owed = True
        self.pending = _Pending(name, position, compact=compact)

    def _program(self, name: str) -> _LayerProgram:
        layer = self.layers[name]
        program = self.programs.get(name)
        if program is None or program.kv_cache is not layer.kv_cache:
            assert self.sel is not None
            program = self.programs[name] = _LayerProgram(
                layer.kv_cache,
                id(layer.impl.index_group),
                self.sel.pack_launch(layer.kv_cache, self.own_wire),
            )
            if program.pack is None:
                logger.warning_once(
                    "PCP + DCP KV prefetch: the union pack cannot read the KV "
                    "cache of %s; steps gather serially.",
                    name,
                )
                self.unpackable.add(name)
            else:
                self.unpackable.discard(name)
        return program

    def _chunk_tables(self, step: _Step, chunk: Any) -> tuple[int, int]:
        """The addresses of the chunk's block table and workspace start; the plan
        checked that the kernels can read them (`tables_readable`)."""
        key = id(chunk)
        tables = step.tables.get(key)
        if tables is None:
            tables = step.tables[key] = (
                chunk.block_table.data_ptr(),
                chunk.workspace_starts.data_ptr(),
            )
        return tables

    def _marks_barrier(self) -> None:
        self.handle.barrier(channel=CHANNEL_SEL_MARKS, timeout_ms=BARRIER_TIMEOUT_MS)

    def _compact_head(
        self,
        name: str,
        position: int,
        chunk: Any,
        converted: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        """On the current (auxiliary) stream: build (or reuse) the union, pack this
        rank's union rows and its rows past `head_rows`, pass HEAD_READY and copy
        the peers' rows: a predicted number of them, completed on the device. A
        layer that builds the union passes its top-k, converted on the main
        stream.

        Every rank takes this path at the same positions and builds at the same
        positions (both agreed in the plan), so the barriers pair up.
        """
        plan, sel, step = self.plan, self.sel, self.step
        assert plan is not None and sel is not None and step is not None
        program = self._program(name)
        group_id = program.group_id
        if position in plan.sel_builds:
            assert converted is not None
            topk, topk_length = converted
            # Converted on the main stream; the build reads it here, and the
            # caller drops it before this stream is done with it.
            topk.record_stream(self.side)
            topk_length.record_stream(self.side)
            sel.build(
                group_id,
                position,
                topk,
                topk_length,
                plan.shard_rows,
                plan.head_rows,
                self._marks_barrier,
                self.side,
            )
        state = sel.state
        assert state is not None and state.group_id == group_id
        shard_rows, head_rows = plan.shard_rows, plan.head_rows
        kv_cache = program.kv_cache
        # A plan is taken only while every layer's cache can be packed.
        launch = program.pack
        assert launch is not None
        tables = self._chunk_tables(step, chunk)
        slots, index = launch.slots, launch.index
        slots[index["block_table_ptr"]] = tables[0]
        slots[index["workspace_starts_ptr"]] = tables[1]
        slots[index["region_ptr"]] = (
            step.regions[position % 2] + self.rank * shard_rows * self.row_bytes
        )
        slots[index["slot_start"]] = self.rank * head_rows
        slots[index["head_rows"]] = head_rows
        launch(self.side_handle, -(-head_rows // PACK_ROWS))
        self._pack_rest(plan, kv_cache, chunk, position)
        self._head_ready()
        rows = predict_rows(
            self.pred_last.get(state.built_position), head_rows, self.world_size
        )
        # The receive kernel also takes what the prediction missed: the head is
        # complete in stream order and ends like any head.
        self._compact_copies(plan, position, rows)
        watch = self.pred_watch
        build = state.sizes.builds
        if not watch or watch[-1].state is not state or watch[-1].build != build:
            watch.append(_PredWatch(state, build))

    def _pred_poll(self) -> None:
        """Keep the union sizes whose readback landed as their group's history;
        never waits. A group's readback is reused by its next build, so sizes not
        seen by then are dropped."""
        left = []
        for watch in self.pred_watch:
            sizes = watch.state.sizes
            if sizes.builds != watch.build:
                continue
            exact = sizes.exact()
            if exact is None:
                left.append(watch)
            else:
                self.pred_last[watch.state.built_position] = exact
        self.pred_watch = left

    def _pack_rest(
        self,
        plan: KVCEOverlapPlan,
        kv_cache: torch.Tensor,
        chunk: Any,
        position: int,
    ) -> None:
        """On the current stream: this rank's rows of a compact head past
        `head_rows` (`_rest_spans`), raw into the buffer peers read and upconverted
        from the cache into its slot of the region of `position`."""
        me = self.rank
        spans = self._rest(plan)[me]
        if not spans:
            return
        region = self._region(position, plan.shard_rows)
        for begin, end, starts in spans:
            ops.cp_gather_cache(
                kv_cache,
                self.own_wire[begin:end],
                chunk.block_table,
                self.wire_bounds,
                1,
                starts,
            )
            start = me * plan.shard_rows
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                kv_cache,
                region[start + begin : start + end],
                chunk.block_table,
                chunk.workspace_starts,
                1,
                starts,
            )

    def _rest(self, plan: KVCEOverlapPlan) -> list[list[Span]]:
        """`_rest_spans` of every rank, computed once per step."""
        step = self.step
        if step is None or step.plan is not plan:
            return [self._rest_spans(plan, peer) for peer in range(self.world_size)]
        if step.rest is None:
            step.rest = [
                self._rest_spans(plan, peer) for peer in range(self.world_size)
            ]
        return step.rest

    def _rest_spans(self, plan: KVCEOverlapPlan, peer: int) -> list[Span]:
        """The rows of `peer`'s slot a compact head copies whole, as
        `(begin, end, offset tensor)`: its old rows past `head_rows` and its
        padding rows."""
        spans: list[Span] = []
        old, new, shard_rows = plan.old_lens[peer], plan.new_lens[peer], plan.shard_rows
        if old > plan.head_rows:
            spans.append((plan.head_rows, old, self.seq_starts))
        if new < shard_rows:
            spans.append((new, shard_rows, self.peer_new[peer : peer + 1]))
        return spans

    def _compact_program(
        self, plan: KVCEOverlapPlan, rows: list[int]
    ) -> _CompactCopies:
        """The copies of a compact head that takes the first `rows[p]` packed rows
        of every peer `p`, in staggered order, and its rows past `head_rows`."""
        entry, wire_rows = KV_ENTRY_BYTES, self.wire_rows
        rest = self._rest(plan)
        spans = []
        copies = []
        for peer in self.pull_order:
            index = self.wire_index[peer]
            peer_spans: list[Span] = []
            if rows[peer] > 0:
                peer_spans.append((0, rows[peer], None))
            peer_spans += rest[peer]
            spans.append((index, peer, peer_spans))
            copies += [
                (
                    self.stage_ptr + (index * wire_rows + begin) * entry,
                    self.remote_ptrs[peer] + begin * entry,
                    (end - begin) * entry,
                )
                for begin, end, _ in peer_spans
                if end > begin
            ]
        whole = all(rows[peer] >= plan.head_rows for peer in self.pull_order)
        return _CompactCopies(head_chunks(copies, entry), whole, spans, [None, None])

    def _compact_copies(
        self, plan: KVCEOverlapPlan, position: int, rows: list[int]
    ) -> None:
        """The copies of a compact head after HEAD_READY: the first `rows[p]` packed
        rows of every peer `p` (in staggered order) and its rows past `head_rows`,
        each upconverted into the peer's slot; the packed rows by one launch for
        all peers, which also takes the rows of a union larger than `rows[p]`
        (`_receive`). This rank's union rows were packed into its slot; its rows
        past `head_rows` were upconverted from its cache."""
        step = self.step
        key = tuple(rows)
        program = None if step is None else step.compact.get(key)
        if program is None:
            program = self._compact_program(plan, rows)
            if step is not None and step.plan is plan:
                step.compact[key] = program
        self._copy(program.chunks)
        shard_rows = plan.shard_rows
        if program.whole:
            self._upconvert_slots(self._region(position, shard_rows), shard_rows)
            return
        # The rows past `head_rows`, where a peer has any (`_rest_spans`).
        parity = position % 2
        views = program.views[parity]
        if views is None:
            region = self._region(position, shard_rows)
            views = program.views[parity] = [
                region[peer * shard_rows + begin : peer * shard_rows + end]
                for _, peer, spans in program.spans
                for begin, end, starts in spans
                if starts is not None
            ]
        at = 0
        upconvert = ops.cp_gather_and_upconvert_fp8_kv_cache
        for index, _, spans in program.spans:
            table = self.wire_part_table[index]
            for _, _, starts in spans:
                if starts is not None:
                    upconvert(
                        self.wire_cache, views[at], table, self.wire_zero, 1, starts
                    )
                    at += 1
        self._receive(plan, position, rows)

    def _receive(self, plan: KVCEOverlapPlan, position: int, rows: list[int]) -> None:
        """On the current (auxiliary) stream, after copy-engine copies of the first
        `rows[p]` packed rows of every peer `p` into the head of `position`: one
        launch of the kernel of `kv_ce_topup` upconverts rows `[0, U_p)` of every
        peer into its slot, from the staging below `rows[p]` and from the peer's
        peer-mapped buffer past it (also into the staging), with `U` read from
        the union's slot ends on the device."""
        launch, step = self.receive_launch, self.step
        assert launch is not None and step is not None
        kv_ce_topup.launch_receive(
            launch,
            current_stream().cuda_stream,
            [rows[peer] for peer in self.pull_order],
            plan.head_rows,
            step.regions[position % 2],
            plan.shard_rows,
        )

    def _copy(self, chunks: list[CopyList]) -> None:
        """Enqueue a head's copies on the current (auxiliary) stream, one call per
        chunk."""
        stream = current_stream().cuda_stream
        for chunk in chunks:
            self.driver.memcpy_list(chunk, stream)

    def _fp8_spans(self, plan: KVCEOverlapPlan) -> list[list[Span]]:
        """Per rank, the shard rows `(begin, end, source offset)` a whole head
        moves: its old rows and its padding rows. The offset tensor holds `begin`
        (None for 0)."""
        spans: list[list[Span]] = []
        for peer, (old, new) in enumerate(zip(plan.old_lens, plan.new_lens)):
            spans.append([(0, old, None)])
            if new < plan.shard_rows:
                spans[-1].append((new, plan.shard_rows, self.peer_new[peer : peer + 1]))
        return spans

    def _span_copies(self, spans: list[list[Span]]) -> list[tuple[int, int, int]]:
        """`(destination, source, bytes)` of every peer's spans: from the buffer
        peers read into the peer's part of the staging."""
        rows, entry = self.wire_rows, KV_ENTRY_BYTES
        return [
            (
                self.stage_ptr + (self.wire_index[peer] * rows + begin) * entry,
                self.remote_ptrs[peer] + begin * entry,
                (end - begin) * entry,
            )
            for peer in self.wire_peers
            for begin, end, _ in spans[peer]
            if end > begin
        ]

    def _slot_runs(self, region: torch.Tensor, shard_rows: int) -> list[tuple]:
        """The arguments that upconvert every peer's whole slot from its staging:
        one launch per run of adjacent slots (this rank's slot splits them)."""
        me = self.rank
        runs = []
        for first, end in ((0, me), (me + 1, self.world_size)):
            count = end - first
            if count <= 0:
                continue
            index = self.wire_index[first]
            runs.append(
                (
                    self.wire_cache,
                    region[first * shard_rows : end * shard_rows],
                    self.wire_table[index : index + count],
                    self.slot_starts[:count],
                    count,
                )
            )
        return runs

    def _upconvert_slots(self, region: torch.Tensor, shard_rows: int) -> None:
        """Upconvert every peer's whole slot, `[0, shard_rows)` of its staging:
        for a head with staged new rows, whose copies miss only the rows the
        staging writes after it."""
        for run in self._slot_runs(region, shard_rows):
            ops.cp_gather_and_upconvert_fp8_kv_cache(*run)

    def _whole_head(self, step: _Step) -> _WholeHead:
        """The step's whole head with staged new rows: each rank's old and padding
        rows."""
        head = step.head
        if head is None:
            plan = step.plan
            me, shard_rows = self.rank, plan.shard_rows
            spans = self._fp8_spans(plan)
            regions = [self._region(parity, shard_rows) for parity in (0, 1)]
            first = me * shard_rows
            own = spans[me]
            head = step.head = _WholeHead(
                [(self.own_wire[begin:end], starts) for begin, end, starts in own],
                (
                    [regions[0][first + b : first + e] for b, e, _ in own],
                    [regions[1][first + b : first + e] for b, e, _ in own],
                ),
                head_chunks(self._span_copies(spans), KV_ENTRY_BYTES),
                (
                    self._slot_runs(regions[0], shard_rows),
                    self._slot_runs(regions[1], shard_rows),
                ),
            )
        return head

    def _fp8_head(
        self,
        kv_cache: torch.Tensor,
        chunk: Any,
        position: int,
        plan: KVCEOverlapPlan,
    ) -> None:
        """The whole head: each rank's old and padding rows. The caller owes
        HEAD_DONE."""
        step = self.step
        assert step is not None and step.plan is plan
        head = self._whole_head(step)
        table, starts_at = chunk.block_table, chunk.workspace_starts
        for wire, starts in head.wire:
            ops.cp_gather_cache(kv_cache, wire, table, self.wire_bounds, 1, starts)
        upconvert = ops.cp_gather_and_upconvert_fp8_kv_cache
        parity = position % 2
        for slot, (_, starts) in zip(head.slot[parity], head.wire):
            upconvert(kv_cache, slot, table, starts_at, 1, starts)
        self._head_ready()
        self._copy(head.chunks)
        # Every peer's whole slot: the copies miss only the rows the staging
        # writes after this head.
        for run in head.runs[parity]:
            upconvert(*run)

    def stage_new_rows(
        self,
        name: str,
        forward: object,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        skip_scale: bool,
    ) -> None:
        """Put layer `name`'s new rows into its gathered region.

        `kv_c` and `k_pe` are the PCP-gathered rows and the arguments the cache
        write of `kv_cache` takes, ready on the current stream. The auxiliary stream
        quantizes them with the same kernel into a scratch block cache and upconverts
        every rank's share into that rank's slot range.
        """
        plan, pending = self.plan, self.pending
        if (
            plan is None
            or pending is None
            or pending.name != name
            or pending.staged
            or self.forward is None
            or self.forward() is not forward
            or not scheduled_step()
        ):
            return
        main = current_stream()
        slots = self._stage_slots(plan, kv_c.shape[0], kv_cache, main)
        if slots is None:
            return
        driver, side = self.driver, self.side_handle
        driver.record(self.fence, main.cuda_stream)
        driver.wait_event(side, self.fence)
        kv_c.record_stream(self.side)
        k_pe.record_stream(self.side)
        torch.cuda.set_stream(self.side)
        try:
            ops.concat_and_cache_mla(
                kv_c,
                k_pe.squeeze(1),
                self.stage_cache,
                slots,
                kv_cache_dtype=kv_cache_dtype,
                scale=k_scale,
                skip_scale=skip_scale,
            )
            # Every rank's rows, upconverted straight into its slot range.
            launch, step = self.stage_launch, self.step
            assert launch is not None and step is not None
            launch.set("region_ptr", step.regions[pending.position % 2])
            launch(side, self.stage_grid, self.world_size)
        finally:
            torch.cuda.set_stream(main)
        pending.staged = True

    def _stage_slots(
        self,
        plan: KVCEOverlapPlan,
        rows: int,
        kv_cache: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor | None:
        """Scratch slot of each gathered row for this step, rank-major: rank `p`'s
        new rows in token order from `p * per_rank`; -1 for PCP padding rows.
        Written on `stream` with the step's staging tables, once per step."""
        key = (self.plan_seq, rows)
        if self.stage_key == key:
            return self.stage_slots
        restore = self.restore_idx
        new_rows = plan.seq_end - plan.context_start
        if (
            restore is None
            or restore.shape[0] != new_rows
            or rows < new_rows
            or new_rows == 0
            or restore.dtype != torch.int64
            or not restore.is_contiguous()
        ):
            return None
        block_size, world = kv_cache.shape[1], self.world_size
        counts = [n - o for n, o in zip(plan.new_lens, plan.old_lens)]
        per_rank = -(-max(counts) // block_size) * block_size
        self._ensure_stage_cache(
            world * per_rank, block_size, kv_cache.shape[-1], kv_cache.dtype
        )
        buffer = self.stage_slot_buffer
        if buffer is None or buffer.shape[0] < rows:
            buffer = self.stage_slot_buffer = torch.empty(
                rows, dtype=torch.int64, device=self.device
            )
            buffer.record_stream(self.side)
        slots = buffer[:rows]
        handle = stream.cuda_stream
        # -1 everywhere, then every token's slot.
        self.driver.fill(buffer.data_ptr(), 0xFF, 8 * rows, handle)
        launch = self.slots_launch
        assert launch is not None
        launch.update(
            restore_ptr=restore.data_ptr(),
            slots_ptr=buffer.data_ptr(),
            context_start=plan.context_start,
            new_rows=new_rows,
            interleave=self.interleave,
            dcp_size=world,
            per_rank=per_rank,
        )(handle, -(-new_rows // kv_ce_rows.SLOT_BLOCK))
        shard_rows = plan.shard_rows
        self.table_writer.write(
            handle,
            1 + 2 * world,
            [
                *(p * per_rank for p in range(world)),
                *(p * shard_rows + old for p, old in enumerate(plan.old_lens)),
                *counts,
            ],
        )
        self.stage_grid = max(-(-max(counts) // kv_ce_rows.STAGE_ROWS), 1)
        self.stage_key, self.stage_slots = key, slots
        return slots

    def _ensure_stage_cache(
        self, rows: int, block_size: int, entry: int, dtype: torch.dtype
    ) -> None:
        """The scratch block cache the new rows are quantized into: at least
        `rows` entries in blocks of the KV cache's shape."""
        blocks = -(-rows // block_size)
        cache = self.stage_cache
        if (
            cache is not None
            and cache.shape[0] >= blocks
            and tuple(cache.shape[1:]) == (block_size, entry)
            and cache.dtype == dtype
        ):
            return
        self.stage_cache = torch.empty(
            (blocks, block_size, entry), dtype=dtype, device=self.device
        )
        self.stage_cache.record_stream(self.side)
        if self.stage_launch is not None:
            self.stage_launch.set("cache_ptr", self.stage_cache.data_ptr())

    def join(self, chunk: Any) -> torch.Tensor | None:
        """Finish this layer's gather, or return None if it must gather serially.

        A layer served with only its selected rows attends with the remapped top-k
        of `take_joined_topk`.
        """
        if self.plan is not None and not self.plan.serves(chunk):
            self._abandon()
            self._serial()
            return None
        pending, self.pending = self.pending, None
        plan = self.plan
        if pending is None or plan is None or pending.position != self.position:
            if pending is not None:
                self._wait(current_stream(), self.side)
            self._serial()
            return None
        main = current_stream()
        self.driver.record(self.fence, self.side_handle)
        self.driver.wait_event(main.cuda_stream, self.fence)
        if not pending.staged:
            # The new rows never reached the region: the layer gathers serially.
            self._serial()
            return None
        region = self._region(pending.position, plan.shard_rows)
        if pending.compact:
            assert self.sel is not None and self.sel.state is not None
            state = self.sel.state
            self.joined_topk = (self.sel.topk[: state.tokens], state.topk_length)
        return region

    def take_joined_topk(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The remapped top-k and lengths of the layer `join` served with only its
        selected rows, once."""
        topk, self.joined_topk = self.joined_topk, None
        return topk


@functools.cache
def get_kv_ce_overlap(
    group: GroupCoordinator,
    device: torch.device,
    shard_capacity: int,
    gathered_capacity: int,
    row_dim: int,
    num_ubatches: int,
    block_size: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> KVCEOverlap | None:
    """The process-wide overlap state, or None where it cannot run. `block_size`
    and `max_num_batched_tokens` size the staging of a chunk's new rows when the
    state is built."""
    if num_ubatches > 1 or not symm_mem_available:
        return None
    if not direct_cp_enabled(group, torch.bfloat16, None):
        return None
    overlap = KVCEOverlap(
        group,
        device,
        shard_capacity,
        gathered_capacity,
        row_dim,
        block_size,
        max_num_batched_tokens,
    )
    return overlap if overlap.ready else None
