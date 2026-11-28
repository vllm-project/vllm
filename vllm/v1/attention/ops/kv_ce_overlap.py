# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap DCP prefill KV all-gathers behind attention using zero-SM copy engines.

While layer `l` attends, an auxiliary stream (`side`) and a dedicated zero-SM
hardware copy-engine stream (`copy_stream`) prepare layer `l + 1`'s gathered
context in an alternating ping-pong region over peer-mapped symmetric memory:

- **Compressed wire & local staging**: raw `fp8_ds_mla` rows (656 B) travel
  over symmetric memory and upconvert on the receiver; new chunk rows already
  held by every rank are staged locally (`stage_new_rows`) and never cross the
  wire twice.
- **Zero-SM conflict-free ring pull & multi-stream pipeline**: peers are pulled
  in staggered ring order (`pull_order`) on `copy_stream` immediately after
  `HEAD_READY`, eliminating `(world - 1)`-to-1 NVLink incast while `side`
  overlaps own-slot upconversion and pipelines per-peer upconverts behind
  per-peer CUDA events (falling back to a batched fast path at short contexts).
- **Deferred two-channel barriers & deterministic planning**: `HEAD_DONE` is
  deferred until the next pack (`head_done_owed`) so no SM-blocking barrier sits
  on `join()`. Every rank derives the same `KVCEOverlapPlan` from
  `DCPPrefillStep` without CPU collectives, guarded by a device-side 64-bit
  `plan_check` hash behind the first `HEAD_READY`.

Enabled by `ParallelConfig.dcp_kv_prefetch`. Ineligible steps stay serial.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
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
from vllm.v1.attention.ops import kv_ce_rows
from vllm.v1.attention.ops.cp_common import (
    direct_cp_enabled,
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

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

# A rank that never reaches a barrier traps its peers instead of hanging them.
BARRIER_TIMEOUT_MS = 60_000
CHANNEL_HEAD_READY = 1
CHANNEL_HEAD_DONE = 2
# Bytes of one fp8_ds_mla cache row: 512 FP8 values, 4 FP32 scales, 64 BF16 rope.
KV_ENTRY_BYTES = 656
# Rows per block when an FP8 staging buffer is read as a paged cache.
STAGE_BLOCK = 64
# A head's copies go out in chunks of at most this many bytes on the batched path.
COPY_CHUNK_BYTES = 16 << 20
# Minimum head rows per rank to pipeline per-peer copy-engine transfers and
# upconverts across `copy_stream` and `side`; smaller contexts use the batched
# fast path to minimize host dispatch overhead.
PIPELINE_MIN_ROWS = 8192
# Plan words of a rank, in peer-mapped memory. Consecutive plans alternate: a peer
# may compare a step's word while this rank already writes the next step's.
PLAN_SLOTS = 2


@dataclass(frozen=True)
class DCPPrefillStep:
    """Step-level prefill schedule shared identically across all DCP ranks.

    Decouples `KVCEOverlap` from PCP-specific managers so both PCP + DCP
    (`restore_idx` mapping global tokens into PCP-gathered rows) and TP + DCP
    (`restore_idx=None`, identity token order) share the same prefetch engine.
    """

    context_start: int
    seq_end: int
    restore_idx: torch.Tensor | None = None


def plan_hash(digest: list[int]) -> tuple[int, int]:
    """The halves of a 64-bit hash of a plan's digest: the plan word the ranks
    compare on the device."""
    packed = struct.pack(f"<{len(digest)}q", *digest)
    low, high = struct.unpack("<II", hashlib.blake2b(packed, digest_size=8).digest())
    return low, high


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

    def digest(self) -> list[int]:
        """Everything of the plan a rank acts on, as integers: the DCP ranks
        compare a hash of it (`plan_hash`)."""
        return [
            1,
            self.shard_rows,
            self.head_rows,
            self.context_start,
            self.seq_end,
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
    # Per-peer copy lists and upconvert runs in staggered `pull_order` for the
    # 2-stream (copy_stream + side) pipeline.
    peer_copies: list[list[CopyList]]
    peer_runs: tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]
    pipelined: bool


class _Step:
    """What the layers of one step share, made once when its plan is taken."""

    __slots__ = (
        "plan",
        "chunks",
        "regions",
        "slot",
        "checked",
        "head",
        "kept",
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
        self.head: _WholeHead | None = None
        # Chunks whose tensors the auxiliary stream already keeps alive.
        self.kept: set[int] = set()


class DCPCopyEngineBase:
    """Shared symmetric-memory, multi-stream copy-engine, and deferred-barrier
    machinery for `KVCEOverlap` (sparse MLA) and `ChunkedDCPKVPrefetch` (dense MLA)."""

    device_group: Any
    cpu_group: Any
    world_size: int
    rank: int
    device: torch.device
    row_dim: int
    head_done_owed: bool
    plan_count: int
    plan_check: KernelLaunch | None
    plan: Any
    step: Any
    pending: Any
    driver: Any
    fence: int
    side: torch.cuda.Stream
    side_handle: int
    copy_stream: torch.cuda.Stream
    copy_handle: int
    copy_ready: int
    peer_events: list[int]
    wire_peers: list[int]
    pull_order: list[int]
    wire_index: dict[int, int]
    own_wire: torch.Tensor
    wire_zero: torch.Tensor
    wire_bounds: torch.Tensor
    plan_words: torch.Tensor
    handle: Any
    remote: list[torch.Tensor]
    remote_ptrs: list[int]
    plan_handle: Any
    plan_remote: list[torch.Tensor]
    plan_deltas: torch.Tensor

    def _init_base(
        self, group: GroupCoordinator, device: torch.device, row_dim: int
    ) -> None:
        self.device_group = group.device_group
        self.cpu_group = group.cpu_group
        self.world_size = group.world_size
        self.rank = group.rank_in_group
        self.device = torch.device(device)
        self.row_dim = row_dim
        self.head_done_owed = False
        self.plan_count = 0
        self.plan_check = None
        self.wire_peers = [p for p in range(self.world_size) if p != self.rank]
        self.pull_order = self._pull_order()
        self.wire_index = {peer: idx for idx, peer in enumerate(self.wire_peers)}

    def _pull_order(self) -> list[int]:
        """Conflict-free staggered ring pull order across DCP peers."""
        world, rank = self.world_size, self.rank
        return [(rank + step) % world for step in range(1, world)]

    def _agree(self, values: list[int]) -> bool:
        """Blocking CPU collective verifying every DCP rank allocated identically."""
        local = torch.tensor(values, dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local, group=self.cpu_group)
        return all(torch.equal(other, local) for other in gathered)

    def _init_common_device_state(self) -> None:
        self.wire_zero = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.wire_bounds = torch.tensor(
            [0, torch.iinfo(torch.int32).max], dtype=torch.int32, device=self.device
        )
        self.driver = cuda_driver()
        self.plan_words = symm_mem.empty(
            PLAN_SLOTS, dtype=torch.int64, device=self.device
        )
        self.fence = self.driver.event()
        self.side = torch.cuda.Stream(device=self.device)
        self.side_handle = self.side.cuda_stream
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.copy_handle = self.copy_stream.cuda_stream
        self.copy_ready = self.driver.event()
        self.peer_events = [self.driver.event() for _ in self.pull_order]

    def _init_paged_staging(
        self, wire_rows: int, wire_cols: int, wire_dtype: torch.dtype
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        stage = torch.empty(
            (len(self.wire_peers) * wire_rows, wire_cols),
            dtype=wire_dtype,
            device=self.device,
        )
        blocks = wire_rows // STAGE_BLOCK
        cache = stage.view(-1, STAGE_BLOCK, wire_cols)
        table = (
            torch.arange(len(self.wire_peers), dtype=torch.int32)[:, None] * blocks
            + torch.arange(blocks, dtype=torch.int32)
        ).to(self.device)
        part_table = [table[i : i + 1] for i in range(len(self.wire_peers))]
        return stage, stage.data_ptr(), cache, table, part_table

    def _connect_symm_buffers(self) -> None:
        self.own_wire.zero_()
        torch.accelerator.synchronize()
        self.handle = symm_mem.rendezvous(self.own_wire, self.device_group.group_name)
        self.handle.barrier()
        self.remote = [
            self.handle.get_buffer(
                peer, list(self.own_wire.shape), self.own_wire.dtype, 0
            )
            for peer in range(self.world_size)
        ]
        self.remote_ptrs = [buf.data_ptr() for buf in self.remote]
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
        self.plan_deltas = torch.tensor(
            [
                words.data_ptr() - self.plan_words.data_ptr()
                for words in self.plan_remote
            ],
            dtype=torch.int64,
            device=self.device,
        )

    def _bind_plan_check(self, check_kernel: DirectKernel) -> None:
        self.plan_check = check_kernel.bind(
            words_ptr=self.plan_words.data_ptr(),
            deltas_ptr=self.plan_deltas.data_ptr(),
            slot=0,
        )
        self.plan_check(self.side_handle, 1)

    def _wait(self, waiter: torch.cuda.Stream, signaler: torch.cuda.Stream) -> None:
        self.driver.record(self.fence, signaler.cuda_stream)
        self.driver.wait_event(waiter.cuda_stream, self.fence)

    def _head_ready(self) -> None:
        """Pass HEAD_READY on the auxiliary stream and verify peer plan hashes once."""
        self.handle.barrier(channel=CHANNEL_HEAD_READY, timeout_ms=BARRIER_TIMEOUT_MS)
        step = self.step
        if step is None or step.checked:
            return
        step.checked = True
        assert self.plan_check is not None
        self.plan_check.set("slot", step.slot)
        self.plan_check(current_stream().cuda_stream, 1)

    def _pay_head_done(self, where: str, scheduled: bool) -> None:
        """Settle an owed HEAD_DONE barrier before packing new rows."""
        if not self.head_done_owed or not scheduled:
            return
        if where == "main":
            self._wait(current_stream(), self.side)
        self.handle.barrier(channel=CHANNEL_HEAD_DONE, timeout_ms=BARRIER_TIMEOUT_MS)
        self.head_done_owed = False

    def _abandon(self) -> None:
        """Forget the step; the auxiliary stream must finish before reuse."""
        if self.pending is not None:
            self._wait(current_stream(), self.side)
            self.pending = None
        self.plan = self.step = None


class KVCEOverlap(DCPCopyEngineBase):
    """Buffers, auxiliary streams and per-step state of the overlapped DCP KV gather."""

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
        self._init_base(group, device, row_dim)
        self.shard_capacity = shard_capacity
        self.gathered_capacity = gathered_capacity
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
        self.slots_identity_launch: KernelLaunch | None = None

        self.order = LayerOrder()
        self.plan: KVCEOverlapPlan | None = None
        self.step: _Step | None = None
        self.pending: _Pending | None = None
        self.position: int | None = None
        self.forward: weakref.ref | None = None
        self.layers: dict[str, Any] = {}

        self.ready = self._allocate()
        if self.ready and block_size and max_num_batched_tokens:
            per_rank = -(-max_num_batched_tokens // self.world_size) + block_size
            self._ensure_stage_cache(
                self.world_size * per_rank, block_size, KV_ENTRY_BYTES, torch.uint8
            )
            torch.accelerator.synchronize()

    def _allocate(self) -> bool:
        try:
            self.shard = torch.empty(
                (self.shard_capacity, self.row_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.gathered = torch.empty(
                (self.gathered_capacity, self.row_dim),
                dtype=torch.bfloat16,
                device=self.device,
            )
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
            self._init_common_device_state()
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
            self.kernels = self._compile_kernels()
            allocated = 1
        except (RuntimeError, ValueError, OSError, AttributeError) as error:
            logger.warning("KV copy-engine overlap is unavailable: %s", error)
            allocated = 0
        all_allocated = self._agree([allocated])
        if not allocated or not all_allocated:
            self.shard = self.gathered = None  # type: ignore[assignment]
            return False
        self._connect_symm_buffers()
        self.gathered_ptr = self.gathered.data_ptr()
        self.row_bytes = self.row_dim * self.gathered.element_size()
        self._load_kernels()
        return True

    def _compile_kernels(
        self,
    ) -> tuple[DirectKernel, DirectKernel, DirectKernel, DirectKernel]:
        """The staging and plan-check kernels, compiled and loaded for direct
        launches; raises where they cannot be launched that way."""
        slots = kv_ce_rows.stage_slots_kernel(has_restore=True)
        slots_id = kv_ce_rows.stage_slots_kernel(has_restore=False)
        rows = kv_ce_rows.stage_rows_kernel()
        check = kv_ce_rows.plan_check_kernel(self.world_size)
        if slots is None or slots_id is None or rows is None or check is None:
            raise RuntimeError("its Triton kernels cannot be launched directly")
        return slots, slots_id, rows, check

    def _load_kernels(self) -> None:
        """Bind the staging and plan-check kernels and run every device operation
        of a head and of the staging once on the auxiliary stream, with nothing
        to do, so the first step with history pays no kernel load or first
        launch."""
        slots, slots_id, rows, check = self.kernels
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
        self.slots_identity_launch = slots_id.bind(
            restore_ptr=anchor,
            slots_ptr=anchor,
            interleave=1,
            dcp_size=1,
            per_rank=1,
        )
        self.slots_identity_launch(handle, 1)
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
        torch.accelerator.synchronize()
        self._bind_plan_check(check)
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
                torch.zeros(
                    (1, kv_ce_rows.FP8_NOPE), dtype=torch.bfloat16, device=self.device
                ),
                torch.zeros(
                    (1, kv_ce_rows.ROPE_DIM), dtype=torch.bfloat16, device=self.device
                ),
                block,
                torch.zeros(1, dtype=torch.int64, device=self.device),
                kv_cache_dtype="fp8_ds_mla",
                scale=torch.ones(1, dtype=torch.float32, device=self.device),
            )
        torch.accelerator.synchronize()

    def _allocate_wire(self) -> None:
        """Allocate symmetric `own_wire` and per-peer paged staging buffers."""
        rows = -(-self.gathered_capacity // (2 * self.world_size))
        rows = -(-rows // STAGE_BLOCK) * STAGE_BLOCK
        self.wire_rows = rows
        self.own_wire = symm_mem.empty(
            (rows, KV_ENTRY_BYTES), dtype=torch.uint8, device=self.device
        )
        (
            self.wire_stage,
            self.stage_ptr,
            self.wire_cache,
            self.wire_table,
            self.wire_part_table,
        ) = self._init_paged_staging(rows, KV_ENTRY_BYTES, torch.uint8)

    def shard_view(self, shard_rows: int) -> torch.Tensor:
        return self.shard[:shard_rows]

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

    def begin_layer(
        self,
        layer: Any,
        forward: object,
        prefill_of: Callable[[str], Any],
        num_decode_tokens: int,
        interleave: int,
        step_info: DCPPrefillStep | None,
    ) -> None:
        """Track the layer's position in `forward`; its first layer plans it.

        A plan and a prefetch belong to the forward that made them: both are
        dropped when another forward starts, so no layer joins rows that were
        laid out for another step's shard length.
        """
        name = layer.layer_name
        self.layers[name] = layer
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
                name, prefill_of, num_decode_tokens, interleave, step_info
            )

    def _derive_plan(
        self,
        name: str,
        prefill_of: Callable[[str], Any],
        num_decode_tokens: int,
        interleave: int,
        step_info: DCPPrefillStep | None,
    ) -> None:
        """Derive the step's plan and take it, or leave the step serial.

        Whether the step is planned and every number of the plan follow from
        `step_info` (which every DCP rank derives identically from the global
        schedule), the configuration and the layer order: no rank asks another.
        This rank's own metadata is checked against them; a mismatch raises
        rather than leaving peers waiting at barriers.
        """
        if step_info is None:
            return
        prefill = prefill_of(name)
        if prefill is None:
            return
        start, end = step_info.context_start, step_info.seq_end
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
                "DCP KV prefetch: this rank's prefill metadata (chunks, "
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
        names = self.order.names
        chunks: list[Any] = []
        # Layers that share a prefill object share the answer.
        served: dict[int, Any] = {}
        for other in names:
            other_prefill = prefill_of(other)
            key = id(other_prefill)
            if key not in served:
                served[key] = (
                    other_prefill.chunks[0]
                    if self._same_plan(other_prefill, plan)
                    else None
                )
            if served[key] is None:
                return
            chunks.append(served[key])
        restore = step_info.restore_idx
        if restore is not None and restore.shape[0] != end - start:
            raise RuntimeError(
                "DCP KV prefetch: restore index covers "
                f"{restore.shape[0]} rows of a request scheduled for "
                f"{end - start}."
            )
        self.interleave = interleave
        self.restore_idx = restore
        old_lens, new_lens = shard_lens(start, end, world, interleave)
        plan = dataclasses.replace(
            plan, old_lens=old_lens, new_lens=new_lens, seq_end=end
        )
        self._take_plan(plan, chunks)

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

    def _serial(self, scheduled: bool) -> None:
        """A layer that gathers serially settles the owed barrier first."""
        self._pay_head_done("main", scheduled)

    def start_next(self) -> None:
        """Start the next layer's head rows; call right before attention launches.

        The next layer is always in this forward: the learned order ends at a
        forward's last layer, and every layer's chunk was checked against the
        plan when the forward was planned.
        """
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
            self._fp8_head(self.layers[name].kv_cache, chunk, position, plan)
        finally:
            torch.cuda.set_stream(main)
        # No barrier follows the head copies; see module doc.
        self.head_done_owed = True
        self.pending = _Pending(name, position)

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

    def _peer_span_copies(
        self, spans: list[list[Span]], peer: int
    ) -> list[tuple[int, int, int]]:
        """`(destination, source, bytes)` of one peer's spans."""
        rows, entry = self.wire_rows, KV_ENTRY_BYTES
        index = self.wire_index[peer]
        remote = self.remote_ptrs[peer]
        return [
            (
                self.stage_ptr + (index * rows + begin) * entry,
                remote + begin * entry,
                (end - begin) * entry,
            )
            for begin, end, _ in spans[peer]
            if end > begin
        ]

    def _span_copies(self, spans: list[list[Span]]) -> list[tuple[int, int, int]]:
        """`(destination, source, bytes)` of every peer's spans in staggered
        `pull_order`: from the buffer peers read into the peer's staging part."""
        out: list[tuple[int, int, int]] = []
        for peer in self.pull_order:
            out.extend(self._peer_span_copies(spans, peer))
        return out

    def _peer_runs(self, region: torch.Tensor, shard_rows: int) -> list[tuple]:
        """Per-peer upconvert arguments in staggered `pull_order`."""
        runs = []
        for peer in self.pull_order:
            index = self.wire_index[peer]
            runs.append(
                (
                    self.wire_cache,
                    region[peer * shard_rows : (peer + 1) * shard_rows],
                    self.wire_part_table[index],
                    self.wire_zero,
                    1,
                )
            )
        return runs

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
            pipelined = plan.head_rows >= PIPELINE_MIN_ROWS and len(self.pull_order) > 1
            peer_copies = [
                [CopyList(c)] if c else []
                for peer in self.pull_order
                for c in (self._peer_span_copies(spans, peer),)
            ]
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
                peer_copies,
                (
                    self._peer_runs(regions[0], shard_rows),
                    self._peer_runs(regions[1], shard_rows),
                ),
                pipelined,
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
        # Peers only read `own_wire`, never `region`: signal HEAD_READY first
        # so zero-SM copy-engine transfers start immediately while `self.side`
        # upconverts this rank's own rows.
        self._head_ready()
        driver = self.driver
        side_h = self.side_handle
        copy_h = self.copy_handle
        driver.record(self.copy_ready, side_h)
        driver.wait_event(copy_h, self.copy_ready)
        upconvert = ops.cp_gather_and_upconvert_fp8_kv_cache
        parity = position % 2
        if head.pipelined:
            for peer_idx, peer_chunks in enumerate(head.peer_copies):
                for copy_list in peer_chunks:
                    driver.memcpy_list(copy_list, copy_h)
                driver.record(self.peer_events[peer_idx], copy_h)
            for slot, (_, starts) in zip(head.slot[parity], head.wire):
                upconvert(kv_cache, slot, table, starts_at, 1, starts)
            for peer_idx, run in enumerate(head.peer_runs[parity]):
                driver.wait_event(side_h, self.peer_events[peer_idx])
                upconvert(*run)
        else:
            for copy_list in head.chunks:
                driver.memcpy_list(copy_list, copy_h)
            if self.peer_events:
                driver.record(self.peer_events[0], copy_h)
            for slot, (_, starts) in zip(head.slot[parity], head.wire):
                upconvert(kv_cache, slot, table, starts_at, 1, starts)
            if self.peer_events:
                driver.wait_event(side_h, self.peer_events[0])
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
        scheduled: bool = True,
    ) -> None:
        """Put layer `name`'s new rows into its gathered region.

        `kv_c` and `k_pe` are the full chunk's new rows ready on the current
        stream. The auxiliary stream quantizes them with the cache-write kernel
        into a scratch block cache and upconverts every rank's share into that
        rank's slot range.
        """
        plan, pending = self.plan, self.pending
        if (
            plan is None
            or pending is None
            or pending.name != name
            or pending.staged
            or self.forward is None
            or self.forward() is not forward
            or not scheduled
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
            assert self.stage_cache is not None
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
        """Scratch slot of each new row for this step, rank-major: rank `p`'s
        new rows in token order from `p * per_rank`; -1 for padding rows.
        Written on `stream` with the step's staging tables, once per step."""
        key = (self.plan_seq, rows)
        if self.stage_key == key:
            return self.stage_slots
        restore = self.restore_idx
        new_rows = plan.seq_end - plan.context_start
        if rows < new_rows or new_rows <= 0:
            return None
        if restore is not None and (
            restore.shape[0] != new_rows
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
        launch = (
            self.slots_launch if restore is not None else self.slots_identity_launch
        )
        assert launch is not None
        launch.update(
            restore_ptr=(
                restore.data_ptr() if restore is not None else buffer.data_ptr()
            ),
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

    def join(self, chunk: Any, scheduled: bool = True) -> torch.Tensor | None:
        """Finish this layer's gather, or return None if it must gather serially."""
        if self.plan is not None and not self.plan.serves(chunk):
            self._abandon()
            self._serial(scheduled)
            return None
        pending, self.pending = self.pending, None
        plan = self.plan
        if pending is None or plan is None or pending.position != self.position:
            if pending is not None:
                self._wait(current_stream(), self.side)
            self._serial(scheduled)
            return None
        main = current_stream()
        self.driver.record(self.fence, self.side_handle)
        self.driver.wait_event(main.cuda_stream, self.fence)
        if not pending.staged:
            # The new rows never reached the region: the layer gathers serially.
            self._serial(scheduled)
            return None
        return self._region(pending.position, plan.shard_rows)


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
