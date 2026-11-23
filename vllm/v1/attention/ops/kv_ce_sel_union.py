# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefetch only the selected KV rows for layers that reuse an earlier layer's top-k.

The KV prefetch (`kv_ce_overlap`) gathers every old row of the next layer while the
current one attends. A layer whose index group already ran its indexer knows its
top-k before the previous layer finishes, so its prefetch can carry only the rows
that some query of this chunk selects. Every byte that crosses GPUs moves by the
copy engines; the kernels here read and write local memory only.

1. Each rank tags the shard rows its queries select (one byte per row, set to the
   build's epoch) in its peer-mapped marks. After a barrier every rank copies the
   marks of every peer with the copy engines, ORs them locally and numbers each
   owner's union rows with a flat scan, so all ranks hold the same numbering.
2. Each owner copies the raw cache entries of its union rows, in row order, to the
   front of the buffer peers read, and upconverts them to the front of its own slot
   of the gathered region.
3. Every rank copies the first `U` rows of each peer's buffer into its staging with
   the copy engines and upconverts them into the peer's slot. `U`, the union size,
   reaches the host by a non-blocking readback (`UnionSizes`); `kv_ce_overlap`
   copies a predicted number of rows when it has not landed, and `kv_ce_topup`
   completes the copy on the device.
4. The top-k indices are remapped once per group onto the compacted slots; rows at
   or above `head_rows` keep their places, so the rows `kv_ce_overlap` moves whole
   (old rows past `head_rows`, the staged new rows and the padding) are
   unchanged.

Each remapped index addresses a row that holds exactly the bytes the whole-context
gather would put at the original index, in the same order, so the attention kernel
reads the same values in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.cuda_direct import (
    POINTER_ALIGNMENT,
    CopyList,
    CudaDriver,
    DirectKernel,
    KernelLaunch,
    cuda_driver,
    direct_launch,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Bytes of one fp8_ds_mla cache entry the upconvert reads, and its BF16 row width.
FP8_NOPE = 512
FP8_SCALE_OFFSET = 512
FP8_ROPE_OFFSET = 528
ROPE_DIM = 64
ROW_DIM = FP8_NOPE + ROPE_DIM

# Marks hold the epoch of the build that set them, so they need no clearing between
# builds; they are cleared once every EPOCHS builds, before an epoch comes back.
EPOCHS = 255

# The same layout, as constants the kernels can read.
_NOPE = tl.constexpr(FP8_NOPE)
_SCALE_OFFSET = tl.constexpr(FP8_SCALE_OFFSET)
_ROPE_OFFSET = tl.constexpr(FP8_ROPE_OFFSET)
_ROPE_DIM = tl.constexpr(ROPE_DIM)
_ROW_DIM = tl.constexpr(ROW_DIM)


@triton.jit(
    do_not_specialize=["shard_rows", "head_rows", "epoch"],
    do_not_specialize_on_alignment=["p_ptr"],
)
def _mark_kernel(
    p_ptr,
    mask_ptr,
    shard_rows: tl.int32,
    head_rows: tl.int32,
    epoch: tl.int32,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """mask[owner * head_rows + row] = epoch for every selected old row."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    p = tl.load(p_ptr + t * K + cols, mask=cols < K, other=-1)
    owner = p // shard_rows
    row = p - owner * shard_rows
    hit = (p >= 0) & (row < head_rows)
    tags = (tl.zeros([BLOCK], dtype=tl.int32) + epoch).to(tl.uint8)
    tl.store(mask_ptr + owner * head_rows + row, tags, mask=hit)


@triton.jit(do_not_specialize=["peer_stride", "n", "rank", "epoch"])
def _union_count_kernel(
    own_ptr,
    peers_ptr,
    peer_stride: tl.int32,
    union_ptr,
    counts_ptr,
    n: tl.int32,
    rank: tl.int32,
    epoch: tl.int32,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Mark a row in `union` when any rank tagged it with this epoch (this rank in
    its own marks, a peer in its local copy); `counts` gets the union rows of each
    block."""
    block = tl.program_id(0)
    offs = block.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = offs < n
    hit = tl.load(own_ptr + offs, mask=inside, other=0) == epoch
    for peer in tl.static_range(WORLD):
        others = inside & (rank != peer)
        tag = tl.load(peers_ptr + peer * peer_stride + offs, mask=others, other=0)
        hit = hit | (tag == epoch)
    tl.store(union_ptr + offs, hit.to(tl.uint8), mask=inside)
    tl.store(counts_ptr + block, tl.sum(hit.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["n", "head_rows"])
def _union_scan_kernel(
    union_ptr,
    counts_ptr,
    cum_ptr,
    ends_ptr,
    n: tl.int32,
    head_rows: tl.int32,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Inclusive scan of the union over all slots into `cum`; `ends[r + 1]` gets
    its value at the end of slot `r` (`ends[0]` stays 0)."""
    block = tl.program_id(0)
    base = tl.sum(tl.zeros([CHUNK], dtype=tl.int32), axis=0)
    for start in range(0, block, CHUNK):
        idx = start + tl.arange(0, CHUNK)
        base += tl.sum(tl.load(counts_ptr + idx, mask=idx < block, other=0), axis=0)
    first = block.to(tl.int64) * BLOCK
    offs = first + tl.arange(0, BLOCK)
    inside = offs < n
    picked = tl.load(union_ptr + offs, mask=inside, other=0).to(tl.int32)
    cum = tl.cumsum(picked, axis=0) + base
    tl.store(cum_ptr + offs, cum, mask=inside)
    for r in tl.static_range(WORLD):
        last = (r + 1) * head_rows.to(tl.int64) - 1
        value = tl.sum(tl.where(offs == last, cum, 0), axis=0)
        tl.store(ends_ptr + r + 1, value, mask=(last >= first) & (last < first + BLOCK))


@triton.jit(
    do_not_specialize=["shard_rows", "head_rows"],
    do_not_specialize_on_alignment=["p_ptr"],
)
def _remap_kernel(
    p_ptr,
    cum_ptr,
    ends_ptr,
    out_ptr,
    shard_rows: tl.int32,
    head_rows: tl.int32,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Old rows move to their place in the owner's compacted slot; the chunk's own
    rows and the -1 padding keep their values. `cum` is the inclusive scan of the
    union over all slots, `ends[r]` its value before slot `r`."""
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inside = cols < K
    p = tl.load(p_ptr + t * K + cols, mask=inside, other=-1)
    owner = p // shard_rows
    row = p - owner * shard_rows
    head = (p >= 0) & (row < head_rows)
    rank = tl.load(cum_ptr + owner * head_rows + row, mask=head, other=1)
    base = tl.load(ends_ptr + owner, mask=head, other=0)
    out = tl.where(head, owner * shard_rows + rank - base - 1, p)
    tl.store(out_ptr + t * K + cols, out, mask=inside)


@triton.jit(
    do_not_specialize=[
        "rank", "slot_start", "head_rows",
        "block_size", "cache_block_stride", "cache_entry_stride",
    ],
    do_not_specialize_on_alignment=["block_table_ptr", "workspace_starts_ptr"],
)  # fmt: skip
def _pack_kernel(
    cache_ptr,
    words_ptr,
    block_table_ptr,
    workspace_starts_ptr,
    union_ptr,
    cum_ptr,
    ends_ptr,
    rank: tl.int32,
    wire_ptr,
    region_ptr,
    slot_start: tl.int32,
    head_rows: tl.int32,
    block_size: tl.int32,
    cache_block_stride: tl.int32,
    cache_entry_stride: tl.int32,
    ROWS: tl.constexpr,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy the raw cache entries of this rank's union rows, in row order, to the
    front of `wire` (`WORDS` int32 words per row; `words_ptr` is the cache read as
    int32 words), and upconvert the same rows to the front of its slot.

    The arithmetic is that of `cp_gather_and_upconvert_fp8_kv_cache`: fp8 to float
    (exact), times the float scale of its 128-element tile, rounded to BF16; the
    RoPE half is copied. Shard row `r` is request token `r - start`; strides are in
    bytes.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    marks = slot_start + rows
    picked = tl.load(union_ptr + marks, mask=rows < head_rows, other=0) != 0
    if tl.max(picked.to(tl.int32), axis=0) == 0:
        return
    base = tl.load(ends_ptr + rank)
    dest = (tl.load(cum_ptr + marks, mask=picked, other=1) - base - 1).to(tl.int64)
    token = rows - tl.load(workspace_starts_ptr)
    block = tl.load(block_table_ptr + token // block_size, mask=picked, other=0)
    entry = (
        block.to(tl.int64) * cache_block_stride
        + (token % block_size).to(tl.int64) * cache_entry_stride
    )
    wcols = tl.arange(0, BLOCK)
    mw = picked[:, None] & (wcols[None, :] < WORDS)
    words = tl.load(words_ptr + entry[:, None] // 4 + wcols[None, :], mask=mw, other=0)
    tl.store(wire_ptr + dest[:, None] * WORDS + wcols[None, :], words, mask=mw)
    cols = tl.arange(0, _NOPE)
    m2 = picked[:, None]
    q = tl.load(cache_ptr + entry[:, None] + cols[None, :], mask=m2, other=0)
    scale_ptr = cache_ptr + entry[:, None] + _SCALE_OFFSET + (cols[None, :] // 128) * 4
    scale = tl.load(scale_ptr.to(tl.pointer_type(tl.float32)), mask=m2, other=0.0)
    nope = (q.to(tl.float8e4nv, bitcast=True).to(tl.float32) * scale).to(tl.bfloat16)
    out = dest[:, None] * _ROW_DIM + cols[None, :]
    tl.store(region_ptr + out, nope, mask=m2)
    rcols = tl.arange(0, _ROPE_DIM)
    rope_ptr = cache_ptr + entry[:, None] + _ROPE_OFFSET + rcols[None, :] * 2
    rope = tl.load(rope_ptr.to(tl.pointer_type(tl.int16)), mask=m2, other=0)
    rout = dest[:, None] * _ROW_DIM + _NOPE + rcols[None, :]
    tl.store(region_ptr.to(tl.pointer_type(tl.int16)) + rout, rope, mask=m2)


MARK_BLOCK = 1024
SCAN_BLOCK = 4096
SCAN_CHUNK = 1024
PACK_ROWS = 8
MAX_WORLD = 8


def topk_rows(metadata: Any) -> tuple[int, int, int] | None:
    """Which rows of the shared top-k buffer a layer's single prefill chunk reads."""
    fp8 = getattr(metadata, "fp8_extra_metadata", None)
    prefill = getattr(fp8, "prefill", None)
    if prefill is None or len(prefill.chunks) != 1:
        return None
    tokens = prefill.chunks[0].tokens_slice
    if tokens.step not in (None, 1):
        return None
    return metadata.topk_start, tokens.start, tokens.stop


@dataclass(frozen=True)
class TopkUse:
    """How one layer reads the shared top-k in a step: what decides whether it
    reuses the top-k of the layer before it. `group`, `buffer` and `metadata` are
    compared by identity."""

    # Its index group, which runs its indexer at `group_index` 0.
    group: Any
    group_index: int
    # Its top-k buffer, and the rows of it its single prefill chunk reads.
    buffer: Any
    rows: tuple[int, int, int]
    # Its attention metadata object.
    metadata: Any


def topk_use(layer: Any, metadata: Any) -> TopkUse | None:
    """`layer`'s use of the top-k with `metadata`, or None when it cannot reuse
    one: no index group with an indexer (or a HiSparse one), no top-k buffer, or
    not exactly one prefill chunk."""
    from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup

    impl = getattr(layer, "impl", None)
    group = getattr(impl, "index_group", None)
    rows = topk_rows(metadata)
    if (
        impl is None
        or group is None
        or isinstance(group, HiSparseMLAIndexGroup)
        or not group.has_indexer
        or impl.topk_indices_buffer is None
        or rows is None
    ):
        return None
    return TopkUse(
        group, impl.index_group_index, impl.topk_indices_buffer, rows, metadata
    )


def sel_union_positions(
    uses: Sequence[TopkUse | None],
) -> tuple[frozenset[int], frozenset[int]]:
    """Positions whose layer reuses the top-k of the layer before it, and those of
    them that number the union themselves; `uses[p]` is the layer at position `p`.

    Such a layer belongs to the same index group as its predecessor, is not the
    group's indexer layer, and reads the same rows of the same top-k buffer, so the
    top-k is final once its predecessor runs. The first of a run builds the union
    from its own converted top-k; the others reuse it, since every input of their
    conversion is the same.

    A run continues only while the metadata object is its predecessor's. A layer
    whose metadata differs from that of a compact predecessor would have to build
    again, and the build rewrites the remapped top-k on the auxiliary stream while
    the predecessor's attention, launched after the build is issued, still reads
    it. Such a layer keeps the whole prefetch instead, and the next one starts a
    new run. So the predecessor of every builder is never compact.
    """
    positions: set[int] = set()
    builds: set[int] = set()
    for position in range(1, len(uses)):
        previous, use = uses[position - 1], uses[position]
        if (
            previous is None
            or use is None
            or use.group_index <= 0
            or use.group is not previous.group
            or use.buffer is not previous.buffer
            or use.rows != previous.rows
        ):
            continue
        if position - 1 in positions:
            if use.metadata is not previous.metadata:
                # Building here would race with the predecessor's attention.
                continue
        else:
            builds.add(position)
        positions.add(position)
    return frozenset(positions), frozenset(builds)


def sel_digest(positions: frozenset[int], builds: frozenset[int]) -> list[int]:
    """What the union positions add to the plan digest: the size of each set,
    then its positions in order."""
    return [len(positions), *sorted(positions), len(builds), *sorted(builds)]


class _Readback:
    """The event recorded after a readback, on a driver handle."""

    def __init__(self, driver: CudaDriver) -> None:
        self.driver = driver
        self.handle = driver.event()

    def query(self) -> bool:
        return self.driver.done(self.handle)


class UnionSizes:
    """Union size of every slot of one index group, read back without blocking.
    The page-locked buffer and the event are made once and reused by every
    build; `builds` tells one build's readback from the next."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        cuda = torch.cuda.is_available()
        self.host = torch.zeros(world_size + 1, dtype=torch.int32, pin_memory=cuda)
        self.host_ptr = self.host.data_ptr()
        self.nbytes = self.host.numel() * self.host.element_size()
        self.driver = cuda_driver() if cuda else None
        self.readback = _Readback(self.driver) if self.driver is not None else None
        # None until a build recorded a readback.
        self.event: _Readback | None = None
        self.builds = 0

    def record(self, ends: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Queue the readback of this build's slot ends on `stream`."""
        assert self.driver is not None and self.readback is not None
        handle = stream.cuda_stream
        self.driver.read(self.host_ptr, ends.data_ptr(), self.nbytes, handle)
        self.driver.record(self.readback.handle, handle)
        self.event = self.readback
        self.builds += 1

    def exact(self) -> list[int] | None:
        """The last build's sizes, if the readback already landed."""
        if self.event is None or not self.event.query():
            return None
        ends = self.host.tolist()
        return [ends[r + 1] - ends[r] for r in range(self.world_size)]


@dataclass
class GroupState:
    """One index group's union in this step, shared by its reusing layers."""

    group_id: int
    tokens: int
    sizes: UnionSizes
    built_position: int
    # The converted top-k lengths of the build, which every reuser keeps.
    topk_length: torch.Tensor


@dataclass
class _UnionLaunches:
    """The build's kernels, loaded for direct launches with their buffers bound."""

    mark: KernelLaunch
    count: KernelLaunch
    scan: KernelLaunch
    remap: KernelLaunch


class SelUnion:
    """Buffers and steps of the union pull, owned by `KVCEOverlap`."""

    def __init__(self, overlap: Any, topk_rows: int, topk_width: int) -> None:
        from vllm.v1.attention.ops.cp_common import symm_mem

        self.overlap = overlap
        self.world_size = overlap.world_size
        assert 1 < self.world_size <= MAX_WORLD
        self.rank = overlap.rank
        device = overlap.device
        n = self.world_size * overlap.shard_capacity
        self.marks = symm_mem.empty((n,), dtype=torch.uint8, device=device)
        # Copy-engine copies of the peers' marks (this rank's row is unused).
        self.peer_marks = torch.empty(
            (self.world_size, n), dtype=torch.uint8, device=device
        )
        self.union = torch.zeros(n, dtype=torch.uint8, device=device)
        self.counts = torch.zeros(
            triton.cdiv(n, SCAN_BLOCK), dtype=torch.int32, device=device
        )
        self.cum = torch.zeros(n, dtype=torch.int32, device=device)
        self.ends = torch.zeros(self.world_size + 1, dtype=torch.int32, device=device)
        self.topk = torch.empty(
            (topk_rows, topk_width), dtype=torch.int32, device=device
        )
        self.topk_width = topk_width
        self.epoch = 0
        self.sizes: dict[int, UnionSizes] = {}
        self.state: GroupState | None = None
        self.launches: _UnionLaunches | None = None
        self.pack_kernel: DirectKernel | None = None
        self.marks_list: tuple[int, CopyList] | None = None

    def connect(self) -> None:
        """Map every rank's marks; collective over the DCP group."""
        from vllm.v1.attention.ops.cp_common import symm_mem

        overlap = self.overlap
        n = self.marks.numel()
        self.marks.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(self.marks, overlap.device_group.group_name)
        handle.barrier()
        self.marks_handle = handle
        self.remote_marks = [
            handle.get_buffer(peer, [n], torch.uint8, 0)
            for peer in range(self.world_size)
        ]
        self.marks_copies = [
            (self.peer_marks[peer].data_ptr(), self.remote_marks[peer].data_ptr())
            for peer in range(self.world_size)
        ]

    def compile_kernels(self) -> None:
        """Compile and load the kernels of the build and the pack for direct
        launches; raises where they cannot be launched that way."""
        world, width = self.world_size, self.topk_width
        block = min(MARK_BLOCK, triton.next_power_of_2(width))
        self.mark_columns = triton.cdiv(width, block)
        i32, u8 = torch.int32, torch.uint8
        mark = direct_launch(
            _mark_kernel, {"p_ptr": i32, "mask_ptr": u8}, K=width, BLOCK=block
        )
        count = direct_launch(
            _union_count_kernel,
            {"own_ptr": u8, "peers_ptr": u8, "union_ptr": u8, "counts_ptr": i32},
            WORLD=world,
            BLOCK=SCAN_BLOCK,
        )
        scan = direct_launch(
            _union_scan_kernel,
            {"union_ptr": u8, "counts_ptr": i32, "cum_ptr": i32, "ends_ptr": i32},
            WORLD=world,
            BLOCK=SCAN_BLOCK,
            CHUNK=SCAN_CHUNK,
        )
        remap = direct_launch(
            _remap_kernel,
            {"p_ptr": i32, "cum_ptr": i32, "ends_ptr": i32, "out_ptr": i32},
            K=width,
            BLOCK=block,
        )
        words = self.overlap.own_wire.shape[1] // 4
        self.pack_kernel = direct_launch(
            _pack_kernel,
            {
                "cache_ptr": u8,
                "words_ptr": i32,
                "block_table_ptr": i32,
                "workspace_starts_ptr": i32,
                "union_ptr": u8,
                "cum_ptr": i32,
                "ends_ptr": i32,
                "wire_ptr": i32,
                "region_ptr": torch.bfloat16,
            },
            ROWS=PACK_ROWS,
            WORDS=words,
            BLOCK=triton.next_power_of_2(words),
        )
        if (
            mark is None
            or count is None
            or scan is None
            or remap is None
            or self.pack_kernel is None
        ):
            raise RuntimeError("the union kernels cannot be launched directly")
        self.kernels = (mark, count, scan, remap)

    def load_kernels(self, stream: torch.cuda.Stream) -> None:
        """Bind the build's buffers to its kernels and launch each kernel once on
        `stream` with nothing to do, so no step pays a load or a first launch."""
        mark, count, scan, remap = self.kernels
        cum, ends = self.cum.data_ptr(), self.ends.data_ptr()
        union, counts = self.union.data_ptr(), self.counts.data_ptr()
        topk = self.topk.data_ptr()
        self.launches = _UnionLaunches(
            mark.bind(p_ptr=topk, mask_ptr=self.marks.data_ptr()),
            count.bind(
                own_ptr=self.marks.data_ptr(),
                peers_ptr=self.peer_marks.data_ptr(),
                peer_stride=self.peer_marks.stride(0),
                union_ptr=union,
                counts_ptr=counts,
                rank=self.rank,
            ),
            scan.bind(union_ptr=union, counts_ptr=counts, cum_ptr=cum, ends_ptr=ends),
            remap.bind(p_ptr=topk, cum_ptr=cum, ends_ptr=ends, out_ptr=topk),
        )
        # A top-k row that selects nothing, zero rows to scan, an empty head.
        self.topk[:1].fill_(-1)
        torch.accelerator.synchronize()
        handle = stream.cuda_stream
        launches = self.launches
        launches.mark.update(shard_rows=1, head_rows=1, epoch=1)
        launches.mark(handle, 1, self.mark_columns)
        launches.count.update(n=0, epoch=1)(handle, 1)
        launches.scan.update(n=0, head_rows=0)(handle, 1)
        launches.remap.update(shard_rows=1, head_rows=1)
        launches.remap(handle, 1, self.mark_columns)
        wire = self.overlap.own_wire
        warm = self.pack_launch(wire.view(-1, 1, wire.shape[1]), wire)
        assert warm is not None
        table = self.ends.data_ptr()
        warm.update(
            block_table_ptr=table,
            workspace_starts_ptr=table,
            region_ptr=self.overlap.gathered.data_ptr(),
            slot_start=0,
            head_rows=0,
        )(handle, 1)
        torch.accelerator.synchronize()

    def pack_launch(
        self, kv_cache: torch.Tensor, wire: torch.Tensor
    ) -> KernelLaunch | None:
        """`_pack_kernel` bound to one layer's cache and this rank's buffers, or
        None for a cache whose rows it cannot read. The caller sets the step's
        `block_table_ptr`, `workspace_starts_ptr`, `region_ptr` (this rank's
        slot), `slot_start` and `head_rows`."""
        cache = kv_cache.view(torch.uint8)
        entry = wire.shape[1]
        if (
            self.pack_kernel is None
            or cache.dim() != 3
            or cache.stride(2) != 1
            or cache.shape[2] < entry
            or cache.stride(0) % 4
            or cache.stride(1) % 4
            or cache.data_ptr() % POINTER_ALIGNMENT
            or wire.dtype != torch.uint8
            or not wire.is_contiguous()
        ):
            return None
        return self.pack_kernel.bind(
            cache_ptr=cache.data_ptr(),
            words_ptr=cache.data_ptr(),
            union_ptr=self.union.data_ptr(),
            cum_ptr=self.cum.data_ptr(),
            ends_ptr=self.ends.data_ptr(),
            rank=self.rank,
            wire_ptr=wire.data_ptr(),
            block_size=cache.shape[1],
            cache_block_stride=cache.stride(0),
            cache_entry_stride=cache.stride(1),
        )

    def reset(self) -> None:
        self.state = None

    def _marks_copy_list(self, n: int) -> CopyList:
        """The copies of every peer's first `n` marks, in staggered order."""
        cached = self.marks_list
        if cached is not None and cached[0] == n:
            return cached[1]
        world, rank = self.world_size, self.rank
        copies = CopyList(
            [(*self.marks_copies[(rank + step) % world], n) for step in range(1, world)]
        )
        self.marks_list = (n, copies)
        return copies

    def build(
        self,
        group_id: int,
        position: int,
        topk: torch.Tensor,
        topk_length: torch.Tensor,
        shard_rows: int,
        head_rows: int,
        barrier: Callable[[], None],
        stream: torch.cuda.Stream,
    ) -> GroupState:
        """Union, slot numbering and remapped top-k of one group; on `stream`.

        `topk`/`topk_length` are the converted top-k of the layer that builds.
        Every rank builds at the same positions (agreed in the plan), so the epochs
        of all ranks advance together.
        """
        world, n = self.world_size, self.world_size * head_rows
        tokens = topk.shape[0]
        self.epoch = epoch = self.epoch % EPOCHS + 1
        if epoch == 1:
            # Tags of EPOCHS builds ago would match again. Peers finished copying
            # this rank's marks before the last HEAD_READY it passed.
            self.marks.zero_()
        handle = stream.cuda_stream
        launches = self.launches
        assert launches is not None
        # The layer's converted chunk of the shared top-k buffer.
        assert topk.dtype == torch.int32 and topk.is_contiguous()
        assert topk.shape[1] == self.topk_width
        p = topk.data_ptr()
        if tokens:
            mark, slots = launches.mark, launches.mark.slots
            index = mark.index
            slots[index["p_ptr"]] = p
            slots[index["shard_rows"]] = shard_rows
            slots[index["head_rows"]] = head_rows
            slots[index["epoch"]] = epoch
            mark(handle, tokens, self.mark_columns)
        barrier()
        # In staggered order, submitted together.
        self.overlap.driver.memcpy_list(self._marks_copy_list(n), handle)
        blocks = triton.cdiv(n, SCAN_BLOCK)
        launches.count.update(n=n, epoch=epoch)(handle, blocks)
        launches.scan.update(n=n, head_rows=head_rows)(handle, blocks)
        if tokens:
            launches.remap.update(p_ptr=p, shard_rows=shard_rows, head_rows=head_rows)
            launches.remap(handle, tokens, self.mark_columns)
        sizes = self.sizes.get(group_id)
        if sizes is None:
            sizes = self.sizes[group_id] = UnionSizes(world)
        sizes.record(self.ends, stream)
        self.state = GroupState(group_id, tokens, sizes, position, topk_length)
        return self.state
