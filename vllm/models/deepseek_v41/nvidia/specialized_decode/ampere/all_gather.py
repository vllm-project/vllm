# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP all-gathers of the SM80 decode step, in CuTe DSL.

Three kinds of tensor are all-gathered across the TP group in a decode step:

* the row-sharded sparse-attention indexer's top-k: in a step of at least
  two requests, each rank scores and top-k's only its share of the
  indexer's rows (`indexer.py`: whole requests, equal shares, a short or
  empty one padded with -1), then gathers the selected int32 indices
  (`index_topk` = 512 a row) so every rank has all rows;
* the vocabulary-parallel logits of the target and the DSpark draft model
  (`columns.py`): every rank has every row's logits over its slice of the
  vocabulary, 16,160 BF16 values at TP8, and needs the whole vocabulary;
* the Engram layers' embedded hash heads and `wkv` output, split by heads
  and by output columns across the ranks.

vLLM gathers the first through its custom all-gather, which loads every
peer's buffer between two barriers (about 15 us a call on 8x A100, median),
and the others through NCCL's ring, 30 to 170 us a call, each followed by a
copy from rank-major blocks into the layout the caller wants.

`AllGatherRows` pushes instead, through a Lamport mailbox as the SM80
all-reduce does (`all_reduce.py`): every slot holds a sentinel, the int32
`INT32_MIN` (FP32 negative zero), until a peer's store replaces it. A call
gathers `rows` rows of `fragments` 16-byte fragments from every rank into
`out`, in one of two layouts:

* rank-major (the indexer): rank `s`'s rows are rows `s * rows` onward of
  `out`, and this rank's own rows are already in place (the indexer scores
  its share into its slice of `out`);
* side by side (logits, Engram): row `j` of `out` is every rank's row
  `j` in rank order, and this rank's own row is copied in from `local`.

A unit of work is a pair of a row `j` and a hop `h` in `1 .. world - 1`;
for it, each thread of a CTA stores its fragments of local row `j` into the
mailbox of rank `rank + h`, then polls the fragments of row `j` that rank
`rank - h` stores into its own mailbox, writes them to `out` and resets
the slots. Side by side, the units of hop 1 also write this rank's own row
from the fragments they push. No phase waits for anything but its own data,
and nothing is copied afterwards.

A unit on one rank waits only for the same unit on one peer. Units go to
CTAs in increasing order, `ctas` apart, with at most one CTA per SM: the
lowest unfinished unit is then in progress on every rank, so every call
completes (the argument of `all_reduce.py`, which also needs every kernel
beside a call to finish without waiting on it).

Each unit keeps the generation (of three) its next call uses, advanced by
every call that has the unit. All ranks make the same calls with the same row
counts, so a unit's generation agrees across ranks. A peer writes the same
slot again three calls later, which needs this rank's pushes of two calls
later, issued after this call completed. The mailbox starts as sentinels and
the generations at zero: no per-launch initialization, CUDA-graph safe.

A sender stores a 32-bit word equal to the sentinel as zero, as vLLM's
Lamport all-gather turns FP32 negative zero into positive zero; the
indexer's rows (indices of at least -1) never hold it, and in BF16 rows it
is a positive zero followed by a negative one, which becomes a positive zero
(the sign of a zero changes no logit's probability and no product). Side by
side, this rank's own row is written the same way, so every rank's `out`
is bitwise the same. Every other value arrives unchanged.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    LAMPORT_GENERATIONS,
    NEGATIVE_ZERO_F32_BITS,
    current_cu_stream,
    fragment_is_dirty,
    store_global_u32x4,
)
from vllm.platforms import current_platform

from .all_reduce import FRAGMENT_BYTES, load_fragment, load_u64
from .codecs import load_s32, store_u32

# int32 values per fragment.
WORDS = 4
# The most threads a CTA has; a thread takes every MAX_THREADS-th fragment of
# a longer row.
MAX_THREADS = 512


@cute.jit
def sanitize_sentinel(word: Uint32) -> Uint32:
    """`word`, or zero if it equals the sentinel."""
    if word == Uint32(NEGATIVE_ZERO_F32_BITS):
        word = Uint32(0)
    return word


@cute.jit
def poll(address: Int64, words: cute.Tensor):
    """Wait until the slot at `address` holds no sentinel, read it into
    `words` and reset it to the sentinel."""
    fragment = load_fragment(address, volatile=True)
    for i in cutlass.range_constexpr(WORDS):
        words[i] = fragment[i]
    dirty = fragment_is_dirty(words)
    while dirty:
        fragment = load_fragment(address, volatile=True)
        for i in cutlass.range_constexpr(WORDS):
            words[i] = fragment[i]
        dirty = fragment_is_dirty(words)
    sentinel = Uint32(NEGATIVE_ZERO_F32_BITS)
    store_global_u32x4(address, [sentinel] * WORDS, volatile=True)


class AllGatherRows:
    """The push-based Lamport all-gather of rows of 16-byte fragments; see
    the module docstring.

    `world` ranks, rows of `fragments` fragments, at most `capacity` rows
    a rank, gathered rank-major or `side_by_side`.
    """

    def __init__(self, fragments: int, world: int, capacity: int, side_by_side: bool):
        self.world = world
        self.capacity = capacity
        self.fragments = fragments
        self.side_by_side = side_by_side
        # A thread takes fragments tid, tid + threads, ...
        self.threads = min(fragments, MAX_THREADS)
        self.passes = -(-fragments // self.threads)

    def mailbox_bytes(self) -> int:
        """Bytes of a rank's mailbox: a slot per generation, source rank and
        row."""
        return (
            LAMPORT_GENERATIONS
            * self.world
            * self.capacity
            * self.fragments
            * FRAGMENT_BYTES
        )

    @cute.jit
    def __call__(
        self,
        local: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(local, out, peers, generations, rank, rows).launch(
            grid=(ctas, 1, 1), block=[self.threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        local: Int64,
        out: Int64,
        peers: Int64,
        generations: Int64,
        rank: Int32,
        rows: Int32,
    ):
        world = self.world
        tid, _, _ = cute.arch.thread_idx()
        unit, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        mailbox = load_u64(peers + Int64(rank) * 8)
        # Units in increasing order, ``ctas`` apart (see the module docstring).
        while unit < rows * (world - 1):
            row = unit // (world - 1)
            hop = unit % (world - 1) + 1
            generation_address = generations + Int64(unit) * 4
            generation = load_s32(generation_address)
            destination = load_u64(peers + Int64((rank + hop) % world) * 8)
            # Every fragment is pushed before the first poll.
            for i in cutlass.range_constexpr(self.passes):
                fragment = tid + i * self.threads
                if fragment < self.fragments:
                    self.push(
                        local,
                        out,
                        destination,
                        generation,
                        rank,
                        row,
                        rows,
                        hop,
                        fragment,
                    )
            source = (rank + world - hop) % world
            for i in cutlass.range_constexpr(self.passes):
                fragment = tid + i * self.threads
                if fragment < self.fragments:
                    self.pull(out, mailbox, generation, source, row, rows, fragment)
            # Every thread has read the generation; advance it for the next call.
            cute.arch.barrier()
            if tid == 0:
                store_u32(
                    generation_address,
                    Uint32((generation + 1) % LAMPORT_GENERATIONS),
                )
            unit += ctas

    @cute.jit
    def push(
        self,
        local: Int64,
        out: Int64,
        destination: Int64,
        generation: Int32,
        rank: Int32,
        row: Int32,
        rows: Int32,
        hop: Int32,
        fragment: Int32,
    ):
        """Store a fragment of local row `row` into the destination's
        mailbox (and, side by side at hop 1, into this rank's `out`)."""
        words = load_fragment(
            local + (Int64(row) * self.fragments + fragment) * FRAGMENT_BYTES
        )
        clean = [sanitize_sentinel(word) for word in words]
        store_global_u32x4(
            destination + self.slot(generation, rank, row, fragment),
            clean,
            volatile=True,
        )
        if cutlass.const_expr(self.side_by_side):  # noqa: SIM102
            if hop == 1:
                store_global_u32x4(
                    out + self.out_offset(rank, row, rows, fragment), clean
                )

    @cute.jit
    def pull(
        self,
        out: Int64,
        mailbox: Int64,
        generation: Int32,
        source: Int32,
        row: Int32,
        rows: Int32,
        fragment: Int32,
    ):
        """Wait for the source's fragment of row `row`, write it to `out`
        and reset its slot."""
        words = cute.make_rmem_tensor(cute.make_layout(WORDS), Uint32)
        poll(mailbox + self.slot(generation, source, row, fragment), words)
        store_global_u32x4(
            out + self.out_offset(source, row, rows, fragment),
            [words[i] for i in range(WORDS)],
        )

    @cute.jit
    def slot(
        self, generation: Int32, source: Int32, row: Int32, fragment: Int32
    ) -> Int64:
        """Byte offset of a slot's fragment in a mailbox."""
        return (
            ((Int64(generation) * self.world + source) * self.capacity + row)
            * self.fragments
            + fragment
        ) * FRAGMENT_BYTES

    @cute.jit
    def out_offset(
        self, source: Int32, row: Int32, rows: Int32, fragment: Int32
    ) -> Int64:
        """Byte offset of rank `source`'s fragment of row `row` in
        `out`."""
        if cutlass.const_expr(self.side_by_side):
            index = (Int64(row) * self.world + source) * self.fragments + fragment
        else:
            index = (Int64(source) * rows + row) * self.fragments + fragment
        return index * FRAGMENT_BYTES


@cache
def _kernel(
    device_index: int, fragments: int, world: int, capacity: int, side_by_side: bool
):
    """`AllGatherRows` compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            AllGatherRows(fragments, world, capacity, side_by_side),
            *(Int64(0),) * 4,
            Int32(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


class AmpereAllGather:
    """The mailbox and per-unit generations of `AllGatherRows` for one TP
    group and row size; construction is collective over the group.

    Args:
        group: The TP group's device process group.
        row_bytes: The bytes of a row a rank contributes, a multiple of 16.
        capacity: The most rows a rank contributes to a call.
        device: This rank's device.
        side_by_side: Lay the ranks' rows out side by side (row `j` of the
            output is every rank's row `j`), else rank-major.

    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        row_bytes: int,
        capacity: int,
        device: torch.device,
        side_by_side: bool = False,
    ):
        assert row_bytes % FRAGMENT_BYTES == 0
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.row_bytes = row_bytes
        self.capacity = capacity
        self.side_by_side = side_by_side
        fragments = row_bytes // FRAGMENT_BYTES
        # The most CTAs a call launches, one per SM (see the module docstring).
        self.ctas = current_platform.num_compute_units(device.index)
        # Compiled here: the first call may come while a CUDA graph is being
        # captured.
        self.kernel = _kernel(
            device.index, fragments, self.world, capacity, side_by_side
        )
        nbytes = AllGatherRows(
            fragments, self.world, capacity, side_by_side
        ).mailbox_bytes()
        self.mailbox = symm_mem.empty(nbytes // 4, dtype=torch.int32, device=device)
        handle = symm_mem.rendezvous(self.mailbox, group)
        self.peers = torch.tensor(handle.buffer_ptrs, dtype=torch.int64, device=device)
        # Every slot starts as the sentinel.
        self.mailbox.fill_(torch.iinfo(torch.int32).min)
        self.generations = torch.zeros(
            capacity * (self.world - 1), dtype=torch.int32, device=device
        )
        torch.accelerator.synchronize()
        dist.barrier(group=group)

    def supports(self, local: torch.Tensor, out: torch.Tensor) -> bool:
        """Whether a call takes `local` into `out`: contiguous 16-byte
        aligned rows of `row_bytes`, at most `capacity` local rows and
        `out` holding every rank's; rank-major, `local` must be this
        rank's slice of `out`."""
        rows = local.shape[0] if local.dim() == 2 else 0
        if not (
            0 < rows <= self.capacity
            and out.dim() == 2
            and local.dtype == out.dtype
            and local.shape[1] * local.element_size() == self.row_bytes
            and local.is_contiguous()
            and out.is_contiguous()
            and local.data_ptr() % FRAGMENT_BYTES == 0
            and out.data_ptr() % FRAGMENT_BYTES == 0
        ):
            return False
        if self.side_by_side:
            return out.shape == (rows, self.world * local.shape[1])
        return (
            out.shape == (self.world * rows, local.shape[1])
            and local.data_ptr() == out[self.rank * rows].data_ptr()
        )

    def __call__(self, local: torch.Tensor, out: torch.Tensor) -> None:
        """Gather every rank's rows of `local` into `out`; rank-major,
        `local` is this rank's slice of `out` (left as it is)."""
        assert self.supports(local, out)
        rows = local.shape[0]
        self.kernel(
            Int64(local.data_ptr()),
            Int64(out.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(self.generations.data_ptr()),
            Int32(self.rank),
            Int32(rows),
            Int32(min(rows * (self.world - 1), self.ctas)),
            current_cu_stream(),
        )

    def gather_columns(self, local: torch.Tensor) -> torch.Tensor | None:
        """`local`'s all-gather along its last dimension, `[rows, world *
        columns]` (side by side), or None when a call does not take it."""
        assert self.side_by_side
        out = local.new_empty(local.shape[0], self.world * local.shape[-1])
        if not self.supports(local, out):
            return None
        self(local, out)
        return out
