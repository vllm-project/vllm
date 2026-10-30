# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing record of the DeepSeek-V4.1 low-batch routed-expert GEMMs in CuTe DSL.

The routed-expert GEMMs (``routed_expert_gemm_cute.py``) multiply token tiles of
at most eight rows by the weights of one expert each. The top-k expert ids of
up to eight rows become the expert-major, tile-padded row permutation and the
per-tile tables both GEMMs read. The result is that of FlashInfer's TRT-LLM-gen
routing kernel with precomputed ids and an 8-token tile:

* local experts (ids in ``[0, experts)``) in ascending id order, each one's
  rows in ascending token order, every expert's segment padded to a multiple
  of the tile (8 rows);
* ``permuted[t, k]`` = the slot's row in that order, -1 for an id outside
  ``[0, experts)``;
* the record (int32, one buffer per call)::

      [0]                      8 * tiles (padded rows)
      [1]                      tiles
      [2, 4)                   zero
      [4, 4 + C)               tile -> local expert
      [4 + C, 4 + 2C)          tile -> row limit min((c + 1) * 8, first + count)
      [4 + 2C, 4 + 2C + 8 M)   padded row -> token (GEMM1 only)

  where ``M`` is :func:`max_tiles` (the most tiles any routing of the call's
  slots needs) and ``C`` is ``M`` rounded up to a multiple of four.

The ids come from a top-k selection, so the valid ids of one row are distinct:
an expert is chosen by at most ``rows <= 8`` slots and owns exactly one tile.
The expert's tile is then the number of distinct selected experts below it (a
presence bitmap), and a slot's rank the number of earlier rows that chose the
expert (a row mask per expert): a few population counts, no scans.

:class:`RoutedExpertRoute` builds the record from precomputed ids in one
64-thread CTA; the fused front builds it with the same helpers.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32

TILE = 8
MAX_EXPERTS = 384
WORDS = MAX_EXPERTS // 32
ROUTE_THREADS = 64  # >= rows * top_k (at most 8 * 8 slots)


@cute.jit
def max_tiles(rows: Int32, top_k: Int32, experts: Int32) -> Int32:
    """The most 8-row token tiles any routing of ``rows * top_k`` slots needs."""
    slots = rows * top_k
    filled = slots
    if experts < slots:
        filled = experts
    return filled + (slots - filled) // TILE


@cute.jit
def tile_words(tiles: Int32) -> Int32:
    """``C``: ``tiles`` rounded up to a multiple of four."""
    return (tiles + 3) // 4 * 4


@cute.jit
def record_words(rows: Int32, top_k: Int32, experts: Int32) -> Int32:
    """int32 length of the record."""
    tiles = max_tiles(rows, top_k, experts)
    return 4 + 2 * tile_words(tiles) + TILE * tiles


@cute.jit
def clear_scratch(
    present: cute.Tensor,
    rows_of: cute.Tensor,
    thread: Int32,
    threads: cutlass.Constexpr[int],
):
    """Zero the expert bitmap and the per-expert row masks; a block barrier
    must separate it from :func:`build_record`."""
    for expert in cutlass.range(thread, MAX_EXPERTS, threads, unroll=1):
        rows_of[expert] = Uint32(0)
        if expert < WORDS:
            present[expert] = Uint32(0)


@cute.jit
def zero_record(
    record: cute.Tensor,
    rows: Int32,
    top_k: Int32,
    experts: Int32,
    thread: Int32,
    threads: cutlass.Constexpr[int],
):
    """Zero the record; :func:`build_record`'s block barrier orders it before
    that function's stores."""
    for word in cutlass.range(
        thread, record_words(rows, top_k, experts), threads, unroll=1
    ):
        record[word] = Int32(0)


@cute.jit
def build_record(
    present: cute.Tensor,
    rows_of: cute.Tensor,
    ids: cute.Tensor,
    permuted: cute.Tensor,
    record: cute.Tensor,
    rows: Int32,
    top_k: Int32,
    experts: Int32,
    thread: Int32,
):
    """Build the permutation and the record from ``ids`` (visible to every
    thread of the block) into a zeroed record. Every thread of the block calls
    it (it synchronizes the block once); thread ``s < rows * top_k`` handles
    slot ``s``. ``experts``: a multiple of 32, at most 384; ``rows <= 8``."""
    slots = rows * top_k
    words = tile_words(max_tiles(rows, top_k, experts))
    row = thread // top_k
    expert = Int32(-1)
    if thread < slots:
        expert = Int32(ids[thread])
        if expert < 0 or expert >= experts:
            expert = Int32(-1)
    if expert >= 0:
        cute.arch.atomic_or(
            present.iterator + (expert >> 5),
            Uint32(1) << Uint32(expert & 31),
            scope="cta",
        )
        cute.arch.atomic_or(
            rows_of.iterator + expert, Uint32(1) << Uint32(row), scope="cta"
        )
    cute.arch.sync_threads()
    if thread < slots:
        tiles = Int32(0)
        tile = Int32(0)
        for word in cutlass.range_constexpr(WORDS):
            bits = Int32(cute.arch.popc(present[word]))
            tiles += bits
            if word < (expert >> 5):  # never for expert -1
                tile += bits
        if thread == 0:
            record[0] = tiles * TILE
            record[1] = tiles
        if expert < 0:
            permuted[thread] = Int32(-1)
        else:
            below = (Uint32(1) << Uint32(expert & 31)) - Uint32(1)
            tile += Int32(cute.arch.popc(present[expert >> 5] & below))
            chosen_by = rows_of[expert]
            count = Int32(cute.arch.popc(chosen_by))
            earlier = (Uint32(1) << Uint32(row)) - Uint32(1)
            rank = Int32(cute.arch.popc(chosen_by & earlier))
            first = tile * TILE
            permuted[thread] = first + rank
            record[4 + 2 * words + first + rank] = row
            if rank == 0:
                record[4 + tile] = expert
                limit = first + count
                if count > TILE:
                    limit = first + TILE
                record[4 + words + tile] = limit


class RoutedExpertRoute:
    """The route kernel: the record from precomputed top-k ids (int32).

    PDL: dependents may launch at once (the trigger is the first instruction),
    so GEMM1's set-up overlaps this kernel; GEMM1 reads the outputs only after
    its own grid-dependency wait. Every global access here follows this
    kernel's own wait.
    """

    @cute.jit
    def __call__(
        self,
        ids_ptr: cute.Pointer,
        permuted_ptr: cute.Pointer,
        record_ptr: cute.Pointer,
        rows: Int32,
        top_k: Int32,
        experts: Int32,
        stream: cuda.CUstream,
    ):
        """``ids``/``permuted``: int32 [rows, top_k]; rows in [1, 8],
        ``rows * top_k <= 64``."""
        slots = cute.make_layout(rows * top_k)
        self.kernel(
            cute.make_tensor(ids_ptr, slots),
            cute.make_tensor(permuted_ptr, slots),
            cute.make_tensor(
                record_ptr, cute.make_layout(record_words(rows, top_k, experts))
            ),
            rows,
            top_k,
            experts,
        ).launch(
            grid=(1, 1, 1), block=[ROUTE_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        ids: cute.Tensor,
        permuted: cute.Tensor,
        record: cute.Tensor,
        rows: Int32,
        top_k: Int32,
        experts: Int32,
    ):
        """One CTA of :data:`ROUTE_THREADS` threads."""
        cute.arch.griddepcontrol_launch_dependents()

        @cute.struct
        class Scratch:
            present: cute.struct.MemRange[Uint32, WORDS]
            rows_of: cute.struct.MemRange[Uint32, MAX_EXPERTS]

        scratch = cutlass.utils.SmemAllocator().allocate(Scratch)
        present = cute.make_tensor(scratch.present.data_ptr(), cute.make_layout(WORDS))
        rows_of = cute.make_tensor(
            scratch.rows_of.data_ptr(), cute.make_layout(MAX_EXPERTS)
        )
        thread, _, _ = cute.arch.thread_idx()
        clear_scratch(present, rows_of, thread, ROUTE_THREADS)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        zero_record(record, rows, top_k, experts, thread, ROUTE_THREADS)
        build_record(
            present, rows_of, ids, permuted, record, rows, top_k, experts, thread
        )
