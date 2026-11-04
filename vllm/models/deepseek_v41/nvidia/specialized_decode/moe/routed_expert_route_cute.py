# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing record of the DeepSeek-V4.1 low-batch routed-expert GEMMs in CuTe DSL.

The routed-expert GEMMs (``routed_expert_gemm_cute.py``) multiply token tiles of
at most eight rows by the weights of one expert each. The top-k expert ids of
up to 192 rows become the expert-major, tile-padded row permutation and the
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
      [4 + 2C + 8 M, + C)      tile -> GEMM1 items done (the ready words), zero

  where ``M`` is :func:`max_tiles` (the most tiles any routing of the call's
  slots needs) and ``C`` is ``M`` rounded up to a multiple of four. GEMM1
  counts up the ready words and GEMM2 waits on them
  (``routed_expert_gemm_cute.py``), so a record serves one GEMM1 and GEMM2
  pair.

The ids come from a top-k selection, so the valid ids of one row are distinct:
an expert is chosen by at most ``rows`` slots, and a slot's rank is the number
of earlier rows that chose the expert (a row mask per expert, one 32-bit word
per 32 rows). Up to eight rows every chosen expert owns exactly one tile, so
its first tile is the number of distinct selected experts below it (a
presence bitmap). Above eight rows an expert chosen by ``n`` rows owns
``ceil(n / 8)`` tiles: level ``l`` of the bitmap marks the experts chosen by
more than ``8 l`` rows, and the first tile is the sum over levels of the
marked experts below it. Either way the record takes a few population
counts, no scans. Above two levels, one warp first sums each bitmap word's
counts over the levels and prefixes them over the words, so a slot reads its
expert's word's prefix instead of summing every level's words.

:class:`RoutedExpertRoute` builds the record from precomputed ids in one CTA
(64 threads per level, at most 1,024; above 16 levels a thread takes two
slots); the fused front builds it with the same helpers.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32

TILE = 8
MAX_EXPERTS = 384
WORDS = MAX_EXPERTS // 32
MAX_THREADS = 1024
# Above this many levels, build_record prefixes the bitmap words' counts.
PREFIX_LEVELS = 2


def levels_for(max_rows: int) -> int:
    """Bitmap levels of the record build for calls of up to ``max_rows`` rows:
    the most tiles one expert can own."""
    return (max_rows + TILE - 1) // TILE


def mask_words_for(levels: int) -> int:
    """32-bit words of an expert's row mask for ``8 * levels`` rows."""
    return (levels * TILE + 31) // 32


def present_words(levels: int) -> int:
    """32-bit words of the ``present`` scratch: the bitmap's ``levels`` levels,
    then, above ``PREFIX_LEVELS`` levels, the word prefix of their counts
    (``WORDS + 1`` words, the last the total)."""
    if levels > PREFIX_LEVELS:
        return levels * WORDS + WORDS + 1
    return levels * WORDS


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
    return 4 + 3 * tile_words(tiles) + TILE * tiles


@cute.jit
def clear_scratch(
    present: cute.Tensor,
    rows_of: cute.Tensor,
    thread: Int32,
    threads: cutlass.Constexpr[int],
    levels: cutlass.Constexpr[int] = 1,
):
    """Zero the expert bitmap's ``levels`` levels and the per-expert row
    masks (not the word prefix, which :func:`build_record` writes whole); a
    block barrier must separate it from :func:`build_record`."""
    for word in cutlass.range(
        thread, MAX_EXPERTS * mask_words_for(levels), threads, unroll=1
    ):
        rows_of[word] = Uint32(0)
        if word < levels * WORDS:
            present[word] = Uint32(0)


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
def slot_expert(ids: cute.Tensor, slot: Int32, slots: Int32, experts: Int32) -> Int32:
    """Slot ``slot``'s expert: its id, or -1 past the call's slots or for an
    id outside ``[0, experts)``."""
    expert = Int32(-1)
    if slot < slots:
        expert = Int32(ids[slot])
        if expert < 0 or expert >= experts:
            expert = Int32(-1)
    return expert


@cute.jit
def mark_slot(
    present: cute.Tensor,
    rows_of: cute.Tensor,
    expert: Int32,
    row: Int32,
    mask_words: cutlass.Constexpr[int],
):
    """Mark ``expert`` in the bitmap's level 0 and ``row`` in its row mask."""
    if expert >= 0:
        cute.arch.atomic_or(
            present.iterator + (expert >> 5),
            Uint32(1) << Uint32(expert & 31),
            scope="cta",
        )
        cute.arch.atomic_or(
            rows_of.iterator + expert * mask_words + (row >> 5),
            Uint32(1) << Uint32(row & 31),
            scope="cta",
        )


@cute.jit
def count_and_rank(
    rows_of: cute.Tensor,
    expert: Int32,
    row: Int32,
    mask_words: cutlass.Constexpr[int],
):
    """The rows that chose ``expert``, and those of them before ``row`` (the
    slot's rank)."""
    count = Int32(0)
    rank = Int32(0)
    if expert >= 0:
        earlier = (Uint32(1) << Uint32(row & 31)) - Uint32(1)
        for word in cutlass.range_constexpr(mask_words):
            chosen_by = rows_of[expert * mask_words + word]
            count += Int32(cute.arch.popc(chosen_by))
            if word < (row >> 5):
                rank += Int32(cute.arch.popc(chosen_by))
            if word == (row >> 5):
                rank += Int32(cute.arch.popc(chosen_by & earlier))
    return count, rank


@cute.jit
def mark_level(present: cute.Tensor, expert: Int32, rank: Int32):
    """Level ``l`` marks the experts chosen by more than ``8 l`` rows; the
    slot of rank ``8 l`` sets it."""
    if expert >= 0 and rank >= TILE and rank % TILE == 0:
        cute.arch.atomic_or(
            present.iterator + (rank // TILE) * WORDS + (expert >> 5),
            Uint32(1) << Uint32(expert & 31),
            scope="cta",
        )


@cute.jit
def write_slot(
    present: cute.Tensor,
    permuted: cute.Tensor,
    record: cute.Tensor,
    slot: Int32,
    slots: Int32,
    words: Int32,
    expert: Int32,
    row: Int32,
    count: Int32,
    rank: Int32,
    levels: cutlass.Constexpr[int],
):
    """Store slot ``slot``'s permuted row, its token and, for the first slot
    of each of its expert's tiles, the tile's expert and row limit; slot 0
    also stores the tile count."""
    if slot < slots:
        prefix = levels * WORDS
        tiles = Int32(0)
        tile = Int32(0)
        if cutlass.const_expr(levels > PREFIX_LEVELS):
            tiles = Int32(present[prefix + WORDS])
            if expert >= 0:
                tile = Int32(present[prefix + (expert >> 5)])
        else:
            for level in cutlass.range_constexpr(levels):
                for word in cutlass.range_constexpr(WORDS):
                    bits = Int32(cute.arch.popc(present[level * WORDS + word]))
                    tiles += bits
                    if word < (expert >> 5):  # never for expert -1
                        tile += bits
        if slot == 0:
            record[0] = tiles * TILE
            record[1] = tiles
        if expert < 0:
            permuted[slot] = Int32(-1)
        else:
            below = (Uint32(1) << Uint32(expert & 31)) - Uint32(1)
            for level in cutlass.range_constexpr(levels):
                tile += Int32(
                    cute.arch.popc(present[level * WORDS + (expert >> 5)] & below)
                )
            first = tile * TILE
            permuted[slot] = first + rank
            record[4 + 2 * words + first + rank] = row
            if rank % TILE == 0:
                # The expert's tile holding ranks rank .. rank + 7.
                own = tile + rank // TILE
                record[4 + own] = expert
                limit = first + count
                if count - rank > TILE:
                    limit = first + rank + TILE
                record[4 + words + own] = limit


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
    threads: cutlass.Constexpr[int],
    levels: cutlass.Constexpr[int] = 1,
):
    """Build the permutation and the record from ``ids`` (visible to every
    thread of the block) into a zeroed record. Every thread of the block's
    ``threads`` calls it (it synchronizes the block up to three times); thread
    ``t`` handles slots ``t``, ``t + threads``, ... below ``rows * top_k``.
    ``experts``: a multiple of 32, at most 384; ``rows <= 8 * levels``,
    ``top_k <= 8``; ``present`` holds ``present_words(levels)`` words and
    ``rows_of`` ``mask_words_for(levels)`` words per expert."""
    mask_words = mask_words_for(levels)
    # Slots a thread handles: up to 8 rows of 8 slots per level.
    per_thread = (64 * levels + threads - 1) // threads
    slots = rows * top_k
    words = tile_words(max_tiles(rows, top_k, experts))
    owned = []
    for j in cutlass.range_constexpr(per_thread):
        slot = thread + j * threads
        expert = slot_expert(ids, slot, slots, experts)
        row = slot // top_k
        mark_slot(present, rows_of, expert, row, mask_words)
        owned.append((slot, expert, row))
    cute.arch.sync_threads()
    counted = []
    for j in cutlass.range_constexpr(per_thread):
        slot, expert, row = owned[j]
        count, rank = count_and_rank(rows_of, expert, row, mask_words)
        if cutlass.const_expr(levels > 1):
            mark_level(present, expert, rank)
        counted.append((slot, expert, row, count, rank))
    if cutlass.const_expr(levels > 1):
        cute.arch.sync_threads()
    if cutlass.const_expr(levels > PREFIX_LEVELS):
        prefix = levels * WORDS
        # Warp 0: lane w < WORDS sums word w's counts over the levels; word
        # w's prefix is the sum of the lanes below, the total at WORDS.
        if thread < 32:
            column = Int32(0)
            if thread < WORDS:
                for level in cutlass.range_constexpr(levels):
                    column += Int32(cute.arch.popc(present[level * WORDS + thread]))
            words_below = Int32(0)
            for word in cutlass.range_constexpr(WORDS):
                column_of_word = cute.arch.shuffle_sync(column, word)
                if word < thread:
                    words_below += column_of_word
            if thread <= WORDS:
                present[prefix + thread] = Uint32(words_below)
        cute.arch.sync_threads()
    for j in cutlass.range_constexpr(per_thread):
        slot, expert, row, count, rank = counted[j]
        write_slot(
            present,
            permuted,
            record,
            slot,
            slots,
            words,
            expert,
            row,
            count,
            rank,
            levels,
        )


class RoutedExpertRoute:
    """The route kernel: the record from precomputed top-k ids (int32), for
    calls of up to ``8 * levels`` rows.

    PDL: dependents may launch at once (the trigger is the first instruction),
    so GEMM1's set-up overlaps this kernel; GEMM1 reads the outputs only after
    its own grid-dependency wait. Every global access here follows this
    kernel's own wait.
    """

    def __init__(self, levels: int = 1) -> None:
        self.levels = levels
        # 8 rows of up to 8 slots per level, at most 1,024 threads (two slots a
        # thread above 16 levels).
        self.threads = min(64 * levels, MAX_THREADS)

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
        """``ids``/``permuted``: int32 [rows, top_k]; rows in
        [1, 8 * levels], ``rows * top_k <= 64 * levels``."""
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
            grid=(1, 1, 1), block=[self.threads, 1, 1], stream=stream, use_pdl=True
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
        """One CTA of ``min(64 * levels, 1024)`` threads."""
        cute.arch.griddepcontrol_launch_dependents()
        bitmap_words = present_words(self.levels)
        mask_size = MAX_EXPERTS * mask_words_for(self.levels)

        @cute.struct
        class Scratch:
            present: cute.struct.MemRange[Uint32, bitmap_words]
            rows_of: cute.struct.MemRange[Uint32, mask_size]

        scratch = cutlass.utils.SmemAllocator().allocate(Scratch)
        present = cute.make_tensor(
            scratch.present.data_ptr(), cute.make_layout(bitmap_words)
        )
        rows_of = cute.make_tensor(
            scratch.rows_of.data_ptr(), cute.make_layout(mask_size)
        )
        thread, _, _ = cute.arch.thread_idx()
        clear_scratch(present, rows_of, thread, self.threads, self.levels)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        zero_record(record, rows, top_k, experts, thread, self.threads)
        build_record(
            present,
            rows_of,
            ids,
            permuted,
            record,
            rows,
            top_k,
            experts,
            thread,
            self.threads,
            self.levels,
        )
