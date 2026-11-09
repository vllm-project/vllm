# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 indexer's decode top-k on SM80, in CuTe DSL.

Each decode step, every indexer layer keeps the `k` best of a row's
`lens[row]` FP32 scores (the rest of the row, up to the CUDA-graph-static
width `max_model_len`, is never read). The candidate source layer also
keeps its row's best key blocks: the top `k` blocks of `group` positions,
a block scoring the maximum of its positions and the newest block pinned on
top. `DecodeTopK` does both, one CTA of 1024 threads per row, so the rows
of a step run side by side and a CTA reads only its row's `lens` scores.

Selection is an exact radix select on the order-preserving 32-bit key of
each score (sign-magnitude FP32 bits made unsigned-comparable):

1. three digit passes (11, 11 and 10 bits, high first) histogram, in shared
   memory, the keys that match the digits found so far, and pick the digit
   holding the `k`-th best key; a pass whose digit holds exactly the keys
   still needed ends the search early;
2. the found prefix splits the row into keys above it (all taken), keys
   matching it (the threshold ties, `need` of them taken) and the rest;
3. each thread owns a contiguous run of the row, counts its taken and tied
   keys, and two block-wide prefix sums give it the run's output offset and
   how many of its ties to take: ties go to the lowest positions.

The first digit pass reads the CTA's elements from global memory (for blocks,
the maximum of their `group` scores) and keeps their keys in shared memory,
where the later passes and the output pass read them: a CTA of the first two
stages selects from at most one chunk of elements (at most 64 KB of keys for
the model's `k`); the last stage reads the chunks' picks from global. A
decode step's rows are few, so these CTAs run one per row or chunk on a
mostly idle GPU, and each re-read of the row from L2 would cost a whole pass
of latency.

The output is the `k` chosen positions (or block indices) in ascending
order, `-1` padded when the row has at most `k` elements (then all of
them, `0 .. n - 1`). It is the same set as any exact top-k of the row; only
threshold ties are resolved, toward lower positions, where vLLM's shared
top-k kernels take any of them.

A long row would leave one CTA streaming it several times, so logits wider
than one chunk (`CHUNK` scores, or `4 k` blocks if more) select in two
launches: CTA `(c, row)` keeps the `k` best of the row's chunk `c` (their
keys and positions, in ascending order) in a workspace, then one CTA per row
selects the `k` best of those picks. An element of the row's top `k`
(ties to the lowest positions) is also among its chunk's top `k`, and the
picks are in position order, so the result is the same. A row within one
chunk gets its answer from the first launch.

A row of compact candidate scores (`mqa_logits.candidate_mqa_logits`)
selects the same way and writes each chosen column's key position, read from
the row's candidate blocks. Its columns at or past the row's key length score
`-inf`, so a row of at most `k` keys takes them all (`0 .. len - 1`).
When the blocks are listed in ascending order (the SM80 candidate source),
the positions come out ascending and ties go to the lowest positions: the
same keys the dense, masked row selects. In any other order (vLLM's
candidate selection), the output is the same set up to threshold ties, not
sorted.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.models.common.ops.moe_tail_fusion.primitives import (
    current_cu_stream,
    load_global_u32x4,
)

from .codecs import load_s32

THREADS = 1024
WARPS = THREADS // 32
# The three radix digits: (shift, bits), high digit first.
DIGITS = ((21, 11), (10, 11), (0, 10))
BINS = 1 << 11
# The key of a pinned (newest) block: above every score's key.
PINNED = 0xFFFFFFFF
# Scores per CTA of a long row's first launch.
CHUNK = 16384


@cute.jit
def _order_key(bits: Uint32) -> Uint32:
    """The unsigned key of FP32 bits that orders like the scores: negative
    values flipped whole, positive ones with the sign bit set."""
    sign = Uint32(Int32(bits) >> 31)
    return bits ^ (sign | Uint32(0x80000000))


def chunk_elements(k: int, group: int) -> int:
    """Elements (scores or blocks) per chunk of a long row."""
    return max(CHUNK // group, 4 * k)


def workspace_bytes(width: int, k: int, group: int = 1) -> int:
    """Bytes per row of the two-launch workspace for logits `width` wide
    (the chunks' picked keys and positions), 0 when a row fits one chunk."""
    elements = (width + group - 1) // group
    size = chunk_elements(k, group)
    if elements <= size:
        return 0
    return (elements + size - 1) // size * k * 8


class DecodeTopK:
    """Top-`k` of each row's first `lens[row]` scores, or of its blocks of
    `group` scores; see the module docstring. `stage` 0 selects whole
    rows, 1 the chunks of long rows, and 2 among the chunks' picks."""

    def __init__(self, k: int, group: int, remap: int = 0, stage: int = 0):
        assert k % 4 == 0 and (group == 1 or group % 4 == 0)
        assert remap == 0 or (group == 1 and stage == 0)
        assert stage in (0, 1, 2)
        self.k = k
        self.group = group
        # Keys per candidate block of compact rows, or 0 for dense rows.
        self.remap = remap
        self.stage = stage
        # Elements per chunk of the two-launch selection.
        self.chunk = chunk_elements(k, group)
        # Keys a CTA keeps in shared memory: stages 0 and 1 select from at
        # most one chunk, each thread at most ``run`` (a multiple of four)
        # elements; the last stage reads the chunks' picks from global.
        self.cache = 0
        if stage < 2:
            self.cache = -(-self.chunk // (4 * THREADS)) * 4 * THREADS

    @cute.jit
    def __call__(
        self,
        logits: cute.Pointer,
        lens: cute.Pointer,
        out: cute.Pointer,
        candidates: Int64,
        key_lens: Int64,
        picked_keys: cute.Pointer,
        picked: cute.Pointer,
        rows: Int32,
        chunks: Int32,
        logits_stride: Int32,
        out_stride: Int32,
        width: Int32,
        candidate_stride: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            hist: cute.struct.MemRange[Int32, BINS]
            sums: cute.struct.MemRange[Int32, 2 * WARPS]
            state: cute.struct.MemRange[Int32, 4]
            keys: cute.struct.Align[
                cute.struct.MemRange[Uint32, max(self.cache, 4)], 16
            ]

        grid = (rows, 1, 1)
        if cutlass.const_expr(self.stage == 1):
            grid = (chunks, rows, 1)
        self.kernel(
            logits,
            lens,
            out,
            candidates,
            key_lens,
            picked_keys,
            picked,
            chunks,
            logits_stride,
            out_stride,
            width,
            candidate_stride,
            Storage,
        ).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
        )

    @cute.jit
    def _position(self, listed: Int64, column: Int32) -> Int32:
        """The key position of a column: itself, or for a compact row the key
        of its candidate block (the row's list at `listed`)."""
        position = column
        if cutlass.const_expr(self.remap > 0):
            block = load_s32(listed + Int64(column // self.remap) * 4)
            position = block * self.remap + column % self.remap
        return position

    @cute.jit
    def _block_key(
        self, row: cute.Pointer, block: Int32, n: Int32, positions: Int32
    ) -> Uint32:
        """The key of a block: the best key of its positions below
        `positions`, or `PINNED` for the newest block (`n - 1`)."""
        best = Uint32(0)
        for half in cutlass.range_constexpr(self.group // 4):
            position = block * self.group + half * 4
            if position < positions:
                words = load_global_u32x4(row + position)
                # Read the vector here: an element read inside a nested branch
                # does not compile.
                keys = [_order_key(words[i]) for i in range(4)]
                for i in cutlass.range_constexpr(4):
                    if position + i < positions and keys[i] > best:
                        best = keys[i]
        if block == n - 1:
            best = Uint32(PINNED)
        return best

    @cute.jit
    def _keys(self, row: cute.Pointer, first: Int32, n: Int32, positions: Int32):
        """Keys of elements `first .. first + 3` (`first % 4 == 0` and
        `first < n`); those at or past `n` are garbage for the caller to
        skip. A row's padded width keeps the four-score load in bounds."""
        if cutlass.const_expr(self.group == 1):
            words = load_global_u32x4(row + first)
            return [
                _order_key(words[0]),
                _order_key(words[1]),
                _order_key(words[2]),
                _order_key(words[3]),
            ]
        return [
            self._block_key(row, first, n, positions),
            self._block_key(row, first + 1, n, positions),
            self._block_key(row, first + 2, n, positions),
            self._block_key(row, first + 3, n, positions),
        ]

    @cute.jit
    def _scan(
        self, a: Int32, b: Int32, sums: cute.Tensor, warp: Int32, lane: Int32
    ) -> tuple[Int32, Int32, Int32, Int32]:
        """Block-wide inclusive prefix sums of two per-thread counts, in
        thread order, and their totals."""
        for level in cutlass.range_constexpr(5):
            offset = 1 << level
            up_a = cute.arch.shuffle_sync(a, offset=lane - offset)
            up_b = cute.arch.shuffle_sync(b, offset=lane - offset)
            if lane >= offset:
                a = a + up_a
                b = b + up_b
        if lane == 31:
            sums[warp] = a
            sums[WARPS + warp] = b
        cute.arch.barrier()
        base_a = Int32(0)
        base_b = Int32(0)
        total_a = Int32(0)
        total_b = Int32(0)
        for w in cutlass.range_constexpr(WARPS):
            warp_a = sums[w]
            warp_b = sums[WARPS + w]
            if w < warp:
                base_a = base_a + warp_a
                base_b = base_b + warp_b
            total_a = total_a + warp_a
            total_b = total_b + warp_b
        cute.arch.barrier()
        return a + base_a, b + base_b, total_a, total_b

    @cute.jit
    def _load(
        self,
        row: cute.Pointer,
        picked_keys: cute.Pointer,
        first: Int32,
        n: Int32,
        positions: Int32,
    ):
        """Keys of elements `first .. first + 3`: the row's, or in the last
        stage the chunks' picks."""
        if cutlass.const_expr(self.stage == 2):
            words = load_global_u32x4(picked_keys + first)
            return [
                Uint32(words[0]),
                Uint32(words[1]),
                Uint32(words[2]),
                Uint32(words[3]),
            ]
        return self._keys(row, first, n, positions)

    @cute.jit
    def _read(
        self,
        row: cute.Pointer,
        picked_keys: cute.Pointer,
        cache: cute.Tensor,
        cached_at: Int32,
        first: Int32,
        n: Int32,
        positions: Int32,
        fill: cutlass.Constexpr,
    ):
        """Keys of elements `first .. first + 3` through the CTA's shared
        key cache: `fill` (the first pass) loads them and stores them,
        later passes read them back. A thread's `i`-th element sits at
        `i * THREADS + tid` (`cached_at` for `first`), so a warp's
        accesses fall in distinct banks. Without a cache (the last stage),
        every pass loads them."""
        if cutlass.const_expr(self.cache == 0):
            return self._load(row, picked_keys, first, n, positions)
        if cutlass.const_expr(fill):
            keys = self._load(row, picked_keys, first, n, positions)
            for j in cutlass.range_constexpr(4):
                cache[cached_at + j * THREADS] = keys[j]
            return keys
        return [
            cache[cached_at],
            cache[cached_at + THREADS],
            cache[cached_at + 2 * THREADS],
            cache[cached_at + 3 * THREADS],
        ]

    @cute.kernel
    def kernel(
        self,
        logits: cute.Pointer,
        lens: cute.Pointer,
        out: cute.Pointer,
        candidates: Int64,
        key_lens: Int64,
        picked_keys: cute.Pointer,
        picked: cute.Pointer,
        chunks: Int32,
        logits_stride: Int32,
        out_stride: Int32,
        width: Int32,
        candidate_stride: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, chunk, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.stage == 1):
            chunk, row, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        k = self.k

        positions = cutlass.max(
            cutlass.min(cute.make_tensor(lens + row, cute.make_layout(1))[0], width),
            Int32(0),
        )
        n = positions
        if cutlass.const_expr(self.group > 1):
            n = (positions + self.group - 1) // self.group
        gOut = cute.make_tensor(
            out + Int64(row) * Int64(out_stride), cute.make_layout(k)
        )
        listed = candidates + Int64(row) * Int64(candidate_stride) * 4
        src = logits + Int64(row) * Int64(logits_stride)
        # The row's picks: chunk c's keys and positions at c * k.
        picks = Int64(row) * Int64(chunks) * k
        stored = picked_keys + picks
        gPickKeys = cute.make_tensor(stored, cute.make_layout(chunks * k))
        gPicks = cute.make_tensor(picked + picks, cute.make_layout(chunks * k))
        # This CTA selects from elements [low, low + count): into the output
        # (direct), into its chunk's picks, or (last stage) from the picks.
        size = self.chunk
        low = Int32(0)
        count = n
        direct = True
        if cutlass.const_expr(self.stage == 1):
            low = chunk * size
            count = cutlass.max(cutlass.min(n - low, size), Int32(0))
            direct = n <= size
        if cutlass.const_expr(self.stage == 2):
            direct = False
            # Every chunk but the last has k picks; a row within one chunk
            # has its answer already.
            last = (n + size - 1) // size - 1
            count = last * k + cutlass.min(k, n - last * size)
            if n <= size:
                count = Int32(-1)
        if cutlass.const_expr(self.remap > 0):
            # A compact row of at most k keys takes them all; its columns past
            # them score -inf.
            keys_len = load_s32(key_lens + Int64(row) * 4)
            if keys_len <= k:
                count = Int32(-1)
                for i in cutlass.range(tid, k, THREADS, unroll=1):
                    chosen = Int32(-1)
                    if i < keys_len:
                        chosen = i
                    gOut[i] = chosen
        if count >= 0 and count <= k:
            if direct:
                if chunk == 0:
                    for i in cutlass.range(tid, k, THREADS, unroll=1):
                        chosen = Int32(-1)
                        if i < count:
                            chosen = self._position(listed, i)
                        gOut[i] = chosen
            else:
                if cutlass.const_expr(self.stage == 2):
                    for i in cutlass.range(tid, k, THREADS, unroll=1):
                        chosen = Int32(-1)
                        if i < count:
                            chosen = gPicks[i]
                        gOut[i] = chosen
                else:
                    for first in cutlass.range(tid * 4, count, THREADS * 4, unroll=1):
                        keys = self._keys(src, low + first, n, positions)
                        for j in cutlass.range_constexpr(4):
                            if first + j < count:
                                gPickKeys[chunk * k + first + j] = keys[j]
                                gPicks[chunk * k + first + j] = low + first + j
        if count > k:
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(Storage)
            hist = storage.hist.get_tensor(cute.make_layout(BINS))
            sums = storage.sums.get_tensor(cute.make_layout(2 * WARPS))
            state = storage.state.get_tensor(cute.make_layout(4))
            cache = storage.keys.get_tensor(cute.make_layout(max(self.cache, 4)))
            # Thread ``tid`` owns elements [begin, end), a multiple of four long.
            run = ((count + THREADS - 1) // THREADS + 3) // 4 * 4
            begin = low + cutlass.min(tid * run, count)
            end = cutlass.min(begin + run, low + count)
            for i in cutlass.range_constexpr(BINS // THREADS):
                hist[tid + i * THREADS] = 0
            cute.arch.barrier()

            # ---- radix select: the prefix of the k-th key ----
            prefix = Uint32(0)
            mask = Uint32(0)
            need = Int32(k)
            done = Int32(0)
            for d in cutlass.range_constexpr(len(DIGITS)):
                shift, bits = DIGITS[d]
                if done == 0:
                    digit_mask = Uint32((1 << bits) - 1)
                    for first in cutlass.range(begin, end, 4, unroll=1):
                        cached_at = (first - begin) * THREADS + tid
                        keys = self._read(
                            src, stored, cache, cached_at, first, n, positions, d == 0
                        )
                        for j in cutlass.range_constexpr(4):
                            key = keys[j]
                            if first + j < end and (key & mask) == prefix:
                                digit = Int32((key >> shift) & digit_mask)
                                cute.arch.atomic_add(hist.iterator + digit, Int32(1))
                    cute.arch.barrier()
                    # Thread t holds the digits (1 << bits) - 1 - per * t - i,
                    # so the prefix sums count the keys above its digits.
                    per = (1 << bits) // THREADS
                    counts = []
                    for i in cutlass.range_constexpr(per):
                        index = (1 << bits) - 1 - per * tid - i
                        counts.append(hist[index])
                        hist[index] = 0
                    mine = counts[0]
                    if cutlass.const_expr(per == 2):
                        mine = counts[0] + counts[1]
                    inclusive, _, _, _ = self._scan(mine, Int32(0), sums, warp, lane)
                    above = inclusive - mine
                    for i in cutlass.range_constexpr(per):
                        if above < need and need <= above + counts[i]:
                            state[0] = (1 << bits) - 1 - per * tid - i
                            state[1] = need - above
                            state[2] = Int32(counts[i] == need - above)
                        above = above + counts[i]
                    cute.arch.barrier()
                    prefix = prefix | (Uint32(state[0]) << shift)
                    mask = mask | (digit_mask << shift)
                    need = state[1]
                    done = state[2]
                    cute.arch.barrier()

            # ---- write the chosen elements in ascending order ----
            taken = Int32(0)
            tied = Int32(0)
            for first in cutlass.range(begin, end, 4, unroll=1):
                cached_at = (first - begin) * THREADS + tid
                keys = self._read(
                    src, stored, cache, cached_at, first, n, positions, False
                )
                for j in cutlass.range_constexpr(4):
                    key = keys[j] & mask
                    if first + j < end:
                        if key > prefix:
                            taken = taken + 1
                        if key == prefix:
                            tied = tied + 1
            taken_end, tied_end, taken_total, tied_total = self._scan(
                taken, tied, sums, warp, lane
            )
            ties_before = tied_end - tied
            slot = taken_end - taken + cutlass.min(ties_before, need)
            for first in cutlass.range(begin, end, 4, unroll=1):
                cached_at = (first - begin) * THREADS + tid
                keys = self._read(
                    src, stored, cache, cached_at, first, n, positions, False
                )
                for j in cutlass.range_constexpr(4):
                    key = keys[j] & mask
                    if first + j < end:
                        take = key > prefix
                        if key == prefix:
                            take = ties_before < need
                            ties_before = ties_before + 1
                        if take:
                            if cutlass.const_expr(self.stage == 2):
                                gOut[slot] = gPicks[first + j]
                            else:
                                if direct:
                                    gOut[slot] = self._position(listed, first + j)
                                else:
                                    gPickKeys[chunk * k + slot] = keys[j]
                                    gPicks[chunk * k + slot] = first + j
                            slot = slot + 1
            if direct or cutlass.const_expr(self.stage == 2):
                chosen = taken_total + cutlass.min(tied_total, need)
                for i in cutlass.range(chosen + tid, k, THREADS, unroll=1):
                    gOut[i] = -1


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to `tensor`'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


@cache
def _kernel(device_index: int, k: int, group: int, remap: int = 0, stage: int = 0):
    """`DecodeTopK` compiled for a device, `k`, block size, candidate
    block size of compact rows (0: dense rows) and stage."""
    fake = (
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
        Int64(0),
        Int64(0),
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 4, cute.AddressSpace.gmem, assumed_align=4),
    )
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            DecodeTopK(k, group, remap, stage),
            *fake,
            *(Int32(1),) * 6,
            make_fake_stream(),
        )


def decode_topk(
    logits: torch.Tensor,
    lens: torch.Tensor,
    out: torch.Tensor,
    group: int = 1,
    candidates: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    key_lens: torch.Tensor | None = None,
) -> None:
    """Top `out.shape[1]` of each row of `logits` (FP32 `[rows, width]`,
    rows 16-byte aligned) over its first `lens[row]` columns, or over its
    blocks of `group` columns, into `out` (int32, unit column stride) in
    ascending order, `-1` padded. With `candidates` (int32
    `[rows, count]`), the rows are compact candidate scores
    (`mqa_logits.candidate_mqa_logits`, `lens` their compact lengths),
    `key_lens` (int32 `[rows]`) the rows' key lengths, and `out` gets
    the chosen keys' positions; compact rows are at most one chunk wide
    (`chunk_elements(k, 1)` columns)."""
    rows, width = logits.shape
    k = out.shape[1]
    assert out.shape[0] >= rows and out.dtype == torch.int32 and out.stride(1) == 1
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert logits.stride(0) % 4 == 0 and logits.data_ptr() % 16 == 0
    assert lens.dtype == torch.int32 and lens.numel() == rows and lens.is_contiguous()
    remap = 0
    if candidates is not None:
        assert group == 1 and candidate_block_size > 0
        assert candidates.dtype == torch.int32 and candidates.stride(1) == 1
        assert candidates.shape[0] >= rows
        assert width <= candidates.shape[1] * candidate_block_size
        assert key_lens is not None and key_lens.dtype == torch.int32
        assert key_lens.numel() == rows and key_lens.is_contiguous()
        remap = candidate_block_size
    if rows == 0:
        return
    elements = (width + group - 1) // group
    size = chunk_elements(k, group)
    # A compact row selects in one launch, from at most one chunk (the CTA's
    # shared key cache); a wider row would overrun it, so this check stays
    # under ``python -O``.
    if remap and elements > size:
        raise ValueError(
            f"compact rows of {elements} columns exceed the {size}-column chunk"
        )
    stages: tuple[int, ...] = (0,)
    chunks = 1
    picked_keys = picked = out
    if remap == 0 and elements > size:
        # Long rows: each chunk's picks, then the pick among them.
        stages = (1, 2)
        chunks = (elements + size - 1) // size
        picked_keys = torch.empty(
            rows, chunks * k, dtype=torch.int32, device=out.device
        )
        picked = torch.empty_like(picked_keys)
    for stage in stages:
        _kernel(logits.device.index, k, group, remap, stage)(
            _pointer(Float32, logits, 16),
            _pointer(Int32, lens, 4),
            _pointer(Int32, out, 4),
            Int64(candidates.data_ptr() if candidates is not None else 0),
            Int64(key_lens.data_ptr() if key_lens is not None else 0),
            _pointer(Uint32, picked_keys, 16),
            _pointer(Int32, picked, 4),
            Int32(rows),
            Int32(chunks),
            Int32(logits.stride(0)),
            Int32(out.stride(0)),
            Int32(width),
            Int32(candidates.stride(0) if candidates is not None else 0),
            current_cu_stream(),
        )
