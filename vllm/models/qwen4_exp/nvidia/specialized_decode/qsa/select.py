# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode top-512 block select of the QSA indexer, with the index expansion,
in CuTe DSL.

The problem. A QSA attention layer of Qwen4Exp attends, for each query row,
only to the tokens of its 512 best-scoring pooled key blocks (a block pools 4
tokens) plus the open block's causal tail. In a decode step the stock chain
is a Triton score kernel (FP32 logits `[rows, columns]`, row `r` valid
below `visible_blocks[r]`), `cooperative_topk` (or `persistent_topk`
above 64 rows) into `block_indices [rows, 512]`, and a Triton expansion of
those blocks into the packed token list the sparse attention reads. At decode
sizes the top-k and the expansion cost about 11 to 13 us per layer-pass,
mostly latency.

What this module computes. `qsa_select_expand` reads the logits and
writes the packed buffer `out [rows, 2052]` that the expansion writes:

* columns `[0, 2048)`: for each of the row's 512 selected blocks `b`, the
  tokens `4 b .. 4 b + 3` in four consecutive columns;
* columns `2048 .. 2050`: the causal tail of the open block (tokens
  `4 floor((p + 1) / 4) ..  p` of a row at position `p`), -1 padded;
* column `2051`: the row's valid-entry count (2048 plus the tail).

A row with at most 512 visible blocks selects all of them in order, as the
stock top-k does, and its packed row is then exactly the stock expansion's.
For longer rows the selected set is exact: every block whose score exceeds
the 512th largest, then enough of those equal to it, lowest block index first
on the usual path below and in CTA, thread and key order on the other (the
stock top-k takes tied blocks in the order of its atomics). The order of the
blocks in the row is this kernel's own; the stock order is that of its
atomics too.

How. One cluster of `C` CTAs per row. CTA `c` holds every `C`-th group
of `4 T` scores (`T` threads a CTA) in registers as order-preserving
32-bit keys of the FP32 scores; the first group is loaded before the row
length is known. The 512th largest key is found by a radix select whose first
round is shared by the cluster and whose later rounds usually run inside one
CTA:

    every CTA: histogram of its keys' top 11 bits (lanes of a warp with the
    same bin add once)
      └► nonzero bins added into every CTA's totals (distributed shared
         memory) ─ cluster barrier ─ every CTA scans its totals: the bin that
         holds the 512th key, how many keys it holds and how many are needed
         ├► usual case (that bin holds at most 4096 keys): every CTA sends its
         │  keys of that bin, with their block indices, to CTA 0 and its count
         │  of keys above the bin to every CTA ─ cluster barrier ─ every CTA
         │  writes its blocks above the bin; CTA 0 selects the needed keys of
         │  the bin by two more radix rounds over its candidates (CTA
         │  barriers only) and writes them
         └► otherwise: rounds over bits 20-10 and 9-0 with CTA 0's totals and
            a broadcast of each result, then every CTA writes its selected
            blocks after an exchange of counts

Every selected block is written as one 16-byte store of its four tokens.
Cluster barriers order the shared-memory traffic: every CTA has zeroed its
totals and its candidate counter before any peer adds into them, the
additions are complete before the totals are scanned, and the candidates and
exchanged counts are complete before they are read. After the last cluster
barrier a CTA touches only its own shared memory, so no CTA exits while a
peer still uses it. The kernel zeroes its shared memory before its
grid-dependency wait and lets its dependents launch once its keys are loaded
(programmatic dependent launch).
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ._select_ptx import (
    cluster_atom_add,
    cluster_red_add,
    cluster_store_x2,
    cluster_store_x4,
    lanes_below,
    load_global_s32,
    load_global_s64,
    load_global_x4,
    match_any,
    peer_address,
    popc,
    shared_load,
    shared_load_x4,
    shared_red_add,
    shared_store,
    shared_zero_x4,
    shfl_idx,
    shfl_up,
    store_global_s32,
    store_global_x4,
)

TOP_BLOCKS = 512  # blocks selected a row (indexer budget 2048 / ratio 4)
RATIO = 4  # tokens a pooled block
TOKENS = TOP_BLOCKS * RATIO
OUT_WIDTH = TOKENS + RATIO - 1  # token columns of a packed row
PACKED_WIDTH = OUT_WIDTH + 1  # plus the valid-entry count
MAX_BLOCKS = 65536  # the longest row every variant holds (262,144 tokens)

# Radix rounds: the digit's lowest key bit and its bin count.
SHIFTS = (21, 10, 0)
LEVEL_BINS = (2048, 2048, 1024)
LEVEL_OFFSET = (0, 2048, 4096)  # words into a CTA's histogram arrays
HIST_WORDS = 5120
NO_BIN = 0xFFFFFFFF  # the round-0 digit of a lane without a key
CAPACITY = 4096  # candidates CTA 0 gathers: (key, block index) words
CTL_BIN, CTL_REMAIN, CTL_COUNT = 0, 1, 2
CTL_WORDS = 4
SCAN_WORDS = 64  # warp totals of the two scanned values
PUB_WORDS = 32  # (above, equal) counts of up to 16 CTAs
COUNTER_WORDS = 4  # word 0: CTA 0's gathered-candidate count

# (CTAs a row, threads a CTA, 16-byte vectors a thread); each holds MAX_BLOCKS.
VARIANTS = (
    (16, 256, 4),
    (16, 512, 2),
    (8, 256, 8),
    (8, 512, 4),
    (4, 512, 8),
    (4, 1024, 4),
    (2, 1024, 8),
)


class QsaSelect:
    """The decode select (module docstring) for one cluster shape.

    Args:
        cluster: CTAs a row (2 to 16).
        threads: Threads a CTA (256, 512 or 1024).
        vectors: 16-byte vectors of four scores a thread holds.

    """

    def __init__(self, cluster: int, threads: int, vectors: int) -> None:
        assert (cluster, threads, vectors) in VARIANTS
        self.cluster = cluster
        self.threads = threads
        self.warps = threads // 32
        self.vectors = vectors
        self.keys = 4 * vectors
        assert cluster * threads * self.keys >= MAX_BLOCKS

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        row_stride: Int32,
        columns: Int32,
        visible: Int64,
        positions: Int64,
        out: Int64,
        out_stride: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream`: FP32 `logits` rows of `columns` scores
        `row_stride` elements apart, int32 `visible [rows]`, int64
        `positions [rows]`, int32 `out` rows `out_stride` elements
        apart."""
        self.kernel(
            logits, row_stride, columns, visible, positions, out, out_stride
        ).launch(
            grid=(rows * self.cluster, 1, 1),
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    def index_of(self, i: int, rank: Int32, tid: Int32) -> Int32:
        """The block index of this thread's key `i`."""
        group = Int32((i // 4) * self.cluster) + rank
        return (group * self.threads + tid) * 4 + (i % 4)

    @cute.jit
    def scan2(self, a: Uint32, b: Uint32, scan: Int32, tid: Int32):
        """Exclusive CTA prefix sums of `a` and `b` over the threads, and
        their totals (every thread calls it): warp scans, one barrier, then
        every thread adds up the warp totals. Two scans must be separated by
        a barrier after the earlier one's reads; every caller has one."""
        lane = tid & 31
        warp = tid >> 5
        in_a = a
        in_b = b
        for shift in cutlass.range_constexpr(5):
            below_a = shfl_up(in_a, 1 << shift)
            below_b = shfl_up(in_b, 1 << shift)
            if lane >= (1 << shift):
                in_a = in_a + below_a
                in_b = in_b + below_b
        if lane == 31:
            shared_store(scan + warp * 4, in_a)
            shared_store(scan + (32 + warp) * 4, in_b)
        cute.arch.sync_threads()
        first_a = in_a - a
        first_b = in_b - b
        total_a = Uint32(0)
        total_b = Uint32(0)
        for q in cutlass.range_constexpr(self.warps // 4):
            sums_a = shared_load_x4(scan + q * 16)
            sums_b = shared_load_x4(scan + 128 + q * 16)
            for e in cutlass.range_constexpr(4):
                total_a = total_a + sums_a[e]
                total_b = total_b + sums_b[e]
                if warp > 4 * q + e:
                    first_a = first_a + sums_a[e]
                    first_b = first_b + sums_b[e]
        return first_a, first_b, total_a, total_b

    @cute.jit
    def find_local(
        self,
        bins: int,
        remain: Uint32,
        tid: Int32,
        counts_at: Int32,
        ctl: Int32,
        scan: Int32,
    ):
        """The bin of this CTA's histogram at `counts_at` that holds the
        `remain`-th largest key, how many of its keys are still needed and
        how many it holds (every thread of the CTA calls it)."""
        per = bins // self.threads
        top = Int32(bins - 1) - tid * per  # this thread's bins, descending
        counts = []
        for i in cutlass.range_constexpr(per):
            counts.append(shared_load(counts_at + (top - i) * 4))
        mine = Uint32(0)
        for i in cutlass.range_constexpr(per):
            mine = mine + counts[i]
        above, _, _, _ = self.scan2(mine, Uint32(0), scan, tid)
        if (above < remain) & (above + mine >= remain):
            found = Int32(-1)
            still = Uint32(0)
            held = Uint32(0)
            run = above
            for i in cutlass.range_constexpr(per):
                if (found < 0) & (run + counts[i] >= remain):
                    found = top - i
                    still = remain - run
                    held = counts[i]
                run = run + counts[i]
            shared_store(ctl + CTL_BIN * 4, Uint32(found))
            shared_store(ctl + CTL_REMAIN * 4, still)
            shared_store(ctl + CTL_COUNT * 4, held)
        cute.arch.sync_threads()
        return (
            shared_load(ctl + CTL_BIN * 4),
            shared_load(ctl + CTL_REMAIN * 4),
            shared_load(ctl + CTL_COUNT * 4),
        )

    @cute.jit
    def count_top(self, keys, n: Int32, rank: Int32, tid: Int32, hist: Int32) -> None:
        """Count this CTA's keys into the histogram of their top 11 bits; the
        lanes of a warp with the same bin add once (the scores crowd into a
        few bins)."""
        below = lanes_below()
        warp_first = tid & Int32(-32)
        for j in cutlass.range_constexpr(self.vectors):
            group = Int32(j * self.cluster) + rank
            if (group * self.threads + warp_first) * 4 < n:  # warp-uniform
                for e in cutlass.range_constexpr(4):
                    i = 4 * j + e
                    digit = Uint32(NO_BIN)
                    if self.index_of(i, rank, tid) < n:
                        digit = keys[i] >> Uint32(SHIFTS[0])
                    same = match_any(digit)
                    if (digit != Uint32(NO_BIN)) & ((same & below) == Uint32(0)):
                        shared_red_add(hist + Int32(digit) * 4, popc(same))
        cute.arch.sync_threads()

    @cute.jit
    def reduce_all(self, rank: Int32, tid: Int32, hist: Int32, total: Int32) -> None:
        """Add this CTA's nonzero top-bit bins into every CTA's totals, then
        a cluster barrier."""
        for b in cutlass.range_constexpr(LEVEL_BINS[0] // self.threads):
            bin_ = b * self.threads + tid
            count = shared_load(hist + bin_ * 4)
            if count != Uint32(0):
                for r in cutlass.range_constexpr(self.cluster):
                    peer = (rank + r) & (self.cluster - 1)
                    cluster_red_add(peer_address(total + bin_ * 4, peer), count)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def count_level(
        self,
        level: int,
        keys,
        n: Int32,
        rank: Int32,
        tid: Int32,
        prefix: Uint32,
        hist: Int32,
    ) -> None:
        """Count this CTA's keys whose higher bits equal `prefix` into the
        histogram of round `level` (1: bits 20-10, 2: bits 9-0)."""
        h = hist + LEVEL_OFFSET[level] * 4
        for i in cutlass.range_constexpr(self.keys):
            key = keys[i]
            if self.index_of(i, rank, tid) < n:
                if cutlass.const_expr(level == 1):  # noqa: SIM102
                    if (key >> Uint32(21)) == prefix:
                        digit = (key >> Uint32(10)) & Uint32(0x7FF)
                        shared_red_add(h + Int32(digit) * 4, Uint32(1))
                if cutlass.const_expr(level == 2):  # noqa: SIM102
                    if (key >> Uint32(10)) == prefix:
                        digit = key & Uint32(0x3FF)
                        shared_red_add(h + Int32(digit) * 4, Uint32(1))
        cute.arch.sync_threads()

    @cute.jit
    def reduce_level(self, level: int, tid: Int32, hist: Int32, total: Int32) -> None:
        """Add this CTA's nonzero bins of the round into CTA 0's totals, then
        a cluster barrier (every CTA)."""
        offset = LEVEL_OFFSET[level] * 4
        remote = peer_address(total + offset, 0)
        for b in cutlass.range_constexpr(LEVEL_BINS[level] // self.threads):
            bin_ = b * self.threads + tid
            count = shared_load(hist + offset + bin_ * 4)
            if count != Uint32(0):
                cluster_red_add(remote + bin_ * 4, count)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

    @cute.jit
    def find_cluster(
        self,
        level: int,
        remain: Uint32,
        tid: Int32,
        rank: Int32,
        total: Int32,
        ctl: Int32,
        scan: Int32,
    ):
        """`find_local` over CTA 0's totals of the round; CTA 0 alone
        scans them and stores the result into every CTA's `ctl`, then a
        cluster barrier (every thread of the cluster calls it)."""
        if rank == 0:
            self.find_local(
                LEVEL_BINS[level],
                remain,
                tid,
                total + LEVEL_OFFSET[level] * 4,
                ctl,
                scan,
            )
            if tid < self.cluster:
                cluster_store_x4(
                    peer_address(ctl, tid),
                    shared_load(ctl + CTL_BIN * 4),
                    shared_load(ctl + CTL_REMAIN * 4),
                    shared_load(ctl + CTL_COUNT * 4),
                    Uint32(0),
                )
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        return (
            shared_load(ctl + CTL_BIN * 4),
            shared_load(ctl + CTL_REMAIN * 4),
            shared_load(ctl + CTL_COUNT * 4),
        )

    @cute.jit
    def store_block(self, out_row: Int64, slot: Uint32, index: Int32) -> None:
        """Write block `index`'s four tokens into output slot `slot`."""
        token = index * RATIO
        store_global_x4(
            out_row + Int64(slot) * 16, token, token + 1, token + 2, token + 3
        )

    @cute.jit
    def store_tail(
        self, tid: Int32, out_row: Int64, tail_count: Int32, tail_first: Int32
    ) -> None:
        """Write the tail columns and the valid-entry count of a long row."""
        if tid < RATIO:
            value = Int32(-1)
            if tid < RATIO - 1:
                if tid < tail_count:
                    value = tail_first + tid
            else:
                value = Int32(TOKENS) + tail_count
            store_global_s32(out_row + Int64(TOKENS + tid) * 4, value)

    @cute.jit
    def chosen(
        self,
        c: Int32,
        held: Int32,
        kth: Uint32,
        tie: Uint32,
        need: Uint32,
        cand: Int32,
    ) -> Uint32:
        """1 if CTA 0's candidate `c` is selected: its key exceeds `kth`,
        or equals it and (with `tie`) fewer than `need` candidates equal to
        `kth` have a lower block index."""
        key = shared_load(cand + c * 8)
        take = Uint32(0)
        if key > kth:
            take = Uint32(1)
        if key == kth:
            if tie == Uint32(0):
                take = Uint32(1)
            else:
                index = shared_load(cand + c * 8 + 4)
                before = Uint32(0)
                for d in cutlass.range(0, held, 1, unroll=1):
                    if shared_load(cand + d * 8) == kth:  # noqa: SIM102
                        if shared_load(cand + d * 8 + 4) < index:
                            before = before + Uint32(1)
                if before < need:
                    take = Uint32(1)
        return take

    @cute.jit
    def select_candidates(
        self,
        bin0: Uint32,
        remain0: Uint32,
        held0: Uint32,
        tid: Int32,
        out_row: Int64,
        hist: Int32,
        ctl: Int32,
        scan: Int32,
        cand: Int32,
    ) -> None:
        """CTA 0: the `remain0` largest of the `held0` gathered candidates
        (all in top-bit bin `bin0`), by rounds over bits 20-10 and 9-0 with
        CTA barriers, written after the blocks above the bin."""
        threads = self.threads
        held = Int32(held0)
        counts_a = hist + LEVEL_OFFSET[1] * 4
        for c in cutlass.range(tid, held, threads, unroll=1):
            key = shared_load(cand + c * 8)
            digit = (key >> Uint32(10)) & Uint32(0x7FF)
            shared_red_add(counts_a + Int32(digit) * 4, Uint32(1))
        cute.arch.sync_threads()
        bin_a, remain_a, held_a = self.find_local(
            LEVEL_BINS[1], remain0, tid, counts_a, ctl, scan
        )
        kth = (bin0 << Uint32(21)) | (bin_a << Uint32(10))
        tie = Uint32(0)
        need = remain_a
        if held_a != remain_a:
            counts_b = hist + LEVEL_OFFSET[2] * 4
            for c in cutlass.range(tid, held, threads, unroll=1):
                key = shared_load(cand + c * 8)
                if (key >> Uint32(10)) == (kth >> Uint32(10)):
                    digit = key & Uint32(0x3FF)
                    shared_red_add(counts_b + Int32(digit) * 4, Uint32(1))
            cute.arch.sync_threads()
            bin_b, remain_b, held_b = self.find_local(
                LEVEL_BINS[2], remain_a, tid, counts_b, ctl, scan
            )
            kth = kth | bin_b
            need = remain_b
            if held_b != remain_b:
                tie = Uint32(1)
        mine = Uint32(0)
        for c in cutlass.range(tid, held, threads, unroll=1):
            mine = mine + self.chosen(c, held, kth, tie, need, cand)
        first, _, _, _ = self.scan2(mine, Uint32(0), scan, tid)
        slot = Uint32(TOP_BLOCKS) - remain0 + first
        for c in cutlass.range(tid, held, threads, unroll=1):
            if self.chosen(c, held, kth, tie, need, cand) != Uint32(0):
                self.store_block(out_row, slot, Int32(shared_load(cand + c * 8 + 4)))
                slot = slot + Uint32(1)

    @cute.jit
    def gather(
        self,
        keys,
        n: Int32,
        rank: Int32,
        tid: Int32,
        bin0: Uint32,
        remain0: Uint32,
        held0: Uint32,
        out_row: Int64,
        tail_count: Int32,
        tail_first: Int32,
        hist: Int32,
        pub: Int32,
        ctl: Int32,
        scan: Int32,
        cand: Int32,
        counter: Int32,
    ) -> None:
        """The usual case (module docstring): the keys of top-bit bin
        `bin0` go to CTA 0, which selects `remain0` of them."""
        above = Uint32(0)
        equal = Uint32(0)
        for i in cutlass.range_constexpr(self.keys):
            if self.index_of(i, rank, tid) < n:
                digit = keys[i] >> Uint32(SHIFTS[0])
                if digit > bin0:
                    above = above + Uint32(1)
                if digit == bin0:
                    equal = equal + Uint32(1)
        # Candidate slots: a warp prefix of `equal` and one atomic a warp on
        # CTA 0's counter, issued before the CTA scan that hides its latency.
        lane = tid & 31
        run = equal
        for shift in cutlass.range_constexpr(5):
            below = shfl_up(run, 1 << shift)
            if lane >= (1 << shift):
                run = run + below
        warp_equal = shfl_idx(run, 31)
        base = Uint32(0)
        if (lane == 0) & (warp_equal != Uint32(0)):
            base = cluster_atom_add(peer_address(counter, 0), warp_equal)
        slot_above, _, cta_above, _ = self.scan2(above, Uint32(0), scan, tid)
        if tid < self.cluster:
            cluster_store_x2(peer_address(pub + rank * 8, tid), cta_above, Uint32(0))
        slot = shfl_idx(base, 0) + run - equal
        remote = peer_address(cand, 0)
        for i in cutlass.range_constexpr(self.keys):
            index = self.index_of(i, rank, tid)
            if index < n:
                key = keys[i]
                if (key >> Uint32(SHIFTS[0])) == bin0:
                    cluster_store_x2(remote + Int32(slot) * 8, key, Uint32(index))
                    slot = slot + Uint32(1)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for r in cutlass.range_constexpr(self.cluster):
            if r < rank:
                slot_above = slot_above + shared_load(pub + r * 8)
        for i in cutlass.range_constexpr(self.keys):
            index = self.index_of(i, rank, tid)
            if index < n:  # noqa: SIM102
                if (keys[i] >> Uint32(SHIFTS[0])) > bin0:
                    self.store_block(out_row, slot_above, index)
                    slot_above = slot_above + Uint32(1)
        if rank == 0:
            self.store_tail(tid, out_row, tail_count, tail_first)
            self.select_candidates(
                bin0, remain0, held0, tid, out_row, hist, ctl, scan, cand
            )

    @cute.jit
    def general(
        self,
        keys,
        n: Int32,
        rank: Int32,
        tid: Int32,
        bin0: Uint32,
        remain0: Uint32,
        held0: Uint32,
        out_row: Int64,
        tail_count: Int32,
        tail_first: Int32,
        hist: Int32,
        total: Int32,
        pub: Int32,
        ctl: Int32,
        scan: Int32,
    ) -> None:
        """Any row (module docstring): cluster rounds over bits 20-10 and
        9-0 while the resolved bin holds more keys than are needed, then every
        CTA writes its keys above the resolved prefix and, while slots remain,
        those equal to it (CTA, thread and key order)."""
        prefix = bin0
        remain = remain0
        shift = Uint32(SHIFTS[0])
        if held0 != remain0:
            self.count_level(1, keys, n, rank, tid, prefix, hist)
            self.reduce_level(1, tid, hist, total)
            bin1, remain1, held1 = self.find_cluster(
                1, remain, tid, rank, total, ctl, scan
            )
            prefix = (prefix << Uint32(11)) | bin1
            remain = remain1
            shift = Uint32(SHIFTS[1])
            if held1 != remain1:
                self.count_level(2, keys, n, rank, tid, prefix, hist)
                self.reduce_level(2, tid, hist, total)
                bin2, remain2, _ = self.find_cluster(
                    2, remain, tid, rank, total, ctl, scan
                )
                prefix = (prefix << Uint32(10)) | bin2
                remain = remain2
                shift = Uint32(SHIFTS[2])
        above = Uint32(0)
        equal = Uint32(0)
        for i in cutlass.range_constexpr(self.keys):
            if self.index_of(i, rank, tid) < n:
                digit = keys[i] >> shift
                if digit > prefix:
                    above = above + Uint32(1)
                if digit == prefix:
                    equal = equal + Uint32(1)
        slot_above, slot_equal, cta_above, cta_equal = self.scan2(
            above, equal, scan, tid
        )
        if tid < self.cluster:
            cluster_store_x2(peer_address(pub + rank * 8, tid), cta_above, cta_equal)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        for r in cutlass.range_constexpr(self.cluster):
            if r < rank:
                slot_above = slot_above + shared_load(pub + r * 8)
                slot_equal = slot_equal + shared_load(pub + r * 8 + 4)
        first_equal = Uint32(TOP_BLOCKS) - remain
        for i in cutlass.range_constexpr(self.keys):
            index = self.index_of(i, rank, tid)
            if index < n:
                digit = keys[i] >> shift
                if digit > prefix:
                    self.store_block(out_row, slot_above, index)
                    slot_above = slot_above + Uint32(1)
                if digit == prefix:
                    if slot_equal < remain:
                        self.store_block(out_row, first_equal + slot_equal, index)
                    slot_equal = slot_equal + Uint32(1)
        if rank == 0:
            self.store_tail(tid, out_row, tail_count, tail_first)

    @cute.jit
    def large(
        self,
        keys,
        n: Int32,
        rank: Int32,
        tid: Int32,
        row_logits: Int64,
        out_row: Int64,
        tail_count: Int32,
        tail_first: Int32,
        hist: Int32,
        total: Int32,
        pub: Int32,
        ctl: Int32,
        scan: Int32,
        cand: Int32,
        counter: Int32,
    ) -> None:
        """A row of more than 512 visible blocks (module docstring)."""
        cluster, threads = self.cluster, self.threads
        # The groups after the first, now that the row length is known.
        for j in cutlass.range_constexpr(1, self.vectors):
            first = ((Int32(j * cluster) + rank) * threads + tid) * 4
            if first < n:
                words = load_global_x4(row_logits + Int64(first) * 4)
                for e in cutlass.range_constexpr(4):
                    keys[4 * j + e] = words[e]
        # Order-preserving keys of the FP32 scores.
        for i in cutlass.range_constexpr(self.keys):
            bits = keys[i]
            keys[i] = bits ^ (Uint32(0x80000000) | (Uint32(0) - (bits >> Uint32(31))))
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.cluster_wait()  # every CTA's totals and counter are zero
        self.count_top(keys, n, rank, tid, hist)
        self.reduce_all(rank, tid, hist, total)
        bin0, remain0, held0 = self.find_local(
            LEVEL_BINS[0], Uint32(TOP_BLOCKS), tid, total, ctl, scan
        )
        if (held0 != remain0) & (held0 <= Uint32(CAPACITY)):
            self.gather(
                keys,
                n,
                rank,
                tid,
                bin0,
                remain0,
                held0,
                out_row,
                tail_count,
                tail_first,
                hist,
                pub,
                ctl,
                scan,
                cand,
                counter,
            )
        else:
            self.general(
                keys,
                n,
                rank,
                tid,
                bin0,
                remain0,
                held0,
                out_row,
                tail_count,
                tail_first,
                hist,
                total,
                pub,
                ctl,
                scan,
            )

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        row_stride: Int32,
        columns: Int32,
        visible: Int64,
        positions: Int64,
        out: Int64,
        out_stride: Int32,
    ):
        """One cluster a row (module docstring)."""
        cluster, threads = self.cluster, self.threads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        row = cta // cluster

        allocator = cutlass.utils.SmemAllocator()
        hist = Int32(
            allocator.allocate_array(Uint32, HIST_WORDS, byte_alignment=16).toint()
        )
        total = Int32(
            allocator.allocate_array(Uint32, HIST_WORDS, byte_alignment=16).toint()
        )
        cand = Int32(
            allocator.allocate_array(Uint32, 2 * CAPACITY, byte_alignment=16).toint()
        )
        pub = Int32(
            allocator.allocate_array(Uint32, PUB_WORDS, byte_alignment=16).toint()
        )
        ctl = Int32(
            allocator.allocate_array(Uint32, CTL_WORDS, byte_alignment=16).toint()
        )
        scan = Int32(
            allocator.allocate_array(Uint32, SCAN_WORDS, byte_alignment=16).toint()
        )
        counter = Int32(
            allocator.allocate_array(Uint32, COUNTER_WORDS, byte_alignment=16).toint()
        )
        # Zero the histograms, the totals and the candidate counter while the
        # previous kernel still runs; peers may add into them once this CTA
        # has arrived.
        for v in cutlass.range(tid, HIST_WORDS // 4, threads, unroll=1):
            shared_zero_x4(hist + v * 16)
            shared_zero_x4(total + v * 16)
        if tid == 0:
            shared_zero_x4(counter)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive()

        cute.arch.griddepcontrol_wait()
        # This thread's first group of four scores, before the row length.
        row_logits = logits + Int64(row) * Int64(row_stride) * 4
        keys = cute.make_rmem_tensor((self.keys,), Uint32)
        for i in cutlass.range_constexpr(self.keys):
            keys[i] = Uint32(0)
        first = (rank * threads + tid) * 4
        if first + 4 <= columns:
            words = load_global_x4(row_logits + Int64(first) * 4)
            for e in cutlass.range_constexpr(4):
                keys[e] = words[e]
        n = cutlass.max(load_global_s32(visible + Int64(row) * 4), Int32(0))
        position = load_global_s64(positions + Int64(row) * 8)
        out_row = out + Int64(row) * Int64(out_stride) * 4
        tail_start = ((position + 1) // RATIO) * RATIO
        tail_count = Int32(position + 1 - tail_start)
        tail_first = Int32(tail_start)

        if n <= TOP_BLOCKS:
            # Every visible block, in order: the stock expansion of the stock
            # top-k's 0 .. n-1.
            if rank == 0:
                expanded = n * RATIO
                for c in cutlass.range(tid, PACKED_WIDTH, threads, unroll=1):
                    value = Int32(-1)
                    if c < expanded:
                        value = c
                    else:
                        if c < OUT_WIDTH:
                            if c - expanded < tail_count:
                                value = tail_first + c - expanded
                        else:
                            value = expanded + tail_count
                    store_global_s32(out_row + Int64(c) * 4, value)
        else:
            self.large(
                keys,
                n,
                rank,
                tid,
                row_logits,
                out_row,
                tail_count,
                tail_first,
                hist,
                total,
                pub,
                ctl,
                scan,
                cand,
                counter,
            )


@cache
def _compiled(device_index: int, cluster: int, threads: int, vectors: int):
    """`QsaSelect` compiled for a device and a cluster shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            QsaSelect(cluster, threads, vectors),
            Int64(0),
            Int32(4),
            Int32(4),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(PACKED_WIDTH),
            Int32(1),
            make_fake_stream(),
        )


def default_variant(rows: int) -> tuple[int, int, int]:
    """The cluster shape for `rows` rows: the widest clusters while every
    row's cluster fits one wave of the GPU."""
    if rows <= 8:
        return (16, 256, 4)
    if rows <= 18:
        return (8, 512, 4)
    if rows <= 72:
        return (4, 512, 8)
    return (2, 1024, 8)


def qsa_select_expand(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    variant: tuple[int, ...] | None = None,
) -> None:
    """Select each row's top 512 blocks and write the packed token list.

    Args:
        logits: FP32 block scores `[rows, columns]` (unit column stride,
            `columns` and the row stride multiples of 4, 16-byte aligned),
            row `r` valid below `visible_blocks[r]`, at most `columns`
            and 65536 (262,144 tokens).
        visible_blocks: Contiguous int32 `[rows]`.
        positions: Contiguous int64 `[rows]` logical query positions.
        out: int32 `[rows, 2052]` (unit column stride, a row stride that is
            a multiple of 4, 16-byte aligned): the packed selection.
        variant: `(CTAs a row, threads, vectors a thread)` from
            `VARIANTS`; default `default_variant`.

    """
    rows, columns = logits.shape
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert columns % 4 == 0 and logits.stride(0) % 4 == 0
    assert logits.data_ptr() % 16 == 0
    assert visible_blocks.dtype == torch.int32 and visible_blocks.is_contiguous()
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert visible_blocks.shape == positions.shape == (rows,)
    assert out.dtype == torch.int32 and out.shape == (rows, PACKED_WIDTH)
    assert out.stride(1) == 1 and out.stride(0) % 4 == 0 and out.data_ptr() % 16 == 0
    if rows == 0:
        return
    variant = variant or default_variant(rows)
    _compiled(logits.device.index, *variant)(
        Int64(logits.data_ptr()),
        Int32(logits.stride(0)),
        Int32(columns),
        Int64(visible_blocks.data_ptr()),
        Int64(positions.data_ptr()),
        Int64(out.data_ptr()),
        Int32(out.stride(0)),
        Int32(rows),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
