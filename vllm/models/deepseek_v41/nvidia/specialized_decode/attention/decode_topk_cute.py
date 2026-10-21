# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode top-k of the dense indexers straight from their BF16 logits, in CuTe DSL.

The problem. In a decode step each dense indexer layer scores every compressed
token of a row (SM100's paged MQA logits are BF16) and keeps the ``k`` largest:
the token top-512, and in the candidate source also the top-2048 of the rows'
8-token block maxima. ``cooperative_topk`` reads FP32, so the logits used to be
upcast first, and it builds a radix histogram of every value with shared-memory
atomics. Real indexer logits crowd a few histogram bins (a row of 150K logits
holds about 200 distinct BF16 values), where those atomics serialize.

What this module computes. ``DecodeTopK`` writes, for each row, the indices of
its ``k`` largest elements in any order (exact: every index whose value exceeds
the k-th largest, then enough of those equal to it), or ``0 .. n-1`` and -1
padding when the row has at most ``k`` elements. An element is one BF16 logit
(token mode), or the NaN-propagating maximum of an aligned group of 8 logits
(block mode, the candidate blocks), with the row's newest block pinned at +inf
and blocks scoring -inf written as -1. Values are compared through an
order-preserving 16-bit key of their BF16 bits, so ties are exact.

How. One thread block cluster of ``C`` CTAs per row: 8 CTAs of 512 threads
while every row's cluster can be resident at once, else 4 of 1024 (a second
wave of clusters would nearly double the time). Each CTA copies one
contiguous chunk of the row into shared memory, as 16-byte vectors (8 logits,
or one block); the chunks split the row's valid length, not its width. A
thread stages 8 vectors and keeps them in registers too, so a cluster holds
262,144 logits. Wider rows need more: at a 1M context the ratio-2 layers'
logits are 524,288 wide, and a second build stages 16 vectors a thread at 8
CTAs, the other 8 in shared memory only. The candidate source layer scores
every token (ratio 1), 1,048,576 at a 1M context, so a third build runs 16
CTAs a row (a non-portable cluster size, which CuTe DSL allows on every
kernel) with 16 vectors a thread. Rows of the wide builds run in waves when
they outnumber the resident clusters; the 16-CTA build takes at most two
waves (`supports_decode_topk`). Two ways to the answer:

- Sampled (token rows longer than one vector a thread)::

      sample the row ──► every CTA: the same cutoff (the q-th largest of the
      row's 2048 samples, q aiming at 3k survivors)
         └► keep keys > cutoff and up to k keys == cutoff, push both to CTA 0
              └► CTA 0: exact k-th largest among <= 8192 survivors (two 8-bit
                 radix passes), write k indices

- Direct (block mode, short token rows, and a sample that misjudged the row:
  more than 8192 survivors, or fewer than k at or above the cutoff): the exact
  k-th largest key from two cluster-wide 8-bit radix histograms, then every
  CTA writes its own selected indices straight to the output.

Both filter the same way: a thread tests each of its vectors' maximum against
the cutoff, the vectors that can hold selected keys go to a CTA queue, and
the CTA's threads unpack the queued vectors evenly; a CTA scan gives each
thread its output slots and one cluster atomic reserves the CTA's range, so no
key costs an atomic. Whole-row histograms spread each warp's atomics over 8
copies (lane l counts into copy l mod 8), since a crowded bin serializes the
atomics on its word; the radix selects over the few thousand samples or
survivors count into one histogram, where the copies' zeroing and summing
cost more than they save.

Synchronization. Cluster barriers separate the phases: every CTA has started
and zeroed its counters before any distributed shared memory access; samples
are complete before the cutoff; survivors are complete before CTA 0 reads
them; histogram contributions are complete before they are read; and CTA 0
waits until every peer is done with its shared memory before it exits. The
kernel waits on its predecessor grid before reading inputs and lets its
dependents launch once its chunk is in shared memory (programmatic dependent
launch).
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ...ops.cute_dsl.primitives import current_cu_stream

SAMPLES = 2048  # keys sampled a row, over the cluster
CAPACITY = 8192  # survivors a row
SURVIVOR_FACTOR = 3  # the cutoff aims at this many times k survivors
VECTORS = 8  # 16-byte vectors a thread keeps in registers (of 8, or 16 for wide rows)
NEG_INF_KEY = 0x007F  # key of BF16 -inf
POS_INF_KEY = 0xFF80  # key of BF16 +inf
# ``ctl`` slots (shared Uint32 words).
BIN, REMAIN = 0, 1  # select_bin's result
BASE_GT, BASE_EQ = 2, 3  # this CTA's reserved first slots
TOTALS = 8  # at CTA 0: (gt, eq) totals, one 8-byte aligned pair a pass
CTL_WORDS = 16
SCAN_WORDS = 160
COPIES = 8  # copies of a whole-row histogram, [copy][bin]; lane l uses copy l % 8
TOP_KS = (512, 1024, 2048)  # the k values a kernel is built for


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(value, loc, ip):
    """IR value of a DSL scalar."""
    return value.ir_value(loc=loc, ip=ip)


def _u32x4(loaded, loc, ip):
    """The four words of an inline-assembly struct result."""
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def load_u32x4(address: Int64, *, loc=None, ip=None):
    """Four 32-bit words at a 16-byte aligned global address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(address, loc, ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        False,
        loc,
        ip,
    )
    return _u32x4(loaded, loc, ip)


@dsl_user_op
def shared_load_u32x4(address: Int32, *, loc=None, ip=None):
    """Four 32-bit words at a 16-byte aligned shared address of this CTA."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_v(Int32(address), loc, ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return _u32x4(loaded, loc, ip)


@dsl_user_op
def shared_store_u32x4(address: Int32, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """Store four 32-bit words at a 16-byte aligned shared address of this CTA."""
    _asm(
        None,
        [_v(Int32(address), loc, ip)]
        + [_v(Uint32(w), loc, ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def store_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """Store a 32-bit word to global memory."""
    _asm(
        None,
        [_v(address, loc, ip), _v(Int32(value), loc, ip)],
        "st.global.s32 [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def max_nan_bf16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Lane-wise maximum of two BF16 pairs, NaN if either is NaN."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip)],
            "max.NaN.bf16x2 $0, $1, $2;",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def shfl_up(value: Uint32, delta: int, *, loc=None, ip=None) -> Uint32:
    """``value`` of the lane ``delta`` below (own value in the lowest lanes)."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Uint32(value), loc, ip)],
            f"shfl.sync.up.b32 $0, $1, {delta}, 0, 0xffffffff;",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def shared_red(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Atomic add to a shared word of this CTA."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def peer_address(address: Int32, rank, *, loc=None, ip=None) -> Int32:
    """The cluster address of this CTA's shared ``address`` in CTA ``rank``."""
    return Int32(
        _asm(
            T.i32(),
            [_v(Int32(address), loc, ip), _v(Int32(rank), loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def cluster_add_u64(address: Int32, value: Int64, *, loc=None, ip=None) -> Int64:
    """Atomic add to an 8-byte aligned cluster shared word pair; returns the
    old value."""
    return Int64(
        _asm(
            T.i64(),
            [_v(Int32(address), loc, ip), _v(Int64(value), loc, ip)],
            "atom.shared::cluster.add.u64 $0, [$1], $2;",
            "=l,r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def cluster_red(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Atomic add to a cluster shared word."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "red.shared::cluster.add.u32 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def cluster_load(address: Int32, *, loc=None, ip=None) -> Uint32:
    """Load a cluster shared word."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(Int32(address), loc, ip)],
            "ld.shared::cluster.u32 $0, [$1];",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def cluster_store_u32(address: Int32, value, *, loc=None, ip=None) -> None:
    """Store a 32-bit word to cluster shared memory."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "st.shared::cluster.u32 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def cluster_store_u16(address: Int32, value, *, loc=None, ip=None) -> None:
    """Store the low half of a word to cluster shared memory."""
    _asm(
        None,
        [_v(Int32(address), loc, ip), _v(Uint32(value), loc, ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.shared::cluster.u16 [$0], h;\n}",
        "r,r",
        True,
        loc,
        ip,
    )


@cute.jit
def order_key(bits: Uint32) -> Uint32:
    """The order-preserving key of the BF16 value in the low half of ``bits``:
    a larger value has a larger key, NaNs of either sign outside the
    numbers."""
    b = bits & Uint32(0xFFFF)
    return b ^ (Uint32(0x8000) | ((Uint32(0) - (b >> Uint32(15))) & Uint32(0x7FFF)))


@cute.jit
def block_key(w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32) -> Uint32:
    """The key of the NaN-propagating maximum of eight BF16 values."""
    m = max_nan_bf16x2(max_nan_bf16x2(w0, w1), max_nan_bf16x2(w2, w3))
    m = max_nan_bf16x2(m, (m >> Uint32(16)) | (m << Uint32(16)))
    return order_key(m)


@cute.jit
def hist_add(
    hist_address: Int32, bin_: Uint32, valid, copies: bool, lane: Int32
) -> None:
    """Count each valid lane's ``bin_`` with one shared atomic. ``copies``:
    lane ``l`` counts into copy ``l % COPIES`` of a ``[COPIES][256]``
    histogram (whole-row histograms: crowded bins would serialize the
    atomics of one copy); else into a plain 256-word one."""
    if valid:
        slot = Int32(bin_)
        if cutlass.const_expr(copies):
            slot = slot + (lane & (COPIES - 1)) * 256
        shared_red(hist_address + slot * 4, Uint32(1))


@cute.jit
def select_bin(hist: cute.Tensor, q: Uint32, ctl: cute.Tensor, lane: Int32) -> None:
    """Warp 0: the bin ``b`` of a 256-bin histogram holding the ``q``-th
    largest entry (q >= 1), into ``ctl[BIN]``, and ``q`` minus the entries
    in the bins above ``b`` into ``ctl[REMAIN]``. Lane ``l`` scans bins
    ``255 - 8 l`` down to ``248 - 8 l``."""
    low = shared_load_u32x4(Int32(hist.iterator.toint()) + (248 - 8 * lane) * 4)
    high = shared_load_u32x4(Int32(hist.iterator.toint()) + (252 - 8 * lane) * 4)
    counts = (high[3], high[2], high[1], high[0], low[3], low[2], low[1], low[0])
    local = Uint32(0)
    for i in cutlass.range_constexpr(8):
        local = local + counts[i]
    inclusive = local
    for shift in cutlass.range_constexpr(5):
        below = shfl_up(inclusive, 1 << shift)
        if lane >= (1 << shift):
            inclusive = inclusive + below
    above = inclusive - local
    if (above < q) & (inclusive >= q):
        found = Int32(-1)
        remain = Uint32(0)
        for i in cutlass.range_constexpr(8):
            if (found < 0) & (above + counts[i] >= q):
                found = Int32(255 - 8 * lane - i)
                remain = q - above
            above = above + counts[i]
        ctl[BIN] = Uint32(found)
        ctl[REMAIN] = remain


class DecodeTopK:
    """The decode top-k (module docstring) of rows of BF16 logits.

    ``k``: 512, 1024 or 2048; ``cluster``: 4, 8 or 16 CTAs a row (4: 1024
    threads, else 512); ``vector_count``: 16-byte vectors a thread stages (8,
    or 16 at 8 or 16 CTAs: the first 8 also stay in registers), so a row
    holds at most ``cluster * threads * vector_count`` vectors;
    ``block_mode``: elements are 8-logit block maxima.
    """

    def __init__(
        self, k: int, cluster: int, vector_count: int, block_mode: bool
    ) -> None:
        assert k in TOP_KS and cluster in (4, 8, 16)
        assert vector_count in vector_choices(cluster)
        self.k = k
        self.cluster = cluster
        self.threads = threads_for(cluster)
        self.warps = self.threads // 32
        self.block_mode = block_mode
        self.per_vector = 1 if block_mode else 8  # elements a 16-byte vector holds
        self.vector_count = vector_count
        self.vectors = self.threads * vector_count  # a CTA's vectors
        self.samples = SAMPLES // cluster  # a CTA's samples, one a thread
        assert self.samples <= self.threads
        self.max_elements = cluster * self.vectors * self.per_vector
        # Token rows of at most one vector a thread take the direct path.
        self.direct_elements = cluster * self.threads * self.per_vector

    @cute.jit
    def __call__(
        self,
        logits_ptr: cute.Pointer,
        lengths_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        rows: Int32,
        row_bytes: Int64,
        width: Int32,
        out_stride: Int32,
        fallback: Int32,
        stream: cuda.CUstream,
    ):
        """``logits``: BF16 rows ``row_bytes`` apart (16-byte aligned), each
        valid below ``lengths[r]`` (int32, clamped to [0, width]); ``out``:
        int32 rows ``out_stride`` apart. ``fallback`` nonzero forces the
        direct path (tests)."""
        self.kernel(
            cute.make_tensor(lengths_ptr, cute.make_layout(rows)),
            cute.make_tensor(out_ptr, cute.make_layout(rows * out_stride)),
            logits_ptr.toint(),
            row_bytes,
            width,
            out_stride,
            fallback,
        ).launch(
            grid=(rows * self.cluster, 1, 1),
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def vector_words(self, data_address: Int32, vector: Int32):
        """The four words of the chunk's vector ``vector`` in shared memory."""
        return shared_load_u32x4(data_address + vector * 16)

    @cute.jit
    def vector_top(self, words, element: Int32, pinned: Int32) -> Uint32:
        """The largest key of a staged vector whose first element is
        ``element`` (block mode: the pinned newest block ranks first)."""
        top = block_key(words[0], words[1], words[2], words[3])
        if cutlass.const_expr(self.block_mode):  # noqa: SIM102
            if element == pinned:
                top = Uint32(POS_INF_KEY)
        return top

    @cute.jit
    def word_key(self, words, e: int, element: Int32, pinned: Int32) -> Uint32:
        """The key of element ``e`` of a vector's ``words`` (``element``: its
        row index, for the pinned block)."""
        key = Uint32(0)
        if cutlass.const_expr(self.block_mode):
            key = block_key(words[0], words[1], words[2], words[3])
            if element == pinned:
                key = Uint32(POS_INF_KEY)
        else:
            key = order_key(Uint32(words[e // 2]) >> Uint32(16 * (e % 2)))
        return key

    @cute.jit
    def scan2(self, a: Uint32, b: Uint32, scan: cute.Tensor, tid: Int32):
        """Exclusive CTA prefix sums of ``a`` and ``b`` over the threads, and
        their totals (every thread calls it; three barriers)."""
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
        cute.arch.sync_threads()  # the previous scan's readers are done
        if lane == 31:
            scan[warp] = in_a
            scan[32 + warp] = in_b
        cute.arch.sync_threads()
        if warp == 0:
            sum_a = Uint32(0)
            sum_b = Uint32(0)
            if lane < self.warps:
                sum_a = Uint32(scan[lane])
                sum_b = Uint32(scan[32 + lane])
            run_a = sum_a
            run_b = sum_b
            for shift in cutlass.range_constexpr(5):
                below_a = shfl_up(run_a, 1 << shift)
                below_b = shfl_up(run_b, 1 << shift)
                if lane >= (1 << shift):
                    run_a = run_a + below_a
                    run_b = run_b + below_b
            scan[64 + lane] = run_a - sum_a
            scan[96 + lane] = run_b - sum_b
            if lane == 31:
                scan[128] = run_a
                scan[129] = run_b
        cute.arch.sync_threads()
        first_a = Uint32(scan[64 + warp]) + in_a - a
        first_b = Uint32(scan[96 + warp]) + in_b - b
        return first_a, first_b, Uint32(scan[128]), Uint32(scan[129])

    @cute.jit
    def radix_select(
        self,
        keys: cute.Tensor,
        count: Int32,
        q: Uint32,
        hist: cute.Tensor,
        hist2: cute.Tensor,
        ctl: cute.Tensor,
        tid: Int32,
    ):
        """The ``q``-th largest of ``keys[0:count]`` (shared Uint16, q <= count)
        and how many of the keys equal to it rank within the top ``q``. Every
        thread calls it."""
        lane = tid & 31
        h1 = Int32(hist.iterator.toint())
        h2 = Int32(hist2.iterator.toint())
        if tid < 256:
            hist[tid] = Uint32(0)
            hist2[tid] = Uint32(0)
        cute.arch.sync_threads()
        iterations = (count + self.threads - 1) // self.threads
        for it in cutlass.range(iterations, unroll=4):
            i = it * self.threads + tid
            valid = i < count
            key = Uint32(0)
            if valid:
                key = Uint32(keys[i])
            hist_add(h1, key >> Uint32(8), valid, False, lane)
        cute.arch.sync_threads()
        if tid < 32:
            select_bin(hist, q, ctl, lane)
        cute.arch.sync_threads()
        high = Uint32(ctl[BIN])
        remain = Uint32(ctl[REMAIN])
        for it in cutlass.range(iterations, unroll=4):
            i = it * self.threads + tid
            key = Uint32(0)
            if i < count:
                key = Uint32(keys[i])
            hist_add(
                h2,
                key & Uint32(0xFF),
                (i < count) & ((key >> Uint32(8)) == high),
                False,
                lane,
            )
        cute.arch.sync_threads()
        if tid < 32:
            select_bin(hist2, remain, ctl, lane)
        cute.arch.sync_threads()
        tau = (high << Uint32(8)) | Uint32(ctl[BIN])
        need = Uint32(ctl[REMAIN])
        cute.arch.sync_threads()
        return tau, need

    @cute.jit
    def queue_vectors(
        self,
        regs,
        data_address: Int32,
        cutoff: Uint32,
        start: Int32,
        length: Int32,
        pinned: Int32,
        tid: Int32,
        scan,
        queue,
    ) -> Int32:
        """Queue this CTA's vectors that can hold a key >= ``cutoff`` (by
        their maximum) and return how many (every thread)."""
        threads, per = self.threads, self.per_vector
        hits = Uint32(0)
        for j in cutlass.range_constexpr(VECTORS):
            first = (j * threads + tid) * per
            if first < length:
                words = (regs[j, 0], regs[j, 1], regs[j, 2], regs[j, 3])
                if self.vector_top(words, start + first, pinned) >= cutoff:
                    hits = hits | Uint32(1 << j)
        if cutlass.const_expr(self.vector_count > VECTORS):  # noqa: SIM102
            # Only a chunk reaching the shared-only vectors visits them.
            if VECTORS * threads * per < length:
                for j in cutlass.range_constexpr(VECTORS, self.vector_count):
                    first = (j * threads + tid) * per
                    if first < length:
                        words = self.vector_words(data_address, j * threads + tid)
                        if self.vector_top(words, start + first, pinned) >= cutoff:
                            hits = hits | Uint32(1 << j)
        slot, _, total, _ = self.scan2(
            Uint32(cute.arch.popc(hits)), Uint32(0), scan, tid
        )
        position = Int32(slot)
        if hits != Uint32(0):
            for j in cutlass.range_constexpr(self.vector_count):
                if (hits >> Uint32(j)) & Uint32(1) != Uint32(0):
                    queue[position] = Uint16(j * threads + tid)
                    position = position + 1
        cute.arch.sync_threads()
        return Int32(total)

    @cute.jit
    def count_queued(
        self,
        data_address: Int32,
        queued: Int32,
        cutoff: Uint32,
        start: Int32,
        length: Int32,
        pinned: Int32,
        tid: Int32,
        queue,
    ):
        """This thread's keys above and equal to ``cutoff`` in its share of
        the queued vectors (entries ``tid``, ``tid + threads``, ...)."""
        per = self.per_vector
        gt = Uint32(0)
        eq = Uint32(0)
        for q in cutlass.range(tid, queued, self.threads, unroll=1):
            vector = Int32(queue[q])
            words = self.vector_words(data_address, vector)
            for e in cutlass.range_constexpr(per):
                element = vector * per + e
                key = self.word_key(words, e, start + element, pinned)
                if element < length:
                    if key > cutoff:
                        gt = gt + Uint32(1)
                    if key == cutoff:
                        eq = eq + Uint32(1)
        return gt, eq

    @cute.jit
    def reserve(self, gt: Uint32, eq: Uint32, totals: int, tid: Int32, ctl, scan):
        """This thread's first slots for its ``gt`` / ``eq`` keys in the
        cluster-wide lists counted at CTA 0's ``ctl[totals]`` pair: a CTA
        scan, then one 64-bit cluster atomic for the CTA."""
        first_gt, first_eq, cta_gt, cta_eq = self.scan2(gt, eq, scan, tid)
        if tid == 0:
            old = cluster_add_u64(
                peer_address(Int32(ctl.iterator.toint()) + totals * 4, 0),
                Int64(cta_gt) | (Int64(cta_eq) << Int64(32)),
            )
            ctl[BASE_GT] = Uint32(old & Int64(0xFFFFFFFF))
            ctl[BASE_EQ] = Uint32(old >> Int64(32))
        cute.arch.sync_threads()
        return Int32(Uint32(ctl[BASE_GT]) + first_gt), Int32(
            Uint32(ctl[BASE_EQ]) + first_eq
        )

    @cute.jit
    def gather(
        self,
        regs,
        data_address: Int32,
        cutoff: Uint32,
        start: Int32,
        length: Int32,
        pinned: Int32,
        tid: Int32,
        ctl,
        scan,
        queue,
        g_key,
        g_idx,
        g_eq,
    ):
        """Sampled path: push this CTA's keys above ``cutoff`` and up to k
        equal to it to CTA 0 and return the cluster's totals (every CTA)."""
        k, per = self.k, self.per_vector
        queued = self.queue_vectors(
            regs, data_address, cutoff, start, length, pinned, tid, scan, queue
        )
        gt, eq = self.count_queued(
            data_address, queued, cutoff, start, length, pinned, tid, queue
        )
        slot_gt, slot_eq = self.reserve(gt, eq, TOTALS, tid, ctl, scan)
        key_remote = peer_address(Int32(g_key.iterator.toint()), 0)
        idx_remote = peer_address(Int32(g_idx.iterator.toint()), 0)
        eq_remote = peer_address(Int32(g_eq.iterator.toint()), 0)
        for q in cutlass.range(tid, queued, self.threads, unroll=1):
            vector = Int32(queue[q])
            words = self.vector_words(data_address, vector)
            for e in cutlass.range_constexpr(per):
                element = vector * per + e
                key = self.word_key(words, e, start + element, pinned)
                if element < length:
                    if key > cutoff:
                        if slot_gt < CAPACITY:
                            cluster_store_u16(key_remote + slot_gt * 2, key)
                            cluster_store_u32(
                                idx_remote + slot_gt * 4, Uint32(start + element)
                            )
                        slot_gt = slot_gt + 1
                    if key == cutoff:
                        if slot_eq < k:
                            cluster_store_u32(
                                eq_remote + slot_eq * 4, Uint32(start + element)
                            )
                        slot_eq = slot_eq + 1
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        totals = peer_address(Int32(ctl.iterator.toint()) + TOTALS * 4, 0)
        return cluster_load(totals), cluster_load(totals + 4)

    @cute.jit
    def count_vector(
        self,
        level: int,
        words,
        first: Int32,
        start: Int32,
        length: Int32,
        pinned: Int32,
        high: Uint32,
        hist_address: Int32,
        lane: Int32,
    ) -> None:
        """Count a staged vector (``words``, its first element ``first``) into
        a whole-row histogram: level 0 the high bytes of its keys, level 1
        the low bytes of those whose high byte is ``high``."""
        per = self.per_vector
        if cutlass.const_expr(level == 0):
            if cute.arch.vote_any_sync(first < length):
                for e in cutlass.range_constexpr(per):
                    key = self.word_key(words, e, start + first + e, pinned)
                    hist_add(
                        hist_address, key >> Uint32(8), first + e < length, True, lane
                    )
        else:
            # Only vectors reaching the selected high byte count.
            top = self.vector_top(words, start + first, pinned)
            reach = (first < length) & ((top >> Uint32(8)) >= high)
            if cute.arch.vote_any_sync(reach):
                for e in cutlass.range_constexpr(per):
                    key = self.word_key(words, e, start + first + e, pinned)
                    hist_add(
                        hist_address,
                        key & Uint32(0xFF),
                        reach & (first + e < length) & ((key >> Uint32(8)) == high),
                        True,
                        lane,
                    )

    @cute.jit
    def exact_cutoff(
        self,
        regs,
        data_address: Int32,
        start: Int32,
        length: Int32,
        pinned: Int32,
        tid: Int32,
        ctl,
        hist,
        hist_a,
        hist_b,
    ):
        """The row's exact k-th largest key and how many keys equal to it
        rank within the top k, from every CTA's vectors: two cluster-wide
        radix histograms gathered at CTA 0 (every CTA)."""
        threads, per = self.threads, self.per_vector
        lane = tid & 31
        hist_address = Int32(hist.iterator.toint())
        high = Uint32(0)
        remain = Uint32(self.k)
        for level in cutlass.range_constexpr(2):
            target = hist_a if level == 0 else hist_b
            for i in cutlass.range(tid, 256 * COPIES, threads, unroll=1):
                hist[i] = Uint32(0)
            cute.arch.sync_threads()
            for j in cutlass.range_constexpr(VECTORS):
                self.count_vector(
                    level,
                    (regs[j, 0], regs[j, 1], regs[j, 2], regs[j, 3]),
                    (j * threads + tid) * per,
                    start,
                    length,
                    pinned,
                    high,
                    hist_address,
                    lane,
                )
            if cutlass.const_expr(self.vector_count > VECTORS):  # noqa: SIM102
                # Only a chunk reaching the shared-only vectors visits them,
                # and only warps reaching a vector load it.
                if VECTORS * threads * per < length:
                    for j in cutlass.range_constexpr(VECTORS, self.vector_count):
                        first = (j * threads + tid) * per
                        if cute.arch.vote_any_sync(first < length):
                            words = self.vector_words(data_address, j * threads + tid)
                            self.count_vector(
                                level,
                                words,
                                first,
                                start,
                                length,
                                pinned,
                                high,
                                hist_address,
                                lane,
                            )
            cute.arch.sync_threads()
            target_remote = peer_address(Int32(target.iterator.toint()), 0)
            if tid < 256:
                count = Uint32(hist[tid])
                for c in cutlass.range_constexpr(1, COPIES):
                    count = count + Uint32(hist[c * 256 + tid])
                if count != Uint32(0):
                    cluster_red(target_remote + tid * 4, count)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tid < 256:
                hist[tid] = cluster_load(target_remote + tid * 4)
            cute.arch.sync_threads()
            if tid < 32:
                select_bin(hist, remain, ctl, lane)
            cute.arch.sync_threads()
            if cutlass.const_expr(level == 0):
                high = Uint32(ctl[BIN])
            else:
                high = (high << Uint32(8)) | Uint32(ctl[BIN])
            remain = Uint32(ctl[REMAIN])
            cute.arch.sync_threads()
        return high, remain

    @cute.jit
    def direct(
        self,
        regs,
        data_address: Int32,
        start: Int32,
        length: Int32,
        pinned: Int32,
        tid: Int32,
        out_row: Int64,
        ctl,
        hist,
        hist_a,
        hist_b,
        scan,
        queue,
    ) -> None:
        """Direct path: the exact cutoff, then every CTA writes its keys
        above it and its share of the ``need`` equal ones to the output."""
        k, per = self.k, self.per_vector
        tau, need = self.exact_cutoff(
            regs, data_address, start, length, pinned, tid, ctl, hist, hist_a, hist_b
        )
        queued = self.queue_vectors(
            regs, data_address, tau, start, length, pinned, tid, scan, queue
        )
        gt, eq = self.count_queued(
            data_address, queued, tau, start, length, pinned, tid, queue
        )
        slot_gt, slot_eq = self.reserve(gt, eq, TOTALS + 2, tid, ctl, scan)
        above = Int32(k) - Int32(need)
        for q in cutlass.range(tid, queued, self.threads, unroll=1):
            vector = Int32(queue[q])
            words = self.vector_words(data_address, vector)
            for e in cutlass.range_constexpr(per):
                element = vector * per + e
                key = self.word_key(words, e, start + element, pinned)
                index = start + element
                if cutlass.const_expr(self.block_mode):  # noqa: SIM102
                    if key == Uint32(NEG_INF_KEY):
                        index = Int32(-1)
                if element < length:
                    if key > tau:
                        if slot_gt < above:
                            store_s32(out_row + Int64(slot_gt) * 4, index)
                        slot_gt = slot_gt + 1
                    if key == tau:
                        if slot_eq < Int32(need):
                            store_s32(out_row + Int64(above + slot_eq) * 4, index)
                        slot_eq = slot_eq + 1

    @cute.jit
    def emit(
        self,
        total_gt: Uint32,
        cutoff: Uint32,
        out_row: Int64,
        tid: Int32,
        ctl,
        hist,
        hist2,
        scan,
        g_key,
        g_idx,
        g_eq,
    ) -> None:
        """CTA 0, sampled path: write the row's k indices from the gathered
        survivors."""
        k, threads = self.k, self.threads
        count = Int32(total_gt)
        if total_gt >= Uint32(k):
            tau, need = self.radix_select(
                g_key, count, Uint32(k), hist, hist2, ctl, tid
            )
            above_n = Uint32(0)
            tie_n = Uint32(0)
            for i in cutlass.range(tid, count, threads, unroll=1):
                key = Uint32(g_key[i])
                if key > tau:
                    above_n = above_n + Uint32(1)
                if key == tau:
                    tie_n = tie_n + Uint32(1)
            slot_above, slot_tie, _, _ = self.scan2(above_n, tie_n, scan, tid)
            above = Int32(k) - Int32(need)
            for i in cutlass.range(tid, count, threads, unroll=1):
                key = Uint32(g_key[i])
                index = g_idx[i]
                if key > tau:
                    store_s32(out_row + Int64(Int32(slot_above)) * 4, index)
                    slot_above = slot_above + Uint32(1)
                if key == tau:
                    if Int32(slot_tie) < Int32(need):
                        store_s32(out_row + Int64(above + Int32(slot_tie)) * 4, index)
                    slot_tie = slot_tie + Uint32(1)
        else:
            for i in cutlass.range(tid, count, threads, unroll=1):
                store_s32(out_row + Int64(i) * 4, g_idx[i])
            for i in cutlass.range(tid, Int32(k) - count, threads, unroll=1):
                store_s32(out_row + Int64(count + i) * 4, g_eq[i])

    @cute.kernel
    def kernel(
        self,
        lengths: cute.Tensor,
        out: cute.Tensor,
        logits_address: Int64,
        row_bytes: Int64,
        width: Int32,
        out_stride: Int32,
        fallback: Int32,
    ):
        """One cluster a row (module docstring)."""
        k, cluster, threads, per = self.k, self.cluster, self.threads, self.per_vector
        samples = self.samples

        @cute.struct
        class Shared:
            data: cute.struct.Align[cute.struct.MemRange[Uint32, self.vectors * 4], 16]
            ctl: cute.struct.Align[cute.struct.MemRange[Uint32, CTL_WORDS], 16]
            scan: cute.struct.MemRange[Uint32, SCAN_WORDS]
            hist: cute.struct.Align[cute.struct.MemRange[Uint32, 256 * COPIES], 16]
            hist2: cute.struct.Align[cute.struct.MemRange[Uint32, 256], 16]
            hist_a: cute.struct.MemRange[Uint32, 256]
            hist_b: cute.struct.MemRange[Uint32, 256]
            g_idx: cute.struct.MemRange[Int32, CAPACITY]
            g_eq: cute.struct.MemRange[Int32, k]
            queue: cute.struct.MemRange[Uint16, self.vectors]
            sample: cute.struct.MemRange[Uint16, SAMPLES]
            g_key: cute.struct.MemRange[Uint16, CAPACITY]

        shared = cutlass.utils.SmemAllocator().allocate(Shared)

        def view(member, size):
            return cute.make_tensor(member.data_ptr(), cute.make_layout(size))

        data = view(shared.data, self.vectors * 4)
        data_address = Int32(data.iterator.toint())
        ctl = view(shared.ctl, CTL_WORDS)
        scan = view(shared.scan, SCAN_WORDS)
        hist = view(shared.hist, 256 * COPIES)
        hist2 = view(shared.hist2, 256)
        hist_a = view(shared.hist_a, 256)
        hist_b = view(shared.hist_b, 256)
        g_idx = view(shared.g_idx, CAPACITY)
        g_eq = view(shared.g_eq, k)
        queue = view(shared.queue, self.vectors)
        sample = view(shared.sample, SAMPLES)
        g_key = view(shared.g_key, CAPACITY)

        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        row = cta // cluster
        if tid < CTL_WORDS:
            ctl[tid] = Uint32(0)
        if tid < 256:
            hist_a[tid] = Uint32(0)
            hist_b[tid] = Uint32(0)
        # Every CTA has started, and CTA 0's counters are zero, before any
        # distributed shared memory access (the wait comes after the loads).
        cute.arch.cluster_arrive()
        cute.arch.griddepcontrol_wait()
        tokens = cutlass.max(cutlass.min(lengths[row], width), Int32(0))
        n = tokens
        if cutlass.const_expr(self.block_mode):
            n = (tokens + 7) // 8
        pinned = n - 1  # block mode: the newest block
        base = logits_address + Int64(row) * row_bytes
        out_row = Int64(out.iterator.toint()) + Int64(row) * Int64(out_stride) * 4
        if n <= k:
            cute.arch.cluster_wait()
            if rank == 0:
                for slot in cutlass.range(tid, k, threads, unroll=1):
                    value = Int32(-1)
                    if slot < n:
                        value = slot
                        if cutlass.const_expr(self.block_mode):
                            if slot == pinned:
                                value = slot
                            else:
                                w = load_u32x4(base + Int64(slot) * 16)
                                if block_key(w[0], w[1], w[2], w[3]) == Uint32(
                                    NEG_INF_KEY
                                ):
                                    value = Int32(-1)
                    store_s32(out_row + Int64(slot) * 4, value)
        else:
            # Chunks start on a 16-byte vector.
            align = 1 if self.block_mode else 8
            chunk = ((n + cluster - 1) // cluster + align - 1) // align * align
            start = rank * chunk
            length = cutlass.max(cutlass.min(n - start, chunk), Int32(0))
            element_bytes = 16 if self.block_mode else 2
            regs = cute.make_rmem_tensor((VECTORS, 4), Uint32)
            for j in cutlass.range_constexpr(VECTORS):
                first = (j * threads + tid) * per
                for w in cutlass.range_constexpr(4):
                    regs[j, w] = Uint32(0)
                if first < length:
                    words = load_u32x4(base + Int64(start + first) * element_bytes)
                    for w in cutlass.range_constexpr(4):
                        regs[j, w] = words[w]
            if cutlass.const_expr(self.vector_count > VECTORS):
                # A wide row's other vectors go through registers to shared
                # memory only.
                more = cute.make_rmem_tensor((self.vector_count - VECTORS, 4), Uint32)
                if VECTORS * threads * per < length:
                    for b in cutlass.range_constexpr(self.vector_count - VECTORS):
                        first = ((VECTORS + b) * threads + tid) * per
                        if first < length:
                            words = load_u32x4(
                                base + Int64(start + first) * element_bytes
                            )
                            for w in cutlass.range_constexpr(4):
                                more[b, w] = words[w]
                    for b in cutlass.range_constexpr(self.vector_count - VECTORS):
                        vector = (VECTORS + b) * threads + tid
                        if vector * per < length:
                            shared_store_u32x4(
                                data_address + vector * 16,
                                more[b, 0],
                                more[b, 1],
                                more[b, 2],
                                more[b, 3],
                            )
            for j in cutlass.range_constexpr(VECTORS):
                if (j * threads + tid) * per < length:
                    shared_store_u32x4(
                        data_address + (j * threads + tid) * 16,
                        regs[j, 0],
                        regs[j, 1],
                        regs[j, 2],
                        regs[j, 3],
                    )
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()
            use_direct = Int32(1)
            cutoff = Uint32(0)
            total_gt = Uint32(0)
            if cutlass.const_expr(self.block_mode):
                cute.arch.cluster_wait()
            else:
                if (n > self.direct_elements) & (fallback == 0):
                    use_direct = Int32(0)
                # Samples: CTA c's valid ones spread over its chunk, at the
                # same density in every CTA; the rest are key 0.
                key = Uint32(0)
                valid_samples = (samples * length) // chunk
                if (use_direct == 0) & (tid < valid_samples):
                    offset = ((2 * tid + 1) * length) // (2 * valid_samples)
                    key = order_key(
                        Uint32(data[offset >> 1]) >> (Uint32(offset & 1) * Uint32(16))
                    )
                cute.arch.cluster_wait()
                if use_direct == 0:
                    if tid < samples:
                        address = (
                            Int32(sample.iterator.toint()) + (rank * samples + tid) * 2
                        )
                        for peer in cutlass.range_constexpr(cluster):
                            cluster_store_u16(peer_address(address, peer), key)
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                    total_samples = Int32(0)
                    for c in cutlass.range_constexpr(cluster):
                        c_length = cutlass.max(
                            cutlass.min(n - c * chunk, chunk), Int32(0)
                        )
                        total_samples = total_samples + (samples * c_length) // chunk
                    q = (SURVIVOR_FACTOR * k * samples + chunk - 1) // chunk
                    q = cutlass.max(cutlass.min(q, total_samples), Int32(1))
                    cutoff, _ = self.radix_select(
                        sample, SAMPLES, Uint32(q), hist, hist2, ctl, tid
                    )
                    total_gt, total_eq = self.gather(
                        regs,
                        data_address,
                        cutoff,
                        start,
                        length,
                        pinned,
                        tid,
                        ctl,
                        scan,
                        queue,
                        g_key,
                        g_idx,
                        g_eq,
                    )
                    if (total_gt > Uint32(CAPACITY)) | (
                        total_gt + total_eq < Uint32(k)
                    ):
                        use_direct = Int32(1)
            if use_direct != 0:
                self.direct(
                    regs,
                    data_address,
                    start,
                    length,
                    pinned,
                    tid,
                    out_row,
                    ctl,
                    hist,
                    hist_a,
                    hist_b,
                    scan,
                    queue,
                )
            # CTA 0 exits only after every peer is done with its shared memory.
            cute.arch.cluster_arrive()
            if rank == 0:
                if cutlass.const_expr(not self.block_mode):  # noqa: SIM102
                    if use_direct == 0:
                        self.emit(
                            total_gt,
                            cutoff,
                            out_row,
                            tid,
                            ctl,
                            hist,
                            hist2,
                            scan,
                            g_key,
                            g_idx,
                            g_eq,
                        )
                cute.arch.cluster_wait()


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to ``tensor``'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _fake(dtype: type[cutlass.Numeric], align: int):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


@cache
def _kernel(
    device_index: int, k: int, cluster: int, vector_count: int, block_mode: bool
):
    """``DecodeTopK`` compiled for a device, k, cluster size, staged vectors
    a thread and mode."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            DecodeTopK(k, cluster, vector_count, block_mode),
            _fake(Uint32, 16),
            _fake(Int32, 4),
            _fake(Int32, 4),
            Int32(1),
            Int64(16),
            Int32(8),
            Int32(k),
            Int32(0),
            make_fake_stream(),
        )


def threads_for(cluster: int) -> int:
    """Threads a CTA: 1024 at 4 CTAs a row, else 512."""
    return 1024 if cluster == 4 else 512


def vector_choices(cluster: int) -> tuple[int, ...]:
    """The 16-byte vectors a thread may stage: 8, or 16 at 512 threads (1024
    threads have the shared memory for 8)."""
    return (8,) if cluster == 4 else (8, 16)


def vectors_for(width: int, cluster: int) -> int | None:
    """The fewest staged vectors a thread whose cluster holds a row of
    ``width`` logits (a 16-byte vector: 8 logits, or one block), or None."""
    for count in vector_choices(cluster):
        if width // 8 <= cluster * threads_for(cluster) * count:
            return count
    return None


@cache
def _resident_clusters(device_index: int, cluster: int) -> int:
    """Clusters of ``cluster`` CTAs the device runs at once, one CTA an SM
    (the kernel's shared memory allows no more): 36 of 4, 15 of 8 and 7 of 16
    on a 152-SM GB300."""
    with torch.accelerator.device_index(device_index):
        return cutlass.utils.HardwareInfo(device_index).get_max_active_clusters(cluster)


def cluster_for(rows: int, width: int, device_index: int) -> int:
    """CTAs a row: 8 while every row's cluster is resident at once, else 4.
    Rows wider than 4 CTAs hold (the ratio-2 layers' at a 1M context) always
    take 8, and rows wider than 8 CTAs hold (the candidate source's) 16, in
    waves when they outnumber the resident clusters."""
    if vectors_for(width, 8) is None:
        return 16
    if rows <= _resident_clusters(device_index, 8) or vectors_for(width, 4) is None:
        return 8
    return 4


def precompile(device_index: int, width: int, k: int, block_mode: bool) -> None:
    """Compile the ``DecodeTopK`` variants `decode_topk` launches on rows of
    ``width`` logits for ``k`` and ``block_mode``, at any row count, and read
    the cluster occupancy it plans with, before graph capture."""
    if k not in TOP_KS:
        return
    if vectors_for(width, 8) is not None:
        clusters = (8,) if vectors_for(width, 4) is None else (8, 4)
    elif vectors_for(width, 16) is not None:
        clusters = (16,)
    else:
        return
    _resident_clusters(device_index, clusters[0])
    for cluster in clusters:
        vector_count = vectors_for(width, cluster)
        assert vector_count is not None
        _kernel(device_index, k, cluster, vector_count, block_mode)


def supports_decode_topk(
    logits: torch.Tensor, rows: int, k: int, block_mode: bool
) -> bool:
    """Whether `decode_topk` takes these BF16 logits (16-byte aligned rows
    whose width, a multiple of 8, fits one cluster of at most 16 CTAs) for
    ``rows`` rows. Rows that need 16 CTAs are taken for at most two waves of
    clusters: beyond, the waves cost more than the upcast and the stock
    top-k."""
    if logits.dtype != torch.bfloat16 or logits.dim() != 2 or logits.stride(1) != 1:
        return False
    width = logits.shape[1]
    if vectors_for(width, 8) is None and (
        vectors_for(width, 16) is None
        or rows > 2 * _resident_clusters(logits.device.index, 16)
    ):
        return False
    return (
        0 < rows <= 64
        and k in TOP_KS
        and width % 8 == 0
        and logits.stride(0) % 8 == 0
        and logits.data_ptr() % 16 == 0
    )


def decode_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    block_mode: bool,
    fallback: bool = False,
) -> None:
    """Launch `DecodeTopK` on the current stream.

    Args:
        logits: [rows, width] BF16, row ``r`` valid below ``lengths[r]``
            (see `supports_decode_topk`).
        lengths: Contiguous [rows] int32 valid lengths in logits columns.
        out: [rows, k] int32 (unit column stride) for the indices: tokens, or
            8-token blocks in block mode (-1 for blocks scoring -inf).
        block_mode: Select 8-token blocks (the candidate source) instead of
            tokens.
        fallback: Force the direct path (tests).

    """
    rows, width = logits.shape
    k = out.shape[1]
    assert supports_decode_topk(logits, rows, k, block_mode)
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[0] == rows
    assert (
        lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and lengths.numel() == rows
    )
    cluster = cluster_for(rows, width, logits.device.index)
    vector_count = vectors_for(width, cluster)
    _kernel(logits.device.index, k, cluster, vector_count, block_mode)(
        _pointer(Uint32, logits, 16),
        _pointer(Int32, lengths, 4),
        _pointer(Int32, out, 4),
        Int32(rows),
        Int64(logits.stride(0) * 2),
        Int32(width),
        Int32(out.stride(0)),
        Int32(1 if fallback else 0),
        current_cu_stream(),
    )
