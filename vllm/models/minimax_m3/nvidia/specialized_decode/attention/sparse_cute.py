# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page selection, sparse attention and o_proj's MXFP8 input, in CuTe DSL.

For each query row (16 heads on one KV head): its 16 pages, the top 15 of its
history pages by index score (ties to the lower page; every history page
when there are at most 15) and its local page, attended causally; the output,
rounded to BF16 as the stock attention stores it, is quantized to MXFP8 as
o_proj's input (E4M3 values, E8M0 scales in the 128x4 layout).

``ctas`` CTAs of eight warps take one row; CTA ``c`` takes ``16 / ctas`` of
its pages, a warp 32 tokens of a page at a time. The row's record (from
``prepare_step``) holds its local page and, for a row of at most 15 history
pages, all of them; each CTA of a longer row first ranks its pages from the
(score, physical page) pairs the indexer wrote (``Ranking``). A warp's
step::

    K (NVFP4 -> E4M3 -> F16) as the B operand of S = Q K^T (m16n8k16 F16,
    FP32; E4M3 values are exact in F16)
      └► online softmax in base 2 per head, P in F16
           └► O += P V (m16n8k16 F16; V NVFP4 -> E4M3 -> F16, the token
              quad of a lane sharing one 4-byte scale word)

A CTA merges its warps through shared memory and stores its partial (per
head the maximum, the sum and the unnormalized output) to a scratch buffer;
the row's last CTA to finish (a counter per row, reset by that CTA) merges
the partials, and four of its threads quantize each 32-value MXFP8 block.
Rows of padding requests (seq_len 0) attend nothing and write zeros.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_front_cute import (  # noqa: E501
    block_scale,
    inverse_scale,
)

from .indexer_cute import RECORD_BYTES
from .ptx import (
    arrive_count,
    atom_add_shared,
    e4m3_byte,
    e4m3x4_f16,
    f16x2,
    ld_cg_f32,
    ld_cg_f32x4,
    ld_f32,
    ld_f32_s32,
    ld_s32,
    ld_s32x2,
    ld_shared_f32x4,
    ld_u16,
    ld_u32,
    ld_u32x2,
    ld_u32x4,
    mma_f16,
    nvfp4_f16x8,
    prmt,
    ranks_before,
    scales_x4,
    st_f32,
    st_f32x4,
    st_shared_f32,
    st_shared_f32x4,
    st_u8,
    st_u32,
)

HEADS = 16
HEAD_DIM = 128
Q_SIZE = HEADS * HEAD_DIM
PAGE = 128
CODE_BYTES = PAGE * HEAD_DIM // 2
SLOT_BYTES = CODE_BYTES + PAGE * HEAD_DIM // 16  # K slot, then the V slot
HISTORY = 15  # ranked history pages; the local page is the 16th
SCORE_STRIDE = 8192
WARPS = 8
RANKERS = WARPS * 32
CAPACITY = 256  # survivors the ranking places
PAIRS = SCORE_STRIDE // RANKERS  # pairs a thread holds while ranking a long row
BIG = 1 << 24
LOG2E = 1.4426950408889634
MAX_ROWS = 32
MAX_CTAS = 8
STAGE_ROW = HEAD_DIM + 4  # floats a head of the staged output (padded)
ML_BYTES = MAX_ROWS * MAX_CTAS * 2 * HEADS * 4  # the partials' maxima and sums


def swizzled_scale_offset(row, block):
    """Byte of MXFP8 scale ``block`` (32 columns) of ``row`` < 128 in the
    128x4 layout."""
    return (block // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + block % 4


@cute.jit
def better(v: Float32, j: Int32, other_v: Float32, other_j: Int32):
    """Whether (``other_v``, ``other_j``) ranks before (``v``, ``j``): by
    score, ties to the lower page."""
    return (other_v > v) | ((other_v == v) & (other_j < j))


class Ranking:
    """A row's top-15 history pages by index score, ranked by the CTA's 256
    threads in shared memory it allocates (ties to the lower page).

    A row of at most 256 pages takes one pair a thread (loaded ahead): each
    warp ranks its 32 pairs by shuffles and stores them in rank order, and a
    pair among its warp's best 15 adds, by binary search, how many pairs of
    each other warp's list rank before it. A longer row (at most 8191 pages)
    takes 32 pairs a thread, all loads at once: the 15th best of 32 maxima,
    each over the pages of one residue mod 32, bounds the selection (15
    pages score at least that); the pages at least the bound survive and
    each is placed by counting the survivors that rank before it, or past
    ``CAPACITY`` survivors 15 rounds of extraction pick them. (Shared memory
    written under a runtime condition is written through local names: the
    DSL carries those, not this object.)"""

    def __init__(self, smem) -> None:
        # [rank, warp]: a warp's pairs in rank order (short rows), or each
        # thread's best pair at [tid % 32, tid // 32] (long rows).
        lists = cute.make_layout((32, WARPS), stride=(1, 32))
        self.list_v = smem.allocate_tensor(Float32, lists, 16)
        self.list_j = smem.allocate_tensor(Int32, lists, 16)
        self.surv_v = smem.allocate_tensor(Float32, cute.make_layout(CAPACITY), 16)
        self.surv_j = smem.allocate_tensor(Int32, cute.make_layout(CAPACITY), 16)
        self.surv_p = smem.allocate_tensor(Int32, cute.make_layout(CAPACITY), 16)
        self.warp_v = smem.allocate_tensor(Float32, cute.make_layout(WARPS + 1), 16)
        # The survivor count, then the bound's (or the extraction's) page.
        self.words = smem.allocate_tensor(Int32, cute.make_layout(WARPS + 2), 16)

    @cute.jit
    def rank(
        self,
        row: Int64,
        count: Int32,
        first_v: Float32,
        first_p: Int32,
        tid: Int32,
        sel: cute.Tensor,
    ):
        """For a row of ``count`` > 15 history pages, ``sel[k]``: the
        physical page of rank ``k`` < 15 among the (score, physical page)
        pairs at ``row``; ``first_v``, ``first_p``: pair ``tid``. Every
        thread calls it."""
        if count > HISTORY:
            if count <= RANKERS:
                self.rank_short(count, first_v, first_p, tid, sel)
            else:
                self.rank_long(row, count, tid, sel)

    @cute.jit
    def rank_short(
        self, count: Int32, first_v: Float32, first_p: Int32, tid: Int32, sel
    ):
        list_v = self.list_v
        list_j = self.list_j
        lane = tid % 32
        warp = tid // 32
        v = first_v
        if tid >= count:
            v = Float32(-math.inf)
        before = Int32(0)
        for k in cutlass.range_constexpr(32):
            other = cute.arch.shuffle_sync(v, k)
            before = before + ranks_before(v, tid, other, tid - lane + k)
        list_v[before, warp] = v
        list_j[before, warp] = tid
        cute.arch.sync_threads()
        if before < HISTORY:
            self.place_short(v, tid, first_p, sel)

    @cute.jit
    def place_short(self, v: Float32, j: Int32, phys: Int32, sel: cute.Tensor):
        """The pairs of every warp's list (its own included) that rank before
        (``v``, ``j``), up to 15 a list: a prefix, found in four steps."""
        place = Int32(0)
        for w in cutlass.range_constexpr(WARPS):
            at = Int32(0)
            for step in (8, 4, 2, 1):
                other_v = self.list_v[at + step - 1, w]
                other_j = self.list_j[at + step - 1, w]
                at = at + step * ranks_before(v, j, other_v, other_j)
            place = place + at
        if place < HISTORY:
            sel[place] = phys

    @cute.jit
    def rank_long(self, row: Int64, count: Int32, tid: Int32, sel: cute.Tensor):
        list_v = self.list_v
        list_j = self.list_j
        words = self.words
        if tid == 0:
            words[0] = 0
        vals = cute.make_rmem_tensor((PAIRS,), Float32)
        pages = cute.make_rmem_tensor((PAIRS,), Int32)
        for i in cutlass.range_constexpr(PAIRS):
            j = tid + RANKERS * i
            v, phys = ld_f32_s32(row + Int64(cute.min(j, count - 1)) * 8)
            if j >= count:
                v = Float32(-math.inf)
            vals[i] = v
            pages[i] = phys
        best_v = Float32(-math.inf)
        best_j = Int32(BIG)
        for i in cutlass.range_constexpr(PAIRS):
            if vals[i] > best_v:
                best_v = vals[i]
                best_j = tid + RANKERS * i
        list_v[tid % 32, tid // 32] = best_v
        list_j[tid % 32, tid // 32] = best_j
        cute.arch.sync_threads()
        if tid < 32:
            self.bound(tid)
        cute.arch.sync_threads()
        self.append(vals, pages, self.warp_v[WARPS], words[1], tid)
        cute.arch.sync_threads()
        survivors = words[0]
        if survivors <= CAPACITY:
            if tid < survivors:
                self.place(survivors, tid, sel)
        else:
            self.extract(vals, pages, tid, sel)

    @cute.jit
    def bound(self, lane: Int32):
        """Warp 0: the 15th best of the 32 residues' best pairs, into
        ``warp_v[WARPS]`` and ``words[1]``."""
        warp_v = self.warp_v
        words = self.words
        best_v = Float32(-math.inf)
        best_j = Int32(BIG) + lane
        for i in cutlass.range_constexpr(WARPS):
            v = self.list_v[lane, i]
            j = self.list_j[lane, i]
            if better(best_v, best_j, v, j):
                best_v = v
                best_j = j
        place = Int32(0)
        for k in cutlass.range_constexpr(32):
            other_v = cute.arch.shuffle_sync(best_v, k)
            other_j = cute.arch.shuffle_sync(best_j, k)
            place = place + ranks_before(best_v, best_j, other_v, other_j)
        if place == HISTORY - 1:
            warp_v[WARPS] = best_v
            words[1] = best_j

    @cute.jit
    def append(
        self,
        vals: cute.Tensor,
        pages: cute.Tensor,
        bound_v: Float32,
        bound_j: Int32,
        tid: Int32,
    ):
        """Append this thread's pairs at least the bound (pairs past the row
        score -inf, below it) to the survivors, up to ``CAPACITY`` (the count
        goes on): one shared atomic a warp, slots from a warp scan."""
        surv_v = self.surv_v
        surv_j = self.surv_j
        surv_p = self.surv_p
        lane = tid % 32
        kept = Int32(0)
        for i in cutlass.range_constexpr(PAIRS):
            kept = kept + 1 - ranks_before(vals[i], tid + RANKERS * i, bound_v, bound_j)
        ahead = kept
        for d in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(ahead, 1 << d)
            if lane >= (1 << d):
                ahead = ahead + other
        base = Int32(0)
        if lane == 31:
            base = atom_add_shared(self.words.iterator.toint(), ahead)
        slot = cute.arch.shuffle_sync(base, 31) + ahead - kept
        for i in cutlass.range_constexpr(PAIRS):
            j = tid + RANKERS * i
            keep = 1 - ranks_before(vals[i], j, bound_v, bound_j)
            if (keep == 1) & (slot < CAPACITY):
                surv_v[slot] = vals[i]
                surv_j[slot] = j
                surv_p[slot] = pages[i]
            slot = slot + keep

    @cute.jit
    def place(self, survivors: Int32, s: Int32, sel: cute.Tensor):
        """Survivor ``s``'s rank among the survivors; ``sel`` gets its page
        when it is below 15."""
        surv_v = self.surv_v
        surv_j = self.surv_j
        v = surv_v[s]
        j = surv_j[s]
        phys = self.surv_p[s]
        place = Int32(0)
        for s2 in cutlass.range(survivors, unroll=4):
            place = place + ranks_before(v, j, surv_v[s2], surv_j[s2])
        if place < HISTORY:
            sel[place] = phys

    @cute.jit
    def extract(
        self, vals: cute.Tensor, pages: cute.Tensor, tid: Int32, sel: cute.Tensor
    ):
        """Rare (more than ``CAPACITY`` survivors): 15 rounds, each the best
        pair after the previous round's."""
        warp_v = self.warp_v
        words = self.words
        warp_p = self.surv_p
        warp = tid // 32
        last_v = Float32(math.inf)
        last_j = Int32(-1)
        for k in cutlass.range(HISTORY, unroll=1):
            best_v = Float32(-math.inf)
            best_j = Int32(BIG)
            best_p = Int32(0)
            for i in cutlass.range_constexpr(PAIRS):
                j = tid + RANKERS * i
                after = better(vals[i], j, last_v, last_j)
                if after & better(best_v, best_j, vals[i], j):
                    best_v = vals[i]
                    best_j = j
                    best_p = pages[i]
            for d in cutlass.range_constexpr(5):
                other_v = cute.arch.shuffle_sync_bfly(best_v, 1 << d)
                other_j = cute.arch.shuffle_sync_bfly(best_j, 1 << d)
                other_p = cute.arch.shuffle_sync_bfly(best_p, 1 << d)
                if better(best_v, best_j, other_v, other_j):
                    best_v = other_v
                    best_j = other_j
                    best_p = other_p
            if tid % 32 == 0:
                warp_v[warp] = best_v
                words[2 + warp] = best_j
                warp_p[warp] = best_p
            cute.arch.sync_threads()
            best_v = warp_v[0]
            best_j = words[2]
            best_p = warp_p[0]
            for w in cutlass.range_constexpr(1, WARPS):
                if better(best_v, best_j, warp_v[w], words[2 + w]):
                    best_v = warp_v[w]
                    best_j = words[2 + w]
                    best_p = warp_p[w]
            if tid == 0:
                sel[k] = best_p
            last_v = best_v
            last_j = best_j
            cute.arch.sync_threads()


class SparseAttention:
    """The module docstring's kernel, ``ctas`` CTAs a row."""

    def __init__(self, ctas: int) -> None:
        assert ctas in (2, 4, 8)
        self.ctas = ctas
        self.slots = 16 // ctas
        self.chunks = self.slots * 4 // WARPS  # 32-token steps a warp

    @cute.jit
    def __call__(
        self,
        q: Int64,
        kv: Int64,
        page_stride: Int64,
        records: Int64,
        scores: Int64,
        k_scale: Int64,
        v_scale: Int64,
        act: Int64,
        act_scale: Int64,
        partial: Int64,
        counters: Int64,
        logit_scale: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Addresses: ``q`` E4M3 [rows, 2048]; ``kv`` NVFP4 pages (K slot,
        V slot) ``page_stride`` bytes apart; ``records`` the step's row
        records (``indexer_cute.IndexWork``); ``scores`` the indexer's;
        ``k_scale``, ``v_scale`` FP32 global scales; ``act`` E4M3 [rows,
        2048] and ``act_scale`` its 128x4 scales. ``logit_scale`` is the
        softmax scale times the query scale and log2(e)."""
        self.kernel(
            q,
            kv,
            page_stride,
            records,
            scores,
            k_scale,
            v_scale,
            act,
            act_scale,
            partial,
            counters,
            logit_scale,
        ).launch(
            grid=(self.ctas, rows, 1),
            block=(WARPS * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        kv: Int64,
        page_stride: Int64,
        records: Int64,
        scores: Int64,
        k_scale: Int64,
        v_scale: Int64,
        act: Int64,
        act_scale: Int64,
        partial: Int64,
        counters: Int64,
        logit_scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        c, r, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        g = lane // 4
        t = lane % 4
        neg_inf = Float32(-math.inf)

        smem = cutlass.utils.SmemAllocator()
        sel = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        ranking = Ranking(smem)
        part_ml = smem.allocate_tensor(Float32, cute.make_layout((2, WARPS, HEADS)), 16)
        part_o = smem.allocate_tensor(
            Float32, cute.make_layout((WARPS, 16, 32, 4), stride=(2048, 128, 4, 1)), 16
        )
        arrived = smem.allocate_tensor(Int32, cute.make_layout(1), 4)

        _, rows, _ = cute.arch.grid_dim()
        if rows > 8:
            cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        # The row's record, its query and (ahead of knowing whether the row
        # ranks) the pair of its page ``tid``, all in flight together.
        record = records + Int64(r) * RECORD_BYTES
        count, local_pos = ld_s32x2(record + 4 * HEADS)
        slot_phys = ld_s32(record + Int64(lane % 16) * 4)
        row_scores = scores + Int64(r) * SCORE_STRIDE * 8
        first_v, first_p = ld_f32_s32(row_scores + Int64(tid) * 8)
        scale = logit_scale * ld_f32(k_scale)
        v_global = ld_f32(v_scale)
        # Heads g and g + 8, dims 32t .. 32t + 31.
        q_words = cute.make_rmem_tensor((2, 2, 4), Uint32)
        for hh in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                words = ld_u32x4(
                    q
                    + Int64(r) * Q_SIZE
                    + Int64((g + 8 * hh) * HEAD_DIM + 32 * t + 16 * half)
                )
                for i in cutlass.range_constexpr(4):
                    q_words[hh, half, i] = words[i]
        # o_proj streams its weights once launched: up to eight rows launch it
        # once their pages are known when they rank at most 256 (its stream
        # does not delay their loads then), others before their wait.
        after_pages = Int32(0)
        if (rows <= 8) & (count <= RANKERS):
            after_pages = Int32(1)
        if (rows <= 8) & (after_pages == 0):
            cute.arch.griddepcontrol_launch_dependents()

        if tid < 16:
            sel[tid] = slot_phys
        ranking.rank(row_scores, count, first_v, first_p, tid, sel)
        cute.arch.sync_threads()
        if after_pages == 1:
            cute.arch.griddepcontrol_launch_dependents()

        # The query's A fragments as F16 pairs (pair p: dims 32t + 2p, + 1).
        qw = cute.make_rmem_tensor((2, 16), Uint32)
        for hh in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(4):
                    pair = e4m3x4_f16(q_words[hh, half, i])
                    qw[hh, 8 * half + 2 * i] = pair[0]
                    qw[hh, 8 * half + 2 * i + 1] = pair[1]

        # Running max and sum (heads g, g + 8) and O (dims 32t .. 32t + 31).
        stat = cute.make_rmem_tensor((4,), Float32)
        stat[0] = neg_inf
        stat[1] = neg_inf
        stat[2] = Float32(0.0)
        stat[3] = Float32(0.0)
        o = cute.make_rmem_tensor((64,), Float32)
        o.fill(0.0)
        for i in cutlass.range_constexpr(self.chunks):
            chunk = warp + WARPS * i
            slot = c * self.slots + chunk // 4
            phys = sel[slot]
            limit = Int32(PAGE - 1)
            if slot == HISTORY:
                limit = local_pos
            if phys >= 0:
                self.attend(
                    kv + Int64(phys) * page_stride,
                    chunk % 4,
                    limit,
                    qw,
                    stat,
                    o,
                    scale,
                    g,
                    t,
                )
        for k in cutlass.range_constexpr(2):
            total = stat[2 + k]
            total = total + cute.arch.shuffle_sync_bfly(total, 1)
            total = total + cute.arch.shuffle_sync_bfly(total, 2)
            stat[2 + k] = total

        # The CTA's warps merged.
        if t == 0:
            for k in cutlass.range_constexpr(2):
                part_ml[0, warp, g + 8 * k] = stat[k]
                part_ml[1, warp, g + 8 * k] = stat[2 + k]
        part_at = part_o.iterator.toint() + lane * 16
        for n in cutlass.range_constexpr(16):
            st_shared_f32x4(
                part_at + (warp * 16 + n) * 512,
                o[4 * n],
                o[4 * n + 1],
                o[4 * n + 2],
                o[4 * n + 3],
            )
        cute.arch.sync_threads()
        factor = cute.make_rmem_tensor((2, WARPS), Float32)
        for k in cutlass.range_constexpr(2):
            head = g + 8 * k
            top = part_ml[0, 0, head]
            for w in cutlass.range_constexpr(1, WARPS):
                top = cute.arch.fmax(top, part_ml[0, w, head])
            base = top
            if top == neg_inf:
                base = Float32(0.0)
            denom = Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                factor[k, w] = cute.math.exp2(part_ml[0, w, head] - base, fastmath=True)
                denom = denom + factor[k, w] * part_ml[1, w, head]
            if (warp == 0) & (t == 0):
                at = partial + Int64((r * self.ctas + c) * 2 * HEADS + head) * 4
                st_f32(at, top)
                st_f32(at + HEADS * 4, denom)
        # This thread's merged fragments (n = 2 warp, + 1; heads g, g + 8; dims
        # 32t + 16 (e % 2) + n) into the row's partials, 16 bytes each.
        o_row = partial + ML_BYTES + Int64((r * self.ctas + c) * Q_SIZE) * 4
        for nn in cutlass.range_constexpr(2):
            n = 2 * warp + nn
            parts = [
                ld_shared_f32x4(part_at + (w * 16 + n) * 512) for w in range(WARPS)
            ]
            acc = [Float32(0.0)] * 4
            for e in cutlass.range_constexpr(4):
                for w in cutlass.range_constexpr(WARPS):
                    acc[e] = acc[e] + factor[e // 2, w] * parts[w][e]
            st_f32x4(o_row + Int64((n * 32 + lane) * 16), *acc)
        cute.arch.sync_threads()
        if tid == 0:
            cute.arch.fence_acq_rel_gpu()
            arrived[0] = arrive_count(counters + Int64(r) * 4)
        cute.arch.sync_threads()
        # The row's last CTA merges the partials.
        if arrived[0] == self.ctas - 1:
            if tid == 0:
                st_u32(counters + Int64(r) * 4, Uint32(0))
            cute.arch.fence_acq_rel_gpu()
            self.finish(
                partial,
                r,
                warp,
                lane,
                act,
                act_scale,
                v_global,
                part_o.iterator.toint(),
            )

    @cute.jit
    def finish(
        self,
        partial: Int64,
        r: Int32,
        warp: Int32,
        lane: Int32,
        act: Int64,
        act_scale: Int64,
        v_global: Float32,
        stage: Int32,
    ):
        """Row ``r``'s output from its CTAs' partials (the fragments of the
        merge above), rounded to BF16 into ``stage`` as [head][132], then
        quantized to MXFP8 by blocks of 32 values, four threads a block."""
        neg_inf = Float32(-math.inf)
        g = lane // 4
        t = lane % 4
        peaks = cute.make_rmem_tensor((2, self.ctas), Float32)
        sums = cute.make_rmem_tensor((2, self.ctas), Float32)
        vals = cute.make_rmem_tensor((self.ctas, 2, 4), Float32)
        for p in cutlass.range_constexpr(self.ctas):
            at = partial + Int64((r * self.ctas + p) * 2 * HEADS + g) * 4
            for k in cutlass.range_constexpr(2):
                peaks[k, p] = ld_cg_f32(at + 32 * k)
                sums[k, p] = ld_cg_f32(at + HEADS * 4 + 32 * k)
            o_row = partial + ML_BYTES + Int64((r * self.ctas + p) * Q_SIZE) * 4
            for nn in cutlass.range_constexpr(2):
                words = ld_cg_f32x4(o_row + Int64(((2 * warp + nn) * 32 + lane) * 16))
                for e in cutlass.range_constexpr(4):
                    vals[p, nn, e] = words[e]
        for k in cutlass.range_constexpr(2):
            top = peaks[k, 0]
            for p in cutlass.range_constexpr(1, self.ctas):
                top = cute.arch.fmax(top, peaks[k, p])
            base = top
            if top == neg_inf:
                base = Float32(0.0)
            denom = Float32(0.0)
            acc = [Float32(0.0)] * 4
            for p in cutlass.range_constexpr(self.ctas):
                f = cute.math.exp2(peaks[k, p] - base, fastmath=True)
                denom = denom + f * sums[k, p]
                for nn in cutlass.range_constexpr(2):
                    for half in cutlass.range_constexpr(2):
                        acc[2 * nn + half] = (
                            acc[2 * nn + half] + f * vals[p, nn, 2 * k + half]
                        )
            norm = Float32(0.0)
            if denom > Float32(0.0):
                norm = v_global / denom
            for nn in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    dim = 32 * t + 16 * half + 2 * warp + nn
                    st_shared_f32(
                        stage + ((g + 8 * k) * STAGE_ROW + dim) * 4,
                        Float32(cutlass.BFloat16(acc[2 * nn + half] * norm)),
                    )
        cute.arch.sync_threads()
        tid = warp * 32 + lane
        head = tid // 16
        out = [
            *ld_shared_f32x4(stage + (head * STAGE_ROW + 8 * (tid % 16)) * 4),
            *ld_shared_f32x4(stage + (head * STAGE_ROW + 8 * (tid % 16) + 4) * 4),
        ]
        amax = Float32(0.0)
        for x in out:
            amax = cute.arch.fmax(amax, cute.arch.fmax(x, -x))
        for d in cutlass.range_constexpr(2):
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << d))
        code = block_scale(amax)
        inverse = inverse_scale(code)
        row = act + Int64(r) * Q_SIZE + Int64(tid * 8)
        for half in cutlass.range_constexpr(2):
            word = Uint32(0)
            for i in cutlass.range_constexpr(4):
                word = word | (e4m3_byte(out[4 * half + i] * inverse) << (8 * i))
            st_u32(row + 4 * half, word)
        if tid % 4 == 0:
            st_u8(act_scale + Int64(swizzled_scale_offset(r, tid // 4)), code)

    @cute.jit
    def attend(
        self,
        page: Int64,
        quarter: Int32,
        limit: Int32,
        qw: cute.Tensor,
        stat: cute.Tensor,
        o: cute.Tensor,
        scale: Float32,
        g: Int32,
        t: Int32,
    ):
        """Tokens 32 ``quarter`` .. + 31 of ``page`` (those up to ``limit``)
        into this warp's softmax state and O."""
        neg_inf = Float32(-math.inf)
        first = quarter * 32
        values = page + SLOT_BYTES
        # K of token 8-tile n: lane group g's token; V: the quad 4t of each
        # 16-token MMA step, bytes 8g .. 8g + 7 (dims 16g ..).
        kw = cute.make_rmem_tensor((4, 4), Uint32)
        ks = cute.make_rmem_tensor((4,), Uint32)
        vw = cute.make_rmem_tensor((2, 4, 2), Uint32)
        vs = cute.make_rmem_tensor((2,), Uint32)
        for n in cutlass.range_constexpr(4):
            token = first + 16 * (n // 2) + 4 * (g // 2) + 2 * (n % 2) + g % 2
            words = ld_u32x4(page + Int64(token * 64 + 16 * t))
            for i in cutlass.range_constexpr(4):
                kw[n, i] = words[i]
            ks[n] = ld_u16(page + Int64(CODE_BYTES + token * 8 + 2 * t))
        for kk in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(4):
                token = first + 16 * kk + 4 * t + i
                words = ld_u32x2(values + Int64(token * 64 + 8 * g))
                vw[kk, i, 0] = words[0]
                vw[kk, i, 1] = words[1]
            quad = (first + 16 * kk) // 4 + t
            vs[kk] = ld_u32(values + Int64(CODE_BYTES + quad * 32 + 4 * g))

        # S = Q K^T: columns 2t, 2t + 1 of tile n are tokens
        # first + 16 (n // 2) + 4t + 2 (n % 2) + {0, 1}.
        s = cute.make_rmem_tensor((16,), Float32)
        for n in cutlass.range_constexpr(4):
            pair = scales_x4(ks[n], 2)
            keys = [nvfp4_f16x8(kw[n, i], pair[i // 2]) for i in range(4)]
            acc = [Float32(0.0)] * 4
            # Step j's columns 2t, 2t + 1, 8 + 2t, 9 + 2t: dims 32t + 4j .. + 3.
            for j in cutlass.range_constexpr(8):
                m = 2 * (j % 2)
                acc = mma_f16(
                    (qw[0, 2 * j], qw[1, 2 * j], qw[0, 2 * j + 1], qw[1, 2 * j + 1]),
                    (keys[j // 2][m], keys[j // 2][m + 1]),
                    acc,
                )
            for e in cutlass.range_constexpr(2):
                token = first + 16 * (n // 2) + 4 * t + 2 * (n % 2) + e
                x0 = acc[e] * scale
                x1 = acc[2 + e] * scale
                if token > limit:
                    x0 = neg_inf
                    x1 = neg_inf
                s[4 * n + e] = x0
                s[4 * n + 2 + e] = x1

        # Online softmax (base 2) of heads g (k = 0) and g + 8 (k = 1).
        alpha = cute.make_rmem_tensor((2,), Float32)
        for k in cutlass.range_constexpr(2):
            top = s[2 * k]
            for n in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(2):
                    top = cute.arch.fmax(top, s[4 * n + 2 * k + e])
            top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, 1))
            top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, 2))
            top = cute.arch.fmax(top, stat[k])
            base = top
            if top == neg_inf:
                base = Float32(0.0)
            alpha[k] = cute.math.exp2(stat[k] - base, fastmath=True)
            total = stat[2 + k] * alpha[k]
            for n in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(2):
                    p = cute.math.exp2(s[4 * n + 2 * k + e] - base, fastmath=True)
                    s[4 * n + 2 * k + e] = p
                    total = total + p
            stat[k] = top
            stat[2 + k] = total
        for n in cutlass.range_constexpr(16):
            for e in cutlass.range_constexpr(4):
                o[4 * n + e] = o[4 * n + e] * alpha[e // 2]

        # O += P V, 16 tokens a step.
        for kk in cutlass.range_constexpr(2):
            pa = (
                f16x2(s[8 * kk], s[8 * kk + 1]),
                f16x2(s[8 * kk + 2], s[8 * kk + 3]),
                f16x2(s[8 * kk + 4], s[8 * kk + 5]),
                f16x2(s[8 * kk + 6], s[8 * kk + 7]),
            )
            pairs = scales_x4(vs[kk], 4)
            h = cute.make_rmem_tensor((4, 8), Uint32)
            for i in cutlass.range_constexpr(4):
                for half in cutlass.range_constexpr(2):
                    words = nvfp4_f16x8(vw[kk, i, half], pairs[i])
                    for m in cutlass.range_constexpr(4):
                        h[i, 4 * half + m] = words[m]
                # Masked tokens may hold any bytes: keep them out of the MMA.
                if first + 16 * kk + 4 * t + i > limit:
                    for m in cutlass.range_constexpr(8):
                        h[i, m] = Uint32(0)
            for m in cutlass.range_constexpr(8):
                for e in cutlass.range_constexpr(2):
                    selector = 0x5410 if e == 0 else 0x7632
                    n = 2 * m + e
                    res = mma_f16(
                        pa,
                        (
                            prmt(h[0, m], h[1, m], selector),
                            prmt(h[2, m], h[3, m], selector),
                        ),
                        (o[4 * n], o[4 * n + 1], o[4 * n + 2], o[4 * n + 3]),
                    )
                    for x in cutlass.range_constexpr(4):
                        o[4 * n + x] = res[x]
