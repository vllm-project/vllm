# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page selection, sparse attention and o_proj's MXFP8 input, in CuTe DSL.

For each query row (16 heads on one KV head): its 16 pages, the top 15 of its
history pages by index score (ties to the lower page; every history page
when there are at most 15) and its local page, attended causally; the output,
rounded to BF16 as the stock attention stores it, is quantized to MXFP8 as
o_proj's input (E4M3 values, E8M0 scales in the 128x4 layout).

``ctas`` CTAs of eight warps take one row; CTA ``c`` takes ``16 / ctas`` of
its pages, a warp 32 tokens of a page at a time. Each CTA first ranks the
row's pages from the (score, physical page) pairs the indexer wrote (warp 0:
the 15th best of the lanes' best pages bounds the survivors, which are ranked
against each other). A warp's step::

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

from .ptx import (
    arrive_count,
    e4m3_byte,
    e4m3x4_f16,
    f16x2,
    ld_cg_f32,
    ld_cg_f32x4,
    ld_f32,
    ld_f32_s32,
    ld_s32,
    ld_shared_f32x4,
    ld_u16,
    ld_u32,
    ld_u32x2,
    ld_u32x4,
    mma_f16,
    nvfp4_f16x8,
    prmt,
    scale_pairs,
    scale_quad,
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
CAPACITY = 128  # survivors of the ranking's first pass
LOADS = 8  # scores a lane loads at once while ranking
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
        block_table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        scores: Int64,
        k_scale: Int64,
        v_scale: Int64,
        act: Int64,
        act_scale: Int64,
        partial: Int64,
        counters: Int64,
        logit_scale: Float32,
        rows: Int32,
        query_len: Int32,
        stream: cuda.CUstream,
    ):
        """Addresses: ``q`` E4M3 [rows, 2048]; ``kv`` NVFP4 pages (K slot,
        V slot) ``page_stride`` bytes apart; ``scores`` the indexer's;
        ``k_scale``, ``v_scale`` FP32 global scales; ``act`` E4M3 [rows,
        2048] and ``act_scale`` its 128x4 scales. ``logit_scale`` is the
        softmax scale times the query scale and log2(e)."""
        self.kernel(
            q,
            kv,
            page_stride,
            block_table,
            table_stride,
            seq_lens,
            scores,
            k_scale,
            v_scale,
            act,
            act_scale,
            partial,
            counters,
            logit_scale,
            query_len,
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
        block_table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        scores: Int64,
        k_scale: Int64,
        v_scale: Int64,
        act: Int64,
        act_scale: Int64,
        partial: Int64,
        counters: Int64,
        logit_scale: Float32,
        query_len: Int32,
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
        ranked = smem.allocate_tensor(Int32, cute.make_layout(16), 16)
        surv_v = smem.allocate_tensor(Float32, cute.make_layout(CAPACITY), 16)
        surv_j = smem.allocate_tensor(Int32, cute.make_layout((2, CAPACITY)), 16)
        part_ml = smem.allocate_tensor(Float32, cute.make_layout((2, WARPS, HEADS)), 16)
        part_o = smem.allocate_tensor(
            Float32, cute.make_layout((WARPS, 16, 32, 4), stride=(2048, 128, 4, 1)), 16
        )
        arrived = smem.allocate_tensor(Int32, cute.make_layout(1), 4)

        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        b = r // query_len
        length = ld_s32(seq_lens + Int64(b) * 4)
        pos = length - query_len + r % query_len
        valid = (length > 0) & (pos >= 0)
        local = Int32(0)
        if valid:
            local = pos // PAGE
        scale = logit_scale * ld_f32(k_scale)
        v_global = ld_f32(v_scale)

        # The query's A fragments: heads g and g + 8, dims 32t .. 32t + 31 as
        # F16 pairs (pair p: dims 32t + 2p, + 1).
        qw = cute.make_rmem_tensor((2, 16), Uint32)
        for hh in cutlass.range_constexpr(2):
            for half in cutlass.range_constexpr(2):
                words = ld_u32x4(
                    q
                    + Int64(r) * Q_SIZE
                    + Int64((g + 8 * hh) * HEAD_DIM + 32 * t + 16 * half)
                )
                for i in cutlass.range_constexpr(4):
                    pair = e4m3x4_f16(words[i])
                    qw[hh, 8 * half + 2 * i] = pair[0]
                    qw[hh, 8 * half + 2 * i + 1] = pair[1]

        if warp == 0:
            # The first 15 pages (the whole history of a short row) and the
            # local page, read alongside the ranking.
            row_table = block_table + Int64(b) * Int64(table_stride) * 4
            early = ld_s32(row_table + Int64(cute.min(lane, HISTORY - 1)) * 4)
            last = ld_s32(row_table + Int64(local) * 4)
            if valid & (local > HISTORY):
                self.rank(
                    scores + Int64(r) * SCORE_STRIDE * 8,
                    local,
                    ranked,
                    surv_v,
                    surv_j,
                    lane,
                )
            cute.arch.sync_warp()
            phys = Int32(-1)
            if valid & (lane < 16):
                if lane == 15:
                    phys = last
                elif local > HISTORY:
                    phys = ranked[lane]
                elif lane < local:
                    phys = early
            if lane < 16:
                sel[lane] = phys
        cute.arch.sync_threads()

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
                limit = pos % PAGE
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
    def rank(
        self,
        row: Int64,
        count: Int32,
        ranked: cute.Tensor,
        surv_v: cute.Tensor,
        surv_j: cute.Tensor,
        lane: Int32,
    ):
        """``ranked[k]``: the physical page of rank ``k`` < 15 among the
        ``count`` > 15 (score, physical page) pairs at ``row`` (by score, ties
        to the lower page). One warp; eight loads a lane in flight."""
        neg_inf = Float32(-math.inf)
        best_v = neg_inf
        best_j = Int32(BIG) + lane
        vals = cute.make_rmem_tensor((LOADS,), Float32)
        pages = cute.make_rmem_tensor((LOADS,), Int32)
        for first in cutlass.range(0, count, 32 * LOADS, unroll=1):
            self.load_scores(row, first, count, vals, pages, lane)
            for i in cutlass.range_constexpr(LOADS):
                if vals[i] > best_v:
                    best_v = vals[i]
                    best_j = first + 32 * i + lane
        # The 15th of the lanes' best pages bounds every selected page.
        place = Int32(0)
        for k in cutlass.range_constexpr(32):
            other_v = cute.arch.shuffle_sync(best_v, k)
            other_j = cute.arch.shuffle_sync(best_j, k)
            if (other_v > best_v) | ((other_v == best_v) & (other_j < best_j)):
                place = place + 1
        hit = cute.arch.vote_ballot_sync(place == HISTORY - 1)
        src = cute.arch.popc(hit - Uint32(1))
        bound_v = cute.arch.shuffle_sync(best_v, src)
        bound_j = cute.arch.shuffle_sync(best_j, src)
        # Survivors: every pair at least the bound, in page order (a row of at
        # most 256 pages still holds its pairs in registers).
        survivors = Int32(0)
        if count <= 32 * LOADS:
            survivors = self.survive(
                vals, pages, 0, count, bound_v, bound_j, survivors, surv_v, surv_j, lane
            )
        else:
            for first in cutlass.range(0, count, 32 * LOADS, unroll=1):
                self.load_scores(row, first, count, vals, pages, lane)
                survivors = self.survive(
                    vals,
                    pages,
                    first,
                    count,
                    bound_v,
                    bound_j,
                    survivors,
                    surv_v,
                    surv_j,
                    lane,
                )
        cute.arch.sync_warp()
        if survivors <= 32:
            # One survivor a lane, ranked against the others by shuffles.
            v = neg_inf
            j = Int32(BIG) + lane
            if lane < survivors:
                v = surv_v[lane]
                j = surv_j[0, lane]
            place = Int32(0)
            for k in cutlass.range_constexpr(32):
                other_v = cute.arch.shuffle_sync(v, k)
                other_j = cute.arch.shuffle_sync(j, k)
                if (other_v > v) | ((other_v == v) & (other_j < j)):
                    place = place + 1
            if (lane < survivors) & (place < HISTORY):
                ranked[place] = surv_j[1, lane]
        elif survivors <= CAPACITY:
            for s in cutlass.range(lane, survivors, 32, unroll=1):
                v = surv_v[s]
                j = surv_j[0, s]
                place = Int32(0)
                for s2 in cutlass.range(survivors, unroll=1):
                    v2 = surv_v[s2]
                    if (v2 > v) | ((v2 == v) & (surv_j[0, s2] < j)):
                        place = place + 1
                if place < HISTORY:
                    ranked[place] = surv_j[1, s]
        else:
            # Rare: rank by repeated extraction over every score.
            last_v = Float32(math.inf)
            last_j = Int32(-1)
            for k in cutlass.range(HISTORY, unroll=1):
                best_v = neg_inf
                best_j = Int32(BIG) + lane
                for j in cutlass.range(lane, count, 32, unroll=1):
                    v = ld_f32_s32(row + Int64(j) * 8)[0]
                    after = (v < last_v) | ((v == last_v) & (j > last_j))
                    if after & (v > best_v):
                        best_v = v
                        best_j = j
                for d in cutlass.range_constexpr(5):
                    other_v = cute.arch.shuffle_sync_bfly(best_v, 1 << d)
                    other_j = cute.arch.shuffle_sync_bfly(best_j, 1 << d)
                    if (other_v > best_v) | ((other_v == best_v) & (other_j < best_j)):
                        best_v = other_v
                        best_j = other_j
                if lane == 0:
                    ranked[k] = ld_f32_s32(row + Int64(best_j) * 8)[1]
                last_v = best_v
                last_j = best_j
        cute.arch.sync_warp()

    @cute.jit
    def load_scores(
        self,
        row: Int64,
        first: Int32,
        count: Int32,
        vals: cute.Tensor,
        pages: cute.Tensor,
        lane: Int32,
    ):
        """Pairs ``first + 32 i + lane`` (score -inf past ``count``), all
        loads in flight at once."""
        for i in cutlass.range_constexpr(LOADS):
            j = first + 32 * i + lane
            v = Float32(-math.inf)
            phys = Int32(0)
            if j < count:
                v, phys = ld_f32_s32(row + Int64(j) * 8)
            vals[i] = v
            pages[i] = phys

    @cute.jit
    def survive(
        self,
        vals: cute.Tensor,
        pages: cute.Tensor,
        first: Int32,
        count: Int32,
        bound_v: Float32,
        bound_j: Int32,
        survivors: Int32,
        surv_v: cute.Tensor,
        surv_j: cute.Tensor,
        lane: Int32,
    ) -> Int32:
        """Append the loaded pairs at least the bound to the survivors."""
        for i in cutlass.range_constexpr(LOADS):
            j = first + 32 * i + lane
            v = vals[i]
            keep = (j < count) & ((v > bound_v) | ((v == bound_v) & (j <= bound_j)))
            mask = cute.arch.vote_ballot_sync(keep)
            slot = survivors + cute.arch.popc(mask & cute.arch.lanemask_lt())
            if keep & (slot < CAPACITY):
                surv_v[slot] = v
                surv_j[0, slot] = j
                surv_j[1, slot] = pages[i]
            survivors = survivors + cute.arch.popc(mask)
        return survivors

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
            pair = scale_pairs(ks[n])
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
            pairs = scale_quad(vs[kk])
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
