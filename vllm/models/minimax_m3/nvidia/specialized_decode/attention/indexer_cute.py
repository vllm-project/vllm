# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index scores of the low-batch attention, in CuTe DSL.

For each request whose last query has more than 15 history pages, every
query row's score of every history page of the last row: the maximum over the
page's 128 tokens of the dot product of the E4M3 index query with the NVFP4
index keys (E4M3-rounded, see ``ptx``), accumulated in FP32. Requests with at
most 15 history pages select all of them and are not scored.

``IndexWork`` runs once per decode step: it lists those pages, flattened in
request order, as (request * 8192 + page, physical page) after their count,
and writes each query row's record (its pages when it does not rank them,
its local page and position) for the attention.
``IndexScores`` runs per layer on a persistent grid of ``ctas`` CTAs of
eight warps: CTA ``c`` takes list entries ``c``, ``c + ctas``, ..., a page a
step, each warp 16 of its tokens against the request's (up to eight) query
rows with eight m16n8k16 F16 MMAs (E4M3 values are exact in F16). The TMA
unit copies each page and its request's query rows into one of ``STAGES``
shared memory stages ahead of the warps (one bulk copy each, completing on
the stage's barrier), and after every 32 pages the CTA writes their scores,
each with its physical page, for the selection.
Everything is sized from the device seq_lens: the grids depend on the device
and the request count only.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from .ptx import (
    bulk_copy,
    e4m3x4_f16,
    ld_s32,
    ld_s32x2,
    ld_shared_u32,
    ld_shared_u32x4,
    mma_f16,
    nvfp4_f16x8,
    scales_x4,
    st_f32_s32,
    st_s32,
    st_s32x2,
)

PAGE = 128
HEAD_DIM = 128
CODE_BYTES = PAGE * HEAD_DIM // 2  # a page's E2M1 codes; its scales follow
TOPK = 16  # 15 ranked history pages and the local page
SCORE_STRIDE = 8192  # pages a score row holds (the block table's width at 1M)
WARPS = 8
MAX_REQS = 32
STAGES = 4
PAGE_BYTES = PAGE * 72  # codes, then scales
STAGE_BYTES = PAGE_BYTES + 8 * HEAD_DIM  # a page and its request's query rows
SPLIT = 8  # IndexWork CTAs a request
ENTRIES = SCORE_STRIDE // SPLIT  # work list entries an IndexWork CTA writes
RECORD_BYTES = 128  # a query row's record (IndexWork)


def work_entries(ctas: int) -> int:
    """Entries of the work list buffer: every page of 32 requests and the
    reads past the end of a CTA's last batch."""
    return MAX_REQS * SCORE_STRIDE + 32 * ctas


@cute.jit
def ranked_pages(length: Int32) -> Int32:
    """History pages a request scores: those of its last query when it has
    more than 15."""
    pages = Int32(0)
    if length > 0:
        history = (length - 1) // PAGE
        if history >= TOPK:
            pages = history
    return pages


class IndexWork:
    """The step's work list (module docstring) and each query row's record:
    words 0-15 the physical pages of its 16 slots (-1 when unused; slots
    0-14 hold the history pages of a row with at most 15, the attention
    ranks the others), 16 its history pages when it ranks them (else 0), 17
    its position in its local page. A grid of (requests, ``SPLIT``) CTAs:
    CTA (b, y) lists request b's pages ``ENTRIES y`` .. and CTA (b, 0)
    writes b's records."""

    @cute.jit
    def __call__(
        self,
        block_table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        work: Int64,
        records: Int64,
        num_reqs: Int32,
        query_len: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            block_table, table_stride, seq_lens, work, records, num_reqs, query_len
        ).launch(
            grid=(num_reqs, SPLIT, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        block_table: Int64,
        table_stride: Int32,
        seq_lens: Int64,
        work: Int64,
        records: Int64,
        num_reqs: Int32,
        query_len: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        b, y, _ = cute.arch.block_idx()
        lane = tid % 32
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        length = Int32(0)
        if lane < num_reqs:
            length = ld_s32(seq_lens + Int64(lane) * 4)
        pages = ranked_pages(length)
        incl = pages
        for d in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_up(incl, 1 << d)
            if lane >= (1 << d):
                incl = incl + other
        total = cute.arch.shuffle_sync(incl, 31)
        first = cute.arch.shuffle_sync(incl - pages, b)
        count = cute.arch.shuffle_sync(pages, b)
        length = cute.arch.shuffle_sync(length, b)
        if (b == 0) & (y == 0) & (tid == 0):
            st_s32x2(work, total, total)
        row = block_table + Int64(b) * Int64(table_stride) * 4
        if y * ENTRIES < count:
            self.list_pages(row, work, b, y * ENTRIES + tid, first, count)
        if y == 0:
            self.write_records(row, records, b, length, query_len, tid)

    @cute.jit
    def list_pages(
        self, row: Int64, work: Int64, b: Int32, j0: Int32, first: Int32, count: Int32
    ):
        """Entries ``j0 + 256 u`` (below ``count``), their table loads in
        flight together."""
        phys = cute.make_rmem_tensor((ENTRIES // 256,), Int32)
        for u in cutlass.range_constexpr(ENTRIES // 256):
            phys[u] = ld_s32(row + Int64(cute.min(j0 + 256 * u, count - 1)) * 4)
        for u in cutlass.range_constexpr(ENTRIES // 256):
            j = j0 + 256 * u
            if j < count:
                entry = work + Int64(first + j + 1) * 8
                st_s32x2(entry, b * SCORE_STRIDE + j, phys[u])

    @cute.jit
    def write_records(
        self,
        row: Int64,
        records: Int64,
        b: Int32,
        length: Int32,
        query_len: Int32,
        tid: Int32,
    ):
        """Word ``tid % 16`` of request ``b``'s row ``tid // 16`` (class
        docstring) and, from slot 0's thread, words 16 and 17."""
        i = tid // 16
        k = tid % 16
        if i < query_len:
            pos = length - query_len + i
            local = pos // PAGE
            phys = Int32(-1)
            if (length > 0) & (pos >= 0):
                if k == TOPK - 1:
                    phys = ld_s32(row + Int64(local) * 4)
                elif (local < TOPK) & (k < local):
                    phys = ld_s32(row + Int64(k) * 4)
            at = records + Int64(b * query_len + i) * RECORD_BYTES
            st_s32(at + 4 * k, phys)
            if k == 0:
                ranked = Int32(0)
                if (length > 0) & (pos >= 0) & (local >= TOPK):
                    ranked = local
                st_s32x2(at + 4 * TOPK, ranked, pos % PAGE)


class IndexScores:
    """``scores[row, page]`` for the listed pages (module docstring)."""

    def __init__(self, ctas: int) -> None:
        self.ctas = ctas

    @cute.jit
    def __call__(
        self,
        iq: Int64,
        cache: Int64,
        page_stride: Int64,
        work: Int64,
        scores: Int64,
        query_len: Int32,
        stream: cuda.CUstream,
    ):
        """Addresses: ``iq`` E4M3 [rows, 128]; ``cache`` NVFP4 index pages
        ``page_stride`` bytes apart; ``work`` the step's list; ``scores``
        (FP32 score, int32 physical page) [rows, 8192]."""
        self.kernel(iq, cache, page_stride, work, scores, query_len).launch(
            grid=(self.ctas, 1, 1),
            block=(WARPS * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        iq: Int64,
        cache: Int64,
        page_stride: Int64,
        work: Int64,
        scores: Int64,
        query_len: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        g = lane // 4
        t = lane % 4
        smem = cutlass.utils.SmemAllocator()
        stages = smem.allocate_array(
            cutlass.Uint8, STAGES * STAGE_BYTES, byte_alignment=128
        )
        full = smem.allocate_array(cutlass.Int64, STAGES, byte_alignment=8)
        red = smem.allocate_tensor(Float32, cute.make_layout((32, WARPS, 8)), 16)
        unit = smem.allocate_tensor(Int32, cute.make_layout((2, 32)), 16)
        keep = Uint32(0xFFFFFFFF)  # query columns past the request's rows are zero
        if g >= query_len:
            keep = Uint32(0)
        if tid == 0:
            for s in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(full + s, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        # This lane's pieces of a stage: keys of tokens 16 warp + g and + 8
        # (bytes 16t ..: dims 32t .. 32t + 31), their scale words, query row g.
        token = warp * 16 + g
        key_at = token * 64 + 16 * t
        scale_at = CODE_BYTES + token * 8 + 4 * (t // 2)
        query_at = PAGE_BYTES + cute.min(g, query_len - 1) * HEAD_DIM + 32 * t
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        total = ld_s32(work)
        steps = Int32(0)
        if cta < total:
            steps = (total - cta + self.ctas - 1) // self.ctas
        # The query's B fragments, converted again only when the request
        # changes from one page to the next.
        query = cute.make_rmem_tensor((8, 2), Uint32)
        query_of = Int32(-1)
        for first in cutlass.range(0, steps, 32, unroll=1):
            key, phys = ld_s32x2(
                work + (Int64(cta) + Int64(first + lane) * self.ctas + 1) * 8
            )
            count = cute.min(steps - first, 32)
            if warp == 0:
                unit[0, lane] = key
                unit[1, lane] = phys
            cute.arch.sync_warp()
            if tid == 0:
                for s in cutlass.range_constexpr(STAGES):
                    if s < count:
                        self.issue(
                            stages.toint(),
                            full,
                            s,
                            unit,
                            iq,
                            cache,
                            page_stride,
                            query_len,
                        )
            for k in cutlass.range(count, unroll=1):
                at = stages.toint() + (k % STAGES) * STAGE_BYTES
                cute.arch.mbarrier_wait(full + k % STAGES, (first + k) // STAGES % 2)
                a0 = ld_shared_u32x4(at + key_at)
                a1 = ld_shared_u32x4(at + key_at + 8 * 64)
                s0 = ld_shared_u32(at + scale_at) >> (16 * (t % 2))
                s1 = ld_shared_u32(at + scale_at + 8 * 8) >> (16 * (t % 2))
                request = unit[0, k] // SCORE_STRIDE
                if request != query_of:
                    self.convert_query(at + query_at, keep, query)
                    query_of = request
                # Every warp has read the stage: refill it.
                cute.arch.sync_threads()
                if (tid == 0) & (k + STAGES < count):
                    self.issue(
                        stages.toint(),
                        full,
                        k + STAGES,
                        unit,
                        iq,
                        cache,
                        page_stride,
                        query_len,
                    )
                # Dims 32t + 4j .. + 3 are MMA step j's columns 2t, 2t + 1,
                # 8 + 2t, 9 + 2t (the same order for keys and query).
                pairs = (scales_x4(s0, 2), scales_x4(s1, 2))
                keys = [
                    [nvfp4_f16x8(words[i], pairs[tok][i // 2]) for i in range(4)]
                    for tok, words in enumerate((a0, a1))
                ]
                acc = [Float32(0.0)] * 4
                for j in cutlass.range_constexpr(8):
                    lo = keys[0][j // 2]
                    hi = keys[1][j // 2]
                    m = 2 * (j % 2)
                    acc = mma_f16(
                        (lo[m], hi[m], lo[m + 1], hi[m + 1]),
                        (query[j, 0], query[j, 1]),
                        acc,
                    )
                # Rows 2t and 2t + 1: the maximum over the warp's 16 tokens.
                m0 = cute.arch.fmax(acc[0], acc[2])
                m1 = cute.arch.fmax(acc[1], acc[3])
                for d in cutlass.range_constexpr(3):
                    m0 = cute.arch.fmax(m0, cute.arch.shuffle_sync_bfly(m0, 4 << d))
                    m1 = cute.arch.fmax(m1, cute.arch.shuffle_sync_bfly(m1, 4 << d))
                if g == 0:
                    red[k, warp, 2 * t] = m0
                    red[k, warp, 2 * t + 1] = m1
            cute.arch.sync_threads()
            k = tid // 8
            row = tid % 8
            if (k < count) & (row < query_len):
                best = red[k, 0, row]
                for w in cutlass.range_constexpr(1, WARPS):
                    best = cute.arch.fmax(best, red[k, w, row])
                key_k = unit[0, k]
                index = (key_k // SCORE_STRIDE * query_len + row) * SCORE_STRIDE
                st_f32_s32(
                    scores + Int64(index + key_k % SCORE_STRIDE) * 8, best, unit[1, k]
                )
            cute.arch.sync_threads()

    @cute.jit
    def convert_query(self, at: Int32, keep: Uint32, query: cute.Tensor):
        """This lane's query words (dims 32t .. 32t + 31 of its row, zero past
        the request's rows) at shared address ``at`` as F16 pairs."""
        words = (*ld_shared_u32x4(at), *ld_shared_u32x4(at + 16))
        for j in cutlass.range_constexpr(8):
            pair = e4m3x4_f16(words[j] & keep)
            query[j, 0] = pair[0]
            query[j, 1] = pair[1]

    @cute.jit
    def issue(
        self,
        stages: Int32,
        full: cute.Pointer,
        k: Int32,
        unit: cute.Tensor,
        iq: Int64,
        cache: Int64,
        page_stride: Int64,
        query_len: Int32,
    ):
        """Bulk-copy step ``k``'s page and its request's query rows into stage
        ``k % STAGES`` (one thread)."""
        at = stages + (k % STAGES) * STAGE_BYTES
        mbar = full + k % STAGES
        rows = query_len * HEAD_DIM
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, PAGE_BYTES + rows)
        page = cache + Int64(unit[1, k]) * page_stride
        bulk_copy(at, page, Int32(PAGE_BYTES), mbar.toint())
        first_row = iq + Int64(unit[0, k] // SCORE_STRIDE) * Int64(rows)
        bulk_copy(at + PAGE_BYTES, first_row, rows, mbar.toint())
