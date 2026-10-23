# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy target head of one TP rank: RMSNorm, vocabulary-shard logits, top-1."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from . import _rt as rt

SHARD = rt.Core.SHARD_ROWS
HIDDEN = rt.Core.HIDDEN
CTAS = 148
PAGE_BYTES = 3 * HIDDEN * 2
STATIC_PAGES = 5624
TAIL_PAGES = 830
MAX_PAGES = 56
RING_SLOTS = 4
COMPUTE_WARPS = 8
COMPUTE_THREADS = 32 * COMPUTE_WARPS
PAGE_PARTIALS = MAX_PAGES * 3
ROW_PARTIALS = COMPUTE_WARPS * PAGE_PARTIALS
NEG_INF, INT_MAX = float("-inf"), 2147483647
BFLY = (16, 8, 4, 2, 1)
SMEM_BYTES = 222208


class DoneWords:
    """``Done`` words: ticket, per (row, CTA) a 2-word tagged record, 2 tail-claim
    counters.
    """

    TICKET = 0

    @staticmethod
    def record(row, cta):
        """First word of CTA ``cta``'s record for ``row``."""
        return 1 + (row * CTAS + cta) * 2

    @staticmethod
    def counter(rows: int) -> int:
        """First tail-claim counter."""
        return 1 + 2 * CTAS * rows


class Mailbox:
    """Peer mailbox: [2 parities][8 source ranks][4 rows] of 16-byte packets."""

    @staticmethod
    def packet(parity, rank, row):
        """Byte offset of ``rank``'s packet for ``row``."""
        return ((parity * 8 + rank) * 4 + row) * 16


def _ld8(vectors: cute.Tensor, i) -> list:
    return rt.Convert.unpack8(rt.Convert.ld_u32x4(vectors[i, None]))


@dataclass(frozen=True)
class LmHeadParams(rt.Params):
    """Kernel parameters of ``lm_head_top1_q1``."""

    X: cute.Pointer = rt.ptr("bfloat16")
    NormWeight: cute.Pointer = rt.ptr("bfloat16")
    LmHead: cute.Pointer = rt.ptr("uint8")
    Logits: cute.Pointer = rt.ptr("float32")
    Partials: cute.Pointer = rt.ptr("uint32")
    RankRecord: cute.Pointer = rt.ptr("uint32")
    Winner: cute.Pointer = rt.ptr("uint32")
    Done: cute.Pointer = rt.ptr("uint32")
    NormRows: cute.Pointer = rt.ptr("bfloat16")
    Tokens: cute.Pointer = rt.ptr("int32")
    peer_rank: Int32
    peer_base0: Int64
    peer_base1: Int64
    peer_base2: Int64
    peer_base3: Int64
    peer_base4: Int64
    peer_base5: Int64
    peer_base6: Int64
    peer_base7: Int64


class LmHeadLayout(rt.SmemLayout):
    """Dynamic shared memory, sized for 4 rows (the 1-row kernel uses the same map)."""

    ROWS = 4
    RING_GAP = 1024

    a = rt.SmemAlloc(SMEM_BYTES)
    ring_full = a.mbarriers(RING_SLOTS, arrivals=1)
    ring_empty = a.mbarriers(RING_SLOTS, arrivals=COMPUTE_THREADS)
    x_full = a.mbarriers(1, arrivals=1)
    ticket_full = a.mbarriers(1, arrivals=1)
    scratch = a.array(Float32, (ROWS, COMPUTE_WARPS, 2), align=1024, also=(Int32,))
    ticket = a.array(Uint32, 1, align=1024)
    page_ids = a.array(Int32, MAX_PAGES, align=256)
    partials = a.array(Float32, (ROWS, COMPUTE_WARPS, PAGE_PARTIALS), align=1024)
    norm = a.array(BFloat16, (ROWS, HIDDEN), align=1024)
    a.pad(RING_GAP)
    ring = a.array(Uint8, (RING_SLOTS, PAGE_BYTES), align=1024)
    assert ring_full.offset == 0  # the mbarrier pointer is the smem base


L = LmHeadLayout


@dataclass(frozen=True)
class LmHeadSmem:
    """Shared-memory pointers and typed views, plus the ``Done`` address."""

    mbar: cute.Pointer
    smem_addr: Int32
    scratch: cute.Tensor
    scratch_i: cute.Tensor
    ticket_slot: cute.Tensor
    page_ids: cute.Tensor
    partials: cute.Tensor
    norm: cute.Tensor
    norm_u4: cute.Tensor
    ring_u4: cute.Tensor
    done_addr: Int64

    @staticmethod
    def build(smem, R: int, done: cute.Pointer) -> "LmHeadSmem":
        """The views over ``smem`` for ``R`` rows."""

        def view(region, dtype, shape, stride=None):
            layout = cute.make_layout(shape, stride=stride)
            return rt.Memory.smem_view(smem, region.offset, dtype, layout)

        return LmHeadSmem(
            mbar=cute.recast_ptr(smem, dtype=cutlass.Int64),
            smem_addr=smem.toint(),
            scratch=view(L.scratch, Float32, L.scratch.size),
            scratch_i=view(L.scratch, Int32, L.scratch.size),
            ticket_slot=view(L.ticket, Uint32, 1),
            page_ids=view(L.page_ids, Int32, MAX_PAGES),
            partials=view(L.partials, Float32, R * ROW_PARTIALS),
            norm=view(L.norm, BFloat16, R * HIDDEN),
            norm_u4=view(L.norm, Uint32, (R * HIDDEN // 8, 4), (4, 1)),
            ring_u4=view(L.ring, Uint32, (RING_SLOTS * PAGE_BYTES // 16, 4), (4, 1)),
            done_addr=done.toint(),
        )


class LmHead(rt.RoleKernel):
    """Compute warps 0-7, loader warps 8-11 (warp 8 issues)."""

    Params = LmHeadParams
    GRID = CTAS
    SMEM_BYTES = SMEM_BYTES
    ROLES = (rt.Role("compute", 0, 7, 216, True), rt.Role("load", 8, 11, 72, False))
    ROWS = 1

    @cute.jit
    def prologue(self, p: LmHeadParams):
        """Own warp/lane form: ``Prologue.thread_ids()``."""
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cute.arch.get_dyn_smem(Uint8, alignment=1024)
        s = LmHeadSmem.build(smem, self.ROWS, p.Done)
        with rt.warp_leader(warp, 0):
            rt.Prologue.init_mbarriers(
                s.smem_addr,
                (
                    (L.ring_full.offset, RING_SLOTS, 1),
                    (L.ring_empty.offset, RING_SLOTS, COMPUTE_THREADS),
                    (L.x_full.offset, 2, 1),
                ),
            )
        cute.arch.sync_threads()
        return rt.ThreadIds(tid, bid, warp, lane), s

    @cute.jit
    def compute(self, p: LmHeadParams, s: LmHeadSmem, ids: rt.ThreadIds):
        """RMSNorm, page FMAs, the CTA's logits and best record, then (CTA 0) the
        exchange.
        """
        R = self.ROWS
        tid, bid, warp, lane = ids.tid, ids.bid, ids.warp, ids.lane
        gamma_g = cute.make_tensor(
            cute.recast_ptr(p.NormWeight, dtype=Uint32),
            cute.make_layout((3, 256, 4), stride=(1024, 4, 1)),
        )
        raw_gamma = [rt.Convert.ld_u32x4(gamma_g[j, tid, None]) for j in range(3)]
        self.set_regs("compute")
        launch_ticket = Uint32(0)
        if tid == 0:
            launch_ticket = cute.arch.atomic_add(p.Done, Uint32(1))
        rt.Sync.pdl_prologue()
        rows_u4 = cute.make_layout((R, 3, 256, 4), stride=(3072, 1024, 4, 1))
        x_g = cute.make_tensor(cute.recast_ptr(p.X, dtype=Uint32), rows_u4)
        x_raw = [
            [rt.Convert.ld_u32x4(x_g[q, j, tid, None]) for j in range(3)]
            for q in range(R)
        ]
        values = [[rt.Convert.unpack8(x_raw[q][j]) for j in range(3)] for q in range(R)]
        squares = [
            rt.Reduce.sum_squares([values[q][j][i] for j in range(3) for i in range(8)])
            for q in range(R)
        ]
        for step in cutlass.range_constexpr(5):
            squares = [
                rt.add(sq, cute.arch.shuffle_sync_down(sq, 16 >> step))
                for sq in squares
            ]
        scratch = s.scratch
        if lane == 0:
            for q in cutlass.range_constexpr(R):
                scratch[q * 8 + warp] = squares[q]
        rt.bar_sync(8, COMPUTE_THREADS)
        gamma = [rt.Convert.unpack8(raw_gamma[j]) for j in range(3)]
        norm_out = cute.make_tensor(s.norm_u4.iterator, rows_u4)
        for q in cutlass.range_constexpr(R):
            inverse = rt.Reduce.rms_inverse(
                rt.Reduce.ordered_sum(lambda w, q=q: scratch[q * 8 + w], 8)
            )
            for j in cutlass.range_constexpr(3):
                normed = cute.make_rmem_tensor(8, Float32)
                for i in cutlass.range_constexpr(8):
                    normed[i] = rt.mul(rt.mul(values[q][j][i], inverse), gamma[j][i])
                rt.Convert.st_f32_bf16x8(normed, norm_out[q, j, tid, None])
        rt.bar_sync(8, COMPUTE_THREADS)
        per_cta = 42 if R == 1 else 167
        element = bid * per_cta + tid
        if (tid < per_cta) & (element < HIDDEN * R):
            # A subscript-store target must be a plain name (the DSL marks it written).
            norm_rows = cute.make_tensor(p.NormRows, cute.make_layout(HIDDEN * R))
            norm_rows[element] = s.norm[element]
        ticket_slot = s.ticket_slot
        if tid == 0:
            ticket_slot[0] = launch_ticket
            cute.arch.mbarrier_arrive(s.mbar + L.ticket_full.offset // 8)
        nrm = [
            [
                _ld8(s.norm_u4, (q * HIDDEN + warp * 768 + j * 256 + lane * 8) // 8)
                for j in range(3)
            ]
            for q in range(R)
        ]
        first_page = bid * STATIC_PAGES // CTAS
        pages = (bid + 1) * STATIC_PAGES // CTAS - first_page
        processed = pages
        page = Int32(0)
        go = Int32(1)
        partials = s.partials
        while (page < MAX_PAGES) & (go != 0):
            slot = page % RING_SLOTS
            cute.arch.mbarrier_wait(s.mbar + slot, page // 4 % 2)
            if pages <= page:  # noqa: SIM102
                if cute.arch.vote_all_sync(s.page_ids[page] < 0):
                    go = Int32(0)
            if go != 0:
                processed = page + 1
                cute.arch.fence_proxy("async.shared", space="cta")
                acc = [[Float32(0.0) for _ in range(R)] for _ in range(3)]
                for r in cutlass.range_constexpr(3):
                    for j in cutlass.range_constexpr(3):
                        wv = _ld8(
                            s.ring_u4,
                            slot * 2304
                            + r * 768
                            + (warp * 768 + j * 256 + lane * 8) // 8,
                        )
                        for q in cutlass.range_constexpr(R):
                            for i in cutlass.range_constexpr(8):
                                acc[r][q] = rt.fma(nrm[q][j][i], wv[i], acc[r][q])
                    if R > 1:
                        for q in cutlass.range_constexpr(R):
                            total = rt.Reduce.fold([acc[r][q]], lane, 16, BFLY)
                            if lane == 0:
                                partials[
                                    (q * 8 + warp) * PAGE_PARTIALS + page * 3 + r
                                ] = total
                if R == 1:
                    a0, a1, a2 = acc[0][0], acc[1][0], acc[2][0]
                    upper = (lane & 16) != 0
                    send = a0 if upper else a2
                    keep = a2 if upper else a0
                    kept = rt.add(keep, cute.arch.shuffle_sync_bfly(send, 16))
                    total0 = rt.add(a1, cute.arch.shuffle_sync_bfly(a1, 16))
                    upper0 = (lane & 8) != 0
                    send0 = kept if upper0 else total0
                    keep0 = total0 if upper0 else kept
                    t = rt.add(keep0, cute.arch.shuffle_sync_bfly(send0, 8))
                    t = rt.Reduce.fold([t], lane, 4, (4, 2, 1))
                    entry = lane >> 3
                    if ((lane & 7) == 0) & (entry < 3):
                        partials[warp * PAGE_PARTIALS + page * 3 + entry] = t
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(s.mbar + L.ring_empty.offset // 8 + slot)
            page = page + 1
        rt.bar_sync(8, COMPUTE_THREADS)
        if tid == 0:
            ticket_slot[0] = launch_ticket
        rows = processed * 3
        first_row = first_page * 3
        logits_t = cute.make_tensor(p.Logits, cute.make_layout(R * SHARD))
        scratch_i = s.scratch_i
        for q in cutlass.range_constexpr(R):
            best = Float32(NEG_INF)
            best_index = Int32(INT_MAX)
            if rows > tid:
                logit = rt.Reduce.ordered_sum(
                    lambda w, q=q: s.partials[
                        q * ROW_PARTIALS + PAGE_PARTIALS * w + tid
                    ],
                    8,
                )
                row = first_row + tid
                if tid >= pages * 3:
                    row = s.page_ids[tid // 3] * 3 + tid % 3
                if row < SHARD:
                    logits_t[q * SHARD + row] = logit
                    best = logit.to(BFloat16).to(Float32)
                    best_index = row
            best, best_index = rt.Reduce.argmax_bfly(best, best_index, BFLY)
            if lane == 0:
                scratch[(q * 8 + warp) * 2] = best
                scratch_i[(q * 8 + warp) * 2 + 1] = best_index
        rt.bar_sync(8, COMPUTE_THREADS)
        if tid < R:
            cta_best = scratch[tid * 16]
            cta_index = scratch_i[tid * 16 + 1]
            for w in cutlass.range_constexpr(1, 8):
                other = scratch[(tid * 8 + w) * 2]
                other_index = scratch_i[(tid * 8 + w) * 2 + 1]
                if rt.Reduce.argmax_better(other, other_index, cta_best, cta_index):
                    cta_best = other
                    cta_index = other_index
            part = cute.make_tensor(p.Partials, cute.make_layout(R * CTAS * 2))
            part[(tid * CTAS + bid) * 2] = rt.Convert.f32_bits(cta_best)
            part[(tid * CTAS + bid) * 2 + 1] = Uint32(cta_index)
            record_tag = rt.TaggedMailbox.tag8(ticket_slot[0] // CTAS) << 24
            record_word = DoneWords.record(tid, bid)
            rt.Memory.st_relaxed_sys(
                s.done_addr + Int64(record_word) * 4,
                record_tag | (rt.Convert.f32_bits(cta_best) >> 16),
            )
            rt.Memory.st_relaxed_sys(
                s.done_addr + Int64(record_word + 1) * 4,
                record_tag | (Uint32(cta_index) & 65535),
            )
        rt.bar_sync(8, COMPUTE_THREADS)
        old = ticket_slot[0]
        if (bid == 0) & (warp < R):
            self.exchange(p, s, old, warp, lane)

    @cute.jit
    def load(self, p: LmHeadParams, s: LmHeadSmem, ids: rt.ThreadIds):
        """Warp 8 streams the static pages, then claims tail pages until they run
        out.
        """
        R = self.ROWS
        bid, warp = ids.bid, ids.warp
        self.set_regs("load")
        load_first = bid * STATIC_PAGES // CTAS
        load_pages = (bid + 1) * STATIC_PAGES // CTAS - load_first
        head = p.LmHead.toint()
        ring_addr = s.smem_addr + L.ring.offset
        for page in cutlass.range_constexpr(RING_SLOTS):
            with rt.warp_leader(warp, 8):
                self.load_page(s, ring_addr, head, page, load_first + page)
        cute.arch.griddepcontrol_wait()
        for page in cutlass.range(RING_SLOTS, load_pages, unroll=1):
            with rt.warp_leader(warp, 8):
                self.load_page(s, ring_addr, head, page, load_first + page)
        with rt.warp_leader(warp, 8):
            cute.arch.mbarrier_wait(s.mbar + L.ticket_full.offset // 8, 0)
            claim_parity = Int32(s.ticket_slot[0] // CTAS % 2)
            counter = DoneWords.counter(R)
            if bid == 0:
                rt.Memory.st_relaxed_sys(
                    s.done_addr + Int64(counter + 1 - claim_parity) * 4, Uint32(0)
                )
            claimed = cute.arch.atomic_add(p.Done + (counter + claim_parity), Uint32(1))
            sequence = load_pages
            step = Int32(0)
            while (step < 18) & (claimed < TAIL_PAGES):
                claimed_page = STATIC_PAGES + Int32(claimed)
                next_claim = Uint32(TAIL_PAGES)
                if sequence + 1 < MAX_PAGES:
                    next_claim = cute.arch.atomic_add(
                        p.Done + (counter + claim_parity), Uint32(1)
                    )
                self.load_page(s, ring_addr, head, sequence, claimed_page, claimed=True)
                sequence = sequence + 1
                claimed = next_claim
                step = step + 1
            if sequence < MAX_PAGES:
                end_slot = sequence % RING_SLOTS
                cute.arch.mbarrier_wait(
                    s.mbar + L.ring_empty.offset // 8 + end_slot,
                    sequence // 4 % 2 ^ 1,
                )
                page_ids = s.page_ids
                page_ids[sequence] = Int32(-1)
                cute.arch.mbarrier_arrive(s.mbar + end_slot)

    @cute.jit
    def load_page(
        self,
        s: LmHeadSmem,
        ring_addr,
        head: Int64,
        sequence,
        page,
        claimed: cutlass.Constexpr = False,
    ):
        """Wait for ``sequence``'s ring slot, then stream page ``page`` into it."""
        slot = sequence % RING_SLOTS
        cute.arch.mbarrier_wait(
            s.mbar + L.ring_empty.offset // 8 + slot, sequence // 4 % 2 ^ 1
        )
        if cutlass.const_expr(claimed):
            page_ids = s.page_ids
            page_ids[sequence] = page
        dst, src = ring_addr + slot * PAGE_BYTES, head + Int64(page) * PAGE_BYTES
        mbar_addr = s.smem_addr + slot * 8
        rt.AsyncCopy.bulk_g2s_evict_first(
            dst, src, PAGE_BYTES, mbar_addr, s.mbar + slot
        )

    @cute.jit
    def exchange(self, p: LmHeadParams, s: LmHeadSmem, old: Uint32, own_row, lane):
        """CTA 0, warp ``own_row``: rank best of the CTA records, then the eight-rank
        top-1.
        """
        peer_rank, bases, done_addr = p.peer_rank, p.peer_bases, s.done_addr
        epoch = old // CTAS
        tag = epoch + 1
        parity = Int32(epoch % 2)
        tag8 = rt.TaggedMailbox.tag8(epoch)
        polled = cute.make_rmem_tensor(10, Uint32)
        polled.fill(0)
        go = Int32(1)
        while go != 0:
            ready = Int32(1)
            for k in cutlass.range_constexpr(5):
                if lane + 32 * k < CTAS:
                    for h in cutlass.range_constexpr(2):
                        polled[2 * k + h] = rt.Memory.ld_relaxed_gpu(
                            done_addr
                            + Int64(1 + h + (own_row * CTAS + lane + 32 * k) * 2) * 4
                        )
                    ready = ready & Int32(
                        ((polled[2 * k] >> 24) == tag8)
                        & ((polled[2 * k + 1] >> 24) == tag8)
                    )
            if cute.arch.vote_all_sync(ready != 0):
                go = Int32(0)
        rank_best = Float32(NEG_INF)
        rank_index = Int32(INT_MAX)
        for k in cutlass.range_constexpr(5):
            if lane + 32 * k < CTAS:
                value = rt.Convert.bits_f32((polled[2 * k] & 65535) << 16)
                index = Int32(polled[2 * k + 1] & 65535)
                if index == 65535:
                    index = Int32(INT_MAX)
                if rt.Reduce.argmax_better(value, index, rank_best, rank_index):
                    rank_best = value
                    rank_index = index
        rank_best, rank_index = rt.Reduce.argmax_bfly(rank_best, rank_index, BFLY)
        best_bits = rt.Convert.f32_bits(rank_best)
        outgoing = [best_bits, Uint32(peer_rank * SHARD + rank_index), tag, tag]
        if lane == 0:
            record_t = cute.make_tensor(p.RankRecord, cute.make_layout(16))
            for k in cutlass.range_constexpr(4):
                record_t[own_row * 4 + k] = outgoing[k]
        if lane < 8:
            rt.TaggedMailbox.push(
                rt.TaggedMailbox.peer_base(lane, bases)
                + Int64(Mailbox.packet(parity, peer_rank, own_row)),
                outgoing,
            )
        source = Mailbox.packet(parity, lane, own_row)
        received_bits = Uint32(0)
        received_index = Uint32(0)
        go = Int32(1)
        while go != 0:
            ready_peers = Int32(1)
            if lane < 8:
                w0, w1, w2, w3 = rt.Memory.ld_volatile_v4(
                    rt.TaggedMailbox.peer_base(peer_rank, bases) + Int64(source)
                )
                ready_peers = Int32((w2 == tag) & (w3 == tag))
                received_bits = w0
                received_index = w1
            if cute.arch.vote_all_sync(ready_peers != 0):
                go = Int32(0)
        winner = Float32(NEG_INF)
        winner_index = Int32(INT_MAX)
        if lane < 8:
            winner = rt.Convert.bits_f32(received_bits)
            winner_index = Int32(received_index)
        winner, winner_index = rt.Reduce.argmax_bfly(winner, winner_index, BFLY)
        if lane == 0:
            win = cute.make_tensor(p.Winner, cute.make_layout(8))
            win[own_row * 2] = rt.Convert.f32_bits(winner)
            win[own_row * 2 + 1] = Uint32(winner_index)
            cute.make_tensor(p.Tokens, cute.make_layout(4))[own_row] = winner_index
