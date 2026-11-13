# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed and shared experts of one GLM-5.3-Flash MoE layer (TP4 rank
shapes: 288 experts of `[1024, 4096]` / `[4096, 512]` block-128 FP8 and the shared
expert of the same shape) for up to 32 decode tokens as one persistent SM90
kernel, finalized: BF16 `[M, 4096]` = sum_k w_k * expert_k(x) + shared(x).

It runs as a programmatic dependent of the router (`moe_router`) and reads its
int32 ids, FP32 weights and FP8 input. Before `griddepcontrol.wait` it only
initializes its barriers and shared tables and prefetches the tensor maps.

Two CTAs per SM, 4 consumer warps and a producer warp each. After the wait,
every CTA builds the same table of the distinct experts the tokens route to
(the shared expert in slot 0, then by expert id) and the (slot, token) -> k map.

- GEMM1: items (slot, 64 gate rows and the matching 64 up rows) over all of K,
  handed out in slot order by an atomic counter (the first item of each CTA is
  its block index). A ring of TMA stages (W gate, W up, the FP8 tokens; the
  first stage also the weight and token scales) feeds `wgmma` with the weights
  as the register A operand and the tokens as the N = 8 * groups B operand;
  every 128-K block's partials take the weight block scale times the token
  scale in FP32. The epilogue applies the clamped SwiGLU, quantizes each
  token's 64 hidden values to FP8 (block 64) and folds the routing weight into
  the hidden scale (0 for tokens not routed to the expert), then releases the
  slot's readiness counter.
- GEMM2: items (slot, 64 output rows) of 32 KB, in slot order, each waiting
  once for its slot's 8 GEMM1 items. With the routing weight in the hidden
  scale, the output of token t's k-th expert is a plain FP32 store to its own
  slot (t, k) of a `[M, 9, 4096]` buffer (slot 8: the shared expert), no atomics.
- After a grid barrier every CTA sums the 9 slots of its share of the output
  in k order and writes BF16, so the output is deterministic.

The grid (2 x SMs) must be co-resident (`residency`): CTAs spin on one
another's counters. The counters reset at the end of each call.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from ._moe_ptx import (
    atom_add_acqrel_gpu,
    atom_add_relaxed,
    bits_f32,
    bulk_g2s,
    cvt_bf16x2,
    cvt_e4m3x4,
    evict_first_policy,
    f32_bits,
    fabs,
    fence_gpu,
    fence_mbar_init,
    fence_proxy_async_global,
    fence_regs,
    fmax,
    griddep_wait,
    ld_acquire_i32,
    ldg_cg_f32,
    ldg_cg_f32x4,
    ldg_cg_i32,
    lds32,
    lds_f32,
    lds_u8,
    make_desc_nosw,
    mbar_arrive,
    mbar_arrive_expect_tx,
    mbar_init,
    mbar_wait,
    named_barrier,
    nanosleep,
    prefetch_tmap,
    rcp,
    red_add_gpu,
    red_max_shared_u32,
    shfl_up_i32,
    shfl_xor_f32,
    smem_addr,
    smem_atom_or,
    stg_f32,
    stg_i32,
    stg_u8,
    stg_u32x2,
    sts32,
    sts_f32,
    sts_u8,
    swiglu,
    tma_load_3d,
    tma_load_4d,
    wgmma_commit,
    wgmma_fence,
    wgmma_rs,
    wgmma_wait_n,
)
from .moe import EXPERTS, HIDDEN, INTER, MAX_TOKENS, SWIGLU_LIMIT, TOP_K
from .moe_router import pow2_ceil

PAIRS = INTER // 64  # GEMM1 items per slot
K1_BLOCKS = HIDDEN // 128
K2_BLOCKS = INTER // 128
OUT_TILES = HIDDEN // 64  # GEMM2 items per slot
SHARED = EXPERTS  # the shared expert's id in the slot table
# Slots: every distinct routed expert of 32 tokens plus the shared expert.
SLOTS = 290
CONSUMERS = 4
PRODUCER = CONSUMERS  # warp index
THREADS = 32 * (CONSUMERS + 1)
CONSUMER_THREADS = 32 * CONSUMERS
END = 0x7FFFFFFF
# Global counters (int32 words): GEMM1 and GEMM2 items, exits, the grid barrier.
C_W1, C_W2, C_EXIT, C_DONE = 1, 2, 3, 4
COUNTERS = 5
# Byte offsets in a GEMM1 / GEMM2 stage.
G1_GATE = 0
G1_UP = 8192
G1_B = 16384
G2_W = 0
G2_B = 8192
# Tensor maps in a layer's blob, 128 B each.
(
    MAP_W13, MAP_W2, MAP_XQ, MAP_IQ, MAP_SHARED_W13, MAP_SHARED_W2,
) = range(6)  # fmt: skip
MAPS = 6
STAGES1 = 4
CTAS_PER_SM = 2


def _align(v, a):
    return (v + a - 1) // a * a


class ExpertsKernel:
    def __init__(self, groups: int, ue8m0: bool):
        assert 1 <= groups <= MAX_TOKENS // 8
        self.groups = groups
        self.rows = 8 * groups
        self.ue8m0 = ue8m0
        mp = self.rows
        self.ns1 = STAGES1
        self.sb1 = _align(G1_B + mp * 128 + 64, 1024)
        self.hdr1 = self.sb1 - 32
        self.sc2 = _align(G2_B + mp * 128, 16)
        self.hdr2 = self.sc2 + 16 + 8 * mp
        self.sb2 = _align(self.hdr2 + 16, 1024)
        self.ns2 = max(2, (self.ns1 * self.sb1) // self.sb2)
        off = _align(max(self.ns1 * self.sb1, self.ns2 * self.sb2), 1024)
        self.tab_e = off
        off += 4 * SLOTS
        self.s_tmask = off
        off += 4 * SLOTS
        self.s_e2s = off
        off += 4 * SLOTS
        # [slot][32 tokens] u8: the slot's k in the token's top-k, 8 for the
        # shared expert, 0xFF if not routed.
        self.s_kmap = _align(off, 16)
        off = self.s_kmap + 32 * SLOTS
        self.s_ids = _align(off, 16)
        off = self.s_ids + 4 * 256
        self.s_tw = off
        off += 4 * 256
        self.s_xs = _align(off, 16)
        off = self.s_xs + mp * 32 * 4
        self.s_ws1 = _align(off, 16)
        off = self.s_ws1 + 2 * 2 * 32 * 4
        self.s_amax = _align(off, 16)
        off = self.s_amax + 4 * mp
        self.full1 = _align(off, 8)
        off = self.full1 + 8 * self.ns1
        self.empty1 = off
        off += 8 * self.ns1
        self.full2 = off
        off += 8 * self.ns2
        self.empty2 = off
        off += 8 * self.ns2
        self.s_tot = off
        off += 8
        self.smem = _align(off, 16)
        assert (self.smem + 1024) * CTAS_PER_SM <= 228 * 1024, self.smem

    @cute.jit
    def __call__(
        self, maps: Int64, w13s: Int64, w2s: Int64, sw13s: Int64, sw2s: Int64,
        xs: Int64, ids: Int64, tw: Int64, iq: Int64, isc: Int64, slots: Int64,
        out: Int64, cnt: Int64, sync: Int64, m: Int32, grid: Int32, stream,
    ):  # fmt: skip
        self.kernel(
            maps, w13s, w2s, sw13s, sw2s, xs, ids, tw, iq, isc, slots, out, cnt,
            sync, m,
        ).launch(
            grid=[grid, 1, 1], block=[THREADS, 1, 1], stream=stream,
            min_blocks_per_mp=CTAS_PER_SM, preferred_smem_carveout=100,
            use_pdl=True,
        )  # fmt: skip

    # ------------------------------------------------------------------ front
    @cute.jit
    def _front_init(self, tid: Int32, s_tmask: Int32, s_kmap: Int32):
        for i in cutlass.range(tid, EXPERTS, THREADS, unroll=1):
            sts32(s_tmask + i * 4, Uint32(0))
        for i in cutlass.range(tid, SLOTS * 8, THREADS, unroll=1):
            sts32(s_kmap + i * 4, Uint32(0xFFFFFFFF))
        cute.arch.sync_threads()

    @cute.jit
    def _front(
        self, ids: Int64, tw: Int64, m: Int32, tid: Int32, warp: Int32,
        lane: Int32, tab_e: Int32, s_tmask: Int32, s_e2s: Int32, s_kmap: Int32,
        s_ids: Int32, s_tw: Int32, s_tot: Int32,
    ):  # fmt: skip
        """The slot table: the shared expert in slot 0, then the routed experts
        in id order; returns the slot count."""
        routes = m * TOP_K
        for i in cutlass.range(tid, routes, THREADS, unroll=1):
            ex = ldg_cg_i32(ids + Int64(i) * 4)
            wv = ldg_cg_f32(tw + Int64(i) * 4)
            sts32(s_ids + i * 4, Uint32(ex))
            sts_f32(s_tw + i * 4, wv)
            smem_atom_or(s_tmask + ex * 4, Uint32(1) << Uint32(i // TOP_K))
        cute.arch.sync_threads()
        if warp == 0:
            per_lane = EXPERTS // 32
            count = Int32(0)
            for j in cutlass.range_constexpr(per_lane):
                if lds32(s_tmask + (lane * per_lane + j) * 4) != Uint32(0):
                    count = count + 1
            inclusive = count
            for i in cutlass.range_constexpr(5):
                o = shfl_up_i32(inclusive, 1 << i)
                if lane >= (1 << i):
                    inclusive = inclusive + o
            r = inclusive - count + 1
            for j in cutlass.range_constexpr(per_lane):
                ex = lane * per_lane + j
                if lds32(s_tmask + ex * 4) != Uint32(0):
                    sts32(tab_e + r * 4, Uint32(ex))
                    sts32(s_e2s + ex * 4, Uint32(r))
                    r = r + 1
            if lane == 31:
                sts32(s_tot, Uint32(inclusive + 1))
        if tid == 0:
            sts32(tab_e, Uint32(SHARED))
        if tid < m:
            sts_u8(s_kmap + tid, Uint32(8))
        cute.arch.sync_threads()
        for i in cutlass.range(tid, routes, THREADS, unroll=1):
            ex = Int32(lds32(s_ids + i * 4))
            r = Int32(lds32(s_e2s + ex * 4))
            sts_u8(s_kmap + r * 32 + i // TOP_K, Uint32(i % TOP_K))
        cute.arch.sync_threads()
        return Int32(lds32(s_tot))

    @cute.jit
    def _loc(self, x: Int32):
        """The 128B-swizzled shared offset of byte `x` of a 128-byte-row tile."""
        return x ^ (((x >> 7) & 7) << 4)

    @cute.jit
    def _route_w(self, e: Int32, n: Int32, m: Int32, s_ids: Int32, s_tw: Int32):
        """Token n's routing weight for expert e (1 for the shared expert, 0 if
        not routed or n >= m)."""
        wv = Float32(0.0)
        if n < m:
            if e == SHARED:
                wv = Float32(1.0)
            else:
                for k in cutlass.range_constexpr(TOP_K):
                    if Int32(lds32(s_ids + (n * TOP_K + k) * 4)) == e:
                        wv = lds_f32(s_tw + (n * TOP_K + k) * 4)
        return wv

    @cute.jit
    def _source(self, e: Int32, routed: Int64, shared: Int64):
        """(tensor map, expert coordinate) of expert e."""
        tmap = routed
        coord = e
        if e == SHARED:
            tmap = shared
            coord = Int32(0)
        return tmap, coord

    # ------------------------------------------------------------------ GEMM1
    @cute.jit
    def _issue1(
        self, st: Int32, fb: Int32, e: Int32, p: Int32, js: Int32, tw13: Int64,
        tsw13: Int64, pol: Int64,
    ):  # fmt: skip
        tmap, coord = self._source(e, tw13, tsw13)
        tma_load_3d(st + G1_GATE, tmap, js * 128, 64 * p, coord, fb, pol)
        tma_load_3d(st + G1_UP, tmap, js * 128, 512 + 64 * p, coord, fb, pol)

    @cute.jit
    def _scales1(
        self, dst: Int32, e: Int32, p: Int32, w13s: Int64, sw13s: Int64,
        fb: Int32, pol: Int64,
    ):  # fmt: skip
        """The gate and up weight scales of item (e, p) (32 x 2 FP32)."""
        base = w13s + Int64(e) * (8 * 128)
        if e == SHARED:
            base = sw13s
        bulk_g2s(dst, base + Int64(p // 2) * 128, 128, fb, pol)
        bulk_g2s(dst + 128, base + Int64(4 + p // 2) * 128, 128, fb, pol)

    @cute.jit
    def _produce1(
        self, tw13: Int64, tsw13: Int64, txq: Int64, w13s: Int64, sw13s: Int64,
        xs: Int64, cnt: Int64, n1: Int32, sbase: Int32, full: Int32,
        empty: Int32, tab_e: Int32, s_xs: Int32, s_ws1: Int32, bid: Int32,
        nblk: Int32,
    ):  # fmt: skip
        sb = self.sb1
        hdr = self.hdr1
        ns = self.ns1
        mp = self.rows
        pol = evict_first_policy()
        work = cnt + C_W1 * 4
        stage = Int32(0)
        phase = Int32(0)
        seq = Int32(0)
        nxt = bid
        running = Int32(1)
        js0 = Int32(0)
        # The first item's stages, W and B together.
        if bid < n1:
            e0 = Int32(lds32(tab_e + (bid // PAIRS) * 4))
            p0 = bid % PAIRS
            for js in cutlass.range_constexpr(ns):
                st = sbase + js * sb
                fb = full + js * 8
                sts32(st + hdr, Uint32(bid))
                if cutlass.const_expr(js == 0):
                    mbar_arrive_expect_tx(fb, G1_B + mp * 128 + 256 + mp * 32 * 4)
                else:
                    mbar_arrive_expect_tx(fb, G1_B + mp * 128)
                self._issue1(st, fb, e0, p0, Int32(js), tw13, tsw13, pol)
                if cutlass.const_expr(js == 0):
                    self._scales1(s_ws1, e0, p0, w13s, sw13s, fb, pol)
                tma_load_3d(st + G1_B, txq, Int32(0), js * 8, Int32(0), fb, pol)
                if cutlass.const_expr(js == 0):
                    bulk_g2s(s_xs, xs, mp * 32 * 4, fb, pol)
            js0 = Int32(ns)
            phase = Int32(1)
        while running == 1:
            ch = nxt
            if ch < n1:
                nxt = nblk + atom_add_relaxed(work, 1)
            r = ch // PAIRS
            p = ch % PAIRS
            e = Int32(0)
            if ch < n1:
                e = Int32(lds32(tab_e + r * 4))
            nst = Int32(K1_BLOCKS)
            if ch >= n1:
                nst = Int32(1)
                running = Int32(0)
                js0 = Int32(0)
            for js in cutlass.range(js0, nst, unroll=1):
                mbar_wait(empty + stage * 8, phase ^ 1)
                st = sbase + stage * sb
                fb = full + stage * 8
                if running == 0:
                    sts32(st + hdr, Uint32(END))
                    mbar_arrive_expect_tx(fb, 0)
                else:
                    sts32(st + hdr, Uint32(ch))
                    extra = Int32(0)
                    if js == 0:
                        extra = Int32(256)
                    mbar_arrive_expect_tx(fb, G1_B + mp * 128 + extra)
                    self._issue1(st, fb, e, p, js, tw13, tsw13, pol)
                    tma_load_3d(st + G1_B, txq, Int32(0), js * 8, Int32(0), fb, pol)
                    if js == 0:
                        slot = seq % 2
                        self._scales1(s_ws1 + slot * 256, e, p, w13s, sw13s, fb, pol)
                stage = stage + 1
                if stage == ns:
                    stage = Int32(0)
                    phase = phase ^ 1
            js0 = Int32(0)
            seq = seq + 1
        for _ in cutlass.range(ns, unroll=1):
            mbar_wait(empty + stage * 8, phase ^ 1)
            stage = stage + 1
            if stage == ns:
                stage = Int32(0)
                phase = phase ^ 1

    @cute.jit
    def _consume1(
        self, stage: Int32, phase: Int32, sbase: Int32, full: Int32, empty: Int32,
        w: Int32, lane: Int32, slot: Int32, s_xs: Int32, s_ws1: Int32,
        ag: cute.Tensor, au: cute.Tensor,
    ):  # fmt: skip
        size = self.sb1
        ns = self.ns1
        groups = self.groups
        row0 = w * 16 + lane // 4
        c = lane % 4
        c4 = c * 4
        for i in cutlass.range_constexpr(4 * groups):
            ag[i] = Float32(0.0)
            au[i] = Float32(0.0)
        dg = cute.make_rmem_tensor((4 * groups,), Float32)
        du = cute.make_rmem_tensor((4 * groups,), Float32)
        for js in cutlass.range(K1_BLOCKS, unroll=1):
            if js > 0:
                mbar_wait(full + stage * 8, phase)
            st = sbase + stage * size
            for i in cutlass.range_constexpr(4 * groups):
                dg[i] = Float32(0.0)
                du[i] = Float32(0.0)
            gr = cute.make_rmem_tensor((16,), Uint32)
            ur = cute.make_rmem_tensor((16,), Uint32)
            for kc in cutlass.range_constexpr(4):
                ks = kc * 32
                for h in cutlass.range_constexpr(2):
                    for v in cutlass.range_constexpr(2):
                        x = (row0 + 8 * v) * 128 + ks + c4 + 16 * h
                        gr[4 * kc + 2 * h + v] = lds32(st + G1_GATE + self._loc(x))
                        ur[4 * kc + 2 * h + v] = lds32(st + G1_UP + self._loc(x))
            wgmma_fence()
            vg = [dg[i] for i in range(4 * groups)]
            vu = [du[i] for i in range(4 * groups)]
            for kc in cutlass.range_constexpr(4):
                bd = make_desc_nosw(st + G1_B + 2 * kc * 128, 128, 1024)
                vg = wgmma_rs(
                    8 * groups, vg, gr[4 * kc], gr[4 * kc + 1], gr[4 * kc + 2],
                    gr[4 * kc + 3], bd, Int32(1),
                )  # fmt: skip
                vu = wgmma_rs(
                    8 * groups, vu, ur[4 * kc], ur[4 * kc + 1], ur[4 * kc + 2],
                    ur[4 * kc + 3], bd, Int32(1),
                )  # fmt: skip
            wgmma_commit()
            wgmma_wait_n(0)
            vg = fence_regs(vg)
            vu = fence_regs(vu)
            for i in cutlass.range_constexpr(4 * groups):
                dg[i] = vg[i]
                du[i] = vu[i]
            wg = lds_f32(s_ws1 + slot * 256 + js * 4)
            wu = lds_f32(s_ws1 + slot * 256 + 128 + js * 4)
            for g in cutlass.range_constexpr(groups):
                sa = lds_f32(s_xs + ((8 * g + 2 * c) * 32 + js) * 4)
                sb = lds_f32(s_xs + ((8 * g + 2 * c + 1) * 32 + js) * 4)
                ag[4 * g + 0] = ag[4 * g + 0] + dg[4 * g + 0] * (sa * wg)
                ag[4 * g + 1] = ag[4 * g + 1] + dg[4 * g + 1] * (sb * wg)
                ag[4 * g + 2] = ag[4 * g + 2] + dg[4 * g + 2] * (sa * wg)
                ag[4 * g + 3] = ag[4 * g + 3] + dg[4 * g + 3] * (sb * wg)
                au[4 * g + 0] = au[4 * g + 0] + du[4 * g + 0] * (sa * wu)
                au[4 * g + 1] = au[4 * g + 1] + du[4 * g + 1] * (sb * wu)
                au[4 * g + 2] = au[4 * g + 2] + du[4 * g + 2] * (sa * wu)
                au[4 * g + 3] = au[4 * g + 3] + du[4 * g + 3] * (sb * wu)
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty + stage * 8)
            stage = stage + 1
            if stage == ns:
                stage = Int32(0)
                phase = phase ^ 1
        return stage, phase

    @cute.jit
    def _epilogue1(
        self, w: Int32, lane: Int32, tid: Int32, r: Int32, p: Int32, m: Int32,
        s_amax: Int32, tab_e: Int32, s_ids: Int32, s_tw: Int32, iq: Int64,
        isc: Int64, sync: Int64, ag: cute.Tensor, au: cute.Tensor,
    ):  # fmt: skip
        """SwiGLU and FP8 (block 64) of this 64-row pair; the hidden scale times
        the routing weight, as `[slot][block][MP]`."""
        groups = self.groups
        c = lane % 4
        mp = self.rows
        h = cute.make_rmem_tensor((4 * groups,), Float32)
        for i in cutlass.range_constexpr(4 * groups):
            h[i] = swiglu(ag[i], au[i], Float32(SWIGLU_LIMIT))
        am = cute.make_rmem_tensor((2 * groups,), Float32)
        for g in cutlass.range_constexpr(groups):
            for vv in cutlass.range_constexpr(2):
                v = fmax(fabs(h[4 * g + vv]), fabs(h[4 * g + vv + 2]))
                for sh in cutlass.range_constexpr(2, 5):
                    v = fmax(v, shfl_xor_f32(v, 1 << sh))
                am[2 * g + vv] = v
        if w == 0 and lane < 4:
            for g in cutlass.range_constexpr(groups):
                for vv in cutlass.range_constexpr(2):
                    sts32(s_amax + (8 * g + 2 * c + vv) * 4, Uint32(0))
        named_barrier(2, CONSUMER_THREADS)
        if lane < 4:
            for g in cutlass.range_constexpr(groups):
                for vv in cutlass.range_constexpr(2):
                    red_max_shared_u32(
                        s_amax + (8 * g + 2 * c + vv) * 4, f32_bits(am[2 * g + vv])
                    )
        named_barrier(3, CONSUMER_THREADS)
        for g in cutlass.range_constexpr(groups):
            for vv in cutlass.range_constexpr(2):
                am[2 * g + vv] = bits_f32(lds32(s_amax + (8 * g + 2 * c + vv) * 4))
        m0 = w * 16 + lane // 4
        e = Int32(lds32(tab_e + r * 4))
        for g in cutlass.range_constexpr(groups):
            a0 = fmax(am[2 * g], Float32(1e-12))
            a1 = fmax(am[2 * g + 1], Float32(1e-12))
            s0 = a0 * Float32(1.0 / 448.0)
            s1 = a1 * Float32(1.0 / 448.0)
            i0 = Float32(448.0) * rcp(a0)
            i1 = Float32(448.0) * rcp(a1)
            if cutlass.const_expr(self.ue8m0):
                s0, i0 = pow2_ceil(s0)
                s1, i1 = pow2_ceil(s1)
            q = cvt_e4m3x4(
                h[4 * g] * i0, h[4 * g + 1] * i1, h[4 * g + 2] * i0, h[4 * g + 3] * i1
            )
            n0 = 8 * g + 2 * c
            n1 = n0 + 1
            row0 = 64 * p + m0
            row1 = row0 + 8
            g0 = Int64(r) * groups + g
            base0 = iq + (g0 * 32 + row0 // 16) * 128 + row0 % 16
            base1 = iq + (g0 * 32 + row1 // 16) * 128 + row1 % 16
            stg_u8(base0 + (n0 % 8) * 16, q & Uint32(0xFF))
            stg_u8(base0 + (n1 % 8) * 16, (q >> Uint32(8)) & Uint32(0xFF))
            stg_u8(base1 + (n0 % 8) * 16, (q >> Uint32(16)) & Uint32(0xFF))
            stg_u8(base1 + (n1 % 8) * 16, (q >> Uint32(24)) & Uint32(0xFF))
            if w == 0 and lane < 4:
                w0 = self._route_w(e, n0, m, s_ids, s_tw)
                w1 = self._route_w(e, n1, m, s_ids, s_tw)
                stg_f32(isc + ((Int64(r) * 8 + p) * mp + n0) * 4, s0 * w0)
                stg_f32(isc + ((Int64(r) * 8 + p) * mp + n1) * 4, s1 * w1)
        fence_proxy_async_global()
        named_barrier(1, CONSUMER_THREADS)
        if tid == 0:
            fence_gpu()
            red_add_gpu(sync + Int64(r) * 4, 1)

    # ------------------------------------------------------------------ GEMM2
    @cute.jit
    def _produce2(
        self, tw2: Int64, tsw2: Int64, tiq: Int64, w2s: Int64, sw2s: Int64,
        isc: Int64, cnt: Int64, sync: Int64, n2: Int32, sbase: Int32,
        full: Int32, empty: Int32, tab_e: Int32, bid: Int32, nblk: Int32,
    ):  # fmt: skip
        sb = self.sb2
        hdr = self.hdr2
        sc = self.sc2
        ns = self.ns2
        mp = self.rows
        pol = evict_first_policy()
        work = cnt + C_W2 * 4
        nbytes = G2_B + mp * 128 + 16 + 8 * mp
        stage = Int32(0)
        phase = Int32(0)
        nxt = bid
        running = Int32(1)
        last_r = Int32(-1)
        while running == 1:
            ch = nxt
            if ch < n2:
                nxt = nblk + atom_add_relaxed(work, 1)
                r = ch // OUT_TILES
                nb = ch % OUT_TILES
                e = Int32(lds32(tab_e + r * 4))
                tmap, coord = self._source(e, tw2, tsw2)
                scales = w2s + Int64(e) * (32 * 16)
                if e == SHARED:
                    scales = sw2s
                for js in cutlass.range_constexpr(K2_BLOCKS):
                    mbar_wait(empty + stage * 8, phase ^ 1)
                    st = sbase + stage * sb
                    fb = full + stage * 8
                    sts32(st + hdr, Uint32(ch))
                    mbar_arrive_expect_tx(fb, nbytes)
                    tma_load_3d(st + G2_W, tmap, js * 128, 64 * nb, coord, fb, pol)
                    bulk_g2s(st + sc, scales + Int64(nb // 2) * 16, 16, fb, pol)
                    if cutlass.const_expr(js == 0):  # noqa: SIM102
                        if r != last_r:
                            while ld_acquire_i32(sync + Int64(r) * 4) < PAIRS:
                                nanosleep(32)
                            fence_proxy_async_global()
                            last_r = r
                    hidden_scales = isc + ((Int64(r) * 8 + 2 * js) * mp) * 4
                    bulk_g2s(st + sc + 16, hidden_scales, 8 * mp, fb, pol)
                    tma_load_4d(st + G2_B, tiq, Int32(0), js * 8, Int32(0), r, fb, pol)
                    stage = stage + 1
                    if stage == ns:
                        stage = Int32(0)
                        phase = phase ^ 1
            else:
                running = Int32(0)
                mbar_wait(empty + stage * 8, phase ^ 1)
                sts32(sbase + stage * sb + hdr, Uint32(END))
                mbar_arrive_expect_tx(full + stage * 8, 0)

    @cute.jit
    def _consume2(
        self, stage: Int32, phase: Int32, sbase: Int32, full: Int32, empty: Int32,
        w: Int32, lane: Int32, acc: cute.Tensor,
    ):  # fmt: skip
        size = self.sb2
        sc = self.sc2
        ns = self.ns2
        groups = self.groups
        mp = self.rows
        row0 = w * 16 + lane // 4
        c = lane % 4
        c4 = c * 4
        for i in cutlass.range_constexpr(4 * groups):
            acc[i] = Float32(0.0)
        d = cute.make_rmem_tensor((4 * groups,), Float32)
        for js in cutlass.range_constexpr(K2_BLOCKS):
            mbar_wait(full + stage * 8, phase)
            st = sbase + stage * size
            ws = lds_f32(st + sc + js * 4)
            for kc2 in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(4 * groups):
                    d[i] = Float32(0.0)
                wgmma_fence()
                vv = [d[i] for i in range(4 * groups)]
                for k1 in cutlass.range_constexpr(2):
                    ks = (2 * kc2 + k1) * 32
                    rows = [(row0 + 8 * v) * 128 + ks + c4 + 16 * h
                            for h in range(2) for v in range(2)]  # fmt: skip
                    a = [lds32(st + G2_W + self._loc(x)) for x in rows]
                    bd = make_desc_nosw(st + G2_B + 2 * (2 * kc2 + k1) * 128, 128, 1024)
                    vv = wgmma_rs(8 * groups, vv, a[0], a[1], a[2], a[3], bd, Int32(1))
                wgmma_commit()
                wgmma_wait_n(0)
                vv = fence_regs(vv)
                for i in cutlass.range_constexpr(4 * groups):
                    d[i] = vv[i]
                for g in cutlass.range_constexpr(groups):
                    n0 = 8 * g + 2 * c
                    sa = lds_f32(st + sc + 16 + (kc2 * mp + n0) * 4) * ws
                    sb = lds_f32(st + sc + 16 + (kc2 * mp + n0 + 1) * 4) * ws
                    acc[4 * g + 0] = acc[4 * g + 0] + d[4 * g + 0] * sa
                    acc[4 * g + 1] = acc[4 * g + 1] + d[4 * g + 1] * sb
                    acc[4 * g + 2] = acc[4 * g + 2] + d[4 * g + 2] * sa
                    acc[4 * g + 3] = acc[4 * g + 3] + d[4 * g + 3] * sb
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty + stage * 8)
            stage = stage + 1
            if stage == ns:
                stage = Int32(0)
                phase = phase ^ 1
        return stage, phase

    @cute.jit
    def _epilogue2(
        self, r: Int32, nb: Int32, m: Int32, w: Int32, lane: Int32, s_kmap: Int32,
        slots: Int64, acc: cute.Tensor,
    ):  # fmt: skip
        """Plain stores of the routed tokens' rows into their (token, k) slots."""
        c = lane % 4
        rb = 64 * nb + 16 * w + lane // 4
        for g in cutlass.range_constexpr(self.groups):
            for vv in cutlass.range_constexpr(2):
                col = 8 * g + 2 * c + vv
                if col < m:
                    k = lds_u8(s_kmap + r * 32 + col)
                    if k != Uint32(0xFF):
                        for hv in cutlass.range_constexpr(2):
                            offset = (Int64(col) * 9 + Int64(k)) * HIDDEN + rb + 8 * hv
                            stg_f32(slots + offset * 4, acc[4 * g + vv + 2 * hv])

    @cute.jit
    def _finish(
        self, m: Int32, tid: Int32, bid: Int32, nblk: Int32, cnt: Int64,
        slots: Int64, out: Int64,
    ):  # fmt: skip
        """Grid barrier, then out[t] = sum_k slot(t, k) in k order over this
        CTA's share of the output (4 consecutive features per thread)."""
        if tid == 0:
            fence_gpu()
            atom_add_relaxed(cnt + C_DONE * 4, 1)
            while ld_acquire_i32(cnt + C_DONE * 4) < nblk:
                nanosleep(64)
        cute.arch.sync_threads()
        fence_gpu()
        quads = m * (HIDDEN // 4)
        per = (quads + nblk - 1) // nblk
        g0 = bid * per
        g1 = g0 + per
        if g1 > quads:
            g1 = quads
        for gi in cutlass.range(g0 + tid, g1, THREADS, unroll=1):
            t = gi // (HIDDEN // 4)
            q = gi % (HIDDEN // 4)
            base = slots + (Int64(t) * 9 * HIDDEN + 4 * q) * 4
            x = cute.make_rmem_tensor((9 * 4,), Float32)
            for k in cutlass.range_constexpr(9):
                v4 = ldg_cg_f32x4(base + k * HIDDEN * 4)
                for j in cutlass.range_constexpr(4):
                    x[4 * k + j] = v4[j]
            s = cute.make_rmem_tensor((4,), Float32)
            for j in cutlass.range_constexpr(4):
                s[j] = x[j]
            for k in cutlass.range_constexpr(1, 9):
                for j in cutlass.range_constexpr(4):
                    s[j] = s[j] + x[4 * k + j]
            stg_u32x2(
                out + (Int64(t) * HIDDEN + 4 * q) * 2,
                cvt_bf16x2(s[0], s[1]),
                cvt_bf16x2(s[2], s[3]),
            )

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self, maps: Int64, w13s: Int64, w2s: Int64, sw13s: Int64, sw2s: Int64,
        xs: Int64, ids: Int64, tw: Int64, iq: Int64, isc: Int64, slots: Int64,
        out: Int64, cnt: Int64, sync: Int64, m: Int32,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        warp = tid // 32
        lane = tid % 32
        tw13 = maps + MAP_W13 * 128
        tw2 = maps + MAP_W2 * 128
        txq = maps + MAP_XQ * 128
        tiq = maps + MAP_IQ * 128
        tsw13 = maps + MAP_SHARED_W13 * 128
        tsw2 = maps + MAP_SHARED_W2 * 128
        smem = cute.arch.alloc_smem(cutlass.Uint8, self.smem, alignment=1024)
        sbase = smem_addr(smem)
        tab_e = sbase + self.tab_e
        s_tmask = sbase + self.s_tmask
        s_e2s = sbase + self.s_e2s
        s_kmap = sbase + self.s_kmap
        s_ids = sbase + self.s_ids
        s_tw = sbase + self.s_tw
        s_xs = sbase + self.s_xs
        s_ws1 = sbase + self.s_ws1
        s_amax = sbase + self.s_amax
        full1 = sbase + self.full1
        empty1 = sbase + self.empty1
        full2 = sbase + self.full2
        empty2 = sbase + self.empty2
        s_tot = sbase + self.s_tot
        if tid == 0:
            for s in cutlass.range_constexpr(self.ns1):
                mbar_init(full1 + s * 8, 1)
                mbar_init(empty1 + s * 8, CONSUMERS)
            for s in cutlass.range_constexpr(self.ns2):
                mbar_init(full2 + s * 8, 1)
                mbar_init(empty2 + s * 8, CONSUMERS)
            fence_mbar_init()
        cute.arch.sync_threads()
        if warp == PRODUCER and lane == 0:
            for i in cutlass.range_constexpr(MAPS):
                prefetch_tmap(maps + i * 128)
        self._front_init(tid, s_tmask, s_kmap)
        griddep_wait()
        fence_proxy_async_global()
        n_slots = self._front(
            ids, tw, m, tid, warp, lane, tab_e, s_tmask, s_e2s, s_kmap, s_ids, s_tw,
            s_tot,
        )  # fmt: skip
        n1 = n_slots * PAIRS
        n2 = n_slots * OUT_TILES
        if warp == PRODUCER and lane == 0:
            self._produce1(
                tw13, tsw13, txq, w13s, sw13s, xs, cnt, n1, sbase, full1,
                empty1, tab_e, s_xs, s_ws1, bid, nblk,
            )  # fmt: skip
            self._produce2(
                tw2, tsw2, tiq, w2s, sw2s, isc, cnt, sync, n2, sbase, full2,
                empty2, tab_e, bid, nblk,
            )  # fmt: skip
        if warp < CONSUMERS:
            w = warp
            ag = cute.make_rmem_tensor((4 * self.groups,), Float32)
            au = cute.make_rmem_tensor((4 * self.groups,), Float32)
            acc = cute.make_rmem_tensor((4 * self.groups,), Float32)
            stage = Int32(0)
            phase = Int32(0)
            item = Int32(0)
            running = Int32(1)
            while running == 1:
                mbar_wait(full1 + stage * 8, phase)
                ch = Int32(lds32(sbase + stage * self.sb1 + self.hdr1))
                if ch >= n1:
                    running = Int32(0)
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(empty1 + stage * 8)
                else:
                    r = ch // PAIRS
                    p = ch % PAIRS
                    stage, phase = self._consume1(
                        stage, phase, sbase, full1, empty1, w, lane, item % 2, s_xs,
                        s_ws1, ag, au,
                    )  # fmt: skip
                    self._epilogue1(
                        w, lane, tid, r, p, m, s_amax, tab_e, s_ids, s_tw, iq, isc,
                        sync, ag, au,
                    )  # fmt: skip
                    item = item + 1
            stage = Int32(0)
            phase = Int32(0)
            running = Int32(1)
            while running == 1:
                mbar_wait(full2 + stage * 8, phase)
                ch = Int32(lds32(sbase + stage * self.sb2 + self.hdr2))
                if ch >= n2:
                    running = Int32(0)
                    cute.arch.sync_warp()
                    if lane == 0:
                        mbar_arrive(empty2 + stage * 8)
                else:
                    r = ch // OUT_TILES
                    nb = ch % OUT_TILES
                    stage, phase = self._consume2(
                        stage, phase, sbase, full2, empty2, w, lane, acc
                    )
                    self._epilogue2(r, nb, m, w, lane, s_kmap, slots, acc)
        cute.arch.sync_threads()
        self._finish(m, tid, bid, nblk, cnt, slots, out)
        cute.arch.sync_threads()
        if tid == 0:
            old = atom_add_acqrel_gpu(cnt + C_EXIT * 4, 1)
            last = Uint32(0)
            if old == nblk - 1:
                last = Uint32(1)
                for s in cutlass.range_constexpr(COUNTERS):
                    stg_i32(cnt + s * 4, 0)
            sts32(s_tot, last)
        cute.arch.sync_threads()
        if lds32(s_tot) == Uint32(1):
            for i in cutlass.range(tid, n_slots, THREADS, unroll=1):
                stg_i32(sync + Int64(i) * 4, 0)
            fence_gpu()


def residency(code, device: int) -> dict:
    """Function attributes and resident CTAs per SM of `code`. Its shared
    memory is static (`alloc_smem`); the function's dynamic limit is only the
    rest of the SM's, so the occupancy query takes no dynamic shared memory."""
    import cuda.bindings.driver as cuda

    from ..cutedsl.host import _checked, resolve_kernel

    _, function = resolve_kernel(code, device)
    attribute = cuda.CUfunction_attribute
    info = {
        name: _checked(cuda.cuFuncGetAttribute(getattr(attribute, key), function))[0]
        for name, key in (
            ("registers", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
            ("static_smem", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"),
            ("dynamic_limit", "CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES"),
            ("carveout", "CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT"),
        )
    }
    occupancy = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
    (info["per_sm"],) = _checked(occupancy(function, THREADS, 0))
    return info


@cache
def compiled(groups: int, ue8m0: bool, device: int):
    """The kernel at `groups` token groups, given the largest shared memory
    carveout and checked to be co-resident on `device`."""
    import torch
    from cutlass.cute.runtime import make_fake_stream

    from ..cutedsl.host import prepare_shared_carveout

    kernel = ExpertsKernel(groups, ue8m0)
    zero = Int64(0)
    with torch.accelerator.device_index(device):
        code = cute.compile(
            kernel, *([zero] * 14), Int32(1), Int32(1), make_fake_stream()
        )
    prepared = prepare_shared_carveout(code, device, 100)
    # CTAs wait on one another: the whole grid must be resident.
    info = residency(code, device)
    if info["per_sm"] < CTAS_PER_SM:
        raise RuntimeError(
            f"GLM-5.3-Flash Hopper MoE experts: {CTAS_PER_SM} CTAs per SM do not fit "
            f"({info})"
        )
    return prepared
