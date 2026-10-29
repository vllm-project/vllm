# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark attention front for decode batches (1 <= M <= 16) in one
launch: the QKV + o-gate projection (split-K BF16 GEMM, mma.sync, FP32
accumulate), then the per-head weightless RMS norm, interleaved RoPE, the y
store and the paged K/V insert. It writes what the projection GEMM followed by
``fused_muse_spark_qknorm_rope_kv_insert`` writes for a BF16 KV cache, up to the
GEMM's summation order.

Structure: a thread-block cluster computes one head (64 output columns), its S
CTAs splitting K into KSTRIPE-tile cyclic stripes. In each CTA a producer thread
streams the W and x boxes into a shared-memory ring with TMA (128B swizzle), four
consumer warps run mma.sync on them and an aux warp fetches the epilogue inputs.
Each CTA sends its partial of token t to the owner CTA (rank t % S) with
st.async; the owner sums the S partials in fixed rank order, rounds to BF16 and
runs the epilogue. No global scratch, no atomics, no state across calls: the
result depends on the launch shape only.

PDL: the kernel may start while its predecessor still runs. Before
griddepcontrol.wait it reads only the weights, which no kernel writes inside a
graph; every thread waits before it reads x, positions, slots or cos|sin and
before any global store.
"""

from collections.abc import Iterable
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream
from cutlass.experimental.cuda import (
    TensorMap,
    TensorMapFloatOOBFill,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

from ..common.host import assert_not_capturing, on_device
from ..common.primitives import (
    WARP_SIZE,
    add_rn,
    bf16x2_to_f32x2,
    fence_mbarrier_init_cluster,
    fma_rn,
    fms_rn,
    load_global_nc_f32_address,
    load_global_s64_address,
    load_shared_u32x4_address,
    map_shared_address_to_peer,
    mbarrier_arrive_address,
    mbarrier_arrive_expect_tx_address,
    mbarrier_init_address,
    mbarrier_wait_address,
    mbarrier_wait_cluster_address,
    mma_m16n8k16_bf16,
    mul_rn,
    pack_bf16x2_rn,
    prefetch_global_l2,
    rsqrt_approx_f32,
    store_async_cluster_f32x2,
    store_global_u32_address,
    tma_load_2d,
    tma_prefetch_2d_l2,
)

HEAD_DIM = 64
HIDDEN = 8192  # the reduction length K
BLOCK_SIZE = 16  # KV cache slots per page
MAX_TOKENS = 16
KT = 128  # k per ring stage: two 64-wide boxes
KTILES = HIDDEN // KT
KSTRIPE = 2  # adjacent k-tiles per rank before cycling across the cluster
BOXK = 64  # TMA box inner extent: 64 BF16 = 128 bytes
NBOX = KT // BOXK
W_BOX_BYTES = HEAD_DIM * BOXK * 2
CONSUMER_WARPS = 4  # 16 W rows each
PRODUCER_WARP = CONSUMER_WARPS
AUX_WARP = CONSUMER_WARPS + 1
WARPS = CONSUMER_WARPS + 2
THREADS = WARPS * WARP_SIZE
MIN_CTAS_PER_SM = 2
STAGES = 5  # keeps two CTAs resident per SM
DEEP_STAGES = 8  # the ring at S = 4 with one CTA per SM
SPLITS = (2, 4, 8)
# SM count of the K-split rule. Fixed, so the reduction order (and the result)
# does not depend on the device.
SPLIT_SMS = 152


def k_split(n: int) -> int:
    """The K split (cluster size) for an N-wide projection: one CTA per SM
    when a split of at least 4 allows it (largest power of two <= 8 with
    (n / 64) * split <= 152 CTAs), else two per SM (the same with 2 * 152
    CTAs). 1 means no kernel fits."""
    heads, one = n // HEAD_DIM, 1
    while one < 8 and heads * one * 2 <= SPLIT_SMS:
        one *= 2
    if one >= 4:
        return one
    split = 1
    while split < 8 and heads * split * 2 <= 2 * SPLIT_SMS:
        split *= 2
    return split


def ring_stages(n: int, split: int) -> int:
    """The ring depth at K split ``split``; it does not change the result."""
    return (
        DEEP_STAGES if (n // HEAD_DIM) * split <= SPLIT_SMS and split == 4 else STAGES
    )


def token_tiles(m: int) -> int:
    """8-token mma column tiles for M tokens."""
    return 1 if m <= 8 else 2


def fixed_rows(m: int) -> int:
    """The kernel variant's row count: 8 specializes M == 8, 0 any other M."""
    return 8 if m == 8 else 0


def _k_tile(i: int, rank: Int32, splits: int) -> Int32:
    """First k of the i-th k-tile cluster rank ``rank`` reduces."""
    return (
        Int32((i // KSTRIPE) * splits * KSTRIPE + i % KSTRIPE) + rank * KSTRIPE
    ) * KT


def _swizzled(row, chunk: int):
    """Byte offset of 16-byte chunk ``chunk`` of ``row`` in a 128B-swizzled
    box (1 KiB aligned)."""
    return row * 128 + ((chunk ^ (row & 7)) << 4)


class QkvFrontKernel:
    """One variant: ``tiles`` 8-token column tiles, a ``stages``-deep ring, K
    split ``splits`` (the cluster size) and ``fixed_m`` (8: M == 8, 0: any M of
    the tile count)."""

    def __init__(self, tiles: int, stages: int, splits: int, fixed_m: int) -> None:
        if tiles not in (1, 2) or splits not in SPLITS or fixed_m not in (0, 8):
            raise ValueError(f"no variant {tiles=} {splits=} {fixed_m=}")
        if stages not in (STAGES, DEEP_STAGES) or (fixed_m and tiles != 1):
            raise ValueError(f"no variant {stages=} {tiles=} {fixed_m=}")
        self.tiles, self.stages, self.splits, self.fixed_m = (
            tiles,
            stages,
            splits,
            fixed_m,
        )
        self.x_box_bytes = tiles * 8 * BOXK * 2
        self.w_bytes = NBOX * W_BOX_BYTES
        self.stage_bytes = self.w_bytes + NBOX * self.x_box_bytes
        self.max_owned = (8 * tiles + splits - 1) // splits
        self.max_recv = splits * self.max_owned
        self.nk = KTILES // splits
        self.npre = min(self.nk, stages)
        # The consumer loop is unrolled fully for two token tiles or eight
        # k-tiles; otherwise the compiler picks the unroll.
        self.full_unroll = tiles == 2 or splits == 8

    @cute.jit
    def __call__(
        self,
        w: Int64,
        x: Int64,
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        kv_cache: Int64,
        y: Int64,
        m: Int32,
        n: Int32,
        hq: Int32,
        hkv: Int32,
        rope: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ) -> None:
        row_units = HIDDEN * 2 // 16
        tm_w = create_tensor_map_tiled(
            w,
            BFloat16,
            [HIDDEN, n],
            [row_units],
            [BOXK, HEAD_DIM],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=TensorMapL2Promotion.l2_256b,
            oob_fill=TensorMapFloatOOBFill.none,
        )
        tm_x = create_tensor_map_tiled(
            x,
            BFloat16,
            [HIDDEN, m],
            [row_units],
            [BOXK, 8 * self.tiles],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=TensorMapL2Promotion.l2_256b,
            oob_fill=TensorMapFloatOOBFill.none,
        )
        self.kernel(
            tm_w, tm_x, positions, cos_sin, slots, kv_cache, y, m, n, hq, hkv, rope, eps
        ).launch(
            grid=((n // HEAD_DIM) * self.splits, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(self.splits, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=MIN_CTAS_PER_SM,
            # No carveout preference: the driver sizes it for the launch.
            preferred_smem_carveout=-1,
        )

    @cute.kernel
    def kernel(
        self,
        tm_w: cutlass.GridConstant[TensorMap],
        tm_x: cutlass.GridConstant[TensorMap],
        positions: Int64,
        cos_sin: Int64,
        slots: Int64,
        kv_cache: Int64,
        y: Int64,
        m: Int32,
        n: Int32,
        hq: Int32,
        hkv: Int32,
        rope: Int32,
        eps: Float32,
    ) -> None:
        splits, stages = self.splits, self.stages
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // WARP_SIZE
        lane = cute.arch.lane_idx()
        rank = cute.arch.block_idx_in_cluster()
        bidx, _, _ = cute.arch.block_idx()
        head = Int32(bidx) // splits
        row0 = head * HEAD_DIM
        w_map = tm_w.get_ptr().toint()
        x_map = tm_x.get_ptr().toint()

        smem = cutlass.utils.SmemAllocator()
        # The ring is 1 KiB aligned by hand: 128B-swizzled boxes need it.
        ring = smem.allocate(stages * self.stage_bytes + 1024, byte_alignment=16)
        bars = smem.allocate_array(Int64, 2 * stages + 1, byte_alignment=8)
        # The cos|sin row of each token this CTA finishes.
        s_cs = smem.allocate_array(
            Float32, self.max_owned * HEAD_DIM, byte_alignment=16
        )
        # [source rank][local token][row] split-K partials.
        s_recv = smem.allocate_array(
            Float32, self.max_recv * HEAD_DIM, byte_alignment=16
        )
        s_slot = smem.allocate_array(Int64, self.max_owned, byte_alignment=8)
        sbase = (Int32(ring.toint()) + 1023) & Int32(-1024)
        fb = Int32(bars.toint())  # full barriers, then empty barriers, then receive
        eb = fb + 8 * stages
        rb = eb + 8 * stages

        if warp == PRODUCER_WARP and lane == 0:
            # Stage 0's weights go first; the rest of the ring is initialized
            # while they are in flight. The full barrier counts bytes, so these
            # W boxes and the x boxes issued after the wait complete one phase.
            mbarrier_init_address(fb, Int32(1))
            mbarrier_init_address(eb, Int32(CONSUMER_WARPS))
            fence_mbarrier_init_cluster()
            mbarrier_arrive_expect_tx_address(fb, Int32(self.stage_bytes))
            k0 = _k_tile(0, rank, splits)
            for j in cutlass.range_constexpr(NBOX):
                tma_load_2d(sbase + j * W_BOX_BYTES, w_map, k0 + j * BOXK, row0, fb)
            for s in cutlass.range_constexpr(1, stages):
                mbarrier_init_address(fb + 8 * s, Int32(1))
                mbarrier_init_address(eb + 8 * s, Int32(CONSUMER_WARPS))
            mbarrier_init_address(rb, Int32(1))
            fence_mbarrier_init_cluster()

        # Local token u of this CTA is global token rank + u * splits.
        active_m = m
        ntok_max = (m + splits - 1) // splits
        ntok = (m - rank + splits - 1) // splits
        if cutlass.const_expr(self.fixed_m != 0):
            active_m = Int32(self.fixed_m)
            ntok_max = Int32(self.fixed_m // splits)
            ntok = Int32(self.fixed_m // splits)
        if warp == PRODUCER_WARP and lane == 0:
            # All splits partials of this CTA's tokens, its own included.
            mbarrier_arrive_expect_tx_address(rb, splits * ntok * HEAD_DIM * 4)
        cute.arch.barrier()
        # Peers wait for this arrive before their first st.async here.
        cute.arch.cluster_arrive_relaxed()

        acc = cute.make_rmem_tensor(cute.make_layout((4, self.tiles, 4)), Float32)
        for a in cutlass.range_constexpr(4):
            for b in cutlass.range_constexpr(self.tiles):
                for c in cutlass.range_constexpr(4):
                    acc[a, b, c] = Float32(0.0)

        if warp == AUX_WARP:
            self.aux(
                positions,
                cos_sin,
                slots,
                kv_cache,
                y,
                n,
                hq,
                hkv,
                rope,
                rank,
                head,
                ntok,
                s_cs,
                s_slot,
            )
        elif warp == PRODUCER_WARP:
            self.produce(w_map, x_map, sbase, fb, eb, rank, row0, lane)
        else:
            self.consume(sbase, fb, eb, warp, lane, acc)
            self.send(acc, s_recv, rb, warp, lane, rank, ntok_max, active_m)
        if warp >= CONSUMER_WARPS:
            # The producer's and aux warp's wait matching the arrive above.
            cute.arch.cluster_wait()
        cute.arch.barrier()  # s_slot and s_cs visible to every warp

        # This warp's first token's epilogue inputs, read before the receive wait.
        cv0 = Float32(0.0)
        sn0 = Float32(0.0)
        slot0 = Int64(-1)
        if warp < ntok:
            cv0 = cute.arch.load((s_cs + warp * HEAD_DIM + lane).llvm_ptr, Float32)
            sn0 = cute.arch.load((s_cs + warp * HEAD_DIM + 32 + lane).llvm_ptr, Float32)
            slot0 = cute.arch.load((s_slot + warp).llvm_ptr, Int64)
        mbarrier_wait_cluster_address(rb, Int32(0))
        # This CTA has received; the wait before exit keeps every CTA alive until
        # all peers have, so no st.async is in flight to or from an exited CTA.
        cute.arch.cluster_arrive_relaxed()
        for u in cutlass.range(warp, ntok, WARPS):
            cv, sn, slot = cv0, sn0, slot0
            if u != warp:
                cv = cute.arch.load((s_cs + u * HEAD_DIM + lane).llvm_ptr, Float32)
                sn = cute.arch.load((s_cs + u * HEAD_DIM + 32 + lane).llvm_ptr, Float32)
                slot = cute.arch.load((s_slot + u).llvm_ptr, Int64)
            token = rank + u * splits
            self.finish(
                s_recv,
                u,
                ntok_max,
                cv,
                sn,
                slot,
                rope,
                head,
                hq,
                hkv,
                eps,
                y + (Int64(token) * Int64(n) + Int64(row0)) * 2,
                kv_cache,
                lane,
            )
        cute.arch.cluster_wait()

    @cute.jit
    def aux(
        self,
        positions,
        cos_sin,
        slots,
        kv_cache,
        y,
        n,
        hq,
        hkv,
        rope,
        rank,
        head,
        ntok,
        s_cs,
        s_slot,
    ) -> None:
        """Slots and cos|sin rows of this CTA's tokens into shared memory, and
        the L2 prefetch of their y and K/V destinations. Slots and positions can
        be written by the kernels before this one in a graph: read after the
        wait."""
        lane = cute.arch.lane_idx()
        splits = self.splits
        cute.arch.griddepcontrol_wait()
        group_k, group_v, group_g = hq, hq + hkv, hq + 2 * hkv
        token = rank + lane * splits
        if lane < ntok:
            slot = load_global_s64_address(slots + Int64(token) * 8, ordered=True)
            cute.arch.store((s_slot + lane).llvm_ptr, slot)
            prefetch_global_l2(
                y + (Int64(token) * Int64(n) + Int64(head * HEAD_DIM)) * 2
            )
            if slot >= 0 and head >= group_k and head < group_g:
                if head < group_v:
                    prefetch_global_l2(_cache_row(kv_cache, slot, hkv, head - group_k))
                else:
                    prefetch_global_l2(
                        _cache_row(kv_cache, slot, hkv, head - group_v) + HEAD_DIM * 2
                    )
        if rope != 0 and head < group_v:
            position = Int64(0)
            if lane < ntok:
                position = load_global_s64_address(
                    positions + Int64(token) * 8, ordered=True
                )
            for u in cutlass.range(ntok):
                p = cute.arch.shuffle_sync(position, u)
                row = cos_sin + p * (HEAD_DIM * 4)
                cute.arch.store(
                    (s_cs + u * HEAD_DIM + lane).llvm_ptr,
                    load_global_nc_f32_address(row + Int64(lane) * 4),
                )
                cute.arch.store(
                    (s_cs + u * HEAD_DIM + 32 + lane).llvm_ptr,
                    load_global_nc_f32_address(row + Int64(32 + lane) * 4),
                )

    @cute.jit
    def produce(self, w_map, x_map, sbase, fb, eb, rank, row0, lane) -> None:
        """The producer thread: the weights of stages 1..npre-1 and an L2
        prefetch of the next ring's weights, then (after the wait) the
        activations of those stages and the steady state."""
        splits, stages, stage_bytes = self.splits, self.stages, self.stage_bytes
        if lane == 0:
            for i in cutlass.range_constexpr(1, self.npre):
                bar = fb + 8 * i
                k0 = _k_tile(i, rank, splits)
                mbarrier_arrive_expect_tx_address(bar, Int32(stage_bytes))
                for j in cutlass.range_constexpr(NBOX):
                    tma_load_2d(
                        sbase + i * stage_bytes + j * W_BOX_BYTES,
                        w_map,
                        k0 + j * BOXK,
                        row0,
                        bar,
                    )
            for i in cutlass.range_constexpr(
                self.npre, min(self.nk, self.npre + stages)
            ):
                k0 = _k_tile(i, rank, splits)
                for j in cutlass.range_constexpr(NBOX):
                    tma_prefetch_2d_l2(w_map, k0 + j * BOXK, row0)
        cute.arch.griddepcontrol_wait()
        if lane == 0:
            for i in cutlass.range_constexpr(self.npre):
                k0 = _k_tile(i, rank, splits)
                for j in cutlass.range_constexpr(NBOX):
                    tma_load_2d(
                        sbase + i * stage_bytes + self.w_bytes + j * self.x_box_bytes,
                        x_map,
                        k0 + j * BOXK,
                        Int32(0),
                        fb + 8 * i,
                    )
            for i in cutlass.range(self.npre, self.nk, 1):
                st = i % stages
                mbarrier_wait_address(eb + 8 * st, ((i // stages) & 1) ^ 1)
                sw = sbase + st * stage_bytes
                bar = fb + 8 * st
                k0 = _k_tile_dynamic(i, rank, splits)
                mbarrier_arrive_expect_tx_address(bar, Int32(stage_bytes))
                for j in cutlass.range_constexpr(NBOX):
                    tma_load_2d(sw + j * W_BOX_BYTES, w_map, k0 + j * BOXK, row0, bar)
                for j in cutlass.range_constexpr(NBOX):
                    tma_load_2d(
                        sw + self.w_bytes + j * self.x_box_bytes,
                        x_map,
                        k0 + j * BOXK,
                        Int32(0),
                        bar,
                    )

    @cute.jit
    def consume(self, sbase, fb, eb, warp, lane, acc) -> None:
        """Consumer warp ``warp``: W rows 16 warp .. 16 warp + 15 over all 128 k
        of every stage. The 32-k block b lives in box b / 2 at 16-byte chunks
        2 t + (b & 1); A and B take the same k permutation. One accumulator set
        per k block."""
        g = lane >> 2
        t = lane & 3
        a_off = []
        x_off = []
        for b in cutlass.range_constexpr(4):
            box, chunk = b >> 1, b & 1
            c = 2 * t + chunk
            a_off.append(
                [
                    box * W_BOX_BYTES + _swizzled(16 * warp + g, c),
                    box * W_BOX_BYTES + _swizzled(16 * warp + g + 8, c),
                ]
            )
            x_off.append(
                [
                    self.w_bytes + box * self.x_box_bytes + _swizzled(8 * nt + g, c)
                    for nt in range(self.tiles)
                ]
            )
        # The epilogue stores of this warp come after the wait; the ring is
        # gated by the full barriers.
        cute.arch.griddepcontrol_wait()
        for i in cutlass.range(self.nk, unroll_full=self.full_unroll):
            st = i % self.stages
            mbarrier_wait_address(fb + 8 * st, (i // self.stages) & 1)
            sw = sbase + st * self.stage_bytes
            for b in cutlass.range_constexpr(4):
                # Rows >= M of the x boxes are zero-filled.
                bx = [
                    load_shared_u32x4_address(sw + x_off[b][nt])
                    for nt in range(self.tiles)
                ]
                a0 = load_shared_u32x4_address(sw + a_off[b][0])
                a1 = load_shared_u32x4_address(sw + a_off[b][1])
                for nt in cutlass.range_constexpr(self.tiles):
                    for half in cutlass.range_constexpr(2):
                        d = mma_m16n8k16_bf16(
                            (
                                a0[2 * half],
                                a1[2 * half],
                                a0[2 * half + 1],
                                a1[2 * half + 1],
                            ),
                            bx[nt][2 * half],
                            bx[nt][2 * half + 1],
                            (
                                acc[b, nt, 0],
                                acc[b, nt, 1],
                                acc[b, nt, 2],
                                acc[b, nt, 3],
                            ),
                        )
                        for e in cutlass.range_constexpr(4):
                            acc[b, nt, e] = d[e]
            if lane == 0:
                mbarrier_arrive_address(eb + 8 * st)

    @cute.jit
    def send(self, acc, s_recv, rb, warp, lane, rank, ntok_max, active_m) -> None:
        """This CTA's partial of rows 16 warp + g (+8) for tokens 8 nt + 2 t
        (+1), the k-block sets summed in fixed order, into the owner rank's
        receive slot [rank][token / splits][row]: rows (r, r + 1) of one token
        per lane, paired by one xor-4 shuffle (lanes g and g ^ 1 share t)."""
        splits = self.splits
        g = lane >> 2
        t = lane & 3
        for nt in cutlass.range_constexpr(self.tiles):
            for e in cutlass.range_constexpr(4):
                acc[0, nt, e] = add_rn(
                    add_rn(add_rn(acc[0, nt, e], acc[1, nt, e]), acc[2, nt, e]),
                    acc[3, nt, e],
                )
        cute.arch.cluster_wait()  # every peer's receive barrier is initialized
        recv = Int32(s_recv.toint())
        odd = g & 1
        for nt in cutlass.range_constexpr(self.tiles):
            for h in cutlass.range_constexpr(2):
                a = acc[0, nt, 2 * h]  # tokens 2t and 2t+1 of row r
                b = acc[0, nt, 2 * h + 1]
                mine = b
                if odd != 0:
                    mine = a
                other = cute.arch.shuffle_sync_bfly(mine, 4)
                lo = a
                hi = other
                if odd != 0:
                    lo = other
                    hi = b
                row = 16 * warp + 8 * h + (g & ~1)  # the pair's even row
                token = 8 * nt + 2 * t + odd
                if token < active_m:
                    owner = token % splits
                    offset = ((rank * ntok_max + token // splits) * HEAD_DIM + row) * 4
                    store_async_cluster_f32x2(
                        map_shared_address_to_peer(recv + offset, owner),
                        lo,
                        hi,
                        map_shared_address_to_peer(rb, owner),
                    )

    @cute.jit
    def finish(
        self,
        s_recv,
        u,
        ntok_max,
        cv,
        sn,
        slot,
        rope,
        head,
        hq,
        hkv,
        eps,
        y_row,
        kv_cache,
        lane,
    ) -> None:
        """Epilogue of local token ``u`` (head columns 2 lane, 2 lane + 1): the
        splits partials summed in rank order and rounded to BF16, then for q and
        k heads the RMS norm and RoPE (``cv``, ``sn``: this lane's cos and
        sin), the y store and, for k and v heads, the cache row of ``slot``."""
        group_k, group_v, group_g = hq, hq + hkv, hq + 2 * hkv
        vx = Float32(0.0)
        vy = Float32(0.0)
        parts = []
        for s in cutlass.range_constexpr(self.splits):
            base = s_recv + (s * ntok_max + u) * HEAD_DIM + 2 * lane
            parts.append(
                (
                    cute.arch.load(base.llvm_ptr, Float32),
                    cute.arch.load((base + 1).llvm_ptr, Float32),
                )
            )
        for s in cutlass.range_constexpr(self.splits):
            vx = add_rn(vx, parts[s][0])
            vy = add_rn(vy, parts[s][1])
        raw = pack_bf16x2_rn(vx, vy)
        out = raw
        if head < group_v:
            h0, h1 = bf16x2_to_f32x2(raw)
            ss = fma_rn(h0, h0, mul_rn(h1, h1))
            for o in cutlass.range_constexpr(5):
                ss = add_rn(ss, cute.arch.shuffle_sync_bfly(ss, 16 >> o))
            rn = rsqrt_approx_f32(fma_rn(ss, Float32(1.0 / HEAD_DIM), eps))
            o0 = mul_rn(h0, rn)
            o1 = mul_rn(h1, rn)
            if rope != 0:
                r0 = fms_rn(o0, cv, mul_rn(o1, sn))
                r1 = fma_rn(o1, cv, mul_rn(o0, sn))
                o0 = r0
                o1 = r1
            out = pack_bf16x2_rn(o0, o1)
            if head >= group_k and slot >= 0:
                store_global_u32_address(
                    _cache_row(kv_cache, slot, hkv, head - group_k) + Int64(lane) * 4,
                    out,
                )
        elif head < group_g and slot >= 0:
            store_global_u32_address(
                _cache_row(kv_cache, slot, hkv, head - group_v)
                + Int64(HEAD_DIM * 2 + lane * 4),
                raw,
            )
        store_global_u32_address(y_row + Int64(lane) * 4, out)


def _k_tile_dynamic(i: Int32, rank: Int32, splits: int) -> Int32:
    """``_k_tile`` of a dynamic tile index."""
    return ((i // KSTRIPE) * (splits * KSTRIPE) + rank * KSTRIPE + i % KSTRIPE) * KT


def _cache_row(kv_cache: Int64, slot: Int64, hkv: Int32, kv_head: Int32) -> Int64:
    """Byte address of the [pages, hkv, 16, 128] BF16 cache row of ``slot``
    for KV head ``kv_head`` (64-bit: cache offsets exceed 2^31)."""
    page = slot >> 4
    row = (page * Int64(hkv) + Int64(kv_head)) * BLOCK_SIZE + (slot & 15)
    return kv_cache + row * (2 * HEAD_DIM * 2)


def variant(n: int, m: int) -> tuple[int, int, int, int]:
    """(tiles, stages, splits, fixed_m) of the kernel for N columns, M rows."""
    splits = k_split(n)
    return token_tiles(m), ring_stages(n, splits), splits, fixed_rows(m)


def supports_shape(n: int, num_kv_heads: int) -> bool:
    """Whether a variant computes an N-wide projection with these KV heads."""
    return (
        n % (2 * HEAD_DIM) == 0
        and 0 < num_kv_heads < n // (2 * HEAD_DIM)
        and k_split(n) in SPLITS
    )


@cache
def compile_qkv_front(
    tiles: int, stages: int, splits: int, fixed_m: int, options: str = ""
):
    """The compiled variant, shared by every layer and shape it serves, with
    these extra compile ``options``; called (TVM FFI) as fn(w, x, positions,
    cos_sin, slots, kv_cache, y, m, n, hq, hkv, rope, eps, stream) with byte
    addresses and the stream handle as ints."""
    assert_not_capturing("the QKV front")
    address = Int64(0)
    return cute.compile(
        QkvFrontKernel(tiles, stages, splits, fixed_m),
        *(address,) * 7,
        Int32(1),
        Int32(HEAD_DIM * 4),
        Int32(1),
        Int32(1),
        Int32(1),
        Float32(1e-5),
        make_fake_stream(),
        options=f"--enable-tvm-ffi {options}".strip(),
    )


def prepare(n: int, rows: Iterable[int] = range(1, MAX_TOKENS + 1)) -> None:
    """Compile the variants that N columns and these row counts take (before
    CUDA graph capture)."""
    for m in rows:
        compile_qkv_front(*variant(n, m))


def hidden_rows_ok(t: torch.Tensor) -> bool:
    """Whether ``t`` is a contiguous, 16-byte aligned bf16 [rows, 8192]
    matrix, as x and the weight must be."""
    return (
        t.dim() == 2
        and t.dtype == torch.bfloat16
        and t.shape[1] == HIDDEN
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


def kv_cache_layout_ok(kv_cache: torch.Tensor, num_kv_heads: int) -> bool:
    """Whether a KV cache has the layout the QKV front and attn2 read: bf16
    ``[pages, num_kv_heads, 16, 128]`` (K | V per slot), dense pages of dense
    heads, 16-byte aligned."""
    row = 2 * HEAD_DIM
    if (
        kv_cache.dtype != torch.bfloat16
        or kv_cache.dim() != 4
        or tuple(kv_cache.shape[1:]) != (num_kv_heads, BLOCK_SIZE, row)
    ):
        return False
    stride = kv_cache.stride()
    return (
        stride[0] == num_kv_heads * BLOCK_SIZE * row
        and (num_kv_heads == 1 or stride[1] == BLOCK_SIZE * row)
        and stride[2] == row
        and stride[3] == 1
        and kv_cache.data_ptr() % 16 == 0
    )


def _vector_ok(t: torch.Tensor, m: int) -> bool:
    return (
        t.dtype == torch.int64 and t.dim() == 1 and t.shape[0] >= m and t.stride(0) == 1
    )


def unsupported(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor | None = None,
) -> str | None:
    """Why the kernel cannot take this call (with ``out``, also its output),
    or None."""
    if not x.is_cuda:
        return "x must be a CUDA tensor"
    if not hidden_rows_ok(x):
        return f"x must be a contiguous, 16-byte aligned bf16 [M, {HIDDEN}] tensor"
    m = x.shape[0]
    if not 1 <= m <= MAX_TOKENS:
        return f"M = {m} is outside 1..{MAX_TOKENS}"
    if not hidden_rows_ok(weight):
        return f"weight must be a contiguous, 16-byte aligned bf16 [N, {HIDDEN}] tensor"
    n = weight.shape[0]
    if kv_cache.dim() != 4:
        return "kv_cache must be 4-D"
    hkv = kv_cache.shape[1]
    if not supports_shape(n, hkv):
        return f"no kernel for N = {n} with {hkv} KV heads"
    if not _vector_ok(positions, m):
        return "positions must be a dense int64 vector of at least M entries"
    if not _vector_ok(slot_mapping, m):
        return "slot_mapping must be a dense int64 vector of at least M entries"
    device = x.device
    if cos_sin_cache is not None and not (
        cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape[1] == HEAD_DIM
        and cos_sin_cache.is_contiguous()
        and cos_sin_cache.device == device
    ):
        return (
            f"cos_sin_cache must be a contiguous fp32 [P, {HEAD_DIM}] tensor on "
            "x's device"
        )
    if not kv_cache_layout_ok(kv_cache, hkv):
        return (
            f"kv_cache must be a 16-byte aligned bf16 [pages, HKV, {BLOCK_SIZE}, "
            f"{2 * HEAD_DIM}] tensor with rows at (page * HKV + head) * 16 + offset"
        )
    tensors = [weight, positions, slot_mapping, kv_cache]
    if out is not None:
        # The y stores are 4-byte words.
        if not (
            out.dtype == torch.bfloat16
            and out.dim() == 2
            and tuple(out.shape) == (m, n)
            and out.is_contiguous()
            and out.data_ptr() % 4 == 0
        ):
            return "out must be a contiguous, 4-byte aligned bf16 [M, N] tensor"
        tensors.append(out)
    if not all(t.device == device for t in tensors):
        return "all tensors must be on x's device"
    return None


def launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """``qkv_front`` without its checks: for a call ``unsupported`` took with
    this ``out``."""
    m, n = x.shape[0], weight.shape[0]
    hkv = kv_cache.shape[1]
    device = x.get_device()
    on_device(
        device,
        _compile_and_call,
        device,
        n,
        m,
        weight.data_ptr(),
        x.data_ptr(),
        positions.data_ptr(),
        0 if cos_sin_cache is None else cos_sin_cache.data_ptr(),
        slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        out.data_ptr(),
        m,
        n,
        n // (2 * HEAD_DIM) - hkv,
        hkv,
        int(cos_sin_cache is not None),
        eps,
    )


def _compile_and_call(device: int, n: int, m: int, *args) -> None:
    """The (N, M) variant called on the current stream of ``device``, which is
    current: a first compile targets the inputs' device too."""
    compile_qkv_front(*variant(n, m))(*args, torch._C._cuda_getCurrentRawStream(device))


def qkv_front(
    x: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """Out [M, N] = x [M, 8192] @ weight [N, 8192]^T with the q and k heads
    RMS-normalized (and rotated when ``cos_sin_cache`` is given), and the
    normalized k and raw v heads written to the kv_cache rows of
    ``slot_mapping`` (-1: none). N = (2 HQ + 2 HKV) * 64, HKV =
    kv_cache.size(1). Launched with PDL on the current stream of x's device;
    raises ValueError for a call outside the kernel (``unsupported``).
    Graph-capturable once ``prepare`` has compiled the variant."""
    reason = unsupported(
        x, weight, positions, cos_sin_cache, slot_mapping, kv_cache, out
    )
    if reason is not None:
        raise ValueError(reason)
    launch(x, weight, positions, cos_sin_cache, slot_mapping, kv_cache, out, eps)
