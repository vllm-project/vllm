# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
"""SM90 FlashTopK refine in CuTe DSL (FP8 E4M3, 32 heads), the second kernel of key
pruning: one CTA per 8-row tile and KV split scores in full the keys prune_sm90.py's
mask keeps for the tile, and leaves select the candidates the scan would.

refine_sm100.py's design on SM90: the gather warpgroup counts the survivors of each
128-key block of the CTA's range into a prefix sum in shared memory, then gather warp w
fills compact tile w (64 keys, the WGMMA's M) with survivors 64t .. 64t + 63 of every
tile t = w (mod 4): each lane locates its two keys' blocks by binary search and their
bits with fns, and copies the K rows (cp.async, 16 bytes at a time, 128-byte swizzle)
with each key's index and kv scale. Math warpgroup g takes the tiles t = g (mod 2), in
turn with the other group (sm90_utils.wgmma_in_turn), one M=64 x N=256 WGMMA per tile,
and scores and appends exactly as scan_sm90.py (score_owned, owned_hits, append_owned),
so pruning selects the scan's bits; like the scan it issues tile t + 2's WGMMA before
writing tile t's hits, which here are dense (the keys passed pruning). Warp roles (384
threads): 0-7 math, 8-11 gather (warp 8 first loads Q and the weights by TMA).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as hopper_helpers
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup

from .common import (
    BLOCK_KV,
    HEAD_DIM,
    NUM_HEADS,
    cp_async_4,
    cp_async_16,
    find_nth_set,
    load_global_v4_b32,
    loose_raw_gate,
    select_f32,
    select_i32,
)
from .common import REFINE_BLOCK_Q as BLOCK_Q
from .scan_sm90 import (
    append_owned,
    fill_head_table,
    head_table_layout,
    owned_hits,
    owned_rows,
    score_owned,
)
from .sm90_utils import (
    CTA_THREADS,
    MATH_BARRIER,
    MATH_THREADS,
    PRODUCER_WARP,
    WARPGROUP_THREADS,
    WARPS_PER_GROUP,
    WGMMA_KEYS,
    cta_kv_range,
    group_blocks,
    issue_q_tma,
    load_row_count,
    partition_q_tma,
    store_row_counts,
    store_row_window,
    wgmma_in_turn,
)

TILE_KEYS = WGMMA_KEYS  # keys of a compact tile: the WGMMA's M
RING_STAGES = 4  # compact tiles in flight: gather warp w fills stage w
N = BLOCK_Q * NUM_HEADS  # the WGMMA's N: the tile's rows x heads
ROW_BYTES = HEAD_DIM  # an FP8 K row
GATHER_THREADS = WARPGROUP_THREADS  # warps 8-11
GATHER_BARRIER = 4  # named barrier of the gather warpgroup
# Mask blocks a CTA's range may span: its survivor prefix is in shared memory.
MAX_BLOCKS = 16384


def _swizzled(slot, chunk):
    """Byte offset of 16-byte chunk `chunk` (< 8) of row `slot` in a 64-row K-major
    tile of 128-byte FP8 rows with the 128-byte swizzle: each row's chunks XOR-ed with
    its row mod 8."""
    return slot * 128 + (((slot % 8) ^ (chunk % 8)) * 16)


class RefineSm90:
    def __init__(self, *, single_split: bool):
        self.single_split = single_split
        self.math_regs = 224
        self.other_regs = 56

    @cute.jit
    def __call__(
        self,
        # [seq_len_kv, 128] uint8 (FP8 E4M3 bits), gathered by cp.async
        kv: cute.Tensor,
        q: cute.Tensor,  # [num_q * 32, 128, 1] uint8 (FP8 E4M3 bits)
        weights: cute.Tensor,  # [num_q, 32] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32
        # [num_q_tiles * mask_blocks * 4] i32: the pruning pass's, one 8-row tile a
        # plane
        mask: cute.Tensor,
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        cand_val: cute.Tensor,
        cand_idx: cute.Tensor,
        cand_cnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        stream: cuda.CUstream,
    ):
        elem = cutlass.Float8E4M3FN
        b = cute.recast_tensor(q, elem)
        mma_tiler = (TILE_KEYS, N, HEAD_DIM)
        tiled_mma = hopper_helpers.make_trivial_tiled_mma(
            elem,
            elem,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            (1, 1, 1),
            (TILE_KEYS, N),
        )
        a_smem_staged = hopper_helpers.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR, mma_tiler, elem, RING_STAGES
        )
        b_smem_staged = hopper_helpers.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR, mma_tiler, elem, 1
        )
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_op, b, cute.slice_(b_smem_staged, (None, None, 0)), (N, HEAD_DIM)
        )
        w_layout = cute.make_layout((BLOCK_Q, NUM_HEADS), stride=(NUM_HEADS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_op, weights, w_layout, (BLOCK_Q, NUM_HEADS)
        )
        # The Q tile (one byte an element) and the weights.
        self.q_tx_bytes = N * HEAD_DIM + N * 4

        @cute.struct
        class SharedStorage:
            # q, count, full_tile[RING_STAGES], empty_tile[RING_STAGES]
            mbar: cute.struct.MemRange[cutlass.Int64, 2 + 2 * RING_STAGES]

        num_tiles = (num_q + BLOCK_Q - 1) // BLOCK_Q
        self.kernel(
            tiled_mma,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            kv,
            kv_scales,
            mask,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            cand_val,
            cand_idx,
            cand_cnt,
            num_q,
            seq_len_kv,
            mask_blocks,
            num_splits,
            a_smem_staged,
            b_smem_staged,
            w_layout,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
            block=[CTA_THREADS, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        mKV: cute.Tensor,
        mScales: cute.Tensor,
        mMask: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mCandVal: cute.Tensor,
        mCandIdx: cute.Tensor,
        mCandCnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        w_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        q_tile = gdimx - 1 - bidx
        query_start = q_tile * BLOCK_Q

        split_lo, split_hi, range_lo, range_hi, has_work = cta_kv_range(
            BLOCK_KV,
            BLOCK_Q,
            query_start,
            bidy,
            num_q,
            seq_len_kv,
            num_splits,
            mRowStart,
            mRowEnd,
        )
        kv_start = range_lo // BLOCK_KV * BLOCK_KV
        # The 128-key mask blocks of the CTA's range.
        nb = Int32(0)
        if has_work:
            nb = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
        # A CTA whose rows have no keys in its split leaves before any setup (adaptive
        # pruning launches the path it did not take with empty windows).
        if has_work:
            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            mbar = storage.mbar.data_ptr()
            q_bar, count_bar = mbar, mbar + 1
            full_tile, empty_tile = mbar + 2, mbar + 2 + RING_STAGES
            sQ = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=b_smem_staged.outer,
                byte_alignment=1024,
                swizzle=b_smem_staged.inner,
            )
            sTile = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=a_smem_staged.outer,
                byte_alignment=1024,
                swizzle=a_smem_staged.inner,
            )
            sMetaIdx = smem.allocate_tensor(
                Int32,
                cute.make_layout((RING_STAGES, TILE_KEYS), stride=(TILE_KEYS, 1)),
                16,
            )
            sMetaScale = smem.allocate_tensor(
                Float32,
                cute.make_layout((RING_STAGES, TILE_KEYS), stride=(TILE_KEYS, 1)),
                16,
            )
            sPrefix = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_BLOCKS + 1,)), 16
            )
            sW = smem.allocate_tensor(Float32, cute.make_layout((N,)), 128)
            sWq = smem.allocate_tensor(Float32, head_table_layout(), 128)
            sGate = smem.allocate_tensor(Float32, cute.make_layout((BLOCK_Q,)), 16)
            sWin = smem.allocate_tensor(
                Int32, cute.make_layout((BLOCK_Q, 2), stride=(2, 1)), 16
            )
            # The survivor count, then the gather warps' totals; then the rows'
            # candidate counts (one KV split).
            sCounts = smem.allocate_tensor(Int32, cute.make_layout((5,)), 16)
            sCnt = smem.allocate_tensor(Int32, cute.make_layout((BLOCK_Q,)), 16)

            if tidx == 0:
                cute.arch.mbarrier_init(q_bar, 1)
                cute.arch.mbarrier_init(count_bar, 1)
                for i in cutlass.range_constexpr(RING_STAGES):
                    # Its gather warp, once the copies have landed.
                    cute.arch.mbarrier_init(full_tile + i, 1)
                    # The consuming group's warps, done with it.
                    cute.arch.mbarrier_init(empty_tile + i, WARPS_PER_GROUP)
                cute.arch.mbarrier_init_fence()
            if tidx < BLOCK_Q:
                row_q = query_start + tidx
                row = min(row_q, num_q - 1)
                sGate[tidx] = loose_raw_gate(
                    mInvDelta[row], mOrigin[row], mThBucket[row]
                )
                store_row_window(
                    sWin,
                    tidx,
                    row_q,
                    num_q,
                    seq_len_kv,
                    split_lo,
                    split_hi,
                    mRowStart,
                    mRowEnd,
                )
                load_row_count(sCnt, tidx, row_q, num_q, mCandCnt, self.single_split)
            cute.arch.sync_threads()

            if warp_idx >= PRODUCER_WARP:
                cute.arch.setmaxregister_decrease(self.other_regs)
                if has_work:
                    if warp_idx == PRODUCER_WARP:
                        q_tma = partition_q_tma(
                            tma_atom_b,
                            mB,
                            sQ,
                            tma_atom_w,
                            mW,
                            sW,
                            w_layout,
                            query_start,
                            NUM_HEADS,
                            N,
                            (BLOCK_Q, NUM_HEADS),
                        )
                        if lane == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                q_bar, self.q_tx_bytes
                            )
                        cute.arch.sync_warp()
                        issue_q_tma(q_tma, q_bar)
                    self._gather(
                        tidx - MATH_THREADS,
                        warp_idx - PRODUCER_WARP,
                        lane,
                        nb,
                        kv_start,
                        q_tile,
                        mask_blocks,
                        mMask,
                        mKV,
                        mScales,
                        sPrefix,
                        sTile,
                        sMetaIdx,
                        sMetaScale,
                        sCounts,
                        count_bar,
                        full_tile,
                        empty_tile,
                    )
            else:
                cute.arch.setmaxregister_increase(self.math_regs)
                self._math(
                    tiled_mma,
                    sTile,
                    sQ,
                    sW,
                    sWq,
                    sMetaIdx,
                    sMetaScale,
                    sGate,
                    sWin,
                    sCounts,
                    sCnt,
                    q_bar,
                    count_bar,
                    full_tile,
                    empty_tile,
                    has_work,
                    query_start,
                    tidx,
                    warp_idx,
                    lane,
                    mCandVal,
                    mCandIdx,
                    mCandCnt,
                )
                if cutlass.const_expr(self.single_split):
                    store_row_counts(sCnt, tidx, query_start, num_q, mCandCnt, BLOCK_Q)

    @cute.jit
    def _gather(
        self,
        gt,
        gw,
        lane,
        nb,
        kv_start,
        q_tile,
        mask_blocks,
        mMask,
        mKV,
        mScales,
        sPrefix,
        sTile,
        sMetaIdx,
        sMetaScale,
        sCounts,
        count_bar,
        full_tile,
        empty_tile,
    ):
        """Gather thread gt (warp gw): survivors per mask block into an exclusive
        prefix, then compact tiles t = gw (mod RING_STAGES)."""
        mask_base = (
            mMask.iterator.toint()
            + (cutlass.Int64(q_tile) * mask_blocks + kv_start // BLOCK_KV) * 16
        )
        # Each gather thread counts a run of consecutive mask blocks.
        run = (nb + (GATHER_THREADS - 1)) // GATHER_THREADS
        b_lo = min(gt * run, nb)
        b_hi = min(b_lo + run, nb)
        local = Int32(0)
        for blk in cutlass.range(b_lo, b_hi, 4, unroll=1):
            words = [
                load_global_v4_b32(
                    mask_base + cutlass.Int64(min(blk + u, b_hi - 1)) * 16
                )
                for u in range(4)
            ]
            for u in cutlass.range_constexpr(4):
                c = Int32(0)
                for word in words[u]:
                    c = c + Int32(cute.arch.popc(word))
                in_run = blk + u < b_hi
                if in_run:
                    sPrefix[blk + u] = c
                local = local + select_i32(in_run, c, 0)
        incl = local
        for off in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, 1 << off, mask_and_clamp=0)
            if lane >= (1 << off):
                incl = incl + up
        if lane == 31:
            sCounts[1 + gw] = incl
        cute.arch.barrier(barrier_id=GATHER_BARRIER, number_of_threads=GATHER_THREADS)
        base = Int32(0)
        total = Int32(0)
        for i in cutlass.range_constexpr(4):
            v = sCounts[1 + i]
            base = base + select_i32(i < gw, v, 0)
            total = total + v
        running = base + incl - local
        for blk in cutlass.range(b_lo, b_hi, unroll=1):
            c = sPrefix[blk]
            sPrefix[blk] = running
            running = running + c
        if gt == 0:
            sPrefix[nb] = total
        cute.arch.barrier(barrier_id=GATHER_BARRIER, number_of_threads=GATHER_THREADS)
        if gt == 0:
            # Review fix (changes codegen): sCounts[0] is written only after the second
            # barrier, so it no longer races (WAR) with the other gather threads' reads
            # of sCounts[1..4] above. The math warps read it after waiting on count_bar,
            # which this arrive publishes it with.
            sCounts[0] = total
            cute.arch.mbarrier_arrive(count_bar)

        tiles = (total + TILE_KEYS - 1) // TILE_KEYS
        tile_bytes = TILE_KEYS * ROW_BYTES
        smem_base = sTile.iterator.toint()
        scale_smem = sMetaScale.iterator.toint()
        kv_base = mKV.iterator.toint()
        scale_base = mScales.iterator.toint()
        i0, i1 = Int32(0), Int32(0)
        if gw < tiles:
            i0, i1 = self._locate(gw, lane, nb, kv_start, total, mask_base, sPrefix)
        for t in cutlass.range(gw, tiles, RING_STAGES, unroll=1):
            s = t % RING_STAGES
            cute.arch.mbarrier_wait(empty_tile + s, (t // RING_STAGES) % 2 ^ 1)
            keys = (i0, i1)
            for j in cutlass.range_constexpr(2):
                slot = j * 32 + lane
                if t * TILE_KEYS + slot < total:
                    src = kv_base + cutlass.Int64(keys[j]) * ROW_BYTES
                    dst = smem_base + s * tile_bytes
                    for chunk in cutlass.range_constexpr(ROW_BYTES // 16):
                        cp_async_16(dst + _swizzled(slot, chunk), src + chunk * 16)
                    sMetaIdx[(s, slot)] = keys[j]
                    cp_async_4(
                        scale_smem + (s * TILE_KEYS + slot) * 4,
                        scale_base + cutlass.Int64(keys[j]) * 4,
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(full_tile + s)
            if t + RING_STAGES < tiles:
                i0, i1 = self._locate(
                    t + RING_STAGES, lane, nb, kv_start, total, mask_base, sPrefix
                )

    @cute.jit
    def _locate(self, t, lane, nb, kv_start, total, mask_base, sPrefix):
        """The keys of tile t's slots 32j + lane: the last block whose prefix is at most
        the survivor's rank (ranks past the last survivor clamped to it), then the
        rank's bit in the block's mask. The two searches interleave."""
        g0 = min(t * TILE_KEYS + lane, total - 1)
        g1 = min(t * TILE_KEYS + 32 + lane, total - 1)
        b0, b1 = Int32(0), Int32(0)
        length = nb
        while length > 1:
            half = length // 2
            b0 = select_i32(sPrefix[b0 + half] <= g0, b0 + half, b0)
            b1 = select_i32(sPrefix[b1 + half] <= g1, b1 + half, b1)
            length = length - half
        blks = (b0, b1)
        words = [load_global_v4_b32(mask_base + cutlass.Int64(b) * 16) for b in blks]
        keys = []
        for j, g in enumerate((g0, g1)):
            r = g - sPrefix[blks[j]]
            w = words[j]
            part = Int32(0)
            word = w[0]
            for i in cutlass.range_constexpr(3):
                cnt = Int32(cute.arch.popc(word))
                step = (part == i) & (r >= cnt)
                r = select_i32(step, r - cnt, r)
                part = select_i32(step, part + 1, part)
                word = select_i32(step, w[i + 1], word)
            keys.append(
                kv_start + blks[j] * BLOCK_KV + part * 32 + find_nth_set(word, r + 1)
            )
        return keys[0], keys[1]

    @cute.jit
    def _math(
        self,
        tiled_mma,
        sTile,
        sQ,
        sW,
        sWq,
        sMetaIdx,
        sMetaScale,
        sGate,
        sWin,
        sCounts,
        sCnt,
        q_bar,
        count_bar,
        full_tile,
        empty_tile,
        has_work,
        query_start,
        tidx,
        warp_idx,
        lane,
        mCandVal,
        mCandIdx,
        mCandCnt,
    ):
        g = warp_idx // WARPS_PER_GROUP
        tid_wg = tidx % WARPGROUP_THREADS
        warp_in_wg = warp_idx % WARPS_PER_GROUP
        thr_mma = tiled_mma.get_slice(tid_wg)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sTile))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((TILE_KEYS, N)), Float32)
        num_k = cute.size(tCrA.shape[2])
        cap = mCandVal.layout.shape[1]
        lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        key_lo = 16 * warp_in_wg + lane // 4
        if has_work:
            cute.arch.mbarrier_wait(count_bar, 0)
            n = sCounts[0]
            tiles = (n + TILE_KEYS - 1) // TILE_KEYS
            cute.arch.mbarrier_wait(q_bar, 0)
            fill_head_table(sW, sWq, tidx)
            cute.arch.barrier(barrier_id=MATH_BARRIER, number_of_threads=MATH_THREADS)
            gates, win_lo, win_len = owned_rows(sGate, sWin, lane)
            count = group_blocks(tiles, g)
            count_other = group_blocks(tiles, 1 - g)
            if count > 0:
                self._issue(
                    tiled_mma,
                    acc,
                    tCrA,
                    tCrB,
                    num_k,
                    full_tile,
                    g,
                    Int32(0),
                    count_other,
                )
            for c in cutlass.range(count, unroll=1):
                t = 2 * c + g
                s = t % RING_STAGES
                warpgroup.wait_group(0)
                valid0 = t * TILE_KEYS + key_lo < n
                valid1 = t * TILE_KEYS + key_lo + 8 < n
                key0 = select_i32(valid0, sMetaIdx[(s, key_lo)], Int32(-1))
                key1 = select_i32(valid1, sMetaIdx[(s, key_lo + 8)], Int32(-1))
                scale0 = select_f32(valid0, sMetaScale[(s, key_lo)], Float32(0.0))
                scale1 = select_f32(valid1, sMetaScale[(s, key_lo + 8)], Float32(0.0))
                # The tile's operands and metadata are used: it may be refilled.
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(empty_tile + s)
                raws = score_owned(acc, sWq, lane, scale0, scale1)
                takes, ballots = owned_hits(raws, key0, key1, gates, win_lo, win_len)
                if c + 1 < count:
                    self._issue(
                        tiled_mma,
                        acc,
                        tCrA,
                        tCrB,
                        num_k,
                        full_tile,
                        g,
                        c + 1,
                        count_other,
                    )
                append_owned(
                    raws,
                    takes,
                    ballots,
                    lane,
                    lt_mask,
                    key0,
                    key1,
                    sCnt,
                    query_start,
                    cap,
                    mCandVal,
                    mCandIdx,
                    mCandCnt,
                    self.single_split,
                )

    @cute.jit
    def _issue(
        self,
        tiled_mma,
        acc,
        tCrA,
        tCrB,
        num_k: cutlass.Constexpr,
        full_tile,
        g,
        c,
        count_other,
    ):
        """Wait for tile 2c + g and issue its WGMMA in turn with the other group."""
        t = 2 * c + g
        s = t % RING_STAGES
        cute.arch.mbarrier_wait(full_tile + s, (t // RING_STAGES) % 2)
        wgmma_in_turn(tiled_mma, acc, tCrA, tCrB, num_k, s, g, c, count_other)
