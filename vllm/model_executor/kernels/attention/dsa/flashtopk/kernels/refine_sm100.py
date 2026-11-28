# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the FlashTopK authors (2026): the epilogue and TMEM read-out of
# scan_sm100.py (derived from FlashInfer's selective-logits candidate kernel) behind a
# gather warpgroup that compacts the pruning pass's surviving keys into tiles by
# cp.async, one MMA per compact tile.
"""SM100 FlashTopK refine in CuTe DSL (FP8 E4M3, 32 heads), the second kernel of key
pruning: one CTA per 8-row tile and KV split scores in full the keys prune_sm100.py's
mask keeps for the tile, and leaves select the candidates the scan would.

  * the gather warpgroup counts the survivors of each 128-key block of the CTA's range
    (each thread a run of consecutive blocks) and turns the counts into an exclusive
    prefix sum over the blocks in shared memory (a warp scan of the run totals, then
    across the four warps), with the survivor total at its end; gather warp w then
    fills compact tile w with survivors 128t .. 128t + 127 of every tile t = w (mod 4):
    each lane finds its keys' blocks by binary search and their bits with fns, and
    copies the K rows (cp.async, 16 bytes at a time, 128-byte swizzle, the layout the
    MMA reads) with each key's index and kv scale;
  * the UMMA warp gives each compact tile one M=128, N=256 MMA with every head of the
    tile's rows, into one of two TMEM stages;
  * math warpgroup w takes the stage w tiles: the scan's arithmetic (scan_sm100.py's
    head sums), the row's raw gate and window, and the candidate buffer (every row's
    slots reserved in one round trip).

The gate stays at the seed. Warp roles (512 threads): 0-7 math, 8-11 gather, 12 Q and
weights loader, 13 UMMA, 14-15 idle.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as blackwell
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05

from .common import (
    BLOCK_KV,
    GATHER_THREADS,
    NUM_HEADS,
    REFINE_BLOCK_Q,
    atomic_add_global_i32,
    fp32_to_i32_bits,
    loose_raw_gate,
    refine_gather,
    select_i32,
    store_global_b32,
)
from .common import (
    REFINE_MAX_BLOCKS as MAX_BLOCKS,
)
from .common import (
    REFINE_STAGES as NUM_TILES,
)
from .sm100_utils import (
    CLUSTER_SHAPE_MN,
    MATH_BARRIER_ID,
    NUM_EPI_SUBTILES,
    NUM_MATH_THREADS,
    NUM_MATH_WARPS,
    NUM_MATH_WG,
    RND_RN,
    TMEM_ALLOC_THREADS,
    UmmaKernel,
    cta_kv_range,
    head_quad_activations,
    head_weight,
    make_tmem_allocator,
    make_tmem_load_atom,
    make_weight_cache,
    named_barrier_sync_unaligned,
    partition_tmem_readout,
    replicate_weights,
    row_window,
    setup_umma,
)

GATHER_BARRIER_ID = 3
THREADS_PER_CTA = 512


class RefineSm100(UmmaKernel):
    def __init__(self):
        self.N = REFINE_BLOCK_Q * NUM_HEADS
        self.math_wg_registers = 216
        self.other_registers = 40

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
        b = cute.recast_tensor(q, cutlass.Float8E4M3FN)
        major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = setup_umma(
            self, b.element_type, b.element_type, major, major, NUM_TILES
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            blackwell.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE_MN, tiled_mma.thr_id),
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        w_layout = cute.make_layout((REFINE_BLOCK_Q, NUM_HEADS), stride=(NUM_HEADS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            weights,
            w_layout,
            (REFINE_BLOCK_Q, NUM_HEADS),
        )
        self.num_q_tma_bytes = (
            cute.size_in_bytes(b.element_type, b_smem_layout) + self.N * 4
        )

        @cute.struct
        class SharedStorage:
            # q, count, full_tile[4], empty_tile[4], full_tmem[2], empty_tmem[2]
            mbar: cute.struct.MemRange[cutlass.Int64, 2 + 2 * NUM_TILES + 4]
            tmem_holding_buf: Int32

        num_tiles = (num_q + REFINE_BLOCK_Q - 1) // REFINE_BLOCK_Q
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
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
            block=[THREADS_PER_CTA, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
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
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        tile = gdimx - 1 - bidx
        query_start = tile * REFINE_BLOCK_Q

        split_lo, split_hi, range_lo, range_hi, _, _ = cta_kv_range(
            mRowStart,
            mRowEnd,
            query_start,
            REFINE_BLOCK_Q,
            num_q,
            seq_len_kv,
            bidy,
            num_splits,
            False,
        )
        has_work = range_hi > range_lo
        kv_start = range_lo // BLOCK_KV * BLOCK_KV
        num_kv = Int32(0)
        if has_work:
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
        # A CTA whose rows have no keys in its split leaves before any setup (adaptive
        # pruning launches the path it did not take with empty windows).
        if has_work:
            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            mbar = storage.mbar.data_ptr()
            q_bar, count_bar = mbar, mbar + 1
            full_tile, empty_tile = mbar + 2, mbar + 2 + NUM_TILES
            full_tmem, empty_tmem = mbar + 2 + 2 * NUM_TILES, mbar + 4 + 2 * NUM_TILES
            sQ = smem.allocate_tensor(
                element_type=self.a_dtype,
                layout=b_smem_layout_staged.outer,
                byte_alignment=1024,
                swizzle=b_smem_layout_staged.inner,
            )
            sTile = smem.allocate_tensor(
                element_type=self.a_dtype,
                layout=a_smem_layout_staged.outer,
                byte_alignment=1024,
                swizzle=a_smem_layout_staged.inner,
            )
            sMetaIdx = smem.allocate_tensor(
                Int32, cute.make_layout((NUM_TILES, BLOCK_KV), stride=(BLOCK_KV, 1)), 16
            )
            sMetaScale = smem.allocate_tensor(
                Float32,
                cute.make_layout((NUM_TILES, BLOCK_KV), stride=(BLOCK_KV, 1)),
                16,
            )
            sPrefix = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_BLOCKS + 1,)), 16
            )
            sW = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (REFINE_BLOCK_Q, NUM_HEADS, 1),
                    stride=(NUM_HEADS, 1, REFINE_BLOCK_Q * NUM_HEADS),
                ),
                128,
            )
            sWarpWeights = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (NUM_MATH_WARPS, REFINE_BLOCK_Q, NUM_HEADS),
                    stride=(self.N + 4, NUM_HEADS, 1),
                ),
                16,
            )
            sGate = smem.allocate_tensor(
                Float32, cute.make_layout((REFINE_BLOCK_Q,)), 16
            )
            sWin = smem.allocate_tensor(
                Int32, cute.make_layout((REFINE_BLOCK_Q, 2), stride=(2, 1)), 16
            )
            # The survivor count, then the gather warps' totals.
            sCounts = smem.allocate_tensor(
                Int32, cute.make_layout((1 + GATHER_THREADS // 32,)), 16
            )

            if tidx == 0:
                cute.arch.mbarrier_init(q_bar, 1)
                cute.arch.mbarrier_init(count_bar, 1)
                for i in cutlass.range_constexpr(NUM_TILES):
                    cute.arch.mbarrier_init(
                        full_tile + i, 32
                    )  # all 32 lanes of its gather warp, once the copies have landed
                    cute.arch.mbarrier_init(
                        empty_tile + i, 1 + 4 * 32
                    )  # the MMA's commit and the epilogue warpgroup's 128 threads
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(full_tmem + i, 1)
                    cute.arch.mbarrier_init(empty_tmem + i, 4 * 32)
                cute.arch.mbarrier_init_fence()
            if tidx < REFINE_BLOCK_Q:
                row_q = query_start + tidx
                row = min(row_q, num_q - 1)
                sGate[tidx] = loose_raw_gate(
                    mInvDelta[row], mOrigin[row], mThBucket[row]
                )
                s, e = row_window(
                    mRowStart, mRowEnd, row_q, num_q, seq_len_kv, split_lo, split_hi
                )
                sWin[(tidx, 0)] = s
                sWin[(tidx, 1)] = max(e - s, Int32(0))
            cute.arch.sync_threads()

            tmem = make_tmem_allocator(storage.tmem_holding_buf)
            thr_mma = tiled_mma.get_slice(0)
            tCrA = tiled_mma.make_fragment_A(sTile)
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))

            if warp_idx == 12:
                cute.arch.setmaxregister_decrease(self.other_registers)
                if has_work:
                    gB = cute.local_tile(
                        cute.domain_offset((query_start * NUM_HEADS, 0, 0), mB_nkl),
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_b,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(thr_mma.partition_B(gB), 0, 3),
                    )
                    gW = cute.local_tile(
                        cute.domain_offset((query_start, 0), mW),
                        (REFINE_BLOCK_Q, NUM_HEADS),
                        (None, None),
                    )
                    tWsW, tWgW = cpasync.tma_partition(
                        tma_atom_w,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sW, 0, 2),
                        cute.group_modes(gW, 0, 2),
                    )
                    if lane == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            q_bar, self.num_q_tma_bytes
                        )
                    cute.arch.sync_warp()
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, 0, 0, 0)],
                        tBsB[(None, 0)],
                        tma_bar_ptr=q_bar,
                    )
                    cute.copy(
                        tma_atom_w,
                        tWgW[(None, 0, 0)],
                        tWsW[(None, 0)],
                        tma_bar_ptr=q_bar,
                    )
            elif warp_idx == 13:
                cute.arch.setmaxregister_decrease(self.other_registers)
                named_barrier_sync_unaligned(TMEM_ALLOC_THREADS)
                tmem_ptr = tmem.retrieve_ptr(Float32)
                if has_work:
                    cute.arch.mbarrier_wait(count_bar, 0)
                    tiles = (sCounts[0] + BLOCK_KV - 1) // BLOCK_KV
                    cute.arch.mbarrier_wait(q_bar, 0)
                    for t in cutlass.range(tiles, unroll=1):
                        s = t % NUM_TILES
                        p = t % 2
                        cute.arch.mbarrier_wait(full_tile + s, (t // NUM_TILES) % 2)
                        cute.arch.mbarrier_wait(empty_tmem + p, (t // 2) % 2 ^ 1)
                        acc = cute.make_tensor(
                            tmem_ptr + p * self.num_tmem_alloc_cols, tCtAcc_fake.layout
                        )[(None, None, None, 0)]
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tCrA[None, None, k, s],
                                tCrB[None, None, k, 0],
                                acc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(full_tmem + p)
                            tcgen05.commit(empty_tile + s)
            elif warp_idx >= 8:
                cute.arch.setmaxregister_decrease(self.other_registers)
                if (warp_idx < 12) & has_work:
                    refine_gather(
                        tidx - NUM_MATH_THREADS,
                        warp_idx - 8,
                        lane,
                        num_kv,
                        kv_start,
                        tile,
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
                        BLOCK_KV,
                        GATHER_BARRIER_ID,
                    )
            else:
                self._run_math(
                    has_work,
                    tidx,
                    lane,
                    warp_idx,
                    tmem,
                    tCtAcc_fake,
                    sW,
                    sWarpWeights,
                    sMetaIdx,
                    sMetaScale,
                    sGate,
                    sWin,
                    sCounts,
                    q_bar,
                    count_bar,
                    full_tmem,
                    empty_tmem,
                    empty_tile,
                    query_start,
                    mCandVal,
                    mCandIdx,
                    mCandCnt,
                )

    @cute.jit
    def _run_math(
        self,
        has_work,
        tidx,
        lane,
        warp_idx,
        tmem,
        tCtAcc_fake,
        sW,
        sWarpWeights,
        sMetaIdx,
        sMetaScale,
        sGate,
        sWin,
        sCounts,
        q_bar,
        count_bar,
        full_tmem,
        empty_tmem,
        empty_tile,
        query_start,
        mCandVal,
        mCandIdx,
        mCandCnt,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        tmem.allocate(NUM_MATH_WG * self.num_tmem_alloc_cols)
        named_barrier_sync_unaligned(TMEM_ALLOC_THREADS)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        wg = warp_idx // 4
        key = tidx % 128
        epi_sub_mn = (BLOCK_KV, self.N // NUM_EPI_SUBTILES)
        copy_atom_t2r = make_tmem_load_atom(self.mma_tiler)
        tCtAcc = cute.make_tensor(
            tmem_ptr + wg * self.num_tmem_alloc_cols, tCtAcc_fake.layout
        )
        tiled_copy_t2r, thr_copy_t2r, tTR_tAcc = partition_tmem_readout(
            copy_atom_t2r, tCtAcc, epi_sub_mn, key
        )
        tTR_tAcc = tTR_tAcc[(None, None, None, None, None, 0)]
        tTR_cC = thr_copy_t2r.partition_D(cute.make_identity_tensor(epi_sub_mn))
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, Float32)
        slot = tTR_cC[0][0]  # the compact tile's row (TMEM lane) this thread reads
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        cap = mCandVal.layout.shape[1]

        w_cache = make_weight_cache(REFINE_BLOCK_Q)
        if has_work:
            cute.arch.mbarrier_wait(count_bar, 0)
            n = sCounts[0]
            tiles = (n + BLOCK_KV - 1) // BLOCK_KV
            cute.arch.mbarrier_wait(q_bar, 0)
            replicate_weights(sW, sWarpWeights, w_cache, warp_idx, lane, REFINE_BLOCK_Q)

            for t in cutlass.range(wg, tiles, 2, unroll=1):
                s = t % NUM_TILES
                cute.arch.mbarrier_wait(full_tmem + wg, (t // 2) % 2)
                # TMEM loads are warp-aligned; a barrier wait may return with the warp
                # diverged.
                cute.arch.sync_warp()
                valid = t * BLOCK_KV + slot < n
                kv_idx = select_i32(valid, sMetaIdx[(s, slot)], 0)
                scale = sMetaScale[(s, slot)]
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc)
                cute.arch.fence_view_async_tmem_load()
                red = []
                for e in cutlass.range_constexpr(NUM_EPI_SUBTILES):
                    if e > 0:
                        cute.arch.fence_view_async_tmem_load()
                    acc = tTR_rAcc.load()
                    if e + 1 < NUM_EPI_SUBTILES:
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, 0, e + 1)],
                            tTR_rAcc,
                        )
                    else:
                        # Every subtile is in registers: hand the stage back to the
                        # tensor core.
                        cute.arch.mbarrier_arrive(empty_tmem + wg)
                    red.extend(
                        self._fast_sums(acc, w_cache, sWarpWeights, warp_idx, 2 * e)
                    )
                masks = []
                for r in cutlass.range_constexpr(0, REFINE_BLOCK_Q, 2):
                    raw = cute.arch.mul_packed_f32x2(
                        (scale, scale), (red[r], red[r + 1]), rnd=RND_RN
                    )
                    for i in cutlass.range_constexpr(2):
                        row = r + i
                        take = (
                            valid
                            & (raw[i] > sGate[row])
                            & (
                                cutlass.Uint32(kv_idx - sWin[(row, 0)])
                                < cutlass.Uint32(sWin[(row, 1)])
                            )
                        )
                        masks.append(
                            (cutlass.Uint32(cute.arch.vote_ballot_sync(take)), raw[i])
                        )
                # Every row's slots in one round trip: lane r reserves row r's.
                lane_rows = cutlass.Uint32(0)
                for r in cutlass.range_constexpr(REFINE_BLOCK_Q):
                    lane_rows = cutlass.Uint32(
                        select_i32(lane == r, Int32(masks[r][0]), Int32(lane_rows))
                    )
                lane_base = Int32(0)
                if lane_rows != cutlass.Uint32(0):
                    lane_base = atomic_add_global_i32(
                        mCandCnt.iterator.toint()
                        + cutlass.Int64(query_start + lane) * 4,
                        Int32(cute.arch.popc(lane_rows)),
                    )
                for r in cutlass.range_constexpr(REFINE_BLOCK_Q):
                    m, raw_r = masks[r]
                    if m != cutlass.Uint32(0):
                        base = cute.arch.shuffle_sync(lane_base, r)
                        if (m >> cutlass.Uint32(lane)) & cutlass.Uint32(
                            1
                        ) != cutlass.Uint32(0):
                            pos = base + Int32(cute.arch.popc(m & lane_mask))
                            if pos < cap:
                                out_row = cutlass.Int64(query_start + r)
                                store_global_b32(
                                    mCandVal.iterator.toint()
                                    + (out_row * mCandVal.layout.stride[0] + pos) * 4,
                                    fp32_to_i32_bits(-raw_r),
                                )
                                store_global_b32(
                                    mCandIdx.iterator.toint()
                                    + (out_row * mCandIdx.layout.stride[0] + pos) * 4,
                                    kv_idx,
                                )
                # The tile's metadata is used: it may be refilled.
                cute.arch.mbarrier_arrive(empty_tile + s)
        cute.arch.barrier(
            barrier_id=MATH_BARRIER_ID, number_of_threads=NUM_MATH_THREADS
        )
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _fast_sums(self, acc, w_cache, sWarpWeights, warp, t0):
        """The subtile's two rows' head sums, in scan_sm100.py's order."""
        out = []
        for r in cutlass.range_constexpr(2):
            t = t0 + r
            s0 = (Float32(0.0), Float32(0.0))
            s1 = (Float32(0.0), Float32(0.0))
            for h in cutlass.range_constexpr(0, NUM_HEADS, 4):
                a01, a23 = head_quad_activations(acc, r * NUM_HEADS + h, h)
                w = [
                    head_weight(w_cache, sWarpWeights, warp, t, h + i) for i in range(4)
                ]
                s0 = cute.arch.fma_packed_f32x2(a01, (w[0], w[1]), s0, rnd=RND_RN)
                s1 = cute.arch.fma_packed_f32x2(a23, (w[2], w[3]), s1, rnd=RND_RN)
            sx, sy = cute.arch.add_packed_f32x2(s0, s1)
            out.append(sx + sy)
        return out
