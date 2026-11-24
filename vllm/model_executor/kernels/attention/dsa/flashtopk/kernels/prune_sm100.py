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
# Modified by the FlashTopK authors (2026): the pipeline of scan_sm100.py (itself
# derived from FlashInfer's selective-logits candidate kernel, adapted from
# TensorRT-LLM's fp8_paged_mqa_logits.py) carrying FlashTopK's key pruning: 16-row tiles
# of the pruning operand, the score-upper-bound epilogue and a bit mask in place of
# scores.
"""SM100 FlashTopK key pruning in CuTe DSL (FP8 E4M3, 32 heads): the score upper
bound, the first of pruning's two kernels.

The structure of scan_sm100.py (its warp roles, per-warpgroup KV rings and TMEM
accumulators, overlapped TMEM subtiles) on 16-row tiles of the pruning operand (plan.py
plans it): per 128-key block one M=128, N=256 MMA gives every row's 16 pruning columns
-- 14 retained heads and the high and low FP8 rows of the negative aggregate -- in pair
order (rows 2p and 2p + 1 interleaved by column), so a thread's TMEM columns hold a
pair's values side by side for packed FP32 arithmetic. The bound, in this order of
operations,

    s_k * (sum_{c < 14} (x_c + |x_c|) (w_c / 2) - relu(x_14 w_14 + x_15 w_15))
        + allowance * (s_k ||k||),

against the row's raw gate (and window off the interior) sets, per 8-row half and key,
whether any of the half's rows keeps it: `mask[2 * tile + half][block][4]`, one bit per
key, every word of the tile's KV range written. refine_sm100.py scores those keys in
full.

Warp roles (384 threads): warps 0-7 two math warpgroups (group w the blocks 2i + w),
warps 8-9 TMA (warp 8 + w group w's KV, scales and key norms; warp 8 also Q and the
weights once), warp 10 UMMA, warp 11 idle.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .common import (
    BLOCK_KV,
    HEAD_DIM,
    PRUNE_BLOCK_Q,
    PRUNE_COLS,
    abs_f32,
    fmax,
    loose_raw_gate,
    store_global_b32,
)
from .sm100_utils import (
    CLUSTER_SHAPE_MN,
    MATH_BARRIER_ID,
    NUM_EPI_SUBTILES,
    NUM_MATH_THREADS,
    NUM_MATH_WG,
    RND_RN,
    TMEM_ALLOC_THREADS,
    UmmaKernel,
    cta_kv_range,
    make_kv_q_operands,
    make_q_kv_umma_pipelines,
    make_tmem_allocator,
    make_tmem_load_atom,
    named_barrier_sync_unaligned,
    partition_tmem_readout,
    row_window,
)

NUM_KV_STAGES = 3
PAIR_FLOATS = 2 * PRUNE_COLS + 4  # the pair's weights by column, allowances, gates
THREADS_PER_CTA = 384


class PruneSm100(UmmaKernel):
    def __init__(self):
        self.N = PRUNE_BLOCK_Q * PRUNE_COLS
        self.math_wg_registers = 216
        self.specialized_wg_registers = 40

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits)
        # [pair_rows * 16, 128, 1] uint8 (FP8 E4M3 bits): the pruning operand in pair
        # order
        q_pairs: cute.Tensor,
        w_prune: cute.Tensor,  # [num_q, 16] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32
        knsc: cute.Tensor,  # [>= seq_len_kv] f32
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        allowance: cute.Tensor,  # [num_q] f32
        mask: cute.Tensor,  # [2 * tiles * mask_blocks * 4] i32
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        stream: cuda.CUstream,
    ):
        (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            b_dtype,
            b_smem_layout,
        ) = make_kv_q_operands(self, kv, q_pairs, seq_len_kv, NUM_KV_STAGES)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        w_layout = cute.make_layout((PRUNE_BLOCK_Q, PRUNE_COLS), stride=(PRUNE_COLS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, w_prune, w_layout, (PRUNE_BLOCK_Q, PRUNE_COLS)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_load_op, knsc, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        self.num_q_tma_bytes = (
            cute.size_in_bytes(b_dtype, b_smem_layout) + PRUNE_BLOCK_Q * PRUNE_COLS * 4
        )
        # One byte per FP8 element, then the block's scales and key norms.
        self.num_kv_tma_bytes = BLOCK_KV * HEAD_DIM + 2 * BLOCK_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, NUM_KV_STAGES * 4]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding_buf: Int32

        num_tiles = (num_q + PRUNE_BLOCK_Q - 1) // PRUNE_BLOCK_Q
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            tma_atom_n,
            tma_tensor_n,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            allowance,
            mask,
            num_q,
            seq_len_kv,
            mask_blocks,
            num_splits,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            w_layout,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
            block=[THREADS_PER_CTA, 1, 1],
            cluster=(*CLUSTER_SHAPE_MN, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        mN: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mAllowance: cute.Tensor,
        mMask: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        warpgroup_idx = warp_idx // 4
        tile = gdimx - 1 - bidx  # longest windows first
        query_start = tile * PRUNE_BLOCK_Q

        # The CTA's KV range starts on a 128-key boundary: mask blocks are global.
        split_lo, split_hi, range_lo, range_hi, cover_lo, cover_hi = cta_kv_range(
            mRowStart,
            mRowEnd,
            query_start,
            PRUNE_BLOCK_Q,
            num_q,
            seq_len_kv,
            bidy,
            num_splits,
            True,
        )
        has_work = range_hi > range_lo
        kv_start = range_lo // BLOCK_KV * BLOCK_KV
        num_kv = Int32(0)
        if has_work:
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
        num_pairs = (num_kv + NUM_MATH_WG - 1) // NUM_MATH_WG

        if (warp_idx == 8) | (warp_idx == 9):
            for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s, tma_atom_n):
                cpasync.prefetch_descriptor(atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        (
            q_pipeline,
            kv_pipes,
            kv_pipeline_math,
            umma_pipes,
            umma_pipeline_math,
            math_ring,
        ) = make_q_kv_umma_pipelines(
            storage,
            NUM_KV_STAGES,
            self.num_q_tma_bytes,
            self.num_kv_tma_bytes,
            cluster_layout_vmnk,
            tidx,
            warpgroup_idx,
        )
        tmem = make_tmem_allocator(storage.tmem_holding_buf)
        pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE_MN, is_relaxed=True)

        sKV = [
            smem.allocate_tensor(
                element_type=self.a_dtype,
                layout=a_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            for _ in range(2)
        ]
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (PRUNE_BLOCK_Q, PRUNE_COLS, 1),
                stride=(PRUNE_COLS, 1, PRUNE_BLOCK_Q * PRUNE_COLS),
            ),
            byte_alignment=128,
        )
        # Per group: [key, stage] scales, then key norms.
        vec_layout = cute.make_layout(
            (BLOCK_KV, NUM_KV_STAGES, 2, 2),
            stride=(
                1,
                BLOCK_KV,
                BLOCK_KV * NUM_KV_STAGES,
                2 * BLOCK_KV * NUM_KV_STAGES,
            ),
        )
        sVec = smem.allocate_tensor(
            element_type=Float32, layout=vec_layout, byte_alignment=128
        )
        sPair = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (PRUNE_BLOCK_Q // 2, PAIR_FLOATS), stride=(PAIR_FLOATS, 1)
            ),
            byte_alignment=16,
        )
        sWin = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((PRUNE_BLOCK_Q, 2), stride=(2, 1)),
            byte_alignment=16,
        )

        thr_mma = tiled_mma.get_slice(0)
        gA = cute.local_tile(
            cute.domain_offset((kv_start, 0, 0), mA_mkl),
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        tCgA = thr_mma.partition_A(gA)
        tA = [
            cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sKV[g], 0, 3),
                cute.group_modes(tCgA, 0, 3),
            )
            for g in range(2)
        ]
        # [group][scales, key norms] -> (smem, gmem) partitions.
        gVec = [
            cute.local_tile(cute.domain_offset((kv_start,), m), (BLOCK_KV,), (None,))
            for m in (mS, mN)
        ]
        tVec = [
            [
                cpasync.tma_partition(
                    atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sVec[(None, None, which, g)], 0, 1),
                    gVec[which],
                )
                for which, atom in enumerate((tma_atom_s, tma_atom_n))
            ]
            for g in range(2)
        ]
        gB = cute.local_tile(
            cute.domain_offset((tile * self.N, 0, 0), mB_nkl),
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
            (PRUNE_BLOCK_Q, PRUNE_COLS),
            (None, None),
        )
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )

        tCrA = [tiled_mma.make_fragment_A(sKV[g]) for g in range(2)]
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        epi_sub_mn = (BLOCK_KV, self.N // NUM_EPI_SUBTILES)
        copy_atom_t2r = make_tmem_load_atom(self.mma_tiler)

        pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MN)

        if (warp_idx == 8) | (warp_idx == 9):
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            group = warp_idx - 8
            kv_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_KV_STAGES
            )
            if (group == 0) & has_work:
                q_prod = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, 1
                )
                q_pipeline.producer_acquire(q_prod)
                q_bar = q_pipeline.producer_get_barrier(q_prod)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, 0, 0)],
                    tBsB[(None, 0)],
                    tma_bar_ptr=q_bar,
                )
                cute.copy(
                    tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar
                )
            for g in cutlass.range_constexpr(2):
                if group == g:
                    tAsA, tAgA = tA[g]
                    for pair in cutlass.range(num_pairs, unroll=1):
                        kv_pipes[g].producer_acquire(kv_prod)
                        bar = kv_pipes[g].producer_get_barrier(kv_prod)
                        stage = kv_prod.index
                        blk = pair * NUM_MATH_WG + g
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, blk, 0, 0)],
                            tAsA[(None, stage)],
                            tma_bar_ptr=bar,
                        )
                        for which, atom in enumerate((tma_atom_s, tma_atom_n)):
                            tVs, tVg = tVec[g][which]
                            cute.copy(
                                atom,
                                tVg[(None, blk)],
                                tVs[(None, stage)],
                                tma_bar_ptr=bar,
                            )
                        kv_prod.advance()
        elif warp_idx == 10:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            named_barrier_sync_unaligned(TMEM_ALLOC_THREADS)
            tmem_ptr = tmem.retrieve_ptr(Float32)
            acc_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[
                (None, None, None, 0)
            ]
            acc_1 = cute.make_tensor(
                tmem_ptr + self.num_tmem_alloc_cols, tCtAcc_fake.layout
            )[(None, None, None, 0)]
            q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            # One named state per pipeline: states advanced in place stay loop-carried.
            kv_cons_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_KV_STAGES
            )
            kv_cons_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_KV_STAGES
            )
            umma_prod_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            umma_prod_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            num_k = cute.size(tCrA[0].shape[2])
            if has_work:
                q_pipeline.consumer_wait(q_cons)
            for _pair in cutlass.range(num_pairs, unroll=1):
                ready = kv_pipes[0].consumer_try_wait(kv_cons_0)
                empty = umma_pipes[0].producer_try_acquire(umma_prod_0)
                kv_pipes[0].consumer_wait(kv_cons_0, ready)
                umma_pipes[0].producer_acquire(umma_prod_0, empty)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(num_k):
                    cute.gemm(
                        tiled_mma,
                        acc_0,
                        tCrA[0][None, None, k, kv_cons_0.index],
                        tCrB[None, None, k, 0],
                        acc_0,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                kv_cons_0.advance()
                umma_pipes[0].producer_commit(umma_prod_0)
                umma_prod_0.advance()

                ready = kv_pipes[1].consumer_try_wait(kv_cons_1)
                empty = umma_pipes[1].producer_try_acquire(umma_prod_1)
                kv_pipes[1].consumer_wait(kv_cons_1, ready)
                umma_pipes[1].producer_acquire(umma_prod_1, empty)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(num_k):
                    cute.gemm(
                        tiled_mma,
                        acc_1,
                        tCrA[1][None, None, k, kv_cons_1.index],
                        tCrB[None, None, k, 0],
                        acc_1,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                kv_cons_1.advance()
                umma_pipes[1].producer_commit(umma_prod_1)
                umma_prod_1.advance()
        elif warp_idx < 8:
            self._run_math_warp(
                has_work,
                num_pairs,
                kv_start,
                num_kv,
                cover_lo,
                cover_hi,
                split_lo,
                split_hi,
                query_start,
                tile,
                num_q,
                seq_len_kv,
                mask_blocks,
                copy_atom_t2r,
                epi_sub_mn,
                kv_pipeline_math,
                q_pipeline,
                sVec,
                sW,
                sPair,
                sWin,
                tCtAcc_fake,
                tidx,
                tmem,
                umma_pipeline_math,
                warpgroup_idx,
                mRowStart,
                mRowEnd,
                mOrigin,
                mInvDelta,
                mThBucket,
                mAllowance,
                mMask,
            )
        else:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self,
        has_work,
        num_pairs,
        kv_start,
        num_kv,
        cover_lo,
        cover_hi,
        split_lo,
        split_hi,
        query_start,
        tile,
        num_q,
        seq_len_kv,
        mask_blocks,
        copy_atom_t2r,
        epi_sub_mn,
        kv_pipeline_math,
        q_pipeline,
        sVec,
        sW,
        sPair,
        sWin,
        tCtAcc_fake,
        tidx,
        tmem,
        umma_pipeline_math,
        warpgroup_idx,
        mRowStart,
        mRowEnd,
        mOrigin,
        mInvDelta,
        mThBucket,
        mAllowance,
        mMask,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        lane = cute.arch.lane_idx()
        quad = cute.arch.make_warp_uniform(cute.arch.warp_idx() % 4)
        tmem.allocate(NUM_MATH_WG * self.num_tmem_alloc_cols)
        named_barrier_sync_unaligned(TMEM_ALLOC_THREADS)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc = cute.make_tensor(
            tmem_ptr + warpgroup_idx * self.num_tmem_alloc_cols, tCtAcc_fake.layout
        )
        epi_tidx = tidx % 128
        tiled_copy_t2r, thr_copy_t2r, tTR_tAcc = partition_tmem_readout(
            copy_atom_t2r, tCtAcc, epi_sub_mn, epi_tidx
        )
        tTR_tAcc = tTR_tAcc[(None, None, None, None, None, 0)]
        cC = cute.make_identity_tensor(epi_sub_mn)
        tTR_rAcc = cute.make_rmem_tensor_like(thr_copy_t2r.partition_D(cC), Float32)
        m_coord = thr_copy_t2r.partition_D(cC)[0][0]

        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        if has_work:
            q_pipeline.consumer_wait(q_cons)
            # The pair layout: retained heads' weights halved (the sum takes x + |x|),
            # the aggregate's scales as they are, then the rows' allowances and raw
            # gates; and the rows' windows.
            if tidx < PRUNE_BLOCK_Q * PRUNE_COLS:
                r = tidx // PRUNE_COLS
                c = tidx % PRUNE_COLS
                w = sW[(r, c, 0)]
                if c < PRUNE_COLS - 2:
                    w = w * Float32(0.5)
                sPair[(r // 2, 2 * c + r % 2)] = w
            if tidx < PRUNE_BLOCK_Q:
                row_q = query_start + tidx
                row = min(row_q, num_q - 1)
                sPair[(tidx // 2, 2 * PRUNE_COLS + tidx % 2)] = mAllowance[row]
                sPair[(tidx // 2, 2 * PRUNE_COLS + 2 + tidx % 2)] = loose_raw_gate(
                    mInvDelta[row], mOrigin[row], mThBucket[row]
                )
                s, e = row_window(
                    mRowStart, mRowEnd, row_q, num_q, seq_len_kv, split_lo, split_hi
                )
                sWin[(tidx, 0)] = s
                sWin[(tidx, 1)] = max(e - s, Int32(0))
            cute.arch.sync_warp()
            pipeline.NamedBarrier(
                barrier_id=MATH_BARRIER_ID, num_threads=NUM_MATH_THREADS
            ).arrive_and_wait()

        kv_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_KV_STAGES
        )
        umma_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        tile_base = (2 * tile * mask_blocks) * 4 + quad
        for pair in cutlass.range(num_pairs, unroll=1):
            j = pair * NUM_MATH_WG + warpgroup_idx
            umma_pipeline_math.consumer_wait(umma_cons)
            # TMEM loads are warp-aligned; a barrier wait may return with the warp
            # diverged.
            cute.arch.sync_warp()
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc)
            scale = sVec[(m_coord, kv_cons.index, 0, warpgroup_idx)]
            norm = sVec[(m_coord, kv_cons.index, 1, warpgroup_idx)]
            cute.arch.fence_view_async_tmem_load()
            blk_lo = kv_start + j * BLOCK_KV
            kv_idx = blk_lo + m_coord
            interior = (blk_lo >= cover_lo) & (blk_lo + BLOCK_KV <= cover_hi)
            surv = [cutlass.Boolean(False), cutlass.Boolean(False)]
            for e in cutlass.range_constexpr(NUM_EPI_SUBTILES):
                if e > 0:
                    cute.arch.fence_view_async_tmem_load()
                if e == NUM_EPI_SUBTILES - 1:
                    umma_pipeline_math.consumer_release(umma_cons)
                    umma_cons.advance()
                acc = tTR_rAcc.load()
                if e + 1 < NUM_EPI_SUBTILES:
                    cute.copy(
                        tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, e + 1)], tTR_rAcc
                    )
                for pp in cutlass.range_constexpr(2):
                    p = 2 * e + pp
                    keep = self._pair_bound(
                        acc, pp * 2 * PRUNE_COLS, p, sPair, scale, norm
                    )
                    for half_row in cutlass.range_constexpr(2):
                        r = 2 * p + half_row
                        in_win = interior | (
                            cutlass.Uint32(kv_idx - sWin[(r, 0)])
                            < cutlass.Uint32(sWin[(r, 1)])
                        )
                        surv[p // 4] = surv[p // 4] | (keep[half_row] & in_win)
            kv_pipeline_math.consumer_release(kv_cons)
            kv_cons.advance()
            m_lo = cute.arch.vote_ballot_sync(surv[0])
            m_hi = cute.arch.vote_ballot_sync(surv[1])
            if (lane == 0) & (j < num_kv):
                gblock = kv_start // BLOCK_KV + j
                base = (
                    mMask.iterator.toint() + cutlass.Int64(tile_base + gblock * 4) * 4
                )
                store_global_b32(base, m_lo)
                store_global_b32(base + cutlass.Int64(mask_blocks * 4) * 4, m_hi)
            cute.arch.sync_warp()

        if has_work:
            q_pipeline.consumer_release(q_cons)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _pair_bound(self, acc, c0, p, sPair, scale, norm):
        """Whether each row of pair p keeps the key: bound > gate (NaN keeps)."""

        def x(c):
            return (acc[c0 + 2 * c], acc[c0 + 2 * c + 1])

        def pos2(v):  # 2 relu(v), exactly
            return cute.arch.add_packed_f32x2(
                v, (abs_f32(v[0]), abs_f32(v[1])), rnd=RND_RN
            )

        def w(i):
            return sPair[(p, i)]

        acc0 = (Float32(0.0), Float32(0.0))
        acc1 = (Float32(0.0), Float32(0.0))
        for c in cutlass.range_constexpr(0, PRUNE_COLS - 2, 2):
            acc0 = cute.arch.fma_packed_f32x2(
                pos2(x(c)), (w(2 * c), w(2 * c + 1)), acc0, rnd=RND_RN
            )
            acc1 = cute.arch.fma_packed_f32x2(
                pos2(x(c + 1)), (w(2 * c + 2), w(2 * c + 3)), acc1, rnd=RND_RN
            )
        k = 2 * (PRUNE_COLS - 2)
        agg = cute.arch.fma_packed_f32x2(
            x(PRUNE_COLS - 1),
            (w(k + 2), w(k + 3)),
            cute.arch.mul_packed_f32x2(x(PRUNE_COLS - 2), (w(k), w(k + 1)), rnd=RND_RN),
            rnd=RND_RN,
        )
        reduced = cute.arch.add_packed_f32x2(
            cute.arch.add_packed_f32x2(acc0, acc1, rnd=RND_RN),
            (-fmax(agg[0], Float32(0.0)), -fmax(agg[1], Float32(0.0))),
            rnd=RND_RN,
        )
        a = 2 * PRUNE_COLS
        bound = cute.arch.fma_packed_f32x2(
            (w(a), w(a + 1)),
            (norm, norm),
            cute.arch.mul_packed_f32x2((scale, scale), reduced, rnd=RND_RN),
            rnd=RND_RN,
        )
        gates = (w(a + 2), w(a + 3))
        return [
            (bound[i] > gates[i]) | (bound[i] != bound[i]) | (gates[i] != gates[i])
            for i in (0, 1)
        ]
