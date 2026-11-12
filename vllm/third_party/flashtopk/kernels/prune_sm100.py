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
# Modified by the FlashTopK authors (2026): the pipeline of scan_sm100.py (itself derived from FlashInfer's
# selective-logits candidate kernel, adapted from TensorRT-LLM's fp8_paged_mqa_logits.py) carrying FlashTopK's key
# pruning: 16-row tiles of the pruning operand, the score-upper-bound epilogue and a bit mask in place of scores.
"""SM100 FlashTopK key pruning in CuTe DSL (FP8 or BF16, 32 heads): the score upper bound, the first of pruning's two
kernels.

The structure of scan_sm100.py (its warp roles, per-warpgroup KV rings and TMEM accumulators, overlapped TMEM
subtiles) on 16-row tiles of the pruning operand (plan.py plans it): per 128-key block one M=128, N=256 MMA gives
every row's 16 pruning columns -- 14 retained heads and the high and low FP8 rows of the negative aggregate --
in pair order (rows 2p and 2p + 1 interleaved by column), so a thread's TMEM columns hold a pair's values side by
side for packed FP32 arithmetic. The bound, in this order of operations,

    s_k * (sum_{c < 14} (x_c + |x_c|) (w_c / 2) - relu(x_14 w_14 + x_15 w_15)) + allowance * (s_k ||k||),

against the row's raw gate (and window off the interior) sets, per 8-row half and key, whether any of the half's
rows keeps it: mask[2 * tile + half][block][4], one bit per key, every word of the tile's KV range written.
refine_sm100.py scores those keys in full.

Warp roles (384 threads): warps 0-7 two math warpgroups (group w the blocks 2i + w), warps 8-9 TMA (Q and weights
once; group w's KV, scales and key norms), warp 10 UMMA.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .common import abs_f32, fmax, loose_raw_gate, select_i32, store_global_b32
from .scan_sm100 import _RND_RN, named_barrier_sync_unaligned

BLOCK_KV = 128
BLOCK_Q = 16
COLS = 16
HEAD_DIM = 128
NUM_KV_STAGES = 3
NUM_EPI_SUBTILES = 4  # 64 columns: two row pairs
PAIR_FLOATS = 2 * COLS + 4  # the pair's weights by column, allowances, gates
NUM_MATH_WG = 2


class PruneSm100:
    def __init__(self, *, bf16: bool = False):
        self.bf16 = bf16
        # A BF16 KV stage is twice an FP8 one: two stages per ring keep shared memory under 227 KB.
        self.num_kv_stages = 2 if bf16 else NUM_KV_STAGES
        self.N = BLOCK_Q * COLS
        self.num_math_threads = 256
        self.threads_per_cta = 384
        self.math_wg_registers = 216
        self.specialized_wg_registers = 40
        self.acc_dtype = Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (BLOCK_KV, self.N)


    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], k * (HEAD_DIM // k))
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, b_dtype, 1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tiled_mma.make_fragment_C(acc_shape))
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits) or bf16
        q_pairs: cute.Tensor,  # [pair_rows * 16, 128, 1] uint8 or bf16: the pruning operand in pair order
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
        if cutlass.const_expr(self.bf16):
            kv_typed, b = kv, q_pairs
        else:
            kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(q_pairs, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator, cute.make_layout((seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0))
        )
        tiled_mma = self._setup_mma(
            a.element_type, b.element_type, utils.LayoutEnum.from_tensor(a).mma_major_mode(),
            utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            a, self.a_smem_layout_staged, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        w_layout = cute.make_layout((BLOCK_Q, COLS), stride=(COLS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(tma_load_op, w_prune, w_layout, (BLOCK_Q, COLS))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_load_op, knsc, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        self.num_q_tma_bytes = cute.size_in_bytes(b.element_type, b_smem_layout) + BLOCK_Q * COLS * 4
        self.num_kv_tma_bytes = BLOCK_KV * HEAD_DIM * (a.element_type.width // 8) + 2 * BLOCK_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 4]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_holding_buf: Int32

        num_tiles = (num_q + BLOCK_Q - 1) // BLOCK_Q
        self.kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_w, tma_tensor_w, tma_atom_s,
            tma_tensor_s, tma_atom_n, tma_tensor_n, row_start, row_end, origin, inv_delta, th_bucket, allowance, mask,
            num_q, seq_len_kv, mask_blocks, num_splits, self.cluster_layout_vmnk, self.a_smem_layout_staged,
            self.b_smem_layout_staged, w_layout, self.epi_tile, SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
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
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        warpgroup_idx = warp_idx // 4
        tile = gdimx - 1 - bidx  # longest windows first
        query_start = tile * BLOCK_Q

        # The CTA's KV range (the rows' windows clipped to the split, from a 128-key boundary: mask blocks are
        # global) and the range every row covers.
        total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
        blocks_per_split = (total_blocks + num_splits - 1) // num_splits
        split_lo = bidy * blocks_per_split * BLOCK_KV
        split_hi = min((bidy + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
        range_lo = seq_len_kv
        range_hi = Int32(0)
        cover_lo = Int32(0)
        cover_hi = seq_len_kv
        for r in cutlass.range_constexpr(BLOCK_Q):
            row = query_start + r
            s = seq_len_kv
            e = Int32(0)
            if row < num_q:
                s = max(min(mRowStart[row], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row], seq_len_kv), split_hi)
            nonempty = e > s
            range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
            range_hi = max(range_hi, select_i32(nonempty, e, 0))
            cover_lo = max(cover_lo, select_i32(nonempty, s, seq_len_kv))
            cover_hi = min(cover_hi, select_i32(nonempty, e, 0))
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
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=1, producer_group=prod_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8), tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk, tidx=tidx, defer_sync=True,
        )
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        kv_pipes = [
            pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar.data_ptr() + g * self.num_kv_stages * 2, num_stages=self.num_kv_stages,
                producer_group=prod_group, consumer_group=kv_cons_group, tx_count=self.num_kv_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk, tidx=tidx, defer_sync=True,
            )
            for g in range(2)
        ]
        math_ring = min(warpgroup_idx, Int32(1))
        kv_pipeline_math = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr() + math_ring * self.num_kv_stages * 2, num_stages=self.num_kv_stages,
            producer_group=prod_group, consumer_group=kv_cons_group, tx_count=self.num_kv_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk, tidx=tidx, defer_sync=True,
        )
        umma_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        umma_pipes = [
            pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar.data_ptr() + g * 2, num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), consumer_group=umma_cons_group,
                cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
            )
            for g in range(2)
        ]
        umma_pipeline_math = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar.data_ptr() + math_ring * 2, num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread), consumer_group=umma_cons_group,
            cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf, barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=288),
            allocator_warp_id=0, is_two_cta=False,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sKV = [
            smem.allocate_tensor(
                element_type=self.a_dtype, layout=a_smem_layout_staged.outer, byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            for _ in range(2)
        ]
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype, layout=b_smem_layout_staged.outer, byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((BLOCK_Q, COLS, 1), stride=(COLS, 1, BLOCK_Q * COLS)),
            byte_alignment=128,
        )
        # Per group: [key, stage] scales, then key norms.
        vec_layout = cute.make_layout((BLOCK_KV, self.num_kv_stages, 2, 2), stride=(1, BLOCK_KV, BLOCK_KV * self.num_kv_stages, 2 * BLOCK_KV * self.num_kv_stages))
        sVec = smem.allocate_tensor(element_type=Float32, layout=vec_layout, byte_alignment=128)
        sPair = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((BLOCK_Q // 2, PAIR_FLOATS), stride=(PAIR_FLOATS, 1)),
            byte_alignment=16,
        )
        sWin = smem.allocate_tensor(element_type=Int32, layout=cute.make_layout((BLOCK_Q, 2), stride=(2, 1)), byte_alignment=16)

        thr_mma = tiled_mma.get_slice(0)
        gA = cute.local_tile(cute.domain_offset((kv_start, 0, 0), mA_mkl), cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        tCgA = thr_mma.partition_A(gA)
        tA = [
            cpasync.tma_partition(tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sKV[g], 0, 3), cute.group_modes(tCgA, 0, 3))
            for g in range(2)
        ]
        # [group][scales, key norms] -> (smem, gmem) partitions.
        gVec = [cute.local_tile(cute.domain_offset((kv_start,), m), (BLOCK_KV,), (None,)) for m in (mS, mN)]
        tVec = [
            [
                cpasync.tma_partition(
                    atom, 0, cute.make_layout(1), cute.group_modes(sVec[(None, None, which, g)], 0, 1), gVec[which]
                )
                for which, atom in enumerate((tma_atom_s, tma_atom_n))
            ]
            for g in range(2)
        ]
        gB = cute.local_tile(
            cute.domain_offset((tile * self.N, 0, 0), mB_nkl), cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3)
        )
        gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), (BLOCK_Q, COLS), (None, None))
        tWsW, tWgW = cpasync.tma_partition(tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW, 0, 2), cute.group_modes(gW, 0, 2))

        tCrA = [tiled_mma.make_fragment_A(sKV[g]) for g in range(2)]
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        epi_sub_mn = (epi_tile[0], self.N // NUM_EPI_SUBTILES)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, utils.LayoutEnum.ROW_MAJOR, self.acc_dtype, self.acc_dtype, (epi_sub_mn[0], 32), False
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if (warp_idx == 8) | (warp_idx == 9):
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            group = warp_idx - 8
            kv_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_kv_stages)
            if (group == 0) & has_work:
                q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                q_pipeline.producer_acquire(q_prod)
                q_bar = q_pipeline.producer_get_barrier(q_prod)
                cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
            for g in cutlass.range_constexpr(2):
                if group == g:
                    tAsA, tAgA = tA[g]
                    for pair in cutlass.range(num_pairs, unroll=1):
                        kv_pipes[g].producer_acquire(kv_prod)
                        bar = kv_pipes[g].producer_get_barrier(kv_prod)
                        stage = kv_prod.index
                        blk = pair * NUM_MATH_WG + g
                        cute.copy(tma_atom_a, tAgA[(None, blk, 0, 0)], tAsA[(None, stage)], tma_bar_ptr=bar)
                        for which, atom in enumerate((tma_atom_s, tma_atom_n)):
                            tVs, tVg = tVec[g][which]
                            cute.copy(atom, tVg[(None, blk)], tVs[(None, stage)], tma_bar_ptr=bar)
                        kv_prod.advance()
        elif warp_idx == 10:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            named_barrier_sync_unaligned(288)
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)[(None, None, None, 0)]
            acc_1 = cute.make_tensor(tmem_ptr + self.num_tmem_alloc_cols, tCtAcc_fake.layout)[(None, None, None, 0)]
            q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            # One named state per pipeline: states advanced in place stay loop-carried.
            kv_cons_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
            kv_cons_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
            umma_prod_0 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            umma_prod_1 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            num_k = cute.size(tCrA[0].shape[2])
            if has_work:
                q_pipeline.consumer_wait(q_cons)
            for pair in cutlass.range(num_pairs, unroll=1):
                ready = kv_pipes[0].consumer_try_wait(kv_cons_0)
                empty = umma_pipes[0].producer_try_acquire(umma_prod_0)
                kv_pipes[0].consumer_wait(kv_cons_0, ready)
                umma_pipes[0].producer_acquire(umma_prod_0, empty)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range_constexpr(num_k):
                    cute.gemm(tiled_mma, acc_0, tCrA[0][None, None, k, kv_cons_0.index], tCrB[None, None, k, 0], acc_0)
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
                    cute.gemm(tiled_mma, acc_1, tCrA[1][None, None, k, kv_cons_1.index], tCrB[None, None, k, 0], acc_1)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                kv_cons_1.advance()
                umma_pipes[1].producer_commit(umma_prod_1)
                umma_prod_1.advance()
        elif warp_idx < 8:
            self._run_math_warp(
                has_work, num_pairs, kv_start, num_kv, cover_lo, cover_hi, split_lo, split_hi, query_start, tile,
                num_q, seq_len_kv, mask_blocks, copy_atom_t2r, epi_sub_mn, kv_pipeline_math, q_pipeline, sVec, sW,
                sPair, sWin, tCtAcc_fake, tidx, tmem, umma_pipeline_math, warpgroup_idx, mRowStart, mRowEnd, mOrigin,
                mInvDelta, mThBucket, mAllowance, mMask,
            )
        else:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self, has_work, num_pairs, kv_start, num_kv, cover_lo, cover_hi, split_lo, split_hi, query_start, tile, num_q,
        seq_len_kv, mask_blocks, copy_atom_t2r, epi_sub_mn, kv_pipeline_math, q_pipeline, sVec, sW, sPair, sWin,
        tCtAcc_fake, tidx, tmem, umma_pipeline_math, warpgroup_idx, mRowStart, mRowEnd, mOrigin, mInvDelta, mThBucket,
        mAllowance, mMask,
    ):
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        lane = cute.arch.lane_idx()
        quad = cute.arch.make_warp_uniform(cute.arch.warp_idx() % 4)
        tmem.allocate(2 * self.num_tmem_alloc_cols)
        named_barrier_sync_unaligned(288)
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr + warpgroup_idx * self.num_tmem_alloc_cols, tCtAcc_fake.layout)
        epi_tidx = tidx % 128
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, None)], epi_sub_mn)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)[(None, None, None, None, None, 0)]
        cC = cute.make_identity_tensor(epi_sub_mn)
        tTR_rAcc = cute.make_rmem_tensor_like(thr_copy_t2r.partition_D(cC), self.acc_dtype)
        m_coord = thr_copy_t2r.partition_D(cC)[0][0]

        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        if has_work:
            q_pipeline.consumer_wait(q_cons)
            # The pair layout: retained heads' weights halved (the sum takes x + |x|), the aggregate's scales as
            # they are, then the rows' allowances and raw gates; and the rows' windows.
            if tidx < BLOCK_Q * COLS:
                r = tidx // COLS
                c = tidx % COLS
                w = sW[(r, c, 0)]
                if c < COLS - 2:
                    w = w * Float32(0.5)
                sPair[(r // 2, 2 * c + r % 2)] = w
            if tidx < BLOCK_Q:
                row_q = query_start + tidx
                row = min(row_q, num_q - 1)
                sPair[(tidx // 2, 2 * COLS + tidx % 2)] = mAllowance[row]
                sPair[(tidx // 2, 2 * COLS + 2 + tidx % 2)] = loose_raw_gate(mInvDelta[row], mOrigin[row], mThBucket[row])
                s = seq_len_kv
                e = Int32(0)
                if row_q < num_q:
                    s = max(min(mRowStart[row_q], seq_len_kv), split_lo)
                    e = min(min(mRowEnd[row_q], seq_len_kv), split_hi)
                sWin[(tidx, 0)] = s
                sWin[(tidx, 1)] = max(e - s, Int32(0))
            cute.arch.sync_warp()
            pipeline.NamedBarrier(barrier_id=2, num_threads=self.num_math_threads).arrive_and_wait()

        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_kv_stages)
        umma_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        tile_base = (2 * tile * mask_blocks) * 4 + quad
        for pair in cutlass.range(num_pairs, unroll=1):
            j = pair * NUM_MATH_WG + warpgroup_idx
            umma_pipeline_math.consumer_wait(umma_cons)
            # TMEM loads are warp-aligned; a barrier wait may return with the warp diverged.
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
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, e + 1)], tTR_rAcc)
                for pp in cutlass.range_constexpr(2):
                    p = 2 * e + pp
                    keep = self._pair_bound(acc, pp * 2 * COLS, p, sPair, scale, norm)
                    for half_row in cutlass.range_constexpr(2):
                        r = 2 * p + half_row
                        in_win = interior | (cutlass.Uint32(kv_idx - sWin[(r, 0)]) < cutlass.Uint32(sWin[(r, 1)]))
                        surv[p // 4] = surv[p // 4] | (keep[half_row] & in_win)
            kv_pipeline_math.consumer_release(kv_cons)
            kv_cons.advance()
            m_lo = cute.arch.vote_ballot_sync(surv[0])
            m_hi = cute.arch.vote_ballot_sync(surv[1])
            if (lane == 0) & (j < num_kv):
                gblock = kv_start // BLOCK_KV + j
                base = mMask.iterator.toint() + cutlass.Int64(tile_base + gblock * 4) * 4
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
            return cute.arch.add_packed_f32x2(v, (abs_f32(v[0]), abs_f32(v[1])), rnd=_RND_RN)

        def w(i):
            return sPair[(p, i)]

        acc0 = (Float32(0.0), Float32(0.0))
        acc1 = (Float32(0.0), Float32(0.0))
        for c in cutlass.range_constexpr(0, COLS - 2, 2):
            acc0 = cute.arch.fma_packed_f32x2(pos2(x(c)), (w(2 * c), w(2 * c + 1)), acc0, rnd=_RND_RN)
            acc1 = cute.arch.fma_packed_f32x2(pos2(x(c + 1)), (w(2 * c + 2), w(2 * c + 3)), acc1, rnd=_RND_RN)
        k = 2 * (COLS - 2)
        agg = cute.arch.fma_packed_f32x2(
            x(COLS - 1), (w(k + 2), w(k + 3)), cute.arch.mul_packed_f32x2(x(COLS - 2), (w(k), w(k + 1)), rnd=_RND_RN),
            rnd=_RND_RN,
        )
        reduced = cute.arch.add_packed_f32x2(
            cute.arch.add_packed_f32x2(acc0, acc1, rnd=_RND_RN),
            (-fmax(agg[0], Float32(0.0)), -fmax(agg[1], Float32(0.0))),
            rnd=_RND_RN,
        )
        a = 2 * COLS
        bound = cute.arch.fma_packed_f32x2(
            (w(a), w(a + 1)), (norm, norm), cute.arch.mul_packed_f32x2((scale, scale), reduced, rnd=_RND_RN), rnd=_RND_RN
        )
        gates = (w(a + 2), w(a + 3))
        return [(bound[i] > gates[i]) | (bound[i] != bound[i]) | (gates[i] != gates[i]) for i in (0, 1)]
