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
# Adapted from TensorRT-LLM (Apache 2.0): its CuTe DSL FP8 paged MQA logits kernel
# (fp8_paged_mqa_logits.py under tensorrt_llm/_torch/cute_dsl_kernels/). Original
# copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modified by the FlashTopK authors (2026): derived from FlashInfer's selective-logits
# candidate kernel (flashinfer/attn_scores/kernels/fp8_paged_mqa_logits.py,
# FP8MQASelectiveLogitsKernel with the Q8/H32/D128 candidate tiling) and specialized to
# the FlashTopK scan. Changes: one (query tile, KV split) work item per CTA over
# contiguous KV with per-row windows, in place of the persistent paged request schedule;
# the gate is FlashTopK's raw gate (score > gate, the loose_raw_gate of the row's bucket
# edge) and the payload FlashTopK's (-score, kv index) pair in separate value / index
# arrays; a row's hits for a pair of blocks reserve their slots at once, from a register
# cursor (one split) or the row's shared count (KV splits), so a row never has per-split
# capacity; a dense variant writes every in-window score; all other modes, dtypes and
# the paged path are removed.

"""SM100 FlashTopK scan in CuTe DSL: FP8 E4M3, 32 heads, head_dim 128, 8-row query
tiles.

A CTA scans one 8-row query tile over the union of its rows' KV windows, clipped to its
KV split, and appends every key whose score passes the row's raw gate to the row's
candidate buffer as (-score, kv index). Warp roles (512 threads):

  * warps 0-7, two math warpgroups: group w reduces the blocks 2i + w from its own TMEM
    accumulator, scores all 8 rows of a key per thread in a fixed order (heads 4j .. 4j
    + 3 in two packed FMA chains, their sum, then the kv scale) and writes them to its
    shared score plane;
  * warps 8-9, TMA: warp 8 + w loads group w's KV blocks and FP8 scales; warp 8 also
    loads Q and the weights once;
  * warp 10, UMMA: one M=128 x N=256 x K=128 MMA per block, both groups;
  * warp 11 idles;
  * warps 12-15, store: warp s owns rows s and s + 4; per pair of blocks each lane gates
    four consecutive keys of each block and the warp appends the hits in ballot order.

The dense variant scores the shared calibration sample and the rows recovery rescores:
every key of a row's window is written to cand_val[row, key] (no gate, no counts), in
the scan's bits.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as blackwell
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .common import (
    BLOCK_KV,
    HEAD_DIM,
    NEXT_N,
    NUM_HEADS,
    atomic_add_global_i32,
    fp32_to_i32_bits,
    loose_raw_gate,
    store_global_b32,
)
from .simt import bits_f32
from .sm100_utils import (
    CLUSTER_SHAPE_MN,
    NUM_EPI_SUBTILES,
    NUM_MATH_THREADS,
    NUM_MATH_WARPS,
    NUM_MATH_WG,
    RND_RN,
    TMEM_ALLOC_THREADS,
    UmmaKernel,
    clip_window,
    create_umma_pipeline,
    cta_kv_range,
    head_quad_activations,
    head_weight,
    make_group_pipelines,
    make_tmem_allocator,
    make_tmem_load_atom,
    make_weight_cache,
    named_barrier_sync_unaligned,
    partition_tmem_readout,
    replicate_weights,
    setup_umma,
)

NUM_KV_STAGES = 3
NUM_UMMA_STAGES = 1  # per group
NUM_Q_STAGES = 1
NUM_STORE_WARPS = 4
FIRST_STORE_WARP = 12
THREADS_PER_CTA = 512
W_STAGE_STRIDE = (NEXT_N * NUM_HEADS * 4 + 127) // 128 * 128 // 4


@dsl_user_op
def store_global_b32_if(predicate, address, value, *, loc=None, ip=None) -> None:
    """One predicated 32-bit store through a raw byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(predicate).ir_value(loc=loc, ip=ip),
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p st.global.b32 [$1], $2; }",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_v4_f32(tensor, row, kv_base, stage, *, loc=None, ip=None):
    """Four contiguous FP32 scores with one 16-byte shared-memory load."""
    elem_offset = cute.crd2idx((row, kv_base, stage), tensor.layout)
    smem_addr = (tensor.iterator + elem_offset).toint(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        bits_f32(
            Int32(llvm.extractvalue(Int32.mlir_type, result, [idx], loc=loc, ip=ip))
        )
        for idx in range(4)
    )


@cute.jit
def make_warp_uniform_i64(value) -> cutlass.Int64:
    bits = cutlass.Uint64(value)
    lo = cute.arch.make_warp_uniform(Int32(bits & 0xFFFFFFFF))
    hi = cute.arch.make_warp_uniform(Int32(bits >> 32))
    return cutlass.Int64(
        cutlass.Uint64(cutlass.Uint32(lo)) | (cutlass.Uint64(cutlass.Uint32(hi)) << 32)
    )


class ScanSm100(UmmaKernel):
    """The FlashTopK scan for one split kind (one split, or KV splits) and output
    (candidates, or every score of the windows: dense)."""

    def __init__(self, *, single_split: bool, dense: bool = False):
        self.single_split = single_split and not dense
        self.dense = dense
        self.N = NEXT_N * NUM_HEADS
        self.math_wg_registers = 224
        self.store_wg_registers = 40
        self.specialized_wg_registers = 24

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, head_dim] uint8 (FP8 E4M3 bits)
        q: cute.Tensor,  # [num_q * num_heads, head_dim, 1] uint8 (FP8 E4M3 bits)
        weights: cute.Tensor,  # [num_q, num_heads] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32
        row_start: cute.Tensor,  # [num_q] i32
        row_end: cute.Tensor,  # [num_q] i32
        origin: cute.Tensor,  # [num_q] f32
        inv_delta: cute.Tensor,  # [num_q] f32
        th_bucket: cute.Tensor,  # [num_q] i32
        # [num_q, cand_cap] f32 (dense: [num_q, >= seq_len_kv] scores)
        cand_val: cute.Tensor,
        cand_idx: cute.Tensor,  # [num_q, cand_cap] i32 (dense: unused)
        cand_cnt: cute.Tensor,  # [num_q] i32 (dense: unused)
        num_q: Int32,
        seq_len_kv: Int32,
        num_splits: Int32,
        stream: cuda.CUstream,
    ):
        kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
        b = cute.recast_tensor(q, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator,
            cute.make_layout(
                (seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0)
            ),
        )
        a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = setup_umma(
            self, a.element_type, b.element_type, a_major, b_major, NUM_KV_STAGES
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        a_op = blackwell.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE_MN, tiled_mma.thr_id)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            self.a_smem_layout_staged,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = blackwell.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE_MN, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        w_smem_layout_staged = cute.make_layout(
            (self.N, NUM_Q_STAGES), stride=(1, W_STAGE_STRIDE)
        )
        w_smem_per_stage = cute.make_layout((NEXT_N, NUM_HEADS), stride=(NUM_HEADS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, w_smem_per_stage, (NEXT_N, NUM_HEADS)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        self.num_q_tma_bytes = (
            cute.size_in_bytes(b.element_type, b_smem_layout) + self.N * 4
        )
        # One byte per FP8 element, then the block's scales.
        self.num_kv_scale_tma_bytes = BLOCK_KV * HEAD_DIM + BLOCK_KV * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, NUM_KV_STAGES * 4]
            q_mbar: cute.struct.MemRange[cutlass.Int64, NUM_Q_STAGES * 2]
            umma_mbar: cute.struct.MemRange[cutlass.Int64, NUM_UMMA_STAGES * 4]
            store_mbar: cute.struct.MemRange[cutlass.Int64, 8]
            tmem_holding_buf: Int32

        num_tiles = (num_q + NEXT_N - 1) // NEXT_N
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
            num_splits,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            w_smem_layout_staged,
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
        mW_tma: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,
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
        num_splits: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()

        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < NUM_MATH_WARPS
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_umma_warp = warp_idx == 10
        is_store_warp = warp_idx >= FIRST_STORE_WARP

        # Longest windows first: under causal masking the last tile's rows see the most
        # keys.
        tile = gdimx - 1 - bidx
        split = bidy
        query_start = tile * NEXT_N

        split_lo, split_hi, range_lo, range_hi, cover_lo, cover_hi = cta_kv_range(
            mRowStart,
            mRowEnd,
            query_start,
            NEXT_N,
            num_q,
            seq_len_kv,
            split,
            num_splits,
            True,
        )
        has_work = range_hi > range_lo
        # The scale TMA's inner coordinate stays 16-byte aligned.
        kv_start = range_lo // 4 * 4
        num_kv = Int32(0)
        if has_work:
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV
        num_pairs = (num_kv + NUM_MATH_WG - 1) // NUM_MATH_WG

        if is_tma_warp_0 | is_tma_warp_1:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=NUM_Q_STAGES,
            producer_group=prod_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        # Per group: a KV ring, a TMEM accumulator and a ring of score planes; the
        # math warpgroup's handles are on its own group's (math_ring).
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        kv_pipeline_0, kv_pipeline_1, kv_pipeline_math, math_ring = (
            make_group_pipelines(
                pipeline.PipelineTmaAsync.create,
                storage.kv_mbar.data_ptr,
                lambda g: g * NUM_KV_STAGES * 2,
                lambda: min(warpgroup_idx, Int32(1)),
                num_stages=NUM_KV_STAGES,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_scale_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
        )
        umma_cons_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MATH_THREADS // 2
        )
        umma_pipeline_0, umma_pipeline_1, umma_pipeline_math, _ = make_group_pipelines(
            create_umma_pipeline,
            storage.umma_mbar.data_ptr,
            lambda g: g * NUM_UMMA_STAGES * 2,
            math_ring,
            consumer_group=umma_cons_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        store_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        store_pipeline_0, store_pipeline_1, store_pipeline_math, _ = (
            make_group_pipelines(
                pipeline.PipelineAsync.create,
                storage.store_mbar.data_ptr,
                lambda g: g * 4,
                math_ring,
                math_kwargs={"defer_sync": True},
                num_stages=2,
                producer_group=store_group,
                consumer_group=store_group,
            )
        )
        tmem = make_tmem_allocator(storage.tmem_holding_buf)

        pipeline_init_arrive(cluster_shape_mn=CLUSTER_SHAPE_MN, is_relaxed=True)

        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=Float32, layout=w_smem_layout_staged, byte_alignment=128
        )
        sW = cute.make_tensor(
            sW.iterator,
            cute.make_layout(
                (NEXT_N, NUM_HEADS, NUM_Q_STAGES),
                stride=(NUM_HEADS, 1, W_STAGE_STRIDE),
            ),
        )
        sWarpWeights = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (NUM_MATH_WARPS, NEXT_N, NUM_HEADS),
                stride=(self.N + 4, NUM_HEADS, 1),
            ),
            byte_alignment=128,
        )
        sScales = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (BLOCK_KV, NUM_KV_STAGES, 2),
                stride=(1, BLOCK_KV, BLOCK_KV * NUM_KV_STAGES),
            ),
            byte_alignment=128,
        )
        sScales_0 = sScales[(None, None, 0)]
        sScales_1 = sScales[(None, None, 1)]
        sStoreScores = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (NEXT_N, BLOCK_KV, 2, 2),
                stride=(BLOCK_KV, 1, NEXT_N * BLOCK_KV, NEXT_N * BLOCK_KV * 2),
            ),
            byte_alignment=128,
        )
        sStoreScores_0 = sStoreScores[(None, None, None, 0)]
        sStoreScores_1 = sStoreScores[(None, None, None, 1)]

        # TMA partitions, rebased once on this CTA's query tile and KV start.
        thr_mma = tiled_mma.get_slice(0)
        shifted_a = cute.domain_offset((kv_start, 0, 0), mA_mkl)
        shifted_gA = cute.local_tile(
            shifted_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        shifted_tCgA = thr_mma.partition_A(shifted_gA)
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sKV_0, 0, 3),
            cute.group_modes(shifted_tCgA, 0, 3),
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sKV_1, 0, 3),
            cute.group_modes(shifted_tCgA, 0, 3),
        )
        shifted_s = cute.domain_offset((kv_start,), mS_tma)
        shifted_gS = cute.local_tile(shifted_s, (BLOCK_KV,), (None,))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            cute.group_modes(sScales_0, 0, 1),
            shifted_gS,
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s,
            0,
            cute.make_layout(1),
            cute.group_modes(sScales_1, 0, 1),
            shifted_gS,
        )
        shifted_b = cute.domain_offset((query_start * NUM_HEADS, 0, 0), mB_nkl)
        shifted_gB = cute.local_tile(
            shifted_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        shifted_tCgB = thr_mma.partition_B(shifted_gB)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(shifted_tCgB, 0, 3),
        )
        shifted_w = cute.domain_offset((query_start, 0), mW_tma)
        shifted_gW = cute.local_tile(shifted_w, (NEXT_N, NUM_HEADS), (None, None))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(shifted_gW, 0, 2),
        )

        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(
            cute.append(acc_shape, NUM_UMMA_STAGES)
        )
        tmem_group_1_offset = self.num_tmem_alloc_cols * NUM_UMMA_STAGES

        epi_sub_mn = (BLOCK_KV, self.N // NUM_EPI_SUBTILES)
        copy_atom_t2r = make_tmem_load_atom(self.mma_tiler)

        pipeline_init_wait(cluster_shape_mn=CLUSTER_SHAPE_MN)

        if is_tma_warp_0:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            q_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_Q_STAGES
            )
            kv_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_KV_STAGES
            )
            if has_work:
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, 0, 0)],
                    tBsB[(None, 0)],
                    tma_bar_ptr=q_bar,
                )
                cute.copy(
                    tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar
                )
                q_prod_state.advance()
            for pair in cutlass.range(num_pairs, unroll=1):
                kv_pipeline_0.producer_acquire(kv_prod_state)
                bar = kv_pipeline_0.producer_get_barrier(kv_prod_state)
                stage = kv_prod_state.index
                blk = pair * NUM_MATH_WG
                cute.copy(
                    tma_atom_a,
                    tAgA_0[(None, blk, 0, 0)],
                    tAsA_0[(None, stage)],
                    tma_bar_ptr=bar,
                )
                cute.copy(
                    tma_atom_s,
                    tSgS_0[(None, blk)],
                    tSsS_0[(None, stage)],
                    tma_bar_ptr=bar,
                )
                kv_prod_state.advance()
        elif is_tma_warp_1:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            kv_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_KV_STAGES
            )
            for pair in cutlass.range(num_pairs, unroll=1):
                kv_pipeline_1.producer_acquire(kv_prod_state)
                bar = kv_pipeline_1.producer_get_barrier(kv_prod_state)
                stage = kv_prod_state.index
                # Past the range (an odd block count) the block's scores are discarded
                # by the store warps' window check; whatever TMA reads there is never
                # published.
                blk = pair * NUM_MATH_WG + 1
                cute.copy(
                    tma_atom_a,
                    tAgA_1[(None, blk, 0, 0)],
                    tAsA_1[(None, stage)],
                    tma_bar_ptr=bar,
                )
                cute.copy(
                    tma_atom_s,
                    tSgS_1[(None, blk)],
                    tSsS_1[(None, stage)],
                    tma_bar_ptr=bar,
                )
                kv_prod_state.advance()
        elif is_umma_warp:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)
            q_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_Q_STAGES
            )
            kv_cons_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_KV_STAGES
            )
            kv_cons_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, NUM_KV_STAGES
            )
            umma_prod_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_UMMA_STAGES
            )
            umma_prod_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, NUM_UMMA_STAGES
            )
            named_barrier_sync_unaligned(TMEM_ALLOC_THREADS)
            tmem_ptr = tmem.retrieve_ptr(Float32)
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            tCtAcc_base_1 = cute.make_tensor(
                tmem_ptr + tmem_group_1_offset, tCtAcc_fake_staged.layout
            )
            num_k_blocks = cute.size(tCrA_0.shape[2])
            if has_work:
                q_pipeline.consumer_wait(q_cons_state)
            for _pair in cutlass.range(num_pairs, unroll=1):
                kv_ready_0 = kv_pipeline_0.consumer_try_wait(kv_cons_state_0)
                umma_empty_0 = umma_pipeline_0.producer_try_acquire(umma_prod_state_0)
                kv_pipeline_0.consumer_wait(kv_cons_state_0, kv_ready_0)
                umma_pipeline_0.producer_acquire(umma_prod_state_0, umma_empty_0)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                kv_stage = kv_cons_state_0.index
                tCtAcc_0 = tCtAcc_base_0[(None, None, None, umma_prod_state_0.index)]
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_0,
                        tCrA_0[None, None, k_block, kv_stage],
                        tCrB[None, None, k_block, 0],
                        tCtAcc_0,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                kv_cons_state_0.advance()
                umma_pipeline_0.producer_commit(umma_prod_state_0)
                umma_prod_state_0.advance()

                kv_ready_1 = kv_pipeline_1.consumer_try_wait(kv_cons_state_1)
                umma_empty_1 = umma_pipeline_1.producer_try_acquire(umma_prod_state_1)
                kv_pipeline_1.consumer_wait(kv_cons_state_1, kv_ready_1)
                umma_pipeline_1.producer_acquire(umma_prod_state_1, umma_empty_1)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                kv_stage_1 = kv_cons_state_1.index
                tCtAcc_1 = tCtAcc_base_1[(None, None, None, umma_prod_state_1.index)]
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc_1,
                        tCrA_1[None, None, k_block, kv_stage_1],
                        tCrB[None, None, k_block, 0],
                        tCtAcc_1,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                kv_cons_state_1.advance()
                umma_pipeline_1.producer_commit(umma_prod_state_1)
                umma_prod_state_1.advance()
        elif is_math_warp:
            self._run_math_warp(
                has_work,
                num_pairs,
                copy_atom_t2r,
                epi_sub_mn,
                kv_pipeline_math,
                q_pipeline,
                sScales,
                sW,
                sWarpWeights,
                sStoreScores,
                store_pipeline_math,
                tCtAcc_fake_staged,
                tidx,
                tmem,
                tmem_group_1_offset,
                umma_pipeline_math,
                warpgroup_idx,
                kv_start,
            )
        elif is_store_warp:
            self._run_store_warp(
                num_pairs,
                kv_start,
                cover_lo,
                cover_hi,
                split_lo,
                split_hi,
                query_start,
                num_q,
                seq_len_kv,
                mRowStart,
                mRowEnd,
                mOrigin,
                mInvDelta,
                mThBucket,
                mCandVal,
                mCandIdx,
                mCandCnt,
                sStoreScores_0,
                sStoreScores_1,
                store_pipeline_0,
                store_pipeline_1,
            )
        else:
            cute.arch.setmaxregister_decrease(self.specialized_wg_registers)

    @cute.jit
    def _run_math_warp(
        self,
        has_work,
        num_pairs,
        copy_atom_t2r,
        epi_sub_mn,
        kv_pipeline_math,
        q_pipeline,
        sScales,
        sW,
        sWarpWeights,
        sStoreScores,
        store_pipeline_math,
        tCtAcc_fake_staged,
        tidx,
        tmem,
        tmem_group_1_offset,
        umma_pipeline_math,
        warpgroup_idx,
        kv_start,
    ):
        """One math warpgroup: scores its blocks into its shared score plane for the
        store warps."""
        cute.arch.setmaxregister_increase(self.math_wg_registers)
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_Q_STAGES
        )
        tmem.allocate(self.num_tmem_alloc_cols * NUM_MATH_WG * NUM_UMMA_STAGES)
        named_barrier_sync_unaligned(TMEM_ALLOC_THREADS)
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tCtAcc_base = cute.make_tensor(
            tmem_ptr + warpgroup_idx * tmem_group_1_offset, tCtAcc_fake_staged.layout
        )

        weight_warp = cute.arch.warp_idx()
        weight_lane = cute.arch.lane_idx()
        epi_tidx = tidx % 128
        cC = cute.make_identity_tensor(epi_sub_mn)
        kv_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_KV_STAGES
        )
        umma_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, NUM_UMMA_STAGES
        )
        store_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2
        )

        tiled_copy_t2r, thr_copy_t2r, tTR_tAcc_base = partition_tmem_readout(
            copy_atom_t2r, tCtAcc_base, epi_sub_mn, epi_tidx
        )
        tTR_cC = thr_copy_t2r.partition_D(cC)
        tTR_rAcc = cute.make_rmem_tensor_like(tTR_cC, Float32)
        m_coord = tTR_cC[0][0]

        w_cache = make_weight_cache(NEXT_N)
        if has_work:
            q_pipeline.consumer_wait(q_cons_state)
            cute.arch.sync_warp()
            replicate_weights(
                sW, sWarpWeights, w_cache, weight_warp, weight_lane, NEXT_N
            )

        num_epi_subtiles = NUM_EPI_SUBTILES
        rows_per_subtile = NEXT_N // num_epi_subtiles
        for _pair in cutlass.range(num_pairs, unroll=1):
            store_pipeline_math.producer_acquire(store_prod_state)
            # The UMMA warp waited on this KV stage before its MMA committed the barrier
            # below, so the stage's scales are visible here too.
            umma_pipeline_math.consumer_wait(umma_cons_state)
            # TMEM loads are warp-aligned; a barrier wait may return with the warp
            # diverged.
            cute.arch.sync_warp()
            tTR_tAcc = tTR_tAcc_base[
                (None, None, None, None, None, umma_cons_state.index)
            ]
            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)], tTR_rAcc)
            kv_stage = kv_cons_state.index
            score_stage = store_prod_state.index
            scale_val = sScales[(m_coord, kv_stage, warpgroup_idx)]
            cute.arch.fence_view_async_tmem_load()
            for epi_subtile in cutlass.range_constexpr(num_epi_subtiles):
                if epi_subtile > 0:
                    cute.arch.fence_view_async_tmem_load()
                if epi_subtile == num_epi_subtiles - 1:
                    umma_pipeline_math.consumer_release(umma_cons_state)
                    umma_cons_state.advance()
                acc_vec = tTR_rAcc.load()
                if epi_subtile + 1 < num_epi_subtiles:
                    # Overlap the next subtile's TMEM load with this subtile's
                    # arithmetic.
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, epi_subtile + 1)],
                        tTR_rAcc,
                    )
                subtile_base_t = epi_subtile * rows_per_subtile
                reduced = self._fast_sums(
                    acc_vec, w_cache, sWarpWeights, weight_warp, subtile_base_t
                )
                for r in cutlass.range_constexpr(rows_per_subtile):
                    sStoreScores[
                        (subtile_base_t + r, m_coord, score_stage, warpgroup_idx)
                    ] = reduced[r] * scale_val
            # Every score store has consumed the register-resident scale.
            kv_pipeline_math.consumer_release(kv_cons_state)
            kv_cons_state.advance()
            store_pipeline_math.producer_commit(store_prod_state)
            store_prod_state.advance()

        if has_work:
            q_pipeline.consumer_release(q_cons_state)
            q_cons_state.advance()
        store_pipeline_math.producer_tail(store_prod_state)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _fast_sums(self, acc_vec, w_cache, sWarpWeights, weight_warp, t0):
        """Each of the subtile's rows' head sum (before the kv scale): heads 4j, 4j + 1
        and 4j + 2, 4j + 3 in two packed FMA chains, then their sum. Head-outer,
        row-inner, so the rows' chains interleave while each keeps its head order."""
        rows = NEXT_N // NUM_EPI_SUBTILES
        sums = cute.make_rmem_tensor((rows, 4), Float32)
        sums.fill(Float32(0.0))
        for h in cutlass.range_constexpr(0, NUM_HEADS, 4):
            for r in cutlass.range_constexpr(rows):
                t = t0 + r
                a01, a23 = head_quad_activations(acc_vec, r * NUM_HEADS + h, h)
                w = [
                    head_weight(w_cache, sWarpWeights, weight_warp, t, h + i)
                    for i in range(4)
                ]
                s0x, s0y = cute.arch.fma_packed_f32x2(
                    a01, (w[0], w[1]), (sums[(r, 0)], sums[(r, 1)]), rnd=RND_RN
                )
                s1x, s1y = cute.arch.fma_packed_f32x2(
                    a23, (w[2], w[3]), (sums[(r, 2)], sums[(r, 3)]), rnd=RND_RN
                )
                sums[(r, 0)] = s0x
                sums[(r, 1)] = s0y
                sums[(r, 2)] = s1x
                sums[(r, 3)] = s1y
        reduced = []
        for r in cutlass.range_constexpr(rows):
            sum_x, sum_y = cute.arch.add_packed_f32x2(
                (sums[(r, 0)], sums[(r, 1)]), (sums[(r, 2)], sums[(r, 3)])
            )
            reduced.append(sum_x + sum_y)
        return reduced

    @cute.jit
    def _run_store_warp(
        self,
        num_pairs,
        kv_start,
        cover_lo,
        cover_hi,
        split_lo,
        split_hi,
        query_start,
        num_q,
        seq_len_kv,
        mRowStart,
        mRowEnd,
        mOrigin,
        mInvDelta,
        mThBucket,
        mCandVal,
        mCandIdx,
        mCandCnt,
        sStoreScores_0,
        sStoreScores_1,
        store_pipeline_0,
        store_pipeline_1,
    ):
        """One store warp: gates and appends the hits of rows s and s + 4 (dense: writes
        every score of their windows at its key)."""
        cute.arch.setmaxregister_decrease(self.store_wg_registers)
        num_q_groups = NEXT_N // NUM_STORE_WARPS
        store_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )
        store_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )
        store_warp = cute.arch.make_warp_uniform(
            cute.arch.warp_idx() - Int32(FIRST_STORE_WARP)
        )
        lane = cute.arch.lane_idx()
        lane_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        cand_cap = mCandVal.layout.shape[1]
        val_addr = make_warp_uniform_i64(mCandVal.iterator.toint())
        idx_addr = make_warp_uniform_i64(mCandIdx.iterator.toint())
        cnt_addr = make_warp_uniform_i64(mCandCnt.iterator.toint())
        val_stride = make_warp_uniform_i64(mCandVal.layout.stride[0])
        idx_stride = make_warp_uniform_i64(mCandIdx.layout.stride[0])

        # Per owned row: validity, raw gate, window [start, start + len), output rows,
        # cursor.
        row_valid = [cutlass.Boolean(False) for _ in range(num_q_groups)]
        gates = [Float32(0.0) for _ in range(num_q_groups)]
        win_start = [Int32(0) for _ in range(num_q_groups)]
        win_len = [Int32(0) for _ in range(num_q_groups)]
        val_rows = [cutlass.Int64(0) for _ in range(num_q_groups)]
        idx_rows = [cutlass.Int64(0) for _ in range(num_q_groups)]
        cursors = [Int32(0) for _ in range(num_q_groups)]
        for g in cutlass.range_constexpr(num_q_groups):
            row = query_start + store_warp + g * NUM_STORE_WARPS
            if row < num_q:
                row_valid[g] = cutlass.Boolean(True)
                if cutlass.const_expr(not self.dense):
                    gates[g] = loose_raw_gate(
                        mInvDelta[row], mOrigin[row], mThBucket[row]
                    )
                s, e = clip_window(
                    mRowStart, mRowEnd, row, seq_len_kv, split_lo, split_hi
                )
                win_start[g] = s
                win_len[g] = max(e - s, Int32(0))
                val_rows[g] = val_addr + cutlass.Int64(
                    row
                ) * val_stride * cutlass.Int64(4)
                idx_rows[g] = idx_addr + cutlass.Int64(
                    row
                ) * idx_stride * cutlass.Int64(4)

        for pair in cutlass.range(num_pairs, unroll=1):
            blk_0 = kv_start + pair * (NUM_MATH_WG * BLOCK_KV)
            key_base_0 = blk_0 + lane * 4
            key_base_1 = key_base_0 + BLOCK_KV
            interior = (blk_0 >= cover_lo) & (
                blk_0 + NUM_MATH_WG * BLOCK_KV <= cover_hi
            )
            store_ready_0 = store_pipeline_0.consumer_try_wait(store_cons_state_0)
            store_ready_1 = store_pipeline_1.consumer_try_wait(store_cons_state_1)
            store_pipeline_0.consumer_wait(store_cons_state_0, store_ready_0)
            store_pipeline_1.consumer_wait(store_cons_state_1, store_ready_1)
            group_scores_0 = []
            group_scores_1 = []
            for g in cutlass.range_constexpr(num_q_groups):
                row_local = store_warp + g * NUM_STORE_WARPS
                group_scores_0.append(
                    load_shared_v4_f32(
                        sStoreScores_0, row_local, lane * 4, store_cons_state_0.index
                    )
                )
                group_scores_1.append(
                    load_shared_v4_f32(
                        sStoreScores_1, row_local, lane * 4, store_cons_state_1.index
                    )
                )
            # Both planes are in registers: the math warpgroups may refill their stages.
            store_pipeline_0.consumer_release(store_cons_state_0)
            store_cons_state_0.advance()
            store_pipeline_1.consumer_release(store_cons_state_1)
            store_cons_state_1.advance()

            if cutlass.const_expr(self.dense):
                # Every in-window score at its key; the window check is per row, never
                # the CTA's cover, which leaves rows with empty windows out.
                for g in cutlass.range_constexpr(num_q_groups):
                    for v in cutlass.range_constexpr(4):
                        for scores, key in (
                            (group_scores_0[g], key_base_0 + v),
                            (group_scores_1[g], key_base_1 + v),
                        ):
                            inside = row_valid[g] & (
                                cutlass.Uint32(key - win_start[g])
                                < cutlass.Uint32(win_len[g])
                            )
                            store_global_b32_if(
                                inside,
                                val_rows[g] + cutlass.Int64(key) * cutlass.Int64(4),
                                fp32_to_i32_bits(scores[v]),
                            )
            else:
                for g in cutlass.range_constexpr(num_q_groups):
                    scores_0 = group_scores_0[g]
                    scores_1 = group_scores_1[g]
                    gate = gates[g]
                    masks = []
                    total = Int32(0)
                    for v in cutlass.range_constexpr(4):
                        # An interior pair is inside every row's window; elsewhere an
                        # invalid or empty row has a zero-length window and takes
                        # nothing.
                        in_0 = interior | (
                            cutlass.Uint32(key_base_0 + v - win_start[g])
                            < cutlass.Uint32(win_len[g])
                        )
                        in_1 = interior | (
                            cutlass.Uint32(key_base_1 + v - win_start[g])
                            < cutlass.Uint32(win_len[g])
                        )
                        sel_0 = (scores_0[v] > gate) & in_0
                        sel_1 = (scores_1[v] > gate) & in_1
                        mask_0 = cutlass.Uint32(cute.arch.vote_ballot_sync(sel_0))
                        mask_1 = cutlass.Uint32(cute.arch.vote_ballot_sync(sel_1))
                        masks.append((mask_0, mask_1))
                        total = (
                            total
                            + Int32(cute.arch.popc(mask_0))
                            + Int32(cute.arch.popc(mask_1))
                        )
                    if total > 0:
                        base = cursors[g]
                        if cutlass.const_expr(self.single_split):
                            cursors[g] = base + total
                        else:
                            reserved = Int32(0)
                            if lane == 0:
                                row = query_start + store_warp + g * NUM_STORE_WARPS
                                reserved = atomic_add_global_i32(
                                    cnt_addr + cutlass.Int64(row) * cutlass.Int64(4),
                                    total,
                                )
                            base = cute.arch.shuffle_sync(reserved, 0)
                        pos = base
                        for v in cutlass.range_constexpr(4):
                            mask_0, mask_1 = masks[v]
                            sel_0 = (mask_0 >> cutlass.Uint32(lane)) & cutlass.Uint32(
                                1
                            ) != cutlass.Uint32(0)
                            sel_1 = (mask_1 >> cutlass.Uint32(lane)) & cutlass.Uint32(
                                1
                            ) != cutlass.Uint32(0)
                            slot_0 = pos + Int32(cute.arch.popc(mask_0 & lane_mask))
                            pos = pos + Int32(cute.arch.popc(mask_0))
                            slot_1 = pos + Int32(cute.arch.popc(mask_1 & lane_mask))
                            pos = pos + Int32(cute.arch.popc(mask_1))
                            ok_0 = sel_0 & (slot_0 < cand_cap)
                            ok_1 = sel_1 & (slot_1 < cand_cap)
                            off_0 = cutlass.Int64(slot_0) * cutlass.Int64(4)
                            off_1 = cutlass.Int64(slot_1) * cutlass.Int64(4)
                            store_global_b32_if(
                                ok_0,
                                val_rows[g] + off_0,
                                fp32_to_i32_bits(-scores_0[v]),
                            )
                            store_global_b32_if(
                                ok_0, idx_rows[g] + off_0, key_base_0 + v
                            )
                            store_global_b32_if(
                                ok_1,
                                val_rows[g] + off_1,
                                fp32_to_i32_bits(-scores_1[v]),
                            )
                            store_global_b32_if(
                                ok_1, idx_rows[g] + off_1, key_base_1 + v
                            )

        if cutlass.const_expr(self.single_split):
            for g in cutlass.range_constexpr(num_q_groups):
                if row_valid[g] & (lane == 0):
                    row = query_start + store_warp + g * NUM_STORE_WARPS
                    store_global_b32(
                        cnt_addr + cutlass.Int64(row) * cutlass.Int64(4), cursors[g]
                    )
