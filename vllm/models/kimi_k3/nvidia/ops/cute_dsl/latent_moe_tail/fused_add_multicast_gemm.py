# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Blackwell GEMM with a fused shared-shard add and multicast epilogue.

Modified from the original CUTLASS CuTe DSL SM100 persistent GEMM tutorial.
The PDL wait before A loading orders the up-projection and shared-expert
inputs after the producer collective.
"""

import math
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import CacheEvictionPriority
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .fused_add_multicast_skinny_gemm import (
    FusedAddMulticastSkinnyGemmKernel,
)
from .primitives import CUDAGraphCompatibleWrapper


def _as_cute(tensor: torch.Tensor, *, dynamic_m: bool = False):
    converted = from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=16
    )
    if dynamic_m:
        converted = converted.mark_compact_shape_dynamic(
            mode=1,
            stride_order=tensor.dim_order(),
        )
    return converted


def validate_configuration(
    *,
    latent_dim: int,
    shard_dim: int,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    b_prime_stages: int,
) -> None:
    """Validate constraints imposed by this BF16 SM100 GEMM."""

    if latent_dim <= 0 or shard_dim <= 0:
        raise ValueError("GEMM K and N must be positive")
    if latent_dim % 8 or shard_dim % 8:
        raise ValueError("GEMM K and N must be divisible by 8 BF16 values")
    if mma_tiler_mn[0] not in (64, 128):
        raise ValueError("MMA M tile must be 64 or 128")
    if mma_tiler_mn[1] not in range(32, 257, 32):
        raise ValueError("MMA N tile must be a multiple of 32 in [32, 256]")
    if (
        len(cluster_shape_mn) != 2
        or any(value <= 0 or value & (value - 1) for value in cluster_shape_mn)
        or math.prod(cluster_shape_mn) > 16
    ):
        raise ValueError("cluster dimensions must be powers of two with product <= 16")
    if not 0 <= b_prime_stages <= math.ceil(latent_dim / 128):
        raise ValueError("b_prime_stages exceeds the GEMM K-tile count")


@cute.jit
def _epilogue_tma_store_add_shared(
    gemm_kernel,
    epi_tidx: cutlass.Int32,
    warp_idx: cutlass.Int32,
    tma_atom_c: cute.CopyAtom,
    tCtAcc_base: cute.Tensor,
    sC: cute.Tensor,
    tCgC_base: cute.Tensor,
    tCgShared_base: cute.Tensor,
    tCcC_base: cute.Tensor,
    mC_mnl: cute.Tensor,
    mShared_mnl: cute.Tensor,
    epi_tile: cute.Tile,
    num_tiles_executed: cutlass.Int32,
    mma_tile_coord_mnl,
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
) -> pipeline.PipelineState:
    """BF16 GEMM rounding + BF16 shared shard addition, then swizzled S2G TMA."""
    sm100 = utils.gemm.sm100
    tCgC = sm100.transform_partitioned_tensor_layout(tCgC_base)
    tCgShared = sm100.transform_partitioned_tensor_layout(tCgShared_base)
    tCcC = sm100.transform_partitioned_tensor_layout(tCcC_base)
    tCtAcc = sm100.transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = sm100.epilogue_tmem_copy_and_partition(
        gemm_kernel,
        epi_tidx,
        tCtAcc,
        tCgC,
        epi_tile,
        False,
    )
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tTR_rShared = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = sm100.epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )
    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gShared_partitioned = thr_copy_t2r.partition_D(
        cute.flat_divide(tCgShared, epi_tile)
    )
    tTR_cC_partitioned = thr_copy_t2r.partition_D(cute.flat_divide(tCcC, epi_tile))
    tTR_gShared = tTR_gShared_partitioned[
        (None, None, None, None, None, *mma_tile_coord_mnl)
    ]
    tTR_cC = tTR_cC_partitioned[(None, None, None, None, None, *mma_tile_coord_mnl)]

    exemplar = tTR_gShared_partitioned[(None, None, None, 0, 0, 0, 0, 0)]
    mcl_r = cute.max_common_layout(tTR_rShared.layout, exemplar.layout)
    shared_copy_bits = min(
        exemplar.iterator.alignment * 8,
        cute.size(mcl_r) * gemm_kernel.c_dtype.width,
        128,
    )
    shared_g2r_atom = cute.make_copy_atom(
        cute.nvgpu.CopyG2ROp(),
        gemm_kernel.c_dtype,
        num_bits_per_copy=shared_copy_bits,
        l1c_evict_priority=CacheEvictionPriority.NO_ALLOCATE,
    )

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gShared = cute.group_modes(tTR_gShared, 3, cute.rank(tTR_gShared))
    tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    previous_subtile_count = num_tiles_executed * subtile_cnt
    cute.arch.griddepcontrol_wait()
    for subtile_idx in range(subtile_cnt):
        tTR_gShared_subtile = tTR_gShared[(None, None, None, subtile_idx)]
        tTR_cC_subtile = tTR_cC[(None, None, None, subtile_idx)]
        pred_shape = (1, *tTR_cC_subtile.shape[1:])
        pred = cute.make_rmem_tensor(pred_shape, cutlass.Boolean)
        for m_idx in range(tTR_cC_subtile.shape[1]):
            for n_idx in range(tTR_cC_subtile.shape[2]):
                pred[(0, m_idx, n_idx)] = cute.elem_less(
                    tTR_cC_subtile[(0, m_idx, n_idx)], mC_mnl.shape
                )
        tTR_rShared.store(cute.zeros_like(tTR_rShared, dtype=gemm_kernel.c_dtype))
        cute.copy(
            shared_g2r_atom,
            tTR_gShared_subtile,
            tTR_rShared,
            pred=pred,
        )

        # Load the shared addend before waiting for the accumulator.
        if subtile_idx == 0:
            acc_pipeline.consumer_wait(acc_consumer_state)
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        gemm_vec = tiled_copy_r2s.retile(tTR_rAcc).load().to(gemm_kernel.c_dtype)
        shared_vec = tiled_copy_r2s.retile(tTR_rShared).load()
        fused_vec = (gemm_vec.to(cutlass.Float32) + shared_vec.to(cutlass.Float32)).to(
            gemm_kernel.c_dtype
        )
        # The symmetric output is an in-band Lamport mailbox whose empty
        # marker contains BF16 -0. Normalize either signed zero to +0 so a
        # legitimate result can never be mistaken for an unwritten fragment.
        fused_vec = cute.where(
            fused_vec == cute.zeros_like(fused_vec),
            cute.zeros_like(fused_vec),
            fused_vec,
        )
        tRS_rC.store(fused_vec)

        c_buffer = (previous_subtile_count + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()
        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, subtile_idx)],
            )
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: tuple[int, int, int],
    a_dtype,
    b_dtype,
    c_dtype,
    smem_capacity: int,
    c_smem_layout,
) -> tuple[int, int, int]:
    """Choose accumulator, mainloop, and epilogue stage counts."""
    num_acc_stage = 2
    num_c_stage = 2
    a_smem_layout_stage_one = utils.sm100.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = utils.sm100.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage
    num_ab_stage = (
        smem_capacity - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage
    num_c_stage += (
        smem_capacity
        - ab_bytes_per_stage * num_ab_stage
        - (mbar_helpers_bytes + c_bytes)
    ) // c_bytes_per_stage
    return num_acc_stage, num_ab_stage, num_c_stage


class FusedAddMulticastGemm:
    """Persistent Blackwell GEMM with a shared-add epilogue.

    B priming may overlap the producer collective. The PDL wait before A
    loading orders both inputs before the shared-add epilogue.
    """

    def __init__(
        self,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        b_prime_stages: int = 2,
    ):
        self.acc_dtype = cutlass.Float32
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        # B primes the combined A+B pipeline before the PDL wait.
        self.b_prime_stages = b_prime_stages
        self.cta_group = tcgen05.CtaGroup.ONE
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
        )
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2

    def _create_tiled_mma(self):
        return utils.sm100.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        """Derive layouts and stage counts from the compiled tensor shapes."""
        tiled_mma = self._create_tiled_mma()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            False,
            self.c_layout,
            self.c_dtype,
        )
        c_smem_layout = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            utils.get_smem_capacity_in_bytes(),
            c_smem_layout,
        )

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )

        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, "sm_100"
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        shared_shard: cute.Tensor,
        c_multicast_i64: cutlass.Int64,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the persistent GEMM."""
        # Preserve C's logical strided layout but point the TMA descriptor at
        # this rank's shard inside the LSA multicast mapping.  One TMA store is
        # therefore replicated into the same shard on all eight ranks.
        c = cute.make_tensor(
            cute.make_ptr(
                c.element_type,
                c_multicast_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            c.layout,
        )

        self.a_dtype: type[cutlass.Numeric] = a.element_type
        self.b_dtype: type[cutlass.Numeric] = b.element_type
        self.c_dtype: type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()

        self._setup_attributes()
        if cutlass.const_expr(self.b_prime_stages > self.num_ab_stage):
            raise ValueError(
                "b_prime_stages exceeds the compiled A/B pipeline stage count"
            )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )

        tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
            shared_shard,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=True,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        shared_shard: cute.Tensor,
    ):
        self._gemm_device(
            tiled_mma,
            tma_atom_a,
            mA_mkl,
            tma_atom_b,
            mB_nkl,
            tma_atom_c,
            mC_mnl,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout_staged,
            c_smem_layout_staged,
            epi_tile,
            tile_sched_params,
            shared_shard,
        )

    @cute.jit
    def _gemm_device(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.Layout | cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        shared_shard: cute.Tensor,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        # Shared shard is physically [M, shard_dim]. Give it the same logical MNL
        # view as C so its epilogue partition is coordinate-identical.
        mShared_mnl = cute.make_tensor(
            shared_shard.iterator,
            cute.append(shared_shard.layout, cute.make_layout((1,), stride=(0,))),
        )
        gShared_mnl = cute.local_tile(
            mShared_mnl,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgShared = thr_mma.partition_C(gShared_mnl)

        # Predicate the partial M tile when M is smaller than the MMA tile.
        idC = cute.make_identity_tensor(mC_mnl.shape)
        cC_mnl = cute.local_tile(
            idC, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        tCcC = thr_mma.partition_C(cC_mnl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        gemm_grid_z = cute.arch.grid_dim()[2]
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            (
                cute.arch.grid_dim()[0],
                cute.arch.grid_dim()[1],
                gemm_grid_z,
            ),
        )
        work_tile = tile_sched.initial_work_tile_info()

        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Prime a short prefix of the existing combined A+B ring with
                # B only.  Its barrier still expects A+B bytes, so the MMA
                # consumer cannot observe a half-filled stage.
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, self.b_prime_stages, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < self.b_prime_stages:
                        peek_ab_empty_status = ab_producer.try_acquire()

                cute.arch.griddepcontrol_wait()

                # Supply A to the very same stages after the producer AR has
                # programmatically released this dependent kernel.
                a_fill_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                for k_tile in cutlass.range(0, self.b_prime_stages, 1, unroll=1):
                    a_barrier = ab_pipeline.producer_get_barrier(a_fill_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_tile)],
                        tAsA[(None, a_fill_state.index)],
                        tma_bar_ptr=a_barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    a_fill_state.advance()

                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(
                    self.b_prime_stages, k_tile_cnt, 1, unroll=1
                ):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        handle.release()

                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group
            )
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed
                acc_consumer_state = _epilogue_tma_store_add_shared(
                    self,
                    tidx,
                    warp_idx,
                    tma_atom_c,
                    tCtAcc_base,
                    sC,
                    tCgC,
                    tCgShared,
                    tCcC,
                    mC_mnl,
                    mShared_mnl,
                    epi_tile,
                    num_tiles_executed,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    c_pipeline,
                )

            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # Allow the Lamport copy to become resident before this grid
        # fully retires. Its griddepcontrol.wait still enforces complete
        # producer-grid ordering before mailbox inspection.
        cute.arch.griddepcontrol_launch_dependents()

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]:
        """Build the static persistent schedule."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(
        tiled_mma: cute.TiledMma,
        mma_tiler: tuple[int, int, int],
        num_acc_stage: int,
        arch: str,
    ) -> int:
        """Return the required tensor-memory column count."""
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake, arch=arch)

        return num_tmem_alloc_cols


@cute.jit
def launch_kernel(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (l, m, k)
    b: cute.Tensor,  # (l, n, k)
    c: cute.Tensor,  # (l, m, n)
    shared_shard: cute.Tensor,  # (m, shard_dim), private TP shard
    rows: cutlass.Int64,
    c_multicast_i64: cutlass.Int64,
    full_hidden_dim: cutlass.Constexpr,
    shard_dim: cutlass.Constexpr,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Launch the fused-add multicast GEMM using PyTorch BMM tensor order."""
    # C is passed as the fixed-capacity [1,max_m,H] symmetric mailbox. Only M
    # is runtime-variable; construct this rank's logical [1,M,S] view here so
    # H and S remain compile-time constants and no dynamic strided host view is
    # needed on every call. __call__ later replaces the local base pointer with
    # the already rank-offset multicast address.
    c = cute.make_tensor(
        c.iterator,
        cute.make_layout(
            (1, rows, shard_dim),
            stride=(0, full_hidden_dim, 1),
        ),
    )
    # (l,m,k) -> (m,k,l)
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    # (l,n,k) -> (n,k,l)
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
    # (l,m,n) -> (m,n,l)
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))

    gemm_op(
        a,
        b,
        c,
        shared_shard,
        c_multicast_i64,
        max_active_clusters,
        stream,
    )


_COMPILED: dict[tuple[object, ...], object] = {}


def compile_kernel(
    mnkl: tuple[int, int, int, int],
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    shared_shard: cute.Tensor,
    full_hidden_dim: int,
    shard_dim: int,
    mma_tiler_mn: tuple[int, int] = (64, 32),
    cluster_shape_mn: tuple[int, int] = (1, 8),
    max_active_clusters: cutlass.Constexpr = None,
    b_prime_stages: int = 2,
):
    key = (
        torch.accelerator.current_device_index(),
        mnkl,
        full_hidden_dim,
        shard_dim,
        mma_tiler_mn,
        cluster_shape_mn,
        max_active_clusters,
        b_prime_stages,
    )
    if key in _COMPILED:
        return _COMPILED[key]

    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    gemm = FusedAddMulticastGemm(
        mma_tiler_mn,
        cluster_shape_mn,
        b_prime_stages,
    )
    validate_configuration(
        latent_dim=mnkl[2],
        shard_dim=mnkl[1],
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        b_prime_stages=b_prime_stages,
    )
    if any(tensor.element_type is not cutlass.BFloat16 for tensor in (a, b, c)):
        raise ValueError("up-projection tensors must be BF16")

    # The producer writes [M, full_hidden_dim]. GEMM receives a rank-local
    # [M, shard_dim] view: contiguous within a row, with full_hidden_dim as
    # its leading dimension.
    shared_shard_compile = make_fake_tensor(
        shared_shard.element_type,
        (mnkl[0], shard_dim),
        stride=(full_hidden_dim, 1),
        assumed_align=16,
    )
    stream = make_fake_stream()
    compiled = cute.compile(
        launch_kernel,
        gemm,
        a,
        b,
        c,
        shared_shard_compile,
        cutlass.Int64(mnkl[0]),
        cutlass.Int64(0),
        full_hidden_dim,
        shard_dim,
        max_active_clusters,
        stream,
    )
    _COMPILED[key] = compiled
    return compiled


class AdaptiveUpProjectionKernel:
    """Dispatch static-M Skinny or dynamic-M WGMMA into one mailbox."""

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        rank: int,
        tp_size: int,
        latent_dim: int,
        hidden_dim: int,
        max_m: int,
        skinny_max_m: int,
        mma_tiler_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        b_prime_stages: int,
    ) -> None:
        if hidden_dim % tp_size:
            raise ValueError("hidden_dim must be divisible by TP size")
        if not 0 <= skinny_max_m <= min(8, max_m):
            raise ValueError("skinny_max_m must be in [0, min(8, max_m)]")
        self.rank = rank
        self.tp_size = tp_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.shard_dim = hidden_dim // tp_size
        self.max_m = max_m
        self.skinny_max_m = skinny_max_m
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.b_prime_stages = b_prime_stages
        device = torch.device("cuda", torch.accelerator.current_device_index())
        self._device = device
        self._dynamic: Any | None = None
        self._skinny_by_m: dict[int, FusedAddMulticastSkinnyGemmKernel] = {}
        validate_configuration(
            latent_dim=latent_dim,
            shard_dim=self.shard_dim,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            b_prime_stages=b_prime_stages,
        )
        if skinny_max_m and self.latent_dim % (224 * 8):
            raise ValueError(
                "Skinny up-projection requires latent_dim divisible by 1792."
            )

        self._mailbox = symm_mem.empty(
            (1, max_m, hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        self._mailbox_symm_mem = symm_mem.rendezvous(self._mailbox, group)
        self._mailbox.view(torch.int32).fill_(-0x80000000)
        multicast_ptr = self._mailbox_symm_mem.multicast_ptr
        if multicast_ptr is None or multicast_ptr == 0:
            raise RuntimeError("mailbox NVLS multicast mapping is unavailable")
        self._mailbox_multicast_ptr = int(multicast_ptr)

        cluster_size = math.prod(cluster_shape_mn)
        self._max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        self._mailbox_c = _as_cute(self._mailbox)

    def compile_dynamic(self) -> None:
        if self._dynamic is not None:
            return
        device = self._device
        with torch.accelerator.device_index(device.index):
            compile_latent = torch.empty(
                (1, self.max_m, self.latent_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            compile_weight = torch.empty(
                (self.shard_dim, self.latent_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            compile_shared = torch.empty(
                (self.max_m, self.hidden_dim),
                dtype=torch.bfloat16,
                device=device,
            )[
                :,
                self.rank * self.shard_dim : (self.rank + 1) * self.shard_dim,
            ]
            compile_latent_c = _as_cute(compile_latent, dynamic_m=True)
            compile_weight_c = _as_cute(compile_weight.unsqueeze(0))
            compile_shared_c = _as_cute(compile_shared)

            self._dynamic = compile_kernel(
                (self.max_m, self.shard_dim, self.latent_dim, 1),
                compile_latent_c,
                compile_weight_c,
                self._mailbox_c,
                compile_shared_c,
                self.hidden_dim,
                self.shard_dim,
                self.mma_tiler_mn,
                self.cluster_shape_mn,
                self._max_active_clusters,
                self.b_prime_stages,
            )

    def compile_skinny(self, m: int) -> None:
        if not 1 <= m <= self.skinny_max_m:
            raise ValueError(
                f"Skinny up-projection requires M in [1, {self.skinny_max_m}]."
            )
        if m in self._skinny_by_m:
            return
        with torch.accelerator.device_index(self._device.index):
            self._skinny_by_m[m] = FusedAddMulticastSkinnyGemmKernel(
                rank=self.rank,
                tp_size=self.tp_size,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                num_rows=m,
            )

    def ensure_compiled(self, m: int) -> None:
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")
        if m <= self.skinny_max_m:
            self.compile_skinny(m)
        else:
            self.compile_dynamic()

    def __call__(
        self,
        latent: torch.Tensor,
        weight: torch.Tensor,
        shared_shard: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("latent must be rank-2")
        m = latent.shape[0]
        device = self._mailbox.device
        expected = (
            (latent, (m, self.latent_dim), "latent"),
            (
                weight,
                (self.shard_dim, self.latent_dim),
                "weight",
            ),
            (
                shared_shard,
                (self.max_m, self.shard_dim),
                "shared_shard",
            ),
        )
        for tensor, shape, name in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != device
            ):
                raise ValueError(f"{name} must be CUDA torch.bfloat16 {list(shape)}")
        if (
            not latent.is_contiguous()
            or not weight.is_contiguous()
            or shared_shard.stride() != (self.hidden_dim, 1)
        ):
            raise ValueError("up-projection inputs have unsupported strides")
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")

        if m <= self.skinny_max_m:
            skinny = self._skinny_by_m.get(m)
            if skinny is None:
                raise RuntimeError(
                    f"Skinny up-projection M={m} was not compiled before launch."
                )
            return skinny(
                latent,
                weight,
                shared_shard,
                self._mailbox,
                self._mailbox_multicast_ptr,
            )

        if self._dynamic is None:
            raise RuntimeError("Dynamic up-projection was not compiled before launch.")
        with torch.accelerator.device_index(device.index):
            stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
            self._dynamic(
                _as_cute(latent.unsqueeze(0), dynamic_m=True),
                _as_cute(weight.unsqueeze(0)),
                self._mailbox_c,
                _as_cute(shared_shard),
                cutlass.Int64(m),
                cutlass.Int64(
                    self._mailbox_multicast_ptr + self.rank * self.shard_dim * 2
                ),
                stream,
            )
        return self._mailbox
