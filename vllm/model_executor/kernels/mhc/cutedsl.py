# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class HCPrenormGemm:
    """Warp-specialized SM100 TF32 GEMM with fused BF16 prenormalization.

    Warp 0 stages A and B with TMA, warps 4-7 convert A to TF32 in TMEM while
    accumulating row square sums, and warp 1 issues UMMA. Warps 0-3 move the
    accumulator through registers and shared memory to a TMA store; split-K
    CTAs produce independent GEMM and square-sum partials.
    """

    def __init__(self, k: int, num_splits: int):
        self.mma_tiler = (64, 32, 64)
        self.num_load_stages = 12
        self.num_transform_stages = 2
        self.num_splits = num_splits
        self.tiles_per_split, self.extra_tiles = divmod(
            k // self.mma_tiler[2], num_splits
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        d: cute.Tensor,
        sqr_sum: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.num_splits == 1):
            d_mnl = cute.make_tensor(
                d.iterator,
                cute.make_layout(
                    (d.shape[0], d.shape[1], 1),
                    stride=(d.stride[0], d.stride[1], 0),
                ),
            )
            sqr_sum_ms = cute.make_tensor(
                sqr_sum.iterator,
                cute.make_layout((sqr_sum.shape[0], 1), stride=(sqr_sum.stride[0], 0)),
            )
        else:
            d_mnl = cute.make_tensor(
                d.iterator,
                cute.make_layout(
                    (d.shape[1], d.shape[2], d.shape[0]),
                    stride=(d.stride[1], d.stride[2], d.stride[0]),
                ),
            )
            sqr_sum_ms = cute.make_tensor(
                sqr_sum.iterator,
                cute.make_layout(
                    (sqr_sum.shape[1], sqr_sum.shape[0]),
                    stride=(sqr_sum.stride[1], sqr_sum.stride[0]),
                ),
            )

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.TFloat32,
            cutlass.TFloat32,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            cutlass.BFloat16,
            self.num_load_stages,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            cutlass.TFloat32,
            self.num_load_stages,
        )
        a_tmem_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            cutlass.TFloat32,
            self.num_transform_stages,
        )

        self.d_layout = utils.LayoutEnum.from_tensor(d_mnl)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.mma_tiler, False, self.d_layout, cutlass.Float32
        )
        d_smem_layout = sm100_utils.make_smem_layout_epi(
            cutlass.Float32, self.d_layout, epi_tile, 1
        )

        cluster_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            load_op,
            a,
            cute.slice_(a_smem_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            load_op,
            b,
            cute.slice_(b_smem_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            d_mnl,
            cute.slice_(d_smem_layout, (None, None, 0)),
            epi_tile,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        transformed_a_fake = tiled_mma.make_fragment_A(a_tmem_smem_layout.outer)
        self.num_tmem_cols = utils.get_num_tmem_alloc_cols(
            [acc_fake, transformed_a_fake]
        )
        assert self.num_tmem_cols == 256

        @cute.struct
        class SharedStorage:
            tmem_holding_buf: cutlass.Int32
            a_full: cute.struct.MemRange[cutlass.Int64, self.num_load_stages]
            a_empty: cute.struct.MemRange[cutlass.Int64, self.num_load_stages]
            b_full: cute.struct.MemRange[cutlass.Int64, self.num_load_stages]
            b_empty: cute.struct.MemRange[cutlass.Int64, self.num_load_stages]
            transform_full: cute.struct.MemRange[
                cutlass.Int64, self.num_transform_stages
            ]
            transform_empty: cute.struct.MemRange[
                cutlass.Int64, self.num_transform_stages
            ]
            acc_full: cute.struct.MemRange[cutlass.Int64, 1]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 1]
            smem_d: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(d_smem_layout.outer)],
                1024,
            ]
            smem_a: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(a_smem_layout.outer)
                ],
                1024,
            ]
            smem_b: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.TFloat32, cute.cosize(b_smem_layout.outer)
                ],
                1024,
            ]

        assert SharedStorage.size_in_bytes() <= 232448  # type: ignore[attr-defined]
        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            sqr_sum_ms,
            cluster_layout_vmnk,
            a_smem_layout,
            b_smem_layout,
            d_smem_layout,
            epi_tile,
            acc_fake.layout,
            transformed_a_fake.layout,
            self.mma_tiler,
            self.num_load_stages,
            self.num_transform_stages,
            self.num_splits,
            self.tiles_per_split,
            self.extra_tiles,
        ).launch(
            grid=(
                cute.ceil_div(a.shape[0], self.mma_tiler[0]) * self.num_splits,
                1,
                1,
            ),
            block=(256, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        mS_ms: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        d_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        acc_tmem_layout: cute.Layout,
        a_tmem_layout: cute.Layout,
        mma_tiler: cutlass.Constexpr,
        num_load_stages: cutlass.Constexpr,
        num_transform_stages: cutlass.Constexpr,
        num_splits: cutlass.Constexpr,
        tiles_per_split: cutlass.Constexpr,
        extra_tiles: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()

        m_block = bidx // num_splits
        split = bidx % num_splits
        tile_begin = split * tiles_per_split + min(split, extra_tiles)
        tile_count = tiles_per_split + (split < extra_tiles)

        # Shared storage and pipelines
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.smem_a.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.smem_b.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = storage.smem_d.get_tensor(d_smem_layout.outer, swizzle=d_smem_layout.inner)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_d)

        thread = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        transform_tidx = tidx % 128
        a_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_full.data_ptr(),
            num_stages=num_load_stages,
            producer_group=thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=mma_tiler[0] * mma_tiler[2] * 2,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_tidx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_full.data_ptr(),
            num_stages=num_load_stages,
            producer_group=thread,
            consumer_group=thread,
            tx_count=mma_tiler[1] * mma_tiler[2] * 4,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )
        transform_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.transform_full.data_ptr(),
            num_stages=num_transform_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=thread,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full.data_ptr(),
            num_stages=1,
            producer_group=thread,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        gA = cute.local_tile(mA_mk, (mma_tiler[0], mma_tiler[2]), (m_block, None))
        gB = cute.local_tile(mB_nk, (mma_tiler[1], mma_tiler[2]), (0, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        a_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_load_stages
        )
        a_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_load_stages
        )
        b_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_load_stages
        )
        b_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_load_stages
        )
        transform_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_transform_stages
        )
        transform_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_transform_stages
        )
        acc_producer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 1
        )
        acc_consumer = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(2, 256),
            allocator_warp_id=2,
            is_two_cta=False,
        )
        pool = tmem.reserve(self.num_tmem_cols)
        accumulators = pool.allocate_tensor(acc_tmem_layout, cutlass.Float32)
        transformed_a = pool.allocate_tensor(a_tmem_layout, cutlass.TFloat32)
        tmem.relinquish_alloc_permit()

        # TMA load warp
        if warp_idx == 0:
            for tile in cutlass.range(0, tile_count, 1, unroll=1):
                a_pipeline.producer_acquire(a_producer)
                b_pipeline.producer_acquire(b_producer)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile_begin + tile)],
                    tAsA[(None, a_producer.index)],
                    tma_bar_ptr=a_pipeline.producer_get_barrier(a_producer),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, tile_begin + tile)],
                    tBsB[(None, b_producer.index)],
                    tma_bar_ptr=b_pipeline.producer_get_barrier(b_producer),
                )
                a_pipeline.producer_commit(a_producer)
                b_pipeline.producer_commit(b_producer)
                a_producer.advance()
                b_producer.advance()
            a_pipeline.producer_tail(a_producer)
            b_pipeline.producer_tail(b_producer)

        # A transform and row-sum warp group
        if warp_idx >= 4:
            a_partition_shape = tiled_mma.partition_shape_A(
                cute.dice(mma_tiler, (1, None, 1))
            )
            copy_atom = mixed_input_utils.get_copy_atom_a_transform(
                cutlass.TFloat32,
                False,
                tcgen05.OperandSource.TMEM,
                a_partition_shape,
                cutlass.BFloat16,
            )
            _, dst_copy, tAsA_input, tAsA_transform = (
                mixed_input_utils.transform_partition(
                    tcgen05.OperandSource.TMEM,
                    mixed_input_utils.TransformMode.ConvertOnly,
                    None,
                    copy_atom,
                    sA,
                    transformed_a,
                    transform_tidx,
                )
            )
            tArA = cute.make_rmem_tensor(
                tAsA_input[(None, None, None, None, 0)].shape,
                cutlass.BFloat16,
            )
            tArA_transform = cute.make_rmem_tensor(tArA.shape, cutlass.TFloat32)
            transform_tiler = cute.make_layout(
                min(cute.size(cute.coalesce(tAsA_input.layout), mode=[0]), 64)
            )
            tArA_load = cute.flat_divide(tArA, transform_tiler)
            tArA_load = cute.group_modes(tArA_load, 1, cute.rank(tArA_load))
            tArA_store = cute.flat_divide(tArA_transform, transform_tiler)
            tArA_store = cute.group_modes(tArA_store, 1, cute.rank(tArA_store))

            sums = cute.make_rmem_tensor((2, 2), cutlass.Float32)
            sums.fill(0.0)
            transform_barrier = pipeline.NamedBarrier(3, 128)
            for _ in cutlass.range(0, tile_count, 1, unroll=1):
                a_pipeline.consumer_wait(a_consumer)
                transform_pipeline.producer_acquire(transform_producer)
                tAsA_stage = tAsA_input[(None, None, None, None, a_consumer.index)]
                tAsA_stage = cute.flat_divide(tAsA_stage, transform_tiler)
                tAsA_stage = cute.group_modes(tAsA_stage, 1, cute.rank(tAsA_stage))
                for i in cutlass.range_constexpr(cute.size(tArA_load, mode=[1])):
                    cute.autovec_copy(tAsA_stage[(None, i)], tArA_load[(None, i)])
                    x = tArA_load[(None, i)].load().to(cutlass.Float32)
                    u = i % 2
                    sums[u, 0], sums[u, 1] = cute.arch.fma_packed_f32x2(
                        (x[0], x[1]),
                        (x[0], x[1]),
                        (sums[u, 0], sums[u, 1]),
                    )
                    tArA_store[(None, i)].store(x.to(cutlass.TFloat32))

                mixed_input_utils.store_transformed_a(
                    tArA_transform,
                    tAsA_transform[(None, None, None, None, transform_producer.index)],
                    dst_copy,
                )
                transform_barrier.arrive_and_wait()
                cute.arch.fence_view_async_tmem_store()
                a_pipeline.consumer_release(a_consumer)
                a_consumer.advance()
                transform_pipeline.producer_commit(transform_producer)
                transform_producer.advance()

            transform_pipeline.producer_tail(transform_producer)
            sqr_sum_tiler = (mma_tiler[0], 1)
            gS = cute.local_tile(mS_ms, sqr_sum_tiler, (m_block, split))
            cS = cute.local_tile(
                cute.make_identity_tensor(mS_ms.shape),
                sqr_sum_tiler,
                (m_block, split),
            )
            row_layout = cute.make_layout(((4, 8, 4), 2), stride=((0, 1, 16), 8))
            for u in cutlass.range_constexpr(2):
                reduced = cute.arch.warp_reduction_sum(
                    sums[u, 0] + sums[u, 1], threads_in_group=4
                )
                local_row = row_layout((transform_tidx, u))
                if lane_idx % 4 == 0 and cute.elem_less(
                    cS[(local_row, 0)], mS_ms.shape
                ):
                    gS[(local_row, 0)] = reduced

        # UMMA warp
        if warp_idx == 1:
            tCrB = tiled_mma.make_fragment_B(sB)
            accumulator = accumulators[(None, None, None, acc_producer.index)]
            acc_pipeline.producer_acquire(acc_producer)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _ in cutlass.range(0, tile_count, 1, unroll=1):
                transform_pipeline.consumer_wait(transform_consumer)
                b_pipeline.consumer_wait(b_consumer)
                for k_block in cutlass.range_constexpr(
                    cute.size(transformed_a, mode=[2])
                ):
                    cute.gemm(
                        tiled_mma,
                        accumulator,
                        transformed_a[(None, None, k_block, transform_consumer.index)],
                        tCrB[(None, None, k_block, b_consumer.index)],
                        accumulator,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                transform_pipeline.consumer_release(transform_consumer)
                b_pipeline.consumer_release(b_consumer)
                transform_consumer.advance()
                b_consumer.advance()
            acc_pipeline.producer_commit(acc_producer)
            acc_producer.advance()

        gD_mnl = cute.local_tile(
            mD_mnl,
            cute.slice_(mma_tiler, (None, None, 0)),
            (None, None, None),
        )
        tCgD = thr_mma.partition_C(gD_mnl)

        # Epilogue warp group: TMEM -> registers -> shared -> global
        if warp_idx < 4:
            d_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            )
            tiled_t2r, tTR_tAcc, tTR_rAcc = (
                mixed_input_utils.epilog_tmem_copy_and_partition(
                    mma_tiler,
                    self.d_layout,
                    cutlass.Float32,
                    cutlass.Float32,
                    tidx,
                    accumulators,
                    tCgD,
                    epi_tile,
                    False,
                )
            )
            tTR_rD = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)
            tiled_r2s, tRS_rD, tRS_sD = (
                mixed_input_utils.epilog_smem_copy_and_partition(
                    self.d_layout,
                    cutlass.Float32,
                    cutlass.Float32,
                    tiled_t2r,
                    tTR_rD,
                    tidx,
                    sD,
                )
            )
            _, bSG_sD, bSG_gD, _, _ = mixed_input_utils.epilog_gmem_copy_and_partition(
                cutlass.Float32,
                tidx,
                tma_atom_d,
                None,
                tCgD,
                None,
                epi_tile,
                sD,
            )
            bSG_gD = bSG_gD[(None, None, None, m_block, 0, split)]
            bSG_gD = cute.group_modes(bSG_gD, 1, cute.rank(bSG_gD))

            acc_pipeline.consumer_wait(acc_consumer)
            tTR_tAcc = tTR_tAcc[(None, None, None, None, None, acc_consumer.index)]
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            cute.copy(
                tiled_t2r,
                tTR_tAcc[(None, None, None, 0)],
                tTR_rAcc,
            )
            cute.arch.fence_view_async_tmem_load()
            tRS_rD.store(tiled_r2s.retile(tTR_rAcc).load())
            cute.copy(
                tiled_r2s,
                tRS_rD,
                tRS_sD[(None, None, None, 0)],
            )
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_consumer)
            acc_consumer.advance()

            cute.arch.fence_view_async_shared()
            pipeline.NamedBarrier(1, 128).arrive_and_wait()
            if warp_idx == 0:
                cute.copy(tma_atom_d, bSG_sD[(None, 0)], bSG_gD[(None, 0)])
                d_pipeline.producer_commit()

            tmem.free(pool.base_ptr)
            if warp_idx == 0:
                d_pipeline.producer_tail()

        if warp_idx == 1:
            acc_pipeline.producer_tail(acc_producer)


@cache
def _compile(k, num_splits):
    m = cute.sym_int()
    a = make_fake_compact_tensor(
        cutlass.BFloat16,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    b = make_fake_compact_tensor(
        cutlass.TFloat32,
        (24, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    if num_splits == 1:
        d = make_fake_compact_tensor(
            cutlass.Float32,
            (m, 24),
            stride_order=(1, 0),
            assumed_align=16,
        )
        sqr_sum = make_fake_compact_tensor(cutlass.Float32, (m,), assumed_align=16)
    else:
        d = make_fake_compact_tensor(
            cutlass.Float32,
            (num_splits, m, 24),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        sqr_sum = make_fake_compact_tensor(
            cutlass.Float32,
            (num_splits, m),
            stride_order=(1, 0),
            assumed_align=16,
        )
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        HCPrenormGemm(k, num_splits),
        a,
        b,
        d,
        sqr_sum,
        stream,
        options="--enable-tvm-ffi",
    )


def can_use_hc_prenorm_gemm(a, b, num_splits):
    return (
        a.ndim == 2
        and b.ndim == 2
        and b.shape[0] == 24
        and a.shape[1] == b.shape[1]
        and a.shape[1] in (5120, 7168, 7680, 16384, 28672)
        and num_splits in (1, 16)
    )


def hc_prenorm_gemm(a, b, d, sqr_sum, num_splits):
    num_splits = num_splits or 1
    assert (
        a.dtype == torch.bfloat16
        and b.dtype == d.dtype == sqr_sum.dtype == torch.float32
    )
    assert (
        a.is_contiguous()
        and b.is_contiguous()
        and d.is_contiguous()
        and sqr_sum.is_contiguous()
    )
    assert can_use_hc_prenorm_gemm(a, b, num_splits)

    _compile(a.shape[1], num_splits)(a, b, d, sqr_sum)


def run_hc_prenorm_gemm(a, b, out, sqr_sum, num_splits):
    if num_splits == 1:
        hc_prenorm_gemm(a, b, out[0], sqr_sum[0], num_splits)
    else:
        hc_prenorm_gemm(a, b, out, sqr_sum, num_splits)
