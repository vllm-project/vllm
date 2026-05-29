# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL split-K A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

Warp-specialized kernel with cluster split-K reduction for low-latency
router GEMM on tall-K, narrow-M problems (e.g. K=14400, N=256, M<=16).
"""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    i32 = _ir.IntegerType.get_signless(32)
    _llvm.inline_asm(
        i32,
        [],
        "barrier.cluster.arrive; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    i32 = _ir.IntegerType.get_signless(32)
    _llvm.inline_asm(
        i32,
        [],
        "barrier.cluster.wait; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def set_block_rank(smem_ptr, peer_rank, *, loc=None, ip=None):
    i32 = _ir.IntegerType.get_signless(32)
    ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    rank_ir = peer_rank.ir_value(loc=loc, ip=ip)
    res = _llvm.inline_asm(
        i32,
        [ptr_i32, rank_ir],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(res)


@dsl_user_op
def st_shared_remote_f32(remote_addr, val, *, loc=None, ip=None):
    i32 = _ir.IntegerType.get_signless(32)
    addr_ir = remote_addr.ir_value(loc=loc, ip=ip)
    val_ir = val.ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        i32,
        [addr_ir, val_ir],
        "st.shared::cluster.f32 [$0], $1; mov.u32 $2, 0;",
        "r,f,=r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
class LLRouterSplitK:
    """Warp-specialized low-latency A GEMM with cluster split-K reduction."""

    def __init__(
        self,
        ab_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        out_dtype=cutlass.Float32,
        tile_n: int = 16,
        tile_k: int = 256,
        num_stages: int = 2,
        num_dma_warps: int = 4,
        split_k: int = 8,
        *,
        loc=None,
        ip=None,
    ):
        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.tile_m = 16
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.split_k = split_k
        self.mma_shape = (16, 8, 16)
        self.atom_layout = (1, 1, 1)
        self.num_mma_warps = 4
        self.num_dma_threads = num_dma_warps * 32
        self.num_mma_threads = self.num_mma_warps * 32
        self.num_threads = self.num_dma_threads + self.num_mma_threads

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        major_size = min(smem_tiler[1], 64)
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        copy_elems = copy_bits // dtype.width
        k_threads = cute.size(self.tile_k) // copy_elems
        thread_layout = cute.make_layout(
            (num_threads // k_threads, k_threads), stride=(k_threads, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def call_splitk(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: CUstream,
        scale: float = 1.0,
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, copy_bits, (bM, bK, self.num_stages)
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, copy_bits, (bN, bK, self.num_stages)
        )
        atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy(
            atom_g2s, mA.element_type, copy_bits, self.num_dma_threads
        )
        tiled_copy_B = self._make_gmem_tiled_copy(
            atom_g2s, mB.element_type, copy_bits, self.num_dma_threads
        )
        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_shape)
        perm_mnk = (
            self.atom_layout[0] * self.mma_shape[0],
            self.atom_layout[1] * self.mma_shape[1] * (self.tile_n // 8),
            self.atom_layout[2] * self.mma_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk
        )
        grid_m = cute.ceil_div(cute.size(mC, mode=[0]), bM)
        grid_n = cute.ceil_div(cute.size(mC, mode=[1]), bN)
        self.kernel_splitk(
            mA,
            mB,
            mC,
            scale,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=[cute.size(grid_m), cute.size(grid_n), self.split_k],
            block=[self.num_threads, 1, 1],
            cluster=[1, 1, self.split_k],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel_splitk(
        self,
        mA,
        mB,
        mC,
        scale: cutlass.Float32,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_z = cute.arch.block_idx()
        warp_idx = tidx // 32
        is_dma = warp_idx < (self.num_dma_threads // 32)
        dma_tidx = tidx
        mma_tidx = tidx - self.num_dma_threads
        N_out = cute.size(mC, mode=[1])
        M_out = cute.size(mC, mode=[0])

        cta_tiler = (bM, bN, bK)
        coord = (bid_m, bid_n, None)
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=coord, proj=(1, 1, None))
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        cA = cute.local_tile(mcA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16
            ]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, num_stages * 2], 8
            ]

        smem = cutlass.utils.SmemAllocator()
        storage_ptr = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)  # type: ignore[attr-defined]
        storage = SharedStorage(storage_ptr)  # type: ignore[call-arg]
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)

        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_dma_threads
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_threads
        )
        mainloop_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
        )

        K_total = cute.size(mA, mode=[1])
        k_tile_count_full = (K_total + bK - 1) // bK
        tiles_per_split = k_tile_count_full // self.split_k
        extra = k_tile_count_full - tiles_per_split * self.split_k
        k_start = bid_z * tiles_per_split + (bid_z if bid_z < extra else extra)
        my_tiles = tiles_per_split + (1 if bid_z < extra else 0)

        if is_dma:
            cute.arch.setmaxregister_decrease(40)
            thr_A = tiled_copy_A.get_slice(dma_tidx)
            thr_B = tiled_copy_B.get_slice(dma_tidx)
            tAgA = thr_A.partition_S(gA)
            tAsA = thr_A.partition_D(sA)
            tBgB = thr_B.partition_S(gB)
            tBsB = thr_B.partition_D(sB)
            tAcA = thr_A.partition_S(cA)
            tBcB = thr_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAgA.shape[0][1],
                        cute.size(tAgA, mode=[1]),
                        cute.size(tAgA, mode=[2]),
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rv in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rv, m, 0] = cute.elem_less(
                        tAcA[(0, rv), m, 0, 0][0], mA.shape[0]
                    )

            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tBgB.shape[0][1],
                        cute.size(tBgB, mode=[1]),
                        cute.size(tBgB, mode=[2]),
                    ),
                    stride=(cute.size(tBgB, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rv in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rv, n, 0] = cute.elem_less(
                        tBcB[(0, rv), n, 0, 0][0], mB.shape[0]
                    )

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_stages
            )

            mainloop_pipeline.producer_acquire(producer_state)
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_start],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )
            cute.arch.griddepcontrol_wait()
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_start],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

            # K-predicate helpers for partial last tile
            k_elems_per_copy = 8
            k_threads = bK // k_elems_per_copy
            k_tidx = dma_tidx % k_threads
            k_offset_in_tile = k_tidx * k_elems_per_copy

            for local_k in range(1, my_tiles):
                k_tile = k_start + local_k
                k_base_global = k_tile * bK + k_offset_in_tile
                for rv in range(tApA.shape[0]):
                    k_valid = (k_base_global + rv) < K_total
                    for m in range(tApA.shape[1]):
                        tApA[rv, m, 0] = (
                            cute.elem_less(tAcA[(0, rv), m, 0, 0][0], mA.shape[0])
                            & k_valid
                        )
                    for n in range(tBpB.shape[1]):
                        tBpB[rv, n, 0] = (
                            cute.elem_less(tBcB[(0, rv), n, 0, 0][0], mB.shape[0])
                            & k_valid
                        )
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile],
                    tAsA[None, None, None, producer_state.index],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile],
                    tBsB[None, None, None, producer_state.index],
                    pred=tBpB,
                )
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            mainloop_pipeline.producer_tail(producer_state)

        else:
            # MMA warps with k-phase interleaving
            cute.arch.setmaxregister_increase(232)
            lane_id = mma_tidx % 32
            mma_warp_idx = mma_tidx // 32
            NUM_MMA_WARPS: cutlass.Constexpr = self.num_mma_warps

            thr_mma = tiled_mma.get_slice(lane_id)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type
            )
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mB.element_type
            )
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane_id)
            thr_s2r_B = tiled_s2r_B.get_slice(lane_id)
            tCsA_v = thr_s2r_A.partition_S(sA)
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCsB_v = thr_s2r_B.partition_S(sB)
            tCrB_v = thr_s2r_B.retile(tCrB)

            num_k_block = cute.size(tCrA, mode=[2])
            K_PER_WARP: cutlass.Constexpr = num_k_block // NUM_MMA_WARPS

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages
            )

            for local_k in range(my_tiles):
                mainloop_pipeline.consumer_wait(consumer_state)
                tCsA_p = tCsA_v[None, None, None, consumer_state.index]
                tCsB_p = tCsB_v[None, None, None, consumer_state.index]
                for ki in cutlass.range(K_PER_WARP, unroll_full=True):
                    k_block = ki * NUM_MMA_WARPS + mma_warp_idx
                    cute.copy(
                        tiled_s2r_A, tCsA_p[None, None, k_block], tCrA_v[None, None, 0]
                    )
                    cute.copy(
                        tiled_s2r_B, tCsB_p[None, None, k_block], tCrB_v[None, None, 0]
                    )
                    cute.gemm(
                        tiled_mma, tCrC, tCrA[None, None, 0], tCrB[None, None, 0], tCrC
                    )
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # === CLUSTER REDUCTION EPILOGUE ===
            # Reduce MMA warps within this CTA
            smem_red_ptr = cute.arch.alloc_smem(
                cutlass.Float32, bM * bN * NUM_MMA_WARPS, alignment=16
            )
            smem_warp = cute.make_tensor(
                smem_red_ptr + mma_warp_idx * bM * bN,
                cute.make_layout((bM, bN), stride=(bN, 1)),
            )
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)
            cute.arch.sync_threads()

            num_elems: cutlass.Constexpr = bM * bN
            elems_per_thread: cutlass.Constexpr = num_elems // self.num_mma_threads

            partial_buf = cute.arch.alloc_smem(
                cutlass.Float32, bM * bN * self.split_k, alignment=16
            )
            cta_rank = cute.arch.block_idx_in_cluster()

            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * self.num_mma_threads + mma_tidx
                m2 = idx // bN
                n2 = idx % bN
                gm2 = bid_m * bM + m2
                gn2 = bid_n * bN + n2
                total = cutlass.Float32(0.0)
                if gm2 < M_out:  # noqa: SIM102
                    if gn2 < N_out:
                        for w in cutlass.range_constexpr(NUM_MMA_WARPS):
                            p = smem_red_ptr + w * bM * bN + idx
                            t = cute.make_tensor(p, cute.make_layout((1,)))
                            r = cute.make_rmem_tensor((1,), cutlass.Float32)
                            cute.autovec_copy(t, r)
                            total = total + r[0]
                        total = total * scale
                pb_p = partial_buf + cta_rank * num_elems + idx
                pb_t = cute.make_tensor(pb_p, cute.make_layout((1,)))
                pb_r = cute.make_rmem_tensor((1,), cutlass.Float32)
                pb_r[0] = total
                cute.autovec_copy(pb_r, pb_t)

            cute.arch.sync_threads()

            # Send partials to all peer CTAs via st.shared::cluster
            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * self.num_mma_threads + mma_tidx
                my_slot = partial_buf + cta_rank * num_elems + idx
                my_val_t = cute.make_tensor(my_slot, cute.make_layout((1,)))
                my_val_r = cute.make_rmem_tensor((1,), cutlass.Float32)
                cute.autovec_copy(my_val_t, my_val_r)
                for peer in cutlass.range_constexpr(self.split_k):
                    remote = set_block_rank(my_slot, cutlass.Int32(peer))
                    st_shared_remote_f32(remote, my_val_r[0])

            cluster_arrive_relaxed()
            cluster_wait()

            if mma_tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()
            cute.arch.sync_threads()

            # Local reduction + global output
            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * self.num_mma_threads + mma_tidx
                m = idx // bN
                n = idx % bN
                global_m = bid_m * bM + m
                global_n = bid_n * bN + n
                if global_m < M_out:  # noqa: SIM102
                    if global_n < N_out:
                        acc = cutlass.Float32(0.0)
                        for sk in cutlass.range_constexpr(self.split_k):
                            cb_p = partial_buf + sk * num_elems + idx
                            cb_t = cute.make_tensor(cb_p, cute.make_layout((1,)))
                            cb_r = cute.make_rmem_tensor((1,), cutlass.Float32)
                            cute.autovec_copy(cb_t, cb_r)
                            acc = acc + cb_r[0]
                        out_p = (mC.iterator + global_m * N_out + global_n).align(2)
                        out_t = cute.make_tensor(out_p, cute.make_layout((1,)))
                        out_r = cute.make_rmem_tensor((1,), self.out_dtype)
                        out_r[0] = acc.to(self.out_dtype)
                        out_t[0] = out_r[0]

        cute.arch.sync_threads()
