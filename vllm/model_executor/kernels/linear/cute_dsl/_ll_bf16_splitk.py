# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

Warp-specialized kernel with cluster split-K reduction.
"""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline


# Takes a shared memory address in this CTA and maps it to the same offset in a peer CTA.
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


# Writes an FP32 value to a REMOTE CTA's shared memory via the mapped address from set_block_rank.
# This goes through the cluster's DSMEM interconnect, not through HBM.
# This reduction needs a plain DSMEM store, not an mbarrier-complete helper.
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
        has_side_effects=True,  # memory store, must not be eliminated or reordered
        loc=loc,
        ip=ip,
    )


def _fill_pred(pred_flat, coord_tensor, k_tile, dim_limit, K_total):
    """Fills predicates from the CuTe coordinate tile selected by k_tile."""
    coord_ktile = coord_tensor[None, None, 0, k_tile]
    num_vec = pred_flat.shape[0]  # = 8 - elements per 128-bit copy
    num_mn = pred_flat.shape[1]  # = 4 - M-rows (or N-rows) this thread handles per tile
    for v in range(num_vec):
        for j in range(num_mn):
            pred_flat[v, j] = cute.elem_less(
                coord_ktile[(0, v), j], (dim_limit, K_total)
            )


def _make_pred(tXcX, k_tile, dim_limit, K_total):
    """Builds a predicate tensor from a selected coordinate tile."""
    num_vec = tXcX.shape[0][1]
    num_mn = cute.size(tXcX, mode=[1])
    pred_flat = cute.make_rmem_tensor(
        cute.make_layout((num_vec, num_mn), stride=(num_mn, 1)),
        cutlass.Boolean,
    )
    _fill_pred(pred_flat, tXcX, k_tile, dim_limit, K_total)
    pred = cute.make_tensor(
        pred_flat.iterator,
        cute.make_layout(
            (num_vec, num_mn, cute.size(tXcX, mode=[2])),
            stride=(num_mn, 1, 0),
        ),
    )
    return pred_flat, pred


@dsl_user_op
class LLBf16SplitK:
    """Warp specialization split: half DMA, half MMA."""

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
        self.mma_shape = (16, 8, 16)  # mma.sync.aligned.m16n8k16
        self.atom_layout = (1, 1, 1)  # No tiling of the MMA atom -- one atom per warp
        self.num_mma_warps = 4
        self.num_dma_threads = num_dma_warps * 32  # 128 threads
        self.num_mma_threads = self.num_mma_warps * 32  # 128 threads
        self.num_threads = (
            self.num_dma_threads + self.num_mma_threads
        )  # total per CTA: 256

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        major_size = min(smem_tiler[1], 64)
        # Maximum swizzle for 128-bit copies - Swizzle XORs the lower address bits to spread
        # bank accesses
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(
            swizzle_bits, 3
        )  # Capped at 3: 64 * 16 / 128 = 8, log2(8) = 3
        # Creates an 8-row x 64-column atom with swizzle, then tiles it to cover the full
        # (M_or_N, K, num_stages) shape. The third dimension (index 2) is the pipeline stage index.
        layout_atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        copy_elems = copy_bits // dtype.width  # 128/16 = 8 elems per copy
        k_threads = cute.size(self.tile_k) // copy_elems  # 256/8 = 32 threads along K
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
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k  # 16, 128, 256
        copy_bits = 128  # TODO (roberto): try 256-bit instructions
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
        )  # cp.async instruction -> async GMEM -> SMEM copy bypassing L1
        tiled_copy_A = self._make_gmem_tiled_copy(
            atom_g2s, mA.element_type, copy_bits, self.num_dma_threads
        )
        tiled_copy_B = self._make_gmem_tiled_copy(
            atom_g2s, mB.element_type, copy_bits, self.num_dma_threads
        )
        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_shape)
        perm_mnk = (
            self.atom_layout[0] * self.mma_shape[0],  # 1 * 16 = 16
            self.atom_layout[1]
            * self.mma_shape[1]
            * (self.tile_n // 8),  # 1 * 8 * 2 = 16
            self.atom_layout[2] * self.mma_shape[2],  # 1 * 16 = 16
        )
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk
        )
        # CTAs within a cluster are hardware-guaranteed to be co-resident on adjacent SMs
        # and can access each other's shared memory via DSMEM.
        grid_m = cute.ceil_div(cute.size(mC, mode=[0]), bM)  # ceil(M/16)
        grid_n = cute.ceil_div(cute.size(mC, mode=[1]), bN)  # ceil(N/16)
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
            grid=[
                cute.size(grid_m),
                cute.size(grid_n),
                self.split_k,
            ],  # [1, 16, 8] = 128 CTAs
            block=[self.num_threads, 1, 1],  # [256, 1, 1]
            cluster=[
                1,
                1,
                self.split_k,
            ],  # [1, 1, 8] -- the 8 CTAs along Z form one cluster
            stream=stream,
            use_pdl=False,
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
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k  # 16, 16, 256
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_z = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        is_dma = warp_idx < (self.num_dma_threads // 32)  # warps 0-3 = DMA
        dma_tidx = tidx  # DMA warps use raw thread index (0-127)
        mma_tidx = tidx - self.num_dma_threads  # MMA warps use offset index (0-127)

        cta_tiler = (bM, bN, bK)  # (16, 16, 256)
        coord = (bid_m, bid_n, None)  # None = all K tiles
        # extracts this CTA portion of the matrix
        gA = cute.local_tile(
            mA, tiler=cta_tiler, coord=coord, proj=(1, None, 1)
        )  # skip N
        gB = cute.local_tile(
            mB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1)
        )  # skip M
        gC = cute.local_tile(
            mC, tiler=cta_tiler, coord=coord, proj=(1, 1, None)
        )  # skip K
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        mcC = cute.make_identity_tensor(mC.layout.shape)
        cA = cute.local_tile(mcA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        cC = cute.local_tile(mcC, tiler=cta_tiler, coord=coord, proj=(1, 1, None))

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16
            ]  # cosize of the swizzled layout, 16-byte aligned
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16
            ]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, num_stages * 2], 8
            ]  # pipeline barriers

        # Dynamic shared memory allocation
        smem = cutlass.utils.SmemAllocator()
        storage_ptr = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)  # type: ignore[attr-defined]
        storage = SharedStorage(storage_ptr)  # type: ignore[call-arg]
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)

        # Creates the async pipeline
        # - Producers (DMA warps) call acquire -> copy -> commit -> advance
        # - Consumers (MMA warps) call wait -> compute -> release -> advance
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

        # Split-K work distribution across 8 CTAs as evenly as possible.
        # Mirrors FlashInfer split-KV: split z handles tiles z, z+split_k, ...
        # For K=7168, 28 K-tiles split as [4, 4, 4, 4, 3, 3, 3, 3].
        K_total = cute.size(mA, mode=[1])
        k_tile_count = cute.ceil_div(K_total, bK)
        k_start = bid_z
        my_tiles = cute.ceil_div(k_tile_count - k_start, self.split_k)

        if is_dma:
            # DMA warps only do memory copies -- they need very few registers
            cute.arch.setmaxregister_decrease(40)
            # Get slice of tiled copy
            thr_A = tiled_copy_A.get_slice(dma_tidx)
            thr_B = tiled_copy_B.get_slice(dma_tidx)
            tAgA = thr_A.partition_S(gA)  # thread's source (global) partition
            tAsA = thr_A.partition_D(sA)  # thread's destination (shared) partition
            tBgB = thr_B.partition_S(gB)
            tBsB = thr_B.partition_D(sB)
            tAcA = thr_A.partition_S(cA)
            tBcB = thr_B.partition_S(cB)

            # Build M/K and N/K predicates, broadcast across K-tile copies.
            tApA_flat, tApA = _make_pred(
                tAcA, k_start, mA.shape[0], K_total
            )
            tBpB_flat, tBpB = _make_pred(
                tBcB, k_start, mB.shape[0], K_total
            )

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_stages
            )

            # First tile load - Acquire first pipeline stage (stage 0)
            mainloop_pipeline.producer_acquire(producer_state)
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_start],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )  # cp.async copy of B tile from GMEM -> SMEM
            #cute.arch.griddepcontrol_wait()  # PDL-wait
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_start],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

            for k_tile in cutlass.range(
                k_start + self.split_k, k_tile_count, self.split_k, unroll=1
            ):
                _fill_pred(tApA_flat, tAcA, k_tile, mA.shape[0], K_total)
                _fill_pred(tBpB_flat, tBcB, k_tile, mB.shape[0], K_total)
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
            cute.arch.setmaxregister_increase(232) # tCrC: 16×16 FP32 = 256 floats = 1024 bytes + plus A/B register fragments for ldmatrix
            
            lane_id = cute.arch.lane_idx()
            mma_warp_idx = warp_idx - (self.num_dma_threads // 32)
            NUM_MMA_WARPS: cutlass.Constexpr = self.num_mma_warps

            thr_mma = tiled_mma.get_slice(lane_id)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)
            
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # loads 8×8 matrices of 16-bit elements from SMEM -> RF
            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type
            ) # False = not transposed. 4 = load 4 matrices per instruction
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mB.element_type
            )

            # Creates SMEM -> RF copy path
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane_id)
            thr_s2r_B = tiled_s2r_B.get_slice(lane_id)
            tCsA_v = thr_s2r_A.partition_S(sA) # _v = views of same underlying tensor, not copies
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCsB_v = thr_s2r_B.partition_S(sB)
            tCrB_v = thr_s2r_B.retile(tCrB)
            tCsA_warp_v = cute.logical_divide(
                tCsA_v, (None, None, NUM_MMA_WARPS, None)
            )
            tCsB_warp_v = cute.logical_divide(
                tCsB_v, (None, None, NUM_MMA_WARPS, None)
            )

            num_k_block = cute.size(tCrA, mode=[2])
            K_PER_WARP: cutlass.Constexpr = num_k_block // NUM_MMA_WARPS # Each warp handles 4 K-blocks per tile

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages
            )

            for local_k in range(my_tiles):
                mainloop_pipeline.consumer_wait(consumer_state)
                for ki in cutlass.range(K_PER_WARP, unroll_full=True): # unroll_full=True unrolls all 4 iterations
                    cute.copy(
                        tiled_s2r_A,
                        tCsA_warp_v[
                            None, None, (mma_warp_idx, ki), consumer_state.index
                        ],
                        tCrA_v[None, None, 0],
                    ) # ldmatrix
                    cute.copy(
                        tiled_s2r_B,
                        tCsB_warp_v[
                            None, None, (mma_warp_idx, ki), consumer_state.index
                        ],
                        tCrB_v[None, None, 0],
                    )
                    cute.gemm(
                        tiled_mma, tCrC, tCrA[None, None, 0], tCrB[None, None, 0], tCrC
                    ) # mma.sync
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # === CLUSTER REDUCTION EPILOGUE ===
            # Reduce MMA warps within this CTA
            num_elems: cutlass.Constexpr = bM * bN
            elems_per_thread: cutlass.Constexpr = num_elems // self.num_mma_threads
            elem_thread_layout = cute.make_layout(
                (elems_per_thread, self.num_mma_threads),
                stride=(self.num_mma_threads, 1),
            )
            smem_red = cute.make_tensor(
                cute.arch.alloc_smem(
                    cutlass.Float32, num_elems * NUM_MMA_WARPS, alignment=16
                ),
                cute.make_layout((NUM_MMA_WARPS, num_elems), stride=(num_elems, 1)),
            )
            smem_warp = cute.make_tensor(
                cute.domain_offset((mma_warp_idx, 0), smem_red).iterator,
                cute.make_layout((bM, bN), stride=(bN, 1)),
            )
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)
            cute.arch.sync_threads()

            partials = cute.make_tensor(
                cute.arch.alloc_smem(
                    cutlass.Float32, num_elems * self.split_k, alignment=16
                ),
                cute.make_layout((self.split_k, num_elems), stride=(num_elems, 1)),
            )
            cta_rank = cute.arch.block_idx_in_cluster()

            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = cute.crd2idx((ei, mma_tidx), elem_thread_layout)
                local_n, local_m = cute.idx2crd(elem_idx, (bN, bM))
                total = cutlass.Float32(0.0)
                if (local_m < cute.size(gC, mode=[0])) & (
                    local_n < cute.size(gC, mode=[1])
                ):
                    total = smem_red[None, elem_idx].load().reduce(
                        cute.ReductionOp.ADD,
                        init_val=cutlass.Float32(0.0),
                        reduction_profile=0,
                    )
                    total = cutlass.Float32(total) * scale
                partials[cta_rank, elem_idx] = total

            cute.arch.sync_threads()

            # Send partials to all peer CTAs via st.shared::cluster
            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = cute.crd2idx((ei, mma_tidx), elem_thread_layout)
                my_slot = cute.domain_offset((cta_rank, elem_idx), partials).iterator
                my_val = partials[cta_rank, elem_idx]
                for peer in cutlass.range_constexpr(self.split_k):
                    remote = set_block_rank(my_slot, cutlass.Int32(peer))
                    st_shared_remote_f32(remote, my_val)

            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            # if mma_tidx == 0:
            #     cute.arch.griddepcontrol_launch_dependents()
            cute.arch.sync_threads()

            # Local reduction + global output
            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = cute.crd2idx((ei, mma_tidx), elem_thread_layout)
                local_n, local_m = cute.idx2crd(elem_idx, (bN, bM))
                global_coord = cC[local_m, local_n]
                if cute.elem_less(global_coord, mC.shape):
                    acc = partials[None, elem_idx].load().reduce(
                        cute.ReductionOp.ADD,
                        init_val=cutlass.Float32(0.0),
                        reduction_profile=0,
                    )
                    gC[local_m, local_n] = cutlass.Float32(acc)

        cute.arch.sync_threads()
