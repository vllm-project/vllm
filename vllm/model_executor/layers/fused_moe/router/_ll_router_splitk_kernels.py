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


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    i32 = _ir.IntegerType.get_signless(32)
    _llvm.inline_asm(
        i32,
        [],
        "barrier.cluster.arrive; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,  # prevents the compiler from reordering or eliminating this barrier
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
        has_side_effects=True,  # prevents the compiler from reordering or eliminating this barrier
        loc=loc,
        ip=ip,
    )


# Takes a shared memory address in THIS CTA (src_addr) and a peer CTA rank (target_rank).
# Returns the address that maps to the SAME offset in the PEER's shared memory
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
        has_side_effects=False,  # pure address computation, no memory access. Safe to reorder
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(res)


# Writes an FP32 value to a REMOTE CTA's shared memory via the mapped address from set_block_rank.
# This goes through the cluster's DSMEM interconnect, not through HBM.
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


def _fill_pred(pred_flat, coord_ktile, dim_limit, k_base, K_total):
    """Fills a predicate tensor in-place. Called both during initial predicate
    construction and in the K-loop to update predicates per tile."""
    num_vec = pred_flat.shape[0]  # = 8 - elements per 128-bit copy
    num_mn = pred_flat.shape[1]  # = 4 - M-rows (or N-rows) this thread handles per tile
    for v in range(num_vec):
        k_valid = (k_base + coord_ktile[(0, v), 0][1]) < K_total
        for j in range(num_mn):
            # Two checks:
            # elem_less: is the M-coord of this elem < total M? Prevents out-of-bounds reads on the M-dim
            # k_valid: is this K element < K_total? Prevents out-of-bounds reads on partial K tiles.
            pred_flat[v, j] = (
                cute.elem_less(coord_ktile[(0, v), j][0], dim_limit)
                & k_valid  # [0] extracts the M-coordinate from the (M, K) coordinate tuple
            )


def _make_pred(tXcX, dim_limit, k_base, K_total):
    """Builds a complete predicate tensor from a coordinate tensor partition.
    Returns a tuple of (coord_ktile, pred_flat, pred_broadcast)."""
    tXcX_ktile = tXcX[
        None, None, 0, 0
    ]  # Slice to one K-tile: None keeps CopyOp and CopyM, 0 selects CopyK=0 and k_tile=0. Result shape: ((8,1), 4).
    num_vec = tXcX.shape[0][
        1
    ]  # = 8 — from the vector part of the hierarchical CopyOp shape (8, 1).
    num_mn = cute.size(
        tXcX, mode=[1]
    )  # = 4 — flattened size of mode 1 (CopyM or CopyN).
    pred_flat = cute.make_rmem_tensor(
        cute.make_layout((num_vec, num_mn), stride=(num_mn, 1)),
        cutlass.Boolean,
    )  # Allocates a 2D boolean tensor in registers: (8, 4) with row-major stride (4, 1)
    _fill_pred(
        pred_flat, tXcX_ktile, dim_limit, k_base, K_total
    )  # Fill with initial values (M/N-bounds AND K-bounds for the first tile).
    pred = cute.make_tensor(
        pred_flat.iterator,
        cute.make_layout(
            (num_vec, num_mn, cute.size(tXcX, mode=[2])),
            stride=(num_mn, 1, 0),  # last-dim stride 0 = broadcast
            # Predicate values are the same for all K-tiles
        ),
    )  # Wraps the same register storage with a 3D layout: (8, 4, 1)
    return tXcX_ktile, pred_flat, pred


@dsl_user_op
class LLRouterSplitK:
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
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k  # 16, 16, 256
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_z = cute.arch.block_idx()
        warp_idx = tidx // 32
        is_dma = warp_idx < (self.num_dma_threads // 32)  # warps 0-3 = DMA
        dma_tidx = tidx  # DMA warps use raw thread index (0-127)
        mma_tidx = tidx - self.num_dma_threads  # MMA warps use offset index (0-127)
        N_out = cute.size(mC, mode=[1])  # total N (not tiled)
        M_out = cute.size(mC, mode=[0])  # total M

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
        cA = cute.local_tile(mcA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))

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

        # Split-K work distribution across 8 CTAs as evenly as possible
        # For K=7168: k_tile_count_full = 28, tiles_per_split = 3, extra = 4.
        # - CTAs 0-3: my_tiles = 4, starting at tiles 0, 4, 8, 12
        # - CTAs 4-7: my_tiles = 3, starting at tiles 16, 19, 22, 25
        K_total = cute.size(mA, mode=[1])
        k_tile_count_full = (K_total + bK - 1) // bK
        tiles_per_split = k_tile_count_full // self.split_k
        extra = k_tile_count_full - tiles_per_split * self.split_k
        k_start = bid_z * tiles_per_split + (bid_z if bid_z < extra else extra)
        my_tiles = tiles_per_split + (1 if bid_z < extra else 0)

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

            # Build M/K and N/K predicates, broadcast across K-tiles via stride-0
            k_base_first = k_start * bK
            tAcA_ktile, tApA_flat, tApA = _make_pred(
                tAcA, mA.shape[0], k_base_first, K_total
            )
            tBcB_ktile, tBpB_flat, tBpB = _make_pred(
                tBcB, mB.shape[0], k_base_first, K_total
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
            cute.arch.griddepcontrol_wait()  # PDL-wait
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_start],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

            for local_k in range(1, my_tiles):  # Remaining tiles loop
                k_tile = k_start + local_k
                k_base_global = k_tile * bK
                _fill_pred(tApA_flat, tAcA_ktile, mA.shape[0], k_base_global, K_total)
                _fill_pred(tBpB_flat, tBcB_ktile, mB.shape[0], k_base_global, K_total)
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

            num_k_block = cute.size(tCrA, mode=[2])
            K_PER_WARP: cutlass.Constexpr = num_k_block // NUM_MMA_WARPS # Each warp handles 4 K-blocks per tile

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages
            )

            for local_k in range(my_tiles):
                mainloop_pipeline.consumer_wait(consumer_state)
                tCsA_p = tCsA_v[None, None, None, consumer_state.index]
                tCsB_p = tCsB_v[None, None, None, consumer_state.index]
                for ki in cutlass.range(K_PER_WARP, unroll_full=True): # unroll_full=True unrolls all 4 iterations
                    k_block = ki * NUM_MMA_WARPS + mma_warp_idx
                    cute.copy(
                        tiled_s2r_A, tCsA_p[None, None, k_block], tCrA_v[None, None, 0]
                    ) # ldmatrix
                    cute.copy(
                        tiled_s2r_B, tCsB_p[None, None, k_block], tCrB_v[None, None, 0]
                    )
                    cute.gemm(
                        tiled_mma, tCrC, tCrA[None, None, 0], tCrB[None, None, 0], tCrC
                    ) # mma.sync
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
