# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline


# Map a local smem address to the same peer CTA DSMEM offset.
@dsl_user_op
def set_block_rank(smem_ptr, peer_rank, *, loc=None, ip=None):
    dsmem_ptr = cute.arch.map_dsmem_ptr(smem_ptr, peer_rank, loc=loc, ip=ip)
    return cutlass.Int32(dsmem_ptr.toint(loc=loc, ip=ip))


# Plain DSMEM store; mbarrier helpers do not model this reduction.
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
        has_side_effects=True,  # keep the inline store ordered
        loc=loc,
        ip=ip,
    )


class LLBf16SplitK:
    """BF16 router GEMM kernel based on clustered split-K MMA.

    This kernel computes C[M, N] = A[M, K] @ B[N, K]^T for bf16 inputs
    and fp32 output. It partitions K across a CTA cluster, uses DMA warps to
    stage A/B tiles with cp.async, uses MMA warps to accumulate fp32 partials,
    and reduces split-K partials through DSMEM before storing C.

    :param ab_dtype: Element type for A and B operands.
    :param acc_dtype: Accumulator type used by MMA and reductions.
    :param out_dtype: Output element type. The public wrapper uses fp32.
    :param tile_n: CTA tile size in N. M is fixed to 16 for router batches.
    :type tile_n: int
    :param tile_k: K tile size staged through shared memory.
    :type tile_k: int
    :param num_stages: Number of cp.async pipeline stages.
    :type num_stages: int
    :param num_dma_warps: Producer warps that issue GMEM to SMEM copies.
    :type num_dma_warps: int
    :param split_k: Number of CTAs in the cluster-level split-K reduction.
    :type split_k: int
    :param use_pdl: Whether to launch with Programmatic Dependent Launch.
    :type use_pdl: bool

    :note: Supported A/B data types:
        - BFloat16/BFloat16
    :note: Supported accumulator data types:
        - Float32
    :note: Supported C data types:
        - Float32
    :note: Constraints:
        - K must preserve 16-byte row alignment for contiguous bf16 inputs.

    :compile-key: ``(split_k, num_stages)`` selects the cluster split
        count and cp.async pipeline depth specialization.
    """

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
        use_pdl: bool = False,
    ):
        """Initialize the split-K kernel configuration.

        This configuration fixes the CTA tile shape, cp.async pipeline depth,
        producer/consumer warp split, and CTA-cluster split count used by the
        DSMEM reduction.

        :param ab_dtype: Element type for A and B operands.
        :param acc_dtype: Accumulator type used by MMA and reductions.
        :param out_dtype: Output element type.
        :param tile_n: CTA tile size in N. M is fixed to 16.
        :type tile_n: int
        :param tile_k: K tile size staged through shared memory.
        :type tile_k: int
        :param num_stages: Number of cp.async pipeline stages.
        :type num_stages: int
        :param num_dma_warps: Producer warps that issue GMEM to SMEM copies.
        :type num_dma_warps: int
        :param split_k: CTAs in the cluster-level split-K reduction.
        :type split_k: int
        :param use_pdl: Whether to launch with Programmatic Dependent Launch.
        :type use_pdl: bool
        """
        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.tile_m = 16
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.copy_bits = 128
        self.num_stages = num_stages
        self.split_k = split_k
        self.use_pdl = use_pdl
        self.mma_shape = (16, 8, 16)  # mma.sync.aligned.m16n8k16
        self.atom_layout = (1, 1, 1)  # one MMA atom per warp
        self.num_dma_warps = num_dma_warps
        self.num_mma_warps = 4
        self.num_dma_threads = self.num_dma_warps * cute.arch.WARP_SIZE
        self.num_mma_threads = self.num_mma_warps * cute.arch.WARP_SIZE
        self.num_threads = self.num_dma_threads + self.num_mma_threads
        self.num_epilogue_elems = self.tile_m * self.tile_n
        self.epilogue_elems_per_thread = self.num_epilogue_elems // self.num_mma_threads

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        """Build the staged swizzled SMEM layout for A or B tiles."""
        major_size = min(smem_tiler[1], 64)
        # Match swizzle span to contiguous K bytes, capped by CuTe 3-bit swizzle.
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        # Tile the swizzled atom across (M_or_N, K, stages).
        layout_atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        """Build the per-thread cp.async vector-copy layout."""
        # Lay threads across K so each lane issues one vector copy.
        copy_elems = copy_bits // dtype.width
        k_threads = cute.size(self.tile_k) // copy_elems  # threads along K
        thread_layout = cute.make_layout(
            (num_threads // k_threads, k_threads), stride=(k_threads, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def _fill_pred(self, pred_flat, coord_tensor, k_tile, dim_limit, K_total):
        # Predicate one K tile and keep the stage-broadcast view in sync.
        coord_ktile = coord_tensor[None, None, 0, k_tile]
        num_vec = pred_flat.shape[0]
        num_mn = pred_flat.shape[1]
        for v in cutlass.range_constexpr(num_vec):
            # pred_flat is (K_VEC, M/N) for one K_TILE.
            for j in cutlass.range_constexpr(num_mn):
                pred_flat[v, j] = cute.elem_less(
                    coord_ktile[(0, v), j], (dim_limit, K_total)
                )

    def _make_pred(self, tXcX, k_tile, dim_limit, K_total):
        # pred_flat is (K_VEC, M/N); pred is (K_VEC, M/N, STAGE).
        num_vec = tXcX.shape[0][1]
        num_mn = cute.size(tXcX, mode=[1])
        pred_flat = cute.make_rmem_tensor(
            cute.make_layout((num_vec, num_mn), stride=(num_mn, 1)),
            cutlass.Boolean,
        )
        pred = cute.make_tensor(
            pred_flat.iterator,
            cute.make_layout(
                (num_vec, num_mn, cute.size(tXcX, mode=[2])),
                stride=(num_mn, 1, 0),
            ),
        )
        return pred_flat, pred

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: CUstream,
        scale: float = 1.0,
    ):
        """Execute the split-K GEMM operation in steps:
        - Build swizzled staged SMEM layouts for A ``[16, tile_k, stages]``
          and B ``[tile_n, tile_k, stages]``.
        - Build 128-bit cp.async GMEM-to-SMEM tiled copies for DMA warps.
        - Build an m16n8k16 BF16 MMA tiled across the CTA N tile.
        - Launch grid ``ceil(M/16) x ceil(N/tile_n) x split_k`` with one
          cluster along split-K, so each cluster rank owns K tiles
          ``rank, rank + split_k, ...``.
        - Reduce MMA-warp partials within the CTA, exchange split-K partials
          through DSMEM, then reduce cluster partials and store C.

        :param mA: Input tensor A with shape ``[M, K]``.
        :type mA: cute.Tensor
        :param mB: Input tensor B with shape ``[N, K]``.
        :type mB: cute.Tensor
        :param mC: Output tensor C with shape ``[M, N]``.
        :type mC: cute.Tensor
        :param stream: CUDA stream for asynchronous execution.
        :type stream: CUstream
        :param scale: Epilogue scale applied before storing C.
        :type scale: float
        """
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        copy_bits: cutlass.Constexpr = self.copy_bits
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, copy_bits, (bM, bK, self.num_stages)
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, copy_bits, (bN, bK, self.num_stages)
        )

        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16
            ]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, self.num_stages * 2], 8
            ]

        atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=copy_bits,
        )  # cp.async GMEM -> SMEM, bypassing L1
        tiled_copy_A = self._make_gmem_tiled_copy(
            atom_g2s, mA.element_type, copy_bits, self.num_dma_threads
        )
        tiled_copy_B = self._make_gmem_tiled_copy(
            atom_g2s, mB.element_type, copy_bits, self.num_dma_threads
        )
        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_shape)
        # Repeat the m16n8k16 atom along N to cover the CTA output tile.
        perm_mnk = (
            self.atom_layout[0] * self.mma_shape[0],
            self.atom_layout[1] * self.mma_shape[1] * (self.tile_n // 8),
            self.atom_layout[2] * self.mma_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk
        )
        tiler_mn = (bM, bN)
        grid_m, grid_n = cute.ceil_div(mC.shape, tiler_mn)
        self.kernel(
            mA,
            mB,
            mC,
            scale,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=[
                cute.size(grid_m),
                cute.size(grid_n),
                self.split_k,
            ],
            block=[self.num_threads, 1, 1],
            cluster=[
                1,
                1,
                self.split_k,
            ],  # split-K CTAs form one cluster
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
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
        shared_storage: cutlass.Constexpr,
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_z = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        num_dma_warps: cutlass.Constexpr = self.num_dma_warps
        is_dma_warp = warp_idx < num_dma_warps
        dma_tidx = tidx
        mma_tidx = tidx - self.num_dma_threads
        mma_warp_idx = warp_idx - num_dma_warps

        cta_tiler = (bM, bN, bK)
        coord = (bid_m, bid_n, None)  # all K tiles
        # CTA-local tiles.
        gA = cute.local_tile(
            mA, tiler=cta_tiler, coord=coord, proj=(1, None, 1)
        )  # skip N
        gB = cute.local_tile(
            mB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1)
        )  # skip M
        gC = cute.local_tile(
            mC, tiler=cta_tiler, coord=coord, proj=(1, 1, None)
        )  # skip K

        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        mcC = cute.make_identity_tensor(mC.layout.shape)
        # Coordinate modes: cA=(M,K,k_tile), cB=(N,K,k_tile), cC=(M,N).
        cA = cute.local_tile(mcA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        cC = cute.local_tile(mcC, tiler=cta_tiler, coord=coord, proj=(1, 1, None))

        # 128-bit cp.async copies require 16-byte aligned GMEM views.
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        smem = cutlass.utils.SmemAllocator()
        storage_ptr = smem.allocate(shared_storage.size_in_bytes(), byte_alignment=16)  # type: ignore[attr-defined]
        storage = shared_storage(storage_ptr)  # type: ignore[call-arg]
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)

        # Pipeline cp.async producers into MMA consumers.
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

        # Round-robin split-K: split z handles tiles z, z+split_k, ...
        K_total = cute.size(mA, mode=[1])
        k_tile_count = cute.size(gA, mode=[2])
        k_start = bid_z
        num_k_tiles = cute.ceil_div(k_tile_count - k_start, self.split_k)

        if is_dma_warp:
            # DMA warps trade registers for copy throughput.
            cute.arch.setmaxregister_decrease(40)
            thr_A = tiled_copy_A.get_slice(dma_tidx)
            thr_B = tiled_copy_B.get_slice(dma_tidx)
            tAgA = thr_A.partition_S(gA)
            tAsA = thr_A.partition_D(sA)
            tBgB = thr_B.partition_S(gB)
            tBsB = thr_B.partition_D(sB)
            tAcA = thr_A.partition_S(cA)
            tBcB = thr_B.partition_S(cB)

            # Build M/K and N/K predicates, broadcast across K-tile copies.
            tApA_flat, tApA = self._make_pred(tAcA, k_start, mA.shape[0], K_total)
            tBpB_flat, tBpB = self._make_pred(tBcB, k_start, mB.shape[0], K_total)
            self._fill_pred(tApA_flat, tAcA, k_start, mA.shape[0], K_total)
            self._fill_pred(tBpB_flat, tBcB, k_start, mB.shape[0], K_total)

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_stages
            )

            # Prime the first pipeline stage.
            mainloop_pipeline.producer_acquire(producer_state)
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_start],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
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
                self._fill_pred(tApA_flat, tAcA, k_tile, mA.shape[0], K_total)
                self._fill_pred(tBpB_flat, tBcB, k_tile, mB.shape[0], K_total)
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
            cute.arch.setmaxregister_increase(232)  # large MMA fragments

            num_mma_warps: cutlass.Constexpr = self.num_mma_warps

            thr_mma = tiled_mma.get_slice(lane_id)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)

            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # ldmatrix moves SMEM fragments into MMA registers.
            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type
            )  # non-transposed, x4
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mB.element_type
            )

            # SMEM -> register copy path.
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane_id)
            thr_s2r_B = tiled_s2r_B.get_slice(lane_id)
            tCsA_v = thr_s2r_A.partition_S(sA)  # views, not copies
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCsB_v = thr_s2r_B.partition_S(sB)
            tCrB_v = thr_s2r_B.retile(tCrB)
            # Split the MMA K-fragments across the MMA warps.
            tCsA_warp_v = cute.logical_divide(tCsA_v, (None, None, num_mma_warps, None))
            tCsB_warp_v = cute.logical_divide(tCsB_v, (None, None, num_mma_warps, None))

            num_k_blocks = cute.size(tCrA, mode=[2])
            k_blocks_per_warp: cutlass.Constexpr = num_k_blocks // num_mma_warps

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages
            )

            # Shape-dynamic split-K count, so this stays a runtime range.
            for _ in cutlass.range(num_k_tiles, unroll_full=True):
                mainloop_pipeline.consumer_wait(consumer_state)
                for ki in cutlass.range_constexpr(k_blocks_per_warp):
                    cute.copy(
                        tiled_s2r_A,
                        tCsA_warp_v[
                            None, None, (mma_warp_idx, ki), consumer_state.index
                        ],
                        tCrA_v[None, None, 0],
                    )  # ldmatrix
                    cute.copy(
                        tiled_s2r_B,
                        tCsB_warp_v[
                            None, None, (mma_warp_idx, ki), consumer_state.index
                        ],
                        tCrB_v[None, None, 0],
                    )
                    cute.gemm(
                        tiled_mma, tCrC, tCrA[None, None, 0], tCrB[None, None, 0], tCrC
                    )  # mma.sync
                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # Cluster reduction epilogue.
            # Reduce per-warp accumulators within this CTA.
            num_elems: cutlass.Constexpr = self.num_epilogue_elems
            elems_per_thread: cutlass.Constexpr = self.epilogue_elems_per_thread
            # Map MMA threads to linear CTA output elements.
            epilogue_thread_layout = cute.make_layout(
                (elems_per_thread, self.num_mma_threads),
                stride=(self.num_mma_threads, 1),
            )
            epilogue_slots = cute.make_tensor(0, epilogue_thread_layout)
            epilogue_slot_coords = cute.make_identity_tensor((bN, bM))
            # Layout: (mma_warp, linear MN element).
            smem_red = cute.make_tensor(
                cute.arch.alloc_smem(
                    cutlass.Float32, num_elems * num_mma_warps, alignment=16
                ),
                cute.make_layout((num_mma_warps, num_elems), stride=(num_elems, 1)),
            )
            smem_warp = cute.make_tensor(
                cute.domain_offset((mma_warp_idx, 0), smem_red).iterator,
                cute.make_layout((bM, bN), stride=(bN, 1)),
            )
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)
            cute.arch.sync_threads()

            # Layout: (split-K rank, linear MN element).
            partials = cute.make_tensor(
                cute.arch.alloc_smem(
                    cutlass.Float32, num_elems * self.split_k, alignment=16
                ),
                cute.make_layout((self.split_k, num_elems), stride=(num_elems, 1)),
            )
            cta_rank = cute.arch.block_idx_in_cluster()

            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = epilogue_slots[ei, mma_tidx]
                local_coord = cute.select(epilogue_slot_coords[elem_idx], mode=[1, 0])
                total = cutlass.Float32(0.0)
                if cute.elem_less(local_coord, gC.shape):
                    total = (
                        smem_red[None, elem_idx]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=cutlass.Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    total = total * scale
                partials[cta_rank, elem_idx] = total

            cute.arch.sync_threads()

            # Broadcast this CTA's partials to peer DSMEM.
            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = epilogue_slots[ei, mma_tidx]
                my_slot = cute.domain_offset((cta_rank, elem_idx), partials).iterator
                my_val = partials[cta_rank, elem_idx]
                for peer in cutlass.range_constexpr(self.split_k):
                    remote = set_block_rank(my_slot, cutlass.Int32(peer))
                    st_shared_remote_f32(remote, my_val)

            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()  # peer DSMEM stores are now visible

            if const_expr(self.use_pdl) and mma_tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()
            cute.arch.sync_threads()

            # Reduce split-K partials and write global output.
            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = epilogue_slots[ei, mma_tidx]
                local_coord = cute.select(epilogue_slot_coords[elem_idx], mode=[1, 0])
                global_coord = cC[local_coord]
                if cute.elem_less(global_coord, mC.shape):
                    acc = (
                        partials[None, elem_idx]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=cutlass.Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    gC[local_coord] = acc

        cute.arch.sync_threads()
