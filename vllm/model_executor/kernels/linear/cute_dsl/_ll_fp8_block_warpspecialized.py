# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL low-latency E4M3 GEMM with packed UE8M0 block scales."""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline


@dsl_user_op
def fused_ue8m0_scale(sa_packed, sb_packed, byte_idx, *, loc=None, ip=None):
    """Fused scale: 2^(ea+eb-254) from packed ue8m0 A and B scales."""
    f32 = cutlass.Float32.mlir_type
    val_a = sa_packed.ir_value(loc=loc, ip=ip)
    val_b = sb_packed.ir_value(loc=loc, ip=ip)
    idx = byte_idx.ir_value(loc=loc, ip=ip)
    res = _llvm.inline_asm(
        f32,
        [val_a, val_b, idx],
        "{"
        ".reg .u32 ea, eb, combined;"
        "prmt.b32 ea, $1, 0, $3;"
        "and.b32 ea, ea, 0xFF;"
        "prmt.b32 eb, $2, 0, $3;"
        "and.b32 eb, eb, 0xFF;"
        "add.u32 combined, ea, eb;"
        "sub.u32 combined, combined, 127;"
        "shl.b32 combined, combined, 23;"
        "mov.b32 $0, combined;"
        "}",
        "=f,r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(res)


class LLFp8BlockGemm:
    """Low-latency block-scaled FP8 GEMM for small-M problems.

    Computes ``C[M, N] = A[M, K] @ B[N, K]^T`` from E4M3 inputs with
    packed UE8M0 scales and BF16 output. DMA warps stage FP8 data and scales
    through a cp.async pipeline while MMA warps accumulate scaled FP32
    partials, reduce them across warps, and store C.

    :note: Supported A/B data types:
        - Float8E4M3FN/Float8E4M3FN
    :note: Supported accumulator data types:
        - Float32
    :note: Supported C data types:
        - BFloat16
    :note: Constraints:
        - M must be in ``[1, 16]``.
        - K must be divisible by 128 FP8 elements.
        - N must be divisible by 8 output elements.
        - A/B must be contiguous row-major tensors.
        - A/B scales must use packed UE8M0 column-major layout.

    :compile-key: ``(tile_n, tile_k, num_stages, num_dma_warps, has_k_tail)``
        controls tiling, pipelining, producer warps, and K-tail predication.
    """

    def __init__(
        self,
        tile_n: int = 16,
        tile_k: int = 256,
        num_stages: int = 2,
        num_dma_warps: int = 4,
        use_pdl: bool = False,
        has_k_tail: bool = False,
    ):
        """Configure the warp-specialized GEMM.

        :param tile_n: Output columns computed by each CTA.
        :param tile_k: K tile in the BF16 view, containing two FP8 values per element.
        :param num_stages: Number of cp.async pipeline stages.
        :param num_dma_warps: Number of producer warps.
        :param use_pdl: Enable programmatic dependent launch.
        :param has_k_tail: Predicate the final K tile.
        """
        self.ab_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.out_dtype = cutlass.BFloat16
        self.tile_m = 16
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.tile_k_fp8 = tile_k * 2
        self.copy_bits = 128
        self.scale_group_size = 128
        self.scales_per_packed = 4
        self.k_blocks_per_scale = 4
        self.num_stages = num_stages
        self.use_pdl = use_pdl
        self.has_k_tail = has_k_tail
        self.mma_shape = (16, 8, 16)  # partition BF16 views for ldmatrix
        self.mma_shape_fp8 = (16, 8, 32)  # mma.sync E4M3 geometry
        self.atom_layout = (1, 1, 1)
        self.num_dma_warps = num_dma_warps
        self.num_mma_warps = 4
        self.num_dma_threads = num_dma_warps * cute.arch.WARP_SIZE
        self.k_blocks_per_warp = self.tile_k // self.mma_shape[2] // self.num_mma_warps
        self.scale_groups_per_warp = self.k_blocks_per_warp // self.k_blocks_per_scale
        self.num_mma_threads = self.num_mma_warps * cute.arch.WARP_SIZE
        self.num_threads = self.num_dma_threads + self.num_mma_threads
        self.num_epilogue_elems = self.tile_m * self.tile_n
        self.epilogue_elems_per_thread = self.num_epilogue_elems // self.num_mma_threads

    @cute.jit
    def _fill_pred(self, pred_flat, coord_tensor, dim_size):
        # pred_flat is (K_VEC, M/N) for the shape-dynamic edge tile.
        num_vec = pred_flat.shape[0]
        num_mn = pred_flat.shape[1]
        for vec in cutlass.range_constexpr(num_vec):
            for mn in cutlass.range_constexpr(num_mn):
                pred_flat[vec, mn] = cute.elem_less(
                    coord_tensor[(0, vec), mn, 0, 0][0], dim_size
                )

    @cute.jit
    def _fill_tail_pred(self, pred_flat, coord_tensor, k_tile, dim_size, k_size):
        # Predicate one K tile and keep the stage-broadcast view in sync.
        coord_ktile = coord_tensor[None, None, 0, k_tile]
        num_vec = pred_flat.shape[0]
        num_mn = pred_flat.shape[1]
        for vec in cutlass.range_constexpr(num_vec):
            for mn in cutlass.range_constexpr(num_mn):
                pred_flat[vec, mn] = cute.elem_less(
                    coord_ktile[(0, vec), mn], (dim_size, k_size)
                )

    def _make_pred(self, coord_tensor):
        # Broadcast the (K_VEC, M/N) predicate across pipeline stages.
        num_vec = coord_tensor.shape[0][1]
        num_mn = cute.size(coord_tensor, mode=[1])
        pred_flat = cute.make_rmem_tensor(
            cute.make_layout((num_vec, num_mn), stride=(num_mn, 1)),
            cutlass.Boolean,
        )
        pred = cute.make_tensor(
            pred_flat.iterator,
            cute.make_layout(
                (num_vec, num_mn, cute.size(coord_tensor, mode=[2])),
                stride=(num_mn, 1, 0),
            ),
        )
        return pred_flat, pred

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        """Build the staged swizzled SMEM layout for A or B tiles."""
        # Match the swizzle span to contiguous K bytes, capped at three bits.
        major_size = min(smem_tiler[1], 64)
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        """Build the per-thread cp.async vector-copy layout."""
        # Lay threads across K so each lane issues one vector copy.
        copy_elems = copy_bits // dtype.width
        k_threads = cute.size(self.tile_k) // copy_elems
        thread_layout = cute.make_layout(
            (num_threads // k_threads, k_threads), stride=(k_threads, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
        stream: CUstream,
    ):
        """Build the copy/MMA layouts and launch the shape-dynamic GEMM.

        :param mA: BF16 view of E4M3 input A with shape ``[M, K / 2]``.
        :param mB: BF16 view of E4M3 weight B with shape ``[N, K / 2]``.
        :param mC: BF16 output tensor with shape ``[M, N]``.
        :param mSA: Packed UE8M0 A scales, shape ``[M, ceil(K / 512)]``.
        :param mSB: Packed UE8M0 B scales, shape ``[N, ceil(K / 512)]``.
        :param stream: CUDA stream for asynchronous execution.
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
            sa_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, bM * self.num_stages], 4
            ]
            sb_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_stages], 4
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
        op_fp8 = cute.nvgpu.warp.MmaFP8Op(
            cutlass.Float8E4M3FN, self.acc_dtype, self.mma_shape_fp8
        )
        perm_mnk_fp8 = (
            self.atom_layout[0] * self.mma_shape_fp8[0],
            self.atom_layout[1]
            * self.mma_shape_fp8[1]
            * (self.tile_n // self.mma_shape_fp8[1]),
            self.atom_layout[2] * self.mma_shape_fp8[2],
        )
        tiled_mma_fp8 = cute.make_tiled_mma(
            op_fp8, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk_fp8
        )
        grid_m = cute.ceil_div(cute.size(mC, mode=[0]), bM)
        grid_n = cute.ceil_div(cute.size(mC, mode=[1]), bN)
        self.kernel(
            mA,
            mB,
            mC,
            mSA,
            mSB,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            tiled_mma_fp8,
            SharedStorage,
        ).launch(
            grid=[cute.size(grid_m), cute.size(grid_n), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mA,
        mB,
        mC,
        mSA,
        mSB,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tiled_mma_fp8: cute.TiledMma,
        shared_storage: cutlass.Constexpr,
    ):
        """Execute the warp-specialized GEMM in three phases.

        - DMA warps stage FP8 tiles and packed scales through cp.async.
        - MMA warps apply each 128-element scale group after FP8 MMA.
        - MMA-warp partials are reduced in shared memory and stored as BF16.

        The layout and tiled-copy arguments are built by :meth:`__call__` and
        specialize shared-memory allocation and copy/MMA partitioning.
        """
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        num_dma_warps: cutlass.Constexpr = self.num_dma_warps
        is_dma_warp = warp_idx < num_dma_warps
        dma_tidx = tidx
        mma_tidx = tidx - self.num_dma_threads
        m_out = cute.size(mC, mode=[0])
        n_out = cute.size(mC, mode=[1])

        cta_tiler = (bM, bN, bK)
        coord = (bid_m, bid_n, None)
        gA = cute.local_tile(mA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, tiler=cta_tiler, coord=coord, proj=(1, 1, None))

        # 128-bit cp.async copies require 16-byte aligned GMEM views.
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        mcC = cute.make_identity_tensor(mC.layout.shape)
        # Coordinate modes: cA=(M,K,k_tile), cB=(N,K,k_tile), cC=(M,N).
        cA = cute.local_tile(mcA, tiler=cta_tiler, coord=coord, proj=(1, None, 1))
        cB = cute.local_tile(mcB, tiler=cta_tiler, coord=coord, proj=(None, 1, 1))
        cC = cute.local_tile(mcC, tiler=cta_tiler, coord=coord, proj=(1, 1, None))

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

        k_size = cute.size(mA, mode=[1])
        k_tile_count = cute.size(gA, mode=[2])
        sSA_ptr = storage.sa_scale.data_ptr()
        sSB_ptr = storage.sb_scale.data_ptr()
        # Map DMA threads onto repeated M scale rows.
        scale_m_layout = cute.make_layout(
            (bM, self.num_dma_threads // bM), stride=(1, 0)
        )
        # Map CTA N tiles onto 128-column scale blocks.
        scale_n_layout = cute.make_layout(
            (
                self.scale_group_size // bN,
                cute.ceil_div(n_out, self.scale_group_size),
            ),
            stride=(0, self.scale_group_size),
        )
        # Map each K tile onto its packed scale column.
        packed_scale_layout = cute.make_layout(
            (k_tile_count,),
            stride=(
                self.tile_k_fp8 // (self.scale_group_size * self.scales_per_packed),
            ),
        )

        if is_dma_warp:
            # Transfer register budget from copy producers to MMA consumers.
            cute.arch.setmaxregister_decrease(40)
            thr_A = tiled_copy_A.get_slice(dma_tidx)
            thr_B = tiled_copy_B.get_slice(dma_tidx)
            tAgA = thr_A.partition_S(gA)
            tAsA = thr_A.partition_D(sA)
            tBgB = thr_B.partition_S(gB)
            tBsB = thr_B.partition_D(sB)
            tAcA = thr_A.partition_S(cA)
            tBcB = thr_B.partition_S(cB)

            tApA_flat, tApA = self._make_pred(tAcA)
            tBpB_flat, tBpB = self._make_pred(tBcB)
            if const_expr(self.has_k_tail):
                self._fill_tail_pred(tApA_flat, tAcA, 0, mA.shape[0], k_size)
                self._fill_tail_pred(tBpB_flat, tBcB, 0, mB.shape[0], k_size)
            else:
                self._fill_pred(tApA_flat, tAcA, mA.shape[0])
                self._fill_pred(tBpB_flat, tBcB, mB.shape[0])

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_stages
            )

            n_repr = scale_n_layout(bid_n)
            m_slot = scale_m_layout(dma_tidx)
            global_m = cA[m_slot, 0, 0][0]
            safe_m = global_m if global_m < m_out else m_out - 1
            sSA = cute.make_tensor(
                sSA_ptr,
                cute.make_layout((bM, num_stages), stride=(1, bM)),
            )
            sSB = cute.make_tensor(
                sSB_ptr,
                cute.make_layout((num_stages,)),
            )

            # Peeled first iteration: B data + B scale before wait,
            # A data after wait, A scale overlaps with cp.async A.
            mainloop_pipeline.producer_acquire(producer_state)
            packed_k = cutlass.Int32(0)
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, 0],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )
            sSB[producer_state.index] = mSB[n_repr, packed_k]

            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, 0],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            sSA[m_slot, producer_state.index] = mSA[safe_m, packed_k]
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                if const_expr(self.has_k_tail):
                    self._fill_tail_pred(tApA_flat, tAcA, k_tile, mA.shape[0], k_size)
                    self._fill_tail_pred(tBpB_flat, tBcB, k_tile, mB.shape[0], k_size)
                mainloop_pipeline.producer_acquire(producer_state)
                packed_k = packed_scale_layout(k_tile)
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
                sSA[m_slot, producer_state.index] = mSA[safe_m, packed_k]
                sSB[producer_state.index] = mSB[n_repr, packed_k]
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            mainloop_pipeline.producer_tail(producer_state)

        else:
            # MMA warps trade copy registers for accumulator capacity.
            cute.arch.setmaxregister_increase(232)
            mma_warp_idx = warp_idx - num_dma_warps
            num_mma_warps: cutlass.Constexpr = self.num_mma_warps

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
            # SMEM-to-register copies feed the FP8 MMA fragments.
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane_id)
            thr_s2r_B = tiled_s2r_B.get_slice(lane_id)
            tCsA_v = thr_s2r_A.partition_S(sA)
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCsB_v = thr_s2r_B.partition_S(sB)
            tCrB_v = thr_s2r_B.retile(tCrB)
            # Split each stage's MMA K fragments across the consumer warps.
            tCsA_warp_v = cute.logical_divide(
                tCsA_v, (None, None, self.k_blocks_per_warp, None)
            )
            tCsB_warp_v = cute.logical_divide(
                tCsB_v, (None, None, self.k_blocks_per_warp, None)
            )

            k_blocks_per_scale: cutlass.Constexpr = self.k_blocks_per_scale
            scale_groups_per_warp: cutlass.Constexpr = self.scale_groups_per_warp
            # (group_in_warp, mma_warp, k_tile) -> unpacked K-scale index.
            scale_group_layout = cute.make_layout(
                (scale_groups_per_warp, num_mma_warps, k_tile_count),
                stride=(
                    1,
                    scale_groups_per_warp,
                    scale_groups_per_warp * num_mma_warps,
                ),
            )
            # (block_in_scale, group_in_warp) -> warp-local MMA K fragment.
            warp_k_layout = cute.make_layout(
                (k_blocks_per_scale, scale_groups_per_warp),
                stride=(1, k_blocks_per_scale),
            )

            # Map each four-lane MMA row group to its two output-scale rows.
            mma_scale_rows = cute.make_tensor(
                0,
                cute.make_layout(((4, 8), 2), stride=((0, 1), 8)),
            )
            # FP8 accumulator lanes use scale rows (0, 0, 1, 1) twice.
            fragment_scale_layout = cute.make_layout((2, 2, 2), stride=(0, 1, 0))

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages
            )

            # Read each stage's packed scales alongside its FP8 tile.
            sSA_mma = cute.make_tensor(
                sSA_ptr,
                cute.make_layout((bM, num_stages), stride=(1, bM)),
            )
            sSB_mma = cute.make_tensor(
                sSB_ptr,
                cute.make_layout((num_stages,)),
            )

            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                mainloop_pipeline.consumer_wait(consumer_state)

                stage = consumer_state.index
                sa0_packed = cute.make_rmem_tensor((1,), cutlass.Int32)
                sa0_packed[0] = sSA_mma[mma_scale_rows[lane_id, 0], stage]
                sa1_packed = cute.make_rmem_tensor((1,), cutlass.Int32)
                sa1_packed[0] = sSA_mma[mma_scale_rows[lane_id, 1], stage]
                sb_packed = cute.make_rmem_tensor((1,), cutlass.Int32)
                sb_packed[0] = sSB_mma[stage]

                for sg in cutlass.range(scale_groups_per_warp, unroll_full=True):
                    scale_k_idx = scale_group_layout((sg, mma_warp_idx, k_tile))
                    byte_k, _ = cute.idx2crd(
                        scale_k_idx,
                        (self.scales_per_packed, k_tile_count),
                    )

                    scales = cute.make_rmem_tensor((2,), cutlass.Float32)
                    scales[0] = fused_ue8m0_scale(sa0_packed[0], sb_packed[0], byte_k)
                    scales[1] = fused_ue8m0_scale(sa1_packed[0], sb_packed[0], byte_k)

                    tCrP = tiled_mma_fp8.make_fragment_C(tCgC)
                    tCrP.fill(0.0)
                    for kb in cutlass.range(k_blocks_per_scale, unroll_full=True):
                        warp_k = warp_k_layout((kb, sg))
                        cute.copy(
                            tiled_s2r_A,
                            tCsA_warp_v[
                                None, None, (warp_k, mma_warp_idx), consumer_state.index
                            ],
                            tCrA_v[None, None, 0],
                        )
                        cute.copy(
                            tiled_s2r_B,
                            tCsB_warp_v[
                                None, None, (warp_k, mma_warp_idx), consumer_state.index
                            ],
                            tCrB_v[None, None, 0],
                        )
                        cute.gemm(
                            tiled_mma_fp8,
                            tCrP,
                            cute.recast_tensor(
                                tCrA[None, None, 0], cutlass.Float8E4M3FN
                            ),
                            cute.recast_tensor(
                                tCrB[None, None, 0], cutlass.Float8E4M3FN
                            ),
                            tCrP,
                        )

                    for value in cutlass.range_constexpr(cute.size(tCrP)):
                        scale_idx = fragment_scale_layout(value)
                        tCrC[value] = tCrC[value] + tCrP[value] * scales[scale_idx]

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # Let dependent grids launch while this CTA reduces and stores C.
            if const_expr(self.use_pdl) and mma_tidx == 0:
                cute.arch.griddepcontrol_launch_dependents()
            cute.arch.sync_threads()

            # Reduce MMA-warp partials and store the CTA output tile.
            num_elems: cutlass.Constexpr = self.num_epilogue_elems
            elems_per_thread: cutlass.Constexpr = self.epilogue_elems_per_thread
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

            for ei in cutlass.range_constexpr(elems_per_thread):
                elem_idx = epilogue_slots[ei, mma_tidx]
                local_coord = cute.select(epilogue_slot_coords[elem_idx], mode=[1, 0])
                global_coord = cC[local_coord]
                if cute.elem_less(global_coord, mC.shape):
                    total = (
                        smem_red[None, elem_idx]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=cutlass.Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    gC[local_coord] = cutlass.Float32(total).to(self.out_dtype)

        cute.arch.sync_threads()
