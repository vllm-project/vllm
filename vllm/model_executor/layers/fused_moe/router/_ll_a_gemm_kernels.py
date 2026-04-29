# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL tiled A GEMM: C[M,N] = A[M,K] @ B[N,K]^T."""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline


@dsl_user_op
def bf16x2_pack(lo, hi, *, loc=None, ip=None):
    """Pack 2 bf16 → 1 uint32 via vector insert + bitcast."""
    lo_ir = lo.ir_value(loc=loc, ip=ip)
    hi_ir = hi.ir_value(loc=loc, ip=ip)
    bf16_ty = lo_ir.type
    vec_ty = _ir.VectorType.get([2], bf16_ty)
    i32 = _ir.IntegerType.get_signless(32)
    c0 = _arith.constant(i32, 0, loc=loc, ip=ip)
    c1 = _arith.constant(i32, 1, loc=loc, ip=ip)
    undef = _llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
    v0 = _llvm.insertelement(vec_ty, undef, lo_ir, c0, loc=loc, ip=ip)
    v1 = _llvm.insertelement(vec_ty, v0, hi_ir, c1, loc=loc, ip=ip)
    packed = _llvm.bitcast(i32, v1, loc=loc, ip=ip)
    return cutlass.Uint32(packed)


@dsl_user_op
def mma_e4m3(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    """mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32"""
    f32 = cutlass.Float32.mlir_type
    args = [
        a0.ir_value(loc=loc, ip=ip),
        a1.ir_value(loc=loc, ip=ip),
        a2.ir_value(loc=loc, ip=ip),
        a3.ir_value(loc=loc, ip=ip),
        b0.ir_value(loc=loc, ip=ip),
        b1.ir_value(loc=loc, ip=ip),
        d0.ir_value(loc=loc, ip=ip),
        d1.ir_value(loc=loc, ip=ip),
        d2.ir_value(loc=loc, ip=ip),
        d3.ir_value(loc=loc, ip=ip),
    ]
    res = _llvm.inline_asm(
        _llvm.StructType.get_literal([f32, f32, f32, f32]),
        args,
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    r0 = _llvm.extractvalue(f32, res, [0], loc=loc, ip=ip)
    r1 = _llvm.extractvalue(f32, res, [1], loc=loc, ip=ip)
    r2 = _llvm.extractvalue(f32, res, [2], loc=loc, ip=ip)
    r3 = _llvm.extractvalue(f32, res, [3], loc=loc, ip=ip)
    return (
        cutlass.Float32(r0),
        cutlass.Float32(r1),
        cutlass.Float32(r2),
        cutlass.Float32(r3),
    )


def _pack2(lo, hi, *, loc=None, ip=None):
    """Pack 2 bf16 → 1 uint32 via vector insert + bitcast.

    Uses LLVM vector ops instead of scalar integer ops.
    LLVM instcombine folds insert(extract(vec,0), extract(vec,1))→vec
    back to the original i32 register from ldmatrix.
    """
    bf16_ty = lo.type
    vec_ty = _ir.VectorType.get([2], bf16_ty)
    i32 = _ir.IntegerType.get_signless(32)
    c0 = _arith.constant(i32, 0, loc=loc, ip=ip)
    c1 = _arith.constant(i32, 1, loc=loc, ip=ip)
    undef = _llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
    v0 = _llvm.insertelement(undef, lo, c0, loc=loc, ip=ip)
    v1 = _llvm.insertelement(v0, hi, c1, loc=loc, ip=ip)
    return _llvm.bitcast(i32, v1, loc=loc, ip=ip)


@dsl_user_op
def fused_fp8_mma_2n(
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    a0_lo,
    a0_hi,
    a1_lo,
    a1_hi,
    a2_lo,
    a2_hi,
    a3_lo,
    a3_hi,
    b0_lo,
    b0_hi,
    b1_lo,
    b1_hi,
    b2_lo,
    b2_hi,
    b3_lo,
    b3_hi,
    *,
    loc=None,
    ip=None,
):
    """Fused: pack bf16 pairs + 2x mma.sync.m16n8k32.e4m3 (both N-atoms)."""
    f32 = cutlass.Float32.mlir_type

    # Pack A: 8 bf16 → 4 uint32
    a0 = _pack2(
        a0_lo.ir_value(loc=loc, ip=ip), a0_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    a1 = _pack2(
        a1_lo.ir_value(loc=loc, ip=ip), a1_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    a2 = _pack2(
        a2_lo.ir_value(loc=loc, ip=ip), a2_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    a3 = _pack2(
        a3_lo.ir_value(loc=loc, ip=ip), a3_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )

    # Pack B N-atom 0: 4 bf16 → 2 uint32
    bn0 = _pack2(
        b0_lo.ir_value(loc=loc, ip=ip), b0_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    bn1 = _pack2(
        b1_lo.ir_value(loc=loc, ip=ip), b1_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )

    # MMA N-atom 0
    r0 = _llvm.inline_asm(
        _llvm.StructType.get_literal([f32, f32, f32, f32]),
        [
            a0,
            a1,
            a2,
            a3,
            bn0,
            bn1,
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )

    # Pack B N-atom 1
    bn2 = _pack2(
        b2_lo.ir_value(loc=loc, ip=ip), b2_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    bn3 = _pack2(
        b3_lo.ir_value(loc=loc, ip=ip), b3_hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )

    # MMA N-atom 1
    r1 = _llvm.inline_asm(
        _llvm.StructType.get_literal([f32, f32, f32, f32]),
        [
            a0,
            a1,
            a2,
            a3,
            bn2,
            bn3,
            c4.ir_value(loc=loc, ip=ip),
            c5.ir_value(loc=loc, ip=ip),
            c6.ir_value(loc=loc, ip=ip),
            c7.ir_value(loc=loc, ip=ip),
        ],
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )

    return (
        cutlass.Float32(_llvm.extractvalue(f32, r0, [0], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r0, [1], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r0, [2], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r0, [3], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r1, [0], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r1, [1], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r1, [2], loc=loc, ip=ip)),
        cutlass.Float32(_llvm.extractvalue(f32, r1, [3], loc=loc, ip=ip)),
    )


class LLAGemm:
    """Warp-specialized low-latency A GEMM with PipelineCpAsync."""

    def __init__(
        self,
        ab_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        out_dtype=cutlass.BFloat16,
        tile_n: int = 32,
        tile_k: int = 512,
        num_stages: int = 3,
        is_fp8: bool = False,
        num_dma_warps: int = 4,
    ):
        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.tile_m = 16
        # min tile_n = 1*8*2 = 16
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.is_fp8 = is_fp8
        self.mma_shape = (16, 8, 16)
        # (1,1,1) = 32 threads per MMA warp
        # 4 MMA warps doing k-phase interleaving on tile_n=16
        self.atom_layout = (1, 1, 1)
        self.num_mma_warps = 4
        self.num_dma_threads = num_dma_warps * 32
        self.num_mma_threads = self.num_mma_warps * 32  # 128
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

    def _make_smem_layout_C(self, dtype, copy_bits, smem_tiler):
        return cute.make_layout(smem_tiler, stride=(smem_tiler[1], 1))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        copy_elems = copy_bits // dtype.width
        k_threads = cute.size(self.tile_k) // copy_elems
        thread_layout = cute.make_layout(
            (num_threads // k_threads, k_threads), stride=(k_threads, 1)
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def __call__(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: CUstream
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        copy_bits = 128

        sA_layout = self._make_smem_layout_AB(
            mA.element_type, copy_bits, (bM, bK, self.num_stages)
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, copy_bits, (bN, bK, self.num_stages)
        )
        sC_layout = self._make_smem_layout_C(mC.element_type, copy_bits, (bM, bN))

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

        atom_s2g = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mC.element_type, num_bits_per_copy=copy_bits
        )
        c_copy_elems = copy_bits // mC.element_type.width
        cn_threads = bN // c_copy_elems
        tiled_copy_C = cute.make_tiled_copy_tv(
            atom_s2g,
            cute.make_layout(
                (self.num_mma_threads // cn_threads, cn_threads), stride=(cn_threads, 1)
            ),
            cute.make_layout((1, c_copy_elems)),
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

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
        ).launch(
            grid=[cute.size(grid_m), cute.size(grid_n), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mA,
        mB,
        mC,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        num_stages = self.num_stages
        tidx, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, _ = cute.arch.block_idx()

        warp_idx = tidx // 32
        is_dma = warp_idx < (self.num_dma_threads // 32)
        dma_tidx = tidx
        mma_tidx = tidx - self.num_dma_threads

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
            c: cute.struct.Align[
                cute.struct.MemRange[mC.element_type, cute.cosize(sC_layout)], 16
            ]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, num_stages * 2], 8
            ]

        smem = cutlass.utils.SmemAllocator()
        storage_ptr = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
        storage = SharedStorage(storage_ptr)
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)
        sC = storage.c.get_tensor(sC_layout)

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

        k_tile_count = cute.size(gA, mode=[2])

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
                tBgB[None, None, None, 0],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )

            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, 0],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            mainloop_pipeline.producer_commit(producer_state)
            producer_state.advance()

            for k_tile in range(1, k_tile_count):
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
            # ===== 4 MMA WARPS with k-phase interleaving =====
            cute.arch.setmaxregister_increase(232)

            lane_id = mma_tidx % 32
            mma_warp_idx = mma_tidx // 32  # 0-3
            NUM_MMA_WARPS: cutlass.Constexpr = self.num_mma_warps

            # Each warp uses the same tiled_mma (32 threads)
            # All warps partition the SAME smem; they'll index different k_blocks
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

            for k_tile in range(k_tile_count):
                mainloop_pipeline.consumer_wait(consumer_state)

                tCsA_p = tCsA_v[None, None, None, consumer_state.index]
                tCsB_p = tCsB_v[None, None, None, consumer_state.index]

                # K-phase: each warp directly computes its k_block index
                # No branch — each warp loops K_PER_WARP times
                # Reuse rmem slot 0 (no need to index by k_block)
                if not self.is_fp8:
                    for ki in cutlass.range(K_PER_WARP, unroll_full=True):
                        k_block = ki * NUM_MMA_WARPS + mma_warp_idx
                        cute.copy(
                            tiled_s2r_A,
                            tCsA_p[None, None, k_block],
                            tCrA_v[None, None, 0],
                        )
                        cute.copy(
                            tiled_s2r_B,
                            tCsB_p[None, None, k_block],
                            tCrB_v[None, None, 0],
                        )
                        cute.gemm(
                            tiled_mma,
                            tCrC,
                            tCrA[None, None, 0],
                            tCrB[None, None, 0],
                            tCrC,
                        )
                else:
                    # fp8: keep accumulators as scalars, avoid
                    # fragment load/store per k_block
                    c0 = tCrC[0]
                    c1 = tCrC[1]
                    c2 = tCrC[2]
                    c3 = tCrC[3]
                    c4 = tCrC[4]
                    c5 = tCrC[5]
                    c6 = tCrC[6]
                    c7 = tCrC[7]
                    a_s = tCrA[None, None, 0]
                    b_s = tCrB[None, None, 0]
                    for ki in cutlass.range(K_PER_WARP, unroll_full=True):
                        k_block = ki * NUM_MMA_WARPS + mma_warp_idx
                        cute.copy(
                            tiled_s2r_A,
                            tCsA_p[None, None, k_block],
                            tCrA_v[None, None, 0],
                        )
                        cute.copy(
                            tiled_s2r_B,
                            tCsB_p[None, None, k_block],
                            tCrB_v[None, None, 0],
                        )
                        c0, c1, c2, c3, c4, c5, c6, c7 = fused_fp8_mma_2n(
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                            a_s[0],
                            a_s[1],
                            a_s[2],
                            a_s[3],
                            a_s[4],
                            a_s[5],
                            a_s[6],
                            a_s[7],
                            b_s[0],
                            b_s[1],
                            b_s[2],
                            b_s[3],
                            b_s[4],
                            b_s[5],
                            b_s[6],
                            b_s[7],
                        )
                    tCrC[0] = c0
                    tCrC[1] = c1
                    tCrC[2] = c2
                    tCrC[3] = c3
                    tCrC[4] = c4
                    tCrC[5] = c5
                    tCrC[6] = c6
                    tCrC[7] = c7

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # Fused epilogue: reduce + direct global store (1 sync, no sC)
            smem_red_ptr = cute.arch.alloc_smem(
                cutlass.Float32, bM * bN * NUM_MMA_WARPS, alignment=16
            )

            # Each warp writes partial C via MMA partition (vectorized)
            smem_warp = cute.make_tensor(
                smem_red_ptr + mma_warp_idx * bM * bN,
                cute.make_layout((bM, bN), stride=(bN, 1)),
            )
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)
            cute.arch.sync_threads()

            # Reduce + write directly to global (skip sC)
            # 128 threads handle 256 elements (2 per thread)
            num_elems: cutlass.Constexpr = bM * bN
            elems_per_thread: cutlass.Constexpr = num_elems // self.num_mma_threads
            N_global = cute.size(mC, mode=[1])
            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * self.num_mma_threads + mma_tidx
                m = idx // bN
                n = idx % bN
                global_m = bid_m * bM + m
                global_n = bid_n * bN + n
                if global_m < cute.size(mC, mode=[0]):
                    if global_n < N_global:
                        total = cutlass.Float32(0.0)
                        for w in cutlass.range_constexpr(NUM_MMA_WARPS):
                            p = smem_red_ptr + w * bM * bN + idx
                            t = cute.make_tensor(p, cute.make_layout((1,)))
                            r = cute.make_rmem_tensor((1,), cutlass.Float32)
                            cute.autovec_copy(t, r)
                            total = total + r[0]
                        # Direct global store via output pointer
                        out_p = (mC.iterator + global_m * N_global + global_n).align(2)
                        out_t = cute.make_tensor(out_p, cute.make_layout((1,)))
                        out_r = cute.make_rmem_tensor((1,), self.out_dtype)
                        out_r[0] = total.to(self.out_dtype)
                        cute.autovec_copy(out_r, out_t)

        cute.arch.sync_threads()
