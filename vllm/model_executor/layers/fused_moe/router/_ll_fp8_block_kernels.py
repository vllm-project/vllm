# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL FP8 block-scaled GEMM: C[M,N] = (sA * A_fp8) @ (sB * B_fp8)^T.

Warp-specialized kernel with cp.async + mma.sync.m16n8k32.e4m3.
FP8 data passed as bf16 view (2 fp8 elements per bf16).
Block scales: packed ue8m0 int32, column-major layout.
Post-MMA scaling: acc *= scale_A[m, kb] * scale_B[n_block, kb] per 128-element K-block.
"""
import math
import torch
import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import sm90 as pipeline
from cutlass.cute.runtime import from_dlpack
from torch.cuda import current_stream


@dsl_user_op
def fused_fp8_mma_2n(
    c0, c1, c2, c3, c4, c5, c6, c7,
    a0_lo, a0_hi, a1_lo, a1_hi, a2_lo, a2_hi, a3_lo, a3_hi,
    b0_lo, b0_hi, b1_lo, b1_hi, b2_lo, b2_hi, b3_lo, b3_hi,
    *, loc=None, ip=None,
):
    """Fused: pack bf16 pairs + 2x mma.sync.m16n8k32.e4m3 (both N-atoms)."""
    f32 = cutlass.Float32.mlir_type
    i32 = _ir.IntegerType.get_signless(32)

    def _pack2(lo_ir, hi_ir):
        bf16_ty = lo_ir.type
        vec_ty = _ir.VectorType.get([2], bf16_ty)
        c0_ = _arith.constant(i32, 0, loc=loc, ip=ip)
        c1_ = _arith.constant(i32, 1, loc=loc, ip=ip)
        undef = _llvm.mlir_undef(vec_ty, loc=loc, ip=ip)
        v0 = _llvm.insertelement(undef, lo_ir, c0_, loc=loc, ip=ip)
        v1 = _llvm.insertelement(v0, hi_ir, c1_, loc=loc, ip=ip)
        return _llvm.bitcast(i32, v1, loc=loc, ip=ip)

    a0 = _pack2(a0_lo.ir_value(loc=loc, ip=ip), a0_hi.ir_value(loc=loc, ip=ip))
    a1 = _pack2(a1_lo.ir_value(loc=loc, ip=ip), a1_hi.ir_value(loc=loc, ip=ip))
    a2 = _pack2(a2_lo.ir_value(loc=loc, ip=ip), a2_hi.ir_value(loc=loc, ip=ip))
    a3 = _pack2(a3_lo.ir_value(loc=loc, ip=ip), a3_hi.ir_value(loc=loc, ip=ip))

    # N-atom 0
    b0_n0 = _pack2(b0_lo.ir_value(loc=loc, ip=ip), b0_hi.ir_value(loc=loc, ip=ip))
    b1_n0 = _pack2(b1_lo.ir_value(loc=loc, ip=ip), b1_hi.ir_value(loc=loc, ip=ip))

    asm_str = ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};")
    constraint = "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3"
    args_n0 = [a0, a1, a2, a3, b0_n0, b1_n0,
               c0.ir_value(loc=loc, ip=ip), c1.ir_value(loc=loc, ip=ip),
               c2.ir_value(loc=loc, ip=ip), c3.ir_value(loc=loc, ip=ip)]
    struct_ty = _llvm.StructType.get_literal([f32, f32, f32, f32])
    res0 = _llvm.inline_asm(struct_ty, args_n0, asm_str, constraint,
                            has_side_effects=True, loc=loc, ip=ip)
    r0 = _llvm.extractvalue(f32, res0, [0], loc=loc, ip=ip)
    r1 = _llvm.extractvalue(f32, res0, [1], loc=loc, ip=ip)
    r2 = _llvm.extractvalue(f32, res0, [2], loc=loc, ip=ip)
    r3 = _llvm.extractvalue(f32, res0, [3], loc=loc, ip=ip)

    # N-atom 1
    b0_n1 = _pack2(b2_lo.ir_value(loc=loc, ip=ip), b2_hi.ir_value(loc=loc, ip=ip))
    b1_n1 = _pack2(b3_lo.ir_value(loc=loc, ip=ip), b3_hi.ir_value(loc=loc, ip=ip))
    args_n1 = [a0, a1, a2, a3, b0_n1, b1_n1,
               c4.ir_value(loc=loc, ip=ip), c5.ir_value(loc=loc, ip=ip),
               c6.ir_value(loc=loc, ip=ip), c7.ir_value(loc=loc, ip=ip)]
    res1 = _llvm.inline_asm(struct_ty, args_n1, asm_str, constraint,
                            has_side_effects=True, loc=loc, ip=ip)
    r4 = _llvm.extractvalue(f32, res1, [0], loc=loc, ip=ip)
    r5 = _llvm.extractvalue(f32, res1, [1], loc=loc, ip=ip)
    r6 = _llvm.extractvalue(f32, res1, [2], loc=loc, ip=ip)
    r7 = _llvm.extractvalue(f32, res1, [3], loc=loc, ip=ip)

    return (cutlass.Float32(r0), cutlass.Float32(r1),
            cutlass.Float32(r2), cutlass.Float32(r3),
            cutlass.Float32(r4), cutlass.Float32(r5),
            cutlass.Float32(r6), cutlass.Float32(r7))


@dsl_user_op
def ue8m0_to_f32(packed_i32, byte_idx, *, loc=None, ip=None):
    """Extract one ue8m0 byte from packed int32 and convert to fp32 scale.

    ue8m0 format: 8-bit exponent, value = 2^(e - 127).
    packed_i32 contains 4 ue8m0 values as bytes.
    byte_idx selects which byte (0-3).
    """
    f32 = cutlass.Float32.mlir_type
    i32 = _ir.IntegerType.get_signless(32)
    val = packed_i32.ir_value(loc=loc, ip=ip)
    idx = byte_idx.ir_value(loc=loc, ip=ip)
    # Extract byte, shift to fp32 exponent position, reinterpret as float
    # f32 = 2^(e-127) is simply the float with exponent=e, mantissa=0
    # IEEE 754: float bits = (e << 23) when sign=0, mantissa=0
    res = _llvm.inline_asm(
        f32, [val, idx],
        "{"
        ".reg .u32 shift, byte_val, f_bits;"
        "shl.b32 shift, $2, 3;"              # shift = byte_idx * 8
        "shr.b32 byte_val, $1, shift;"       # byte_val = packed >> shift
        "and.b32 byte_val, byte_val, 0xFF;"  # mask to 8 bits
        "shl.b32 f_bits, byte_val, 23;"      # place as fp32 exponent
        "mov.b32 $0, f_bits;"                # reinterpret as float
        "}",
        "=f,r,r", has_side_effects=False, loc=loc, ip=ip)
    return cutlass.Float32(res)


@dsl_user_op
class LLFp8BlockGemm:
    def __init__(
        self,
        tile_n: int = 16,
        tile_k: int = 256,  # in bf16 units = 512 fp8 elements = 4 scale blocks
        num_stages: int = 2,
        num_dma_warps: int = 4,
        *, loc=None, ip=None,
    ):
        self.ab_dtype = cutlass.BFloat16  # bf16 view of fp8
        self.acc_dtype = cutlass.Float32
        self.out_dtype = cutlass.BFloat16
        self.tile_m = 16
        self.tile_n = tile_n
        self.tile_k = tile_k  # bf16 units
        self.tile_k_fp8 = tile_k * 2  # actual fp8 elements
        self.num_stages = num_stages
        self.mma_shape = (16, 8, 16)  # bf16 view MMA shape
        self.atom_layout = (1, 1, 1)
        self.num_mma_warps = 4
        self.num_dma_threads = num_dma_warps * 32
        self.num_mma_threads = self.num_mma_warps * 32
        self.num_threads = self.num_dma_threads + self.num_mma_threads
        # Number of 128-element FP8 scale blocks per K-tile
        self.scale_blocks_per_tile = self.tile_k_fp8 // 128

    def _make_smem_layout_AB(self, dtype, copy_bits, smem_tiler):
        major_size = min(smem_tiler[1], 64)
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = cute.make_layout(
            (8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy(self, atom_copy, dtype, copy_bits, num_threads):
        copy_elems = copy_bits // dtype.width
        k_threads = cute.size(self.tile_k) // copy_elems
        thread_layout = cute.make_layout(
            (num_threads // k_threads, k_threads), stride=(k_threads, 1))
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 mSA: cute.Tensor, mSB: cute.Tensor,
                 stream: CUstream):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, copy_bits, (bM, bK, self.num_stages))
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, copy_bits, (bN, bK, self.num_stages))
        atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type, num_bits_per_copy=copy_bits)
        tiled_copy_A = self._make_gmem_tiled_copy(
            atom_g2s, mA.element_type, copy_bits, self.num_dma_threads)
        tiled_copy_B = self._make_gmem_tiled_copy(
            atom_g2s, mB.element_type, copy_bits, self.num_dma_threads)
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_shape)
        perm_mnk = (
            self.atom_layout[0] * self.mma_shape[0],
            self.atom_layout[1] * self.mma_shape[1] * (self.tile_n // 8),
            self.atom_layout[2] * self.mma_shape[2])
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk)
        grid_m = cute.ceil_div(cute.size(mC, mode=[0]), bM)
        grid_n = cute.ceil_div(cute.size(mC, mode=[1]), bN)
        self.kernel(
            mA, mB, mC, mSA, mSB,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B, tiled_mma,
        ).launch(
            grid=[cute.size(grid_m), cute.size(grid_n), 1],
            block=[self.num_threads, 1, 1],
            stream=stream, use_pdl=False,
        )

    @cute.kernel
    def kernel(
        self, mA, mB, mC, mSA, mSB,
        sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
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
                cute.struct.MemRange[mA.element_type,
                                     cute.cosize(sA_layout)], 16]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type,
                                     cute.cosize(sB_layout)], 16]
            mbar: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, num_stages * 2], 8]

        smem = cutlass.utils.SmemAllocator()
        storage_ptr = smem.allocate(
            SharedStorage.size_in_bytes(), byte_alignment=16)  # type: ignore[attr-defined]
        storage = SharedStorage(storage_ptr)  # type: ignore[call-arg]
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)

        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_dma_threads)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_threads)
        mainloop_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
        )

        k_tile_count = cute.size(gA, mode=[2])

        if is_dma:
            # ===== DMA WARPS: load A/B tiles =====
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
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]),
                     cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0)),
                cutlass.Boolean)
            for rv in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rv, m, 0] = cute.elem_less(
                        tAcA[(0, rv), m, 0, 0][0], mA.shape[0])

            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBgB.shape[0][1], cute.size(tBgB, mode=[1]),
                     cute.size(tBgB, mode=[2])),
                    stride=(cute.size(tBgB, mode=[1]), 1, 0)),
                cutlass.Boolean)
            for rv in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rv, n, 0] = cute.elem_less(
                        tBcB[(0, rv), n, 0, 0][0], mB.shape[0])

            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_stages)

            for k_tile in range(k_tile_count):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(tiled_copy_A,
                          tAgA[None, None, None, k_tile],
                          tAsA[None, None, None, producer_state.index],
                          pred=tApA)
                cute.copy(tiled_copy_B,
                          tBgB[None, None, None, k_tile],
                          tBsB[None, None, None, producer_state.index],
                          pred=tBpB)
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            mainloop_pipeline.producer_tail(producer_state)

        else:
            # ===== MMA WARPS: FP8 MMA with block-scale application =====
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
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
                mA.element_type)
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
                mB.element_type)
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
            SCALE_BLOCKS: cutlass.Constexpr = self.scale_blocks_per_tile

            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_stages)

            # MMA thread → output position mapping for m16n8k32:
            # Each thread owns 2 M-rows and 2 N-positions (across 2 N-atoms)
            # m_row_0 = lane_id // 4, m_row_1 = m_row_0 + 8
            m_row_0 = lane_id // 4
            m_row_1 = m_row_0 + 8
            # N-block index for scale_B (all N within tile share same block
            # since tile_n < 128)
            n_block_idx = bid_n * bN // 128

            # K dimension in FP8 elements (bf16 view K * 2)
            K_fp8 = cute.size(mA, mode=[1]) * 2

            for k_tile in range(k_tile_count):
                mainloop_pipeline.consumer_wait(consumer_state)

                tCsA_p = tCsA_v[None, None, None, consumer_state.index]
                tCsB_p = tCsB_v[None, None, None, consumer_state.index]

                # FP8 MMA with per-scale-block accumulation
                # Each K-tile (tile_k bf16 = tile_k*2 fp8) contains
                # SCALE_BLOCKS groups of 128 fp8 elements.
                # mma.sync.m16n8k32 processes 32 fp8 elements per instruction.
                # K_PER_WARP k_blocks per warp, each k_block = 16 bf16 = 32 fp8
                # So 128 fp8 = 4 k_blocks = 1 scale group.
                # With 4 MMA warps interleaving, each warp does K_PER_WARP
                # k_blocks per tile.

                a_s = tCrA[None, None, 0]
                b_s = tCrB[None, None, 0]

                for ki in cutlass.range(K_PER_WARP, unroll_full=True):
                    k_block = ki * NUM_MMA_WARPS + mma_warp_idx
                    cute.copy(tiled_s2r_A,
                              tCsA_p[None, None, k_block],
                              tCrA_v[None, None, 0])
                    cute.copy(tiled_s2r_B,
                              tCsB_p[None, None, k_block],
                              tCrB_v[None, None, 0])

                    # Partial MMA for this k_block (32 fp8 elements)
                    p0 = cutlass.Float32(0.0)
                    p1 = cutlass.Float32(0.0)
                    p2 = cutlass.Float32(0.0)
                    p3 = cutlass.Float32(0.0)
                    p4 = cutlass.Float32(0.0)
                    p5 = cutlass.Float32(0.0)
                    p6 = cutlass.Float32(0.0)
                    p7 = cutlass.Float32(0.0)
                    p0, p1, p2, p3, p4, p5, p6, p7 = \
                        fused_fp8_mma_2n(
                            p0, p1, p2, p3, p4, p5, p6, p7,
                            a_s[0], a_s[1], a_s[2], a_s[3],
                            a_s[4], a_s[5], a_s[6], a_s[7],
                            b_s[0], b_s[1], b_s[2], b_s[3],
                            b_s[4], b_s[5], b_s[6], b_s[7])

                    # Block scale index: which 128-fp8-element group
                    # k_block processes 32 fp8 elements (= 16 bf16)
                    TILE_K_FP8: cutlass.Constexpr = self.tile_k_fp8
                    k_fp8_offset = k_tile * TILE_K_FP8 + k_block * 32
                    scale_k_idx = k_fp8_offset // 128

                    # Packed int32 index and byte position
                    packed_k = scale_k_idx // 4
                    byte_k = scale_k_idx - packed_k * 4  # modulo without %

                    # Scale layout: [M, K_packed] int32, COLUMN-MAJOR
                    # stride=(1, M) → element [m, kp] at offset kp * M + m
                    global_m0 = bid_m * bM + m_row_0
                    global_m1 = bid_m * bM + m_row_1
                    # Clamp to valid M range (MMA tile=16 may exceed actual M)
                    safe_m0 = global_m0 if global_m0 < M_out else M_out - 1
                    safe_m1 = global_m1 if global_m1 < M_out else M_out - 1

                    # Load packed int32 for scale_A (column-major: kp * M + m)
                    sa0_p = (mSA.iterator + packed_k * M_out + safe_m0).align(4)
                    sa0_t = cute.make_tensor(sa0_p, cute.make_layout((1,)))
                    sa0_r = cute.make_rmem_tensor((1,), cutlass.Int32)
                    cute.autovec_copy(sa0_t, sa0_r)
                    sa1_p = (mSA.iterator + packed_k * M_out + safe_m1).align(4)
                    sa1_t = cute.make_tensor(sa1_p, cute.make_layout((1,)))
                    sa1_r = cute.make_rmem_tensor((1,), cutlass.Int32)
                    cute.autovec_copy(sa1_t, sa1_r)

                    # Extract ue8m0 byte and convert to fp32
                    scale_a_m0 = ue8m0_to_f32(sa0_r[0], byte_k)
                    scale_a_m1 = ue8m0_to_f32(sa1_r[0], byte_k)

                    # Load packed int32 for scale_B (column-major: kp * N + n)
                    # weight_scale shape: [N, K_packed] stride=(1, N)
                    n_repr = n_block_idx * 128
                    sb_p = (mSB.iterator + packed_k * N_out + n_repr).align(4)
                    sb_t = cute.make_tensor(sb_p, cute.make_layout((1,)))
                    sb_r = cute.make_rmem_tensor((1,), cutlass.Int32)
                    cute.autovec_copy(sb_t, sb_r)
                    scale_b_val = ue8m0_to_f32(sb_r[0], byte_k)

                    # Apply: acc += partial * scale_A * scale_B
                    scale_m0 = scale_a_m0 * scale_b_val
                    scale_m1 = scale_a_m1 * scale_b_val
                    tCrC[0] = tCrC[0] + p0 * scale_m0
                    tCrC[1] = tCrC[1] + p1 * scale_m0
                    tCrC[2] = tCrC[2] + p2 * scale_m1
                    tCrC[3] = tCrC[3] + p3 * scale_m1
                    tCrC[4] = tCrC[4] + p4 * scale_m0
                    tCrC[5] = tCrC[5] + p5 * scale_m0
                    tCrC[6] = tCrC[6] + p6 * scale_m1
                    tCrC[7] = tCrC[7] + p7 * scale_m1

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # Epilogue: warp reduction + global store
            smem_red_ptr = cute.arch.alloc_smem(
                cutlass.Float32, bM * bN * NUM_MMA_WARPS, alignment=16)
            smem_warp = cute.make_tensor(
                smem_red_ptr + mma_warp_idx * bM * bN,
                cute.make_layout((bM, bN), stride=(bN, 1)))
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)
            cute.arch.sync_threads()

            num_elems: cutlass.Constexpr = bM * bN
            elems_per_thread: cutlass.Constexpr = (
                num_elems // self.num_mma_threads)
            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * self.num_mma_threads + mma_tidx
                m = idx // bN
                n = idx % bN
                global_m = bid_m * bM + m
                global_n = bid_n * bN + n
                if global_m < M_out:  # noqa: SIM102
                    if global_n < N_out:
                        total = cutlass.Float32(0.0)
                        for w in cutlass.range_constexpr(NUM_MMA_WARPS):
                            p = smem_red_ptr + w * bM * bN + idx
                            t = cute.make_tensor(
                                p, cute.make_layout((1,)))
                            r = cute.make_rmem_tensor(
                                (1,), cutlass.Float32)
                            cute.autovec_copy(t, r)
                            total = total + r[0]
                        out_p = (mC.iterator
                                 + global_m * N_out + global_n).align(2)
                        out_t = cute.make_tensor(
                            out_p, cute.make_layout((1,)))
                        out_r = cute.make_rmem_tensor(
                            (1,), self.out_dtype)
                        out_r[0] = total.to(self.out_dtype)
                        out_t[0] = out_r[0]

        cute.arch.sync_threads()


