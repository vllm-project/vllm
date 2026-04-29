import math
import ctypes

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm, llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op
from cuda.bindings.driver import CUstream, cuuint64_t, cuuint32_t
from cuda.bindings import driver as cu

from ._ll_a_gemm_kernels import fused_fp8_mma_2n

TMA_BOX_K = 64
NUM_SUB = 4
NUM_STAGES = 8


@dsl_user_op
def tma_load_2d(desc_ptr, smem_ptr, mbar_ptr, coord_x, coord_y, *, loc=None, ip=None):
    desc_llvm = desc_ptr.to_llvm_ptr(loc=loc, ip=ip) if hasattr(desc_ptr, 'to_llvm_ptr') else desc_ptr.ir_value(loc=loc, ip=ip)
    smem_llvm = smem_ptr.to_llvm_ptr(loc=loc, ip=ip) if hasattr(smem_ptr, 'to_llvm_ptr') else smem_ptr.ir_value(loc=loc, ip=ip)
    mbar_llvm = mbar_ptr.to_llvm_ptr(loc=loc, ip=ip) if hasattr(mbar_ptr, 'to_llvm_ptr') else mbar_ptr.ir_value(loc=loc, ip=ip)
    coords = [coord_x.ir_value(loc=loc, ip=ip), coord_y.ir_value(loc=loc, ip=ip)]
    nvvm.CpAsyncBulkTensorGlobalToSharedClusterOp(
        dstMem=smem_llvm, tmaDescriptor=desc_llvm,
        coordinates=coords, mbar=mbar_llvm,
        im2colOffsets=[], loc=loc, ip=ip)


@dsl_user_op
def prefetch_tensormap(desc_ptr, *, loc=None, ip=None):
    desc_llvm = desc_ptr.to_llvm_ptr(loc=loc, ip=ip) if hasattr(desc_ptr, "to_llvm_ptr") else desc_ptr.ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        res=None, operands_=[desc_llvm],
        asm_string="prefetch.tensormap [$0];",
        constraints="l",
        has_side_effects=True, loc=loc, ip=ip)

@dsl_user_op
def mbarrier_try_wait(mbar_ptr, phase, *, loc=None, ip=None):
    """Non-blocking try_wait. Returns 1 if phase completed, 0 otherwise."""
    mbar_llvm = mbar_ptr.to_llvm_ptr(loc=loc, ip=ip) if hasattr(mbar_ptr, 'to_llvm_ptr') else mbar_ptr.ir_value(loc=loc, ip=ip)
    phase_ir = phase.ir_value(loc=loc, ip=ip)
    i32 = cutlass.Int32.mlir_type
    result = _llvm.inline_asm(
        i32, [mbar_llvm, phase_ir],
        "{\n.reg .pred P1;\nmbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\nselp.b32 $2, 1, 0, P1;\n}\n",
        "r,r,=r",
        has_side_effects=True, loc=loc, ip=ip)
    return cutlass.Int32(result)


@dsl_user_op
def mbarrier_arrive(mbar_ptr, *, loc=None, ip=None):
    mbar_llvm = mbar_ptr.to_llvm_ptr(loc=loc, ip=ip) if hasattr(mbar_ptr, 'to_llvm_ptr') else mbar_ptr.ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        res=None, operands_=[mbar_llvm],
        asm_string="mbarrier.arrive.shared::cta.b64 _, [$0];",
        constraints="r",
        has_side_effects=True, loc=loc, ip=ip)


def create_tma_descriptor(tensor, box_rows, box_cols=TMA_BOX_K):
    rows, cols = tensor.shape
    err, desc = cu.cuTensorMapEncodeTiled(
        cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, tensor.data_ptr(),
        [cuuint64_t(cols), cuuint64_t(rows)],
        [cuuint64_t(cols * 2)],
        [cuuint32_t(box_cols), cuuint32_t(box_rows)],
        [cuuint32_t(1), cuuint32_t(1)],
        cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
    host_ptr = desc.getPtr()
    desc_bytes = (ctypes.c_ubyte * 128).from_address(host_ptr)
    return bytearray(desc_bytes)


class LLAGemmTmaPrefetch:
    """Pipelined TMA A GEMM with 3-barrier protocol, k-phase interleaving."""

    def __init__(self, tile_k=256, num_stages=NUM_STAGES, K_eff=0, prefetch_tiles=0,
                 is_fp8=False):
        self.tile_m = 16
        self.tile_n = 16
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.is_fp8 = is_fp8
        self.K_eff = K_eff
        k_tiles = prefetch_tiles if prefetch_tiles > 0 else (K_eff // tile_k if K_eff > 0 else 0)
        self.k_tiles = k_tiles
        self.k_loops_dma = (k_tiles + 1) // 2
        self.mma_shape = (16, 8, 16)
        self.atom_layout = (1, 1, 1)
        self.num_mma_warps = 4
        self.num_threads = 256

    def _make_smem_layout_AB(self, dtype, smem_tiler):
        copy_bits = 128
        major_size = min(smem_tiler[1], 64)
        swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = cute.make_layout(
            (8, major_size), stride=(major_size, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer)
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    @cute.jit
    def __call__(self, mA, mB, mC,
                 descA_tensor, descB_tensor,
                 stream: CUstream):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k

        op = cute.nvgpu.warp.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, self.mma_shape)
        perm_mnk = (
            self.atom_layout[0] * self.mma_shape[0],
            self.atom_layout[1] * self.mma_shape[1] * 2,
            self.atom_layout[2] * self.mma_shape[2])
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout(self.atom_layout), permutation_mnk=perm_mnk)

        sA_layout = self._make_smem_layout_AB(
            mA.element_type, (bM, bK, self.num_stages))
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, (bN, bK, self.num_stages))

        grid_m = cute.ceil_div(cute.size(mC, mode=[0]), bM)
        grid_n = cute.ceil_div(cute.size(mC, mode=[1]), bN)

        self.kernel(
            mA, mB, mC,
            descA_tensor, descB_tensor,
            sA_layout, sB_layout,
            tiled_mma,
        ).launch(
            grid=[cute.size(grid_m), cute.size(grid_n), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, mA, mB, mC,
               descA_tensor: cute.Tensor,
               descB_tensor: cute.Tensor,
               sA_layout: cute.ComposedLayout,
               sB_layout: cute.ComposedLayout,
               tiled_mma: cute.TiledMma):
        bM, bN, bK = self.tile_m, self.tile_n, self.tile_k
        STAGES: cutlass.Constexpr = self.num_stages
        K_TILES: cutlass.Constexpr = self.k_tiles
        K_LOOPS_DMA: cutlass.Constexpr = self.k_loops_dma
        NUM_MMA_WARPS: cutlass.Constexpr = self.num_mma_warps
        SUB_K: cutlass.Constexpr = TMA_BOX_K
        SUB_STRIDE: cutlass.Constexpr = 1024
        TILE_STRIDE: cutlass.Constexpr = bM * bK

        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // 32
        lane_id = tidx % 32
        bid_m, bid_n, _ = cute.arch.block_idx()

        # Shared storage
        @cute.struct
        class SharedStorage:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 128]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 128]
            bar_b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, STAGES], 8]
            bar_a: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, STAGES], 8]
            bar_consumed: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, STAGES], 8]
            red: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, bM * bN * NUM_MMA_WARPS], 16]

        smem = cutlass.utils.SmemAllocator()
        sptr = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=128)
        storage = SharedStorage(sptr)
        sA = storage.a.get_tensor(sA_layout)
        sB = storage.b.get_tensor(sB_layout)

        bar_b_base = storage.bar_b.data_ptr()
        bar_a_base = storage.bar_a.data_ptr()
        bar_c_base = storage.bar_consumed.data_ptr()
        sA_base = storage.a.data_ptr()
        sB_base = storage.b.data_ptr()
        descA_ptr = descA_tensor.iterator
        descB_ptr = descB_tensor.iterator
        smem_red_ptr = storage.red.data_ptr()

        # Init barriers: consumed=128 (all 4 compute warps × 32 threads)
        if tidx == 0:
            for i in range(STAGES):
                cute.arch.mbarrier_init(bar_b_base + i, 1)
                cute.arch.mbarrier_init(bar_a_base + i, 1)
                cute.arch.mbarrier_init(bar_c_base + i, 128)
        if tidx == 0:
            prefetch_tensormap(descA_ptr)
            prefetch_tensormap(descB_ptr)
        cute.arch.sync_threads()

        # === A DMA warps (6-7) ===
        if warp_id >= 6:
            cute.arch.setmaxregister_decrease(40)
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

            a_tile_bytes: cutlass.Constexpr = bM * SUB_K * mA.element_type.width // 8 * NUM_SUB
            dma_off_a = (warp_id - 6) % 2

            if lane_id == 0:
                stg_a = dma_off_a
                ph_a = cutlass.Int32(0)
                for ki in range(K_LOOPS_DMA):
                    kt_a = ki * 2 + dma_off_a
                    cute.arch.mbarrier_wait(bar_c_base + stg_a, ph_a ^ 1)
                    bar_a_ptr = bar_a_base + stg_a
                    if kt_a < K_TILES:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_a_ptr, a_tile_bytes)
                        for j in range(NUM_SUB):
                            dest_a = sA_base + stg_a * TILE_STRIDE + j * SUB_STRIDE
                            ck_a = kt_a * bK + j * SUB_K
                            tma_load_2d(descA_ptr, dest_a, bar_a_ptr,
                                        cutlass.Int32(ck_a), cutlass.Int32(bid_m * bM))
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(bar_a_ptr, 0)
                    stg_a = stg_a + 2
                    if stg_a >= STAGES:
                        stg_a = dma_off_a
                        ph_a = ph_a ^ 1
                for i in range(STAGES // 2 - 1):
                    cute.arch.mbarrier_wait(bar_c_base + stg_a, ph_a ^ 1)
                    stg_a = stg_a + 2
                    if stg_a >= STAGES:
                        stg_a = dma_off_a
                        ph_a = ph_a ^ 1

        # === B DMA warps (4-5) ===
        if warp_id >= 4:
            if warp_id < 6:
                cute.arch.setmaxregister_decrease(40)
                b_tile_bytes: cutlass.Constexpr = bN * SUB_K * mB.element_type.width // 8 * NUM_SUB
                dma_off_b = (warp_id - 4) % 2

                if lane_id == 0:
                    stg_b = dma_off_b
                    ph_b = cutlass.Int32(0)
                    for ki in range(K_LOOPS_DMA):
                        kt_b = ki * 2 + dma_off_b
                        cute.arch.mbarrier_wait(bar_c_base + stg_b, ph_b ^ 1)
                        bar_b_ptr = bar_b_base + stg_b
                        if kt_b < K_TILES:
                            cute.arch.mbarrier_arrive_and_expect_tx(bar_b_ptr, b_tile_bytes)
                            for j in range(NUM_SUB):
                                dest_b = sB_base + stg_b * TILE_STRIDE + j * SUB_STRIDE
                                ck_b = kt_b * bK + j * SUB_K
                                tma_load_2d(descB_ptr, dest_b, bar_b_ptr,
                                            cutlass.Int32(ck_b), cutlass.Int32(bid_n * bN))
                        else:
                            cute.arch.mbarrier_arrive_and_expect_tx(bar_b_ptr, 0)
                        stg_b = stg_b + 2
                        if stg_b >= STAGES:
                            stg_b = dma_off_b
                            ph_b = ph_b ^ 1
                    for i in range(STAGES // 2 - 1):
                        cute.arch.mbarrier_wait(bar_c_base + stg_b, ph_b ^ 1)
                        stg_b = stg_b + 2
                        if stg_b >= STAGES:
                            stg_b = dma_off_b
                            ph_b = ph_b ^ 1

        else:
            # === Compute warps 0-3: k-phase interleaving ===
            cute.arch.setmaxregister_increase(232)
            compute_warp = warp_id

            thr_mma = tiled_mma.get_slice(lane_id)
            gC = cute.local_tile(mC, (bM, bN), (bid_m, bid_n))
            tCgC = thr_mma.partition_C(gC)

            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mA.element_type)
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), mB.element_type)
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(lane_id)
            thr_s2r_B = tiled_s2r_B.get_slice(lane_id)

            tCsA_v = thr_s2r_A.partition_S(sA)
            tCsB_v = thr_s2r_B.partition_S(sB)

            tCsA_mma = thr_mma.partition_A(sA)
            tCsB_mma = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA_mma[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB_mma[None, None, None, 0])
            tCrA_v = thr_s2r_A.retile(tCrA)
            tCrB_v = thr_s2r_B.retile(tCrB)

            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            num_k_block = cute.size(tCrA, mode=[2])
            K_PER_WARP: cutlass.Constexpr = num_k_block // NUM_MMA_WARPS

            # Main loop: ALL warps process ALL k_tiles, k-phase interleaved
            # Pre-check first stage (speculative prefetch for fp8 try_wait path)
            b_rdy = mbarrier_try_wait(bar_b_base + 0, cutlass.Int32(0))
            a_rdy = mbarrier_try_wait(bar_a_base + 0, cutlass.Int32(0))

            for k_tile in range(K_TILES):
                STAGE: cutlass.Constexpr = k_tile % STAGES
                PHASE: cutlass.Constexpr = (k_tile // STAGES) % 2

                if not self.is_fp8:
                    # bf16: blocking wait
                    cute.arch.mbarrier_wait(bar_b_base + STAGE, PHASE)
                    cute.arch.mbarrier_wait(bar_a_base + STAGE, PHASE)
                else:
                    # fp8: try_wait spin loop with speculative prefetch
                    while b_rdy + a_rdy < 2:
                        b_rdy = mbarrier_try_wait(bar_b_base + STAGE, PHASE)
                        a_rdy = mbarrier_try_wait(bar_a_base + STAGE, PHASE)

                # MMA: k-phase interleaving (each warp does K_PER_WARP k_blocks)
                tCsA_p = tCsA_v[None, None, None, STAGE]
                tCsB_p = tCsB_v[None, None, None, STAGE]

                if not self.is_fp8:
                    for ki in cutlass.range(K_PER_WARP, unroll_full=True):
                        k_block = ki * NUM_MMA_WARPS + compute_warp
                        cute.copy(tiled_s2r_A, tCsA_p[None, None, k_block],
                                  tCrA_v[None, None, 0])
                        cute.copy(tiled_s2r_B, tCsB_p[None, None, k_block],
                                  tCrB_v[None, None, 0])
                        cute.gemm(tiled_mma, tCrC,
                                  tCrA[None, None, 0], tCrB[None, None, 0], tCrC)
                else:
                    c0 = tCrC[0]; c1 = tCrC[1]; c2 = tCrC[2]; c3 = tCrC[3]
                    c4 = tCrC[4]; c5 = tCrC[5]; c6 = tCrC[6]; c7 = tCrC[7]
                    a_s = tCrA[None, None, 0]
                    b_s = tCrB[None, None, 0]
                    for ki in cutlass.range(K_PER_WARP, unroll_full=True):
                        k_block = ki * NUM_MMA_WARPS + compute_warp
                        cute.copy(tiled_s2r_A, tCsA_p[None, None, k_block],
                                  tCrA_v[None, None, 0])
                        cute.copy(tiled_s2r_B, tCsB_p[None, None, k_block],
                                  tCrB_v[None, None, 0])
                        c0, c1, c2, c3, c4, c5, c6, c7 = fused_fp8_mma_2n(
                            c0, c1, c2, c3, c4, c5, c6, c7,
                            a_s[0], a_s[1], a_s[2], a_s[3],
                            a_s[4], a_s[5], a_s[6], a_s[7],
                            b_s[0], b_s[1], b_s[2], b_s[3],
                            b_s[4], b_s[5], b_s[6], b_s[7])
                    tCrC[0] = c0; tCrC[1] = c1; tCrC[2] = c2; tCrC[3] = c3
                    tCrC[4] = c4; tCrC[5] = c5; tCrC[6] = c6; tCrC[7] = c7

                # Speculatively prefetch next stage barriers (fp8 only)
                if self.is_fp8:
                    if k_tile + 1 < K_TILES:
                        NEXT_STAGE: cutlass.Constexpr = (k_tile + 1) % STAGES
                        NEXT_PHASE: cutlass.Constexpr = ((k_tile + 1) // STAGES) % 2
                        b_rdy = mbarrier_try_wait(bar_b_base + NEXT_STAGE, NEXT_PHASE)
                        a_rdy = mbarrier_try_wait(bar_a_base + NEXT_STAGE, NEXT_PHASE)

                # All 128 compute threads signal data consumed
                mbarrier_arrive(bar_c_base + STAGE)

            # Write accum to reduction buffer
            smem_warp = cute.make_tensor(
                smem_red_ptr + compute_warp * bM * bN,
                cute.make_layout((bM, bN), stride=(bN, 1)))
            tCsC_partial = thr_mma.partition_C(smem_warp)
            cute.autovec_copy(tCrC, tCsC_partial)

        # ALL 256 threads sync
        cute.arch.sync_threads()

        # All 4 compute warps reduce and store (128 threads, 2 elems each)
        if warp_id < 4:
            num_elems: cutlass.Constexpr = bM * bN
            elems_per_thread: cutlass.Constexpr = num_elems // (NUM_MMA_WARPS * 32)
            N_global = cute.size(mC, mode=[1])
            for ei in cutlass.range_constexpr(elems_per_thread):
                idx = ei * (NUM_MMA_WARPS * 32) + tidx
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
                        out_p = (mC.iterator + global_m * N_global + global_n).align(2)
                        out_t = cute.make_tensor(out_p, cute.make_layout((1,)))
                        out_r = cute.make_rmem_tensor((1,), cutlass.BFloat16)
                        out_r[0] = total.to(cutlass.BFloat16)
                        cute.autovec_copy(out_r, out_t)
