# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 dense GEMM1 (gate_up) with fused clamped-SwiGLU + MXFP8 quant epilogue.

One persistent SM100 tcgen05 MXFP8 GEMM tuned for prefill shapes (BM=128,
BN=128 per gate/up half, BK=128, cta_group=2; 6-warp CTA: warps 0-3
epilogue, warp 4 MMA, warp 5 TMA) computes ``gate = X @ Wg^T`` and
``up = X @ Wu^T`` and applies clamped SwiGLU and MXFP8 quantization. No
BF16 intermediate.

The tmem accumulator is double-buffered with a 32-col overlap (even tiles
[0,256), odd [224,480), SF fixed [500,512)): the epilogue reads the shared
32-col group first, so the next tile's MMA waits on a 32-col read instead
of 256. The single epilogue warpgroup processes a tile in 32-col chunks
(load -> act+quant -> store), overlap chunk first.

Tile order: flat (n-fastest) when the B operand fits in half of L2, else
grouped m-fastest (GROUP_M=32) so B panels stay L2-resident (dense TP1:
B=151MB > 126MB L2).
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    Float8E4M3FN,
    Float8E8M0FNU,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    cute,
)
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from vllm.cute_utils import _tcgen05, mbarrier, simple_tma_copy, to_cta0_smem
from vllm.cute_utils.cvt import fp32x4_to_fp8x4
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform


class Sm100DenseGemm1Mxfp8:
    def __init__(self, GROUP_M: int, use_pdl: bool = True) -> None:
        BM, BN, BK = 128, 128, 128
        self.cta_tile = (BM, BN, BK)
        self.GROUP_M = GROUP_M
        self.use_pdl = use_pdl

        smem_bytes = cutlass.memory.SmemAllocator.capacity_in_bytes()
        # A + Bg/2 + Bu/2 tiles + SFA + SFBg + SFBu (2-CTA: B halves, SF full)
        self.stage_size = (BM + BN) * BK + (BM + 256) * (BK // 32)
        self.num_stages = smem_bytes // self.stage_size

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: int, BK: int) -> cpasync.TmaInfo:
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cta_group=tcgen05.CtaGroup.TWO)
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        s_layout = cute.make_layout((BM, BK, self.num_stages), stride=(BK, 1, BM * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BM, BK))

    @cute.jit
    def prepare_SF(
        self,
        SF: cute.Tensor,
        R: Int32,
        Ksf: Int32,
        BR: int,
        BKsf: int,
        tma_op: cpasync.CopyG2STileBaseOp,
    ) -> cpasync.TmaInfo:
        g_layout = cute.make_layout((Ksf * 128, cute.ceil_div(R, 128)))
        SF_i64 = cute.recast_tensor(cute.make_tensor(SF.iterator, g_layout), Int64)
        tiler = (128 * BKsf // 8, BR // 128)
        s_layout = cute.make_layout((*tiler, self.num_stages))
        return cpasync.make_tiled_tma_atom(tma_op, SF_i64, s_layout, tiler)

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        SFA: cute.Tensor,
        SFB: cute.Tensor,
        Q: cute.Tensor,
        SFQ: cute.Tensor,
        limit: Float32,
        alpha: Float32,
        beta: Float32,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        M, K = A.shape
        N2, _ = B.shape  # 2 * N (gate|up)
        BM, BN, BK = self.cta_tile

        A_tma = self.prepare_AB(A, BM, BK)
        B_tma = self.prepare_AB(B, BN, BK)
        SFA_tma = self.prepare_SF(
            SFA,
            M,
            K // 32,
            BM,
            BK // 32,
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO),
        )
        SFB_tma = self.prepare_SF(
            SFB,
            N2,
            K // 32,
            BN,
            BK // 32,
            cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO),
        )

        self.kernel(A_tma, B_tma, SFA_tma, SFB_tma, Q, SFQ, limit, alpha, beta).launch(
            grid=(grid_size, 1, 1),
            block=(6 * 32, 1, 1),
            cluster=(2, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        SFA_tma: cpasync.TmaInfo,
        SFB_tma: cpasync.TmaInfo,
        Q_tensor: cute.Tensor,
        SFQ_tensor: cute.Tensor,
        limit: Float32,
        alpha: Float32,
        beta: Float32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        cta_rank = raw_bid % 2

        BM, BN, BK = self.cta_tile
        GROUP_M = self.GROUP_M
        num_stages = self.num_stages

        smem = cutlass.memory.SmemAllocator()
        sA = smem.allocate_tensor(
            Float8E4M3FN,
            A_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=A_tma.smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            Float8E4M3FN,
            B_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=B_tma.smem_layout.inner,
        )
        sSFA = smem.allocate_tensor(Int64, SFA_tma.smem_layout, byte_alignment=128)
        # SFB slots: rank r's multicast copy fills slot r of BOTH CTAs;
        # slice shape/stride per slot matches the atom's (64,1):(1,64) tile
        sSFB = smem.allocate_tensor(
            Int64,
            cute.make_layout((64, 1, 2, num_stages), stride=(1, 64, 64, 128)),
            byte_alignment=128,
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_partial_mbar = smem.allocate_array(Int64, 1)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)
        taddr = smem.allocate(Int32, 4)

        # allocate named barrier
        BAR_EPI = 1
        BAR_TMEM_ALLOC = 2

        M, K = A_tma.tma_tensor.shape
        N2, _ = B_tma.tma_tensor.shape
        N = N2 // 2
        grid_m = cute.ceil_div(M, BM * 2) * 2
        grid_n = N // BN
        total_tiles = grid_m * grid_n
        # output SF swizzle: NUM_K_TILES = ceil(N/32/4) tiles of 512B per 128 rows
        NUM_KT = cute.ceil_div(N // 32, 4)

        def compute_bid(bid, GROUP_M, grid_m, grid_n):
            if cutlass.const_expr(GROUP_M == 0):
                bid_m = (bid // (grid_n * 2)) * 2 + bid % 2
                bid_n = (bid // 2) % grid_n
            else:
                group_size = GROUP_M * grid_n
                group_id = bid // group_size
                group_m = min(grid_m - group_id * GROUP_M, GROUP_M)
                local_bid = bid - group_id * group_size
                bid_m = group_id * GROUP_M + local_bid % group_m
                bid_n = local_bid // group_m
            return bid_m, bid_n

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 2)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 256)
                cute.arch.mbarrier_init(tmem_partial_mbar, 256)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)
            cpasync.prefetch_descriptor(SFA_tma.atom)
            cpasync.prefetch_descriptor(SFB_tma.atom)

        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if warp_id == 5:
            # TMA warp
            tma_stage = 0
            parity = 1
            tma_full_mbar_ = to_cta0_smem(tma_full_mbar)

            gA_tiles = cute.zipped_divide(A_tma.tma_tensor, (BM, BK))
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN, BK))
            gSFA_tiles = cute.zipped_divide(
                SFA_tma.tma_tensor, (128 * (BK // 32) // 8, BM // 128)
            )
            gSFB_tiles = cute.zipped_divide(
                SFB_tma.tma_tensor, (128 * (BK // 32) // 8, BN // 128)
            )

            for work_id in range(raw_bid, total_tiles, num_bids):
                bid_m, bid_n = compute_bid(work_id, GROUP_M, grid_m, grid_n)

                for iter_k in range(cute.ceil_div(K, BK)):
                    mbar = tma_full_mbar_ + tma_stage
                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)
                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                    simple_tma_copy(
                        A_tma.atom,
                        gA_tiles[None, (bid_m, iter_k)],
                        sA[None, None, tma_stage],
                        mbar,
                    )
                    simple_tma_copy(
                        B_tma.atom,
                        gB_tiles[None, (cta_rank * grid_n + bid_n, iter_k)],
                        sB[None, None, tma_stage],
                        mbar,
                    )
                    simple_tma_copy(
                        SFA_tma.atom,
                        gSFA_tiles[None, (iter_k, bid_m)],
                        sSFA[None, None, tma_stage],
                        mbar,
                    )
                    # TMA multicast. rank0 loads gate, rank1 loads up
                    simple_tma_copy(
                        SFB_tma.atom,
                        gSFB_tiles[None, (iter_k, bid_n + cta_rank * grid_n)],
                        sSFB[None, None, cta_rank, tma_stage],
                        mbar,
                        mcast_mask=Uint16(3),
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        parity ^= 1

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        elif warp_id == 4:
            # MMA warp (2-CTA collective alloc; only cta_rank 0 issues)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            tma_stage = 0
            tma_full_parity = 0

            sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
            # SF smem descriptor: no swizzle, SBO = 8 * 16 bytes.
            sdesc_sf = Uint64((8 << 32) | (1 << 46))
            idesc = _tcgen05.make_mxfp8_idesc(BM * 2, BN * 2)
            multicast_mask = Uint16(3)

            # each [128,128] MXFP8 tile needs 4 tmem columns for SF
            sfa_tmem = 500
            sfbg_tmem = sfa_tmem + 4
            sfbu_tmem = sfbg_tmem + 4

            if cta_rank == 0:
                tmem_stage = 0
                tmem_parity = 1
                for _ in range(raw_bid, total_tiles, num_bids):
                    acc_base = 224 * tmem_stage
                    cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_parity)
                    # wait for the previous tile's overlap-region read; the
                    # single partial barrier phases once per tile
                    cute.arch.mbarrier_wait(tmem_partial_mbar, tmem_stage ^ 1)
                    _tcgen05.fence_after_thread_sync()

                    for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                        a_desc = sdesc | (
                            sA[None, None, tma_stage].iterator.toint() >> 4
                        )
                        b_desc = sdesc | (
                            sB[None, None, tma_stage].iterator.toint() >> 4
                        )
                        sfa_desc = sdesc_sf | (
                            sSFA[None, None, tma_stage].iterator.toint() >> 4
                        )
                        sfbg_desc = sdesc_sf | (
                            sSFB[None, None, 0, tma_stage].iterator.toint() >> 4
                        )
                        sfbu_desc = sdesc_sf | (
                            sSFB[None, None, 1, tma_stage].iterator.toint() >> 4
                        )

                        cute.arch.mbarrier_wait(
                            tma_full_mbar + tma_stage, tma_full_parity
                        )
                        _tcgen05.fence_after_thread_sync()

                        _tcgen05.cp(sfa_tmem, sfa_desc, "32x128b", "warpx4", 2)
                        _tcgen05.cp(sfbg_tmem, sfbg_desc, "32x128b", "warpx4", 2)
                        _tcgen05.cp(sfbu_tmem, sfbu_desc, "32x128b", "warpx4", 2)

                        for k in cutlass.range_constexpr(BK // 32):
                            enable_input_d = iter_k > 0 or k > 0
                            # select this k-block's SF byte inside the staged
                            # 4-col tmem atoms (idesc SF-id fields)
                            idesc_k = idesc + ((k << 4) | (k << 29))
                            _tcgen05.mma_mxfp8(
                                acc_base,
                                a_desc,
                                b_desc,
                                idesc_k,
                                sfa_tmem,
                                sfbg_tmem,
                                enable_input_d,
                                2,
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                        _tcgen05.commit(tma_empty_mbar + tma_stage, multicast_mask, 2)

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(tmem_full_mbar + tmem_stage, multicast_mask, 2)
                    tmem_stage ^= 1
                    if tmem_stage == 0:
                        tmem_parity ^= 1

        else:
            # epilogue warps (one warpgroup; tid < 128 indexes the tmem lane)
            if warp_id == 0:
                _tcgen05.alloc(taddr, cta_group=2)
                cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            exp_scale = -1.4426950216293334961 * alpha

            # ((32,4), (M, N/BN))
            Q_ = cute.zipped_divide(Q_tensor, (1, cute.make_layout((32, 4))))
            Q_ = Q_[(0, None), None]

            f8x32_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float8E4M3FN,
                num_bits_per_copy=256,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            partial_ = to_cta0_smem(tmem_partial_mbar)
            empty_ = to_cta0_smem(tmem_empty_mbar)

            tmem_stage = 0
            tmem_parity = 0
            for work_id in range(raw_bid, total_tiles, num_bids):
                bid_m, bid_n = compute_bid(work_id, GROUP_M, grid_m, grid_n)
                m = bid_m * BM + tid
                acc_base = 224 * tmem_stage

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, tmem_parity)
                cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=4 * 32)
                _tcgen05.fence_after_thread_sync()

                rGate = cute.make_rmem_tensor(32, Float32)
                rUp = cute.make_rmem_tensor(32, Float32)

                # cc=0 is the overlap chunk: cols [224,256) (up[96:128] on
                # even tiles, gate[0:32] on odd) plus the matching 32 cols of
                # the other operand. cc=1..3: the remaining chunk-pairs in
                # order. Each chunk: load -> act+quant -> store.
                for cc in cutlass.range_constexpr(4):
                    if cutlass.const_expr(cc == 0):
                        c = 3 - 3 * tmem_stage  # even tiles: 3; odd: 0
                        gate_col = 96 + 128 * tmem_stage
                        up_col = 224 + 128 * tmem_stage
                    else:
                        c = cc - 1 + tmem_stage  # even: 0,1,2; odd: 1,2,3
                        gate_col = acc_base + c * 32
                        up_col = acc_base + 128 + c * 32
                    rGate.store(_tcgen05.ld(tid // 32 * 32, gate_col, "32x32b", 32))
                    rUp.store(_tcgen05.ld(tid // 32 * 32, up_col, "32x32b", 32))
                    _tcgen05.wait_ld()
                    if cutlass.const_expr(cc == 0):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(partial_, "cluster")
                    if cutlass.const_expr(cc == 3):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(empty_ + tmem_stage, "cluster")

                    q_rmem = cute.make_rmem_tensor(32, Float8E4M3FN)
                    q_i32 = cute.recast_tensor(q_rmem, Uint32)

                    # activation
                    for j in cutlass.range_constexpr(32):
                        rGate[j] = cute.arch.fmin(rGate[j], limit)
                        rUp[j] = cute.arch.fmin(cute.arch.fmax(rUp[j], -limit), limit)

                    af = cute.make_rmem_tensor(32, Float32)
                    for j in cutlass.range(32, vectorize=True):
                        e = cute.math.exp2(rGate[j] * exp_scale, fastmath=True)
                        gate = rGate[j] * cute.arch.rcp_approx(1.0 + e)
                        af[j] = gate * (rUp[j] + beta)

                    # quantization
                    amax0 = cute.math.abs(af[0])
                    amax1 = cute.math.abs(af[1])
                    for j in cutlass.range_constexpr(15):
                        amax0 = cute.arch.fmax(amax0, cute.math.abs(af[j * 2 + 2]))
                        amax1 = cute.arch.fmax(amax1, cute.math.abs(af[j * 2 + 3]))
                    amax = cute.arch.fmax(amax0, amax1)

                    # Scale byte via exact integer math:
                    # ceil(log2(amax/448)) = E - 8 + (amax > 1.75 * 2^E),
                    # E = exponent(amax); amax=0 (all-zero block) clamps to 0.
                    bits = amax.bitcast(Int32)
                    exp_bits = bits & 0x7F800000
                    thr = (exp_bits | 0x600000).bitcast(Float32)  # 1.75 * 2^E
                    sb = (exp_bits >> 23) - 8 + Int32(amax > thr)
                    sb = cutlass.min(cutlass.max(sb, Int32(0)), Int32(254))

                    inv = ((Int32(254) - sb) << 23).bitcast(Float32)
                    for j in cutlass.range(32, vectorize=True):
                        af[j] *= inv

                    for j in cutlass.range_constexpr(8):
                        q_i32[j] = fp32x4_to_fp8x4(*[af[4 * j + t] for t in range(4)])

                    kb = bid_n * (BN // 32) + c
                    sf_off = (
                        (bid_m * NUM_KT + kb // 4) * 512
                        + (m % 32) * 16
                        + ((m // 32) % 4) * 4
                        + kb % 4
                    )
                    if m < M:
                        cute.copy(f8x32_atom, q_rmem, Q_[(None, c), (m, bid_n)])
                        SFQ_tensor[sf_off] = Uint8(sb)

                tmem_stage ^= 1
                if tmem_stage == 0:
                    tmem_parity ^= 1

            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp_id == 0:
                _tcgen05.dealloc(cta_group=2)

    @cache
    @staticmethod
    def compile(GROUP_M: int, use_pdl: bool = True):
        M = cute.sym_int()
        K = cute.sym_int()
        N2 = cute.sym_int()

        A = make_fake_tensor(
            Float8E4M3FN,
            (M, K),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        B = make_fake_tensor(
            Float8E4M3FN,
            (N2, K),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        SFA = make_fake_tensor(
            Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )
        SFB = make_fake_tensor(
            Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )
        Q = make_fake_tensor(
            Float8E4M3FN,
            (M, cute.sym_int(divisibility=256)),
            (cute.sym_int64(divisibility=256), 1),
            assumed_align=32,
        )
        SFQ = make_fake_tensor(
            Uint8, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )

        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100DenseGemm1Mxfp8(GROUP_M, use_pdl)
        return cute.compile(
            kernel,
            A,
            B,
            SFA,
            SFB,
            Q,
            SFQ,
            Float32(7.0),
            Float32(1.702),
            Float32(1.0),
            148,
            stream,
            options="--enable-tvm-ffi",
        )


# Below this the unfused GEMM -> act -> quant chain is as fast (TP4 shared
# expert, I=768: break-even at ~1024 tokens).
MIN_TOKENS = 1024


def _linear_kernel(linear):
    method = getattr(linear, "scheme", getattr(linear, "quant_method", None))
    return getattr(method, "kernel", None)


def can_fuse_gemm1(gate_up_proj, down_proj, act_fn) -> bool:
    """Whether fused_gemm1 can replace gate_up_proj + act_fn for down_proj.

    Called at init, before weight loading: kernels are selected at
    create_weights time, so the decision does not depend on the weights.
    """
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return False
    if not isinstance(act_fn, SiluAndMulWithClamp):
        return False
    # The kernel reads gate_up's fp8 weight and F8_128x4-swizzled weight
    # scale directly, so it depends on these kernels' weight layout.
    if not isinstance(
        _linear_kernel(gate_up_proj),
        (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel),
    ):
        return False
    down_kernel = _linear_kernel(down_proj)
    if down_kernel is None or down_kernel.input_quant_key() != kMxfp8Dynamic:
        return False
    # I % 256 == 0 (output row stride), K % 128 == 0 (BK).
    n = gate_up_proj.output_size_per_partition // 2
    k = gate_up_proj.input_size_per_partition
    return n % 256 == 0 and k % 128 == 0


def fused_gemm1(x: torch.Tensor, gate_up_proj, act_fn) -> QuantizedActivation:
    """act_fn(gate_up_proj(x)), MXFP8-quantized for an MXFP8 down_proj.

    Requires can_fuse_gemm1(gate_up_proj, down_proj, act_fn).
    """
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
    )

    assert x.ndim == 2 and x.dtype == torch.bfloat16
    weight = gate_up_proj.weight
    # FlashInferCutedsl stores the weight as a column-major [K, 2I] view.
    if isinstance(_linear_kernel(gate_up_proj), FlashInferCutedslMxfp8LinearKernel):
        weight = weight.t()

    M = x.shape[0]
    N = weight.shape[0] // 2
    q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty(
        ((M + 127) // 128) * 128 * (N // 32), dtype=torch.uint8, device=x.device
    )

    xq, xsf = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
    run(
        xq,
        weight,
        xsf.view(-1).view(torch.float8_e8m0fnu),
        gate_up_proj.weight_scale.view(torch.float8_e8m0fnu),
        q,
        sf,
        act_fn.swiglu_limit,
        act_fn.alpha,
        act_fn.beta,
    )
    return QuantizedActivation(
        data=q,
        scale=sf,
        orig_dtype=x.dtype,
        orig_shape=x.shape[:-1] + (N,),
        quant_key=kMxfp8Dynamic,
    )


def run(
    xq: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    limit: float,
    alpha: float,
    beta: float,
    grid: int | None = None,
) -> None:
    """Functional interface to the fused kernel (hides compile + grid sizing).

    xq: [M, K] fp8e4m3; weight: [2N, K] fp8e4m3 row-major; x_scale/weight_scale:
    F8_128x4-swizzled E8M0 (flat, any uint8-viewable dtype); q: [M, N] fp8e4m3;
    sf: flat uint8 scale buffer (round_up(M,128) * N/32 bytes; rows >= M are
    not written).
    """
    M, K = xq.shape
    _, N = q.shape
    if grid is None:
        grid_m2 = ((M + 127) // 128 + 1) // 2 * 2
        num_sms = torch.cuda.get_device_properties(xq.device).multi_processor_count
        grid = min(num_sms, grid_m2 * (N // 128))

    # Grouped (m-fastest) rasterization only pays when B (n2*k fp8 bytes)
    # exceeds half of L2; below that the flat order is faster (no div/mod
    # overhead in compute_bid)
    GROUP_M = 32 if N * 2 * K > (63 << 20) else 0
    kernel = Sm100DenseGemm1Mxfp8.compile(GROUP_M)
    kernel(xq, weight, x_scale, weight_scale, q, sf, limit, alpha, beta, grid)
