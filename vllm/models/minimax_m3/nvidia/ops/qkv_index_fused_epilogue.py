# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 QKV+indexer GEMM with a fused QK-norm/RoPE + KV-insert epilogue.

Replaces mm_mxfp8 + fused_minimax_m3_qknorm_rope_kv_insert for the MiniMax-M3
qkv_proj with one persistent SM100 kernel. Per 128-col head tile, the
epilogue applies Gemma RMSNorm + partial NeoX RoPE to the f32 accumulator
(no bf16 round trip) and writes q (bf16, optionally fp8) and index_q (fp8),
or NVFP4-quantizes k / v / index_k into the paged caches. The qkv projection
is never materialized.
"""

from dataclasses import dataclass
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
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
from cutlass.cute.runtime import make_fake_tensor

from vllm.cute_utils import _tcgen05, mbarrier, simple_tma_copy, to_cta0_smem
from vllm.cute_utils.cvt import (
    bf16x2_to_fp32x2,
    fp32x8_to_fp4x8,
)
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import VllmCuTeDSLJitKernel

HEAD_DIM = 128  # one 128-col N-tile per head
# epilogue outputs are consumed by other kernels: stores bypass L1
_NO_L1_ALLOC = dict(level1_eviction_priority="evict_no_allocate")


@cute.jit
def _rms_norm(acc, sW, eps):
    """Gemma RMS-norm the 128-col head row in acc in place; sW is the head
    type's bf16 weight as (8, 16) tiles."""
    acc1 = cute.make_tensor(acc.iterator, cute.make_layout(HEAD_DIM))
    # 4 independent fma chains; starting from 0 makes every step
    # fma(x, x, ss) (x0 * x0 + x1 * x1 would let ptxas fuse either product)
    ss = cute.full(4, 0.0, Float32)
    for j in cutlass.range_constexpr(HEAD_DIM // 4):
        x = cute.local_tile(acc1, (4,), (j,)).load()
        ss = ss + x * x
    t = (ss[0] + ss[1] + ss[2] + ss[3]) * Float32(1.0 / HEAD_DIM) + eps
    # t >= eps, so rsqrt.approx needs no subnormal handling
    rms = cute.math.rsqrt(t, approx=True)

    # load 8 norm weights at a time -> 16-byte loads
    for j in cutlass.range_constexpr(HEAD_DIM // 8):
        w_f32 = bf16x2_to_fp32x2(sW[None, j].load())
        x = cute.local_tile(acc1, (8,), (j,))
        n = x.load() * rms
        x.store(n + n * w_f32)


def _rope64(acc, CS, pos):
    """NeoX-rope cols [0, 64) of acc in place, rotating chunk 0 (x) with
    chunk 1 (y); CS is the bf16 cos | sin cache as [32, (pos, 2)]."""
    cos = bf16x2_to_fp32x2(CS[None, (pos, 0)].load())
    sin = bf16x2_to_fp32x2(CS[None, (pos, 1)].load())
    x = acc[None, 0].load()
    y = acc[None, 1].load()
    acc[None, 0].store(x * cos - y * sin)
    acc[None, 1].store(y * cos + x * sin)


def _nvfp4_cache_views(C, bs, nslots):
    """Views of the [num_pages, page_bytes] NVFP4 cache C. A page holds nslots
    slots of bs tokens' E2M1 data (64B each), then their E4M3 scales (8B
    each). Returns the data as u32 words [8, half, token, slot, page], the
    linear scales as [8, token, slot, page], and the V-swizzled scales as
    [(2, 4), token, slot, page]: byte h of token t at
    (t // 4 * 4 + h // 2) * 8 + h % 2 * 4 + t % 4."""
    slot_bytes = bs * (HEAD_DIM // 2 + HEAD_DIM // 16)
    sf_off = bs * HEAD_DIM // 2
    nb, page = C.shape[0], C.stride[0]
    data = cute.make_tensor(
        C.iterator,
        cute.make_layout((32, 2, bs, nslots, nb), stride=(1, 32, 64, slot_bytes, page)),
    )
    sf = cute.make_tensor(
        C.iterator + sf_off,
        cute.make_layout((8, bs, nslots, nb), stride=(1, 8, slot_bytes, page)),
    )
    vsf = cute.make_tensor(
        C.iterator + sf_off,
        cute.make_layout(
            ((2, 4), (4, bs // 4), nslots, nb),
            stride=((4, 8), (1, 32), slot_bytes, page),
        ),
    )
    return cute.recast_tensor(data, Uint32), sf, vsf


def _nvfp4_quant32(vals, words, sf, sf_scale):
    """NVFP4-quantize 32 f32 (two 16-value groups) into 4 data words and the
    2 scale bytes `sf`, following
    fused_minimax_m3_qknorm_rope_kv_insert's quantizer. sf_scale is
    1 / (global dequant scale), or None for unit scale."""
    # E4M3(amax / 6) floored at 2^-9 (satfinite caps it at 448); the unfused
    # op's 1e-12 amax floor lies below that and is omitted. Both groups'
    # scales share one e4m3x2 cvt.
    amax_6 = cute.make_rmem_tensor(2, Float32)
    for g in range(2):
        gmax = Float32(0.0)
        for i in range(16 * g, 16 * g + 16):
            gmax = cute.arch.fmax(gmax, cute.math.abs(vals[i]))
        a = gmax * Float32(1.0 / 6.0)
        if sf_scale is not None:
            a = a * sf_scale
        amax_6[g] = cute.arch.fmax(a, Float32(2.0**-9))

    sf8 = amax_6.load().to(Float8E4M3FN)
    sf.store(sf8.bitcast(Uint8))
    sf_f32 = sf8.to(Float32)

    q = cute.make_rmem_tensor(32, Float32)
    for g in range(2):
        # rcp.approx.ftz is exact on every positive E4M3 value
        r = cute.arch.rcp_approx(sf_f32[g])
        if sf_scale is not None:
            r = r * sf_scale
        # rz: r >= 1 / sf, so a rounded product could step past an exact E2M1
        # midpoint and round away from even; truncation lands back on it (raw
        # v accumulators do hit exact midpoints, rarely)
        for i in range(16 * g, 16 * g + 16, 2):
            q[i], q[i + 1] = cute.arch.mul_packed_f32x2(
                (vals[i], vals[i + 1]), (r, r), rnd="rz"
            )

    # clear the sign of zero-magnitude nibbles (+7 carries into bit 3 exactly
    # when the 3-bit magnitude is nonzero)
    m = Uint32(0x77777777)
    for j in range(4):
        w = fp32x8_to_fp4x8(q, 8 * j)
        words[j] = w & (((w & m) + m) | m)


def _nvfp4_quant64(vals, sf_scale=None):
    """NVFP4-quantize 64 f32 (a 64-col half, one 32-col chunk at a time);
    returns the 8 data words and the 4 scale bytes."""
    words = cute.make_rmem_tensor(8, Uint32)
    sf = cute.make_rmem_tensor(4, Uint8)
    for c in range(2):
        _nvfp4_quant32(
            cute.local_tile(vals, (32,), (c,)),
            cute.local_tile(words, (4,), (c,)),
            cute.local_tile(sf, (2,), (c,)),
            sf_scale,
        )
    return words.load(), sf.load()


class QkvIndexFusedEpilogueKernel:
    """Persistent MXFP8 GEMM with a BM x BN = 128 x (128 * cta_group) output
    tile per CTA (one head per epilogue warpgroup). 2-SM (cta_group=2): a CTA
    pair runs one m256n256 MMA per k-step over two adjacent M-tiles, each CTA
    loading half of the BN rows of B. 1-SM (cta_group=1): one m128n128 MMA per
    CTA (finer tiles for small M). Warps: 4 * cta_group epilogue, then MMA and
    TMA; cta_group=2 adds 2 padding warps so setmaxnreg can move registers to
    the epilogue (at cta_group=1, 6 warps already fit 255 regs each).
    """

    def __init__(
        self,
        nkv: int,
        block_size: int,
        cta_group: int = 2,
        group_m: int = 1,
        use_pdl: bool = True,
    ) -> None:
        assert nkv in (1, 2, 4)
        assert block_size % 4 == 0  # V scales swizzle over 4-token quads
        assert cta_group in (1, 2)
        self.cta_group = cta_group
        self.group_m = group_m
        self.num_warps = 12 if cta_group == 2 else 6
        self.tma_group = (
            tcgen05.CtaGroup.TWO if cta_group == 2 else tcgen05.CtaGroup.ONE
        )
        self.use_pdl = use_pdl
        self.nkv = nkv
        self.block_size = block_size

        # The N dim is a list of 128-col heads, [q x 16nkv | k x nkv |
        # v x nkv | index_q x nkv | index_k x 1]; first head of each role
        # (q starts at 0). An odd head count leaves a phantom head in the
        # last tile at cta_group=2 (TMA zero-fill, never stored).
        self.k_start = 16 * nkv
        self.v_start = self.k_start + nkv
        self.iq_start = self.v_start + nkv
        self.ik_start = self.iq_start + nkv

        BM, BN, BK = 128, cta_group * HEAD_DIM, 128
        self.cta_tile = (BM, BN, BK)
        # per CTA and stage: A + this CTA's share of B (cta_group=2 splits B
        # across the pair) + SFA + full SFB (each CTA multicasts its half)
        self.stage_size = (BM + BN // cta_group) * BK + (BM + BN) * (BK // 32)
        self.num_stages = (
            cutlass.memory.SmemAllocator.capacity_in_bytes() // self.stage_size
        )

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BM: int, BK: int) -> cpasync.TmaInfo:
        tma_op = cpasync.CopyBulkTensorTileG2SOp(cta_group=self.tma_group)
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
        tma_op: cpasync.CopyG2STileBaseOp,
    ) -> cpasync.TmaInfo:
        # SF grid [R, Ksf] padded to R % 128 == 0 and permuted as
        # [R/128, Ksf/4, 32, 4, 4]; the [32, 4, 4] atom is tcgen05's opaque
        # 512B block, so treat it as bytes. One box = one atom (128 rows x
        # BK / 32 = 4 K-groups) = 64 Int64 x 1 row-block.
        g_layout = cute.make_layout((Ksf * 128, cute.ceil_div(R, 128)))
        SF_i64 = cute.recast_tensor(cute.make_tensor(SF.iterator, g_layout), Int64)
        s_layout = cute.make_layout((64, 1, self.num_stages))
        return cpasync.make_tiled_tma_atom(tma_op, SF_i64, s_layout, (64, 1))

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        SFA: cute.Tensor,
        SFB: cute.Tensor,
        Q: cute.Tensor | None,
        WQ: cute.Tensor,
        WK: cute.Tensor,
        WIQ: cute.Tensor,
        WIK: cute.Tensor,
        COS_SIN: cute.Tensor,
        POS: cute.Tensor,
        KVC: cute.Tensor,
        IDXC: cute.Tensor,
        SLOT: cute.Tensor,
        IDX_SLOT: cute.Tensor,
        IQFP8: cute.Tensor,
        QFP8: cute.Tensor | None,
        eps: Float32,
        q_fp8_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        M, K = A.shape
        N, _ = B.shape
        BM, BN, BK = self.cta_tile

        A_tma = self.prepare_AB(A, BM, BK)
        B_tma = self.prepare_AB(B, BN // self.cta_group, BK)
        SFA_tma = self.prepare_SF(
            SFA, M, K // 32, cpasync.CopyBulkTensorTileG2SOp(self.tma_group)
        )
        sfb_op = (
            cpasync.CopyBulkTensorTileG2SMulticastOp(self.tma_group)
            if self.cta_group == 2
            else cpasync.CopyBulkTensorTileG2SOp(self.tma_group)
        )
        SFB_tma = self.prepare_SF(SFB, N, K // 32, sfb_op)

        self.kernel(
            A_tma,
            B_tma,
            SFA_tma,
            SFB_tma,
            Q,
            WQ,
            WK,
            WIQ,
            WIK,
            COS_SIN,
            POS,
            KVC,
            IDXC,
            SLOT,
            IDX_SLOT,
            IQFP8,
            QFP8,
            eps,
            q_fp8_scale,
            k_scale,
            v_scale,
        ).launch(
            grid=(grid_size, 1, 1),
            block=(self.num_warps * 32, 1, 1),
            cluster=(self.cta_group, 1, 1),
            min_blocks_per_mp=1,
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
        Q_tensor: cute.Tensor | None,
        WQ_tensor: cute.Tensor,
        WK_tensor: cute.Tensor,
        WIQ_tensor: cute.Tensor,
        WIK_tensor: cute.Tensor,
        COS_SIN_tensor: cute.Tensor,
        POS_tensor: cute.Tensor,
        KVC_tensor: cute.Tensor,
        IDXC_tensor: cute.Tensor,
        SLOT_tensor: cute.Tensor,
        IDX_SLOT_tensor: cute.Tensor,
        IQFP8_tensor: cute.Tensor,
        QFP8_tensor: cute.Tensor | None,
        eps: Float32,
        q_fp8_scale: Float32,
        k_scale: Float32,
        v_scale: Float32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        cg = self.cta_group
        cta_rank = raw_bid % cg

        BM, BN, BK = self.cta_tile
        num_stages = self.num_stages

        # tmem (512 cols): double-buffered accumulators, then SFA/SFB0/SFB1.
        # At cta_group=2, 2 x 256 + SF cols don't fit, so they overlap by 32
        # cols (even tiles at [0, 256), odd at [224, 480)).
        acc_odd = BN - 32 if cg == 2 else BN
        sfa_tmem = 496
        sfb0_tmem = sfa_tmem + 4
        sfb1_tmem = sfb0_tmem + 4
        # q / k / index_q / index_k norm weight index in sW
        W_Q, W_K, W_IQ, W_IK = range(4)
        bs = self.block_size

        BAR_EPI = 1
        BAR_TMEM_ALLOC = 2
        cta_pair = Uint16(3) if cg == 2 else None  # multicast / commit mask

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
        # SFB slots: CTA r's multicast copy (head tile 2 * bid_n + r) fills slot
        # r of both CTAs; each slot matches the atom's (64, 1):(1, 64) tile
        sSFB = smem.allocate_tensor(
            Int64,
            cute.make_layout((64, 1, cg, num_stages), stride=(1, 64, 64, 64 * cg)),
            byte_alignment=128,
        )
        # norm weights as (8, 16) tiles per head type
        sW = smem.allocate_tensor(
            BFloat16, cute.make_layout((8, 16, 4)), byte_alignment=16
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_partial_mbar = smem.allocate_array(Int64, 1)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)
        taddr = smem.allocate(Int32, 4)

        M, K = A_tma.tma_tensor.shape
        N, _ = B_tma.tma_tensor.shape

        def num_tiles(M, N, cg):
            # per-CTA M-tiles (a multiple of cg) x BN-wide N-tiles. Each role
            # recomputes these after setmaxnreg: a value kept live across it
            # is spilled.
            grid_n = cute.ceil_div(N, 128 * cg)
            grid_mp = cute.ceil_div(M, 128 * cg)
            return grid_mp, grid_n, grid_mp * cg * grid_n

        def tile_coords(bid, grid_mp, grid_n, cg, G):
            # CTA group p = bid // cg covers M-tiles cg * m + rank (one per
            # CTA) and N-tile n. Rasterized in groups of G M-tile groups, m
            # fastest within each N-tile of a group (G = 1: n fastest).
            p = bid // cg
            if cutlass.const_expr(G == 1):
                return (p // grid_n) * cg + bid % cg, p % grid_n
            group = p // (G * grid_n)
            first = group * G
            size = cutlass.min(grid_mp - first, G)
            local = p - group * (G * grid_n)
            return (first + local % size) * cg + bid % cg, local // size

        if warp_id == 0:
            # counts: tma_full cg CTAs; tmem_empty cg CTAs x 128 * cg epilogue
            # threads; tmem_partial 2 CTAs x 128 (the overlap warpgroup)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cg)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cg * cg)
                cute.arch.mbarrier_init(tmem_partial_mbar, 256)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)
            cpasync.prefetch_descriptor(SFA_tma.atom)
            cpasync.prefetch_descriptor(SFB_tma.atom)
        elif warp_id < 4:
            # norm weights (static, so before the PDL wait): 4 x 256B, one 16B
            # cp.async per thread of warps 2-3
            t = tid - 64
            cp_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(), BFloat16, num_bits_per_copy=128
            )
            for i, Wt in enumerate((WQ_tensor, WK_tensor, WIQ_tensor, WIK_tensor)):
                if t // 16 == i:
                    cute.copy(
                        cp_atom,
                        cute.zipped_divide(Wt, (8,))[None, t % 16],
                        sW[None, t % 16, i],
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()  # sW visible (cluster_arrive is relaxed)
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # warps: 4 * cg epilogue (one warpgroup per head), then MMA, TMA (and 2
        # padding warps at cg=2, completing the producer warpgroup)
        MMA_WARP = 4 * cg
        TMA_WARP = MMA_WARP + 1
        # setmaxnreg (cg=2): the producer warpgroup drops to 40 regs, the
        # epilogue rises to 232 (2*128*232 + 128*40 <= 64K)
        REG_DEC = 40
        REG_INC = 232

        if warp_id > TMA_WARP:
            # padding warps
            if cutlass.const_expr(cg == 2):
                cute.arch.setmaxregister_decrease(REG_DEC)

        elif warp_id == TMA_WARP:
            # TMA warp; both CTAs load, copies arrive on CTA0's mbarrier.
            # SFB is multicast: the pair's MMA needs all 256 rows in each CTA.
            if cutlass.const_expr(cg == 2):
                cute.arch.setmaxregister_decrease(REG_DEC)
            grid_mp, grid_n, total_tiles = num_tiles(M, N, cg)
            tma_stage = 0
            tma_empty_parity = 1
            tma_full_mbar_ = to_cta0_smem(tma_full_mbar)

            gA_tiles = cute.zipped_divide(A_tma.tma_tensor, (BM, BK))
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN // cg, BK))
            # SF boxes: one [32, 4, 4] atom = (64, 1) Int64 (see prepare_SF)
            gSFA_tiles = cute.zipped_divide(SFA_tma.tma_tensor, (64, 1))
            gSFB_tiles = cute.zipped_divide(SFB_tma.tma_tensor, (64, 1))

            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m, bid_n = tile_coords(bid, grid_mp, grid_n, cg, self.group_m)

                for iter_k in cutlass.range(cute.ceil_div(K, BK), unroll=1):
                    mbar = tma_full_mbar_ + tma_stage
                    cute.arch.mbarrier_wait(
                        tma_empty_mbar + tma_stage, tma_empty_parity
                    )
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
                        gB_tiles[None, (cg * bid_n + cta_rank, iter_k)],
                        sB[None, None, tma_stage],
                        mbar,
                    )
                    simple_tma_copy(
                        SFA_tma.atom,
                        gSFA_tiles[None, (iter_k, bid_m)],
                        sSFA[None, None, tma_stage],
                        mbar,
                    )
                    simple_tma_copy(
                        SFB_tma.atom,
                        gSFB_tiles[None, (iter_k, cg * bid_n + cta_rank)],
                        sSFB[None, None, cta_rank, tma_stage],
                        mbar,
                        mcast_mask=cta_pair,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_empty_parity ^= 1

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        elif warp_id == MMA_WARP:
            # MMA warp (2-SM collective; only cta_rank 0 issues). tmem is
            # allocated by epilogue warp 0.
            if cutlass.const_expr(cg == 2):
                cute.arch.setmaxregister_decrease(REG_DEC)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            grid_mp, grid_n, total_tiles = num_tiles(M, N, cg)
            tma_stage = 0
            tma_full_parity = 0

            sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
            # SF smem descriptor: no swizzle, SBO = 8 * 16 bytes
            sdesc_sf = Uint64((8 << 32) | (1 << 46))
            idesc = _tcgen05.make_mxfp8_idesc(cg * BM, BN)

            if cta_rank == 0:
                tmem_stage = 0
                tmem_parity = 1
                for _ in range(raw_bid, total_tiles, num_bids):
                    acc_base = acc_odd * tmem_stage
                    cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_parity)
                    # the previous tile's overlap-region read (cols [224, 256));
                    # the single partial barrier phases once per tile
                    if cutlass.const_expr(cg == 2):
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
                        sfb0_desc = sdesc_sf | (
                            sSFB[None, None, 0, tma_stage].iterator.toint() >> 4
                        )

                        cute.arch.mbarrier_wait(
                            tma_full_mbar + tma_stage, tma_full_parity
                        )
                        _tcgen05.fence_after_thread_sync()

                        _tcgen05.cp(sfa_tmem, sfa_desc, "32x128b", "warpx4", cg)
                        _tcgen05.cp(sfb0_tmem, sfb0_desc, "32x128b", "warpx4", cg)
                        if cutlass.const_expr(cg == 2):
                            sfb1_desc = sdesc_sf | (
                                sSFB[None, None, 1, tma_stage].iterator.toint() >> 4
                            )
                            _tcgen05.cp(sfb1_tmem, sfb1_desc, "32x128b", "warpx4", 2)

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
                                sfb0_tmem,
                                enable_input_d,
                                cg,
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                        _tcgen05.commit(tma_empty_mbar + tma_stage, cta_pair, cg)

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(tmem_full_mbar + tmem_stage, cta_pair, cg)
                    tmem_stage ^= 1
                    if tmem_stage == 0:
                        tmem_parity ^= 1

        else:
            # epilogue warps
            if cutlass.const_expr(cg == 2):
                cute.arch.setmaxregister_increase(REG_INC)
            if warp_id == 0:
                _tcgen05.alloc(taddr, cta_group=cg)
                cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            # warp w reads tmem rows (w % 4) * 32 of head cg * bid_n + w // 4;
            # thread t owns row t % 128. Padding rows (m >= M) compute garbage
            # from clamped metadata and skip only the stores.
            warp_e = warp_id % 4
            wg_id = warp_id // 4
            tid_ = tid % 128

            # Q / QFP8 are each optional (None -> compiled out)
            has_q = cutlass.const_expr(Q_tensor is not None)
            has_qfp8 = cutlass.const_expr(QFP8_tensor is not None)
            # [32, (pos, cos | sin)]
            CS_CHUNKS = cute.zipped_divide(COS_SIN_tensor, (1, 32))[(0, None), None]
            # KV head h's K / V go to slots 2h / 2h + 1; index_k has one slot
            KV_DATA, KV_SF, KV_VSF = _nvfp4_cache_views(KVC_tensor, bs, 2 * self.nkv)
            IDX_DATA, IDX_SF, _ = _nvfp4_cache_views(IDXC_tensor, bs, 1)

            partial_ = to_cta0_smem(tmem_partial_mbar)
            empty_ = to_cta0_smem(tmem_empty_mbar)

            grid_mp, grid_n, total_tiles = num_tiles(M, N, cg)
            tmem_stage = 0
            tmem_parity = 0
            for bid in range(raw_bid, total_tiles, num_bids):
                bid_m, bid_n = tile_coords(bid, grid_mp, grid_n, cg, self.group_m)
                acc_base = acc_odd * tmem_stage

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, tmem_parity)
                cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=cg * 128)
                _tcgen05.fence_after_thread_sync()

                head = cg * bid_n + wg_id  # this warpgroup's head
                m = bid_m * BM + tid_

                # Drain this warp's 32 rows of its head into registers
                row = warp_e * 32
                base = acc_base + wg_id * HEAD_DIM
                acc = cute.make_rmem_tensor((32, 4), Float32)  # 4 x 32 cols
                owns_overlap = False
                if cutlass.const_expr(cg == 2):
                    owns_overlap = wg_id == 1 - tmem_stage
                if owns_overlap:
                    # This warpgroup holds cols [224, 256), which the next
                    # tile's MMA reuses: load and release them first. They are
                    # chunk 0 of the head on odd tiles, chunk 3 on even.
                    if tmem_stage == 1:
                        acc[None, 0].store(_tcgen05.ld(row, base, "32x32b", 32))
                        _tcgen05.wait_ld()
                        mbarrier.arrive(partial_, "cluster")
                        for c in cutlass.range_constexpr(1, 4):
                            col = base + 32 * c
                            acc[None, c].store(_tcgen05.ld(row, col, "32x32b", 32))
                    else:
                        acc[None, 3].store(_tcgen05.ld(row, base + 96, "32x32b", 32))
                        _tcgen05.wait_ld()
                        mbarrier.arrive(partial_, "cluster")
                        for c in cutlass.range_constexpr(3):
                            col = base + 32 * c
                            acc[None, c].store(_tcgen05.ld(row, col, "32x32b", 32))
                else:
                    acc.store(_tcgen05.ld(row, base, "32x32b", HEAD_DIM))
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                mbarrier.arrive(empty_ + tmem_stage, "cluster")

                # rope dim is 64: store the unrotated half (cols 64-127) first
                # to free its registers before rope. Only the stores are
                # predicated on the row / slot.
                halves = cute.make_tensor(acc.iterator, cute.make_layout((64, 2)))
                m_safe = cutlass.min(m, M - 1)

                if head < self.k_start:
                    # q: bf16 and / or scaled e4m3
                    _rms_norm(acc, sW[None, None, W_Q], eps)
                    if cutlass.const_expr(has_qfp8):
                        qfp8_inv = cute.math.rcp(q_fp8_scale)

                    for h in (1, 0):  # NoPE, then RoPE
                        if cutlass.const_expr(h == 0):
                            _rope64(acc, CS_CHUNKS, POS_tensor[m_safe].to(Int32))

                        x = halves[None, h].load()
                        col = HEAD_DIM * head + 64 * h
                        if cutlass.const_expr(has_q):
                            assert Q_tensor is not None
                            q = x.to(BFloat16)
                            if m < M:
                                dst = Q_tensor[m, None].iterator + col
                                cute.arch.store(dst, q, **_NO_L1_ALLOC)
                        # fp8 is stored as u32 words: fp8 vectors don't lower
                        # in cute.arch.store (NVIDIA/cutlass#3712)
                        if cutlass.const_expr(has_qfp8):
                            assert QFP8_tensor is not None
                            q = (x * qfp8_inv).to(Float8E4M3FN).bitcast(Uint32)
                            if m < M:
                                dst = QFP8_tensor[m, None].iterator + col
                                dst = cute.recast_ptr(dst, dtype=Uint32)
                                cute.arch.store(dst, q, **_NO_L1_ALLOC)

                elif head < self.v_start:
                    # k: NVFP4 into its KV head's K slot, linear scale bytes
                    _rms_norm(acc, sW[None, None, W_K], eps)
                    kv_slot = 2 * (head - self.k_start)
                    slot = SLOT_tensor[m_safe]
                    page, token = slot // bs, slot % bs
                    k_sf = cute.math.rcp(k_scale)
                    sf = cute.make_rmem_tensor(8, Uint8)

                    for h in (1, 0):  # NoPE, then RoPE
                        if cutlass.const_expr(h == 0):
                            _rope64(acc, CS_CHUNKS, POS_tensor[m_safe].to(Int32))

                        words, sf_h = _nvfp4_quant64(halves[None, h], k_sf)
                        cute.local_tile(sf, (4,), (h,)).store(sf_h)
                        if m < M and slot >= 0:
                            dst = KV_DATA[None, h, token, kv_slot, page].iterator
                            cute.arch.store(dst, words, **_NO_L1_ALLOC)

                    if m < M and slot >= 0:
                        dst = KV_SF[None, token, kv_slot, page].iterator
                        cute.arch.store(dst, sf.load(), **_NO_L1_ALLOC)

                elif head < self.iq_start:
                    # v: raw accumulator, NVFP4 into its KV head's V slot,
                    # swizzled scale bytes
                    kv_slot = 2 * (head - self.v_start) + 1
                    slot = SLOT_tensor[m_safe]
                    page, token = slot // bs, slot % bs
                    v_sf = cute.math.rcp(v_scale)

                    for h in (0, 1):
                        words, sf = _nvfp4_quant64(halves[None, h], v_sf)
                        if m < M and slot >= 0:
                            dst = KV_DATA[None, h, token, kv_slot, page].iterator
                            cute.arch.store(dst, words, **_NO_L1_ALLOC)
                            vsf = KV_VSF[None, token, kv_slot, page]
                            cute.local_tile(vsf, (4,), (h,)).store(sf)

                elif head < self.ik_start:
                    # index_q: e4m3
                    _rms_norm(acc, sW[None, None, W_IQ], eps)
                    iq_col = HEAD_DIM * (head - self.iq_start)

                    for h in (1, 0):  # NoPE, then RoPE
                        if cutlass.const_expr(h == 0):
                            _rope64(acc, CS_CHUNKS, POS_tensor[m_safe].to(Int32))

                        # u32 words: see the q branch
                        q = halves[None, h].load().to(Float8E4M3FN).bitcast(Uint32)
                        if m < M:
                            dst = IQFP8_tensor[m, None].iterator + iq_col + 64 * h
                            dst = cute.recast_ptr(dst, dtype=Uint32)
                            cute.arch.store(dst, q, **_NO_L1_ALLOC)

                elif head == self.ik_start:
                    # index_k: NVFP4 into the index cache (one slot per page,
                    # linear scale bytes)
                    _rms_norm(acc, sW[None, None, W_IK], eps)
                    slot = IDX_SLOT_tensor[m_safe]
                    page, token = slot // bs, slot % bs
                    sf = cute.make_rmem_tensor(8, Uint8)

                    for h in (1, 0):  # NoPE, then RoPE
                        if cutlass.const_expr(h == 0):
                            _rope64(acc, CS_CHUNKS, POS_tensor[m_safe].to(Int32))

                        words, sf_h = _nvfp4_quant64(halves[None, h])
                        cute.local_tile(sf, (4,), (h,)).store(sf_h)
                        if m < M and slot >= 0:
                            dst = IDX_DATA[None, h, token, 0, page].iterator
                            cute.arch.store(dst, words, **_NO_L1_ALLOC)

                    if m < M and slot >= 0:
                        dst = IDX_SF[None, token, 0, page].iterator
                        cute.arch.store(dst, sf.load(), **_NO_L1_ALLOC)

                # else: phantom head (odd head count), nothing to store

                tmem_stage ^= 1
                if tmem_stage == 0:
                    tmem_parity ^= 1

            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp_id == 0:
                _tcgen05.dealloc(cta_group=cg)


class _QkvIndexFusedEpilogueJit(
    VllmCuTeDSLJitKernel["_QkvIndexFusedEpilogueJit.CompileKey"]
):
    """Compile keys, warmup (fake) inputs and the compiled executors of
    QkvIndexFusedEpilogueKernel."""

    @dataclass(frozen=True)
    class CompileKey:
        has_q: bool
        has_qfp8: bool
        nkv: int
        block_size: int
        group_m: int
        cta_group: int
        use_pdl: bool = True

    @staticmethod
    def kernel(key: CompileKey) -> QkvIndexFusedEpilogueKernel:
        return QkvIndexFusedEpilogueKernel(
            key.nkv, key.block_size, key.cta_group, key.group_m, key.use_pdl
        )

    def get_warmup_keys(self, **fields) -> list[CompileKey]:
        # a layer's keys differ only in the tile config, picked per batch size
        return [self.CompileKey(cta_group=cg, **fields) for cg in (1, 2)]

    def warmup_inputs(self, key: CompileKey) -> tuple:
        M = cute.sym_int()
        N = cute.sym_int()
        K = cute.sym_int()
        P = cute.sym_int()
        slot_bytes = key.block_size * (HEAD_DIM // 2 + HEAD_DIM // 16)

        A = make_fake_tensor(
            Float8E4M3FN, (M, K), (cute.sym_int64(divisibility=16), 1), assumed_align=16
        )
        B = make_fake_tensor(
            Float8E4M3FN, (N, K), (cute.sym_int64(divisibility=16), 1), assumed_align=16
        )
        SFA = make_fake_tensor(
            Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )
        SFB = make_fake_tensor(
            Float8E8M0FNU, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )
        q_cols = 16 * key.nkv * HEAD_DIM
        Q = (
            make_fake_tensor(BFloat16, (M, q_cols), (q_cols, 1), assumed_align=32)
            if key.has_q
            else None
        )
        WQ, WK, WIQ, WIK = (
            make_fake_tensor(BFloat16, (HEAD_DIM,), (1,), assumed_align=16)
            for _ in range(4)
        )
        COS_SIN = make_fake_tensor(BFloat16, (P, 64), (64, 1), assumed_align=16)
        POS = make_fake_tensor(Int64, (M,), (1,), assumed_align=16)
        # caches as [nb, page_bytes] with a dynamic page stride: no dynamic dim
        # may exceed int32, and cross-layer layouts space pages wider than a page
        KVC = make_fake_tensor(
            Uint8,
            (cute.sym_int(), 2 * key.nkv * slot_bytes),
            (cute.sym_int64(divisibility=32), 1),
            assumed_align=32,
        )
        IDXC = make_fake_tensor(
            Uint8,
            (cute.sym_int(), slot_bytes),
            (cute.sym_int64(divisibility=32), 1),
            assumed_align=32,
        )
        SLOT = make_fake_tensor(Int64, (M,), (1,), assumed_align=16)
        IDX_SLOT = make_fake_tensor(Int64, (M,), (1,), assumed_align=16)
        IQFP8 = make_fake_tensor(
            Float8E4M3FN,
            (M, key.nkv * HEAD_DIM),
            (key.nkv * HEAD_DIM, 1),
            assumed_align=32,
        )
        QFP8 = (
            make_fake_tensor(Float8E4M3FN, (M, q_cols), (q_cols, 1), assumed_align=32)
            if key.has_qfp8
            else None
        )

        return (
            A,
            B,
            SFA,
            SFB,
            Q,
            WQ,
            WK,
            WIQ,
            WIK,
            COS_SIN,
            POS,
            KVC,
            IDXC,
            SLOT,
            IDX_SLOT,
            IQFP8,
            QFP8,
            # runtime scalars (eps, q/k/v scales, grid): placeholder values
            Float32(1e-6),
            Float32(1.0),
            Float32(1.0),
            Float32(1.0),
            148,
        )


_JIT = _QkvIndexFusedEpilogueJit()


@cache
def _device_info(device: int) -> tuple[int, int]:
    """(SM count, L2 bytes) of a CUDA device."""
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count, props.L2_cache_size


def _group_m(weight_bytes: int, l2_bytes: int) -> int:
    # Group M-tiles 16 at a time when the weight is a large fraction of L2, so
    # a wave reads a few N-panels instead of all of B (GB300: helps TP1's 60MB
    # weight by 1.5-5%; slows TP2 / TP4, whose weights stay L2-resident anyway)
    return 16 if weight_bytes >= l2_bytes // 4 else 1


def register_qkv_index_fused_epilogue_warmup(
    *, has_q: bool, has_qfp8: bool, nkv: int, block_size: int, weight_bytes: int
) -> None:
    """Compile this layer's kernels (both tile configs) during JIT warmup."""
    _, l2_bytes = _device_info(torch.accelerator.current_device_index())
    _JIT.register_warmup(
        has_q=has_q,
        has_qfp8=has_qfp8,
        nkv=nkv,
        block_size=block_size,
        group_m=_group_m(weight_bytes, l2_bytes),
    )


def minimax_m3_qkv_index_fused_epilogue(
    xq: torch.Tensor,
    xsf: torch.Tensor,
    wq: torch.Tensor,
    wsf: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    eps: float,
    *,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    index_cache: torch.Tensor,
    index_q_out: torch.Tensor,
    q_out: torch.Tensor | None = None,
    q_fp8_out: torch.Tensor | None = None,
    q_fp8_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    cta_group: int | None = None,
    use_pdl: bool = True,
    grid: int | None = None,
) -> None:
    """QKV+indexer projection with fused epilogue: writes q (bf16 and / or
    fp8; at least one) and index_q, and NVFP4-inserts k / v / index_k into the
    paged caches.
    nkv in {1, 2, 4} KV heads per rank is inferred from N = (19 * nkv + 1) * 128.

    Args:
        xq: MXFP8 activation [M, K].
        xsf: xq's F8_128x4-swizzled E8M0 scales.
        wq: K-major MXFP8 weight [N, K]: [q | k | v | index_q | index_k].
        wsf: wq's F8_128x4-swizzled E8M0 scales.
        positions: [M] int64 token positions.
        cos_sin_cache: [max_pos, 64] bf16 RoPE cache (cos | sin).
        q_norm_weight: [128] bf16 Gemma RMSNorm weight; likewise the next three.
        k_norm_weight: see q_norm_weight.
        index_q_norm_weight: see q_norm_weight.
        index_k_norm_weight: see q_norm_weight.
        eps: RMSNorm epsilon.
        slot_mapping: [M] int64 KV-cache slots (-1 = skip).
        index_slot_mapping: [M] int64 index-cache slots (-1 = skip).
        kv_cache: NVFP4 [nb, 2 * nkv, block_size, 72] uint8; packed pages at
            any page stride. block_size % 4 == 0.
        index_cache: NVFP4 [nb, block_size, 72] uint8; packed pages at any
            page stride.
        index_q_out: [M, nkv * 128] e4m3 normed + roped index_q.
        q_out: optional [M, 16 * nkv * 128] bf16 normed + roped q.
        q_fp8_out: optional [M, 16 * nkv * 128] e4m3 normed + roped q.
        q_fp8_scale: dequantization scale of q_fp8_out.
        k_scale: dequantization (global) scale of the NVFP4 k entries.
        v_scale: dequantization (global) scale of the NVFP4 v entries.
        cta_group: 2 (2-SM: CTA pairs, 128 x 256 tiles), 1 (1-SM: 128 x 128
            tiles) or None (pick by wave count).
        use_pdl: launch with programmatic dependent launch.
        grid: persistent grid size override (testing).

    """
    M, K = xq.shape
    N = wq.shape[0]
    nkv = (N // HEAD_DIM - 1) // 19
    assert nkv in (1, 2, 4) and (19 * nkv + 1) * HEAD_DIM == N, N
    assert K % 128 == 0 and wq.shape[1] == K and wq.stride() == (K, 1)
    assert index_q_out.shape == (M, nkv * HEAD_DIM)
    assert index_q_out.dtype == torch.float8_e4m3fn and index_q_out.is_contiguous()
    assert q_out is not None or q_fp8_out is not None, "no q output"
    for t in (positions, slot_mapping, index_slot_mapping):
        assert t.shape == (M,) and t.dtype == torch.int64
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    for w in (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight):
        assert w.shape == (HEAD_DIM,) and w.dtype == torch.bfloat16
    for q, dtype in ((q_out, torch.bfloat16), (q_fp8_out, torch.float8_e4m3fn)):
        if q is not None:
            assert q.shape == (M, 16 * nkv * HEAD_DIM)
            assert q.dtype == dtype and q.is_contiguous()
    assert kv_cache.dtype == torch.uint8 and index_cache.dtype == torch.uint8
    block_size, token_bytes = kv_cache.shape[2:]
    assert token_bytes == HEAD_DIM // 2 + HEAD_DIM // 16
    assert kv_cache.shape[1] == 2 * nkv
    assert kv_cache.stride()[1:] == (block_size * token_bytes, token_bytes, 1)
    assert index_cache.shape[1:] == (block_size, token_bytes)
    assert index_cache.stride()[1:] == (token_bytes, 1)
    for c in (kv_cache, index_cache):  # 32B NVFP4 data stores
        assert c.data_ptr() % 32 == 0 and c.stride(0) % 32 == 0
    num_sms, l2_bytes = _device_info(xq.device.index)

    def num_tiles(cg):
        tile = 128 * cg  # M rows per CTA group, N cols per tile
        return -(-M // tile) * cg * -(-N // tile)

    if cta_group is None:
        # a 128 x 128 tile costs ~2/3 of a 128 x 256 one (GB300): take 1-SM
        # when that wins after wave quantization (odd M-tile counts, few tiles)
        waves1, waves2 = (-(-num_tiles(cg) // num_sms) for cg in (1, 2))
        cta_group = 1 if 2 * waves1 < 3 * waves2 else 2
    if grid is None:
        grid = min(num_sms, num_tiles(cta_group))
    # caches as [nb, page_bytes], keeping the page stride (a flat 1-D view
    # exceeds the DSL's int32 shapes at production sizes)
    slot_bytes = block_size * token_bytes
    kvc = kv_cache.as_strided(
        (kv_cache.size(0), 2 * nkv * slot_bytes), (kv_cache.stride(0), 1)
    )
    idxc = index_cache.as_strided(
        (index_cache.size(0), slot_bytes), (index_cache.stride(0), 1)
    )
    key = _JIT.CompileKey(
        has_q=q_out is not None,
        has_qfp8=q_fp8_out is not None,
        nkv=nkv,
        block_size=block_size,
        group_m=_group_m(wq.numel(), l2_bytes),
        cta_group=cta_group,
        use_pdl=use_pdl,
    )
    _JIT._get_or_compile(key)(
        xq,
        wq,
        xsf.view(-1).view(torch.float8_e8m0fnu),
        wsf.view(-1).view(torch.float8_e8m0fnu),
        q_out,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
        cos_sin_cache,
        positions,
        kvc,
        idxc,
        slot_mapping,
        index_slot_mapping,
        index_q_out,
        q_fp8_out,
        eps,
        q_fp8_scale,
        k_scale,
        v_scale,
        grid,
    )
