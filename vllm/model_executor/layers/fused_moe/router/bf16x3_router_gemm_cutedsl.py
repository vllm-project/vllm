# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL BF16x3 router GEMM.

Computes ``X @ W.T`` for BF16 ``X`` with shape ``[M, K]`` and FP32 router
weights ``W`` with shape ``[N, K]`` by decomposing each FP32 weight value into
three BF16 residual terms, then accumulating the three BF16 MMA results into
FP32 TMEM output. Two tiers, dispatched on token count inside the custom op:

- Small token counts: ``Sm100BF16x3RouterGemmSmallM`` decomposes the weights inside
  the kernel per tile (swap-AB, decomped W stored directly to TMEM).
- Large token counts: a Triton kernel first decomposes the weights into a
  stacked ``[3, N, K]`` BF16 buffer, then ``Sm100BF16x3RouterGemmLargeM``
  ingests the planes via TMA in direct orientation (tokens on the MMA M side,
  256-bit epilogue stores).

The TMEM accumulation chain is bounded to ``num_tmem_acc`` K-tiles: at chunk
boundaries the epilogue warps drain the main-term accumulator into a register
master accumulator and the MMA warp resets it.
"""

from functools import cache
from typing import NamedTuple

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint16, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.memory import get_smem_capacity_in_bytes
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import (
    EVICT_FIRST,
    EVICT_LAST,
    _tcgen05,
    mbarrier,
    simple_tma_copy,
    to_cta0_smem,
)
from vllm.triton_utils import tl, triton
from vllm.utils import math_utils
from vllm.utils.torch_utils import direct_register_custom_op

__all__ = ["bf16x3_router_gemm"]


@dsl_user_op
def _decompose_fp32x2_to_3xbf16x2(
    w0: Float32,
    w1: Float32,
    *,
    loc=None,
    ip=None,
) -> tuple[Uint32, Uint32, Uint32]:
    # this PTX snippets does the following
    #   out0 = BF16(in);   res =  in - FP32(out0)
    #   out1 = BF16(res);  res = res - FP32(out1)
    #   out2 = BF16(res)
    #
    # for normal FP32, this decomposition is exact
    # i.e. in = FP32(out0) + FP32(out1) + FP32(out2)
    #
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [w0.ir_value(loc=loc, ip=ip), w1.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 r1_lo, r1_hi, r2_lo, r2_hi;\n\t"
        ".reg .b32 w0_lo, w0_hi, w1_lo, w1_hi;\n\t"
        ".reg .b64 a_pair, w0_pair, w1_pair, r1_pair, r2_pair;\n\t"
        "cvt.rn.bf16x2.f32 $0, $4, $3;\n\t"
        "shl.b32 w0_lo, $0, 16;\n\t"
        "and.b32 w0_hi, $0, 0xffff0000;\n\t"
        "mov.b64 a_pair, {$3, $4};\n\t"
        "mov.b64 w0_pair, {w0_lo, w0_hi};\n\t"
        "sub.rn.f32x2 r1_pair, a_pair, w0_pair;\n\t"
        "mov.b64 {r1_lo, r1_hi}, r1_pair;\n\t"
        "cvt.rn.bf16x2.f32 $1, r1_hi, r1_lo;\n\t"
        "shl.b32 w1_lo, $1, 16;\n\t"
        "and.b32 w1_hi, $1, 0xffff0000;\n\t"
        "mov.b64 w1_pair, {w1_lo, w1_hi};\n\t"
        "sub.rn.f32x2 r2_pair, r1_pair, w1_pair;\n\t"
        "mov.b64 {r2_lo, r2_hi}, r2_pair;\n\t"
        "cvt.rn.bf16x2.f32 $2, r2_hi, r2_lo;\n\t"
        "}\n",
        "=r,=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [2], loc=loc, ip=ip)),
    )


class Sm100BF16x3RouterGemmSmallM:
    """Swap-AB: experts (BN) on the MMA M side with the decomposed W in TMEM,
    tokens (BM) on the MMA N side, so the accumulators hold BM columns."""

    def __init__(self, BM: int = 128) -> None:
        self.cta_tile = (BM, 128, 64)  # (BM tokens, BN experts, BK)
        self.num_stages = 2
        self.num_warps = 10
        self.num_tmem_acc = 8

    @cute.jit
    def _make_tma(self, tensor: cute.Tensor, rows: int, BK: int):
        op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // tensor.element_type.width  # 128B
        slayout = cute.make_layout(
            (rows, (elems, BK // elems), self.num_stages),
            stride=(elems, (1, rows * elems), rows * BK),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(op, tensor, slayout, (rows, BK))

    @cute.jit
    def __call__(
        self,
        X: cute.Tensor,
        W: cute.Tensor,
        out: cute.Tensor,
        split_k: Int32,
        stream: CUstream,
    ):
        BM, BN, BK = self.cta_tile
        W_tma = self._make_tma(W, BN, BK)
        X_tma = self._make_tma(X, BM, BK)

        grid_n = cute.ceil_div(W.shape[0], BN)
        grid_m = cute.ceil_div(X.shape[0], BM)

        self.kernel(X_tma, W_tma, out).launch(
            grid=(grid_n, grid_m, split_k),
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, X_tma: cpasync.TmaInfo, W_tma: cpasync.TmaInfo, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid_n, bid_m, bid_k = cute.arch.block_idx()
        _, _, split_k = cute.arch.grid_dim()

        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        BM, BN, BK = self.cta_tile
        num_stages = self.num_stages
        num_tmem_acc = self.num_tmem_acc

        M, K = X_tma.tma_tensor.shape
        N, _ = W_tma.tma_tensor.shape
        k_tiles = cute.ceil_div(K, BK)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            BFloat16,
            X_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=X_tma.smem_layout.inner,
        )
        sW = smem.allocate_tensor(
            Float32,
            W_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=W_tma.smem_layout.inner,
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        w_full_mbar = smem.allocate_array(Int64, num_stages)
        acc_full_mbar = smem.allocate_array(Int64, 1)
        acc_empty_mbar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        BAR_TMEM_ALLOC = 1
        BAR_PREP = 2
        BAR_EPI = 3

        # tmem "allocation"
        # acc_main is for the 1st BF16 term. acc_res is for the 2nd and 3rd
        # BF16 terms, which are much smaller than the first term.
        acc_main = 0
        acc_res = BM
        w_tmem_base = BM * 2

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(w_full_mbar + i, 128)
                cute.arch.mbarrier_init(acc_full_mbar, 1)
                cute.arch.mbarrier_init(acc_empty_mbar, 128)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(X_tma.atom)
            cpasync.prefetch_descriptor(W_tma.atom)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()

        if warp_id == 9:
            # TMA warp
            stage_id = 0
            parity = 1

            # (BN, BK, K/BK)
            gW_tiles = cute.local_tile(W_tma.tma_tensor, (BN, BK), (bid_n, None))
            gX_tiles = cute.local_tile(X_tma.tma_tensor, (BM, BK), (bid_m, None))

            for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                mbar = tma_full_mbar + stage_id
                with cute.arch.elect_one():
                    stage_bytes = BM * BK * 2 + BN * BK * 4
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, stage_bytes)
                simple_tma_copy(
                    W_tma.atom,
                    gW_tiles[None, None, tile_k],
                    sW[None, None, stage_id],
                    mbar,
                )
                simple_tma_copy(
                    X_tma.atom,
                    gX_tiles[None, None, tile_k],
                    sX[None, None, stage_id],
                    mbar,
                )

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

        elif warp_id == 8:
            # MMA warp
            tma_stage_id = 0
            tma_parity = 0

            tmem_acc_count = 0
            tmem_parity = 1

            MMA_M, MMA_N = BN, BM  # swap-AB
            idesc = _tcgen05.make_bf16_idesc(MMA_M, MMA_N)
            sdesc = _tcgen05.make_sdesc_128B_swizzle(0)

            for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                if tmem_acc_count == 0:
                    cute.arch.mbarrier_wait(acc_empty_mbar, tmem_parity)
                cute.arch.mbarrier_wait(tma_full_mbar + tma_stage_id, tma_parity)
                cute.arch.mbarrier_wait(w_full_mbar + tma_stage_id, tma_parity)
                _tcgen05.fence_after_thread_sync()

                w_tmem = w_tmem_base + tma_stage_id * (BK // 2 * 3)
                x_desc = sdesc | (sX[None, None, tma_stage_id].iterator.toint() >> 4)

                for k in cutlass.range_constexpr(BK // 16):
                    enable_d_main = (tmem_acc_count > 0) or (k > 0)
                    enable_d_res = (tile_k > bid_k) or (k > 0)
                    _tcgen05.mma_ts_f16(
                        acc_main, w_tmem + k * 8, x_desc, idesc, enable_d_main
                    )
                    _tcgen05.mma_ts_f16(
                        acc_res, w_tmem + 32 + k * 8, x_desc, idesc, enable_d_res
                    )
                    _tcgen05.mma_ts_f16(
                        acc_res, w_tmem + 64 + k * 8, x_desc, idesc, True
                    )
                    x_desc += 32 >> 4

                _tcgen05.commit(tma_empty_mbar + tma_stage_id)

                tma_stage_id = (tma_stage_id + 1) % num_stages
                if tma_stage_id == 0:
                    tma_parity ^= 1

                tmem_acc_count = (tmem_acc_count + 1) % num_tmem_acc
                if tmem_acc_count == 0:
                    _tcgen05.commit(acc_full_mbar)
                    tmem_parity ^= 1

            if tmem_acc_count != 0:
                _tcgen05.commit(acc_full_mbar)

        elif warp_id >= 4:
            # prep warps: decompose FP32 W into 3xBF16
            warp_id_ = warp_id % 4

            stage_id = 0
            parity = 0

            # ld.shared.v4.f32
            op = cute.nvgpu.CopyUniversalOp()
            cp_atom = cute.make_copy_atom(op, Float32, num_bits_per_copy=128)

            # sW_view: ((4, 1), (BK/4, num_stages))
            row = warp_id_ * 32 + lane_id
            sW_view = cute.zipped_divide(sW[row, None, None], (4, 1))

            for _ in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar + stage_id, parity)
                cute.arch.barrier(barrier_id=BAR_PREP, number_of_threads=128)

                row = warp_id_ * 32 + lane_id
                w_tmem = w_tmem_base + stage_id * (BK // 2 * 3)
                for kblock in cutlass.range_constexpr(BK // 4):
                    w0 = cute.make_rmem_tensor(2, Uint32)
                    w1 = cute.make_rmem_tensor(2, Uint32)
                    w2 = cute.make_rmem_tensor(2, Uint32)

                    w_tmp = cute.make_rmem_tensor(4, Float32)
                    cute.copy(cp_atom, sW_view[None, (kblock, stage_id)], w_tmp)
                    w0[0], w1[0], w2[0] = _decompose_fp32x2_to_3xbf16x2(
                        w_tmp[0], w_tmp[1]
                    )
                    w0[1], w1[1], w2[1] = _decompose_fp32x2_to_3xbf16x2(
                        w_tmp[2], w_tmp[3]
                    )

                    tcol = kblock * 2
                    _tcgen05.st(warp_id_ * 32, w_tmem + 0 + tcol, "32x32b", 2, w0)
                    _tcgen05.st(warp_id_ * 32, w_tmem + 32 + tcol, "32x32b", 2, w1)
                    _tcgen05.st(warp_id_ * 32, w_tmem + 64 + tcol, "32x32b", 2, w2)

                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(w_full_mbar + stage_id)

                stage_id = (stage_id + 1) % self.num_stages
                if stage_id == 0:
                    parity ^= 1

        else:
            # epilogue warps
            if warp_id == 0:
                _tcgen05.alloc(taddr)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=128)

            tiles_local = cute.ceil_div(k_tiles - bid_k, split_k)
            num_chunks = cute.ceil_div(tiles_local, num_tmem_acc)

            WIDTH = 8
            main_regs = cute.make_rmem_tensor(WIDTH, Float32)
            res_regs = cute.make_rmem_tensor(WIDTH, Float32)

            if num_chunks == 1:
                # single chunk
                cute.arch.mbarrier_wait(acc_full_mbar, 0)
                _tcgen05.fence_after_thread_sync()
                w_row_idx = bid_n * BN + tid
                for i in cutlass.range_constexpr(BM // WIDTH):
                    tcol = i * WIDTH
                    main_regs.store(_tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH))
                    res_regs.store(
                        _tcgen05.ld(warp_id * 32, BM + tcol, "32x32b", WIDTH)
                    )
                    _tcgen05.wait_ld()

                    # CuteDSL will codegen add.f32x2
                    for j in cutlass.range(WIDTH, vectorize=True):
                        main_regs[j] += res_regs[j]

                    for j in cutlass.range_constexpr(WIDTH):
                        x_row_idx = bid_m * BM + i * WIDTH + j
                        if x_row_idx < M and w_row_idx < N:
                            out[bid_k, x_row_idx, w_row_idx] = main_regs[j]

            else:
                # multiple chunks
                master_acc = cute.make_rmem_tensor(BM, Float32)

                # chunk 0: pure TMEM load, no accumulation.
                cute.arch.mbarrier_wait(acc_full_mbar, 0)
                _tcgen05.fence_after_thread_sync()
                master_acc.store(_tcgen05.ld(warp_id * 32, 0, "32x32b", BM))
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(acc_empty_mbar)

                # accumulate main tmem acc
                for chunk in cutlass.range(1, num_chunks, unroll=1):
                    cute.arch.mbarrier_wait(acc_full_mbar, chunk & 1)
                    _tcgen05.fence_after_thread_sync()

                    for i in cutlass.range_constexpr(BM // WIDTH):
                        tcol = i * WIDTH
                        main_regs.store(
                            _tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH)
                        )
                        _tcgen05.wait_ld()
                        # CuteDSL refuses to compile
                        # master_acc[i * WIDTH + j] += main_regs[j]
                        # when vectorize=True
                        master_view = cute.local_tile(master_acc, (WIDTH,), (i,))
                        for j in cutlass.range(WIDTH, vectorize=True):
                            master_view[j] += main_regs[j]

                    _tcgen05.fence_before_thread_sync()
                    cute.arch.mbarrier_arrive(acc_empty_mbar)

                # fold in the residual accumulator
                for i in cutlass.range_constexpr(BM // WIDTH):
                    tcol = i * WIDTH
                    res_regs.store(
                        _tcgen05.ld(warp_id * 32, BM + tcol, "32x32b", WIDTH)
                    )
                    _tcgen05.wait_ld()
                    master_view = cute.local_tile(master_acc, (WIDTH,), (i,))
                    for j in cutlass.range(WIDTH, vectorize=True):
                        master_view[j] += res_regs[j]

                    w_row_idx = bid_n * BN + tid
                    for j in cutlass.range_constexpr(WIDTH):
                        x_row_idx = bid_m * BM + i * WIDTH + j
                        if x_row_idx < M and w_row_idx < N:
                            out[bid_k, x_row_idx, w_row_idx] = master_acc[i * WIDTH + j]

            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=128)
            if warp_id == 0:
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile(K: int, BM: int = 128):
        M = cute.sym_int()
        N = cute.sym_int()
        SPLIT_K = cute.sym_int()
        X = make_fake_tensor(BFloat16, (M, K), divisibility=8)
        W = make_fake_tensor(Float32, (N, K), divisibility=4)
        out = make_fake_tensor(Float32, (SPLIT_K, M, N), divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100BF16x3RouterGemmSmallM(BM)
        return cute.compile(
            kernel, X, W, out, Int32(1), stream, options="--enable-tvm-ffi"
        )


@triton.jit
def _splitk_reduce_kernel(
    partials,
    out,
    M,
    N: tl.constexpr,
    split_stride,
    k_splits,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_s = tl.arange(0, BS)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    vals = tl.load(
        partials
        + offs_s[:, None, None] * split_stride
        + offs_m[None, :, None] * N
        + offs_n[None, None, :],
        mask=(
            (offs_s[:, None, None] < k_splits)
            & (offs_m[None, :, None] < M)
            & (offs_n[None, None, :] < N)
        ),
        other=0.0,
    )
    acc = tl.sum(vals, axis=0)
    tl.store(
        out + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _splitk_reduce_config(split_k: int) -> tuple[int, int, int]:
    block_s = triton.next_power_of_2(split_k)
    if block_s >= 64:
        return 1, 32, block_s
    if block_s >= 8:
        return 1, 256, block_s
    return min(16, 32 // block_s), 32, block_s


def splitk_reduce_triton(partials: torch.Tensor, out: torch.Tensor):
    split_k, M, N = partials.shape
    split_stride = partials.stride(0)
    BM, BN, block_s = _splitk_reduce_config(split_k)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _splitk_reduce_kernel[grid](
        partials,
        out,
        M,
        N,
        split_stride,
        split_k,
        BM=BM,
        BN=BN,
        BS=block_s,
        USE_PDL=True,
        num_warps=4,
        launch_pdl=True,
    )


@triton.jit
def _bf16x3_decomp_kernel(W, W3, numel, BLOCK: tl.constexpr = 4096):
    # W is the contiguous [N, K] weight, viewed flat. numel stays specialized:
    # its divisibility by 16 is what vectorizes the masked loads/stores to
    # 128-bit (scalar without it, 1.5x slower). The PDL wait guards W3, a
    # fresh allocation whose block a still-running predecessor may be reading.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    w = tl.load(W + offs, mask=mask)
    # RN casts and fp32 subs: bit-exact with the torch fp32 -> bf16 chain
    w0 = w.to(tl.bfloat16)
    r1 = w - w0.to(tl.float32)
    w1 = r1.to(tl.bfloat16)
    r2 = r1 - w1.to(tl.float32)
    w2 = r2.to(tl.bfloat16)
    tl.store(W3 + offs, w0, mask=mask)
    tl.store(W3 + numel + offs, w1, mask=mask)
    tl.store(W3 + 2 * numel + offs, w2, mask=mask)


class Sm100BF16x3RouterGemmLargeM:
    """Direct-orientation BF16x3 router GEMM over pre-decomposed W3.

    cta_group=2: m256n128 tile per CTA pair. Each CTA loads its own 128
    token rows (X) but only half of the W3 splits (64 experts), halving
    per-SM ingest. Only cta_rank 0 issues MMAs; commits multicast to both
    CTAs. cta_group=1: m128n128 per CTA, full 128-expert W3 stages (2x
    per-CTA W ingest, but 1-CTA work granularity -> finer split-K for small
    token counts). TMEM per CTA: acc_main x2 (double-buffered, 128 cols
    each) + acc_res (128 cols) = 384/512 cols. Warps 0-3: epilogue. Warp 4:
    MMA. Warp 5: TMA.
    """

    def __init__(self, cta_group: int) -> None:
        self.cta_group = cta_group
        self.cta_tile = (128, 128, 64)  # (BM per CTA, BN experts, BK)
        self.num_warps = 6
        # drain acc_main every 512 K-elements (matches the small-M kernel's
        # 8-tile chunks at BK=64) to preserve bit-exact accumulation grouping
        self.num_tmem_acc = 512 // 64
        # a stage holds this CTA's X tile and its BN // cta_group-expert slice
        # of the three W3 planes
        BM, BN, BK = self.cta_tile
        self.stage_size = (BM + 3 * (BN // cta_group)) * BK * 2
        self.num_stages = get_smem_capacity_in_bytes() // self.stage_size

    @cute.jit
    def _make_tma(self, tensor: cute.Tensor, BM: int, BK: int):
        group = tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        op = cpasync.CopyBulkTensorTileG2SOp(cta_group=group)
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // tensor.element_type.width  # 128B
        slayout = cute.make_layout(
            (BM, (elems, BK // elems), self.num_stages),
            stride=(elems, (1, BM * elems), BM * BK),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(op, tensor, slayout, (BM, BK))

    @cute.jit
    def _make_tma_w3(self, tensor: cute.Tensor, BN: int, BK: int):
        """One 3D TMA box (3 terms, BN experts, BK) over the stacked [3,N,K]
        W tensor; smem holds the three term blocks contiguously (8 KB each),
        so the MMA descriptors for terms 1/2 are base + 512/1024."""
        group = tcgen05.CtaGroup.TWO if self.cta_group == 2 else tcgen05.CtaGroup.ONE
        op = cpasync.CopyBulkTensorTileG2SOp(cta_group=group)
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // tensor.element_type.width  # 128B
        slayout = cute.make_layout(
            (3, BN, (elems, BK // elems), self.num_stages),
            stride=(BN * BK, elems, (1, BN * elems), 3 * BN * BK),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(op, tensor, slayout, (3, BN, BK))

    @cute.jit
    def __call__(
        self,
        X: cute.Tensor,
        W3: cute.Tensor,
        out: cute.Tensor,
        split_k: Int32,
        stream: CUstream,
    ):
        BM, BN, BK = self.cta_tile
        cg = self.cta_group
        X_tma = self._make_tma(X, BM, BK)
        W3_tma = self._make_tma_w3(W3, BN // cg, BK)

        # one work item per (cg*128-token tile, 128-expert n-tile, k-slice);
        # the work item's cg CTAs are adjacent flat bids
        pair_tiles = cute.ceil_div(X.shape[0], BM * cg)
        n_tiles = cute.ceil_div(W3.shape[1], BN)
        grid_x = cg * pair_tiles * n_tiles * split_k

        self.kernel(X_tma, W3_tma, out, split_k).launch(
            grid=(grid_x, 1, 1),
            block=(self.num_warps * 32, 1, 1),
            cluster=(cg, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        X_tma: cpasync.TmaInfo,
        W3_tma: cpasync.TmaInfo,
        out: cute.Tensor,
        split_k: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        BM, BN, BK = self.cta_tile
        cg = self.cta_group
        cta_rank = raw_bid % cg
        BN_HALF = BN // cg
        num_stages = self.num_stages
        num_tmem_acc = self.num_tmem_acc

        M, K = X_tma.tma_tensor.shape
        _, N, _ = W3_tma.tma_tensor.shape
        n_tiles = cute.ceil_div(N, BN)

        # one work item per CTA (pair): k-slice fastest, then the BN-expert
        # tile, then the (cg * BM)-token tile; each cg member covers its own
        # BM token rows and BN // cg experts of the item
        work_id = raw_bid // cg
        bid_k = work_id % split_k
        bid_n = work_id // split_k % n_tiles
        bid_m = work_id // (split_k * n_tiles) * cg + cta_rank
        # this slice's K tiles are bid_k, bid_k + split_k, ...
        k_tiles = cute.ceil_div(K, BK)
        num_k_iters = cute.ceil_div(k_tiles - bid_k, split_k)
        num_chunks = cute.ceil_div(num_k_iters, num_tmem_acc)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            BFloat16,
            X_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=X_tma.smem_layout.inner,
        )
        sW3 = smem.allocate_tensor(
            BFloat16,
            W3_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=W3_tma.smem_layout.inner,
        )

        # the full and tmem_empty barriers live in CTA0's smem: both CTAs'
        # TMA copies and epilogues arrive there
        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)
        taddr = smem.allocate(Int32, 4)

        BAR_TMEM_ALLOC = 1

        # TMEM: acc_main double-buffered at [0, 2 * BN), acc_res at 2 * BN
        acc_res = 2 * BN

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cg)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cg)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(X_tma.atom)
            cpasync.prefetch_descriptor(W3_tma.atom)
        if cutlass.const_expr(cg == 2):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()

        if warp_id == 5:
            # TMA warp (all ranks)
            tma_stage = 0
            parity = 1
            if cutlass.const_expr(cg == 2):
                tma_full_mbar_ = to_cta0_smem(tma_full_mbar)
                tma_scope = "cluster"
            else:
                tma_full_mbar_ = tma_full_mbar
                tma_scope = "cta"

            gX_tiles = cute.local_tile(X_tma.tma_tensor, (BM, BK), (bid_m, None))
            gW3_tiles = cute.local_tile(
                W3_tma.tma_tensor, (3, BN_HALF, BK), (0, bid_n * cg + cta_rank, None)
            )

            for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                mbar = tma_full_mbar_ + tma_stage
                cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)
                with cute.arch.elect_one():
                    mbarrier.arrive_expect_tx(mbar, self.stage_size, tma_scope)
                simple_tma_copy(
                    X_tma.atom,
                    gX_tiles[None, None, tile_k],
                    sX[None, None, tma_stage],
                    mbar,
                    # streamed once: keep it from displacing W in L2
                    cache_policy=EVICT_FIRST,
                )
                simple_tma_copy(
                    W3_tma.atom,
                    gW3_tiles[None, None, None, tile_k],
                    sW3[None, None, None, tma_stage],
                    mbar,
                    # re-read by every CTA
                    cache_policy=EVICT_LAST,
                )
                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    parity ^= 1

        elif warp_id == 4:
            # MMA warp (collective TMEM alloc with epilogue warp 0; only
            # cta_rank 0 issues MMAs)
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_empty_parity = 1

                MMA_M = BM * cg
                MMA_N = BN
                idesc = _tcgen05.make_bf16_idesc(MMA_M, MMA_N)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                multicast_mask = Uint16(3) if cutlass.const_expr(cg == 2) else None

                for chunk in cutlass.range(num_chunks, unroll=1):
                    # acc_main accumulates up to num_tmem_acc K tiles per chunk
                    cute.arch.mbarrier_wait(
                        tmem_empty_mbar + tmem_stage, tmem_empty_parity
                    )
                    acc_main = tmem_stage * BN
                    chunk_iters = cutlass.min(
                        num_tmem_acc, num_k_iters - chunk * num_tmem_acc
                    )

                    for acc_iter in cutlass.range(chunk_iters, unroll=1):
                        cute.arch.mbarrier_wait(
                            tma_full_mbar + tma_stage, tma_full_parity
                        )
                        _tcgen05.fence_after_thread_sync()

                        x_desc = sdesc | (
                            sX[None, None, tma_stage].iterator.toint() >> 4
                        )
                        w0_desc = sdesc | (
                            sW3[None, None, None, tma_stage].iterator.toint() >> 4
                        )
                        # term t's [BN_HALF, BK] block sits t * BN_HALF * BK * 2 B in
                        w1_desc = w0_desc + (BN_HALF * BK * 2 >> 4)
                        w2_desc = w0_desc + 2 * (BN_HALF * BK * 2 >> 4)

                        for k in cutlass.range_constexpr(BK // 16):
                            enable_main = acc_iter > 0 or k > 0
                            enable_res = chunk > 0 or enable_main
                            _tcgen05.mma_f16(
                                acc_main, x_desc, w0_desc, idesc, enable_main, cg
                            )
                            _tcgen05.mma_f16(
                                acc_res, x_desc, w1_desc, idesc, enable_res, cg
                            )
                            _tcgen05.mma_f16(acc_res, x_desc, w2_desc, idesc, True, cg)
                            # BK = 64 is one 128B swizzle atom: 32 B per k-step
                            x_desc += 32 >> 4
                            w0_desc += 32 >> 4
                            w1_desc += 32 >> 4
                            w2_desc += 32 >> 4

                        _tcgen05.commit(tma_empty_mbar + tma_stage, multicast_mask, cg)
                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(tmem_full_mbar + tmem_stage, multicast_mask, cg)
                    tmem_stage = (tmem_stage + 1) % 2
                    if tmem_stage == 0:
                        tmem_empty_parity ^= 1

        else:
            # epilogue warps 0-3 (each CTA drains its own TMEM lanes)
            if warp_id == 0:
                _tcgen05.alloc(taddr, cta_group=cg)
                cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            WIDTH = 8
            main_regs = cute.make_rmem_tensor(WIDTH, Float32)
            res_regs = cute.make_rmem_tensor(WIDTH, Float32)
            master_acc = cute.make_rmem_tensor(BN, Float32)

            if cutlass.const_expr(cg == 2):
                tmem_empty_mbar_ = to_cta0_smem(tmem_empty_mbar)
                arrive_scope = "cluster"
            else:
                tmem_empty_mbar_ = tmem_empty_mbar
                arrive_scope = "cta"

            tmem_stage = 0
            parity = 0

            if num_chunks == 1:
                # main + residual in one pass
                cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                _tcgen05.fence_after_thread_sync()
                for i in cutlass.range_constexpr(BN // WIDTH):
                    main_regs.store(
                        _tcgen05.ld(warp_id * 32, i * WIDTH, "32x32b", WIDTH)
                    )
                    res_regs.store(
                        _tcgen05.ld(warp_id * 32, acc_res + i * WIDTH, "32x32b", WIDTH)
                    )
                    _tcgen05.wait_ld()
                    # CuteDSL will codegen add.f32x2
                    master_view = cute.local_tile(master_acc, (WIDTH,), (i,))
                    for j in cutlass.range(WIDTH, vectorize=True):
                        master_view[j] = main_regs[j] + res_regs[j]

            else:
                # chunk 0: pure TMEM load, no accumulation
                cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                _tcgen05.fence_after_thread_sync()
                master_acc.store(_tcgen05.ld(warp_id * 32, 0, "32x32b", BN))
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, arrive_scope)
                tmem_stage = 1

                for _ in cutlass.range(1, num_chunks, unroll=1):
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, parity)
                    _tcgen05.fence_after_thread_sync()
                    for i in cutlass.range_constexpr(BN // WIDTH):
                        tcol = tmem_stage * BN + i * WIDTH
                        main_regs.store(
                            _tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH)
                        )
                        _tcgen05.wait_ld()
                        master_view = cute.local_tile(master_acc, (WIDTH,), (i,))
                        for j in cutlass.range(WIDTH, vectorize=True):
                            master_view[j] += main_regs[j]
                    _tcgen05.fence_before_thread_sync()
                    mbarrier.arrive(tmem_empty_mbar_ + tmem_stage, arrive_scope)
                    tmem_stage = (tmem_stage + 1) % 2
                    if tmem_stage == 0:
                        parity ^= 1

                # fold in the residual accumulator
                for i in cutlass.range_constexpr(BN // WIDTH):
                    res_regs.store(
                        _tcgen05.ld(warp_id * 32, acc_res + i * WIDTH, "32x32b", WIDTH)
                    )
                    _tcgen05.wait_ld()
                    master_view = cute.local_tile(master_acc, (WIDTH,), (i,))
                    for j in cutlass.range(WIDTH, vectorize=True):
                        master_view[j] += res_regs[j]

            # one 256-bit L1::no_allocate store per 8 experts; N % 8 == 0,
            # so groups are full or fully OOB, and the fake tensor's
            # divisibility=8 proves 32B dst alignment at compile time
            m_row = bid_m * BM + tid
            for i in cutlass.range_constexpr(BN // WIDTH):
                n_col = bid_n * BN + i * WIDTH
                if m_row < M and n_col + WIDTH <= N:
                    v = cute.local_tile(master_acc, (WIDTH,), (i,)).load()
                    dst = out[bid_k, m_row, None].iterator + n_col
                    cute.arch.store(
                        dst, v, level1_eviction_priority="evict_no_allocate"
                    )

            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp_id == 0:
                _tcgen05.dealloc(cta_group=cg)

    @cache
    @staticmethod
    def compile(K: int, cta_group: int):
        M = cute.sym_int()
        N = cute.sym_int()
        SPLIT_K = cute.sym_int()
        X = make_fake_tensor(BFloat16, (M, K), divisibility=8)
        W3 = make_fake_tensor(BFloat16, (3, N, K), divisibility=8)
        # the 256-bit epilogue stores need 32B-aligned rows (N % 8 == 0)
        out = make_fake_tensor(Float32, (SPLIT_K, M, N), divisibility=8)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100BF16x3RouterGemmLargeM(cta_group)
        return cute.compile(
            kernel, X, W3, out, Int32(1), stream, options="--enable-tvm-ffi"
        )


# Dispatch and tuning. Two tiers, chosen per call by token count M:
# - small-M: Sm100BF16x3RouterGemmSmallM (in-kernel decomp, swap-AB); wins
#   where the call is launch- and decomp-latency bound.
# - large-M: decomp kernel + Sm100BF16x3RouterGemmLargeM.
# Both write split-K partials that the Triton reduce sums.


class SmallMConfig(NamedTuple):
    """Sm100BF16x3RouterGemmSmallM with BM-token tiles."""

    BM: int
    split_k: int | None = None  # None: fill one wave of SMs


class LargeMConfig(NamedTuple):
    """Decomp kernel + Sm100BF16x3RouterGemmLargeM."""

    cta_group: int
    split_k: int | None = None  # None: fill one wave of SMs


RouterGemmConfig = SmallMConfig | LargeMConfig
_ConfigTable = list[tuple[int | None, RouterGemmConfig]]

# Ordered (max tokens, config) entries; the first entry with M <= max tokens
# (None: no limit) wins. Tuned shapes are keyed by (K, N experts), from GB300
# cold-L2 sweeps of both tiers; fixed split_k entries launch at most 148 CTAs
# over their range, so they stay one wave on B200 as on GB300. GateLinear runs
# MiniMax-M2/M3 at <= FP32_MAX_TOKENS on the FP32 router kernel instead.
_TUNED_CONFIGS: dict[tuple[int, int], _ConfigTable] = {
    (6144, 128): [  # MiniMax-M3
        (48, SmallMConfig(16, 48)),
        (64, SmallMConfig(32, 50)),
        (128, SmallMConfig(32)),
        (256, LargeMConfig(1, 64)),
        (512, LargeMConfig(1)),
        (1024, LargeMConfig(2, 16)),
        (None, LargeMConfig(2)),
    ],
    (3072, 256): [  # MiniMax-M2
        (48, SmallMConfig(16)),
        (96, SmallMConfig(32)),
        (128, SmallMConfig(32, 16)),
        (224, SmallMConfig(32)),
        (512, LargeMConfig(1, 16)),
        (2048, LargeMConfig(1)),
        (None, LargeMConfig(2)),
    ],
    (4096, 192): [  # Hunyuan-V3
        (8, SmallMConfig(8)),
        (32, SmallMConfig(16)),
        (88, SmallMConfig(32)),
        (128, SmallMConfig(32, 16)),
        (264, LargeMConfig(1)),
        (512, LargeMConfig(1, 16)),
        (2048, LargeMConfig(1)),
        (None, LargeMConfig(2)),
    ],
    (2816, 256): [  # Hunyuan-V4
        (8, SmallMConfig(8)),
        (48, SmallMConfig(16)),
        (96, SmallMConfig(32)),
        (128, SmallMConfig(32, 16)),
        (256, SmallMConfig(32)),
        (512, LargeMConfig(1, 16)),
        (2048, LargeMConfig(1)),
        (None, LargeMConfig(2)),
    ],
}
_DEFAULT_CONFIGS: _ConfigTable = [
    (16, SmallMConfig(16)),
    (128, SmallMConfig(32)),
    (512, LargeMConfig(1)),
    (None, LargeMConfig(2)),
]


def _pick_config(M: int, K: int, N: int, num_sms: int) -> tuple[RouterGemmConfig, int]:
    """Return the config for M tokens and its resolved split_k. Shared by
    dispatch and warmup."""
    configs = _TUNED_CONFIGS.get((K, N), _DEFAULT_CONFIGS)
    config = next(c for max_m, c in configs if max_m is None or max_m >= M)
    if isinstance(config, LargeMConfig) and (K % 64 or N % 8):
        # the large-M kernel needs 64-wide K tiles and 8-expert 256-bit stores
        config = SmallMConfig(128)
    split_k = config.split_k
    if split_k is None:
        if isinstance(config, LargeMConfig):
            tiles = config.cta_group * math_utils.cdiv(M, 128 * config.cta_group)
        else:
            tiles = math_utils.cdiv(M, config.BM)
        split_k = max(1, num_sms // (tiles * math_utils.cdiv(N, 128)))
    return config, min(split_k, math_utils.cdiv(K, 64))


def warmup_bf16x3_router_gemm(
    K: int,
    N: int,
    min_num_tokens: int,
    max_num_tokens: int,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Compile every configuration the dispatch reaches for M in
    [min_num_tokens, max_num_tokens]; return the compiled tiles per tier
    (hashable, for ``logger.info_once``)."""
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        triton_scalar_specialization_rep,
    )

    device = torch.accelerator.current_device_index()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    small_m_BMs: set[int] = set()
    large_m_cta_groups: set[int] = set()
    reduce_configs: set[tuple[int, int, int, int, int, int]] = set()
    for M in range(min_num_tokens, max_num_tokens + 1):
        config, split_k = _pick_config(M, K, N, num_sms)
        if isinstance(config, LargeMConfig):
            large_m_cta_groups.add(config.cta_group)
        else:
            small_m_BMs.add(config.BM)
        if split_k > 1:
            reduce_configs.add(
                (
                    triton_scalar_specialization_rep(M),
                    triton_scalar_specialization_rep(M * N),
                    triton_scalar_specialization_rep(split_k),
                    *_splitk_reduce_config(split_k),
                )
            )

    for BM in sorted(small_m_BMs):
        Sm100BF16x3RouterGemmSmallM.compile(K, BM)
    if large_m_cta_groups:
        _bf16x3_decomp_kernel.warmup(
            TritonWarmupTensor(torch.float32),
            TritonWarmupTensor(torch.bfloat16),
            triton_scalar_specialization_rep(N * K),
            launch_pdl=True,
            grid=(1,),
        )
    for cta_group in sorted(large_m_cta_groups):
        Sm100BF16x3RouterGemmLargeM.compile(K, cta_group)
    for M, split_stride, split_k, BM, BN, block_s in sorted(reduce_configs):
        _splitk_reduce_kernel.warmup(
            TritonWarmupTensor(torch.float32),
            TritonWarmupTensor(torch.float32),
            M,
            N,
            split_stride,
            split_k,
            BM=BM,
            BN=BN,
            BS=block_s,
            USE_PDL=True,
            num_warps=4,
            launch_pdl=True,
            grid=(1, 1),
        )
    return (
        ("small_m_BM", tuple(sorted(small_m_BMs))),
        ("large_m_cta_group", tuple(sorted(large_m_cta_groups))),
    )


def _bf16x3_router_gemm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Return ``X @ W.T`` using the SM100 BF16x3 router GEMM kernels."""
    M, K = X.shape
    N, _ = W.shape
    num_sms = torch.cuda.get_device_properties(X.device).multi_processor_count
    config, split_k = _pick_config(M, K, N, num_sms)

    partials = X.new_empty(split_k, M, N, dtype=torch.float32)
    if isinstance(config, LargeMConfig):
        if not W.is_contiguous():
            # the decomp kernel indexes W flat; the small-M kernel takes strides
            raise ValueError("BF16x3 router GEMM: large-M path needs a contiguous W")
        w3 = W.new_empty(3, N, K, dtype=torch.bfloat16)
        _bf16x3_decomp_kernel[lambda meta: (triton.cdiv(N * K, meta["BLOCK"]),)](
            W, w3, N * K, launch_pdl=True
        )
        Sm100BF16x3RouterGemmLargeM.compile(K, config.cta_group)(
            X, w3, partials, split_k
        )
    else:
        Sm100BF16x3RouterGemmSmallM.compile(K, config.BM)(X, W, partials, split_k)

    if split_k == 1:
        return partials.squeeze(0)
    out = X.new_empty(M, N, dtype=torch.float32)
    splitk_reduce_triton(partials, out)
    return out


def _bf16x3_router_gemm_fake(
    X: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    return X.new_empty((X.shape[0], W.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="bf16x3_router_gemm",
    op_func=_bf16x3_router_gemm,
    fake_impl=_bf16x3_router_gemm_fake,
)


def bf16x3_router_gemm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.bf16x3_router_gemm(X, W)
