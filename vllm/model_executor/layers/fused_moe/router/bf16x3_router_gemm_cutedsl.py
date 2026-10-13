# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL BF16x3 router GEMM.

Computes ``X @ W.T`` for BF16 ``X`` with shape ``[N, K]`` and FP32 router
weights ``W`` with shape ``[M, K]`` by decomposing each FP32 weight value into
three BF16 residual terms inside the kernel, then accumulating the three BF16
MMA results into FP32 TMEM output.

The TMEM accumulation chain is bounded to ``num_tmem_acc`` K-tiles: at chunk
boundaries the epilogue warps drain the main-term accumulator into a register
master accumulator and the MMA warp resets it.
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import _tcgen05, simple_tma_copy
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


class Sm100BF16x3RouterGemm:
    def __init__(self, BN: int = 128) -> None:
        self.cta_tile = (BN, 128, 64)
        self.num_stages = 2
        self.num_warps = 10
        self.num_tmem_acc = 8

    @cute.jit
    def _make_tma(self, tensor: cute.Tensor, BM: int, BK: int):
        op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // tensor.element_type.width  # 128B
        slayout = cute.make_layout(
            (BM, (elems, BK // elems), self.num_stages),
            stride=(elems, (1, BM * elems), BM * BK),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(op, tensor, slayout, (BM, BK))

    @cute.jit
    def __call__(
        self,
        X: cute.Tensor,
        W: cute.Tensor,
        out: cute.Tensor,
        split_k: Int32,
        stream: CUstream,
    ):
        BN, BM, BK = self.cta_tile
        W_tma = self._make_tma(W, BM, BK)
        X_tma = self._make_tma(X, BN, BK)

        grid_m = cute.ceil_div(W.shape[0], BM)
        grid_n = cute.ceil_div(X.shape[0], BN)

        self.kernel(X_tma, W_tma, out).launch(
            grid=(grid_m, grid_n, split_k),
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, X_tma: cpasync.TmaInfo, W_tma: cpasync.TmaInfo, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_k = cute.arch.block_idx()
        _, _, split_k = cute.arch.grid_dim()

        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        BN, BM, BK = self.cta_tile
        num_stages = self.num_stages
        num_tmem_acc = self.num_tmem_acc

        N, K = X_tma.tma_tensor.shape
        M, _ = W_tma.tma_tensor.shape
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
        acc_res = BN
        w_tmem_base = BN * 2

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

            # (BM, BK, K/BK)
            gW_tiles = cute.local_tile(W_tma.tma_tensor, (BM, BK), (bid_m, None))
            gX_tiles = cute.local_tile(X_tma.tma_tensor, (BN, BK), (bid_n, None))

            for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                cute.arch.mbarrier_wait(tma_empty_mbar + stage_id, parity)
                mbar = tma_full_mbar + stage_id
                with cute.arch.elect_one():
                    stage_bytes = BN * BK * 2 + BM * BK * 4
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

            idesc = _tcgen05.make_bf16_idesc(BM, BN)
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
                w_row_idx = bid_m * BM + tid
                for i in cutlass.range_constexpr(BN // WIDTH):
                    tcol = i * WIDTH
                    main_regs.store(_tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH))
                    res_regs.store(
                        _tcgen05.ld(warp_id * 32, BN + tcol, "32x32b", WIDTH)
                    )
                    _tcgen05.wait_ld()

                    # CuteDSL will codegen add.f32x2
                    for j in cutlass.range(WIDTH, vectorize=True):
                        main_regs[j] += res_regs[j]

                    for j in cutlass.range_constexpr(WIDTH):
                        x_row_idx = bid_n * BN + i * WIDTH + j
                        if x_row_idx < N and w_row_idx < M:
                            out[bid_k, x_row_idx, w_row_idx] = main_regs[j]

            else:
                # multiple chunks
                master_acc = cute.make_rmem_tensor(BN, Float32)

                # chunk 0: pure TMEM load, no accumulation.
                cute.arch.mbarrier_wait(acc_full_mbar, 0)
                _tcgen05.fence_after_thread_sync()
                master_acc.store(_tcgen05.ld(warp_id * 32, 0, "32x32b", BN))
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(acc_empty_mbar)

                # accumulate main tmem acc
                for chunk in cutlass.range(1, num_chunks, unroll=1):
                    cute.arch.mbarrier_wait(acc_full_mbar, chunk & 1)
                    _tcgen05.fence_after_thread_sync()

                    for i in cutlass.range_constexpr(BN // WIDTH):
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
                for i in cutlass.range_constexpr(BN // WIDTH):
                    tcol = i * WIDTH
                    res_regs.store(
                        _tcgen05.ld(warp_id * 32, BN + tcol, "32x32b", WIDTH)
                    )
                    _tcgen05.wait_ld()
                    master_view = cute.local_tile(master_acc, (WIDTH,), (i,))
                    for j in cutlass.range(WIDTH, vectorize=True):
                        master_view[j] += res_regs[j]

                    w_row_idx = bid_m * BM + tid
                    for j in cutlass.range_constexpr(WIDTH):
                        x_row_idx = bid_n * BN + i * WIDTH + j
                        if x_row_idx < N and w_row_idx < M:
                            out[bid_k, x_row_idx, w_row_idx] = master_acc[i * WIDTH + j]

            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=128)
            if warp_id == 0:
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile(K: int, BN: int = 128):
        N = cute.sym_int()
        M = cute.sym_int()
        SPLIT_K = cute.sym_int()
        X = make_fake_tensor(BFloat16, (N, K), divisibility=8)
        W = make_fake_tensor(Float32, (M, K), divisibility=4)
        out = make_fake_tensor(Float32, (SPLIT_K, N, M), divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100BF16x3RouterGemm(BN)
        return cute.compile(
            kernel, X, W, out, Int32(1), stream, options="--enable-tvm-ffi"
        )


@triton.jit
def _splitk_reduce_kernel(
    partials,
    out,
    N,
    M: tl.constexpr,
    split_stride,
    k_splits,
    BN: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_s = tl.arange(0, BS)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    vals = tl.load(
        partials
        + offs_s[:, None, None] * split_stride
        + offs_n[None, :, None] * M
        + offs_m[None, None, :],
        mask=(
            (offs_s[:, None, None] < k_splits)
            & (offs_n[None, :, None] < N)
            & (offs_m[None, None, :] < M)
        ),
        other=0.0,
    )
    acc = tl.sum(vals, axis=0)
    tl.store(
        out + offs_n[:, None] * M + offs_m[None, :],
        acc,
        mask=(offs_n[:, None] < N) & (offs_m[None, :] < M),
    )


def _splitk_reduce_config(split_k: int) -> tuple[int, int, int]:
    block_s = triton.next_power_of_2(split_k)
    if block_s >= 64:
        return 1, 32, block_s
    if block_s >= 8:
        return 1, 256, block_s
    return min(16, 32 // block_s), 32, block_s


def splitk_reduce_triton(partials: torch.Tensor, out: torch.Tensor):
    split_k, N, M = partials.shape
    split_stride = partials.stride(0)
    BN, BM, block_s = _splitk_reduce_config(split_k)
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
    _splitk_reduce_kernel[grid](
        partials,
        out,
        N,
        M,
        split_stride,
        split_k,
        BN=BN,
        BM=BM,
        BS=block_s,
        USE_PDL=True,
        num_warps=4,
        launch_pdl=True,
    )


# Accuracy-constrained (BN, split_k) configs from GB300 sweeps.
_TILE_CONFIG_OVERRIDES = {
    (2816, 256): {  # Hunyuan-V4
        64: (16, 16),
        128: (32, 16),
        256: (64, 16),
        512: (128, 16),
    },
    (3072, 256): {  # MiniMax-M2
        64: (16, 16),
        128: (32, 16),
        256: (64, 16),
        512: (128, 16),
    },
    (4096, 192): {  # Hunyuan-V3
        64: (32, 32),
        128: (64, 32),
        256: (128, 32),
        512: (128, 19),
    },
    (6144, 128): {  # MiniMax-M3
        64: (32, 64),
        128: (64, 64),
        256: (64, 32),
        512: (128, 32),
    },
}


def _pick_tile_config(N: int, K: int, M: int, num_sms: int) -> tuple[int, int]:
    """Return (BN, split_k): tuned table mid-range, generic rule otherwise."""
    k_tiles = math_utils.cdiv(K, 64)

    if 32 < N <= 512:
        token_bucket = triton.next_power_of_2(N)
        overrides = _TILE_CONFIG_OVERRIDES.get((K, M))
        if overrides is not None:
            BN, split_k = overrides[token_bucket]
            return BN, min(split_k, k_tiles)

    # next power of 2 within 8 and 128
    BN = triton.next_power_of_2(N)
    BN = min(max(BN, 8), 128)

    grid_m = math_utils.cdiv(M, 128)
    grid_n = math_utils.cdiv(N, BN)
    base_ctas = grid_m * grid_n
    split_k = min(k_tiles, max(1, num_sms // base_ctas))
    return BN, split_k


def warmup_bf16x3_router_gemm(
    K: int,
    M: int,
    min_num_tokens: int,
    max_num_tokens: int,
) -> tuple[int, ...]:
    """Compile every BF16x3 router GEMM configuration reachable at runtime."""
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        triton_scalar_specialization_rep,
    )

    device = torch.accelerator.current_device_index()
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    gemm_configs: set[int] = set()
    reduce_configs: set[tuple[int, int, int, int, int, int]] = set()
    for N in range(min_num_tokens, max_num_tokens + 1):
        BN, split_k = _pick_tile_config(N, K, M, num_sms)
        gemm_configs.add(BN)
        if split_k == 1:
            continue
        reduce_BN, reduce_BM, block_s = _splitk_reduce_config(split_k)
        reduce_configs.add(
            (
                triton_scalar_specialization_rep(N),
                triton_scalar_specialization_rep(N * M),
                triton_scalar_specialization_rep(split_k),
                reduce_BN,
                reduce_BM,
                block_s,
            )
        )

    configs = tuple(sorted(gemm_configs))
    for BN in configs:
        Sm100BF16x3RouterGemm.compile(K, BN)
    for N, split_stride, split_k, BN, BM, block_s in sorted(reduce_configs):
        partials = TritonWarmupTensor(torch.float32)
        out = TritonWarmupTensor(torch.float32)
        _splitk_reduce_kernel.warmup(
            partials,
            out,
            N,
            M,
            split_stride,
            split_k,
            BN=BN,
            BM=BM,
            BS=block_s,
            USE_PDL=True,
            num_warps=4,
            launch_pdl=True,
            grid=(1, 1),
        )
    return configs


def _bf16x3_router_gemm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Return ``X @ W.T`` using the SM100 BF16x3 router GEMM kernel."""
    N, K = X.shape
    M, _ = W.shape
    num_sms = torch.cuda.get_device_properties(X.device).multi_processor_count

    BN, split_k = _pick_tile_config(N, K, M, num_sms)

    partials = X.new_empty(split_k, N, M, dtype=torch.float32)
    Sm100BF16x3RouterGemm.compile(K, BN)(X, W, partials, split_k)

    if split_k == 1:
        return partials.squeeze(0)

    out = X.new_empty(N, M, dtype=torch.float32)
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
