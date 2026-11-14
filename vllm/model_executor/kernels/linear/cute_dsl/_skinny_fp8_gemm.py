# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

# Weight quantization block (N and K) of the block-scaled FP8 checkpoints.
_QUANT_BLOCK = 128
# A is read in 128-bit vectors of BF16.
_A_VECTOR = 8


class CuteSkinnyFp8BlockGemm:
    """W8A16 block-scaled FP8 GEMM for small token counts.

    Computes ``C[M, N] = A[M, K] @ (B[N, K] * S)^T`` where ``A`` is BF16, ``B``
    is FP8 E4M3 and ``S[N / 128, K / 128]`` holds one FP32 scale per 128x128
    weight block. The activation is not quantized. Accumulation is FP32 and the
    output is BF16.

    One CTA computes ``outputs_per_block`` outputs (all inside one 128-row
    scale block); its threads stride over K in ``vector_width``-wide FP8
    vectors, each of which lies inside one 128-wide scale block, so a thread
    scales its partial dot product once per vector. Partial sums are reduced
    per warp and then across warps through shared memory. N and K are runtime
    values; the tiny M dimension is unrolled.
    """

    def __init__(
        self,
        *,
        num_rows: int,
        block_size: int,
        outputs_per_block: int,
        vector_width: int = 8,
        k_unroll: int = 1,
        use_pdl: bool = False,
    ) -> None:
        if block_size % cute.arch.WARP_SIZE != 0:
            raise ValueError("block_size must be a multiple of the warp size")
        if vector_width not in (8, 16):
            raise ValueError("vector_width must be 8 or 16")
        if _QUANT_BLOCK % outputs_per_block != 0:
            raise ValueError("outputs_per_block must divide the 128-row scale block")
        self.num_rows = num_rows
        self.block_size = block_size
        self.outputs_per_block = outputs_per_block
        self.vector_width = vector_width
        self.a_vectors = vector_width // _A_VECTOR
        self.k_unroll = k_unroll
        self.use_pdl = use_pdl
        self.num_warps = block_size // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        stream: CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        k = cute.size(gA, mode=[1])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=_A_VECTOR * cutlass.BFloat16.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Float8E4M3FN,
            num_bits_per_copy=self.vector_width * cutlass.Float8E4M3FN.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gB, gS, gC, k, copy_a, copy_b).launch(
            grid=[cute.ceil_div(n, self.outputs_per_block), 1, 1],
            block=[self.block_size, 1, 1],
            smem=self.num_rows * self.outputs_per_block * self.num_warps * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        k_extent: cutlass.Int32,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.num_rows
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        a_vectors: cutlass.Constexpr = self.a_vectors
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        k_tile_size: cutlass.Constexpr = block_size * vector_width
        vectors_per_scale: cutlass.Constexpr = _QUANT_BLOCK // vector_width

        acc = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
            ),
            cutlass.Float32,
        )
        acc.fill(0.0)

        n_base = block_idx * outputs_per_block
        n_scale = n_base // _QUANT_BLOCK

        # (M, (8, K / 8)) for A and (N, (vector_width, K / vector_width)) for B.
        gA_vec = cute.logical_divide(gA, (None, _A_VECTOR))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        a_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, a_vectors, _A_VECTOR),
                stride=(a_vectors * _A_VECTOR, _A_VECTOR, 1),
            ),
            cutlass.BFloat16,
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, vector_width), stride=(vector_width, 1)
            ),
            cutlass.Float8E4M3FN,
        )

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        num_k_tiles = k_extent // k_tile_size
        for k_tile in cutlass.range(num_k_tiles, unroll=self.k_unroll):
            vec_idx = k_tile * block_size + tidx
            for ni in cutlass.range_constexpr(outputs_per_block):
                cute.copy(
                    copy_b, gB_vec[n_base + ni, (None, vec_idx)], b_regs[ni, None]
                )
            for mi in cutlass.range_constexpr(num_rows):
                for ai in cutlass.range_constexpr(a_vectors):
                    cute.copy(
                        copy_a,
                        gA_vec[mi, (None, vec_idx * a_vectors + ai)],
                        a_regs[mi, ai, None],
                    )
            scale = gS[n_scale, vec_idx // vectors_per_scale]
            b_f32 = b_regs.load().to(cutlass.Float32)
            a_f32 = a_regs.load().to(cutlass.Float32)
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    part = cutlass.Float32(0.0)
                    for vi in cutlass.range_constexpr(vector_width):
                        part = (
                            part
                            + a_f32[mi, vi // _A_VECTOR, vi % _A_VECTOR] * b_f32[ni, vi]
                        )
                    acc[mi, ni] = acc[mi, ni] + part * scale

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (num_rows, outputs_per_block, num_warps),
                stride=(outputs_per_block * num_warps, num_warps, 1),
            ),
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    total = (
                        partials[mi, ni, None]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=cutlass.Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    gC[mi, n_base + ni] = cutlass.Float32(total).to(cutlass.BFloat16)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
