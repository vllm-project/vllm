# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

_PREFETCH_TILES = 2


class CuteSkinnyGemm:
    """Shape-dynamic low-latency GEMM for small token counts.

    Computes ``C[M, N] = A[M, K] @ B[N, K].T + residual`` with BF16 or FP16
    inputs, FP32 accumulators, and an output matching the input dtype. The
    residual term is optional and is added before the output conversion. N and
    K are runtime values unless the configuration specializes K at compile
    time. The tiny M dimension is fully unrolled alongside a small set of
    tuning parameters.
    """

    def __init__(
        self,
        *,
        element_type,
        num_rows: int,
        block_size: int,
        outputs_per_block: int,
        vector_width: int = 8,
        k_unroll: int = 1,
        has_residual: bool = False,
        use_pdl: bool = False,
        static_k: int | None = None,
    ) -> None:
        if block_size % cute.arch.WARP_SIZE != 0:
            raise ValueError("block_size must be a multiple of the warp size")
        k_tile_size = block_size * vector_width
        if static_k is not None and (
            static_k % k_tile_size or static_k < _PREFETCH_TILES * k_tile_size
        ):
            raise ValueError("static K must contain at least two complete tiles")
        self.element_type = element_type
        self.num_rows = num_rows
        self.block_size = block_size
        self.outputs_per_block = outputs_per_block
        self.vector_width = vector_width
        self.k_unroll = k_unroll
        self.has_residual = has_residual
        self.use_pdl = use_pdl
        self.static_k = static_k
        self.num_warps = block_size // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gResidual: cute.Tensor,
        gC: cute.Tensor,
        stream: CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        k = cute.size(gA, mode=[1])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        copy_b_prefetch = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            self.element_type,
            num_bits_per_copy=self.vector_width * self.element_type.width,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        self.kernel(
            gA,
            gB,
            gResidual,
            gC,
            k,
            copy_a,
            copy_b,
            copy_b_prefetch,
        ).launch(
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
        gResidual: cute.Tensor,
        gC: cute.Tensor,
        k_extent: cutlass.Int32,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
        copy_b_prefetch: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.num_rows
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps

        acc_layout = cute.make_layout(
            (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
        )
        acc = cute.make_rmem_tensor(acc_layout, cutlass.Float32)
        acc.fill(0.0)

        n_base = block_idx * outputs_per_block
        k_tile_size: cutlass.Constexpr = block_size * vector_width

        gA_vec = cute.logical_divide(gA, (None, vector_width))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        # Layout after both divides is (M/N, K_TILE, K_LANE, K_VEC).
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_rows, vector_width), stride=(vector_width, 1)),
            self.element_type,
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, vector_width), stride=(vector_width, 1)
            ),
            self.element_type,
        )

        if const_expr(self.static_k is not None):
            num_k_tiles: cutlass.Constexpr = self.static_k // k_tile_size
            prefetched_b = cute.make_rmem_tensor(
                cute.make_layout(
                    (_PREFETCH_TILES, outputs_per_block, vector_width),
                    stride=(outputs_per_block * vector_width, vector_width, 1),
                ),
                self.element_type,
            )
            for k_tile in cutlass.range_constexpr(_PREFETCH_TILES):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    tB_static = tB_all[n_base + ni, (None, (tidx, None))]
                    cute.copy(
                        copy_b_prefetch,
                        tB_static[None, k_tile],
                        prefetched_b[k_tile, ni, None],
                    )

            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            for k_tile in cutlass.range_constexpr(num_k_tiles):
                if const_expr(k_tile < _PREFETCH_TILES):
                    b_f32 = prefetched_b[k_tile, None, None].load().to(cutlass.Float32)
                else:
                    for ni in cutlass.range_constexpr(outputs_per_block):
                        tB_static = tB_all[n_base + ni, (None, (tidx, None))]
                        cute.copy(
                            copy_b,
                            tB_static[None, k_tile],
                            b_regs[ni, None],
                        )
                    b_f32 = b_regs.load().to(cutlass.Float32)

                for mi in cutlass.range_constexpr(num_rows):
                    cute.copy(copy_a, tA[mi, None, k_tile], a_regs[mi, None])
                a_f32 = a_regs.load().to(cutlass.Float32)

                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] += a_f32[mi, vi] * b_f32[ni, vi]
        else:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            num_k_tiles = k_extent // k_tile_size
            for k_tile in cutlass.range(num_k_tiles, unroll=self.k_unroll):
                for mi in cutlass.range_constexpr(num_rows):
                    cute.copy(copy_a, tA[mi, None, k_tile], a_regs[mi, None])

                for ni in cutlass.range_constexpr(outputs_per_block):
                    n_idx = n_base + ni
                    tB_dynamic = tB_all[n_idx, (None, (tidx, None))]
                    cute.copy(
                        copy_b,
                        tB_dynamic[None, k_tile],
                        b_regs[ni, None],
                    )

                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(
                                cutlass.Float32
                            ) * b_regs[ni, vi].to(cutlass.Float32)

        for mi in cutlass.range_constexpr(num_rows):
            for ni in cutlass.range_constexpr(outputs_per_block):
                acc[mi, ni] = cute.arch.warp_reduction_sum(acc[mi, ni])

        smem_layout = cute.make_layout(
            (num_rows, outputs_per_block, num_warps),
            stride=(outputs_per_block * num_warps, num_warps, 1),
        )
        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    partials[mi, ni, warp_idx] = acc[mi, ni]

        cute.arch.sync_threads()
        if tidx == 0:
            for mi in cutlass.range_constexpr(num_rows):
                for ni in cutlass.range_constexpr(outputs_per_block):
                    n_idx = n_base + ni
                    total = (
                        partials[mi, ni, None]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=cutlass.Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    if const_expr(self.has_residual):
                        total += gResidual[mi, n_idx].to(cutlass.Float32)
                    gC[mi, n_idx] = cutlass.Float32(total).to(self.element_type)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
