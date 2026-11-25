# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

_PREFETCH_TILES = 2
# FP8 elements per vectorized (16-byte) load.
_VEC = 16
# K and N extent of one weight scale block, and K extent of one activation
# scale group.
_BLOCK = 128


class CuteSkinnyFp8BlockGemm:
    """Low-latency block-scaled FP8 GEMM for small token counts.

    Computes `C[M, N] = (A[M, K] * As) @ (B[N, K] * Bs).T` for FP8 (e4m3)
    activations with one FP32 scale per (row, 128 columns) of K (`As`, stored
    column-major as the CUTLASS block-FP8 kernel takes it) and FP8 weights with
    one FP32 scale per 128 x 128 block (`Bs`), accumulating in FP32 and
    writing BF16/FP16.

    The structure follows CuteSkinnyGemm: each CTA owns `outputs_per_block`
    consecutive output columns, every thread streams 16 contiguous FP8 values
    of K per tile (one 16-byte load per operand row), and partial sums are
    reduced across the CTA. A thread's 16 values always lie in one 128-wide K
    group, so its partial dot product is scaled by that group's activation and
    weight scales before it is accumulated. `outputs_per_block` divides 128,
    so a CTA's columns share one row of weight scales. K is static.
    """

    def __init__(
        self,
        *,
        out_type,
        num_rows: int,
        block_size: int,
        outputs_per_block: int,
        static_k: int,
        use_pdl: bool = False,
    ) -> None:
        if block_size % cute.arch.WARP_SIZE != 0:
            raise ValueError("block_size must be a multiple of the warp size")
        if _BLOCK % outputs_per_block != 0:
            raise ValueError("outputs_per_block must divide 128")
        k_tile_size = block_size * _VEC
        if k_tile_size % _BLOCK != 0:
            raise ValueError("block_size * 16 must be a multiple of 128")
        if static_k % k_tile_size or static_k < k_tile_size:
            raise ValueError("static K must be a whole number of tiles")
        self.out_type = out_type
        self.num_rows = num_rows
        self.block_size = block_size
        self.outputs_per_block = outputs_per_block
        self.static_k = static_k
        self.use_pdl = use_pdl
        self.num_warps = block_size // cute.arch.WARP_SIZE
        self.num_prefetch = min(_PREFETCH_TILES, static_k // k_tile_size)

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gAs: cute.Tensor,
        gB: cute.Tensor,
        gBs: cute.Tensor,
        gC: cute.Tensor,
        stream: CUstream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Float8E4M3FN,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Float8E4M3FN,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gAs, gB, gBs, gC, copy_a, copy_b).launch(
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
        gAs: cute.Tensor,
        gB: cute.Tensor,
        gBs: cute.Tensor,
        gC: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        num_rows: cutlass.Constexpr = self.num_rows
        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        k_tile_size: cutlass.Constexpr = block_size * _VEC
        num_k_tiles: cutlass.Constexpr = self.static_k // k_tile_size
        num_prefetch: cutlass.Constexpr = self.num_prefetch
        # K groups (of 128) per tile; this thread's group within a tile.
        groups_per_tile: cutlass.Constexpr = k_tile_size // _BLOCK
        lanes_per_group: cutlass.Constexpr = _BLOCK // _VEC
        thread_group = tidx // lanes_per_group

        acc = cute.make_rmem_tensor(
            cute.make_layout(
                (num_rows, outputs_per_block), stride=(outputs_per_block, 1)
            ),
            cutlass.Float32,
        )
        acc.fill(0.0)

        n_base = block_idx * outputs_per_block
        n_block = n_base // _BLOCK

        gA_vec = cute.logical_divide(gA, (None, _VEC))
        gB_vec = cute.logical_divide(gB, (None, _VEC))
        # (M/N, K_TILE, K_LANE, K_VEC) after both divides.
        tA_all = cute.logical_divide(gA_vec, (None, (None, block_size)))
        tB_all = cute.logical_divide(gB_vec, (None, (None, block_size)))
        tA = tA_all[None, (None, (tidx, None))]

        a_regs = cute.make_rmem_tensor(
            cute.make_layout((num_rows, _VEC), stride=(_VEC, 1)),
            cutlass.Float8E4M3FN,
        )
        b_tiles = cute.make_rmem_tensor(
            cute.make_layout(
                (num_prefetch, outputs_per_block, _VEC),
                stride=(outputs_per_block * _VEC, _VEC, 1),
            ),
            cutlass.Float8E4M3FN,
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout((outputs_per_block, _VEC), stride=(_VEC, 1)),
            cutlass.Float8E4M3FN,
        )

        # Weights do not depend on the previous kernel: prefetch before the
        # PDL wait so their loads overlap its tail.
        for k_tile in cutlass.range_constexpr(num_prefetch):
            for ni in cutlass.range_constexpr(outputs_per_block):
                tB_n = tB_all[n_base + ni, (None, (tidx, None))]
                cute.copy(copy_b, tB_n[None, k_tile], b_tiles[k_tile, ni, None])

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        for k_tile in cutlass.range_constexpr(num_k_tiles):
            if const_expr(k_tile < num_prefetch):
                b_f32 = b_tiles[k_tile, None, None].load().to(cutlass.Float32)
            else:
                for ni in cutlass.range_constexpr(outputs_per_block):
                    tB_n = tB_all[n_base + ni, (None, (tidx, None))]
                    cute.copy(copy_b, tB_n[None, k_tile], b_regs[ni, None])
                b_f32 = b_regs.load().to(cutlass.Float32)

            for mi in cutlass.range_constexpr(num_rows):
                cute.copy(copy_a, tA[mi, None, k_tile], a_regs[mi, None])
            a_f32 = a_regs.load().to(cutlass.Float32)

            k_group = k_tile * groups_per_tile + thread_group
            w_scale = gBs[n_block, k_group]
            for mi in cutlass.range_constexpr(num_rows):
                scale = gAs[mi, k_group] * w_scale
                for ni in cutlass.range_constexpr(outputs_per_block):
                    dot = cutlass.Float32(0.0)
                    for vi in cutlass.range_constexpr(_VEC):
                        dot += a_f32[mi, vi] * b_f32[ni, vi]
                    acc[mi, ni] += dot * scale

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
                    total = (
                        partials[mi, ni, None]
                        .load()
                        .reduce(
                            cute.ReductionOp.ADD,
                            init_val=cutlass.Float32(0.0),
                            reduction_profile=0,
                        )
                    )
                    gC[mi, n_base + ni] = cutlass.Float32(total).to(self.out_type)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
