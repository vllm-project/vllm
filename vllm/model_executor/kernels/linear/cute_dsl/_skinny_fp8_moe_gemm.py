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


class CuteSkinnyFp8BlockMoeGemm:
    """Grouped W8A16 block-scaled FP8 GEMM for the routed experts of a MoE
    layer at small token counts.

    For every (token, expert) pair ``p`` computes
    ``C[p, :] = A[row(p), :] @ (B[e, :, :] * S[e])^T`` (times the routing
    weight of the pair when ``mul_routed_weight``), where ``e = ids[p]``,
    ``A`` is BF16, ``B[E, N, K]`` is FP8 E4M3 and ``S[E, N / 128, K / 128]``
    holds one FP32 scale per 128x128 weight block. With ``a_per_token`` the
    activation row is the pair's token (``p // top_k``, first GEMM of the
    layer); otherwise it is the pair itself (second GEMM). The activation is
    not quantized. Accumulation is FP32 and the output is BF16. A pair whose
    expert id is negative (padding) or not below ``E`` gets a zero output row.

    One CTA computes ``outputs_per_block`` outputs of one pair; the grid is
    ``(N / outputs_per_block, num_pairs)``. Each CTA's threads stride over K in
    ``vector_width``-wide FP8 vectors, each of which lies inside one 128-wide
    scale block, so a thread scales its partial dot product once per vector.
    Partial sums are reduced per warp and then across warps through shared
    memory. ``N`` must be a multiple of 128 so that an expert's rows start on
    a scale block.
    """

    def __init__(
        self,
        *,
        block_size: int,
        outputs_per_block: int,
        vector_width: int = 8,
        k_unroll: int = 1,
        a_per_token: bool = True,
        mul_routed_weight: bool = False,
        use_pdl: bool = False,
    ) -> None:
        if block_size % cute.arch.WARP_SIZE != 0:
            raise ValueError("block_size must be a multiple of the warp size")
        if vector_width not in (8, 16):
            raise ValueError("vector_width must be 8 or 16")
        if _QUANT_BLOCK % outputs_per_block != 0:
            raise ValueError("outputs_per_block must divide the 128-row scale block")
        self.block_size = block_size
        self.outputs_per_block = outputs_per_block
        self.vector_width = vector_width
        self.a_vectors = vector_width // _A_VECTOR
        self.k_unroll = k_unroll
        self.a_per_token = a_per_token
        self.mul_routed_weight = mul_routed_weight
        self.use_pdl = use_pdl
        self.num_warps = block_size // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gIds: cute.Tensor,
        gW: cute.Tensor,
        gC: cute.Tensor,
        n_per_expert: cutlass.Int32,
        top_k: cutlass.Int32,
        num_experts: cutlass.Int32,
        stream: CUstream,
    ) -> None:
        num_pairs = cute.size(gC, mode=[0])
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
        self.kernel(
            gA,
            gB,
            gS,
            gIds,
            gW,
            gC,
            n_per_expert,
            top_k,
            num_experts,
            k,
            copy_a,
            copy_b,
        ).launch(
            grid=[cute.ceil_div(n_per_expert, self.outputs_per_block), num_pairs, 1],
            block=[self.block_size, 1, 1],
            smem=self.outputs_per_block * self.num_warps * 4,
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
        gIds: cute.Tensor,
        gW: cute.Tensor,
        gC: cute.Tensor,
        n_per_expert: cutlass.Int32,
        top_k: cutlass.Int32,
        num_experts: cutlass.Int32,
        k_extent: cutlass.Int32,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, pair, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        outputs_per_block: cutlass.Constexpr = self.outputs_per_block
        vector_width: cutlass.Constexpr = self.vector_width
        a_vectors: cutlass.Constexpr = self.a_vectors
        block_size: cutlass.Constexpr = self.block_size
        num_warps: cutlass.Constexpr = self.num_warps
        k_tile_size: cutlass.Constexpr = block_size * vector_width
        vectors_per_scale: cutlass.Constexpr = _QUANT_BLOCK // vector_width

        acc = cute.make_rmem_tensor(
            cute.make_layout((outputs_per_block,), stride=(1,)), cutlass.Float32
        )
        acc.fill(0.0)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        expert = gIds[pair]
        a_row = pair
        if const_expr(self.a_per_token):
            a_row = pair // top_k
        n_local = block_idx * outputs_per_block
        n_base = expert * n_per_expert + n_local
        n_scale = n_base // _QUANT_BLOCK

        # ((8, K / 8)) for the A row and (E * N, (vector_width, K / vector_width))
        # for B.
        gA_vec = cute.logical_divide(gA, (None, _A_VECTOR))
        gB_vec = cute.logical_divide(gB, (None, vector_width))
        a_regs = cute.make_rmem_tensor(
            cute.make_layout((a_vectors, _A_VECTOR), stride=(_A_VECTOR, 1)),
            cutlass.BFloat16,
        )
        b_regs = cute.make_rmem_tensor(
            cute.make_layout(
                (outputs_per_block, vector_width), stride=(vector_width, 1)
            ),
            cutlass.Float8E4M3FN,
        )

        # A negative expert id marks a pair with no expert (padding in warmup
        # and profiling runs); it and any id outside the stack give a zero row.
        valid = (expert >= 0) & (expert < num_experts)
        if valid:
            num_k_tiles = k_extent // k_tile_size
            for k_tile in cutlass.range(num_k_tiles, unroll=self.k_unroll):
                vec_idx = k_tile * block_size + tidx
                for ni in cutlass.range_constexpr(outputs_per_block):
                    cute.copy(
                        copy_b,
                        gB_vec[n_base + ni, (None, vec_idx)],
                        b_regs[ni, None],
                    )
                for ai in cutlass.range_constexpr(a_vectors):
                    cute.copy(
                        copy_a,
                        gA_vec[a_row, (None, vec_idx * a_vectors + ai)],
                        a_regs[ai, None],
                    )
                scale = gS[n_scale, vec_idx // vectors_per_scale]
                b_f32 = b_regs.load().to(cutlass.Float32)
                a_f32 = a_regs.load().to(cutlass.Float32)
                for ni in cutlass.range_constexpr(outputs_per_block):
                    part = cutlass.Float32(0.0)
                    for vi in cutlass.range_constexpr(vector_width):
                        part = (
                            part
                            + a_f32[vi // _A_VECTOR, vi % _A_VECTOR] * b_f32[ni, vi]
                        )
                    acc[ni] = acc[ni] + part * scale

        for ni in cutlass.range_constexpr(outputs_per_block):
            acc[ni] = cute.arch.warp_reduction_sum(acc[ni])

        smem = cutlass.utils.SmemAllocator()
        partials = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((outputs_per_block, num_warps), stride=(num_warps, 1)),
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for ni in cutlass.range_constexpr(outputs_per_block):
                partials[ni, warp_idx] = acc[ni]

        cute.arch.sync_threads()
        if tidx == 0:
            routed_weight = cutlass.Float32(1.0)
            # Compile-time and runtime conditions cannot share one `if`.
            if const_expr(self.mul_routed_weight):  # noqa: SIM102
                if valid:
                    routed_weight = gW[pair]
            for ni in cutlass.range_constexpr(outputs_per_block):
                total = (
                    partials[ni, None]
                    .load()
                    .reduce(
                        cute.ReductionOp.ADD,
                        init_val=cutlass.Float32(0.0),
                        reduction_profile=0,
                    )
                )
                gC[pair, n_local + ni] = (cutlass.Float32(total) * routed_weight).to(
                    cutlass.BFloat16
                )

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
