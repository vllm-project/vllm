# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuteDSL A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

Dot-product kernel for low-latency problems using FMA instructions.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
class LLBf16Dotprod:
    """Dot-product router GEMM for low-M bf16 inputs."""

    def __init__(
        self,
        k: int,
        bs: int = 128,
        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        *,
        loc=None,
        ip=None,
    ):
        self.bs = bs
        self.main_vec_width = main_vec_width
        self.tail_vec_width = tail_vec_width
        self.num_warps = bs // cute.arch.WARP_SIZE

        self.k_main_elems = (k // (main_vec_width * bs)) * main_vec_width * bs
        self.k_after_main = k - self.k_main_elems
        self.k_tail_elems = (
            self.k_after_main // (tail_vec_width * bs)
        ) * tail_vec_width * bs
        self.k_done_all = self.k_main_elems + self.k_tail_elems
        self.scalar_rem = k - self.k_done_all
        self.ks_full = self.scalar_rem // bs
        self.ks_part = self.scalar_rem % bs
        self.k_scalar_full = self.ks_full * bs
        self.k_part_offset = self.k_done_all + self.k_scalar_full
        self.main_tiles = self.k_main_elems // (main_vec_width * bs)
        self.tail_tiles = self.k_tail_elems // (tail_vec_width * bs)

    def _vector_dotprod(
        self,
        acc: cute.Tensor,
        tA: cute.Tensor,
        tB: cute.Tensor,
        M: cutlass.Constexpr,
        num_tiles: cutlass.Constexpr,
        align_bytes: cutlass.Constexpr,
    ):
        for tile in range(num_tiles):
            bt = tB[None, tile]
            bt_a = cute.make_tensor(bt.iterator.align(align_bytes), bt.layout)
            br = cute.make_rmem_tensor_like(bt_a)
            cute.autovec_copy(bt_a, br)
            br_f32 = br.load().to(cutlass.Float32)

            for m in range(M):
                at = tA[m, None, tile]
                at_a = cute.make_tensor(at.iterator.align(align_bytes), at.layout)
                ar = cute.make_rmem_tensor_like(at_a)
                cute.autovec_copy(at_a, ar)
                vec_width: cutlass.Constexpr = cute.size(ar)
                for v in range(vec_width):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

    def _make_thread_vector_slice(
        self,
        gA_vec: cute.Tensor,
        gB_vec: cute.Tensor,
        tidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        bs: cutlass.Constexpr,
    ):
        tA = cute.logical_divide(gA_vec, (None, (None, bs)))
        tB = cute.logical_divide(gB_vec, (None, (None, bs)))
        return tA[None, (None, (tidx, None))], tB[n_idx, (None, (tidx, None))]

    def _make_k_slice(
        self,
        gX: cute.Tensor,
        k_offset: cutlass.Constexpr,
        k_extent: cutlass.Constexpr,
    ):
        k_layout_extent: cutlass.Constexpr = k_extent
        if cutlass.const_expr(k_extent == 0):
            k_layout_extent = 1
        if cutlass.const_expr(k_offset == 0):
            return cute.local_tile(
                gX, (cute.size(gX, mode=[0]), k_layout_extent), (0, 0)
            )
        return cute.local_tile(
            cute.domain_offset((0, k_offset), gX),
            (cute.size(gX, mode=[0]), k_layout_extent),
            (0, 0),
        )

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        M: cutlass.Constexpr,
        K_dim: cutlass.Constexpr,
        N_dim: cutlass.Int32,
        stream: CUstream,
    ):
        main_vec_width: cutlass.Constexpr = self.main_vec_width
        tail_vec_width: cutlass.Constexpr = self.tail_vec_width
        bs: cutlass.Constexpr = self.bs
        k_main_elems: cutlass.Constexpr = self.k_main_elems
        k_tail_elems: cutlass.Constexpr = self.k_tail_elems
        k_done_all: cutlass.Constexpr = self.k_done_all
        ks_full: cutlass.Constexpr = self.ks_full
        ks_part: cutlass.Constexpr = self.ks_part
        k_scalar_full: cutlass.Constexpr = self.k_scalar_full
        k_part_offset: cutlass.Constexpr = self.k_part_offset
        main_tiles: cutlass.Constexpr = self.main_tiles
        tail_tiles: cutlass.Constexpr = self.tail_tiles

        self.kernel(
            gA,
            gB,
            gC,
            M,
            main_vec_width,
            tail_vec_width,
            bs,
            self.num_warps,
            k_main_elems,
            k_tail_elems,
            k_done_all,
            ks_full,
            ks_part,
            k_scalar_full,
            k_part_offset,
            main_tiles,
            tail_tiles,
        ).launch(
            grid=[N_dim, 1, 1],
            block=[bs, 1, 1],
            smem=M * 4 * self.num_warps,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        M: cutlass.Constexpr,
        main_vec_width: cutlass.Constexpr,
        tail_vec_width: cutlass.Constexpr,
        bs: cutlass.Constexpr,
        num_warps: cutlass.Constexpr,
        k_main_elems: cutlass.Constexpr,
        k_tail_elems: cutlass.Constexpr,
        k_done_all: cutlass.Constexpr,
        ks_full: cutlass.Constexpr,
        ks_part: cutlass.Constexpr,
        k_scalar_full: cutlass.Constexpr,
        k_part_offset: cutlass.Constexpr,
        main_tiles: cutlass.Constexpr,
        tail_tiles: cutlass.Constexpr,
    ):
        cute.arch.setmaxregister_increase(128)

        tidx, _, _ = cute.arch.thread_idx()
        n_idx, _, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()

        # One FP32 accumulator per token.
        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        acc.fill(0.0)

        cute.arch.griddepcontrol_wait()  # PDL wait

        # 128-bit vectorized main loop
        if cutlass.const_expr(k_main_elems > 0):
            gA_main = self._make_k_slice(gA, 0, k_main_elems)
            gB_main = self._make_k_slice(gB, 0, k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, main_vec_width))
            tA, tB = self._make_thread_vector_slice(gA_vec, gB_vec, tidx, n_idx, bs)
            self._vector_dotprod(acc, tA, tB, M, main_tiles, 16)

        # 64-bit vectorized tail (K remainder after main loop)
        if cutlass.const_expr(k_tail_elems > 0):
            gA_tail = self._make_k_slice(gA, k_main_elems, k_tail_elems)
            gB_tail = self._make_k_slice(gB, k_main_elems, k_tail_elems)
            gA_tail_vec = cute.logical_divide(gA_tail, (None, tail_vec_width))
            gB_tail_vec = cute.logical_divide(gB_tail, (None, tail_vec_width))
            tA_t, tB_t = self._make_thread_vector_slice(
                gA_tail_vec, gB_tail_vec, tidx, n_idx, bs
            )
            self._vector_dotprod(acc, tA_t, tB_t, M, tail_tiles, 8)

        # Full scalar rounds use CuTe width-1 tiles; KS_PART is the ragged tail.
        if cutlass.const_expr(ks_full > 0):
            gA_scalar = self._make_k_slice(gA, k_done_all, k_scalar_full)
            gB_scalar = self._make_k_slice(gB, k_done_all, k_scalar_full)
            gA_scalar_vec = cute.logical_divide(gA_scalar, (None, 1))
            gB_scalar_vec = cute.logical_divide(gB_scalar, (None, 1))
            tA_s, tB_s = self._make_thread_vector_slice(
                gA_scalar_vec, gB_scalar_vec, tidx, n_idx, bs
            )
            self._vector_dotprod(acc, tA_s, tB_s, M, ks_full, 2)

        # Only threads below KS_PART load the ragged tail.
        if cutlass.const_expr(ks_part > 0):
            gA_part = self._make_k_slice(gA, k_part_offset, ks_part)
            gB_part = self._make_k_slice(gB, k_part_offset, ks_part)
            if tidx < ks_part:
                bv2 = gB_part[n_idx, tidx].to(cutlass.Float32)
                for m in cutlass.range_constexpr(M):
                    acc[m] = acc[m] + gA_part[m, tidx].to(cutlass.Float32) * bv2

        # Intra-warp shuffle reduction
        for m in cutlass.range_constexpr(M):
            acc[m] = cute.arch.warp_reduction_sum(acc[m])

        # Cross-warp reduction via shared memory
        smem_red_layout = cute.make_layout((M, num_warps), stride=(num_warps, 1))
        smem = cutlass.utils.SmemAllocator()
        sm = smem.allocate_tensor(cutlass.Float32, smem_red_layout, byte_alignment=16)
        with cute.arch.elect_one():
            for m in cutlass.range_constexpr(M):
                sm[m, wid] = acc[m]

        # Final reduction and output
        cute.arch.sync_threads()
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                partials = sm[m, None].load()
                gC[m, n_idx] = partials.reduce(
                    cute.ReductionOp.ADD,
                    init_val=cutlass.Float32(0.0),
                    reduction_profile=0,
                )
        cute.arch.griddepcontrol_launch_dependents()  # PDL signal


def make_host_bf16(k_val: int, bs: int = 128):
    return LLBf16Dotprod(k=k_val, bs=bs)
