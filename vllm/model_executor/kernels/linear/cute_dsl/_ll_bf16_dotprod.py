# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


class LLBf16Dotprod:
    """BF16 router GEMM kernel based on CTA-local dot products.

    This kernel computes C[M, N] = A[M, K] @ B[N, K]^T for bf16 inputs
    and fp32 output. It launches one CTA per output column, distributes K
    across CTA threads with vectorized loads, accumulates one fp32 dot product
    per token, and reduces through warp shuffles plus shared memory.

    :param k: Compile-time K dimension specialized into the generated kernel.
    :type k: int
    :param bs: Threads per CTA and K-stripe width used by the reduction.
    :type bs: int
    :param main_vec_width: bf16 elements loaded per thread in the main loop.
    :type main_vec_width: int
    :param tail_vec_width: bf16 elements loaded per thread in the vector tail.
    :type tail_vec_width: int
    :param use_pdl: Whether to launch with Programmatic Dependent Launch.
    :type use_pdl: bool

    :note: Supported A/B data types:
        - BFloat16/BFloat16
    :note: Supported accumulator data types:
        - Float32
    :note: Supported C data types:
        - Float32
    :note: Constraints:
        - K must preserve 16-byte row alignment for contiguous bf16 inputs.

    :note: K is handled as vectorized main/tail loops plus scalar remainder.

    :compile-key: ``(M, K, bs)`` selects the token count, hidden size,
        and CTA thread/K-stripe width specialization.
    """

    def __init__(
        self,
        k: int,
        bs: int = 128,
        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        use_pdl: bool = False,
    ):
        """Initialize the dot-product kernel configuration.

        This configuration fixes the CTA thread count, reduction warp count,
        bf16 vector widths, and K-loop decomposition used by the generated
        kernel.

        :param k: Hidden size K used to specialize the K-loop decomposition.
        :type k: int
        :param bs: Threads per CTA and K-stripe width for the reduction.
        :type bs: int
        :param main_vec_width: BF16 elements loaded per thread in the main loop.
        :type main_vec_width: int
        :param tail_vec_width: BF16 elements loaded per thread in the vector tail.
        :type tail_vec_width: int
        :param use_pdl: Whether to launch with Programmatic Dependent Launch.
        :type use_pdl: bool
        """
        self.bs = bs
        self.main_vec_width = main_vec_width
        self.tail_vec_width = tail_vec_width
        self.use_pdl = use_pdl
        self.num_warps = bs // cute.arch.WARP_SIZE
        self._init_k_tiles(k)

    def _vectorized_elems(self, k_extent: int, vec_width: int) -> int:
        vector_tile = vec_width * self.bs
        return (k_extent // vector_tile) * vector_tile

    def _init_k_tiles(self, k: int) -> None:
        """Split K into vector loops, scalar rounds, and ragged tail."""
        self.k_main_elems = self._vectorized_elems(k, self.main_vec_width)
        self.k_after_main = k - self.k_main_elems
        self.k_tail_elems = self._vectorized_elems(
            self.k_after_main, self.tail_vec_width
        )
        self.k_done_all = self.k_main_elems + self.k_tail_elems
        self.scalar_rem = k - self.k_done_all
        self.ks_full = self.scalar_rem // self.bs
        self.ks_part = self.scalar_rem % self.bs
        self.k_scalar_full = self.ks_full * self.bs
        self.k_part_offset = self.k_done_all + self.k_scalar_full
        self.main_tiles = self.k_main_elems // (self.main_vec_width * self.bs)
        self.tail_tiles = self.k_tail_elems // (self.tail_vec_width * self.bs)

    @cute.jit
    def _vector_dotprod(
        self,
        acc: cute.Tensor,
        tA: cute.Tensor,
        tB: cute.Tensor,
        M: cutlass.Constexpr,
        num_tiles: cutlass.Constexpr,
        align_bytes: cutlass.Constexpr,
    ):
        for tile in cutlass.range_constexpr(num_tiles):
            bt = tB[None, tile]
            br = cute.make_rmem_tensor_like(bt)
            cute.autovec_copy(bt, br)
            br_f32 = br.load().to(cutlass.Float32)

            for m in cutlass.range_constexpr(M):
                at = tA[m, None, tile]
                ar = cute.make_rmem_tensor_like(at)
                cute.autovec_copy(at, ar)
                vec_width: cutlass.Constexpr = cute.size(ar)
                for v in cutlass.range_constexpr(vec_width):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

    def _make_thread_vector_slice(
        self,
        gA_vec: cute.Tensor,
        gB_vec: cute.Tensor,
        tidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        bs: cutlass.Constexpr,
    ):
        # (M/N, K_TILE, K_LANE, K_VEC); tidx selects K_LANE.
        tA = cute.logical_divide(gA_vec, (None, (None, bs)))
        tB = cute.logical_divide(gB_vec, (None, (None, bs)))
        return tA[None, (None, (tidx, None))], tB[n_idx, (None, (tidx, None))]

    def _make_k_slice(
        self,
        gX: cute.Tensor,
        k_offset: cutlass.Constexpr,
        k_extent: cutlass.Constexpr,
    ):
        k_layout_extent: cutlass.Constexpr = (
            1 if const_expr(k_extent == 0) else k_extent
        )

        if const_expr(k_offset == 0):
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
        """Execute the dot-product GEMM operation in steps:
        - Launch one CTA per output column ``n`` with ``bs`` threads.
        - Keep one FP32 accumulator per token ``m`` in each thread.
        - Traverse K with vectorized 128-bit, vectorized 64-bit, scalar, and
          ragged-tail loops from the precomputed K decomposition.
        - Reduce each token accumulator first within the warp, then across
          warps through shared memory, and store ``C[:, n]``.

        :param gA: Input tensor A with shape ``[M, K]``.
        :type gA: cute.Tensor
        :param gB: Input tensor B with shape ``[N, K]``.
        :type gB: cute.Tensor
        :param gC: Output tensor C with shape ``[M, N]``.
        :type gC: cute.Tensor
        :param M: Token count selected by the compile key.
        :type M: cutlass.Constexpr
        :param K_dim: Hidden size selected by the compile key.
        :type K_dim: cutlass.Constexpr
        :param N_dim: Output column count used for the launch grid.
        :type N_dim: cutlass.Int32
        :param stream: CUDA stream for asynchronous execution.
        :type stream: CUstream
        """
        self.kernel(
            gA,
            gB,
            gC,
            M,
            self.main_vec_width,
            self.tail_vec_width,
            self.bs,
            self.num_warps,
            self.k_main_elems,
            self.k_tail_elems,
            self.k_done_all,
            self.ks_full,
            self.ks_part,
            self.k_scalar_full,
            self.k_part_offset,
            self.main_tiles,
            self.tail_tiles,
        ).launch(
            grid=[N_dim, 1, 1],
            block=[self.bs, 1, 1],
            smem=M * 4 * self.num_warps,
            stream=stream,
            use_pdl=self.use_pdl,
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
        tidx, _, _ = cute.arch.thread_idx()
        n_idx, _, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()

        # One FP32 accumulator per token.
        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        acc.fill(0.0)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # 128-bit vectorized main loop
        if const_expr(k_main_elems > 0):
            gA_main = self._make_k_slice(gA, 0, k_main_elems)
            gB_main = self._make_k_slice(gB, 0, k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, main_vec_width))
            tA, tB = self._make_thread_vector_slice(gA_vec, gB_vec, tidx, n_idx, bs)
            self._vector_dotprod(acc, tA, tB, M, main_tiles, 16)

        # 64-bit vectorized tail (K remainder after main loop)
        if const_expr(k_tail_elems > 0):
            gA_tail = self._make_k_slice(gA, k_main_elems, k_tail_elems)
            gB_tail = self._make_k_slice(gB, k_main_elems, k_tail_elems)
            gA_tail_vec = cute.logical_divide(gA_tail, (None, tail_vec_width))
            gB_tail_vec = cute.logical_divide(gB_tail, (None, tail_vec_width))
            tA_t, tB_t = self._make_thread_vector_slice(
                gA_tail_vec, gB_tail_vec, tidx, n_idx, bs
            )
            self._vector_dotprod(acc, tA_t, tB_t, M, tail_tiles, 8)

        # Full scalar rounds use CuTe width-1 tiles; KS_PART is the ragged tail.
        if const_expr(ks_full > 0):
            gA_scalar = self._make_k_slice(gA, k_done_all, k_scalar_full)
            gB_scalar = self._make_k_slice(gB, k_done_all, k_scalar_full)
            gA_scalar_vec = cute.logical_divide(gA_scalar, (None, 1))
            gB_scalar_vec = cute.logical_divide(gB_scalar, (None, 1))
            tA_s, tB_s = self._make_thread_vector_slice(
                gA_scalar_vec, gB_scalar_vec, tidx, n_idx, bs
            )
            self._vector_dotprod(acc, tA_s, tB_s, M, ks_full, 2)

        # Only threads below KS_PART load the ragged tail.
        if const_expr(ks_part > 0):
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
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def make_host_bf16(k_val: int, bs: int = 128):
    return LLBf16Dotprod(k=k_val, bs=bs)
