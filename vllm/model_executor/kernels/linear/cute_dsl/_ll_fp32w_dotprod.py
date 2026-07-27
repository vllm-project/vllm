# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


class LLFp32WDotprod:
    """FP32-weight router GEMM kernel based on CTA-local dot products.

    This kernel computes C[M, N] = A[M, K] @ B[N, K]^T for bf16/fp16/fp32
    activations, fp32 weights, and fp32 output. It launches one CTA per
    small output-column tile, distributes K across CTA threads with vectorized
    loads, accumulates one fp32 dot product per token/expert, and reduces
    through warp shuffles plus shared memory.

    :param k: Compile-time K dimension specialized into the generated kernel.
    :type k: int
    :param bs: Threads per CTA and K-stripe width used by the reduction.
    :type bs: int
    :param main_vec_width: Elements loaded per thread in the main loop.
    :type main_vec_width: int
    :param tail_vec_width: Elements loaded per thread in the vector tail.
    :type tail_vec_width: int
    :param use_pdl: Whether to launch with Programmatic Dependent Launch.
    :type use_pdl: bool

    :note: Supported A/B data types:
        - BFloat16/Float32
        - Float16/Float32
        - Float32/Float32
    :note: Supported accumulator data types:
        - Float32
    :note: Supported C data types:
        - Float32
    :note: Constraints:
        - Inputs must be contiguous row-major 2D tensors.

    :note: K is handled as vectorized main/tail loops plus scalar remainder.

    :compile-key: ``(M, K, bs, epb, act_dtype)`` selects the token count,
        hidden size, CTA thread/K-stripe width, experts per CTA, and activation
        input type.
    """

    def __init__(
        self,
        m: int,
        k: int,
        bs: int = 128,
        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        token_groups: int = 1,
        epb: int = 1,
        use_pdl: bool = False,
    ):
        """Initialize the dot-product kernel configuration.

        This configuration fixes the CTA thread count, reduction warp count,
        vector widths, and K-loop decomposition used by the generated kernel.

        :param m: Token count M selected by the compile key.
        :type m: int
        :param k: Hidden size K used to specialize the K-loop decomposition.
        :type k: int
        :param bs: Threads per CTA and K-stripe width for the reduction.
        :type bs: int
        :param main_vec_width: Elements loaded per thread in the main loop.
        :type main_vec_width: int
        :param tail_vec_width: Elements loaded per thread in the vector tail.
        :type tail_vec_width: int
        :param use_pdl: Whether to launch with Programmatic Dependent Launch.
        :type use_pdl: bool
        """
        self.m = m
        self.m_per_group = m // token_groups
        self.bs = bs
        self.main_vec_width = main_vec_width
        self.tail_vec_width = tail_vec_width
        self.token_groups = token_groups
        self.epb = epb
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
        EPB: cutlass.Constexpr,
        num_tiles: cutlass.Constexpr,
    ):
        for tile in cutlass.range_constexpr(num_tiles):
            bt0 = tB[0, None, tile]
            br0 = cute.make_rmem_tensor_like(bt0)
            cute.autovec_copy(bt0, br0)
            br0_f32 = br0.load().to(cutlass.Float32)

            if const_expr(EPB == 2):
                bt1 = tB[1, None, tile]
                br1 = cute.make_rmem_tensor_like(bt1)
                cute.autovec_copy(bt1, br1)
                br1_f32 = br1.load().to(cutlass.Float32)

            for m in cutlass.range_constexpr(M):
                at = tA[m, None, tile]
                ar = cute.make_rmem_tensor_like(at)
                cute.autovec_copy(at, ar)
                vec_width: cutlass.Constexpr = cute.size(ar)
                for v in cutlass.range_constexpr(vec_width):
                    av = ar[v].to(cutlass.Float32)
                    acc[m, 0] = acc[m, 0] + av * br0_f32[v]
                    if const_expr(EPB == 2):
                        acc[m, 1] = acc[m, 1] + av * br1_f32[v]

    def _make_thread_vector_slice(
        self,
        gA_vec: cute.Tensor,
        gB_vec: cute.Tensor,
        tidx: cutlass.Int32,
        bs: cutlass.Constexpr,
    ):
        # A: (M, K_TILE, K_LANE, K_VEC); B: (EPB, K_TILE, K_LANE, K_VEC).
        tA = cute.logical_divide(gA_vec, (None, (None, bs)))
        tB = cute.logical_divide(gB_vec, (None, (None, bs)))
        return tA[None, (None, (tidx, None))], tB[None, (None, (tidx, None))]

    def _make_m_slice(
        self,
        gX: cute.Tensor,
        m_offset: cutlass.Int32,
        m_extent: cutlass.Constexpr,
    ):
        return cute.local_tile(
            cute.domain_offset((m_offset, 0), gX),
            (m_extent, cute.size(gX, mode=[1])),
            (0, 0),
        )

    def _make_n_tile(
        self,
        gX: cute.Tensor,
        n_tile_idx: cutlass.Int32,
        n_extent: cutlass.Constexpr,
    ):
        return cute.local_tile(
            gX,
            (n_extent, cute.size(gX, mode=[1])),
            (n_tile_idx, 0),
        )

    def _make_sm_m_slice(
        self,
        sm: cute.Tensor,
        m_offset: cutlass.Int32,
        m_extent: cutlass.Constexpr,
        epb: cutlass.Constexpr,
        num_warps: cutlass.Constexpr,
    ):
        return cute.local_tile(
            cute.domain_offset((m_offset, 0, 0), sm),
            (m_extent, epb, num_warps),
            (0, 0, 0),
        )

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
    def _compute_token_group(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        sm: cute.Tensor,
        n_tile_idx: cutlass.Int32,
        tidx: cutlass.Int32,
        local_wid: cutlass.Int32,
        M_PER_GROUP: cutlass.Constexpr,
        M_OFFSET: cutlass.Int32,
        main_vec_width: cutlass.Constexpr,
        tail_vec_width: cutlass.Constexpr,
        bs: cutlass.Constexpr,
        epb: cutlass.Constexpr,
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
        # Each CTA reuses A for epb adjacent experts.
        acc = cute.make_rmem_tensor((M_PER_GROUP, epb), cutlass.Float32)
        acc.fill(0.0)

        gA_group = self._make_m_slice(gA, M_OFFSET, M_PER_GROUP)
        gB_epb = self._make_n_tile(gB, n_tile_idx, epb)
        sm_group = self._make_sm_m_slice(sm, M_OFFSET, M_PER_GROUP, epb, num_warps)

        if const_expr(k_main_elems > 0):
            gA_main = self._make_k_slice(gA_group, 0, k_main_elems)
            gB_main = self._make_k_slice(gB_epb, 0, k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, main_vec_width))
            tA, tB = self._make_thread_vector_slice(gA_vec, gB_vec, tidx, bs)
            self._vector_dotprod(acc, tA, tB, M_PER_GROUP, epb, main_tiles)

        if const_expr(k_tail_elems > 0):
            gA_tail = self._make_k_slice(gA_group, k_main_elems, k_tail_elems)
            gB_tail = self._make_k_slice(gB_epb, k_main_elems, k_tail_elems)
            gA_tail_vec = cute.logical_divide(gA_tail, (None, tail_vec_width))
            gB_tail_vec = cute.logical_divide(gB_tail, (None, tail_vec_width))
            tA_t, tB_t = self._make_thread_vector_slice(
                gA_tail_vec, gB_tail_vec, tidx, bs
            )
            self._vector_dotprod(acc, tA_t, tB_t, M_PER_GROUP, epb, tail_tiles)

        if const_expr(ks_full > 0):
            gA_scalar = self._make_k_slice(gA_group, k_done_all, k_scalar_full)
            gB_scalar = self._make_k_slice(gB_epb, k_done_all, k_scalar_full)
            gA_scalar_vec = cute.logical_divide(gA_scalar, (None, 1))
            gB_scalar_vec = cute.logical_divide(gB_scalar, (None, 1))
            tA_s, tB_s = self._make_thread_vector_slice(
                gA_scalar_vec, gB_scalar_vec, tidx, bs
            )
            self._vector_dotprod(acc, tA_s, tB_s, M_PER_GROUP, epb, ks_full)

        if const_expr(ks_part > 0):
            gA_part = self._make_k_slice(gA_group, k_part_offset, ks_part)
            gB_part = self._make_k_slice(gB_epb, k_part_offset, ks_part)
            if tidx < ks_part:
                for local_m in cutlass.range_constexpr(M_PER_GROUP):
                    av = gA_part[local_m, tidx].to(cutlass.Float32)
                    for e in cutlass.range_constexpr(epb):
                        bv = gB_part[e, tidx].to(cutlass.Float32)
                        acc[local_m, e] = acc[local_m, e] + av * bv

        for local_m in cutlass.range_constexpr(M_PER_GROUP):
            for e in cutlass.range_constexpr(epb):
                acc[local_m, e] = cute.arch.warp_reduction_sum(acc[local_m, e])

        with cute.arch.elect_one():
            for local_m in cutlass.range_constexpr(M_PER_GROUP):
                for e in cutlass.range_constexpr(epb):
                    sm_group[local_m, e, local_wid] = acc[local_m, e]

    @cute.jit
    def _store_outputs(
        self,
        gC: cute.Tensor,
        sm: cute.Tensor,
        n_tile_idx: cutlass.Int32,
        N_dim: cutlass.Int32,
        tidx: cutlass.Int32,
        group_idx: cutlass.Int32,
        M: cutlass.Constexpr,
        EPB: cutlass.Constexpr,
    ):
        n_tile_layout = cute.make_layout(
            (cute.ceil_div(N_dim, EPB), EPB), stride=(EPB, 1)
        )
        n_coords = cute.make_tensor(0, n_tile_layout)

        if group_idx == 0 and tidx < M:
            for e in cutlass.range_constexpr(EPB):
                n = n_coords[n_tile_idx, e]
                if n < N_dim:
                    partials = sm[tidx, e, None].load()
                    gC[tidx, n] = partials.reduce(
                        cute.ReductionOp.ADD,
                        init_val=cutlass.Float32(0.0),
                        reduction_profile=0,
                    )

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        N_dim: cutlass.Int32,
        stream: CUstream,
    ):
        """Execute the dot-product GEMM operation in steps:
        - Launch one CTA per small output-column tile with ``bs`` threads.
        - Keep one FP32 accumulator per token/expert in each thread.
        - Traverse K with vectorized main/tail, scalar, and ragged-tail loops
          from the precomputed K decomposition.
        - Reduce each token accumulator first within the warp, then across
          warps through shared memory, and store ``C[:, n]``.

        :param gA: Input tensor A with shape ``[M, K]``.
        :type gA: cute.Tensor
        :param gB: Input tensor B with shape ``[N, K]``.
        :type gB: cute.Tensor
        :param gC: Output tensor C with shape ``[M, N]``.
        :type gC: cute.Tensor
        :param N_dim: Output column count used for the launch grid.
        :type N_dim: cutlass.Int32
        :param stream: CUDA stream for asynchronous execution.
        :type stream: CUstream
        """
        self.kernel(
            gA,
            gB,
            gC,
            N_dim,
            self.m,
            self.m_per_group,
            self.main_vec_width,
            self.tail_vec_width,
            self.bs,
            self.num_warps,
            self.token_groups,
            self.epb,
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
            grid=[cute.ceil_div(N_dim, self.epb), 1, 1],
            block=[self.bs, self.token_groups, 1],
            smem=self.m * self.epb * 4 * self.num_warps,
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
        N_dim: cutlass.Int32,
        M: cutlass.Constexpr,
        m_per_group: cutlass.Constexpr,
        main_vec_width: cutlass.Constexpr,
        tail_vec_width: cutlass.Constexpr,
        bs: cutlass.Constexpr,
        num_warps: cutlass.Constexpr,
        token_groups: cutlass.Constexpr,
        epb: cutlass.Constexpr,
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
        tidx, group_idx, _ = cute.arch.thread_idx()
        n_tile_idx, _, _ = cute.arch.block_idx()

        local_wid = tidx // cute.arch.WARP_SIZE

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        smem_red_layout = cute.make_layout(
            (M, epb, num_warps), stride=(epb * num_warps, num_warps, 1)
        )
        smem = cutlass.utils.SmemAllocator()
        sm = smem.allocate_tensor(cutlass.Float32, smem_red_layout, byte_alignment=16)

        if const_expr(token_groups == 1):
            m_offset: cutlass.Constexpr = 0
        else:
            m_offset = group_idx * m_per_group
        self._compute_token_group(
            gA,
            gB,
            sm,
            n_tile_idx,
            tidx,
            local_wid,
            m_per_group,
            m_offset,
            main_vec_width,
            tail_vec_width,
            bs,
            epb,
            num_warps,
            k_main_elems,
            k_tail_elems,
            k_done_all,
            ks_full,
            ks_part,
            k_scalar_full,
            k_part_offset,
            main_tiles,
            tail_tiles,
        )

        cute.arch.sync_threads()
        self._store_outputs(gC, sm, n_tile_idx, N_dim, tidx, group_idx, M, epb)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def make_host_fp32w(
    k_val: int, bs: int = 128, token_groups: int = 1, epb: int = 1, m: int = 1
):
    return LLFp32WDotprod(m=m, k=k_val, bs=bs, token_groups=token_groups, epb=epb)
