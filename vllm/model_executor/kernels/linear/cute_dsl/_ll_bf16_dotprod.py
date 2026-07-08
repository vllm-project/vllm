# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuteDSL A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

Dot-product kernel for low-latency problems using FMA instructions.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream


def _vector_dotprod(
    acc: cute.Tensor,              # [M] fp32 accumulators in registers
    tA: cute.Tensor,               # this thread's A view: [M, vec_width, num_tiles]
    tB: cute.Tensor,               # this thread's B view: [vec_width, num_tiles]
    M: cutlass.Constexpr,          # must be constexpr for loop unrolling
    num_tiles: cutlass.Constexpr,  # tiles this thread processes
    align_bytes: cutlass.Constexpr,  # 16 for 128-bit loads, 8 for 64-bit tail
):
    """FMA dot-product over all K-tiles.

    B is loaded once per tile and converted to fp32, then reused across all M
    tokens to avoid redundant type conversion.
    Alignment hints are re-applied after logical_divide to restore vectorized loads.
    """
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
    gA_vec: cute.Tensor,  # [M, (vec_width, num_vecs)]
    gB_vec: cute.Tensor,  # [N, (vec_width, num_vecs)]
    tidx: cutlass.Int32,
    n_idx: cutlass.Int32,
    bs: cutlass.Constexpr,
):
    """Partition vectorized K-tiles across BS threads, return this thread's view."""
    tA = cute.logical_divide(gA_vec, (None, (None, bs)))
    tB = cute.logical_divide(gB_vec, (None, (None, bs)))
    return tA[None, (None, (tidx, None))], tB[n_idx, (None, (tidx, None))]


def _make_k_slice(
    gX: cute.Tensor,
    k_offset: cutlass.Constexpr,
    k_extent: cutlass.Constexpr,
):
    """Extract a K-dimension slice [*leading_dims, k_offset:k_offset+k_extent]."""
    # CuTe layouts require positive dimensions even when a compile-time branch
    # is eliminated. Empty K regions use a one-element layout and zero tiles.
    k_layout_extent: cutlass.Constexpr = max(k_extent, 1)
    if k_offset == 0:
        return cute.local_tile(gX, (cute.size(gX, mode=[0]), k_layout_extent), (0, 0))
    return cute.local_tile(
        cute.domain_offset((0, k_offset), gX),
        (cute.size(gX, mode=[0]), k_layout_extent),
        (0, 0),
    )


# Returns a compiled host function specialized for a specific K.
# K-derived constants are computed from the CuTe layout so loop bounds remain
# compile-time constants.
# TODO (roberto): add 256-bit instructions support.
def make_host_bf16(k_val: int, bs: int = 128):
    _MAIN_VEC_WIDTH = 8  # bf16 elements per 128-bit vectorized load
    _TAIL_VEC_WIDTH = 4  # bf16 elements per 64-bit vectorized load
    _BS = bs


    @cute.kernel
    def dotprod_bf16_lf(
        gA: cute.Tensor,          # activations [M, K], row-major bf16
        gB: cute.Tensor,          # weights     [N, K], row-major bf16
        gC: cute.Tensor,          # output      [M, N], row-major fp32
        M: cutlass.Constexpr,     # number of tokens
        K_dim: cutlass.Constexpr, # hidden dimension (K)
    ):
        cute.arch.setmaxregister_increase(128)

        tidx  = cute.arch.thread_idx()[0]   # [0, BS-1]
        n_idx = cute.arch.block_idx()[0]    # which expert [0, N-1]

        # Compile-time constants
        MAIN_VEC_WIDTH: cutlass.Constexpr = _MAIN_VEC_WIDTH
        TAIL_VEC_WIDTH: cutlass.Constexpr = _TAIL_VEC_WIDTH
        BS: cutlass.Constexpr = _BS
        K_TOTAL: cutlass.Constexpr = K_dim
        K_MAIN_ELEMS: cutlass.Constexpr = (
            (K_TOTAL // (MAIN_VEC_WIDTH * BS)) * MAIN_VEC_WIDTH * BS
        )
        K_AFTER_MAIN: cutlass.Constexpr = K_TOTAL - K_MAIN_ELEMS
        K_TAIL_ELEMS: cutlass.Constexpr = (
            (K_AFTER_MAIN // (TAIL_VEC_WIDTH * BS)) * TAIL_VEC_WIDTH * BS
        )
        K_DONE_ALL: cutlass.Constexpr = K_MAIN_ELEMS + K_TAIL_ELEMS
        SCALAR_REM: cutlass.Constexpr = K_TOTAL - K_DONE_ALL
        KS_FULL: cutlass.Constexpr = SCALAR_REM // BS
        KS_PART: cutlass.Constexpr = SCALAR_REM % BS
        K_SCALAR_FULL: cutlass.Constexpr = KS_FULL * BS
        K_PART_OFFSET: cutlass.Constexpr = K_DONE_ALL + K_SCALAR_FULL
        MAIN_TILES: cutlass.Constexpr = K_MAIN_ELEMS // (MAIN_VEC_WIDTH * BS)
        TAIL_TILES: cutlass.Constexpr = K_TAIL_ELEMS // (TAIL_VEC_WIDTH * BS)

        # One FP32 accumulator per token, in registers
        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        acc.fill(0.0)

        #cute.arch.griddepcontrol_wait()  # PDL wait

        # 128-bit vectorized main loop
        if K_MAIN_ELEMS > 0:
            gA_main = _make_k_slice(gA, 0, K_MAIN_ELEMS)
            gB_main = _make_k_slice(gB, 0, K_MAIN_ELEMS)
            gA_vec = cute.logical_divide(gA_main, (None, MAIN_VEC_WIDTH))
            gB_vec = cute.logical_divide(gB_main, (None, MAIN_VEC_WIDTH))
            tA, tB = _make_thread_vector_slice(gA_vec, gB_vec, tidx, n_idx, BS)
            _vector_dotprod(acc, tA, tB, M, MAIN_TILES, 16)

        # 64-bit vectorized tail (K remainder after main loop)
        if K_TAIL_ELEMS > 0:
            gA_tail = _make_k_slice(gA, K_MAIN_ELEMS, K_TAIL_ELEMS)
            gB_tail = _make_k_slice(gB, K_MAIN_ELEMS, K_TAIL_ELEMS)
            gA_tail_vec = cute.logical_divide(gA_tail, (None, TAIL_VEC_WIDTH))
            gB_tail_vec = cute.logical_divide(gB_tail, (None, TAIL_VEC_WIDTH))
            tA_t, tB_t = _make_thread_vector_slice(
                gA_tail_vec, gB_tail_vec, tidx, n_idx, BS
            )
            _vector_dotprod(acc, tA_t, tB_t, M, TAIL_TILES, 8)

        # Scalar remainder after vectorized loops. Full BS-wide rounds are
        # expressed as width-1 CuTe partitions; only the final ragged tile needs
        # a runtime guard for threads beyond KS_PART.
        if KS_FULL > 0:
            gA_scalar = _make_k_slice(gA, K_DONE_ALL, K_SCALAR_FULL)
            gB_scalar = _make_k_slice(gB, K_DONE_ALL, K_SCALAR_FULL)
            gA_scalar_vec = cute.logical_divide(gA_scalar, (None, 1))
            gB_scalar_vec = cute.logical_divide(gB_scalar, (None, 1))
            tA_s, tB_s = _make_thread_vector_slice(
                gA_scalar_vec, gB_scalar_vec, tidx, n_idx, BS
            )
            _vector_dotprod(acc, tA_s, tB_s, M, KS_FULL, 2)

        # Compile-time guard: no partial remainder eliminates this block.
        # Runtime guard: only the first KS_PART threads participate.
        # Remaining threads do not execute any out-of-bounds loads.
        if KS_PART > 0:
            gA_part = _make_k_slice(gA, K_PART_OFFSET, KS_PART)
            gB_part = _make_k_slice(gB, K_PART_OFFSET, KS_PART)
            if tidx < KS_PART:
                bv2 = gB_part[n_idx, tidx].to(cutlass.Float32)
                for m in cutlass.range_constexpr(M):
                    acc[m] = acc[m] + gA_part[m, tidx].to(cutlass.Float32) * bv2

        # Intra-warp shuffle reduction
        WS: cutlass.Constexpr = 32
        NW: cutlass.Constexpr = BS // WS # = 8 warps
        for m in cutlass.range_constexpr(M):
            acc[m] = cute.arch.warp_reduction_sum(acc[m])

        # Cross-warp reduction via shared memory
        wid = cute.arch.warp_idx()
        smem_red_layout = cute.make_layout((M, NW), stride=(NW, 1))
        sp = cute.arch.alloc_smem(
            cutlass.Float32, cute.cosize(smem_red_layout), alignment=16
        )
        sm = cute.make_tensor(sp, smem_red_layout)
        with cute.arch.elect_one():
            for m in cutlass.range_constexpr(M):
                sm[m, wid] = acc[m]

        # Final reduction and output
        cute.arch.sync_threads()
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                partials = sm[m, None].load()
                gC[m, n_idx] = partials.reduce(
                    cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0
                )
        #cute.arch.griddepcontrol_launch_dependents()  # PDL signal

    @cute.jit
    def host_bf16_lf(
        gA: cute.Tensor,          # [M, K] bf16 row-major
        gB: cute.Tensor,          # [N, K] bf16 row-major
        gC: cute.Tensor,          # [M, N] fp32 row-major
        M: cutlass.Constexpr,
        K_dim: cutlass.Constexpr,
        N_dim: cutlass.Int32,
        stream: CUstream,
    ):
        dotprod_bf16_lf(gA, gB, gC, M, K_dim).launch(
            grid=[N_dim, 1, 1], # one CTA per expert
            block=[bs, 1, 1],
            smem=M * 4 * (bs // 32),
            stream=stream,
            use_pdl=False,
            min_blocks_per_mp=1,
        )

    return host_bf16_lf
