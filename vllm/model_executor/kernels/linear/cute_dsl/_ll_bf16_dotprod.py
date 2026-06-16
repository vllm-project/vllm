# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuteDSL A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

Dot-product kernel for low-latency problems using FMA instructions.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream


def _vector_dotprod(
    acc: cute.Tensor,
    tA_vec: cute.Tensor,  # layout: (M, vector_width, num_tiles)
    tB_vec: cute.Tensor,  # layout: (vector_width, num_tiles)
):
    for tile in range(cute.size(tA_vec, mode=[2])):
        bt = tB_vec[None, tile]
        br = cute.make_rmem_tensor_like(bt)
        cute.autovec_copy(bt, br)
        br_f32 = br.load().to(cutlass.Float32)

        for m in range(cute.size(tA_vec, mode=[0])):
            at = tA_vec[m, None, tile]
            ar = cute.make_rmem_tensor_like(at)
            cute.autovec_copy(at, ar)
            vec_width: cutlass.Constexpr = cute.size(ar)
            for v in range(vec_width):
                acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]


def _make_thread_vector_slice(
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
    gX: cute.Tensor,
    k_offset: cutlass.Constexpr,
    k_extent: cutlass.Constexpr,
):
    # CuTe layouts require positive dimensions even when a compile-time branch
    # is eliminated. Empty K regions use a one-element layout and zero tiles.
    k_layout_extent: cutlass.Constexpr = max(k_extent, 1)
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
    _BS = bs  # block size

    @cute.kernel
    def dotprod_bf16_lf(
        gA: cute.Tensor,  # activations, [M, K]
        gB: cute.Tensor,  # weights, [N, K]
        gC: cute.Tensor,  # output, [M, N]
        M: cutlass.Constexpr,  # number of tokens (compile-time constant)
    ):
        cute.arch.setmaxregister_increase(128)

        tidx = cute.arch.thread_idx()[0]  # [0, 255]
        n_idx = cute.arch.block_idx()[0]  # which expert this CTA computes. [0, N-1]

        # Compile-time constants
        MAIN_VEC_WIDTH: cutlass.Constexpr = _MAIN_VEC_WIDTH
        BS: cutlass.Constexpr = _BS
        K_TOTAL: cutlass.Constexpr = cute.size(gA, mode=[1])
        K_MAIN_ELEMS: cutlass.Constexpr = (
            (K_TOTAL // (MAIN_VEC_WIDTH * BS)) * MAIN_VEC_WIDTH * BS
        )

        # One FP32 accumulator per token, in registers
        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        acc.fill(0.0)

        cute.arch.griddepcontrol_wait()  # PDL wait

        if K_MAIN_ELEMS > 0:
            gA_main = _make_k_slice(gA, 0, K_MAIN_ELEMS)
            gB_main = _make_k_slice(gB, 0, K_MAIN_ELEMS)
            gA_vec = cute.logical_divide(gA_main, (None, MAIN_VEC_WIDTH))
            gB_vec = cute.logical_divide(gB_main, (None, MAIN_VEC_WIDTH))
            tA_vec, tB_vec = _make_thread_vector_slice(gA_vec, gB_vec, tidx, n_idx, BS)
            _vector_dotprod(acc, tA_vec, tB_vec)

        TAIL_VEC_WIDTH: cutlass.Constexpr = _TAIL_VEC_WIDTH
        K_AFTER_MAIN: cutlass.Constexpr = K_TOTAL - K_MAIN_ELEMS
        K_TAIL_ELEMS: cutlass.Constexpr = (
            (K_AFTER_MAIN // (TAIL_VEC_WIDTH * BS)) * TAIL_VEC_WIDTH * BS
        )
        if K_TAIL_ELEMS > 0:
            gA_tail = _make_k_slice(gA, K_MAIN_ELEMS, K_TAIL_ELEMS)
            gB_tail = _make_k_slice(gB, K_MAIN_ELEMS, K_TAIL_ELEMS)
            gA_tail_vec = cute.logical_divide(gA_tail, (None, TAIL_VEC_WIDTH))
            gB_tail_vec = cute.logical_divide(gB_tail, (None, TAIL_VEC_WIDTH))
            tA_tail_vec, tB_tail_vec = _make_thread_vector_slice(
                gA_tail_vec, gB_tail_vec, tidx, n_idx, BS
            )
            _vector_dotprod(acc, tA_tail_vec, tB_tail_vec)

        # Scalar tail (one element per thread for non-aligned K)
        K_DONE_ALL: cutlass.Constexpr = K_MAIN_ELEMS + K_TAIL_ELEMS
        SCALAR_REM: cutlass.Constexpr = K_TOTAL - K_DONE_ALL
        KS_FULL: cutlass.Constexpr = SCALAR_REM // BS
        for si in cutlass.range_constexpr(KS_FULL):
            ko = K_DONE_ALL + si * BS + tidx
            bv = gB[n_idx, ko].to(cutlass.Float32)
            for m in cutlass.range_constexpr(M):
                acc[m] = acc[m] + gA[m, ko].to(cutlass.Float32) * bv
        # Compile-time guard: no partial remainder eliminates this block.
        # Runtime guard: only the first KS_PART threads participate.
        # Remaining threads do not execute any out-of-bounds loads.
        KS_PART: cutlass.Constexpr = SCALAR_REM % BS
        if KS_PART > 0:
            ko_p = K_DONE_ALL + KS_FULL * BS + tidx
            if tidx < KS_PART:
                bv2 = gB[n_idx, ko_p].to(cutlass.Float32)
                for m in cutlass.range_constexpr(M):
                    acc[m] = acc[m] + gA[m, ko_p].to(cutlass.Float32) * bv2

        # Reduction

        # Intra-warp shuffle reduction
        WS: cutlass.Constexpr = 32
        NW: cutlass.Constexpr = BS // WS  # = 8 warps
        for m in cutlass.range_constexpr(M):
            acc[m] = cute.arch.warp_reduction_sum(acc[m])

        # Cross-warp reduction via shared memory
        wid = tidx // WS
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
                    cute.ReductionOp.ADD,
                    init_val=0.0,
                    reduction_profile=0,
                )
        cute.arch.griddepcontrol_launch_dependents()  # PDL signal

    @cute.jit
    def host_bf16_lf(
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        M: cutlass.Constexpr,
        K_dim: cutlass.Constexpr,
        N_dim: cutlass.Int32,
        stream: CUstream,
    ):
        mA = cute.make_tensor(
            gA.iterator,
            cute.make_layout((M, K_dim), stride=(K_dim, 1)),
        )
        mB = cute.make_tensor(
            gB.iterator,
            cute.make_layout((N_dim, K_dim), stride=(K_dim, 1)),
        )
        mC = cute.make_tensor(
            gC.iterator,
            cute.make_layout((M, N_dim), stride=(N_dim, 1)),
        )
        dotprod_bf16_lf(mA, mB, mC, M).launch(
            grid=[N_dim, 1, 1],  # one CTA per expert
            block=[bs, 1, 1],
            smem=M * 4 * (bs // 32),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    return host_bf16_lf
