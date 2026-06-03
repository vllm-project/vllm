# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cuteDSL A GEMM: C[M,N] = A[M,K] @ B[N,K]^T.

Dot-product kernel for low-latency problems using FMA instructions.
"""

import operator

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream


# Returns a compiled host function specialized for a specific K.
# The closure captures all K-derived constants, so the kernel has zero runtime
# loop overhead
# TODO (roberto): add 256-bit instructions support.
def make_host_bf16(k_val: int, bs: int = 128):
    # Main loop constants
    _VPT = 8  # vectors per thread: 8 × bf16 (2 bytes) = 16 bytes = 128-bit load
    _BS = bs  # block size: 256 threads
    _KPI = _VPT * _BS  # = 2048 K elements processed per main-loop iteration
    _k_main = k_val // _KPI  # main loop iters

    # Tail loop constants
    _VPT_T = 4  # 4 × bf16 = 8 bytes = 64-bit load
    _KPT = _VPT_T * _BS  # = 1024 K elements per tail iteration
    _k_tail = (k_val - _k_main * _KPI) // _KPT  # Number of 64-bit tail iterations

    # Scalar remainder
    _k_done = _k_main * _KPI + _k_tail * _KPT  # elements handled by vectorized loops
    _scalar_rem = k_val - _k_done  # leftover
    _ks_full = _scalar_rem // _BS  # full scalar iterations (256 threads, 1 elem each)
    _ks_part = _scalar_rem % _BS  # partial: fewer than 256 active threads

    # ^ Every one of these becomes a Constexpr inside the kernel,
    # so the compiler statically removes dead branches.

    @cute.kernel
    def dotprod_bf16_lf(
        gA: cute.Tensor,  # activations, flat [M*K]
        gB: cute.Tensor,  # weights, flat [N*K]
        gC: cute.Tensor,  # output, flat [M*N]
        M: cutlass.Constexpr,  # number of tokens (compile-time constant)
        K_dim: cutlass.Constexpr,  # hidden dimension (compile-time constant)
        N_dim: cutlass.Int32,  # number of experts (runtime value)
    ):
        cute.arch.setmaxregister_increase(128)

        tidx = cute.arch.thread_idx()[0]  # [0, 255]
        n_idx = cute.arch.block_idx()[0]  # which expert this CTA computes. [0, N-1]

        # Compile-time constants
        VPT: cutlass.Constexpr = _VPT
        BS: cutlass.Constexpr = _BS
        KPI: cutlass.Constexpr = _KPI
        K_MAIN: cutlass.Constexpr = _k_main

        elem = gB.element_type
        b_base = (
            gB.iterator + n_idx * K_dim
        )  # Pointer to the start of this expert's weight row in the flat B tensor
        tid_off = (
            tidx * VPT
        )  # This thread's starting offset within a K-chunk (256 threads, 8 bf16 elements each)

        # One FP32 accumulator per token, in registers
        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        for m in cutlass.range_constexpr(M):  # fully unrolls
            acc[m] = cutlass.Float32(0.0)

        cute.arch.griddepcontrol_wait()  # PDL wait

        # Main K-loop (fully unrolled via range_constexpr)
        for ki in cutlass.range_constexpr(K_MAIN):
            # Load B tile - B is loaded once from GMEM per K-iteration, then reused across all M tokens
            kb = ki * KPI + tid_off
            bp = (b_base + kb).align(16)  # pointer with 16-byte alignment guarantees
            bt = cute.make_tensor(
                bp, cute.make_layout((VPT,))
            )  # creates a 1D tensor view of 8 elements
            br = cute.make_rmem_tensor(
                (VPT,), elem
            )  # allocates 8 registers for the loaded data
            cute.autovec_copy(bt, br)  # GMEM -> RF
            # Convert B to fp32 once, reuse across all M tokens
            br_f32 = cute.make_rmem_tensor((VPT,), cutlass.Float32)
            for v in cutlass.range_constexpr(VPT):
                br_f32[v] = br[v].to(cutlass.Float32)
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + kb)).align(16)
                at = cute.make_tensor(ap, cute.make_layout((VPT,)))
                ar = cute.make_rmem_tensor((VPT,), elem)
                cute.autovec_copy(at, ar)
                for v in cutlass.range_constexpr(VPT):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

        VPT_T: cutlass.Constexpr = _VPT_T
        KPT: cutlass.Constexpr = _KPT
        K_DONE: cutlass.Constexpr = _k_main * _KPI
        tid_off_t = tidx * VPT_T
        # Vectorized tail (64-bit loads for K remainder) - same structure as main loop
        for ti in cutlass.range_constexpr(_k_tail):
            kb = K_DONE + ti * KPT + tid_off_t
            bp = (b_base + kb).align(8)
            bt = cute.make_tensor(bp, cute.make_layout((VPT_T,)))
            br = cute.make_rmem_tensor((VPT_T,), elem)
            cute.autovec_copy(bt, br)
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + kb)).align(8)
                at = cute.make_tensor(ap, cute.make_layout((VPT_T,)))
                ar = cute.make_rmem_tensor((VPT_T,), elem)
                cute.autovec_copy(at, ar)
                for v in cutlass.range_constexpr(VPT_T):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br[v].to(
                        cutlass.Float32
                    )

        # Scalar tail (one element per thread for non-aligned K)
        K_DONE_ALL: cutlass.Constexpr = _k_done
        for si in cutlass.range_constexpr(_ks_full):
            ko = K_DONE_ALL + si * BS + tidx
            bp = (b_base + ko).align(2)
            bt = cute.make_tensor(bp, cute.make_layout((1,)))
            br = cute.make_rmem_tensor((1,), elem)
            cute.autovec_copy(bt, br)
            bv = br[0].to(cutlass.Float32)
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + ko)).align(2)
                at = cute.make_tensor(ap, cute.make_layout((1,)))
                ar = cute.make_rmem_tensor((1,), elem)
                cute.autovec_copy(at, ar)
                acc[m] = acc[m] + ar[0].to(cutlass.Float32) * bv
        # Compile-time guard (_ks_part > 0) — if there's no partial remainder, this entire
        # block is eliminated from the binary.
        # Runtime guard (tidx < KS_PART) — only the first _ks_part threads participate.
        # Remaining threads are idle but don't execute any loads (no out-of-bounds access).
        if _ks_part > 0:
            KS_PART: cutlass.Constexpr = _ks_part
            ko_p = K_DONE_ALL + _ks_full * BS + tidx
            if tidx < KS_PART:
                bp2 = (b_base + ko_p).align(2)
                bt2 = cute.make_tensor(bp2, cute.make_layout((1,)))
                br2 = cute.make_rmem_tensor((1,), elem)
                cute.autovec_copy(bt2, br2)
                bv2 = br2[0].to(cutlass.Float32)
                for m in cutlass.range_constexpr(M):
                    ap2 = (gA.iterator + (m * K_dim + ko_p)).align(2)
                    at2 = cute.make_tensor(ap2, cute.make_layout((1,)))
                    ar2 = cute.make_rmem_tensor((1,), elem)
                    cute.autovec_copy(at2, ar2)
                    acc[m] = acc[m] + ar2[0].to(cutlass.Float32) * bv2

        # Reduction

        # Intra-warp shuffle reduction
        WS: cutlass.Constexpr = 32
        NW: cutlass.Constexpr = BS // WS  # = 8 warps
        for m in cutlass.range_constexpr(M):
            acc[m] = cute.arch.warp_reduction(acc[m], operator.add)
            # TODO (roberto): uses __shfl_xor_sync -> try redux.sync.add instead

        # Cross-warp reduction via shared memory
        wid = tidx // WS
        lid = tidx % WS
        sp = cute.arch.alloc_smem(cutlass.Float32, M * NW, alignment=16)
        sm = cute.make_tensor(
            sp, cute.make_layout((M, NW))
        )  # token partials are contiguous in M-dim
        for m in cutlass.range_constexpr(M):
            if lid == 0:  # Lane 0 of each warp writes its partial sum
                sm[m, wid] = acc[m]

        # Final reduction and output
        cute.arch.sync_threads()
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                t = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NW):
                    t = t + sm[m, w]
                gC[m * N_dim + n_idx] = t
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
        dotprod_bf16_lf(gA, gB, gC, M, K_dim, N_dim).launch(
            grid=[N_dim, 1, 1],  # one CTA per expert
            block=[bs, 1, 1],
            smem=M * 4 * (bs // 32),
            stream=stream,
            use_pdl=True, 
            min_blocks_per_mp=1,
        )

    return host_bf16_lf
