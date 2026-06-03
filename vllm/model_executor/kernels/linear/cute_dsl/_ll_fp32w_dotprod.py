# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import operator

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream


def make_host_fp32w(k_val: int, bs: int = 128):
    _VPT = 8
    _BS = 128
    _KPI = _VPT * _BS  # 1024
    _k_main = k_val // _KPI

    _VPT_T = 4
    _KPT = _VPT_T * _BS  # 512
    _k_tail = (k_val - _k_main * _KPI) // _KPT

    _k_done = _k_main * _KPI + _k_tail * _KPT
    _scalar_rem = k_val - _k_done
    _ks_full = _scalar_rem // _BS
    _ks_part = _scalar_rem % _BS

    @cute.kernel
    def dotprod_fp32w_lf(
        gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
        M: cutlass.Constexpr, K_dim: cutlass.Constexpr, N_dim: cutlass.Int32,
    ):
        cute.arch.setmaxregister_increase(128)
        tidx = cute.arch.thread_idx()[0]
        n_idx = cute.arch.block_idx()[0]

        VPT: cutlass.Constexpr = _VPT
        BS: cutlass.Constexpr = _BS
        KPI: cutlass.Constexpr = _KPI
        K_MAIN: cutlass.Constexpr = _k_main

        elem_a = gA.element_type
        elem_b = gB.element_type
        b_base = gB.iterator + n_idx * K_dim
        tid_off = tidx * VPT

        acc = cute.make_rmem_tensor((M,), cutlass.Float32)
        for m in cutlass.range_constexpr(M):
            acc[m] = cutlass.Float32(0.0)

        #cute.arch.griddepcontrol_wait()

        # Main K-loop: load B, then per-token load A + FMA (no ar_all staging)
        for ki in cutlass.range_constexpr(K_MAIN):
            kb = ki * KPI + tid_off
            bp = (b_base + kb).align(32)
            bt = cute.make_tensor(bp, cute.make_layout((VPT,)))
            br = cute.make_rmem_tensor((VPT,), elem_b)
            cute.autovec_copy(bt, br)
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + kb)).align(16)
                at = cute.make_tensor(ap, cute.make_layout((VPT,)))
                ar = cute.make_rmem_tensor((VPT,), elem_a)
                cute.autovec_copy(at, ar)
                for v in cutlass.range_constexpr(VPT):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br[v]

        VPT_T: cutlass.Constexpr = _VPT_T
        KPT: cutlass.Constexpr = _KPT
        K_DONE: cutlass.Constexpr = _k_main * _KPI
        tid_off_t = tidx * VPT_T
        for ti in cutlass.range_constexpr(_k_tail):
            kb = K_DONE + ti * KPT + tid_off_t
            bp = (b_base + kb).align(16)
            bt = cute.make_tensor(bp, cute.make_layout((VPT_T,)))
            br = cute.make_rmem_tensor((VPT_T,), elem_b)
            cute.autovec_copy(bt, br)
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + kb)).align(8)
                at = cute.make_tensor(ap, cute.make_layout((VPT_T,)))
                ar = cute.make_rmem_tensor((VPT_T,), elem_a)
                cute.autovec_copy(at, ar)
                for v in cutlass.range_constexpr(VPT_T):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br[v]

        K_DONE_ALL: cutlass.Constexpr = _k_done
        for si in cutlass.range_constexpr(_ks_full):
            ko = K_DONE_ALL + si * BS + tidx
            bp = (b_base + ko).align(4)
            bt = cute.make_tensor(bp, cute.make_layout((1,)))
            br = cute.make_rmem_tensor((1,), elem_b)
            cute.autovec_copy(bt, br)
            bv = br[0]
            for m in cutlass.range_constexpr(M):
                ap = (gA.iterator + (m * K_dim + ko)).align(2)
                at = cute.make_tensor(ap, cute.make_layout((1,)))
                ar = cute.make_rmem_tensor((1,), elem_a)
                cute.autovec_copy(at, ar)
                acc[m] = acc[m] + ar[0].to(cutlass.Float32) * bv

        if _ks_part > 0:
            KS_PART: cutlass.Constexpr = _ks_part
            ko_p = K_DONE_ALL + _ks_full * BS + tidx
            if tidx < KS_PART:
                bp2 = (b_base + ko_p).align(4)
                bt2 = cute.make_tensor(bp2, cute.make_layout((1,)))
                br2 = cute.make_rmem_tensor((1,), elem_b)
                cute.autovec_copy(bt2, br2)
                bv2 = br2[0]
                for m in cutlass.range_constexpr(M):
                    ap2 = (gA.iterator + (m * K_dim + ko_p)).align(2)
                    at2 = cute.make_tensor(ap2, cute.make_layout((1,)))
                    ar2 = cute.make_rmem_tensor((1,), elem_a)
                    cute.autovec_copy(at2, ar2)
                    acc[m] = acc[m] + ar2[0].to(cutlass.Float32) * bv2

        WS: cutlass.Constexpr = 32
        NW: cutlass.Constexpr = BS // WS
        for m in cutlass.range_constexpr(M):
            acc[m] = cute.arch.warp_reduction(acc[m], operator.add)
        wid = tidx // WS
        lid = tidx % WS
        sp = cute.arch.alloc_smem(cutlass.Float32, M * NW, alignment=16)
        sm = cute.make_tensor(sp, cute.make_layout((M, NW)))
        for m in cutlass.range_constexpr(M):
            if lid == 0:
                sm[m, wid] = acc[m]
        cute.arch.sync_threads()
        if tidx == 0:
            for m in cutlass.range_constexpr(M):
                t = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(NW):
                    t = t + sm[m, w]
                gC[m * N_dim + n_idx] = t
        #cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def host_fp32w_lf(
        gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
        M: cutlass.Constexpr, K_dim: cutlass.Constexpr, N_dim: cutlass.Int32,
        stream: CUstream,
    ):
        dotprod_fp32w_lf(gA, gB, gC, M, K_dim, N_dim).launch(
            grid=[N_dim, 1, 1], block=[128, 1, 1],
            smem=M * 4 * 4, stream=stream, use_pdl=False, min_blocks_per_mp=1,
        )

    return host_fp32w_lf
