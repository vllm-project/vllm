# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono MoE launch: Kimi-K3's routed experts (biased sigmoid top-k, sort,
a4w4 gemm1 and gemm2) and its bf16 shared expert, in one persistent
FlyDSL launch (README.md).

Workgroups loop on a global ticket counter and the ticket picks the work item:

    0 .. M-1                    top-k of token t, zero its output row
    M .. M+GU-1                 shared expert gate/up K splits
    M+GU .. M+GU+G1-1           gemm1 tile
    .. +DN                      shared expert down tiles
    ..                          gemm2 tile, m-block major

The workgroup that finishes the last top-k also sorts the routes and raises
the routed flag. A gemm1 tile waits for that flag; a gemm2 tile also waits for
every gemm1 n-block of its m-block; a shared down tile waits for every gate/up
pair. Each wait names an item of a smaller ticket, and a workgroup holding a
ticket works on it until done, so the grid cannot deadlock however many
workgroups are resident.
"""

import hashlib
import pathlib

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels import mxfp4_gemm1, mxfp4_gemm_common
from aiter.ops.flydsl.kernels.mxfp4_gemm1 import (
    _bm_constants,
    default_epi_splits,
    default_k_stages,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import compile_gemm2_a4w4_port
from flydsl.expr import const_expr, gpu, ptrtoint, rocdl
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.mono.common import ops
from vllm.models.kimi_k3.amd.mono.common.plan import (
    BM,
    MB_STRIDE,
    N_WAVES,
    SPIN_SLEEP,
    THREADS,
    W_EPOCH,
    W_MBLOCK,
    W_ROUTED,
    W_TICKET,
    W_TOPK,
    sh_done_word,
    sh_pair_word,
    sh_pairs,
    ws_layout,
)
from vllm.models.kimi_k3.amd.mono.common.sync import (
    bump,
    count,
    grab,
    raise_flag,
    wait_ge,
)
from vllm.models.kimi_k3.amd.mono.stages.gemm1 import _gemm1_body
from vllm.models.kimi_k3.amd.mono.stages.route import sort_routes, topk_token
from vllm.models.kimi_k3.amd.mono.stages.shared import shared_down, shared_gate_up

G1_BK = 256
G2_BK = 128

# FlyDSL keys its compile cache on the kernel's source and scalar closure
# values, not on the stages it calls; the package's sources and the AITER
# kernel sources it traces go into the key.
_SOURCES = hashlib.sha256(
    b"".join(
        p.read_bytes()
        for p in (
            *sorted(pathlib.Path(__file__).parent.rglob("*.py")),
            *(pathlib.Path(m.__file__) for m in (mxfp4_gemm1, mxfp4_gemm_common)),
        )
    )
).hexdigest()[:16]


def compile_mono_moe(
    *,
    M_MAX,
    NE,
    TOPK,
    D_HIDDEN,
    D_INTER,
    G1_BN,
    G2_BN,
    G1_BN_WIDE=0,
    G1_WIDE_MB=96,
    situ_beta,
    situ_linear_beta,
    SH_INTER,
    SH_HIDDEN,
    sh_beta,
    sh_linear_beta,
    SH_KS=4,
    SH_DN_BN=64,
    TRACE=False,
):
    """Compile the launch; returns the launcher.

    G1_BN_WIDE switches gemm1 to that tile width once routing yields
    G1_WIDE_MB m-blocks or more (fewer, wider tiles amortise the fixed costs;
    with few m-blocks narrow tiles keep more CUs streaming weights).

    TRACE: arg_trace receives ROUTE_MARKS i64 for the routing phase, then four
    i64 per ticket (start, dependencies met, end); the unit is the 100 MHz
    s_memrealtime clock.

    The bf16 shared expert (this rank's SH_INTER columns of gate and up,
    hidden SH_HIDDEN) runs as SH_INTER / 16 * SH_KS gate/up tickets after
    the top-k ones, so they run while routing holds the other workgroups back,
    and SH_HIDDEN / SH_DN_BN down tickets after gemm1, where gate/up is long
    done. sh_linear_beta <= 0 passes up through unclipped. M_MAX must be <= BM:
    the shared expert's tiles cover one m-block.
    """
    assert NE % N_WAVES == 0 and TOPK <= 64 and M_MAX <= BM
    assert SH_INTER % 32 == 0 and SH_HIDDEN % (SH_KS * N_WAVES * 32) == 0
    assert SH_DN_BN % (16 * N_WAVES) == 0 and SH_HIDDEN % SH_DN_BN == 0
    SH_GU_T = sh_pairs(SH_INTER) * SH_KS
    SH_DN_T = SH_HIDDEN // SH_DN_BN
    N_OUT1 = 2 * D_INTER
    NNB1 = N_OUT1 // G1_BN
    NNB2 = D_HIDDEN // G2_BN
    K_TILES1 = D_HIDDEN // G1_BK

    def g1_tile(bn):
        epi = default_epi_splits(BM, bn)
        ks = default_k_stages(BM, bn, G1_BK // 2, K_TILES1, N_OUT1, 1, epi)
        lds = _bm_constants(BM, bn, G1_BK // 2, K_TILES1, 1, epi, ks)[3]
        kw = dict(
            BM=BM, BN=bn, BK=G1_BK, inline_quant=True, prefetch_hidden=True,
            a_dtype="fp4", out_dtype="fp4", act="situv2", situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta, swiglu_limit=7.0, enable_bias=False,
            K=D_HIDDEN, N_OUT=N_OUT1, NE=NE, interleave=False,
            native_scale_layout=True, num_waves=4, k_wave=1, epi_splits=epi,
            k_stages=ks, wt_out=True,
        )  # fmt: skip
        return kw, lds

    g1_kw, g1_lds_bytes = g1_tile(G1_BN)
    g1w_kw, NNB1W = {}, NNB1
    if G1_BN_WIDE:
        g1w_kw, lds_w = g1_tile(G1_BN_WIDE)
        g1_lds_bytes = max(g1_lds_bytes, lds_w)
        NNB1W = N_OUT1 // G1_BN_WIDE

    # Routing LDS (i32 words). The sort's per-m-block fill shares the count
    # words' tail and its bitmaps sit in the next M_MAX * TOPK words.
    CNT_WORDS = (NE + THREADS - 1) // THREADS * THREADS
    L_CNT = 0
    L_BFILL = L_CNT + CNT_WORDS
    L_BMAP = L_BFILL + CNT_WORDS
    L_WTOT = L_BMAP + M_MAX * TOPK
    L_PART = L_WTOT + N_WAVES
    L_PIV = L_PART + THREADS
    L_CV = L_PIV + N_WAVES
    L_CI = L_CV + NE
    L_CS = L_CI + NE
    L_SEL = L_CS + NE
    L_SLOT = L_SEL + TOPK
    route_lds_words = L_SLOT + 8
    assert ((M_MAX + BM - 1) // BM) * ((NE + 31) // 32) <= M_MAX * TOPK
    # One LDS slot per control-word call site (common/sync.py).
    S_TOPK_COUNT, S_FIRST, S_TOPK, S_GEMM, S_SHARED, S_SH_PAIR = (
        L_SLOT + 1, L_SLOT + 3, L_SLOT + 4, L_SLOT + 5, L_SLOT + 6, L_SLOT + 7,
    )  # fmt: skip

    topk_kw = dict(
        NE=NE, TOPK=TOPK, D_HIDDEN=D_HIDDEN, TRACE=TRACE, L_PART=L_PART,
        L_PIV=L_PIV, L_WTOT=L_WTOT, L_CV=L_CV, L_CI=L_CI, L_CS=L_CS, L_SEL=L_SEL,
    )  # fmt: skip
    sort_kw = dict(
        NE=NE, TOPK=TOPK, M_MAX=M_MAX, TRACE=TRACE, L_CNT=L_CNT, L_BMAP=L_BMAP,
        L_BFILL=L_BFILL,
    )  # fmt: skip
    sh_kw = dict(SH_HIDDEN=SH_HIDDEN, SH_INTER=SH_INTER)
    W_SH_PAIR = sh_pair_word(M_MAX, TOPK)
    W_SH_DONE = sh_done_word(M_MAX, TOPK, SH_INTER)
    ws_offs = ws_layout(M_MAX, TOPK, D_INTER, SH_INTER, SH_KS)[0]

    name = (
        f"k3_mono_moe_m{M_MAX}_ne{NE}_k{TOPK}_h{D_HIDDEN}_i{D_INTER}"
        f"_g1bn{G1_BN}_g1w{G1_BN_WIDE}t{G1_WIDE_MB}_g2bn{G2_BN}"
        f"_b{situ_beta:g}l{situ_linear_beta:g}"
        f"_sh{SH_HIDDEN}x{SH_INTER}ks{SH_KS}dn{SH_DN_BN}b{sh_beta:g}l{sh_linear_beta:g}"
        f"{'_trace' if TRACE else ''}"
    )
    cache_tag = repr(
        (name, _SOURCES, sorted(topk_kw.items()), sorted(sort_kw.items()),
         sorted(g1_kw.items()), sorted(g1w_kw.items()))
    )  # fmt: skip

    def compose(*, module_name, emit_gemm2_tile, shared_storage):
        @fx.struct
        class RouteStorage:
            raw: fx.Array[fx.Int32, route_lds_words, 16]

        @fx.struct
        class G1Storage:
            raw: fx.Array[fx.Uint8, g1_lds_bytes, 16]

        @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
        def mono_moe_kernel(
            arg_logits: fx.Int64,
            arg_bias: fx.Int64,
            arg_x: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1s: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2s: fx.Int64,
            arg_out: fx.Int64,
            arg_tw: fx.Int64,
            arg_ti: fx.Int64,
            arg_ws: fx.Int64,
            arg_sh_x: fx.Int64,
            arg_sh_wgu: fx.Int64,
            arg_sh_wdn: fx.Int64,
            arg_sh_out: fx.Int64,
            arg_trace: fx.Int64,
            i32_M: fx.Int32,
            i32_grid: fx.Int32,
        ):
            _ = cache_tag
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
            smem = fx.SharedAllocator()
            route_lds = smem.allocate(RouteStorage).peek().raw.ptr
            g1_lds = smem.allocate(G1Storage).peek().raw.ptr
            g2_lds = smem.allocate(shared_storage).peek()
            lds_base = fx.Int32(ptrtoint(route_lds))

            ctrl = fx.Int64(arg_ws)
            arg_stids = ctrl + fx.Int64(ws_offs["stids"])
            arg_sw = ctrl + fx.Int64(ws_offs["sw"])
            arg_eids = ctrl + fx.Int64(ws_offs["eids"])
            arg_cumsum = ctrl + fx.Int64(ws_offs["cumsum"])
            arg_mind = ctrl + fx.Int64(ws_offs["mind"])
            arg_inter = ctrl + fx.Int64(ws_offs["inter"])
            arg_inter_scale = ctrl + fx.Int64(ws_offs["inter_scale"])
            a_ticket = ctrl + fx.Int64(W_TICKET * 4)
            a_routed = ctrl + fx.Int64(W_ROUTED * 4)
            a_epoch = ctrl + fx.Int64(W_EPOCH * 4)
            a_topk = ctrl + fx.Int64(W_TOPK * 4)
            a_mblock = ctrl + fx.Int64(W_MBLOCK * 4)
            a_sh_pair = ctrl + fx.Int64(W_SH_PAIR * 4)
            a_sh_done = ctrl + fx.Int64(W_SH_DONE * 4)
            arg_sh_part = ctrl + fx.Int64(ws_offs["sh_part"])
            arg_sh_h = ctrl + fx.Int64(ws_offs["sh_h"])

            def mark(t, k):
                if const_expr(TRACE):
                    ops.ticket_mark(tid, arg_trace, t, k)

            def rmark(k):
                if const_expr(TRACE):
                    ops.route_mark(tid, arg_trace, k)

            mb_max = i32_M * fx.Int32(TOPK)
            # The epoch word holds a launch count (mod 2^30). The routed flag is
            # never reset: this launch stores epoch + 1 into it and waiters wait
            # for equality, so the epoch can wrap. It is read before the first
            # grab, so it cannot move under us (the last grab moves it).
            epoch = rocdl.readfirstlane(
                T.i32, fx.Int32(comm_ops.load_i32_global_agent(a_epoch))
            )
            routed_at = (epoch + fx.Int32(1)) & fx.Int32((1 << 30) - 1)
            t = grab(wave, lane, lds_base, a_ticket, S_FIRST)

            while t < i32_M:
                mark(t, 0)
                topk_token(
                    lds_base, arg_logits, arg_bias, arg_tw, arg_ti, arg_out, t,
                    tid, lane, wave, arg_trace, **topk_kw,
                )  # fmt: skip
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                rmark(8)
                n_topk = count(wave, lane, lds_base, a_topk, S_TOPK_COUNT, False)
                rmark(9)
                mark(t, 1)
                if n_topk == i32_M - fx.Int32(1):
                    wait_ge(wave, a_topk, i32_M, True, SPIN_SLEEP)
                    rmark(0)
                    sort_routes(
                        lds_base, arg_tw, arg_ti, arg_stids, arg_sw, arg_eids,
                        arg_cumsum, arg_mind, i32_M, tid, lane, wave, arg_trace,
                        **sort_kw,
                    )  # fmt: skip
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    rmark(12)
                    raise_flag(wave, a_routed, routed_at, False)
                    rmark(1)
                mark(t, 2)
                t = grab(wave, lane, lds_base, a_ticket, S_TOPK)

            while t < i32_M + fx.Int32(SH_GU_T):
                mark(t, 0)
                shared_gate_up(
                    t - i32_M, tid, lane, wave, lds_base, i32_M, arg_sh_x,
                    arg_sh_wgu, arg_sh_part, arg_sh_h, a_sh_pair, a_sh_done,
                    SH_KS=SH_KS, PAIR_STRIDE=MB_STRIDE, SLOT=S_SH_PAIR,
                    beta=sh_beta, linear_beta=sh_linear_beta, **sh_kw,
                )  # fmt: skip
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                mark(t, 2)
                t = grab(wave, lane, lds_base, a_ticket, S_SHARED)

            wait_ge(wave, a_routed, routed_at, True, SPIN_SLEEP, True)
            total_mb = fx.Int32(global_typed_ptr(arg_cumsum, T.i32)[0]) // fx.Int32(BM)
            wide = fx.Int32(0)
            nnb1 = fx.Int32(NNB1)
            if const_expr(G1_BN_WIDE):
                q = total_mb // fx.Int32(G1_WIDE_MB)
                wide = fx.Int32(1) - fx.Int32(1) // (q + fx.Int32(1))
                nnb1 = fx.Int32(NNB1) - fx.Int32(NNB1 - NNB1W) * wide
            n_g1 = total_mb * nnb1
            t0 = i32_M + fx.Int32(SH_GU_T)
            g2_0 = n_g1 + fx.Int32(SH_DN_T)
            n_work = t0 + g2_0 + total_mb * fx.Int32(NNB2)

            while t < n_work:
                mark(t, 0)
                wk = t - t0
                if wk < n_g1:
                    mb1 = wk // nnb1
                    mark(t, 1)
                    if const_expr(G1_BN_WIDE):
                        if wide == fx.Int32(1):
                            _gemm1_body(
                                g1_lds, arg_x, arg_x, arg_w1, arg_w1s, arg_eids,
                                arg_mind, arg_inter, arg_inter_scale, arg_x,
                                fx.Int64(0), wk, lane, wave, True, i32_M, total_mb,
                                **g1w_kw,
                            )  # fmt: skip
                        if wide == fx.Int32(0):
                            _gemm1_body(
                                g1_lds, arg_x, arg_x, arg_w1, arg_w1s, arg_eids,
                                arg_mind, arg_inter, arg_inter_scale, arg_x,
                                fx.Int64(0), wk, lane, wave, True, i32_M, total_mb,
                                **g1_kw,
                            )  # fmt: skip
                    else:
                        _gemm1_body(
                            g1_lds, arg_x, arg_x, arg_w1, arg_w1s, arg_eids,
                            arg_mind, arg_inter, arg_inter_scale, arg_x,
                            fx.Int64(0), wk, lane, wave, True, i32_M, total_mb,
                            **g1_kw,
                        )  # fmt: skip
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    bump(
                        wave, lane, a_mblock + fx.Int64(mb1) * fx.Int64(4 * MB_STRIDE),
                        False,
                    )  # fmt: skip
                    mark(t, 2)
                # Traced conditions: nested ifs, as `and` would short-circuit
                # on the host.
                if wk >= n_g1:  # noqa: SIM102
                    if wk < g2_0:
                        mark(t, 1)
                        shared_down(
                            wk - n_g1, lane, wave, i32_M, arg_sh_h, arg_sh_wdn,
                            arg_sh_out, a_sh_done, SH_DN_BN=SH_DN_BN,
                            SPIN=SPIN_SLEEP, **sh_kw,
                        )  # fmt: skip
                        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        mark(t, 2)
                if wk >= g2_0:
                    u = wk - g2_0
                    mb2 = u // fx.Int32(NNB2)
                    nb2 = u - mb2 * fx.Int32(NNB2)
                    a_mb2 = a_mblock + fx.Int64(mb2) * fx.Int64(4 * MB_STRIDE)
                    wait_ge(wave, a_mb2, nnb1, True, SPIN_SLEEP)
                    mark(t, 1)
                    emit_gemm2_tile(
                        arg_inter, arg_inter_scale, arg_w2, arg_w2s, arg_eids,
                        arg_stids, arg_sw, arg_w2, arg_out, mb2, nb2, lane, wave,
                        i32_M, mb_max, fx.Int32(D_INTER), fx.Int32(D_HIDDEN), g2_lds,
                    )  # fmt: skip
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    mark(t, 2)
                t = grab(wave, lane, lds_base, a_ticket, S_GEMM)

            # Every workgroup ends holding exactly one ticket >= n_work, taken
            # after its last item. The holder of the largest one is the last
            # workgroup to use the ticket, top-k and m-block words; the others
            # only still read the routed flag, which is not reset.
            if t == n_work + i32_grid - fx.Int32(1):
                ctrl32 = global_typed_ptr(arg_ws, T.i32)
                for i in range(tid, mb_max, fx.Int32(THREADS)):
                    ctrl32[fx.Int32(W_MBLOCK) + i * fx.Int32(MB_STRIDE)] = fx.Int32(0)
                if tid <= fx.Int32(sh_pairs(SH_INTER)):
                    ctrl32[fx.Int32(W_SH_PAIR) + tid * fx.Int32(MB_STRIDE)] = fx.Int32(
                        0
                    )
                if tid == fx.Int32(0):
                    ctrl32[W_TICKET] = fx.Int32(0)
                    ctrl32[W_TOPK] = fx.Int32(0)
                    ctrl32[W_EPOCH] = routed_at

        @flyc.jit
        def launch_mono_moe(
            arg_logits: fx.Int64,
            arg_bias: fx.Int64,
            arg_x: fx.Int64,
            arg_w1: fx.Int64,
            arg_w1s: fx.Int64,
            arg_w2: fx.Int64,
            arg_w2s: fx.Int64,
            arg_out: fx.Int64,
            arg_tw: fx.Int64,
            arg_ti: fx.Int64,
            arg_ws: fx.Int64,
            arg_sh_x: fx.Int64,
            arg_sh_wgu: fx.Int64,
            arg_sh_wdn: fx.Int64,
            arg_sh_out: fx.Int64,
            arg_trace: fx.Int64,
            i32_M: fx.Int32,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            mono_moe_kernel(
                arg_logits,
                arg_bias,
                arg_x,
                arg_w1,
                arg_w1s,
                arg_w2,
                arg_w2s,
                arg_out,
                arg_tw,
                arg_ti,
                arg_ws,
                arg_sh_x,
                arg_sh_wgu,
                arg_sh_wdn,
                arg_sh_out,
                arg_trace,
                i32_M,
                i32_grid,
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1), block=(THREADS, 1, 1), stream=stream
            )

        return launch_mono_moe

    launch = compile_gemm2_a4w4_port(
        BM=BM, BN=G2_BN, BK=G2_BK, use_nt=False, HIDDEN_MAX=8192, epilog="atomic",
        INTER_MAX=D_INTER, a_dtype="fp4", b_dtype="fp4", SBM=BM, g2_kstatic=True,
        _composition=compose,
    )  # fmt: skip
    launch.work_meta = dict(NNB1=NNB1, NNB2=NNB2, SH_T=SH_GU_T + SH_DN_T)
    return launch
