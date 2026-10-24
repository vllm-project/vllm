# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/stages/router.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the router GEMV (and the MXFP8 operand helpers the experts share)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from ...config import (
    HIDDEN,
    N_ROUTED,
    SHARED_EXPERT,
    TOP_K,
)
from ...layout import (
    N_ROUTER,
    SCORE_COPIES,
)
from ...support.device.mx import (
    FP8_MAX,
    clamp_fp8,
    code_ceil,
    pow2,
)
from ...support.device.ops import (
    CM_DEV,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    rsrc,
    traced,
    wave_sum,
)
from ...support.device.sync import (
    preg,
    publish,
)
from ...support.plan.execution import (
    THREADS,
    WAVES,
)
from ..common import (
    ld_bf16x12,
)
from .shared import (
    XN_SLICE,
)


@traced
def router_defs(k4ctx):
    """The functions of the router GEMV (and the MXFP8 operand helpers the experts
    share)."""
    AG_RS = k4ctx["AG_RS"]
    CM_K1 = k4ctx["CM_K1"]
    G = k4ctx["G"]
    SY = k4ctx["SY"]
    WIDE = k4ctx["WIDE"]
    XN8_WORDS = k4ctx["XN8_WORDS"]
    XSC_WORDS = k4ctx["XSC_WORDS"]
    acquire = k4ctx["acquire"]
    bias = k4ctx["bias"]
    eps = k4ctx["eps"]
    fuse_k1 = k4ctx["fuse_k1"]
    g4 = k4ctx["g4"]
    g_post = k4ctx["g_post"]
    h_in = k4ctx["h_in"]
    h_mid = k4ctx["h_mid"]
    k1_poll = k4ctx["k1_poll"]
    lane = k4ctx["lane"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put = k4ctx["mb_put"]
    mb_put_words = k4ctx["mb_put_words"]
    mr = k4ctx["mr"]
    rdone_mb = k4ctx["rdone_mb"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    route_key = k4ctx["route_key"]
    route_scale = k4ctx["route_scale"]
    rwt = k4ctx["rwt"]
    scores_copy = k4ctx["scores_copy"]
    shared_weight = k4ctx["shared_weight"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    v4f = k4ctx["v4f"]
    w_gate = k4ctx["w_gate"]
    wave = k4ctx["wave"]
    xs = k4ctx["xs"]

    # ======================================================= MoE helpers
    def e8m0_load(r_s, byte_idx):
        """An e8m0 scale byte, still in flight (the MFMA takes it as is)."""
        return fx.Int32(bo.buffer_load(r_s, byte_idx, vec_width=1, dtype=T.i8))

    def group_scale_byte(expert, rows, rg, kc, cols):
        """scale_index(expert rows + 16 rg + l16, 4 kc + g4, cols): this lane's
        byte for row group rg, 128-k chunk kc, folded to wave-uniform terms plus
        the lane (the general form's per-lane arithmetic cost the S = 16 up /
        gate loop ~8 us)."""
        base = (expert * (rows // 32) + rg // 2) * (cols // 8) + kc // 2
        return (base * 64 + lane) * 4 + (kc % 2) * 2 + rg % 2

    def route_top4(k, k0, sc0, k1, sc1, sig, wave_mask=None, mask_bit=0):
        """route[8k : 8k+4] / rwt[..] := token k's sigmoid top-k gating from the
        router tasks' routing keys ``k0`` / ``k1`` (experts ``lane`` / ``lane +
        64``, sigmoid + bias, see route_key) and unbiased sigmoids (wave k % WAVES
        computes it, looking the sigmoids up in LDS at ``sig``): a pick is one
        wave max; the weights are the picks' sigmoids renormalized and scaled by
        route_scale. Slot 4 is the fused shared expert. ``wave_mask``: this
        wave's expert -> token mask, ``mask_bit`` or-ed in at the picks (they
        are distinct, and the wave's tokens run in program order): bit k, or 0
        for a pad row (``row_live``), whose experts then join no token's work.
        The caller syncs."""
        if wave == k % WAVES:
            # the unbiased weights, looked up by expert once the picks are known
            fx.ptr_store(sc0, sig + (k * N_ROUTED + lane))
            fx.ptr_store(sc1, sig + (k * N_ROUTED + 64 + lane))
            pid = fx.Int32(0)
            for p in range_constexpr(TOP_K):
                m = butterfly(fx.max(k0, k1), (32, 16, 8, 4, 2, 1), fx.max)
                hit0 = k0 == m
                k0 = hit0.select(fx.Int32(-(2**31)), k0)
                k1 = ((k1 == m) & (k0 != m)).select(fx.Int32(-(2**31)), k1)
                pid = (lane == p).select(127 - (m & 127), pid)
            if lane < TOP_K:
                w = fx.ptr_load(sig + (k * N_ROUTED + pid))
                tot = butterfly(w, (1, 2))
                fx.ptr_store(pid, route + (8 * k + lane))
                fx.ptr_store(
                    w * (route_scale / fx.max(tot, 1e-20)), rwt + (8 * k + lane)
                )
                if const_expr(wave_mask is not None):
                    was = fx.Int32(fx.ptr_load(wave_mask + pid))
                    fx.ptr_store(was | mask_bit, wave_mask + pid)
            if lane == 0:
                fx.ptr_store(fx.Int32(SHARED_EXPERT), route + (8 * k + TOP_K))
                fx.ptr_store(fx.Float32(shared_weight), rwt + (8 * k + TOP_K))

    def mx8_scale(vals, offsets):
        """E8M0 byte (>= 1) of the 32-block the lanes xor-``offsets`` apart share,
        aiter's MXFP8 RoundUp rule ceil_pow2(amax / 448), and its reciprocal."""
        amax = fx.max(vals[0], -vals[0])
        for v in vals[1:]:
            amax = fx.max(amax, fx.max(v, -v))
        amax = butterfly(amax, offsets, fx.max)
        e = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        return e, hw_rcp(pow2(e))

    def mx8_q(v, inv):
        return clamp_fp8(v * inv)

    def lds_b8(chunk_word):
        """This lane's B operand of the 128-k chunk at LDS word ``chunk_word``:
        fp8 k 16 g .. + 16 then 64 + 16 g .. + 16 (g = lane // 16), the
        f8f6f4 MFMA's B layout (harness/mfma_f4f8_probe.py)."""
        h = [
            fx.Vector(
                fx.ptr_load(xs + (chunk_word + 16 * j + 4 * g4), result_type=v4f)
            ).bitcast(fx.Int32)
            for j in range(2)
        ]
        return fx.Vector.from_elements(
            [h[j][e] for j in range(2) for e in range(4)], fx.Int32
        )

    def bf16x2(word):
        return fx.Vector.from_elements([word], fx.Int32).bitcast(fx.BFloat16)

    def lds_i32(word):
        return fx.ptr_load(xs + word).bitcast(fx.Int32)

    # ============================================================ 4. router
    def stage_router():
        r_g = rsrc(g_post)
        r_gate = rsrc(w_gate)
        # this thread's 12 consecutive elements of the row (512 x 12 = HIDDEN):
        # the poll is 3 specs per rank, 12 in one batch, every thread busy
        RE = HIDDEN // THREADS
        rw0 = tid * (RE // 2)

        # one task per (token, 8 experts): the tokens' norms and dot products run
        # on different CTAs, not one after another in each (S = 4: 74.7 -> 64.1 us)
        for tt in range(start("router"), N_ROUTER * tokens, G):
            tt = fx.Int32(tt)
            k = tt // N_ROUTER
            t = tt % N_ROUTER
            e_r = t * WAVES + wave
            bias_r = fx.Float32(
                bo.buffer_load(rsrc(bias), e_r, vec_width=1, dtype=T.f32)
            )
            gw = [  # bf16 pairs
                fx.Vector(
                    bo.buffer_load(
                        r_gate,
                        (e_r * HIDDEN + (i * 64 + lane) * 8) // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for i in range(HIDDEN // 512)
            ]
            gp = ld_bf16x12(r_g, rw0)
            if const_expr(fuse_k1):
                # the residual (K1's norm / GEMV task)
                if wave == 0:
                    k1_poll([(rdone_mb, k, 1)])
                gpu.barrier()
                acquire()
            hres = ld_bf16x12(rsrc(h_in), k * (HIDDEN // 2) + rw0, CM_K1)
            # the all-reduced post-attention row (the o tasks finished it; with
            # AG_RS the owner ranks gathered it into this rank's a_ag)
            a_src = preg(sym, SY["a_ag"], "a_ag") if AG_RS else mr("a")
            arow = mb_poll(
                [
                    (a_src, (k * HIDDEN + tid * RE) // 2 + 2 * h, 2)
                    for h in range(RE // 4)
                ]
            )
            stamp(22)
            v = [
                f + fx.Float32(hres[4 * h + 2 * qq + i])
                for h in range(RE // 4)
                for qq in range(2)
                for i, f in enumerate(bf2_f32(arow[h][qq]))
            ]
            if const_expr(AG_RS):  # noqa: SIM102
                # h_mid = bf16(bf16(sum) + residual): task t writes slice t
                if tid // (XN_SLICE // RE) == t:
                    hw = (k * HIDDEN + tid * RE) // 2
                    hb = fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16)
                    hi = hb.bitcast(fx.Int32)
                    bo.buffer_store(
                        fx.Vector.from_elements([hi[e] for e in range(4)], fx.Int32),
                        rsrc(h_mid),
                        hw,
                    )
                    bo.buffer_store(
                        fx.Vector.from_elements([hi[4], hi[5]], fx.Int32),
                        rsrc(h_mid),
                        hw + 4,
                    )
            # the row's sum of squares: a wave sum each, one barrier, every lane
            # adds the 8
            ssw = fx.Float32(0.0)
            for e in range_constexpr(RE):
                ssw = fmath.fma(v[e], v[e], ssw)
            ssw = wave_sum(ssw)
            if lane == 0:
                fx.ptr_store(ssw, red + wave)
            gpu.barrier()
            ss = fx.ptr_load(red)
            for w in range_constexpr(1, WAVES):
                ss = ss + fx.ptr_load(red + w)
            stamp(24)
            rstd = hw_rsq(ss / float(HIDDEN) + eps)
            for j in range_constexpr(RE // 2):
                fx.ptr_store(
                    bf16_pair(
                        v[2 * j] * rstd * (fx.Float32(gp[2 * j]) + 1.0),
                        v[2 * j + 1] * rstd * (fx.Float32(gp[2 * j + 1]) + 1.0),
                    ),
                    xs + (rw0 + j),
                )
            gpu.barrier()
            stamp(25)
            if tid < XN_SLICE // 4:  # 4 elements a lane, 8 lanes a 32-block
                w0 = t * (XN_SLICE // 2) + tid * 2
                f = [
                    v
                    for j in range(2)
                    for v in bf2_f32(fx.ptr_load(xs + (w0 + j)).bitcast(fx.Int32))
                ]
                e, inv = mx8_scale(f, (1, 2, 4))
                word = fp8_pack4(*[mx8_q(v, inv) for v in f])
                xw = k * XN8_WORDS + t * (XN_SLICE // 4) + tid
                xsw = k * XSC_WORDS + t * (XN_SLICE // 32) + tid // 8
                if const_expr(WIDE):
                    # plain words, flagged once the task is done (below)
                    bo.buffer_store(word, rsrc(mb("xn8p")), xw, cache_modifier=CM_DEV)
                    if tid % 8 == 0:
                        bo.buffer_store(e, rsrc(mb("xscp")), xsw, cache_modifier=CM_DEV)
                else:
                    mb_put_words(mr("xn8"), xw, [word])
                    if tid % 8 == 0:
                        mb_put_words(mr("xsc"), xsw, [e])
            stamp(26)
            # bf16 pair dot products, one accumulator per pair position: four
            # 12-long chains instead of one 96-long FMA chain behind 192 widenings
            accs = [fx.Float32(0.0) for _ in range(4)]
            for i in range_constexpr(HIDDEN // 512):
                xv = fx.Vector(
                    fx.ptr_load(xs + (i * 64 + lane) * 4, result_type=v4f)
                ).bitcast(fx.Int32)
                for j in range_constexpr(4):
                    accs[j] = fx.Float32(
                        rocdl.fdot2_f32_bf16(
                            T.f32,
                            as_ir_value(bf16x2(gw[i][j])),
                            as_ir_value(bf16x2(xv[j])),
                            as_ir_value(accs[j]),
                            clamp=False,
                        ).result
                    )
            logit = wave_sum((accs[0] + accs[1]) + (accs[2] + accs[3]))
            stamp(23)
            # the routing key and sigmoid go out, not the logit: the consumers'
            # routing starts at the max rounds
            sc = hw_rcp(1.0 + hw_exp2(-bf16_round(logit) * LOG2E))
            rk = route_key(sc + bias_r, e_r)
            if lane < SCORE_COPIES:
                mb_put_words(scores_copy(lane, k), e_r * 2, [rk, sc.bitcast(fx.Int32)])
            if const_expr(WIDE):
                # this task's xn slice is out: its flag (after the logits, not
                # in front of them)
                publish(mb_put, mr("xndone"), tt, fx.Int32(1), tid == 0)

    return {
        "e8m0_load": e8m0_load,
        "group_scale_byte": group_scale_byte,
        "route_top4": route_top4,
        "mx8_scale": mx8_scale,
        "mx8_q": mx8_q,
        "lds_b8": lds_b8,
        "bf16x2": bf16x2,
        "lds_i32": lds_i32,
        "stage_router": stage_router,
    }
