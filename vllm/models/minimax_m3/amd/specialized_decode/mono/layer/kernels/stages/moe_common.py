# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/stages/moe_common.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the narrow MoE's up / gate / down tasks (the FFN all-reduce:
``ffn_reduce``)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from ...config import (
    HIDDEN,
    INTER,
    MOE_SLOTS,
    SHARED_EXPERT,
    TOP_K,
)
from ...layout import (
    SH_ROWS,
    UG_PER_SLOT,
    UG_ROWS,
    XN8_ROW,
    XSC_ROW,
)
from ...support.device.mx import (
    FP4,
    fp4_tile_load,
    mfma_scaled,
    scale_index,
)
from ...support.device.ops import (
    CM_NT,
    bf16_round,
    row_sum,
    rsrc,
    traced,
    uniform,
)
from .shared import (
    S2_BYTES,
    S13_BYTES,
    W2_BYTES,
    W2_SCALE_COLS,
    W13_BYTES,
    W13_ROWS,
    W13_SCALE_COLS,
)


@traced
def moe_common_defs(k4ctx):
    """The functions of the narrow MoE's up / gate / down tasks."""
    CPT = k4ctx["CPT"]
    DN_KC = k4ctx["DN_KC"]
    DN_UNITS = k4ctx["DN_UNITS"]
    DN_UPW = k4ctx["DN_UPW"]
    G = k4ctx["G"]
    ROUTED_TASKS = k4ctx["ROUTED_TASKS"]
    SH_UPW = k4ctx["SH_UPW"]
    TPT = k4ctx["TPT"]
    UG_CPW = k4ctx["UG_CPW"]
    XSC_LDS = k4ctx["XSC_LDS"]
    bid = k4ctx["bid"]
    e8m0_load = k4ctx["e8m0_load"]
    g4 = k4ctx["g4"]
    group_scale_byte = k4ctx["group_scale_byte"]
    l16 = k4ctx["l16"]
    lane = k4ctx["lane"]
    lds_b8 = k4ctx["lds_b8"]
    lds_i32 = k4ctx["lds_i32"]
    mb_put = k4ctx["mb_put"]
    mr = k4ctx["mr"]
    opart = k4ctx["opart"]
    red = k4ctx["red"]
    route = k4ctx["route"]
    rwt = k4ctx["rwt"]
    s13 = k4ctx["s13"]
    s2 = k4ctx["s2"]
    shared_weight = k4ctx["shared_weight"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    swiglu_limit = k4ctx["swiglu_limit"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    w13 = k4ctx["w13"]
    w2 = k4ctx["w2"]
    wave = k4ctx["wave"]

    def ug_task_id(it):
        """(this CTA's it-th routed task, clamped; whether it exists)."""
        if const_expr(CPT is not None):
            u = bid % CPT + it * CPT
            return (bid // CPT) * TPT + fx.min(u, TPT - 1), u < TPT
        t = start("ug") + it * G
        return fx.min(t, ROUTED_TASKS - 1), t < ROUTED_TASKS

    def ug_task(t):
        """(shared expert?, token, slot, row group) of up / gate task t."""
        shared = t >= ROUTED_TASKS
        r = fx.min(t, ROUTED_TASKS - 1)
        tk = r // (TOP_K * UG_PER_SLOT)
        slot = shared.select(
            fx.Int32(TOP_K), (r % (TOP_K * UG_PER_SLOT)) // UG_PER_SLOT
        )
        return shared, tk, slot, t % UG_PER_SLOT

    def ug_load(t):
        """Up / gate weights and scales of task t."""
        shared, tk, slot, cg = ug_task(t)
        # the bound keeps a corrupt route from addressing past the expert table
        e_u = shared.select(
            fx.Int32(SHARED_EXPERT),
            fx.min(uniform(fx.ptr_load(route + (8 * tk + slot))), SHARED_EXPERT),
        )
        r_w = rsrc(w13 + fx.Int64(e_u) * W13_BYTES)
        r_s = rsrc(s13)
        wts: list = []  # [gate, up][chunk]
        scs: list = []
        for gu in range_constexpr(2):
            rg = gu * UG_PER_SLOT + cg
            wts.append([])
            scs.append([])
            for i in range_constexpr(UG_CPW):
                kc = wave * UG_CPW + i
                wts[gu].append(fp4_tile_load(r_w, 0, rg, HIDDEN, kc, lane * 4))
                scs[gu].append(
                    e8m0_load(
                        r_s,
                        group_scale_byte(e_u, W13_ROWS, rg, kc, W13_SCALE_COLS),
                    )
                )
        return wts, scs

    def ug_compute(t, wts, scs):
        """Up / gate GEMV of task t -> swiglu -> its 16 mid values (of every token
        for the shared expert: B column l % 16 is token l % 16)."""
        shared, tk, slot, cg = ug_task(t)
        col = shared.select(fx.min(l16, tokens - 1), tk)
        cg_ = fx.Vector.filled(4, 0.0, fx.Float32)
        cu_ = fx.Vector.filled(4, 0.0, fx.Float32)
        # every chunk's B scale up front: read at use, each MFMA waited out an LDS
        # latency (S = 4 up / gate 3.5 us over the unscaled floor)
        sbs = [
            lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            xb = lds_b8(col * XN8_ROW + kc * 32)
            cg_ = mfma_scaled(wts[0][i], xb, cg_, scs[0][i], sbs[i], a_fmt=FP4)
            cu_ = mfma_scaled(wts[1][i], xb, cu_, scs[1][i], sbs[i], a_fmt=FP4)
        stamp(9)
        # gate partials in red, up partials in opart (free once the split is done)
        fx.ptr_store(cg_, red + (wave * 64 + lane) * 4)
        fx.ptr_store(cu_, opart + (wave * 64 + lane) * 4)
        gpu.barrier()
        # row tid % 16; a shared task's thread tid reads token tid / 16's column
        if tid < shared.select(fx.Int32(UG_ROWS * tokens), fx.Int32(UG_ROWS)):
            r = tid % UG_ROWS
            col = tid // UG_ROWS
            kk = shared.select(col, tk)
            gsum = row_sum(red, r, col)
            usum = row_sum(opart, r, col)
            y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
            mb_put(
                mr("mid"),
                (kk * MOE_SLOTS + slot) * INTER + cg * UG_ROWS + r,
                bf16_round(y),
            )
        gpu.barrier()

    def sh_load(j, has=None):
        """The shared expert's half task j: gate rows 8 j .. + 8 on lanes l16 < 8,
        the up rows alike on the rest -- one MFMA tile; the waves split k."""
        row = (l16 < 8).select(SH_ROWS * j + l16, INTER + SH_ROWS * j + l16 - 8)
        rg = row // 16
        if const_expr(has is None):
            r_w = rsrc(w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES)
            r_s = rsrc(s13)
        else:  # a CTA without the task reads through zero-sized buffers
            r_w = rsrc(
                w13 + fx.Int64(SHARED_EXPERT) * W13_BYTES,
                has.select(fx.Int32(W13_BYTES), fx.Int32(0)),
            )
            r_s = rsrc(s13, has.select(fx.Int32(S13_BYTES), fx.Int32(0)))
        wts = []
        scs = []
        # the row's part of scale_index, once: a chunk adds 256 (kc / 2) + 2 (kc % 2)
        sc_row = scale_index(SHARED_EXPERT * W13_ROWS + row, g4, W13_SCALE_COLS)
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            wts.append(
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        (
                            (rg * (HIDDEN // 64) + kc * 2) * 512
                            + (g4 * 16 + row % 16) * 16
                        )
                        // 4,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
            )
            scs.append(e8m0_load(r_s, sc_row + kc // 2 * 256 + kc % 2 * 2))
        return wts, scs

    def sh_compute(j, wts, scs):
        """Half task j's GEMV (B column l % 16 = token l % 16) -> swiglu -> the
        shared slot's mid rows 8 j .. + 8 of every token."""
        col = fx.min(l16, tokens - 1)
        c = fx.Vector.filled(4, 0.0, fx.Float32)
        sbs = [
            lds_i32(XSC_LDS + col * XSC_ROW + (wave * UG_CPW + i) * 4 + g4)
            for i in range(UG_CPW)
        ]
        for i in range_constexpr(UG_CPW):
            kc = wave * UG_CPW + i
            c = mfma_scaled(
                wts[i], lds_b8(col * XN8_ROW + kc * 32), c, scs[i], sbs[i], a_fmt=FP4
            )
        fx.ptr_store(c, red + (wave * 64 + lane) * 4)
        gpu.barrier()
        # mid row r = tid % 8 of token tid / 8: gate in tile row r, up in r + 8
        if tid < SH_ROWS * tokens:
            r = tid % SH_ROWS
            col = tid // SH_ROWS
            gsum = row_sum(red, r, col)
            usum = row_sum(red, r + 8, col)
            y = swiglu_mul_batch([gsum], [usum], fx.Float32(-swiglu_limit))[0]
            mb_put(
                mr("mid"),
                (col * MOE_SLOTS + TOP_K) * INTER + j * SH_ROWS + r,
                bf16_round(y),
            )
        gpu.barrier()

    def down_load(k, td, has_dn):
        """Token k's route-weighted down units of row group td; a CTA without a
        down task reads through zero-sized buffers."""
        dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
        dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
        rg_d = td * 2 + wave // 4
        units = []
        for i in range_constexpr(DN_UPW):
            j = (wave % 4) + 4 * i
            jj = fx.min(j, DN_UNITS - 1)
            sl = jj // DN_KC
            kc = jj % DN_KC
            e_d = fx.min(uniform(fx.ptr_load(route + (8 * k + sl))), SHARED_EXPERT)
            wv = fp4_tile_load(
                rsrc(w2 + fx.Int64(e_d) * W2_BYTES, dn_bytes),
                0,
                rg_d,
                INTER,
                kc,
                lane * 4,
            )
            sc = e8m0_load(
                rsrc(s2, dn_scale_bytes),
                group_scale_byte(e_d, HIDDEN, rg_d, kc, W2_SCALE_COLS),
            )
            wgt = (j < DN_UNITS).select(
                fx.ptr_load(rwt + (8 * k + sl)), fx.Float32(0.0)
            )
            units.append((wv, sc, sl, kc, wgt))
        return units

    def down_load_shared(td, has_dn):
        """The shared expert's down units of row group td (every token's)."""
        dn_bytes = has_dn.select(fx.Int32(W2_BYTES), fx.Int32(0))
        dn_scale_bytes = has_dn.select(fx.Int32(S2_BYTES), fx.Int32(0))
        rg_d = td * 2 + wave // 4
        units = []
        for i in range_constexpr(SH_UPW):
            j = (wave % 4) + 4 * i
            kc = fx.min(j, DN_KC - 1)
            wv = fp4_tile_load(
                rsrc(w2 + fx.Int64(SHARED_EXPERT) * W2_BYTES, dn_bytes),
                0,
                rg_d,
                INTER,
                kc,
                lane * 4,
            )
            sc = e8m0_load(
                rsrc(s2, dn_scale_bytes),
                group_scale_byte(SHARED_EXPERT, HIDDEN, rg_d, kc, W2_SCALE_COLS),
            )
            wgt = (j < DN_KC).select(fx.Float32(shared_weight), fx.Float32(0.0))
            units.append((wv, sc, kc, wgt))
        return units

    return {
        "ug_task_id": ug_task_id,
        "ug_task": ug_task,
        "ug_load": ug_load,
        "ug_compute": ug_compute,
        "sh_load": sh_load,
        "sh_compute": sh_compute,
        "down_load": down_load,
        "down_load_shared": down_load_shared,
    }
