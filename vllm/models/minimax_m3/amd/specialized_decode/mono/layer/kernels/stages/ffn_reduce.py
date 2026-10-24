# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from ROCm/ATOM: atom/models/minimax_m3/mono/kernels/stages/ffn_reduce.py
# Original code is MIT-licensed; see mono/layer/LICENSE and the provenance in
# specialized_decode/README.md.
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layer-kernel stage: the FFN all-reduce of a down task's partial rows (the
sparse layers' MoE and the dense layers' MLP)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr

from ...config import (
    HIDDEN,
    MAX_TOKENS,
)
from ...layout import DN_ROWS
from ...support.device.ops import (
    bf2_f32,
    rsrc,
    traced,
)
from ...support.device.ranks import (
    sum_partials,
)
from ...support.device.sync import (
    preg,
)


@traced
def ffn_reduce_defs(k4ctx):
    """The FFN all-reduce of a down task's rows."""
    AG_RS = k4ctx["AG_RS"]
    SY = k4ctx["SY"]
    W = k4ctx["W"]
    ar_out = k4ctx["ar_out"]
    mb_poll = k4ctx["mb_poll"]
    mb_put_bf = k4ctx["mb_put_bf"]
    misc = k4ctx["misc"]
    peer_addr = k4ctx["peer_addr"]
    push_rows = k4ctx["push_rows"]
    rank = k4ctx["rank"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]

    def ffn_finish(t):
        """Misc (row group t's bf16 partial rows, token k's DN_ROWS from k DN_ROWS)
        -> every peer -> rank-ordered sum -> ar_out."""
        # every token's rows: reduce threads take token tid // 16, pair tid % 16
        row_tok = tid // (DN_ROWS // 2)
        row_pair = tid % (DN_ROWS // 2)
        if const_expr(W > 1):  # push bf16 partial rows: wave w -> peer w
            # (with AG_RS to the row group's owner rank only)
            push_rows("ffn", t * DN_ROWS, DN_ROWS, t % W if AG_RS else None)
            gpu.barrier()
        if tid < DN_ROWS // 2 * tokens:
            row = t * DN_ROWS + row_pair * 2
            if const_expr(W == 1):
                t0 = fx.ptr_load(misc + tid * 2)
                t1 = fx.ptr_load(misc + (tid * 2 + 1))
            else:
                own = preg(sym, SY["ffn"], "ffn")

                def rank_sum():
                    return sum_partials(
                        mb_poll,
                        own,
                        lambda src: ((src * MAX_TOKENS + row_tok) * HIDDEN + row) // 2,
                        W,
                    )

                if const_expr(AG_RS):
                    t0 = fx.Float32(0.0)
                    t1 = fx.Float32(0.0)
                    if t % W == rank:  # the owner: sum, gather to every rank
                        t0, t1 = rank_sum()
                        for w in range_constexpr(W):
                            mb_put_bf(
                                preg(peer_addr(w), SY["ffn_ag"], "ffn_ag"),
                                row_tok * HIDDEN + row,
                                [t0, t1],
                            )
                    else:  # the owner's sums (bf16, as ar_out stores them)
                        got = mb_poll(
                            [
                                (
                                    preg(sym, SY["ffn_ag"], "ffn_ag"),
                                    (row_tok * HIDDEN + row) // 2,
                                    1,
                                )
                            ]
                        )
                        t0, t1 = bf2_f32(got[0][0])
                else:
                    t0, t1 = rank_sum()
            bo.buffer_store(
                fx.Vector.from_elements([t0, t1], fx.Float32).to(fx.BFloat16),
                rsrc(ar_out),
                row_tok * HIDDEN + row,
            )

    return {"ffn_finish": ffn_finish}
