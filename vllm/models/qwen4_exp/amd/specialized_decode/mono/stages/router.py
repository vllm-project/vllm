# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE block's router and shared-expert gate on the block-input rows in
LDS ``xl``: bf16 router logits -> RLOG (16 experts a task), and a token's
sigmoid(bf16(x . w_sg)) -> SGATE."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    bf_hi,
    bf_lo,
    row_sum,
    rsrc,
    traced,
    wave_sum,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import WAVES
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import gemv
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.moe import (
    rows_at,
    sigmoid,
    tiled,
    tok,
)

H, E, ROWS = L.HIDDEN, L.E, L.ROWS
KT = L.TOPK + 1  # picks a token: the top-k + the fused shared expert
XROW = H + gemv.LDS_PAD


def router_loads(c, i):
    return gemv.loads(c["lane"], c["wave"], c["args"]["w_gate"], i * ROWS, H)


def stage_router(c, i, ops):
    """Experts 16 i ..: bf16 logits of the block-input rows in LDS -> RLOG."""
    lane, wave = c["lane"], c["wave"]
    tiled(
        c,
        lambda t0, n: gemv.mfmas(
            lane, wave, rows_at(c["xl"], XROW, t0), XROW, H, n, ops
        ),
        lambda t0, n: _router_out(c, i, t0, n),
    )


@traced
def _router_out(c, i, t0, n):
    tid = c["tid"]
    if tid < (ROWS // 2) * n:
        r = 2 * (tid % (ROWS // 2))
        j = tid // (ROWS // 2)
        vs = [bf16_round(row_sum(c["red"], r + d, j)) for d in range(2)]
        c["put_bf"](c["rlog"], tok(t0, j) * E + i * ROWS + r, vs)
    gpu.barrier()


@traced
def stage_sgate(c, t):
    """Token t's shared-expert weight: sigmoid(bf16(x . w_sg)) in fp32 (the
    top-k kernel's fused sigmoid on the combined gate's bf16 logit) -> SGATE."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    n = H // 8
    kk = fx.min(tid, n - 1) * 8
    xw = fx.Vector(
        fx.ptr_load(
            c["xl"] + (t * XROW + kk) // 2,
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    ww = fx.Vector(bo.buffer_load(rsrc(a["w_sg"]), kk // 2, vec_width=4, dtype=T.i32))
    acc = fx.Float32(0.0)
    for d in range_constexpr(4):
        acc = acc + bf_lo(xw[d]) * bf_lo(ww[d]) + bf_hi(xw[d]) * bf_hi(ww[d])
    acc = wave_sum((tid < n).select(acc, fx.Float32(0.0)))
    if lane == 0:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    if tid == 0:
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, WAVES):
            tot = tot + fx.ptr_load(red + w)
        c["put"](c["sgate"], t, sigmoid(bf16_round(tot)))
    gpu.barrier()
