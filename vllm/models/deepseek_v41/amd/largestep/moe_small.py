# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The small launches around the gfx942 large-step stages.

`build_quant_h`: bf16 rows to MXFP8 FNUZ per 32 columns, the rule of
`moe_host.quant_rows`. `build_combine`: each token's 6 weighted rows summed
in f32 in top-k order, to bf16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.attention.device import kernel_symbol
from vllm.models.deepseek_v41.amd.mono.common.mx import FP8_MAX, code_ceil, rcp_pow2
from vllm.models.deepseek_v41.amd.mono.common.ops import (
    butterfly,
    fp8_pack4,
    lane_gather,
    rsrc,
)

from .moe_host import HIDDEN, TOPK

# The default stream of a launch: the current one (`mono/layer.py`).
_STREAM = fx.Stream(None)
THREADS = 256


def build_quant_h(inter: int, even_odd: bool = True):
    """MXFP8 of h [P, I]: grid (P * I / 32 / 4,), 4 waves a CTA, a wave a
    32-column block. `even_odd`: each 8-value chunk even K first, for the
    routed experts' FP4 weights (`fp4x8_fnuz` order); False keeps natural
    order, for the shared expert's FP8 copies (`linear_copy`)."""
    groups = inter // 32
    name = kernel_symbol("dsv41_ls_quant_h", i=inter, eo=int(even_odd))

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(h: Int64, hq: Int64, hqs: Int64, n_blocks: Int32):
        tid = fx.thread_idx.x
        lane = tid % 64
        blk = fx.block_idx.x * 4 + tid // 64
        if blk < n_blocks:
            p = blk // groups
            gi = blk % groups
            col = gi * 32 + fx.min(lane, 31)
            v = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(rsrc(h), p * inter + col, vec_width=1, dtype=T.bf16)
                )
            )
            v = (lane < 32).select(v, fx.Float32(0.0))
            amax = fx.max(
                butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(1e-10)
            )
            # code_ceil of amax / 448 in OCP terms; the FNUZ code is one more
            # (`mono.common.mx`: FP8_MAX is 224 on gfx942, so this is
            # already the FNUZ code).
            code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
            q = fx.max(
                fx.min(v * rcp_pow2(code), fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX)
            )
            # Pack: lane l (an 8-value chunk's start, l % 8 == 0) writes the
            # chunk's even values then its odd values as two dwords.
            got = [
                lane_gather(q.bitcast(fx.Int32), fx.min(lane + i, 63)).bitcast(
                    fx.Float32
                )
                for i in range(8)
            ]
            if (lane < 32) & (lane % 8 == 0):
                if const_expr(even_odd):
                    lo = fp8_pack4(got[0], got[2], got[4], got[6])
                    hi = fp8_pack4(got[1], got[3], got[5], got[7])
                else:
                    lo = fp8_pack4(got[0], got[1], got[2], got[3])
                    hi = fp8_pack4(got[4], got[5], got[6], got[7])
                at = (p * inter + gi * 32 + lane) // 4
                bo.buffer_store(lo, rsrc(hq), at)
                bo.buffer_store(hi, rsrc(hq), at + 1)
            if lane == 0:
                # one byte a code: 4 neighbouring blocks share a dword, so
                # the byte is written with a byte store
                bo.buffer_store(code.to(fx.Int8), rsrc(hqs), p * groups + gi)
        return

    @flyc.jit
    def launch(
        h: Int64,
        hq: Int64,
        hqs: Int64,
        n_blocks: Int32,
        stream: fx.Stream = _STREAM,
    ):
        kernel(h, hq, hqs, n_blocks).launch(
            grid=((n_blocks + 3) // 4,), block=(THREADS,), stream=stream
        )

    return launch


def build_combine():
    """out[t] = bf16(sum over k of y[6 t + k]) in f32, top-k order: grid
    (S * 5120 / 8 / THREADS,), a thread 8 columns of one token."""
    name = kernel_symbol("dsv41_ls_combine", h=HIDDEN)
    chunks = HIDDEN // 8

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(y: Int64, out: Int64, s: Int32):
        i = fx.block_idx.x * THREADS + fx.thread_idx.x
        if i < s * chunks:
            t = i // chunks
            c8 = i % chunks
            acc = [fx.Float32(0.0) for _ in range(8)]
            for k in range_constexpr(TOPK):
                v = fx.Vector(
                    bo.buffer_load(
                        rsrc(y),
                        ((t * TOPK + k) * HIDDEN + c8 * 8) // 2,
                        vec_width=4,
                        dtype=T.i32,
                    )
                ).bitcast(fx.BFloat16)
                for c in range_constexpr(8):
                    acc[c] = acc[c] + fx.Float32(v[c])
            res = (
                fx.Vector.from_elements(acc, fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int32)
            )
            bo.buffer_store(res, rsrc(out), (t * HIDDEN + c8 * 8) // 2)
        return

    @flyc.jit
    def launch(y: Int64, out: Int64, s: Int32, stream: fx.Stream = _STREAM):
        kernel(y, out, s).launch(
            grid=((s * chunks + THREADS - 1) // THREADS,),
            block=(THREADS,),
            stream=stream,
        )

    return launch
