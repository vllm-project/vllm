# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MoE block's input rows to MXFP8 FNUZ, in both orders from one read of
x: each 8-value chunk even K first for the routed FP4 weights, and natural
order for the shared expert's FP8 copies.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.attention.device import kernel_symbol
from vllm.models.deepseek_v41.amd.mono.common.mx import FP8_MAX, code_ceil, rcp_pow2
from vllm.models.deepseek_v41.amd.mono.common.ops import (
    butterfly,
    fp8_pack4,
    lane_gather,
    rsrc,
)

from .moe_host import HIDDEN

THREADS = 256
_STREAM = fx.Stream(None)
GROUPS = HIDDEN // 32


def build_quant_x():
    name = kernel_symbol("dsv41_ls_quant_x", h=HIDDEN)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(x: Int64, xb: Int64, xs: Int64, xbn: Int64, xsn: Int64, s: Int32):
        tid = fx.thread_idx.x
        lane = tid % 64
        blk = fx.block_idx.x * 4 + tid // 64
        if blk < s * GROUPS:
            t = blk // GROUPS
            gi = blk % GROUPS
            col = gi * 32 + fx.min(lane, 31)
            v = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(rsrc(x), t * HIDDEN + col, vec_width=1, dtype=T.bf16)
                )
            )
            v = (lane < 32).select(v, fx.Float32(0.0))
            amax = fx.max(
                butterfly(abs(v), (1, 2, 4, 8, 16), fx.max), fx.Float32(1e-10)
            )
            # FP8_MAX is 224 on gfx942, so this is the FNUZ code directly
            # (`mono.common.mx`).
            code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
            q = fx.max(
                fx.min(v * rcp_pow2(code), fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX)
            )
            got = [
                lane_gather(q.bitcast(fx.Int32), fx.min(lane + i, 63)).bitcast(
                    fx.Float32
                )
                for i in range(8)
            ]
            if (lane < 32) & (lane % 8 == 0):
                at = (t * HIDDEN + gi * 32 + lane) // 4
                eo_lo = fp8_pack4(got[0], got[2], got[4], got[6])
                eo_hi = fp8_pack4(got[1], got[3], got[5], got[7])
                bo.buffer_store(eo_lo, rsrc(xb), at)
                bo.buffer_store(eo_hi, rsrc(xb), at + 1)
                nat_lo = fp8_pack4(got[0], got[1], got[2], got[3])
                nat_hi = fp8_pack4(got[4], got[5], got[6], got[7])
                bo.buffer_store(nat_lo, rsrc(xbn), at)
                bo.buffer_store(nat_hi, rsrc(xbn), at + 1)
            if lane == 0:
                bo.buffer_store(code.to(fx.Int8), rsrc(xs), t * GROUPS + gi)
                bo.buffer_store(code.to(fx.Int8), rsrc(xsn), t * GROUPS + gi)
        return

    @flyc.jit
    def launch(
        x: Int64,
        xb: Int64,
        xs: Int64,
        xbn: Int64,
        xsn: Int64,
        s: Int32,
        stream: fx.Stream = _STREAM,
    ):
        kernel(x, xb, xs, xbn, xsn, s).launch(
            grid=((s * GROUPS + 3) // 4,), block=(THREADS,), stream=stream
        )

    return launch
