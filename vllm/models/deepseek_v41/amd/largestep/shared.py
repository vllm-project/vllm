# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared expert of a large decode step on gfx942, as GEMMs over tiles of
16 tokens on the mono kernels' gfx942 FP8 copies (`mono.weights942.linear_copy`):
gate_up with vLLM's clamped SwiGLU, then down.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.deepseek_v41.amd.mono.attention.device import kernel_symbol
from vllm.models.deepseek_v41.amd.mono.common.gfx942 import (
    mfma_fp8_scaled,
    pair_i64,
    pin_acc,
)
from vllm.models.deepseek_v41.amd.mono.common.ops import (
    CM_NT,
    bf16_round,
    rsrc,
)
from vllm.models.deepseek_v41.amd.mono.common.plan import BLOCKS
from vllm.models.deepseek_v41.amd.mono.stages.dims import MOE_INTER

from .moe_host import HIDDEN, LIMIT

THREADS = 512
WAVES = THREADS // 64
_STREAM = fx.Stream(None)


def _step_load(w, ws, kp, rg, st, lane):
    """K step `st` (128 wide) of rows 16 rg .. of a `linear_copy` weight with
    K padded to kp: the lane's two 16-byte halves (blocks 0-1, then 2-3,
    row lane % 16, chunk lane / 16) and the step's four FNUZ codes of the
    rows' 32-row block, one dword. As `mono.stages.gemv._fp8_step_load_942`."""
    base = (rg * 16 * kp + 2048 * st) // 4 + 4 * lane
    lo = fx.Vector(
        bo.buffer_load(rsrc(w), base, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    hi = fx.Vector(
        bo.buffer_load(
            rsrc(w), base + 256, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
        )
    )
    wsi = (rg // 2) * (kp // 32) + st * 4
    word = fx.Int32(bo.buffer_load(rsrc(ws), wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, word


def _x_block(xb, xs, tok, k, b, lane):
    """The lane's B operand of K block b of token `tok`'s MXFP8 row of width
    k: 8 bytes (K 32 b + 8 (lane / 16) .. + 8) and the block's code."""
    j = lane // 16
    v = fx.Vector(
        bo.buffer_load(
            rsrc(xb), (tok * k + 32 * b + 8 * j) // 4, vec_width=2, dtype=T.i32
        )
    )
    cw = fx.Int32(
        bo.buffer_load(rsrc(xs), (tok * (k // 32) + b) // 4, vec_width=1, dtype=T.i32)
    )
    code = (cw >> (8 * ((tok * (k // 32) + b) % 4))) & 0xFF
    return pair_i64(v[0], v[1]), code


def _step(acc, ops, xs4, b0):
    """Four FP8 MFMAs of one 128-wide K step: each 32-wide block times its
    code factor (weight code of the 32 x 32 block + activation code)."""
    lo, hi, word = ops
    for b in range_constexpr(4):
        w = lo if b < 2 else hi
        a = pair_i64(w[2 * (b % 2)], w[2 * (b % 2) + 1])
        bx, cx = xs4[b0 + b]
        acc = mfma_fp8_scaled(acc, a, bx, ((word >> (8 * b)) & 0xFF) + cx)
    return pin_acc(acc)


def build_shared_gate_up(tp: int):
    inter = MOE_INTER // tp
    kp = HIDDEN  # 5120 is a multiple of 128
    steps = kp // 128  # 40
    steps_wave = steps // WAVES  # 5
    col_blocks = inter // 16
    name = kernel_symbol("dsv41_ls_shared_gu", tp=tp)

    @fx.struct
    class _Lds:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(sgu: Int64, sgu_s: Int64, xb: Int64, xs: Int64, s: Int32, sh: Int64):
        lds = fx.SharedAllocator().allocate(_Lds).peek()
        red = lds.red.ptr
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        tok_tiles = (s + 15) // 16
        n_units = tok_tiles * col_blocks
        for unit in range(fx.block_idx.x, n_units, BLOCKS):
            tt = unit // col_blocks
            cb = unit % col_blocks
            tok = fx.min(tt * 16 + lane % 16, s - 1)
            st0 = wave * steps_wave
            xs4 = [
                _x_block(xb, xs, tok, HIDDEN, 4 * st0 + b, lane)
                for b in range(4 * steps_wave)
            ]
            accs = []
            # gate rows cb 16 .. + 15 are tile cb; up rows I + 16 cb .. are
            # tile inter / 16 + cb of the [2 I, K] copy
            for rg in (cb, col_blocks + cb):
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                for i in range_constexpr(steps_wave):
                    ops = _step_load(sgu, sgu_s, kp, rg, st0 + i, lane)
                    acc = _step(acc, ops, xs4, 4 * i)
                accs.append(acc)
            for t in range_constexpr(2):
                fx.ptr_store(accs[t], red + ((wave * 2 + t) * 64 + lane) * 4)
            gpu.barrier()
            if tid < 256:
                pk = tid % 16
                r = tid // 16
                at = (16 * (r // 4) + pk) * 4 + r % 4
                gu = []
                for t in range_constexpr(2):
                    acc = fx.Float32(0.0)
                    for wv in range_constexpr(WAVES):
                        acc = acc + fx.ptr_load(red + (wv * 2 + t) * 256 + at)
                    gu.append(bf16_round(acc))
                g = bf16_round(fx.min(gu[0], fx.Float32(LIMIT)))
                u = fx.max(fx.min(gu[1], fx.Float32(LIMIT)), fx.Float32(-LIMIT))
                act = bf16_round(g / (fx.Float32(1.0) + fx.Float32(fx.math.exp(-g))))
                t_ = tt * 16 + pk
                if t_ < s:
                    bo.buffer_store(
                        (act * u).to(fx.BFloat16), rsrc(sh), t_ * inter + 16 * cb + r
                    )
            gpu.barrier()
        return

    @flyc.jit
    def launch(
        sgu: Int64,
        sgu_s: Int64,
        xb: Int64,
        xs: Int64,
        s: Int32,
        sh: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(sgu, sgu_s, xb, xs, s, sh).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


def build_shared_down(tp: int):
    inter = MOE_INTER // tp
    kp = -(-inter // 128) * 128  # 384 at TP8, 640 at TP4
    steps = kp // 128
    name = kernel_symbol("dsv41_ls_shared_down", tp=tp)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(sw2: Int64, sw2_s: Int64, hb: Int64, hs: Int64, s: Int32, out: Int64):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        tok_tiles = (s + 15) // 16
        n_units = tok_tiles * (HIDDEN // 16)
        for unit in range(fx.block_idx.x * WAVES + wave, n_units, BLOCKS * WAVES):
            tt = unit // (HIDDEN // 16)
            rg = unit % (HIDDEN // 16)
            tok = fx.min(tt * 16 + lane % 16, s - 1)
            # blocks past the real K (inter) are zero weights; their
            # activation bytes are zeroed and their factor is 0 (code 127)
            xs4 = []
            for b in range_constexpr(4 * steps):
                if b < inter // 32:
                    xs4.append(_x_block(hb, hs, tok, inter, b, lane))
                else:
                    xs4.append((pair_i64(fx.Int32(0), fx.Int32(0)), fx.Int32(127)))
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(steps):
                ops = _step_load(sw2, sw2_s, kp, rg, st, lane)
                acc = _step(acc, ops, xs4, 4 * st)
            t_ = tt * 16 + lane % 16
            if t_ < s:
                row0 = rg * 16 + 4 * (lane // 16)
                for q in range_constexpr(4):
                    bo.buffer_store(
                        acc[q].to(fx.BFloat16), rsrc(out), t_ * HIDDEN + row0 + q
                    )
        return

    @flyc.jit
    def launch(
        sw2: Int64,
        sw2_s: Int64,
        hb: Int64,
        hs: Int64,
        s: Int32,
        out: Int64,
        stream: fx.Stream = _STREAM,
    ):
        kernel(sw2, sw2_s, hb, hs, s, out).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
