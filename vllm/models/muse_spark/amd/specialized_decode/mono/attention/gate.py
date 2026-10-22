# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(gated) phase 1 core load (DESIGN_BC 6, Phase B2): the attention's
output norm and gate, ``o_norm(attn) * sigmoid(og)``, computed in the core
load instead of by vLLM's ``rms_norm_gated`` launch.

Stock reference: FLA's ``layer_norm_gated_fwd_kernel``
(``vllm/third_party/flash_linear_attention/ops/kda.py``, weightless
RMS norm over a head's D = 64, ``activation="sigmoid"``, eps 1e-5), Triton
3.8 on gfx950, the instance the served launch compiles (16-byte aligned
pointers, DESIGN_BC 1.4)::

    tot  = sum(x^2)                    the served tree: leaf RN(x_2c^2) +
                                       RN(x_2c+1^2), butterfly 16, 8, 4, 2, 1
                                       over the 32 leaves (``ops.tl_sum64``)
    rstd = 1 / sqrt(tot / 64 + eps)    fma(tot, 2^-6, eps), raw v_sqrt_f32,
                                       IEEE divide
    y    = bf16((x rstd) sigmoid(g))   sigmoid = 1 / (1 + exp2(-g log2 e)),
                                       IEEE divide (``ops.ieee_sigmoid``)

Inputs (plain loads: written by earlier launches): ``attn`` bf16 [S, 1024],
the stock core's output, and the o_gate rows of ``qkvo`` bf16 [S, 2304]
(columns ``ROW_G`` ..), the stock qkv buffer (or K1's). Every CTA computes
all S x 16 heads (redundant, no exchange): pass k in 0 .. 2S - 1 gives wave w
the row (t, h) = divmod(8 k + w, 16), lane d its dim d; the gated value goes
to LDS ``xl[t, 64 h + d]`` (back's row layout, ``ds_write_b16``), a pad row
as zeros (``select``: a NaN in a pad row never reaches o_proj).

Order (rule 8, E11): ``core_loads`` is the CTA's first load, before the
o_proj weights; ``load_core_gated`` only waits for its own operands (an
in-order ``vmcnt`` leaves the weights in flight). ``o_gate`` is the per-row
gate Phase C's combine reuses.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    hw_sqrt,
    ieee_sigmoid,
    ld_i32,
    rsrc,
    tl_sum64,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    WAVES,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    CORE,
    HEAD_DIM,
    HEADS,
    QKV_ROWS,
    RMS_EPS,
    ROW_G,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import gemv

XL_ROW = CORE + gemv.LDS_PAD  # bf16 a padded LDS core row (back.XL_ROW)
INV_D = 2.0**-6  # 1 / HEAD_DIM, exact: FLA's mean is an fma with it
assert HEAD_DIM == 64 and HEADS == 2 * WAVES  # a pass: a row a wave, lane = d


def passes(s: int) -> int:
    """Rows a wave gates: S tokens x 16 heads over 8 waves."""
    return s * HEADS // WAVES


def pass_row(k: int, wave):
    """Pass k's row of wave ``wave``: (token t, Python int; head h)."""
    return k // 2, (k % 2) * WAVES + wave


def _ld_bf16(addr, i):
    """bf16 element i of the buffer at ``addr``: one 2-byte load, raw."""
    return bo.buffer_load(rsrc(addr), i, vec_width=1, dtype=T.bf16)


def _f32(v):
    return fx.BFloat16(v).to(fx.Float32)


def core_loads(c) -> dict:
    """This lane's operands of every pass, the CTA's first loads after the
    context's (before the o_proj weights), kept raw (no arithmetic on them
    here: anything computed from a load before the weights are issued would
    wait for it): the pad words (``ld_pad``'s, a word 4 rows), then ``x``
    (attn) and ``g`` (o_gate), 2S each."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    pads = [ld_i32(a["pad"], w) for w in range((s + 3) // 4)]
    rocdl.sched_barrier(0)  # pass 0's select waits for the pads, not every row
    xs, gs = [], []
    for k in range(passes(s)):
        t, h = pass_row(k, wave)
        col = h * HEAD_DIM + lane
        xs.append(_ld_bf16(a["attn"], t * CORE + col))
        gs.append(_ld_bf16(a["qkvo"], t * QKV_ROWS + ROW_G + col))
    return {"x": xs, "g": gs, "pad": pads}


def _dead(pads, t):
    """Whether row t is a pad row: ``ld_pad`` on the loaded word."""
    return ((pads[t // 4] >> (8 * (t % 4))) & 0xFF) != 0


def o_gate(x, g):
    """One row's ``rms_norm_gated`` (lane = dim): bf16(RN(x rstd)
    sigmoid(g)), returned as its f32. Every lane of the wave must be active
    (``tl_sum64``)."""
    tot = tl_sum64(x)
    q = fx.Float32(fmath.fma(tot, fx.Float32(INV_D), fx.Float32(RMS_EPS)))
    rstd = fx.Float32(1.0) / hw_sqrt(q)
    return bf16_round((x * rstd) * ieee_sigmoid(g))


def load_core_gated(c, pre) -> None:
    """``pre`` gated -> LDS ``xl`` (back's padded rows), a pad row as zeros;
    ends with the hand-over barrier."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    xl = fx.recast_iter(fx.BFloat16, c["lds"]["back"]["xl"])
    zero = fx.Float32(0.0)
    for k in range_constexpr(passes(s)):
        t, h = pass_row(k, wave)
        y = o_gate(_f32(pre["x"][k]), _f32(pre["g"][k]))
        y = fx.Float32(_dead(pre["pad"], t).select(zero, y))
        fx.ptr_store(y.to(fx.BFloat16), xl + (t * XL_ROW + h * HEAD_DIM + lane))
    gpu.barrier()
