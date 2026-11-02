# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2a ``attn_post``: one Kimi-K3 KDA layer from the gated core output to the
MoE's input row -- o_proj, its TP all-reduce inside the kernel, and the MLP
AttnRes seam.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``:

    gemv CTAs 56 ..   o_proj (448 x 16 rows, K = 1536 local): bf16 GEMV of the
                      core rows (plain loads: the previous launch wrote them) ->
                      bf16 partial -> pushed to every rank's AR region (system
                      scope, this rank's slot)
    attn CTAs 0 .. 55 per 128 columns: the 8 ranks' partials summed in rank
                      order (fp32 -> bf16: the custom all-reduce's result) ->
                      LDS delta; then K1's AttnRes stages with the MLP's weights:
                      prefix + delta -> prefix out, the sources' softmax mix,
                      post_attention_layernorm -> the MoE input x

The AR region is double-buffered by the step epoch's parity: a rank a step
ahead writes the other half, so no peer's unread partials are overwritten.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
    COLS,
    HIDDEN,
    LDS_PAD,
    NA,
    PROJ,
    ROWS,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ops import (
    CM_NT,
    bf16_round,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    pair_layout,
)
from vllm.models.kimi_k3.amd.specialized_decode.mono.common.ranks import sum_partials

TP = 8
OROWS = HIDDEN  # o_proj rows (full hidden, a rank's K slice)
OTASKS = OROWS // ROWS  # 448
OK = PROJ  # K = 1536 a rank
OKCH = OK // 128  # 12 chunks
GEMV0 = NA  # the first gemv CTA
GEMV_N = BLOCKS - NA  # 200

_STREAM = fx.Stream(None)


def ar_half_pairs(s):
    """One parity's AR region: ``[src rank][token][column pair]``."""
    return TP * s * HIDDEN // 2


def scratch_layout(s: int, nblocks: int) -> dict:
    ns = nblocks + 1
    pairs = {
        "part": NA * s * ns * 2,
        "wgt": s * (ns + 1),
        "msq": NA * s,
        "xrdy": NA,
    }
    return pair_layout(pairs.items())


def o_loads(c, rg):
    """Rows 16 rg .. of o_proj, this wave's K chunks (wave, wave + 8 < 12): a lane
    64 B of its row a chunk."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_o"])
    row = rg * ROWS + lane % ROWS
    g = lane // ROWS
    out = []
    for i in range(2):
        ch = fx.min(wave + WAVES * i, OKCH - 1)
        base = (row * OK + ch * 128 + g * 8) // 2
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_NT,
                    )
                )
                for q in range(4)
            ]
        )
    return out


@traced
def load_core(c):
    """The core rows (S x 1536 bf16) -> LDS, rows padded."""
    s, tid = c["S"], c["tid"]
    words = s * OK // 8
    row = OK + LDS_PAD
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(
            bo.buffer_load(rsrc(c["args"]["core"]), e * 4, vec_width=4, dtype=T.i32)
        )
        t = e // (OK // 8)
        k = e % (OK // 8) * 8
        if tid + THREADS * i < words:
            fx.ptr_store(v, c["xl"] + (t * row + k) // 2)
    gpu.barrier()


@traced
def stage_o(c, rg, ops):
    """Rows 16 rg ..: GEMV -> bf16 partials -> every rank's AR region, this
    rank's slot (thread: token, row pair; a put a peer)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    g = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    row = OK + LDS_PAD
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(2):
        ch = wave + WAVES * i
        live = ch < OKCH
        k = fx.min(ch, OKCH - 1) * 128 + g * 8
        for q in range_constexpr(4):
            xb = fx.Vector(
                fx.ptr_load(
                    c["xl"] + (t * row + k + 32 * q) // 2,
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            xb = fx.Vector.from_elements(
                [live.select(xb[d], fx.Int32(0)) for d in range(4)], fx.Int32
            )
            acc = mfma_bf16(
                ops[i][q].bitcast(fx.BFloat16), xb.bitcast(fx.BFloat16), acc
            )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, tt))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tt))
        at = (c["rank"] * s + tt) * HIDDEN + rg * ROWS + 2 * rp
        for p in range_constexpr(TP):
            c["put_bf"](c["ar_peer"][p], at, [v0, v1])
    gpu.barrier()


@traced
def stage_reduce(c, task):
    """Columns 128 task ..: the 8 ranks' partials, rank order, -> bf16 delta in
    LDS (thread: token, column pair)."""
    s, tid = c["S"], c["tid"]
    if tid < s * (COLS // 2):
        t = tid // (COLS // 2)
        cp = tid % (COLS // 2)
        col = task * COLS + 2 * cp
        lo, hi = sum_partials(
            c["poll"], c["ar_own"], lambda src: ((src * s + t) * HIDDEN + col) // 2, TP
        )
        fx.ptr_store(bf16_round(lo), c["delta_lds"] + (t * COLS + 2 * cp))
        fx.ptr_store(bf16_round(hi), c["delta_lds"] + (t * COLS + 2 * cp + 1))
    gpu.barrier()
