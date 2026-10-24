# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN block's input projections on the block-input rows in LDS ``xl``:
in_proj_qkvz through MXFP4 (two 16-row groups a CTA, bid and bid + 256) ->
bf16 PROJ, in_proj_ba as 3 row groups x 2 K halves on the unquantized rows
(CTAs ``BA0`` ..) -> fp32 partials the GDN tasks sum (``ba_sums``)."""

from flydsl.expr import gpu

from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    row_sum,
    traced,
)
from vllm.models.qwen4_exp.amd.specialized_decode.mono.common.plan import BLOCKS
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import gdn as G
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages import gemv
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.gdn import NVCH
from vllm.models.qwen4_exp.amd.specialized_decode.mono.stages.moe import (
    fp4_gemv,
    fp4_gemv_loads,
    rows_at,
    tiled,
    tok,
)

H, ROWS = L.HIDDEN, L.ROWS
SL = L.CONV_W - 1  # conv state entries (no spec decode)
BA_PARTS = 2
BA_K = H // BA_PARTS
BA0 = 160  # in_proj_ba's first CTA
XROW = H + gemv.LDS_PAD


def ba_tasks(lay):
    return lay.BA // ROWS * BA_PARTS


def gdn_tasks_of(lay):
    """GDN tasks a token."""
    return lay.NV * NVCH


def gdn_offset(tokens: int, tp: int) -> int:
    """The first GDN CTA: the high CTAs while a single round fits."""
    n = tokens * gdn_tasks_of(L.Layout(tp))
    return BLOCKS - n if n < BLOCKS else 0


def qkvz_loads(c, rg):
    a = c["args"]
    return fp4_gemv_loads(
        c["lane"],
        c["wave"],
        a["w_qkvz"],
        a["w_qkvzs"],
        rg,
        H,
        c["lay"].QKVZ,
        c["shuffled"],
    )


def stage_qkvz(c, rg, ops):
    """in_proj_qkvz rows 16 rg .. on the MXFP4-exact x rows -> bf16 PROJ."""
    lane, wave = c["lane"], c["wave"]
    tiled(
        c,
        lambda t0, n: fp4_gemv(
            lane,
            wave,
            rows_at(c["xl"], XROW, t0),
            XROW,
            H,
            n,
            rg,
            ops,
            c["shuffled"],
        ),
        lambda t0, n: _qkvz_out(c, rg, t0, n),
    )


@traced
def _qkvz_out(c, rg, t0, n):
    tid = c["tid"]
    if tid < ROWS * n:
        r = tid % ROWS
        j = tid // ROWS
        v = bf16_round(row_sum(c["red"], r, j))
        c["put"](c["proj"], tok(t0, j) * c["lay"].QKVZ + rg * ROWS + r, v)
    gpu.barrier()


def ba_of(c, t, h):
    """Head h's b and a of token t: the sums of in_proj_ba's K halves (bf16)."""
    return G.ba_sums(c, t, h, BA_PARTS)
