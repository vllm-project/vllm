# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 1: one Muse Spark layer's o_proj and the push half of its TP
all-reduce (``ar``), on every CTA.

    o_proj      8192 x 1024 bf16 a rank (RowParallel: K sharded), 512 tasks of
                16 output rows; CTA b runs tasks b and b + BLOCKS, both tasks'
                weights in flight before the first MFMA (K = 1024 = 8 chunks of
                128: one chunk a wave, ``gemv.loads``)
    core        the gated attention core [S, 1024] (stock attention wrote it
                before the launch: plain loads) -> LDS ``xl``, rows padded,
                pad rows stored as zeros (``select``: a NaN in a pad row's core
                never reaches the partials). By ``K2Build.core``: "tensor"
                loads vLLM's gated core after the weights (Phase A); "gated"
                (``attention/gate.py``, Phase B2) loads the attention output and
                the o_gate rows *before* the weights (``core_loads``) and gates
                them into ``xl`` (DESIGN_BC 6.2, rule 8); "split"
                (``attention/split.py``, Phase C) copies the in-kernel
                combine's gated rows ``xo`` behind their ``crdy`` flags
                (DESIGN_BC 7.4), the split and combine having run before the
                weights were issued (7.2, ``layer.split_roles``)
    partial     ``bf16(sum_k core[t, k] w_o[n, k])``: the waves' accumulators
                summed in wave order (``row_sum``), rounded once -> every
                rank's ``ar`` region, this rank's slot ``(rank MT + t) 8192 +
                n`` (system scope, a ``put_bf`` a peer)

The reduce half (the 8 partials in rank order) is the seam's
(``stages/seam.py``). ``ar`` is sized by MAX_TOKENS rows, so its offsets do
not depend on the launch's width; a step's parity half keeps a rank one step
ahead off the partials a peer has not read yet.

LDS: the stage struct ``lds_struct(S)`` under the stage key ``"back"``
(``c["lds"]["back"]["xl"]``) and ``red``. ``load_core`` and ``stage_o`` end
with a barrier (the hand-over rule).

Adapted from the Kimi-K3 AMD specialized decode (vLLM #60683,
``kimi_k3/amd/specialized_decode/mono/attention/back.py``: ``o_loads``,
``load_core``, ``stage_o``) with Muse's K = 1024, the width-independent
``MAX_TOKENS`` stride of the AR slot and the pad rows.
"""

import functools

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (
    gate,
    split,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf16_round,
    ld_pad,
    row_sum,
    rsrc,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (
    THREADS,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    CORE,
    HIDDEN,
    K2_CORES,
    MAX_TOKENS,
    O_TASKS,
    ROWS,
    TP,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import gemv

OK = CORE  # o_proj's K a rank: 16 heads x 64
OKW = OK // 8  # dwordx4 a core row (8 bf16 each)
XL_ROW = OK + gemv.LDS_PAD  # bf16 a padded LDS core row (1032)
assert OK % (gemv.KCH * 8) == 0  # one 128-wide K chunk a wave, no masked tail
assert O_TASKS * ROWS == HIDDEN
assert gate.XL_ROW == XL_ROW  # the gated core load writes these rows
assert split.XL_ROW == XL_ROW and split.OKW == OKW  # ... and the split's
# the cores whose load this module has (K2Build.core; the split's is
# attention/split.py's)
CORES = ("tensor", "gated", "split")


@functools.cache
def lds_struct(s: int):
    """The stage's LDS: the core rows, ``s`` x (1024 + 8) bf16 (16,512 B at
    s = 8). Cached: one struct type a width."""

    @fx.struct
    class BackLds:
        xl: fx.Array[fx.Int32, s * XL_ROW // 2, 16]

    return BackLds


def o_loads(c, task):
    """o_proj rows 16 task .., this wave's K chunk (one of 8): a lane 64 B of
    its row (4 dwordx4, nontemporal). Issued before any wait of the CTA."""
    w = c["args"]["w_o"]
    return gemv.loads(c["lane"], c["wave"], w, task * ROWS, OK)


def _check_core(core: str) -> None:
    if core not in K2_CORES:
        raise ValueError(f"K2 core {core!r} (one of {K2_CORES})")
    if core not in CORES:
        raise NotImplementedError(f"K2 core {core!r}: no core load yet")


def core_loads(c, core: str = "tensor"):
    """The core's own operands, the CTA's first loads (before ``o_loads``):
    "gated" -> ``gate.core_loads`` (rule 8: its math then waits on them, never
    on the weights), "tensor" -> None (Phase A loads the core after the
    weights, in ``load_core``), "split" -> None (its operands are the
    combine's rows, loaded behind ``crdy`` in ``load_core``; the split and
    combine stages ran before the weights). Scheduling barriers on both sides
    keep them in place, the weights after (the context's scalar words load at
    ``settle``, past the weights)."""
    _check_core(core)
    if core in ("tensor", "split"):
        return None
    rocdl.sched_barrier(0)
    pre = gate.core_loads(c)
    rocdl.sched_barrier(0)
    return pre


def load_core(c, core: str = "tensor", pre=None):
    """The core -> LDS ``xl`` by ``K2Build.core`` (``pre``: ``core_loads``'
    result). Ends with the hand-over barrier."""
    _check_core(core)
    if core == "tensor":
        _load_tensor_core(c)
        return
    if core == "split":  # polls crdy with only the o_proj weights in flight
        split.load_core(c)
        return
    if pre is None:  # loading them here would put them behind the weights
        raise ValueError(f"K2 core {core!r}: load_core needs core_loads' result")
    # the o_proj weights stay issued above the gate's math
    rocdl.sched_barrier(0)
    gate.load_core_gated(c, pre)


@traced
def _load_tensor_core(c):
    """The core rows (S x 1024 bf16) -> LDS ``xl``, rows padded; a pad row is
    stored as zeros (``select``, never a multiply)."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    xl = c["lds"]["back"]["xl"]
    words = s * OKW
    for i in range_constexpr((words + THREADS - 1) // THREADS):
        e = fx.min(tid + THREADS * i, words - 1)
        v = fx.Vector(bo.buffer_load(rsrc(a["core"]), e * 4, vec_width=4, dtype=T.i32))
        t = e // OKW
        k = e % OKW * 8
        dead = ld_pad(a["pad"], t)
        v = fx.Vector.from_elements(
            [dead.select(fx.Int32(0), v[d]) for d in range(4)], fx.Int32
        )
        if tid + THREADS * i < words:
            fx.ptr_store(v, xl + (t * XL_ROW + k) // 2)
    gpu.barrier()


@traced
def stage_o(c, task, ops):
    """Rows 16 task ..: GEMV -> bf16 partials -> every rank's ``ar`` region,
    this rank's slot (thread: token, row pair; a ``put_bf`` a peer)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    xl = c["lds"]["back"]["xl"]
    acc = gemv.mfmas(lane, wave, xl, XL_ROW, OK, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        t = tid // (ROWS // 2)
        rp = tid % (ROWS // 2)
        v0 = bf16_round(row_sum(red, 2 * rp, t))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, t))
        # element index (put_bf): 2 x spec.ar_pair(rank, t, n)
        at = (c["rank"] * MAX_TOKENS + t) * HIDDEN + task * ROWS + 2 * rp
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["ar"][p], at, [v0, v1])
    gpu.barrier()
