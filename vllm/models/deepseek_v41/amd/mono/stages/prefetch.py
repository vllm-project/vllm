# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tuned builds past one token tile (``RouteShape.shared_v2``): the shared
expert's gate / up rows pulled into each
consumer XCD's L2 by the CTAs the FFN seam leaves idle (no slice, no norm task),
while the seam runs -- HBM is idle until the routing lands, and the shared
units otherwise stream their 160 KB a CTA cold, behind which their ug start
waited. A unit (task, token tile) runs on CTA ``shared0`` + task + tasks x
tile, so every tile of a task is on the XCD of CTA ``shared0`` + task: each
task's lines are touched once, on that XCD. The loaded words are folded and
discarded: no value of the layer depends on them (the single-tile shapes'
equivalent is the attention stage's ``touch_plan``)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from vllm.models.deepseek_v41.amd.mono.common.ops import fresh, rsrc, traced
from vllm.models.deepseek_v41.amd.mono.common.plan import BLOCKS, THREADS
from vllm.models.deepseek_v41.amd.mono.stages.gemv import ROWS, token_tiles
from vllm.models.deepseek_v41.amd.mono.stages.seam import HIDDEN

XCDS = 8  # CTA b runs on XCD b % 8
LINE = 64  # bytes one touch (a dword load) pulls into L2
SH_ROW = HIDDEN  # a shared gate / up e4m3 row, bytes
SH_SROW = HIDDEN // 32  # its E8M0 scale row (32 rows a scale row), bytes


def shared_touch_plan(s, slices, rs):
    """The idle CTAs [slices, slices + s) and [slices + 2 s, BLOCKS) of a
    multi-tile shape, by XCD: (k-th, K) per CTA as Python tables, the tasks of
    each XCD. None: not ``shared_v2``, or an XCD without an idle CTA."""
    if not rs.shared_v2:
        return None
    idle = [b for b in range(slices, BLOCKS) if not (slices + s <= b < slices + 2 * s)]
    per = [[b for b in idle if b % XCDS == r] for r in range(XCDS)]
    if any(len(v) == 0 for v in per):
        return None
    tasks = rs.dims.shared_tasks
    tiles = len(token_tiles(s))
    xcd = [rs.shared_task_cta(t, 0) % XCDS for t in range(tasks)]
    # every tile's unit of a task on the task's XCD
    for t in range(tasks):
        for j in range(tiles):
            assert rs.shared_task_cta(t, j) % XCDS == xcd[t]
    xtasks = [[t for t in range(tasks) if xcd[t] == r] for r in range(XCDS)]
    # XCD r's tasks: unit p_r + 8 m' of a tile, its per consecutive tasks
    per_u = rs.shared_per_unit
    p_r = [(r - rs.shared0) % XCDS for r in range(XCDS)]
    for r in range(XCDS):
        assert xtasks[r] == [
            per_u * (p_r[r] + XCDS * (m // per_u)) + m % per_u
            for m in range(len(xtasks[r]))
        ]
    return {
        "per_u": per_u,
        "p_r": p_r,
        "n_r": [len(v) for v in xtasks],
        "idle": idle,
        "k": {b: per[b % XCDS].index(b) for b in idle},
        "K": [len(v) for v in per],
        "xtasks": xtasks,
        "per_xcd": max(len(v) for v in xtasks),
        "sh": rs.dims.sh_inter,
        "lo2": slices + 2 * s,
        "hi1": slices + s,
        "slices": slices,
    }


def _sel(r, table):
    """table[r] for traced r in [0, len(table)) (a select chain)."""
    v = fx.Int32(table[0])
    for i in range(1, len(table)):
        v = (r == i).select(fx.Int32(table[i]), v)
    return v


def _touch(base, byte, live):
    nb = live.select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    return fx.Int32(bo.buffer_load(rsrc(base, nb), byte // 4, vec_width=1, dtype=T.i32))


@traced
def stage_shared_touch(tid, bid, sgu, sgu_s, plan):
    """This idle CTA's share of its XCD's shared tasks' gate / up lines and
    their scale lines (``shared_touch_plan``)."""
    r = bid % XCDS
    # k-th idle CTA of K on this XCD: the first range's by (bid - slices) / 8,
    # the second's past the first range's on this XCD
    n1 = [
        len([b for b in plan["idle"] if b < plan["hi1"] and b % XCDS == q])
        for q in range(XCDS)
    ]
    lo2r = [plan["lo2"] + (q - plan["lo2"]) % XCDS for q in range(XCDS)]
    in1 = bid < plan["hi1"]
    k = in1.select(
        (bid - plan["slices"]) // XCDS, _sel(r, n1) + (bid - _sel(r, lo2r)) // XCDS
    )
    kk = _sel(r, plan["K"])
    sh = plan["sh"]
    per_u = plan["per_u"]
    p_r = _sel(r, plan["p_r"])
    n_r = _sel(r, plan["n_r"])
    task_lines = 2 * ROWS * SH_ROW // LINE  # gate + up rows of a task
    total = plan["per_xcd"] * task_lines
    total_r = n_r * task_lines
    kmin = min(plan["K"])
    acc = fx.Int32(0)
    for i in range_constexpr(-(-total // (THREADS * kmin))):
        ln = tid + THREADS * (k + kk * i)
        live = ln < total_r
        ln = fx.min(ln, total_r - 1)
        m = ln // task_lines
        rem = ln - m * task_lines
        half = rem // (task_lines // 2)
        rem = rem - half * (task_lines // 2)
        row = rem // (SH_ROW // LINE)
        chunk = rem - row * (SH_ROW // LINE)
        task = per_u * (p_r + XCDS * (m // per_u)) + m % per_u
        byte = (half * sh + task * ROWS + row) * SH_ROW + chunk * LINE
        acc = acc ^ _touch(sgu, byte, live)
    # the scale lines: 3 a task half (a 160-byte scale row), on the XCD's last
    q = fx.min(tid, n_r * 6 - 1)
    m = q // 6
    hs = (q - m * 6) // 3
    task = per_u * (p_r + XCDS * (m // per_u)) + m % per_u
    sbyte = ((hs * sh + task * ROWS) // 32) * SH_SROW + (q % 3) * LINE
    acc = acc ^ _touch(
        sgu_s,
        fx.min(sbyte, (2 * sh // 32) * SH_SROW - 4),
        (k == kk - 1) & (tid < n_r * 6),
    )
    _ = fresh(acc)
