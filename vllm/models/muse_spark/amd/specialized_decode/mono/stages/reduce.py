# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 4a ``owner``: the column-owner per-route reduce of the routed
experts' rows, the post-expert RMSNorm (its sum of squares exchanged between
the owners), the route weighting and the top-k sum -> every rank's ``moe``.

Stock (``MuseSparkTopKWeightAndReduce`` then ``fused_post_expert_finalize``,
common/ops/fused_finalize.py:10-70): ``R = all_reduce(rows.float())`` [S, 8,
4096], ``x = R (w + 1)``, ``scale = rsqrt(sum_c x^2 / 4096 + 1e-8) tw``,
``moe = bf16(sum_k x scale)``.

Here rank r owns the 512 pooled columns 512 r .. (``OWN_COLS``); its 32 owner
CTAs (j, columns 16 j .. of the 512) take every pick's 16 columns:

    R      the 8 ranks' bf16 rows of the pick (``rsx``, pushed by the down CTAs
           to the column owner only), summed in fp32 in rank order, never
           rounded (README, Numerics: the deviation)
    x      R (w_postexp + GAIN): the weight first, as stock
    sqx    the pick's sum of x^2 over the CTA's 16 columns -> every rank (one
           cross-rank hop, no rank-local one); each owner sums, for each rank
           in rank order, that rank's 32 in a fixed order (4 a lane, then the
           8 lanes): the pick's sum of squares over all 4096 columns, the same
           bits on every rank
    xs     x rsqrt(ss / 4096 + POST_EXP_EPS) tw, 0 on a pad row and on a pick
           of route weight 0 (``select``: NaN in a dead pick's rows cannot
           leak)
    moe    the 8 picks of a row summed as a fixed tree, bf16 -> every rank
           (``moe`` all-gather: rank r writes columns 512 r ..)

The tree sum and the per-pick sum of squares are bit-identical on every rank
(same values, same order), so every rank's ``moe`` row is the same.
"""

import functools

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from vllm.models.muse_spark.amd.specialized_decode.mono.common import route_lds
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (
    bf2_f32,
    butterfly,
    hw_rsq,
    ld_i32,
    ld_pad,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import (
    sum_partials,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
    uniform_addr,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import (
    GAIN,
    MOE_HIDDEN,
    OWN_COLS,
    POST_EXP_EPS,
    RED_TASKS,
    ROWS,
    TOPK,
    TP,
    rsx_pair,
    sqx_pair,
)

PICK_LANES = ROWS // 2  # threads a pick: 2 of its 16 columns each
PSS_PER_LANE = RED_TASKS // PICK_LANES  # 4 of the 32 owner slices a lane sums
assert PSS_PER_LANE * PICK_LANES == RED_TASKS
assert PSS_PER_LANE == 4  # a lane's slices of a rank: two 2-pair loads
assert TOPK * PICK_LANES == 64  # a row's picks fill one wave: S rows, S waves


@functools.cache
def lds_struct(s: int):
    """``xs``: every pick's 16 weighted, normed columns (f32 [s 8 16])."""

    @fx.struct
    class ReduceLds:
        xs: fx.Array[fx.Float32, s * TOPK * ROWS, 16]

    return ReduceLds


# Index helpers: plain integer arithmetic on Python ints and traced values
# alike (the CPU tests check the protocol with the same formulas).


def owner_col(j, rp):
    """The owned column (0 .. OWN_COLS) of owner CTA j's lane pair rp."""
    return j * ROWS + 2 * rp


def global_col(rank, cc):
    """Owned column ``cc`` of rank ``rank`` in the pooled width (0 .. 4096)."""
    return rank * OWN_COLS + cc


def pss_lane_slices(rp):
    """The owner slices a pick's lane rp sums (4 of 32, in order)."""
    return [PSS_PER_LANE * rp + i for i in range(PSS_PER_LANE)]


def moe_elem(t, col):
    """``moe`` element index (``put_bf``) of row t, pooled column col: 2
    ``spec.moe_pair`` for an even col."""
    return t * MOE_HIDDEN + col


def topk_tree(xs):
    """The fixed sum of a row's 8 picks: ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7))."""
    assert len(xs) == TOPK == 8
    return ((xs[0] + xs[1]) + (xs[2] + xs[3])) + ((xs[4] + xs[5]) + (xs[6] + xs[7]))


def _f32(w):
    return w.bitcast(fx.Float32)


@traced
def stage_owner(c, j):
    """Owner CTA j of this rank (columns 16 j .. of its 512): a thread a pick's
    two columns (pick = t 8 + k, ``rsx`` -> R -> x -> the pick's sum of squares
    via ``sqx`` -> xs in LDS), then a thread a row's two columns
    (the top-k sum -> every rank's ``moe``). Ends with a barrier (``xs`` is in
    the stage union)."""
    s, tid, wave = c["S"], c["tid"], c["wave"]
    a, rt, rank = c["args"], c["rt"], c["rank"]
    xs = c["lds"]["reduce"]["xs"]
    # thread tid < S 64 -> (pick, rp): the first S waves (S 8 picks x 8 lanes
    # is S whole waves, so the butterflies below see whole 8-lane groups);
    # the other waves poll nothing and only take the barriers
    if wave < s:
        pick = tid // PICK_LANES
        t = pick // TOPK
        k = pick % TOPK
        rp = tid % PICK_LANES
        cc = owner_col(j, rp)
        # weights and route data first (none waits on a producer)
        w0, w1 = bf2_f32(ld_i32(a["w_postexp"], global_col(rank, cc) // 2))
        w0 = w0 + fx.Float32(GAIN)
        w1 = w1 + fx.Float32(GAIN)
        tw = route_lds.load_weight(rt, t, k)
        pad_t = ld_pad(a["pad"], t)
        # R: the 8 ranks' partials in rank order, fp32
        r0, r1 = sum_partials(
            c["poll"], c["peer"]["rsx_own"], lambda src: rsx_pair(src, pick, cc), TP
        )
        x0 = r0 * w0
        x1 = r1 * w1
        # the pick's sum of squares: 16 columns (lanes rp ^ 4, 2, 1 of the pick)
        # -> every rank's slice j of this rank
        ps = butterfly(x0 * x0 + x1 * x1, (4, 2, 1))
        if rp == 0:
            for p in range_constexpr(TP):
                c["put"](c["peer"]["sqx"][p], sqx_pair(rank, j, pick), ps)
        # ... all 4096, one batch: rank src's 512 as lane rp's slices
        # 4 rp .. 4 rp + 3 summed in order, then the 8 lanes; then the ranks
        # in rank order
        # (polls after a predicated put go through ``uniform_addr``: no
        # readfirstlane waterfall around the poll loads)
        sl = pss_lane_slices(rp)[0]
        sqx = uniform_addr(c["peer"]["sqx_own"])
        sq = c["poll"](
            [
                (sqx, sqx_pair(src, sl + 2 * h, pick), 2)
                for src in range(TP)
                for h in range(2)
            ],
            batch=2 * TP,
        )
        owns = []
        for src in range_constexpr(TP):
            lo, hi = sq[2 * src], sq[2 * src + 1]
            own = ((_f32(lo[0]) + _f32(lo[1])) + _f32(hi[0])) + _f32(hi[1])
            owns.append(butterfly(own, (4, 2, 1)))
        ss = owns[0]
        for src in range_constexpr(1, TP):
            ss = ss + owns[src]
        scale = (
            hw_rsq(ss * fx.Float32(1.0 / MOE_HIDDEN) + fx.Float32(POST_EXP_EPS)) * tw
        )
        dead = pad_t | (tw == fx.Float32(0.0))
        y0 = dead.select(fx.Float32(0.0), x0 * scale)
        y1 = dead.select(fx.Float32(0.0), x1 * scale)
        fx.ptr_store(fx.Float32(y0), xs + (pick * ROWS + 2 * rp))
        fx.ptr_store(fx.Float32(y1), xs + (pick * ROWS + 2 * rp + 1))
    gpu.barrier()
    # the top-k sum: thread (row tt, pair rp2) -> 2 columns of the row
    if tid < s * PICK_LANES:
        tt = tid // PICK_LANES
        rp2 = tid % PICK_LANES
        m0 = topk_tree(
            [fx.ptr_load(xs + ((tt * TOPK + q) * ROWS + 2 * rp2)) for q in range(TOPK)]
        )
        m1 = topk_tree(
            [
                fx.ptr_load(xs + ((tt * TOPK + q) * ROWS + 2 * rp2 + 1))
                for q in range(TOPK)
            ]
        )
        pad_tt = ld_pad(a["pad"], tt)
        m0 = pad_tt.select(fx.Float32(0.0), m0)
        m1 = pad_tt.select(fx.Float32(0.0), m1)
        col = global_col(rank, owner_col(j, rp2))
        for p in range_constexpr(TP):
            c["put_bf"](c["peer"]["moe"][p], moe_elem(tt, col), [m0, m1])
    gpu.barrier()  # the LDS hand-over: the unpooler's xl reuses xs's bytes
