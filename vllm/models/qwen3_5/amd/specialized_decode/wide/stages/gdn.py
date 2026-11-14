# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gated delta net of a GDN layer (vLLM's `QwenGatedDeltaNetAttention`
decode on one rank, the flat `[q | k | v | z]` layout) as grid phases over
every row of the step.

- `proj`: `[q | k | v | z] = x @ W_qkvz.T` in three K slices a 64-row
  group (72 groups, 216 units), folded by the group's last arriving unit,
  whose epilogue runs the causal conv on `q | k | v` with its state update:
  `silu(w0 s0 + w1 s1 + w2 s2 + w3 x)`, new state `[s1, s2, x]`, `x` the
  projection rounded to bf16 (the stock GEMM's output); `z` is stored as
  is. `W_qkvz` is bf16, or MXFP4 against the MXFP4 rows of `x` (W4A4, the
  online-quantized projection; with a Hadamard rotation those rows are of
  `x H32` and the weight is stored rotated). The bf16 `[b | a]` rows run
  as split-K units on the CTAs past the projection's; the recurrence sums
  their partials.
- `recur`: a unit is (row, value head), a CTA: the gated delta rule on the
  head's 128 x 128 fp32 state in place, then the gated RMSNorm of the output
  (`RMSNormGated`, `norm_before_gate`, SiLU gate: `norm(o) w silu(z)`),
  rounded to bf16 (the stock norm's output); for an MXFP4 `out_proj`
  quantized (after the Hadamard rotation, when it has one). A row whose state
  slot is the null block (<= 0, CUDA graph padding) skips its state and
  outputs zeros, as stock does.
- `out`: `core @ W_out.T` (bf16 or MXFP4; the attention layer's bf16
  `o_proj` with the output gate applied as its activations are staged) in
  two K slices a 64-column group, folded by the group's last arriver, which
  pushes the group to the TP exchange.

Up to `tiled.RESIDENT_MAX` rows (the split launch, `layout.SPLIT_MAX`) the
projection and the recurrence are one phase: the projection's outputs reach
the recurrence as tagged pairs (`tagged`) instead of through a grid
barrier; `out` runs only past it, in the MLP half's launch below.
"""

from __future__ import annotations

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen3_5.amd.specialized_decode.layout import (
    BA,
    BLOCKS,
    CONV_DIM,
    GDN_DK,
    GDN_DV,
    HIDDEN,
    KEY_DIM,
    NK,
    NV,
    QKVZ,
    VALUE_DIM,
)
from vllm.models.qwen3_5.amd.specialized_decode.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    exp,
    gld,
    gst,
    ld,
    ld_i32x4,
    lds_ld,
    lds_st,
    pack_bf16,
    pack_bf16x8,
    permlane_swap,
    rsqrt,
    sigmoid,
    silu,
    softplus,
    st,
    traced,
    uniform,
    unpack_bf16x8,
    xshfl,
)
from vllm.models.qwen3_5.amd.specialized_decode.wide.grid import TraceInvariant
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen3_5.amd.specialized_decode.wide.stages import (
    mx,
    norm,
    tagged,
    tiled,
)

PROJ_SPLITS = 3  # 72 row groups x 3 = 216 units
BA_SPLITS = 32  # [b | a] units: one 64-row group (32 rows) x 32 slices of 2 chunks
OUT_SPLITS = 2  # 128 column groups x 2 = 256 units
HEADS_PER_K = NV // NK  # value heads a key head serves
L2_EPS = 1e-6  # the in-kernel q / k L2 norm's
# widths past tiled.RESIDENT_MAX up to which a projection group's conv loads
# are issued by its last arriver alone
CONV_LATE_ROWS = 48


def proj_plan():
    return tiled.SplitK(QKVZ, HIDDEN, PROJ_SPLITS)


def ba_plan():
    return tiled.SplitK(BA, HIDDEN, BA_SPLITS)


BA_CTA0 = proj_plan().units  # the [b | a] units' first CTA
assert BA_CTA0 + ba_plan().units <= BLOCKS


def out_plan(k: int = VALUE_DIM):
    return tiled.SplitK(HIDDEN, k, OUT_SPLITS)


def i32(v):
    return fx.Int32(v)


def _halves(word):
    return bf_lo(word), bf_hi(word)


class Bufs(TraceInvariant):
    """Named addresses (Int64) and trace-time constants of a launch."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(
                self, k, v if isinstance(v, (bool, int, float, str)) else fx.Int64(v)
            )


# --------------------------------------------------------------------------
# input projection + conv
# --------------------------------------------------------------------------


def handoff_words(rows: int) -> tuple[int, int, int]:
    """i32 words of the `qkv`, `ba` and `core` buffers: plain, or at the
    resident widths tagged pairs (`[rows, QKVZ / 2]` bf16 pairs, the
    partials' f32 values, `[rows, VALUE_DIM / 2]` bf16 pairs)."""
    if tiled.resident(rows):
        return rows * QKVZ, 2 * ba_plan().partial_words(rows), rows * VALUE_DIM
    return rows * CONV_DIM // 2, ba_plan().partial_words(rows), rows * VALUE_DIM // 2


def proj_weights(w):
    """`w.qkvz_fmt`: "bf16", or "mxfp4" (`qkvz_s` its scales,
    `shuffled` its layout)."""
    if w.qkvz_fmt == "bf16":
        return tiled.Bf16Weights(w.qkvz, QKVZ, HIDDEN)
    return tiled.Fp4Weights(w.qkvz, w.qkvz_s, QKVZ, HIDDEN, w.shuffled)


def proj_source(w, b):
    if w.qkvz_fmt == "bf16":
        return tiled.Source(b.x, HIDDEN)
    return tiled.Mx4Source(b.x4, b.x4s, HIDDEN)


def ba_weights(w):
    return tiled.Bf16Weights(w.ba, BA, HIDDEN)


@traced
def proj(g, gemm, w, b, rows: int, fold0: int, pending=None, tag=None):
    """The input projection and its conv. `w`: `qkvz` (and `qkvz_s`),
    `ba` (bf16), `conv` ([CONV_DIM, 4] bf16); `b`: `x` (and the MXFP4
    rows `x4` / `x4s`), `qkv`, `z`, `ba` (partials), `part`,
    `st_idx`, `conv_state` (a slot [3, CONV_DIM] bf16, slots
    `conv_stride` elements apart). Fold counters `fold0 ..`.
    `pending`: the first unit's weights (`proj_issue`'s), or None.
    With `tag` (resident widths) the outputs are tagged
    pairs the recurrence polls, with no grid barrier between: `qkv` holds
    `[rows, QKVZ / 2]` (bf16 pair, tag) pairs, z included, `ba` the
    partials' (value, tag) pairs."""
    plan = proj_plan()
    if g.bid < i32(plan.units):
        if const_expr(gemm.resident):
            _proj_resident(g, gemm, w, b, rows, fold0, pending, tag)
        else:
            _proj_streamed(g, gemm, w, b, rows, fold0, pending)
    else:
        _ba_units(g, gemm, w, b, tag)


def _proj_resident(g, gemm, w, b, rows: int, fold0: int, pending, tag):
    plan = proj_plan()

    def finish(group, first, lane, acc):
        conv_rows(w, b, rows, first + lane // 16 * 4, lane, acc, gemm.tiles, tag)

    epilogue, after = tiled.fold(g, gemm, plan, b.part, fold0, finish)
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        proj_source(w, b),
        proj_weights(w),
        plan,
        epilogue,
        after,
        pending=pending,
    )


@traced
def _ba_units(g, gemm, w, b, tag):
    """The [b | a] split-K units on CTAs `BA_CTA0 ..`, their partials plain
    or (with `tag`) tagged."""
    bplan = ba_plan()
    if const_expr(tag is None):
        epi = tiled.partial_epilogue(gemm, bplan, b.ba)
    else:
        epi = _tagged_partials(gemm, bplan, b.ba, tag)
    tiled.gemm_units(
        g.bid - i32(BA_CTA0),
        BLOCKS,
        gemm,
        tiled.Source(b.x, HIDDEN),
        ba_weights(w),
        bplan,
        epi,
    )


def _proj_streamed(g, gemm, w, b, rows: int, fold0: int, pending):
    """`proj` past `tiled.RESIDENT_MAX`. The conv's inputs do not depend on
    the projection, so they leave the group's fold chain: the slots load
    before the GEMM. Up to `CONV_LATE_ROWS` the last arriver issues the taps
    and states with its partial loads (`tiled.fold`'s `begin`), so their
    round trips overlap; above, every unit issues them after its partials,
    before the fold arrival, which then drains them."""
    plan = proj_plan()
    tokens, _ = _conv_tokens(rows, gemm.lane, gemm.tiles)
    slots = [ld(b.st_idx, tk, fx.Int32, cm=CM_DEV) for tk in tokens]
    late = rows <= CONV_LATE_ROWS
    staged: list = []

    def stage(first, lane):
        # a z row reads the last conv rows' and uses nothing
        row = fx.min(first + lane // 16 * 4, i32(CONV_DIM - 4))
        staged[:] = [conv_loads(w, b, row, tokens, slots)]

    def begin(group, first, lane):
        stage(first, lane)

    def finish(group, first, lane, acc):
        row = first + lane // 16 * 4
        conv_rows(w, b, rows, row, lane, acc, gemm.tiles, loads=staged[0])

    epilogue, after = tiled.fold(
        g, gemm, plan, b.part, fold0, finish, begin=begin if late else None
    )

    def staging_epilogue(unit, split, first_row, lane, acc):
        epilogue(unit, split, first_row, lane, acc)
        stage(first_row, lane)

    # the staged loads are the CTA's one unit's
    assert late or plan.units <= BLOCKS, plan.units
    tiled.gemm_units(
        g.bid,
        BLOCKS,
        gemm,
        proj_source(w, b),
        proj_weights(w),
        plan,
        epilogue if late else staging_epilogue,
        after,
        pending,
    )


def _tagged_partials(gemm, plan, part, tag):
    """`tiled.partial_epilogue` storing (value, tag) pairs."""

    @traced
    def epilogue(unit, split, first_row, lane, acc):
        for t in range_constexpr(gemm.tiles):
            token = i32(t * 16) + lane % 16
            if token < gemm.batch:
                at = (split * gemm.batch + token) * plan.padded + first_row
                at = at + lane // 16 * 4
                tagged.put(part, at, [acc[t][e] for e in range(4)], tag)

    return epilogue


def proj_issue(g, w):
    """The first unit's weights of `proj` at the resident widths, in flight
    (a CTA past the projection's units: its [b | a] unit's)."""
    plan = proj_plan()
    return tiled.issue_unit(g.bid, g.tid, proj_weights(w), plan)


def _conv_tokens(rows: int, lane, tiles: int):
    """A lane's token of each tile (a padding lane takes the last token) and
    whether it is a real one."""
    tokens = [fx.min(i32(t * 16) + lane % 16, i32(rows - 1)) for t in range(tiles)]
    valid = [i32(t * 16) + lane % 16 < rows for t in range(tiles)]
    return tokens, valid


def conv_loads(w, b, row, tokens, slots=None):
    """The conv's loads for projection rows `row .. row + 3` (conv rows):
    their taps and each token's state (`slots`: the `tokens`' state slots,
    loaded here if None)."""
    words = [
        ld_i32x4(w.conv, row * 2),
        ld_i32x4(w.conv, row * 2 + 4),
    ]  # the rows' 16 taps
    if slots is None:
        slots = [ld(b.st_idx, tk, fx.Int32, cm=CM_DEV) for tk in tokens]
    live = [sl > 0 for sl in slots]
    # a slot's [3, CONV_DIM] state, slots ``conv_stride`` elements apart
    sbs = [
        b.conv_state
        + (fx.Int64(live[t].select(slots[t], i32(0))) * b.conv_stride + fx.Int64(row))
        * fx.Int64(2)
        for t in range(len(tokens))
    ]
    states = [
        [gld(sbs[t] + fx.Int64(jj * CONV_DIM * 2), fx.Int32, 2) for jj in range(3)]
        for t in range(len(tokens))
    ]
    return words, live, sbs, states


@traced
def conv_rows(w, b, rows: int, row, lane, acc, tiles: int, tag=None, loads=None):
    """Projection rows `row .. row + 3` of every token (a fold's summed
    accumulators): conv + silu with the state update for q | k | v, the plain
    store for z. A group of four never straddles the two. `loads`: the rows'
    `conv_loads`, issued before the summed partials were loaded; without
    them every token tile's slot and state loads are issued here before the
    first use. A padding token's lane reads the last token's and stores
    nothing. With `tag` q | k | v and z are stored as tagged pairs
    (`proj`)."""
    tokens, valid = _conv_tokens(rows, lane, tiles)
    if row < CONV_DIM:
        if const_expr(loads is None):
            words, live, sbs, states = conv_loads(w, b, row, tokens)
        else:
            words, live, sbs, states = loads
        tap = []
        for e in range_constexpr(4):
            w01 = _halves(words[e // 2][2 * (e % 2)])
            w23 = _halves(words[e // 2][2 * (e % 2) + 1])
            tap.append([w01[0], w01[1], w23[0], w23[1]])
        for t in range_constexpr(tiles):
            s = states[t]
            xs = [bf16_round(acc[t][e]) for e in range(4)]
            out = []
            for e in range_constexpr(4):
                sv = [
                    live[t].select(_halves(s[jj][e // 2])[e % 2], fx.Float32(0.0))
                    for jj in range(3)
                ]
                out.append(
                    silu(
                        tap[e][0] * sv[0]
                        + tap[e][1] * sv[1]
                        + tap[e][2] * sv[2]
                        + tap[e][3] * xs[e]
                    )
                )
            if valid[t]:
                packed = (
                    fx.Vector.from_elements(out, fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)
                )
                if const_expr(tag is None):
                    st(b.qkv, (tokens[t] * CONV_DIM + row) // 2, packed, CM_DEV)
                else:
                    at = (tokens[t] * QKVZ + row) // 2
                    tagged.put(b.qkv, at, [packed[0], packed[1]], tag)
                if live[t]:
                    gst(sbs[t], s[1])
                    gst(sbs[t] + fx.Int64(CONV_DIM * 2), s[2])
                    gst(
                        sbs[t] + fx.Int64(2 * CONV_DIM * 2),
                        fx.Vector.from_elements(xs, fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32),
                    )
    else:
        for t in range_constexpr(tiles):
            if valid[t]:
                v = fx.Vector.from_elements([acc[t][e] for e in range(4)], fx.Float32)
                packed = v.to(fx.BFloat16).bitcast(fx.Int32)
                if const_expr(tag is None):
                    at = (tokens[t] * VALUE_DIM + row - CONV_DIM) // 2
                    st(b.z, at, packed, CM_DEV)
                else:
                    at = (tokens[t] * QKVZ + row) // 2
                    tagged.put(b.qkv, at, [packed[0], packed[1]], tag)


# --------------------------------------------------------------------------
# recurrence + gated RMSNorm
# --------------------------------------------------------------------------

# The state of a value head is [DV rows (value), DK cols (key)] fp32, key
# contiguous. Lane ``l`` of wave ``w`` holds key quad ``l % 32`` of rows ``8 i
# + 2 w + l // 32`` (i = 0..15): every access is 16 bytes a lane and a wave
# instruction covers two whole rows (1 KiB). Dots over the key dimension
# reduce over the 32 lanes of a row.
ROWS_PER_THREAD = GDN_DV // 8
HEAD_BYTES = GDN_DV * GDN_DK * 4
# LDS f32 offsets of a unit's staged values
Q_AT, K_AT, V_AT, Z_AT = 0, GDN_DK, 2 * GDN_DK, 2 * GDN_DK + GDN_DV
G_AT = 2 * GDN_DK + 2 * GDN_DV  # [beta, decay]
N_AT = G_AT + 4  # the norm's wave partials
B_AT = N_AT + 4  # k . h of each wave's rows, [wave][half][16]
Y_AT = B_AT + 4 * 32  # the head's gated-norm outputs


def recur_lds_words() -> int:
    return Y_AT + GDN_DV


def _row_sum(x):
    return butterfly(x, (16, 8, 4, 2, 1))


def _state_offsets(g):
    """Element offsets of this thread's 16 state quads in a head."""
    kq, rp = g.lane % 32, g.lane // 32
    return [
        (i32(8 * i) + g.wave * 2 + rp) * GDN_DK + kq * 4 for i in range(ROWS_PER_THREAD)
    ]


def state_base(b, slot):
    """Byte address of slot `slot`'s state (clamped to the null block 0 for
    a padding row, whose state is never written); slots are `ssm_stride`
    f32 apart, a slot's [NV, DV, DK] heads dense."""
    s = (slot > 0).select(slot, i32(0))
    return b.ssm + fx.Int64(s) * b.ssm_stride * fx.Int64(4)


def load_head(b, unit, at):
    """This thread's 16 state quads of value head `unit % NV` of row `unit //
    NV`."""
    row, vh = unit // NV, unit % NV
    slot = uniform(ld(b.st_idx, row, fx.Int32, cm=CM_DEV))
    hb = state_base(b, slot) + fx.Int64(vh) * fx.Int64(HEAD_BYTES)
    return [ld(hb, a, fx.Float32, 4, cm=CM_NT) for a in at]


def recur_first(g, b, rows: int):
    """The state loads of the CTA's first unit. No phase of the launch writes
    them before the recurrence, so they may be issued ahead (inside the
    barrier before it)."""
    units = rows * NV
    unit = (g.bid < i32(units)).select(g.bid, i32(units - 1))
    return load_head(b, unit, _state_offsets(g))


@traced
def recur(g, w, b, lds, rows: int, eps: float, core: str, h=None, tag=None):
    """`w`: `a_log` (f32), `dt_bias` (bf16), `norm` ([GDN_DV] bf16);
    `b`: `qkv`, `z`, `ba` (partials), `st_idx`, `ssm` (a slot [NV,
    DV, DK] f32, slots `ssm_stride` apart), `core` ([rows, VALUE_DIM]
    bf16, or for `core` "mxfp4*" the codes with `core_s` the scale
    words); `lds`: `recur_lds_words()` f32 words. CTA `bid` takes
    units (row, value head) `bid, bid + BLOCKS, ..`, the next unit's state
    loads in flight while one computes. With `tag` (resident widths) the
    projection's outputs are tagged pairs (`proj`). `core`: its format,
    "bf16", "bf16_h32" (Hadamard-rotated bf16 rows, for an MXFP4 out_proj
    that quantizes them itself), "mxfp4" or "mxfp4_h32", "_t" for the
    Triton GEMM's quantization. `h`: the first unit's state loads
    (`recur_first`), or None."""
    kq = g.lane % 32
    k0 = kq * 4
    rp = g.lane // 32
    at = _state_offsets(g)
    units = rows * NV
    # the norm weight of the one row each lane stores (``head``)
    vmine = (kq // 2) * 8 + g.wave * 2 + rp
    nw = bf_lo(ld(w.norm, vmine // 2, fx.Int32) >> ((vmine % 2) * 16))
    if const_expr(h is None):
        h = recur_first(g, b, rows)
    for unit in range(g.bid, units, BLOCKS):
        row, vh = unit // NV, unit % NV
        slot = uniform(ld(b.st_idx, row, fx.Int32, cm=CM_DEV))
        stage(g, w, b, lds, rows, row, vh, tag)
        q = lds_ld(lds + Q_AT + k0, 4, fx.Float32)
        k = lds_ld(lds + K_AT + k0, 4, fx.Float32)
        nxt = load_head(b, (i32(unit) + BLOCKS < units).select(unit + BLOCKS, unit), at)
        if slot > 0:
            hb = state_base(b, slot) + fx.Int64(vh) * fx.Int64(HEAD_BYTES)
            head(g, nw, h, hb, at, q, k, lds, rp, kq, eps)
        else:
            if g.tid < GDN_DV:
                lds_st(lds + Y_AT + g.tid, fx.Float32(0.0))
            fx.barrier()
        publish(g, b, lds, row, vh, core)
        h = nxt
        fx.barrier()


@traced
def stage(g, w, b, lds, rows: int, row, vh, tag):
    """q, k (L2-normalized, q scaled) of the value head's key head, its v and
    z, and its gates into LDS: thread `t` < 128 four of q | k | v | z, the
    next 64 the [b | a] split partials (beta's, then the decay's), summed over
    32 lanes and rounded to bf16 (the stock GEMM's output). With `tag` the
    values are polled from the projection's tagged pairs."""
    bplan = ba_plan()
    t = g.tid
    kh = vh // HEADS_PER_K
    scale = fx.Float32(1.0 / math.sqrt(GDN_DK))
    tt = fx.min(t, i32(127))  # a gate thread reads z's last quad and drops it
    col = (tt < 32).select(
        kh * GDN_DK + tt * 4,
        (tt < 64).select(
            KEY_DIM + kh * GDN_DK + (tt - 32) * 4,
            (tt < 96).select(
                2 * KEY_DIM + vh * GDN_DV + (tt - 64) * 4,
                CONV_DIM + vh * GDN_DV + (tt - 96) * 4,
            ),
        ),
    )
    gl = fx.max(t - 128, i32(0))
    live = (t >= 128) & (gl < 2 * bplan.splits)
    j, s = gl // bplan.splits, gl % bplan.splits
    gate_row = (j == 0).select(vh, NV + vh)
    gate_at = live.select((s * rows + row) * bplan.padded + gate_row, i32(0))
    if const_expr(tag is None):
        is_z = tt >= 96
        qkv_at = (row * CONV_DIM + fx.min(col, i32(CONV_DIM - 4))) // 2
        z_at = (row * VALUE_DIM + fx.max(col - CONV_DIM, i32(0))) // 2
        wq = ld(b.qkv, qkv_at, fx.Int32, 2, cm=CM_DEV)
        wz = ld(b.z, z_at, fx.Int32, 2, cm=CM_DEV)
        wv = [is_z.select(wz[e], wq[e]) for e in range(2)]
        part = ld(b.ba, gate_at, fx.Float32, cm=CM_DEV)
    else:
        got = tagged.poll(
            g,
            [
                (b.qkv, (row * QKVZ + fx.min(col, i32(QKVZ - 4))) // 2, 2),
                (b.ba, gate_at, 1),
            ],
            tag,
        )
        wv = got[0]
        part = got[1][0].bitcast(fx.Float32)
    x = [bf_lo(wv[0]), bf_hi(wv[0]), bf_lo(wv[1]), bf_hi(wv[1])]
    if t < 64:  # q | k: L2-normalized, q scaled
        ss = _row_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
        r = rsqrt(ss + fx.Float32(L2_EPS)) * (t < 32).select(scale, fx.Float32(1.0))
        lds_st(
            lds + Q_AT + t * 4,
            fx.Vector.from_elements([x[e] * r for e in range(4)], fx.Float32),
        )
    else:
        if t < 128:
            lds_st(lds + V_AT + (t - 64) * 4, fx.Vector.from_elements(x, fx.Float32))
    # the gates: wave 2's lane halves sum beta's and the decay's partials
    tot = _row_sum(live.select(part, fx.Float32(0.0)))
    if live & (s == 0):
        a_log = ld(w.a_log, vh, fx.Float32)
        dt = bf_lo(ld(w.dt_bias, vh // 2, fx.Int32) >> ((vh % 2) * 16))
        v = bf16_round(tot)
        gate = (j == 0).select(sigmoid(v), exp(-exp(a_log) * softplus(v + dt)))
        lds_st(lds + G_AT + j, gate)
    fx.barrier()


def _halve(v, lane):
    """Sums over the 32 lanes of a row half (lane bits 0-4) of the 16 values a
    lane holds, one a state row, by halving: at each level a lane keeps half
    its values and adds its partner's copy of that half, so 16 rows cost
    16 exchanges, not 80. Returns the sum of
    row `(lane % 32) // 2`, in both lanes of each pair."""
    # level 1 (lane ^ 16): v_permlane16_swap of (low half, high half) leaves
    # the row-0 lanes' low and the row-1 lanes' high partial pairs in place
    out = []
    for j in range(8):
        a, b = _permlane16(v[j], v[j + 8])
        out.append(a + b)
    v = out
    for level, n in ((8, 4), (4, 2), (2, 1)):
        hi = (lane & level) != 0
        out = []
        for j in range(n):
            keep = hi.select(v[j + n], v[j])
            send = hi.select(v[j], v[j + n])
            out.append(keep + xshfl(send, level))
        v = out
    return v[0] + xshfl(v[0], 1)


def _permlane16(x, y):
    a, b = permlane_swap(16, x.bitcast(fx.Int32), y.bitcast(fx.Int32))
    return a.bitcast(fx.Float32), b.bitcast(fx.Float32)


@traced
def head(g, nw, h, hb, at, q, k, lds, rp, kq, eps):
    """One value head of one row: the delta rule on the state (`h`, this
    thread's 16 rows of 4 keys, loaded), stored back in place; then the gated
    RMSNorm of the head's output into LDS (`Y_AT`). `k . h` of each row is
    reduced by halving (`_halve`) and broadcast to the row's lanes through
    the wave's LDS words (`B_AT`); `q . h` stays in the lanes that hold it."""
    beta = lds_ld(lds + G_AT)
    decay = lds_ld(lds + G_AT + 1)
    vl, zl, nl = lds + V_AT, lds + Z_AT, lds + N_AT
    bl = lds + B_AT + g.wave * 32 + rp * 16  # this half-wave's 16 rows
    x = [[h[i][e] * decay for e in range(4)] for i in range(ROWS_PER_THREAD)]
    kv = _halve(
        [
            x[i][0] * k[0] + x[i][1] * k[1] + x[i][2] * k[2] + x[i][3] * k[3]
            for i in range(ROWS_PER_THREAD)
        ],
        g.lane,
    )
    mine = kq // 2  # the row whose sum this lane holds
    if kq % 2 == 0:
        lds_st(bl + mine, kv)
    rows_kv = [lds_ld(bl + 4 * c, 4, fx.Float32) for c in range(4)]
    outs = []
    for i in range_constexpr(ROWS_PER_THREAD):
        v = 8 * i + g.wave * 2 + rp
        dv = (lds_ld(vl + v) - rows_kv[i // 4][i % 4]) * beta
        x[i] = [x[i][e] + dv * k[e] for e in range(4)]
        st(hb, at[i], fx.Vector.from_elements(x[i], fx.Float32), cm=CM_NT)
        outs.append(x[i][0] * q[0] + x[i][1] * q[1] + x[i][2] * q[2] + x[i][3] * q[3])
    o = bf16_round(
        _halve(outs, g.lane)
    )  # row ``mine``'s output, rounded as stock's core
    # sum of squares over the head's 128 outputs: one lane of each pair counts
    sq = (kq % 2 == 0).select(o * o, fx.Float32(0.0))
    sq = butterfly(sq, (32, 16, 8, 4, 2, 1))
    if g.lane == 0:
        lds_st(nl + g.wave, sq)
    fx.barrier()
    tot = lds_ld(nl) + lds_ld(nl + 1) + lds_ld(nl + 2) + lds_ld(nl + 3)
    rr = rsqrt(tot / fx.Float32(GDN_DV) + fx.Float32(eps))
    v = 8 * mine + g.wave * 2 + rp
    if kq % 2 == 0:
        lds_st(lds + Y_AT + v, o * rr * nw * silu(lds_ld(zl + v)))
    fx.barrier()


@traced
def publish(g, b, lds, row, vh, core: str):
    """The head's 128 gated-norm outputs (`Y_AT`) to `core`: bf16 (a
    thread four), or for an MXFP4
    `out_proj` a thread a 32-group rounded to bf16 (the stock norm's
    output), Hadamard-rotated for "mxfp4_h32" (rounded again), quantized
    (`mx.scale_code`; "_t": as the Triton FP4 GEMM does); "bf16_h32"
    stores the rotated group as bf16 instead."""
    h32, rceil = core.startswith("mxfp4_h32"), not core.endswith("_t")
    if const_expr(core == "bf16"):
        if g.tid < GDN_DV // 4:
            ys = lds_ld(lds + Y_AT + g.tid * 4, 4, fx.Float32)
            words = [pack_bf16(ys[0], ys[1]), pack_bf16(ys[2], ys[3])]
            first = (row * VALUE_DIM + vh * GDN_DV) // 2 + g.tid * 2
            st(b.core, first, fx.Vector.from_elements(words, fx.Int32), CM_DEV)
    elif const_expr(core == "bf16_h32"):
        if g.tid < GDN_DV // mx.GROUP:
            vals = []
            for m in range_constexpr(mx.GROUP // 4):
                y4 = lds_ld(lds + Y_AT + g.tid * mx.GROUP + 4 * m, 4, fx.Float32)
                vals += [bf16_round(y4[e]) for e in range(4)]
            vals = norm.hadamard32(vals)
            words = [
                pack_bf16(vals[2 * e], vals[2 * e + 1]) for e in range(mx.GROUP // 2)
            ]
            first = (row * VALUE_DIM + vh * GDN_DV + g.tid * mx.GROUP) // 2
            for q in range_constexpr(mx.GROUP // 8):
                st(
                    b.core,
                    first + 4 * q,
                    fx.Vector.from_elements(words[4 * q : 4 * q + 4], fx.Int32),
                    CM_DEV,
                )
    else:
        if g.tid < GDN_DV // mx.GROUP:
            vals = []
            for m in range_constexpr(mx.GROUP // 4):
                y4 = lds_ld(lds + Y_AT + g.tid * mx.GROUP + 4 * m, 4, fx.Float32)
                vals += [bf16_round(y4[e]) for e in range(4)]
            if const_expr(h32):
                vals = [bf16_round(v) for v in norm.hadamard32(vals)]
            code = mx.scale_code(mx.amax(vals), rceil)
            words = mx.quant_f32_group(vals, code)
            el = row * VALUE_DIM + vh * GDN_DV + g.tid * mx.GROUP
            st(b.core, el // 8, fx.Vector.from_elements(words, fx.Int32), CM_DEV)
            st(b.core_s, el // mx.GROUP, code, CM_DEV)


# --------------------------------------------------------------------------
# output projection -> TP exchange
# --------------------------------------------------------------------------


class GatedCore(tiled.Source):
    """An attention layer's o_proj input: `bf16(attn * bf16(sigmoid(gate)))`,
    as the stock path's two bf16 tensors multiply."""

    def __init__(self, attn, gate, ld_elems: int):
        self.attn, self.gate, self.ld = fx.Int64(attn), fx.Int64(gate), ld_elems

    def issue(self, token, k):
        at = (token * i32(self.ld) + k) // i32(2)
        return [ld_i32x4(self.attn, at), ld_i32x4(self.gate, at)]

    def convert(self, raw):
        a, gt = unpack_bf16x8(raw[0]), unpack_bf16x8(raw[1])
        return pack_bf16x8([a[e] * bf16_round(sigmoid(gt[e])) for e in range(8)])


def out_weights(w_out, w_out_s, fmt: str, shuffled: bool, k: int = VALUE_DIM):
    if fmt == "bf16":
        return tiled.Bf16Weights(w_out, HIDDEN, k)
    return tiled.Fp4Weights(w_out, w_out_s, HIDDEN, k, shuffled)


def out(g, gemm, weights, source, part, tile, xch, k: int, fold0: int):
    """`source @ weights.T` (K = `k`) to the exchange `xch`: each
    64-column group's last arriving unit sums its slices into the LDS
    `tile` and pushes the group's lines."""
    plan = out_plan(k)

    def finish(group, first, lane, acc):
        tile.put_acc(first - group * ex_.COLS, lane, acc, gemm.tiles)

    def done(group):
        fx.barrier()
        ex_.push_tile(g, xch, tile, group)
        fx.barrier()

    epilogue, after = tiled.fold(g, gemm, plan, part, fold0, finish, done)
    tiled.gemm_units(g.bid, BLOCKS, gemm, source, weights, plan, epilogue, after)
