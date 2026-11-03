# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's fused QSA prepare (``ops.qsa_prepare``) for one decode row, inside
the indexer's score launch: from the row's projections (qsa_front's ``qkv``
and index ``qk``) and its metadata, a wave at a time.

- ``main_head``: one of the main attention's heads, Gemma RMSNorm and
  partial NeoX RoPE (the first 64 of 256 dims; MRoPE axes interleaved). A
  query head goes to ``q_out`` and copies its gate; the key head goes to
  the paged K | V cache at the row's slot with its value.
- ``index_query``: an indexer query head (128 dims), normed and rotated
  with the cache's bf16 cos / sin and bf16 arithmetic as stock rounds it.
- ``index_key``: commits the row's raw indexer key to the request's ring
  (with its MRoPE position when the cache keeps them) and, when the token
  completes a group of ``ratio``, pools the group (the ring's keys and this
  one), norms and rotates it at the group's first position and stores it in
  the compressed cache; returns it for the row's new score column.

The rounding points follow ``ops.qsa_prepare``: bf16 inputs, f32 norm, the
rotated half rounded to bf16 before the main heads' f32 rotation, bf16 out.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    bf16_round,
    bf_hi,
    bf_lo,
    ld,
    pack_bf16,
    rsqrt,
    st,
    traced,
    uniform,
    wave_sum,
)

i32 = fx.Int32
f32 = fx.Float32
i64 = fx.Int64

MAIN_DIM = 256
INDEX_DIM = 128
ROTARY = 64  # rotated dims of either head (cos / sin rows of 32 pairs each)
PAIRS = ROTARY // 2


def _axis(pair, mrope, strict: bool):
    """The MRoPE axis (0 t, 1 h, 2 w) of rotary pair ``pair``: ``mrope`` is
    (h, w) sections, or None for plain RoPE. ``strict``: the main heads'
    bound (``<``), else the indexer's (``<=``)."""
    if mrope is None:
        return None
    h, w = mrope
    lim_h, lim_w = (3 * h, 3 * w)
    in_h = (pair < i32(lim_h)) if strict else (pair <= i32(lim_h))
    in_w = (pair < i32(lim_w)) if strict else (pair <= i32(lim_w))
    is_h = (pair % i32(3) == i32(1)) & in_h
    is_w = (pair % i32(3) == i32(2)) & in_w
    return is_h.select(i32(1), is_w.select(i32(2), i32(0)))


def _pos(axes, axis):
    """The position of ``axis`` among ``axes`` (t, h, w) or plain."""
    if axis is None:
        return axes[0]
    return (axis == i32(1)).select(axes[1], (axis == i32(2)).select(axes[2], axes[0]))


def _cos_sin(cos_sin, pos, pair):
    """bf16 cos and sin of rotary ``pair`` at position ``pos`` as f32."""
    at = pos * i32(ROTARY) + pair
    c = ld(cos_sin, at, fx.BFloat16)
    s = ld(cos_sin, at + i32(PAIRS), fx.BFloat16)
    return f32(c), f32(s)


def _gemma_weight(w, at, n: int):
    """``n`` bf16 norm weights at element ``at`` plus one, as f32."""
    words = ld(w, at // i32(2), fx.Int32, n // 2)
    words = [words[e] for e in range(n // 2)] if n > 2 else [words]
    out = []
    for word in words:
        out += [bf_lo(word) + f32(1.0), bf_hi(word) + f32(1.0)]
    return out


@traced
def main_head(lane, src_at, qkv, w, eps: float, cos_sin, axes, mrope):
    """The lane's four dims ``4 lane ..`` of a main head at element ``src_at``
    of ``qkv``, normed and rotated, as two packed bf16 words."""
    words = ld(qkv, src_at // i32(2) + lane * i32(2), fx.Int32, 2)
    x = [bf_lo(words[0]), bf_hi(words[0]), bf_lo(words[1]), bf_hi(words[1])]
    ss = wave_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3])
    inv = rsqrt(ss / f32(MAIN_DIM) + f32(eps))
    wt = _gemma_weight(w, lane * i32(4), 4)
    y = [x[e] * inv * wt[e] for e in range(4)]
    # the rotated dims, rounded to bf16 first; the partner 32 dims away is
    # lane ^ 8 (four dims a lane)
    yr = [bf16_round(v) for v in y]
    partner = [v.shuffle_xor(8, 64) for v in yr]
    low = lane < i32(8)
    rotated = lane < i32(16)
    out = []
    for e in range_constexpr(4):
        pair = (lane * i32(4) + i32(e)) % i32(PAIRS)
        c, s = _cos_sin(cos_sin, _pos(axes, _axis(pair, mrope, True)), pair)
        o = low.select(yr[e] * c - partner[e] * s, yr[e] * c + partner[e] * s)
        out.append(rotated.select(o, y[e]))
    return fx.Vector.from_elements(
        [pack_bf16(out[0], out[1]), pack_bf16(out[2], out[3])], fx.Int32
    )


def _bf(x):
    return bf16_round(x)


@traced
def index_norm_rope(lane, x, w, eps: float, cos_sin, axes, mrope):
    """The lane's two dims ``2 lane ..`` of an indexer head ``x`` (two f32),
    normed and rotated like ``qsa_prepare._norm_rope`` with a bf16 cos /
    sin cache (bf16 products and sums), as two f32 holding bf16 values."""
    ss = wave_sum(x[0] * x[0] + x[1] * x[1])
    rr = rsqrt(ss / f32(INDEX_DIM) + f32(eps))
    wt = _gemma_weight(w, lane * i32(2), 2)
    y = [_bf(x[e] * rr * wt[e]) for e in range(2)]
    partner = [v.shuffle_xor(16, 64) for v in y]
    low = lane < i32(16)
    rotated = lane < i32(32)
    out = []
    for e in range_constexpr(2):
        pair = (lane * i32(2) + i32(e)) % i32(PAIRS)
        c, s = _cos_sin(cos_sin, _pos(axes, _axis(pair, mrope, False)), pair)
        a = _bf(y[e] * c)
        o = low.select(_bf(a - _bf(partner[e] * s)), _bf(a + _bf(partner[e] * s)))
        out.append(rotated.select(o, y[e]))
    return out


def load_pair(base, at):
    """Two bf16 at element ``at`` (even) of ``base`` as f32."""
    word = ld(base, at // i32(2), fx.Int32)
    return [bf_lo(word), bf_hi(word)]


def store_pair(base, at, vals):
    st(base, at // i32(2), pack_bf16(vals[0], vals[1]))


def low_word(base, i):
    """The low word of int64 element ``i`` (a slot or position)."""
    return ld(base, i * i32(2), fx.Int32)


@traced
def index_key(
    lane, row, lp, live, req, k, ix_eps: float, w, cos_sin, mrope, tail: bool, raw, comp
):
    """The row's raw indexer key ``k`` (the lane's two f32 dims) into its
    request's ring, and when the token completes a group (its compressed
    slot set) the group's normed, rotated mean into the compressed cache.
    ``raw``: (ring base, block bytes, row bytes, rows a ring, rings, slots,
    block table, its stride); ``comp``: (cache base, block bytes, rows a
    block, blocks, slots, ratio). Returns (group completed, the compressed
    key's two dims as stored)."""
    base, block_bytes, row_bytes, size, rings, slots, table, table_stride = raw
    cbase, cblock_bytes, cpage, cblocks, cslots, ratio = comp
    cslot = uniform(low_word(cslots, row))
    boundary = live & (cslot >= i32(0)) & (cslot < cblocks * i32(cpage))
    key = [f32(0.0), f32(0.0)]
    if boundary:
        ring_no = uniform(ld(table, req * table_stride, fx.Int32))
        ring_ok = (ring_no >= i32(0)) & (ring_no < rings)
        ring = base + i64(fx.max(ring_no, i32(0))) * i64(block_bytes)
        group = []
        for d in range_constexpr(ratio - 1):
            at = ring + i64(((lp - i32(ratio - 1 - d)) % size) * row_bytes)
            v = load_pair(at, lane * i32(2))
            group.append([ring_ok.select(v[e], f32(0.0)) for e in range(2)])
        group.append(k)
        pooled = []
        for e in range_constexpr(2):
            total = (group[0][e] + group[1][e]) + (group[2][e] + group[3][e])
            pooled.append(_bf(total / f32(ratio)))
        first = lp - i32(ratio - 1)
        axes = [first, first, first]
        if const_expr(tail):
            at = ring + i64((first % size) * row_bytes + 2 * INDEX_DIM)
            axes = [
                ring_ok.select(uniform(ld(at, 2 * a, fx.Int32)), i32(0))
                for a in range(3)
            ]
        key = index_norm_rope(lane, pooled, w, ix_eps, cos_sin, axes, mrope)
        dst = (
            cbase
            + i64(cslot // i32(cpage)) * i64(cblock_bytes)
            + i64((cslot % i32(cpage)) * (2 * INDEX_DIM))
        )
        store_pair(dst, lane * i32(2), key)
    return boundary, key


@traced
def commit_raw(lane, row, live, k, axes, tail: bool, raw):
    """The row's raw indexer key ``k`` (and its MRoPE ``axes``) at its ring
    slot, after the group's reads."""
    base, block_bytes, row_bytes, size, rings, slots, _, _ = raw
    slot = uniform(low_word(slots, row))
    if live & (slot >= i32(0)) & (slot < rings * size):
        dst = (
            base + i64(slot // size) * i64(block_bytes) + i64((slot % size) * row_bytes)
        )
        store_pair(dst, lane * i32(2), k)
        if const_expr(tail):
            _store_axes(lane, dst + i64(2 * INDEX_DIM), axes)


@traced
def _store_axes(lane, at, axes):
    """Lanes 0..2 store the MRoPE positions (t, h, w) as int64 at ``at``."""
    if lane < i32(3):
        pos = (lane == i32(1)).select(
            axes[1], (lane == i32(2)).select(axes[2], axes[0])
        )
        words = fx.Vector.from_elements([pos, pos >> i32(31)], fx.Int32)
        st(at, lane * i32(2), words)
