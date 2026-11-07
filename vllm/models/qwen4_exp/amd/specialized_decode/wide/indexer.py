# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA indexer scores of a decode batch over vLLM's paged compressed-key
cache, in the stock scoring op's format: ``logits[row, c]`` = sum over the
indexer heads of ``relu(q_h . k_c)`` / ``divisor`` for the row's visible
columns ``c < visible[row]`` (columns past it unwritten but for the last
4-column store), and ``visible`` = complete compressed blocks the row sees.

``splits`` CTAs a row each score a contiguous range of 256-column rounds:
the CTA first loads the row's page table into LDS (byte offsets of its pages
in the cache); a wave then multiplies four 16-column tiles of keys with the
row's queries on the matrix unit (``S^T = K Q^T``, 16x16x32, the indexer
heads in four of the 16 columns) per round, with the next round's key loads
in flight.

With a selection budget (``ScoreBuild.top``, the decode top-k's blocks) the
launch also selects, replacing ``top_k_per_row_decode``: a row with at most
``top`` visible blocks is not scored (its first CTA lists them all), and in
a longer row the CTA that arrives last on the row's counter selects the
``top`` best (``wide.select``) into ``sel``.
"""

import math
from dataclasses import astuple, dataclass
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen4_exp.amd.specialized_decode.wide import prepare as prep_
from vllm.models.qwen4_exp.amd.specialized_decode.wide import select as sel_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.attention import (
    COUNTER_WORDS,
    _arrive,
    splits_for,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    _xshfl,
    bf_hi,
    bf_lo,
    gld,
    ld,
    lds_ld,
    lds_st,
    mfma_bf16,
    pack_bf16,
    st,
    traced,
    uniform,
    wave_sum,
    zeros,
)

i32 = fx.Int32
f32 = fx.Float32

THREADS = 256
WAVES = THREADS // 64
DIM = 128  # an indexer head's dims
TILES = 4  # 16-column tiles a wave scores a round
ROUND = WAVES * TILES * 16  # columns a CTA scores a round
LDS_BYTES = 160 * 1024
MAIN_HEADS = 12  # the main attention's query heads a rank
QKV_ROW = (2 * MAIN_HEADS + 2) * 256  # qkv_proj's row: [q | gate] a head, k, v
IX_ROW = 5 * DIM  # index_qk_proj's row: the query heads, then the key


@dataclass(frozen=True)
class ScoreBuild:
    rows: int
    splits: int
    page: int  # compressed keys a cache block
    pages: int  # the page table's width
    heads: int  # indexer heads
    ratio: int
    pos64: bool  # query positions are int64
    divisor: float
    top: int = 0  # the selection budget in blocks (0: scores only)
    # vLLM's fused prepare in the launch (``wide.prepare``): (main eps, index
    # eps, MRoPE (h, w) sections or None, MRoPE positions kept in the ring);
    # () runs none (the caller prepared ``q``, the caches)
    prep: tuple = ()


def score_splits(rows: int) -> int:
    """Score CTAs a row: 64 up to 4 rows (a 128K row's 129 rounds 2-3 a CTA,
    against 8-9 at 16), else attention's ``splits_for``."""
    return 64 if rows <= 4 else splits_for(rows)


def _head_sum(vals, heads: int):
    """Sums over the 16-lane group's first ``heads`` lanes (the others hold
    zeros), into its lane 0."""
    off = 1
    while off < heads:
        vals = [v + _xshfl(v, off) for v in vals]
        off *= 2
    return vals


@traced
def _select_last(tid, end, counters, row, splits: int, verdict, keys, staged, *args):
    """A row past the budget: the row's last-arriving CTA selects its blocks
    (``select.select(..., *args)``) from the scores every CTA wrote."""
    if end > i32(0):
        last = _arrive(counters, row, splits, tid, verdict)
        if last:
            logits, logits_stride, sel, sel_stride, top = args
            sel_.select(
                tid, keys, staged, logits, row, logits_stride, end, sel, sel_stride, top
            )


@traced
def _store_scores(logits, base, c, end, n, vals, cm: int):
    """Lane ``n == 0``'s four scores of columns ``c ..``: whole when they are
    all before ``end``, else one by one (the columns at ``end`` and past it
    are not this tile's)."""
    if (n == 0) & (c < end):
        if c + i32(4) <= end:
            st(logits, base + c, fx.Vector.from_elements(vals, fx.Float32), cm)
        else:
            for e in range_constexpr(4):
                if c + i32(e) < end:
                    st(logits, base + c + i32(e), vals[e], cm)


@traced
def _index_queries(wave, lane, row, heads: int, ixqk, qx, w, eps: float, *args):
    """Wave ``w``'s indexer query head ``w`` of the row, normed and rotated,
    into LDS (``qx``) as bf16 pairs."""
    cos_sin, axes, mrope = args
    if wave < i32(heads):
        xq = prep_.load_pair(ixqk, row * IX_ROW + wave * DIM + lane * 2)
        yq = prep_.index_norm_rope(lane, xq, w, eps, cos_sin, axes, mrope)
        lds_st(qx + wave * (DIM // 2) + lane, pack_bf16(yq[0], yq[1]))


@traced
def _main_heads(first, step: int, lane, row, qkv, q_out, gate_out, *args):
    """The row's main query heads (and their gates) and its key | value
    head, items ``first, first + step, ..`` of 13 (a wave each)."""
    qn_w, kn_w, eps, cos_sin, axes, mrope, kv, kv_block, kv_page, slots = args
    for item in range(first, i32(MAIN_HEADS + 1), i32(step)):
        if item < i32(MAIN_HEADS):
            src = row * QKV_ROW + item * (2 * prep_.MAIN_DIM)
            o = prep_.main_head(lane, src, qkv, qn_w, eps, cos_sin, axes, mrope)
            at = ((row * MAIN_HEADS + item) * prep_.MAIN_DIM) // 2 + lane * 2
            st(q_out, at, o)
            gate = ld(qkv, (src + prep_.MAIN_DIM) // 2 + lane * 2, Int32, 2)
            st(gate_out, at, gate)
        else:
            src = row * QKV_ROW + MAIN_HEADS * 2 * prep_.MAIN_DIM
            o = prep_.main_head(lane, src, qkv, kn_w, eps, cos_sin, axes, mrope)
            slot = uniform(prep_.low_word(slots, row))
            if slot >= i32(0):
                dst = (
                    kv
                    + Int64(slot // kv_page) * kv_block
                    + Int64((slot % kv_page) * (4 * prep_.MAIN_DIM))
                )
                st(dst, lane * 2, o)
                v = ld(qkv, (src + prep_.MAIN_DIM) // 2 + lane * 2, Int32, 2)
                st(dst, prep_.MAIN_DIM // 2 + lane * 2, v)


@traced
def _new_key(lane, row, lp, live, req, k, axes, fresh, visible, qx, heads: int, *args):
    """The row's indexer key: the ring commit and, at a completed group, the
    compressed key and (``fresh``) its score at column ``visible - 1``."""
    eps, w, cos_sin, mrope, tail, raw, comp, logits, base, inv = args
    done, key = prep_.index_key(
        lane, row, lp, live, req, k, eps, w, cos_sin, mrope, tail, raw, comp
    )
    prep_.commit_raw(lane, row, live, k, axes, tail, raw)
    if fresh != i32(0):
        total = f32(0.0)
        for h in range_constexpr(heads):
            word = lds_ld(qx + h * (DIM // 2) + lane)
            dot = wave_sum(bf_lo(word) * key[0] + bf_hi(word) * key[1])
            total = total + fx.max(dot, f32(0.0))
        if lane == i32(0):
            st(logits, base + visible - i32(1), total * f32(inv), CM_DEV)
    _ = done


_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def build(key: ScoreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    assert key.heads <= 16
    splits, page, heads, ratio = key.splits, key.page, key.heads, key.ratio
    table_rounds = -(-key.pages // THREADS)
    inv = 1.0 / key.divisor
    keyed = astuple(key)

    top = key.top
    table_words = 2 * table_rounds * THREADS
    staged = sel_.lds_keys(LDS_BYTES - 4 * (table_words + 64)) if top else 0
    select_words = sel_.lds_words(staged) if top else 4
    # the last arriver reads the row's scores: device-coherent stores
    cm = CM_DEV if top else 0

    prep = key.prep
    if prep:
        main_eps, ix_eps, mrope, tail = prep
        assert heads * DIM // 2 == 256 and ratio == 4

    @fx.struct
    class Lds:
        table: fx.Array[fx.Int32, table_words, 16]
        keys: fx.Array[fx.Int32, select_words, 16]
        verdict: fx.Array[fx.Int32, 4, 16]
        qx: fx.Array[fx.Int32, (heads * DIM // 2) if prep else 4, 16]

    @flyc.kernel(
        name=f"q4w_qsa_score_s{key.rows}_x{splits}"
        + (f"_k{top}" if top else "")
        + ("_prep" if prep else ""),
        known_block_size=[THREADS, 1, 1],
    )
    def kernel(
        q: Int64,
        kc: Int64,
        page_bytes: Int64,
        bt: Int64,
        bt_stride: Int32,
        reqs: Int32,
        token_to_req: Int64,
        positions: Int64,
        seq_lens: Int64,
        logits: Int64,
        logits_stride: Int32,
        columns: Int32,
        visible_out: Int64,
        sel: Int64,
        sel_stride: Int32,
        counters: Int64,
        ixqk: Int64,
        qkv: Int64,
        q_out: Int64,
        gate_out: Int64,
        rope_pos: Int64,
        pos_axis: Int32,
        cos_sin: Int64,
        qn_w: Int64,
        kn_w: Int64,
        iqn_w: Int64,
        ikn_w: Int64,
        kv: Int64,
        kv_block: Int64,
        kv_page: Int32,
        main_slots: Int64,
        raw: Int64,
        raw_block: Int64,
        raw_row: Int32,
        raw_size: Int32,
        raw_rings: Int32,
        raw_slots: Int64,
        raw_bt: Int64,
        raw_bt_stride: Int32,
        comp_slots: Int64,
        comp_blocks: Int32,
    ):
        _ = keyed
        tid = i32(fx.thread_idx.x)
        bid = i32(fx.block_idx.x)
        alloc = fx.SharedAllocator()
        sm = alloc.allocate(Lds).peek()
        table = fx.recast_iter(fx.Int64, sm.table.ptr)
        lane, wave = tid % 64, tid // 64
        n, g = lane % 16, lane // 16
        row, member = bid // splits, bid % splits

        if const_expr(prep):
            # the main heads need nothing but the row: they go first
            axes = [
                uniform(prep_.low_word(rope_pos, a * pos_axis + row)) for a in range(3)
            ]
            _main_heads(
                member * WAVES + wave,
                splits * WAVES,
                lane,
                row,
                qkv,
                q_out,
                gate_out,
                qn_w,
                kn_w,
                main_eps,
                cos_sin,
                axes,
                mrope,
                kv,
                kv_block,
                kv_page,
                main_slots,
            )

        req = uniform(ld(token_to_req, row, Int32))
        live = (req >= 0) & (req < reqs)
        req = live.select(req, i32(0))
        # an int64 position's low word
        pos = uniform(ld(positions, row * (2 if key.pos64 else 1), Int32))
        seq = live.select(uniform(ld(seq_lens, req, Int32)), i32(0))
        a, b = (pos + 1) // ratio, seq // ratio
        visible = (a < b).select(a, b)
        end = live.select((visible < columns).select(visible, columns), i32(0))
        if (member == 0) & (tid == 0):
            st(visible_out, row, visible)
        if const_expr(top):
            # within the budget the row takes every block: nothing to score
            short = end <= i32(top)
            if short & (member == 0):
                sel_.identity(sel, row, sel_stride, end, top, tid)
            end = short.select(i32(0), end)
        # the launch's own new compressed key (a completed group) is scored
        # apart, from registers: its column stays out of the tiles
        fresh = i32(0)
        if const_expr(prep):
            cslot = uniform(prep_.low_word(comp_slots, row))
            fresh = (
                live & (cslot >= i32(0)) & (end == visible) & (end > i32(0))
            ).select(i32(1), i32(0))
        tile_end = end - fresh
        rounds = (tile_end + ROUND - 1) // ROUND
        r0 = member * rounds // splits
        local = (member + 1) * rounds // splits - r0

        # a row with tiles to score: its pages (byte offsets in the cache, -1:
        # none) and, preparing, its indexer queries (wave w's head w, bf16
        # pairs) into LDS; a row within the budget needs neither
        if tile_end > i32(0):
            for tr in range_constexpr(table_rounds):
                p = tid + tr * THREADS
                inside = p < key.pages
                entry = ld(bt, req * bt_stride + inside.select(p, i32(0)), Int32)
                ok = inside & (entry >= 0)
                lds_st(table + p, ok.select(Int64(entry) * page_bytes, Int64(-1)))
            if const_expr(prep):
                _index_queries(
                    wave,
                    lane,
                    row,
                    heads,
                    ixqk,
                    sm.qx.ptr,
                    iqn_w,
                    ix_eps,
                    cos_sin,
                    axes,
                    mrope,
                )
            fx.barrier()

        qs = []
        if const_expr(prep):
            raw_args = (
                raw,
                raw_block,
                raw_row,
                raw_size,
                raw_rings,
                raw_slots,
                raw_bt,
                raw_bt_stride,
            )
            k_raw = prep_.load_pair(ixqk, row * IX_ROW + heads * DIM + lane * 2)
            for kq in range_constexpr(DIM // 32):
                at = (n < heads).select(n, i32(0)) * (DIM // 2) + kq * 16 + g * 4
                v = lds_ld(sm.qx.ptr + at, 4)
                qs.append((n < heads).select(v, zeros(4, Int32)).bitcast(fx.BFloat16))
            # the row's last wave: no main head at 4 splits a row or more
            if (member == splits - 1) & (wave == WAVES - 1):
                _new_key(
                    lane,
                    row,
                    pos,
                    live,
                    req,
                    k_raw,
                    axes,
                    fresh,
                    visible,
                    sm.qx.ptr,
                    heads,
                    ix_eps,
                    ikn_w,
                    cos_sin,
                    mrope,
                    tail,
                    raw_args,
                    (kc, page_bytes, page, comp_blocks, comp_slots, ratio),
                    logits,
                    row * logits_stride,
                    inv,
                )
        else:
            for kq in range_constexpr(DIM // 32):
                v = ld(
                    q,
                    (
                        (row * heads + (n < heads).select(n, i32(0))) * DIM
                        + kq * 32
                        + g * 8
                    )
                    // 2,
                    Int32,
                    4,
                )
                qs.append((n < heads).select(v, zeros(4, Int32)).bitcast(fx.BFloat16))

        def key_loads(r):
            """Round ``r``'s keys of the lane's column of each tile: its
            quarter (``g``) of each 32-dim step, as stored."""
            out = []
            for u in range_constexpr(TILES):
                c = (r0 + r) * ROUND + (wave * TILES + u) * 16 + n
                ok = (r < local) & (c < tile_end)
                c = ok.select(c, i32(0))
                base = lds_ld(table + c // page)
                ok = ok & (base >= 0)
                at = (
                    kc
                    + ok.select(base + Int64(c % page) * Int64(DIM * 2), Int64(0))
                    + Int64(g * 16)
                )
                out.append(
                    [gld(at + Int64(k * 64), Int32, 4) for k in range(DIM // 32)]
                )
            return out

        current = key_loads(i32(0))
        for r in range(i32(0), local, i32(1)):
            pending = key_loads(r + 1)
            for u in range_constexpr(TILES):
                d = zeros(4)
                for k in range_constexpr(DIM // 32):
                    d = mfma_bf16(current[u][k].bitcast(fx.BFloat16), qs[k], d)
                # relu, then the sum over the heads (lanes n of the 16-lane group)
                s = _head_sum([fx.max(d[e], f32(0.0)) for e in range(4)], heads)
                c = (r0 + r) * ROUND + (wave * TILES + u) * 16 + g * 4
                vals = [s[e] * f32(inv) for e in range(4)]
                _store_scores(logits, row * logits_stride, c, tile_end, n, vals, cm)
            current = pending

        if const_expr(top):
            _select_last(
                tid,
                end,
                counters,
                row,
                splits,
                sm.verdict.ptr,
                sm.keys.ptr,
                staged,
                logits,
                logits_stride,
                sel,
                sel_stride,
                top,
            )

    @flyc.jit
    def launch(
        q: Int64,
        kc: Int64,
        page_bytes: Int64,
        bt: Int64,
        bt_stride: Int32,
        reqs: Int32,
        token_to_req: Int64,
        positions: Int64,
        seq_lens: Int64,
        logits: Int64,
        logits_stride: Int32,
        columns: Int32,
        visible_out: Int64,
        sel: Int64,
        sel_stride: Int32,
        counters: Int64,
        ixqk: Int64,
        qkv: Int64,
        q_out: Int64,
        gate_out: Int64,
        rope_pos: Int64,
        pos_axis: Int32,
        cos_sin: Int64,
        qn_w: Int64,
        kn_w: Int64,
        iqn_w: Int64,
        ikn_w: Int64,
        kv: Int64,
        kv_block: Int64,
        kv_page: Int32,
        main_slots: Int64,
        raw: Int64,
        raw_block: Int64,
        raw_row: Int32,
        raw_size: Int32,
        raw_rings: Int32,
        raw_slots: Int64,
        raw_bt: Int64,
        raw_bt_stride: Int32,
        comp_slots: Int64,
        comp_blocks: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(
            q,
            kc,
            page_bytes,
            bt,
            bt_stride,
            reqs,
            token_to_req,
            positions,
            seq_lens,
            logits,
            logits_stride,
            columns,
            visible_out,
            sel,
            sel_stride,
            counters,
            ixqk,
            qkv,
            q_out,
            gate_out,
            rope_pos,
            pos_axis,
            cos_sin,
            qn_w,
            kn_w,
            iqn_w,
            ikn_w,
            kv,
            kv_block,
            kv_page,
            main_slots,
            raw,
            raw_block,
            raw_row,
            raw_size,
            raw_rings,
            raw_slots,
            raw_bt,
            raw_bt_stride,
            comp_slots,
            comp_blocks,
        ).launch(grid=(key.rows * splits,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


_COUNTERS: dict = {}


def _counters(rows: int, device) -> torch.Tensor:
    """The rows' arrival counters (zero; the last arriver leaves its counter
    zero)."""
    k = (rows, str(device))
    if k not in _COUNTERS:
        _COUNTERS[k] = torch.zeros(
            rows * COUNTER_WORDS, dtype=torch.int32, device=device
        )
    return _COUNTERS[k]


class Prep(NamedTuple):
    """What the score launch needs to run vLLM's fused prepare for a decode
    batch (``ScoreBuild.prep``): the projections, the outputs, the rope, the
    norms and every cache with its slots."""

    ixqk: torch.Tensor  # [rows, (heads + 1) 128] bf16: index_qk_proj(x)
    qkv: torch.Tensor  # [rows, QKV_ROW] bf16: qkv_proj(x)
    query: torch.Tensor  # [rows, 12, 256] bf16 out: the main attention's q
    gate: torch.Tensor  # [rows, 12, 256] bf16 out
    rope_pos: torch.Tensor  # int64 [3, >= rows] (MRoPE) or [>= rows]
    cos_sin: torch.Tensor  # bf16 [positions, 64]
    q_norm: torch.Tensor  # the main attention's Gemma norms (256)
    k_norm: torch.Tensor
    ix_q_norm: torch.Tensor  # the indexer's (128)
    ix_k_norm: torch.Tensor
    main_eps: float
    ix_eps: float
    mrope: tuple | None  # (h, w) sections
    kv: torch.Tensor  # [blocks, 1, page, 512] bf16
    main_slots: torch.Tensor  # int64 [>= rows]
    ring: torch.Tensor  # [rings, ring rows, 128 (+ 12 with MRoPE positions)] bf16
    ring_slots: torch.Tensor  # int64 [>= rows]
    ring_table: torch.Tensor  # int32 [reqs, >= 1]: a request's ring at column 0
    comp_slots: torch.Tensor  # int64 [>= rows]: a completed group's row, else -1


def _prep_args(p: Prep | None, rows: int, comp_blocks: int) -> list[int]:
    if p is None:
        return [0] * 25
    assert p.qkv.dtype == p.ixqk.dtype == p.kv.dtype == p.ring.dtype == torch.bfloat16
    assert p.qkv.shape[1] == QKV_ROW and p.qkv.is_contiguous()
    assert p.ixqk.stride(0) == IX_ROW and p.ixqk.stride(1) == 1
    assert p.query.is_contiguous() and p.gate.is_contiguous()
    assert p.query.shape == p.gate.shape == (rows, MAIN_HEADS, 256)
    assert p.cos_sin.dtype == torch.bfloat16 and p.cos_sin.shape[1] == 64
    assert p.cos_sin.is_contiguous()
    for t in (p.rope_pos, p.main_slots, p.ring_slots, p.comp_slots):
        assert t.dtype == torch.int64 and t.stride(-1) == 1
    assert p.ring_table.dtype == torch.int32
    blocks, kv_heads, page, rec = p.kv.shape
    assert kv_heads == 1 and rec == 512 and p.kv.stride(2) == rec
    assert p.kv.stride(3) == 1
    tail = p.ring.shape[2] > DIM
    assert p.ring.stride(2) == 1 and (not tail or p.ring.shape[2] >= DIM + 12)
    axis = p.rope_pos.stride(0) if p.rope_pos.dim() == 2 else 0
    return [
        p.ixqk.data_ptr(),
        p.qkv.data_ptr(),
        p.query.data_ptr(),
        p.gate.data_ptr(),
        p.rope_pos.data_ptr(),
        axis,
        p.cos_sin.data_ptr(),
        p.q_norm.data_ptr(),
        p.k_norm.data_ptr(),
        p.ix_q_norm.data_ptr(),
        p.ix_k_norm.data_ptr(),
        p.kv.data_ptr(),
        p.kv.stride(0) * 2,
        page,
        p.main_slots.data_ptr(),
        p.ring.data_ptr(),
        p.ring.stride(0) * 2,
        p.ring.stride(1) * 2,
        p.ring.shape[1],
        p.ring.shape[0],
        p.ring_slots.data_ptr(),
        p.ring_table.data_ptr(),
        p.ring_table.stride(0),
        p.comp_slots.data_ptr(),
        comp_blocks,
    ]


def score(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    ratio: int,
    logits: torch.Tensor,
    visible: torch.Tensor,
    divisor: float | None = None,
    selected: torch.Tensor | None = None,
    prep: Prep | None = None,
) -> None:
    """``qsa_mqa_paged`` into ``logits`` [rows, columns] f32 and ``visible``
    [rows] int32: ``q`` [rows, heads, 128] bf16, ``k_cache`` [pages,
    page_size, 1, 128] bf16. With ``selected`` [rows, top] int32, also the
    decode top-k's selection (``top_k_per_row_decode(logits, 1, visible,
    selected, ...)``'s set, ascending; -1 past a short row's blocks); a row
    within the budget then leaves its logits unwritten. With ``prep`` (a
    decode batch, ``q`` None), first vLLM's fused prepare of the rows
    (``ops.qsa_prepare``): the queries from the projections, the caches;
    ``positions`` are then the rows' logical positions."""
    from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

    if prep is None:
        rows, heads, dim = q.shape
        assert q.dtype == torch.bfloat16 and q.is_contiguous()
    else:
        rows, heads, dim = logits.shape[0], prep.ixqk.shape[1] // DIM - 1, DIM
    _, page, kv_heads, kdim = k_cache.shape
    assert dim == kdim == DIM and kv_heads == 1
    assert k_cache.dtype == torch.bfloat16
    assert k_cache.stride(1) == DIM and k_cache.stride(3) == 1
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert logits.shape[1] % 4 == 0 and logits.stride(0) % 4 == 0
    assert visible.dtype == token_to_req.dtype == seq_lens.dtype == torch.int32
    assert positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
    assert page_table.dtype == torch.int32 and page_table.stride(1) == 1
    top = 0
    if selected is not None:
        assert selected.dtype == torch.int32 and selected.stride(1) == 1
        assert selected.shape[0] == rows
        top = selected.shape[1]
    divisor = math.sqrt(dim) if divisor is None else divisor
    key = ScoreBuild(
        rows,
        score_splits(rows),
        page,
        page_table.shape[1],
        heads,
        ratio,
        positions.dtype == torch.int64,
        float(divisor),
        top,
        ()
        if prep is None
        else (
            float(prep.main_eps),
            float(prep.ix_eps),
            None if prep.mrope is None else tuple(int(v) for v in prep.mrope),
            prep.ring.shape[2] > DIM,
        ),
    )
    _run_compiled(
        build(key),
        0 if q is None else q.data_ptr(),
        k_cache.data_ptr(),
        k_cache.stride(0) * k_cache.element_size(),
        page_table.data_ptr(),
        page_table.stride(0),
        page_table.shape[0],
        token_to_req.data_ptr(),
        positions.data_ptr(),
        seq_lens.data_ptr(),
        logits.data_ptr(),
        logits.stride(0),
        logits.shape[1],
        visible.data_ptr(),
        0 if selected is None else selected.data_ptr(),
        0 if selected is None else selected.stride(0),
        0 if selected is None else _counters(rows, logits.device).data_ptr(),
        *_prep_args(prep, rows, k_cache.shape[0]),
        torch.cuda.current_stream(),
    )
