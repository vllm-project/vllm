# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QSA sparse attention of a decode batch over vLLM's paged K/V cache.

A row attends to the tokens of the compressed blocks its indexer selected
(``sel``: [rows, top_blocks] the top-k's block indices; all of them while
the row has at most ``top_blocks``), each block ``ratio`` tokens, then to
the open tail of its last incomplete block: the list the stock expansion
builds. ``splits`` CTAs a row each attend a contiguous range of the list's
64-token stages:

* The CTA first expands its tokens and translates them through the block
  table into an LDS table of byte offsets into the cache (-1: not
  attended), every load issued before the first use.
* A stage is 16 tokens a wave. The wave multiplies their keys with the kv
  head's queries on the matrix unit (``S^T = K Q^T``, 16x16x32) and keeps a
  running max, sum and full-width output of its own (``O^T += V^T P^T``,
  16x16x16, V read transposed from LDS), with the next stage's K and V loads
  in flight while it computes. The softmax is wave-private: the bounds of a
  joint one, not its reduction order.
* The four waves merge through LDS; with several splits, each stores its
  partial and the row's last-arriving CTA merges them in the ``exp2`` domain
  and resets the row's counter for the next launch. With ``spread`` (four
  splits a row or more, the small batches) every split instead waits until
  the row's splits have all arrived and merges its own slice of the output
  (one CTA reading every partial is the slowest part of a short row).

The kernel reads the cache in place: a token's K and V are one 1 KiB record
(``[blocks, kv heads, block_size, 2 head_dim]``, one kv head on a rank), pages
any stride apart. Rows past the batch's requests (``token_to_req`` out of
range) output zeros, like the stock kernel.
"""

import math
from dataclasses import astuple, dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import WAIT_TICKS
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    CM_DEV,
    LOG2E,
    atomic_add,
    drain_stores,
    gld,
    hw_exp2,
    hw_rcp,
    ld,
    lds_ld,
    lds_ld_tr16,
    lds_st,
    memrealtime,
    mfma_bf16,
    mfma_bf16_k16,
    pack_bf16,
    st,
    traced,
    uniform,
    zeros,
)

i32 = fx.Int32
f32 = fx.Float32

THREADS = 256
WAVES = THREADS // 64
HEAD_DIM = 256
STAGE = 64  # tokens a CTA stages at a time, 16 a wave
COLS = 16  # query-head columns of the matrix instruction
RECORD = 2 * HEAD_DIM * 2  # bytes of a token's K | V
NEG = -1e30
COUNTER_WORDS = 16  # one 64-byte line a row's arrival counter


@dataclass(frozen=True)
class AttnBuild:
    rows: int
    splits: int
    page: int  # tokens a cache block
    top_blocks: int  # the selection's blocks a row
    group: int  # query heads a kv head
    ratio: int  # the indexer's compression ratio
    # a row with at most ``top_blocks`` complete blocks attends all of them
    # without reading ``sel`` (the selection stock would make, in order)
    dense: bool = False
    # every split merges a slice of the row's output once all have arrived
    # (else the last to arrive merges all of it)
    spread: bool = False


def splits_for(rows: int) -> int:
    """CTAs a row: the 256 CUs shared by the rows, at most 16."""
    return max(1, min(16, 1 << max(0, (256 // max(rows, 1)).bit_length() - 1)))


def _stages(key: AttnBuild) -> int:
    """Stages of the longest split, plus the one the last stage prefetches."""
    total = -(-(key.top_blocks * key.ratio + key.ratio - 1) // STAGE)
    return -(-total // key.splits) + 1


def _table_rounds(key: AttnBuild) -> int:
    return -(-_stages(key) * STAGE // THREADS)


_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def build(key: AttnBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    assert key.group <= COLS
    rounds = _table_rounds(key)
    tbl_words = 2 * rounds * THREADS
    splits, group = key.splits, key.group
    top_blocks, ratio = key.top_blocks, key.ratio
    scale = LOG2E / math.sqrt(HEAD_DIM)

    @fx.struct
    class Lds:
        # V stages (bf16 [wave][32 dim chunks][16 tokens][8]), then the waves'
        # outputs (f32 [wave][16 heads][head_dim])
        big: fx.Array[fx.Int32, WAVES * COLS * HEAD_DIM, 16]
        aux: fx.Array[fx.Int32, 2 * WAVES * COLS + 4, 16]
        tbl: fx.Array[fx.Int32, tbl_words, 16]

    name = f"q4w_qsa_attn_s{key.rows}_x{splits}" + ("_d" if key.dense else "")
    name += "_m" if key.spread else ""
    # FlyDSL keys its compile cache on closure scalars and tuples, not on the
    # dataclass: both functions reference the key's fields
    keyed = astuple(key)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        q: Int64,
        kv: Int64,
        page_bytes: Int64,
        sel: Int64,
        sel_stride: Int32,
        bt: Int64,
        bt_stride: Int32,
        bt_width: Int32,
        reqs: Int32,
        token_to_req: Int64,
        seq_lens: Int64,
        out: Int64,
        part: Int64,
        counters: Int64,
    ):
        _ = keyed
        tid = i32(fx.thread_idx.x)
        bid = i32(fx.block_idx.x)
        alloc = fx.SharedAllocator()
        sm = alloc.allocate(Lds).peek()
        vbuf = fx.recast_iter(fx.BFloat16, sm.big.ptr)
        obuf = fx.recast_iter(fx.Float32, sm.big.ptr)
        aux = fx.recast_iter(fx.Float32, sm.aux.ptr)
        verdict = sm.aux.ptr + 2 * WAVES * COLS
        tbl = fx.recast_iter(fx.Int64, sm.tbl.ptr)
        lane, wave = tid % 64, tid // 64
        n, g = lane % 16, lane // 16
        row, member = bid // splits, bid % splits

        req = uniform(ld(token_to_req, row, Int32))
        live = (req >= 0) & (req < reqs)
        req = live.select(req, i32(0))
        seq = live.select(uniform(ld(seq_lens, req, Int32)), i32(0))
        # the list: ``blocks`` selected blocks, then the open tail
        complete = seq // ratio
        blocks = (complete < top_blocks).select(complete, i32(top_blocks))
        expanded = blocks * ratio
        count = expanded + seq % ratio
        total = (count + STAGE - 1) // STAGE
        s0 = member * total // splits
        local = (member + 1) * total // splits - s0

        # the split's token positions, then their byte offsets
        first = s0 * STAGE
        selected = []
        listed = None
        sel_bytes = None
        if const_expr(key.dense):
            listed = complete > i32(top_blocks)
            sel_bytes = i32(key.rows) * sel_stride * i32(4)
        for r in range_constexpr(rounds):
            j = first + tid + r * THREADS
            bi = (j < expanded).select(j // ratio, i32(0))
            if const_expr(key.dense):
                # a short row's list is its blocks in order: no load (masked)
                v = ld(sel, row * sel_stride + bi, Int32, nbytes=sel_bytes, mask=listed)
                selected.append(listed.select(v, bi))  # type: ignore[union-attr]
            else:
                selected.append(ld(sel, row * sel_stride + bi, Int32))
        pos = []
        for r in range_constexpr(rounds):
            j = first + tid + r * THREADS
            in_blocks = j < expanded
            tok = in_blocks.select(
                selected[r] * ratio + j % ratio, complete * ratio + j - expanded
            )
            keep = (
                (j < count)
                & (tok >= 0)
                & (tok < seq)
                & (in_blocks.select(selected[r], i32(0)) >= 0)
            )
            pos.append(keep.select(tok, i32(-1)))
        pages = []
        for r in range_constexpr(rounds):
            slot = (pos[r] >= 0).select(pos[r] // key.page, i32(0))
            slot = (slot < bt_width).select(slot, bt_width - 1)
            pages.append(ld(bt, req * bt_stride + slot, Int32))
        for r in range_constexpr(rounds):
            mapped = (pos[r] >= 0) & (pages[r] >= 0)
            record = Int64(pages[r]) * page_bytes + Int64(pos[r] % key.page) * Int64(
                RECORD
            )
            lds_st(tbl + tid + r * THREADS, mapped.select(record, Int64(-1)))
        fx.barrier()

        head = (n < group).select(n, i32(group - 1))
        qrow = (row * group + head) * (HEAD_DIM // 2)
        qs = [
            ld(q, qrow + k * 16 + g * 4, Int32, 4).bitcast(fx.BFloat16)
            for k in range(8)
        ]

        def kv_loads(s):
            """Stage ``s``'s K and V of the lane's token (``n``): the 32-dim
            steps' 8 dims of the lane's quarter (``g``), as stored."""
            off = lds_ld(tbl + s * STAGE + wave * 16 + n)
            # past the split (the last stage's prefetch) or no token: the
            # cache's first record, one hot line for every lane
            ok = (s < local) & (off >= 0)
            at = kv + ok.select(off, Int64(0)) + Int64(g * 16)
            return [gld(at + Int64(k * 64), Int32, 4) for k in range(8)] + [
                gld(at + Int64(HEAD_DIM * 2 + k * 64), Int32, 4) for k in range(8)
            ]

        current = kv_loads(i32(0))
        top, den = f32(NEG), f32(0.0)
        acc = [zeros(4) for _ in range(16)]
        for s in range(i32(0), local, i32(1)):
            pending = kv_loads(s + 1)
            scores = zeros(4)
            mine = lds_ld(tbl + s * STAGE + wave * 16 + n) >= 0
            for k in range_constexpr(8):
                scores = mfma_bf16(current[k].bitcast(fx.BFloat16), qs[k], scores)
                v = mine.select(current[8 + k], zeros(4, Int32))
                lds_st(
                    vbuf + wave * 4096 + (4 * k + g) * 128 + n * 8,
                    v.bitcast(fx.BFloat16),
                )
            base = s * STAGE + wave * 16 + g * 4
            live4 = [lds_ld(tbl + base + e) >= 0 for e in range(4)]
            sc = [live4[e].select(scores[e] * f32(scale), f32(NEG)) for e in range(4)]
            m = fx.max(fx.max(sc[0], sc[1]), fx.max(sc[2], sc[3]))
            m = fx.max(m, m.shuffle_xor(16, 64))
            m = fx.max(m, m.shuffle_xor(32, 64))
            new_top = fx.max(top, m)
            corr = hw_exp2(top - new_top)
            probs = [
                live4[e].select(hw_exp2(sc[e] - new_top), f32(0.0)) for e in range(4)
            ]
            den = den * corr + probs[0] + probs[1] + probs[2] + probs[3]
            p = (
                fx.Vector.from_elements(probs, fx.Float32)
                .to(fx.BFloat16)
                .bitcast(fx.Int16)
            )
            for mt in range_constexpr(16):
                at = (
                    wave * 4096
                    + (2 * mt + (n % 4) // 2) * 128
                    + (4 * g + n // 4) * 8
                    + (n % 2) * 4
                )
                vt = lds_ld_tr16(vbuf + at)
                scaled = fx.Vector.from_elements(
                    [acc[mt][e] * corr for e in range(4)], fx.Float32
                )
                acc[mt] = mfma_bf16_k16(vt, p, scaled)
            top, current = new_top, pending
        den = den + den.shuffle_xor(16, 64)
        den = den + den.shuffle_xor(32, 64)

        # the waves' merge
        fx.barrier()
        if g == 0:
            lds_st(aux + wave * 16 + n, top)
            lds_st(aux + WAVES * 16 + wave * 16 + n, den)
        for mt in range_constexpr(16):
            lds_st(obuf + (wave * 16 + n) * HEAD_DIM + mt * 16 + g * 4, acc[mt])
        fx.barrier()
        tops = [lds_ld(aux + w * 16 + n) for w in range(WAVES)]
        m = fx.max(fx.max(tops[0], tops[1]), fx.max(tops[2], tops[3]))
        factors = [hw_exp2(tops[w] - m) for w in range(WAVES)]
        tot = f32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + factors[w] * lds_ld(aux + WAVES * 16 + w * 16 + n)
        dims = [wave * 64 + mt * 16 + g * 4 for mt in range(4)]
        nums = []
        for d in dims:
            vals = [f32(0.0)] * 4
            for w in range_constexpr(WAVES):
                o = lds_ld(obuf + (w * 16 + n) * HEAD_DIM + d, 4, fx.Float32)
                vals = [vals[e] + o[e] * factors[w] for e in range(4)]
            nums.append(vals)
        orow = (row * group + n) * (HEAD_DIM // 2)

        if const_expr(splits == 1):
            _store_out(out, orow, n < group, dims, nums, tot)
        else:
            # partial: [row][member][head][head_dim] f32, then
            # [row][member][head] (max, sum)
            pbase = (row * splits + member) * COLS
            if n < group:
                for i in range_constexpr(4):
                    st(
                        part,
                        (pbase + n) * HEAD_DIM + dims[i],
                        fx.Vector.from_elements(nums[i], fx.Float32),
                        CM_DEV,
                    )
            ml = Int64(key.rows * splits * COLS * HEAD_DIM)
            if (wave == 0) & (g == 0):
                st(
                    part + ml * Int64(4),
                    (pbase + n) * 2,
                    fx.Vector.from_elements([m, tot], fx.Float32),
                    CM_DEV,
                )
            if const_expr(key.spread):
                _gather(counters, row, splits, tid)
                _merge_slice(part, ml, out, row, member, splits, group, tid)
                _depart(counters, row, splits, tid)
            elif _arrive(counters, row, splits, tid, verdict):
                mls = [
                    ld(
                        part + ml * Int64(4),
                        ((row * splits + j) * COLS + n) * 2,
                        fx.Float32,
                        2,
                        cm=CM_DEV,
                    )
                    for j in range(splits)
                ]
                mx = mls[0][0]
                for j in range_constexpr(1, splits):
                    mx = fx.max(mx, mls[j][0])
                fs = [hw_exp2(mls[j][0] - mx) for j in range(splits)]
                tot = f32(0.0)
                for j in range_constexpr(splits):
                    tot = tot + fs[j] * mls[j][1]
                merged = []
                for i in range_constexpr(4):
                    vals = [f32(0.0)] * 4
                    for j in range_constexpr(splits):
                        o = ld(
                            part,
                            ((row * splits + j) * COLS + n) * HEAD_DIM + dims[i],
                            fx.Float32,
                            4,
                            cm=CM_DEV,
                        )
                        vals = [vals[e] + o[e] * fs[j] for e in range(4)]
                    merged.append(vals)
                _store_out(out, orow, n < group, dims, merged, tot)

    @flyc.jit
    def launch(
        q: Int64,
        kv: Int64,
        page_bytes: Int64,
        sel: Int64,
        sel_stride: Int32,
        bt: Int64,
        bt_stride: Int32,
        bt_width: Int32,
        reqs: Int32,
        token_to_req: Int64,
        seq_lens: Int64,
        out: Int64,
        part: Int64,
        counters: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(
            q,
            kv,
            page_bytes,
            sel,
            sel_stride,
            bt,
            bt_stride,
            bt_width,
            reqs,
            token_to_req,
            seq_lens,
            out,
            part,
            counters,
        ).launch(grid=(key.rows * splits,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


@traced
def _store_out(out, orow, ok, dims, nums, tot):
    """A thread's 16 normalized outputs of its head (``ok``) as bf16; zeros
    for a row without attended tokens."""
    inv = (tot > f32(0.0)).select(hw_rcp(tot), f32(0.0))
    if ok:
        for i in range_constexpr(4):
            v = [nums[i][e] * inv for e in range(4)]
            words = fx.Vector.from_elements(
                [pack_bf16(v[0], v[1]), pack_bf16(v[2], v[3])], Int32
            )
            st(out, orow + dims[i] // 2, words)


@traced
def _gather(counters, row, arrivals: int, tid):
    """The split's arrival on its row's counter after its partial, then a
    wait (bounded) until every split of the row has arrived: the row's
    splits are co-resident (at most one CTA a CU)."""
    drain_stores()
    fx.barrier()
    if tid == 0:
        addr = counters + Int64(row * (COUNTER_WORDS * 4))
        atomic_add(addr, 1)
        t0 = memrealtime()
        seen = ld(addr, 0, Int32, cm=CM_DEV)
        late = Int64(0)
        while (seen < i32(arrivals)) & (late < Int64(WAIT_TICKS)):
            seen = ld(addr, 0, Int32, cm=CM_DEV)
            late = memrealtime() - t0
    fx.barrier()


@traced
def _depart(counters, row, arrivals: int, tid):
    """After the split's merge: the row's last to leave resets both counters
    for the next launch (every split has seen all arrivals by then)."""
    fx.barrier()
    if tid == 0:
        addr = counters + Int64(row * (COUNTER_WORDS * 4))
        old = atomic_add(addr + Int64(4 * (COUNTER_WORDS // 2)), 1)
        if old == i32(arrivals - 1):
            st(addr, 0, i32(0), cm=CM_DEV)
            st(addr, COUNTER_WORDS // 2, i32(0), cm=CM_DEV)


@traced
def _merge_slice(part, ml, out, row, member, splits: int, group: int, tid):
    """Split ``member``'s slice of the row's output (``HEAD_DIM / splits``
    dims of every head) from all splits' partials, in the ``exp2`` domain."""
    span = HEAD_DIM // splits
    quads = span // 4
    chunks = group * quads
    for r in range_constexpr(-(-chunks // THREADS)):
        c = tid + i32(r * THREADS)
        if c < i32(chunks):
            head = c // i32(quads)
            d = member * i32(span) + c % i32(quads) * i32(4)
            mls = [
                ld(
                    part + ml * Int64(4),
                    ((row * splits + j) * COLS + head) * 2,
                    fx.Float32,
                    2,
                    cm=CM_DEV,
                )
                for j in range(splits)
            ]
            parts = [
                ld(
                    part,
                    ((row * splits + j) * COLS + head) * HEAD_DIM + d,
                    fx.Float32,
                    4,
                    cm=CM_DEV,
                )
                for j in range(splits)
            ]
            mx = mls[0][0]
            for j in range_constexpr(1, splits):
                mx = fx.max(mx, mls[j][0])
            fs = [hw_exp2(mls[j][0] - mx) for j in range(splits)]
            tot = f32(0.0)
            vals = [f32(0.0)] * 4
            for j in range_constexpr(splits):
                tot = tot + fs[j] * mls[j][1]
                vals = [vals[e] + parts[j][e] * fs[j] for e in range(4)]
            inv = (tot > f32(0.0)).select(hw_rcp(tot), f32(0.0))
            words = fx.Vector.from_elements(
                [
                    pack_bf16(vals[0] * inv, vals[1] * inv),
                    pack_bf16(vals[2] * inv, vals[3] * inv),
                ],
                Int32,
            )
            st(out, ((row * group + head) * HEAD_DIM + d) // 2, words)


@traced
def _arrive(counters, row, arrivals: int, tid, verdict):
    """The split's arrival on its row's counter after its partial: True on
    the row's last CTA, which resets the counter."""
    drain_stores()
    fx.barrier()
    if tid == 0:
        addr = counters + Int64(row * (COUNTER_WORDS * 4))
        old = atomic_add(addr, 1)
        last = old == i32(arrivals - 1)
        if last:
            st(addr, 0, i32(0), cm=CM_DEV)
        lds_st(verdict, last.select(i32(1), i32(0)))
    fx.barrier()
    is_last = lds_ld(verdict) != i32(0)
    return is_last


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

_WORKSPACES: dict = {}


def _workspace(rows: int, splits: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The splits' partials and the rows' arrival counters (zero; the last
    arriver leaves its counter zero)."""
    k = (rows, splits, str(device))
    if k not in _WORKSPACES:
        part = torch.empty(
            rows * splits * COLS * (HEAD_DIM + 2), dtype=torch.float32, device=device
        )
        counters = torch.zeros(rows * COUNTER_WORDS, dtype=torch.int32, device=device)
        _WORKSPACES[k] = (part, counters)
    return _WORKSPACES[k]


def sparse_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    seq_lens: torch.Tensor,
    ratio: int,
    out: torch.Tensor,
    splits: int | None = None,
    dense: bool = False,
    spread: bool | None = None,
) -> torch.Tensor:
    """``out`` [rows, heads, 256] bf16: attention of ``q`` [rows, heads, 256]
    over the tokens of the compressed blocks ``selected`` [rows, top_blocks]
    lists (the indexer top-k's) and the open tail, in ``kv_cache`` [blocks, 1,
    block_size, 512] bf16 (K | V a token). ``dense``: a row with at most
    ``top_blocks`` complete blocks attends all of them, its list unread.
    ``spread``: every split merges a slice of the output (``AttnBuild.spread``;
    default: at 4 splits a row or more, where it is faster)."""
    from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

    rows, group, hd = q.shape
    blocks, kv_heads, page, rec = kv_cache.shape
    assert hd == HEAD_DIM and kv_heads == 1 and rec == 2 * HEAD_DIM
    assert q.dtype == kv_cache.dtype == out.dtype == torch.bfloat16
    assert q.is_contiguous() and out.is_contiguous() and out.shape == q.shape
    assert kv_cache.stride(3) == 1 and kv_cache.stride(2) == rec
    assert selected.dtype == block_table.dtype == token_to_req.dtype == torch.int32
    assert seq_lens.dtype == torch.int32 and selected.stride(1) == 1
    assert block_table.stride(1) == 1
    splits = splits or splits_for(rows)
    key = AttnBuild(
        rows,
        splits,
        page,
        selected.shape[1],
        group,
        ratio,
        dense,
        (splits >= 4 if spread is None else spread) and splits > 1,
    )
    part, counters = _workspace(rows, splits, q.device)
    _run_compiled(
        build(key),
        q.data_ptr(),
        kv_cache.data_ptr(),
        kv_cache.stride(0) * kv_cache.element_size(),
        selected.data_ptr(),
        selected.stride(0),
        block_table.data_ptr(),
        block_table.stride(0),
        block_table.shape[1],
        block_table.shape[0],
        token_to_req.data_ptr(),
        seq_lens.data_ptr(),
        out.data_ptr(),
        part.data_ptr(),
        counters.data_ptr(),
        torch.cuda.current_stream(),
    )
    return out
