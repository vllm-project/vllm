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
"""

import math
from dataclasses import astuple, dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen4_exp.amd.specialized_decode.wide.attention import splits_for
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    _xshfl,
    gld,
    ld,
    lds_ld,
    lds_st,
    mfma_bf16,
    st,
    uniform,
    zeros,
)

i32 = fx.Int32
f32 = fx.Float32

THREADS = 256
WAVES = THREADS // 64
DIM = 128  # an indexer head's dims
TILES = 4  # 16-column tiles a wave scores a round
ROUND = WAVES * TILES * 16  # columns a CTA scores a round


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


def _head_sum(vals, heads: int):
    """Sums over the 16-lane group's first ``heads`` lanes (the others hold
    zeros), into its lane 0."""
    off = 1
    while off < heads:
        vals = [v + _xshfl(v, off) for v in vals]
        off *= 2
    return vals


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

    @fx.struct
    class Lds:
        table: fx.Array[fx.Int32, 2 * table_rounds * THREADS, 16]

    @flyc.kernel(
        name=f"q4w_qsa_score_s{key.rows}_x{splits}", known_block_size=[THREADS, 1, 1]
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
        rounds = (end + ROUND - 1) // ROUND
        r0 = member * rounds // splits
        local = (member + 1) * rounds // splits - r0

        # the row's pages: byte offsets in the cache (-1: none)
        for tr in range_constexpr(table_rounds):
            p = tid + tr * THREADS
            inside = p < key.pages
            entry = ld(bt, req * bt_stride + inside.select(p, i32(0)), Int32)
            ok = inside & (entry >= 0)
            lds_st(table + p, ok.select(Int64(entry) * page_bytes, Int64(-1)))
        fx.barrier()

        qs = []
        for kq in range_constexpr(DIM // 32):
            v = ld(
                q,
                ((row * heads + (n < heads).select(n, i32(0))) * DIM + kq * 32 + g * 8)
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
                ok = (r < local) & (c < end)
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
                if (n == 0) & (c < end):
                    vals = fx.Vector.from_elements(
                        [s[e] * f32(inv) for e in range(4)], fx.Float32
                    )
                    st(logits, row * logits_stride + c, vals)
            current = pending

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
        ).launch(grid=(key.rows * splits,), block=(THREADS,), stream=stream)

    _BUILDS[key] = launch
    return launch


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
) -> None:
    """``qsa_mqa_paged`` into ``logits`` [rows, columns] f32 and ``visible``
    [rows] int32: ``q`` [rows, heads, 128] bf16, ``k_cache`` [pages,
    page_size, 1, 128] bf16."""
    from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

    rows, heads, dim = q.shape
    _, page, kv_heads, kdim = k_cache.shape
    assert dim == kdim == DIM and kv_heads == 1
    assert q.dtype == k_cache.dtype == torch.bfloat16 and q.is_contiguous()
    assert k_cache.stride(1) == DIM and k_cache.stride(3) == 1
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert logits.shape[1] % 4 == 0 and logits.stride(0) % 4 == 0
    assert visible.dtype == token_to_req.dtype == seq_lens.dtype == torch.int32
    assert positions.dtype in (torch.int32, torch.int64) and positions.is_contiguous()
    assert page_table.dtype == torch.int32 and page_table.stride(1) == 1
    divisor = math.sqrt(dim) if divisor is None else divisor
    key = ScoreBuild(
        rows,
        splits_for(rows),
        page,
        page_table.shape[1],
        heads,
        ratio,
        positions.dtype == torch.int64,
        float(divisor),
    )
    _run_compiled(
        build(key),
        q.data_ptr(),
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
        torch.cuda.current_stream(),
    )
