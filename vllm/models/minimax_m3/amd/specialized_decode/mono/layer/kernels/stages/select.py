# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K4 stages: the indexer's top-k selection (every block ranked, or a long context's
shares)."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from ...config import (
    MAX_INDEX_BLOCKS,
    SPARSE_BLOCK,
    TOPK_BLOCKS,
    TP,
)
from ...layout import (
    N_SPLIT,
    PAGES_PER_BLOCK,
    SPARSE_TABLE_WORDS,
)
from ...support.device.ops import (
    rsrc,
    traced,
    unlikely,
)
from ...support.device.select import (
    BATCH,
    KEY_LEAST,
    compact,
    count_beaten,
    group_floor,
    put_key,
)
from ...support.device.sync import (
    preg,
    shift,
)
from ...support.plan.arith import (
    pick,
)
from ...support.plan.execution import (
    THREADS,
    WAVES,
)
from .shared import (
    CAND_BATCH,
    CAND_CAP,
)


@traced
def select_defs(k4ctx):
    """The functions of the indexer's top-k selection (every block ranked, or a long
    context's shares)."""
    SY = k4ctx["SY"]
    blk = k4ctx["blk"]
    block_table = k4ctx["block_table"]
    bt_width = k4ctx["bt_width"]
    fuse_k1 = k4ctx["fuse_k1"]
    ih_count = k4ctx["ih_count"]
    ih_own = k4ctx["ih_own"]
    init_blocks = k4ctx["init_blocks"]
    keys = k4ctx["keys"]
    local_blocks = k4ctx["local_blocks"]
    long_rows = k4ctx["long_rows"]
    mb = k4ctx["mb"]
    mb_poll = k4ctx["mb_poll"]
    mb_put_words = k4ctx["mb_put_words"]
    mr = k4ctx["mr"]
    n_blks = k4ctx["n_blks"]
    peer_addr = k4ctx["peer_addr"]
    rank = k4ctx["rank"]
    scored = k4ctx["scored"]
    seq_lens_k = k4ctx["seq_lens_k"]
    stamp = k4ctx["stamp"]
    start = k4ctx["start"]
    sym = k4ctx["sym"]
    tid = k4ctx["tid"]
    tokens = k4ctx["tokens"]
    lane = k4ctx["lane"]
    wave = k4ctx["wave"]
    # ``keys`` (LDS words): a share's entries, CAND_BATCH a thread (two words
    # each), then the candidates, their counter, the groups' maxima
    cand_at = 2 * THREADS * CAND_BATCH
    cand = keys + cand_at
    counter = keys + (cand_at + 2 * CAND_CAP)
    floor_red = counter + 1
    # ``keys`` is the activation pool's start (``post_attn``)
    assert k4ctx["POOL"] >= cand_at + 2 * CAND_CAP + 2 + WAVES * TOPK_BLOCKS
    # a short context's last round of BATCH entries stays in ``keys``
    assert THREADS * CAND_BATCH % BATCH == 0

    def bt_row(k):
        """Token k's block-table row (rows are ``bt_width`` apart)."""
        return block_table + fx.Int64(k) * fx.Int64(bt_width) * 4

    def index_key(b, word, n_blk):
        """Sort key of block b's raw score word: forced blocks pinned, then a
        signed int ordered like the float (-0 == 0, NaN below -inf)."""
        sc = word.bitcast(fx.Float32)
        sc = (b < init_blocks).select(fx.Float32(1e30), sc)
        sc = (b >= n_blk - local_blocks).select(fx.Float32(1e29), sc)
        bits = sc.bitcast(fx.Int32)
        bits = (bits == fx.Int32(-(2**31))).select(fx.Int32(0), bits)
        key = (bits < 0).select(bits ^ 0x7FFFFFFF, bits)
        return ((bits & 0x7FFFFFFF) > 0x7F800000).select(fx.Int32(-(2**31)), key)

    def rank_in_cta(entries, m):
        """Each of this thread's ``entries`` ((key, block, live), entry tid +
        THREADS i of the CTA's m) ranked among the m: the count of entries that
        beat it, exact for the TOPK_BLOCKS best (``device.select``: only the
        entries at least the groups' floor are counted against each other, all
        of them only when more than CAND_CAP tie past it). Barriers included.
        -> the counts."""
        mines = [
            put_key(keys, tid + THREADS * i, h, b, lv)
            for i, (h, b, lv) in enumerate(entries)
        ]
        if tid < CAND_CAP:  # dead: past the candidates every count reads
            fx.ptr_store(fx.Int32(0), cand + 2 * tid)
            fx.ptr_store(fx.Int32(KEY_LEAST), cand + (2 * tid + 1))
        if tid == 0:
            fx.ptr_store(fx.Int32(0), counter)
        stamp(18)
        floor = group_floor(
            [lv.select(h, fx.Int32(KEY_LEAST)) for h, _, lv in entries],
            TOPK_BLOCKS, floor_red, lane, wave,
        )  # fmt: skip
        compact(cand, counter, CAND_CAP, entries, floor)
        gpu.barrier()
        stamp(20)
        n_cand = fx.ptr_load(counter)
        every = n_cand > CAND_CAP
        return count_beaten(
            keys + every.select(0, cand_at), mines, every.select(m, n_cand)
        )

    def rank_blocks(iscore, n_blk, r_bt, scored_k):
        """Block tid of up to LONG_FROM_BLOCKS, ranked by the count of blocks
        that beat it (``rank_in_cta``). ``scored_k``: the scores are read (else
        the blocks rank by id). -> ((key, block, rank, live), its page, the
        tail block's key)."""
        b = tid
        page = fx.Int32(
            bo.buffer_load(r_bt, fx.min(b, n_blk - 1), vec_width=1, dtype=T.i32)
        )
        # a pinned block's slot holds no score (index_key forces its key); past
        # n_blk the key is never read
        if const_expr(fuse_k1):
            # the scorers are CTAs of this launch: poll (a pinned block's slot
            # polls a scored block's)
            slot = fx.min(fx.max(b, init_blocks), n_blk - 1 - local_blocks)
            word = fx.Int32(0)
            if scored_k:
                word = mb_poll([(iscore, slot, 1)])[0][0]
        else:
            # K1 (or a harness's scorer) wrote every score before this launch
            word = fx.Int32(
                bo.buffer_load(rsrc(iscore.value), 2 * b, vec_width=1, dtype=T.i32)
            )
        key = scored_k.select(index_key(b, word, n_blk), -b)
        live = b < n_blk
        (rank,) = rank_in_cta([(key, b, live)], n_blk)
        stamp(19)
        kt = fx.ptr_load(keys + (2 * (n_blk - 1) + 1))
        return (key, b, rank, live), page, kt

    def place(entry, kt, n_blk, seq_len):
        """A ranked block's slot, as the Triton selector emits them: full
        blocks by rank, the tail block (the one holding the current token)
        last. ``entry``: this thread's (key, block, rank, live) -> ((selected,
        slot), sparse context length). One barrier: whether the tail is
        selected is known only once every thread's entry is."""
        key, bb, rank, live = entry
        n_sel = fx.min(n_blk, TOPK_BLOCKS)
        tail = n_blk - 1
        sel = live & (rank < n_sel)
        is_tail = sel & (bb == tail)
        if is_tail:
            fx.ptr_store(fx.Int32(1), blk + 17)
        gpu.barrier()
        tail_sel = fx.ptr_load(blk + 17) != 0
        n_full = n_sel - tail_sel.select(fx.Int32(1), fx.Int32(0))
        tail_first = (kt > key) | ((kt == key) & (tail > bb))
        slot = is_tail.select(
            n_full, rank - tail_first.select(fx.Int32(1), fx.Int32(0))
        )
        n_ctx = tail_sel.select(
            n_full * SPARSE_BLOCK + seq_len - tail * SPARSE_BLOCK,
            fx.min(n_sel * SPARSE_BLOCK, seq_len),
        )
        return (sel, slot), n_ctx

    def write_slot(sel, slot, page, n_ctx, part, table):
        """A selected block's pages into blk (when its slot is split task
        ``part``'s) and, part 0, into the token's sparse table; thread 0 the
        sparse context length."""
        if sel & (slot // 2 == part):
            for j in range_constexpr(PAGES_PER_BLOCK):
                fx.ptr_store(
                    page * (2 * PAGES_PER_BLOCK) + j,
                    blk + ((slot % 2) * PAGES_PER_BLOCK + j),
                )
        if sel & (part == 0):
            for j in range_constexpr(PAGES_PER_BLOCK):
                bo.buffer_store(
                    page * (2 * PAGES_PER_BLOCK) + j, table, slot * PAGES_PER_BLOCK + j
                )
        if tid == 0:
            fx.ptr_store(n_ctx, blk + 16)
            if part == 0:
                bo.buffer_store(n_ctx, table, TOPK_BLOCKS * PAGES_PER_BLOCK)

    def table_of(k):
        return rsrc(mb("sparse_table") + fx.Int64(k) * (SPARSE_TABLE_WORDS * 4))

    def select_blocks(k, t):
        """blk[0:16] := the pages of selection slots 2t, 2t+1 and blk[16] := the
        sparse context length, exactly as the Triton selector emits them: top
        TOPK_BLOCKS by (score, block id) descending with the init blocks pinned
        to 1e30 and the local ones to 1e29, full blocks in that order and the
        tail block (the one holding the current token) last. A block's slot
        follows from its rank, the count of blocks that beat it. k: token.
        A long request's blk is ``stage_select_long``'s: nothing runs here."""
        if ~pick(long_rows, k):
            place_short(k, t)
        gpu.barrier()

    def place_short(k, t):
        """``select_blocks`` of a short request's token k (every thread)."""
        iscore = shift(mr("iscore"), fx.Int64(k) * (MAX_INDEX_BLOCKS * 8))
        n_blk = pick(n_blks, k)
        seq_len = pick(seq_lens_k, k)
        r_bt = rsrc(bt_row(k))
        table = table_of(k)
        if tid < 2 * PAGES_PER_BLOCK + 2:
            fx.ptr_store(fx.Int32(0), blk + tid)
        entry, page, kt = rank_blocks(iscore, n_blk, r_bt, pick(scored, k))
        (sel, slot), n_ctx = place(entry, kt, n_blk, seq_len)
        write_slot(sel, slot, page, n_ctx, t, table)

    def reuse_selection(k, t):
        """Blk as ``select_blocks`` leaves it, read from token k's sparse table:
        a layer reusing a selection (the original path's ``skip_index_topk``)
        attends over the one the step's last selecting layer wrote. Its slots
        past the selection are 0, as a selection leaves blk: the table is
        cleared at the step's start and every selection of a token fills the
        same slots."""
        table = table_of(k)
        if tid < 2 * PAGES_PER_BLOCK + 1:
            j = (tid < 2 * PAGES_PER_BLOCK).select(
                2 * t * PAGES_PER_BLOCK + tid, fx.Int32(TOPK_BLOCKS * PAGES_PER_BLOCK)
            )
            fx.ptr_store(
                fx.Int32(bo.buffer_load(table, j, vec_width=1, dtype=T.i32)), blk + tid
            )
        gpu.barrier()

    def select_long(k, part):
        """Split task (k, part) of a long request's token, before the
        split stage. Its share of the blocks (a thread keys CAND_BATCH) -> each
        wave's TOPK_BLOCKS best -> the CTA's, to a candidate mailbox; then the
        N_SPLIT shares' candidates (the context's TOPK_BLOCKS best are among
        them) are ranked and placed -> blk as ``select_blocks`` leaves it, and
        (part 0) the sparse table. The share: an eighth of the blocks, to this
        rank's sel_long; with every index head (indexer context parallelism)
        head part // 2's scores of half part % 2 of this rank's blocks, to the
        head's rank's sel_cp -- the shares its own head's candidates come in."""
        n_blk = pick(n_blks, k)
        iscore = shift(mr("iscore"), fx.Int64(k) * (MAX_INDEX_BLOCKS * 8))
        first = k * (N_SPLIT * TOPK_BLOCKS)  # token k's candidates (pair pairs)
        table = table_of(k)
        if tid == 0:
            fx.ptr_store(fx.Int32(0), blk + 17)
        if const_expr(ih_count > 1):
            own0 = init_blocks + (ih_own - init_blocks) % TP  # this rank's first
            n_own = (n_blk - own0 + TP - 1) // TP
            # the last scored one: the local blocks' slots hold no score
            last = own0 + TP * ((n_blk - local_blocks - own0 + TP - 1) // TP - 1)
            span = (n_own + 1) // 2
            lo = (part % 2) * span
            hi = fx.min(lo + span, n_own)
            js = [lo + tid + THREADS * i for i in range(CAND_BATCH)]
            blocks = [own0 + TP * j for j in js]
            live = [j < hi for j in js]
            src = shift(iscore, fx.Int64(part // 2) * (MAX_INDEX_BLOCKS // TP * 8))
            slots = [fx.min(b, last) // TP for b in blocks]
            out, out_slot = (
                preg(peer_addr(part // 2), SY["sel_cp"], "sel_cp"),
                2 * rank,
            )
            out_slot = out_slot + part % 2
            got_from = preg(sym, SY["sel_cp"], "sel_cp")
        else:
            span = (n_blk + N_SPLIT - 1) // N_SPLIT
            lo = part * span
            hi = fx.min(lo + span, n_blk)
            blocks = [lo + tid + THREADS * i for i in range(CAND_BATCH)]
            live = [b < hi for b in blocks]
            src = iscore
            slots = [
                fx.min(fx.max(b, init_blocks), n_blk - 1 - local_blocks) for b in blocks
            ]
            out, out_slot = mr("sel_long"), part
            got_from = out
        if const_expr(fuse_k1):
            words = [w[0] for w in mb_poll([(src, sl, 1) for sl in slots])]
        else:
            words = [
                fx.Int32(
                    bo.buffer_load(rsrc(src.value), 2 * sl, vec_width=1, dtype=T.i32)
                )
                for sl in slots
            ]
        # the share's entry tid + THREADS i is its block i of this thread's
        hs = [index_key(b, w, n_blk) for b, w in zip(blocks, words)]
        cnts = rank_in_cta(list(zip(hs, blocks, live)), hi - lo)
        # a share holds >= TOPK_BLOCKS blocks: its 16 best have distinct ranks
        for h, b, lv, c in zip(hs, blocks, live, cnts):
            if lv & (c < TOPK_BLOCKS):
                mb_put_words(out, 2 * (first + out_slot * TOPK_BLOCKS + c), [h, b])
        gpu.barrier()
        # the shares' candidates (every share sends TOPK_BLOCKS distinct ranks:
        # all live), a thread each, ranked among themselves (``count_beaten``);
        # a candidate's page load goes out before the count, under it
        n_c = N_SPLIT * TOPK_BLOCKS
        if tid < n_c:
            got = mb_poll([(got_from, 2 * (first + tid), 2)])[0]
            fx.ptr_store(got[1], keys + 2 * tid)  # put_key's (block, key)
            fx.ptr_store(got[0], keys + (2 * tid + 1))
        gpu.barrier()
        held = tid < n_c
        cl = fx.ptr_load(keys + 2 * fx.min(tid, n_c - 1))
        ch = fx.ptr_load(keys + (2 * fx.min(tid, n_c - 1) + 1))
        page = fx.Int32(
            bo.buffer_load(rsrc(bt_row(k)), fx.max(cl, 0), vec_width=1, dtype=T.i32)
        )
        (cnt,) = count_beaten(keys, [(fx.Int64(ch) << 32) | fx.Int64(cl)], n_c)
        stamp(19)
        # the tail block is a local one: its key is forced
        kt = index_key(n_blk - 1, fx.Int32(0), n_blk)
        (sel, slot), n_ctx = place((ch, cl, cnt, held), kt, n_blk, pick(seq_lens_k, k))
        write_slot(sel, slot, page, n_ctx, part, table)
        gpu.barrier()

    def stage_select_long():
        """The split tasks of a long request's token select before the
        split stage (``select_long``), each leaving its blk for its split task
        (a CTA runs at most one). Out of the split stage and cold: inline
        there, this code slowed the split for every context (64k: +1 us)."""
        ts = start("split")
        k = ts // N_SPLIT
        if unlikely((ts < N_SPLIT * tokens) & pick(long_rows, k)):
            select_long(k, ts % N_SPLIT)

    return {
        "bt_row": bt_row,
        "index_key": index_key,
        "rank_blocks": rank_blocks,
        "place": place,
        "table_of": table_of,
        "select_blocks": select_blocks,
        "reuse_selection": reuse_selection,
        "select_long": select_long,
        "stage_select_long": stage_select_long,
    }
