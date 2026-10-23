# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 prefill index scorer for MiniMax-M3 (FlyDSL, gfx950).

Per (query row, block): the max of q . k over the row's causal tokens, with
AITER's init / local sentinels, on e4m3_rn_satfinite(e2m1 x scale) keys. A
workgroup takes 512 query rows (token * H + head) x a segment of the request's
blocks; Q^T stays in VGPRs and each 16-token tile is one
v_mfma_f32_16x16x128_f8f6f4 per 16-row N-tile. Raw NVFP4 pages arrive in LDS by
DMA one block ahead, and block b + 1 is dequantized into the other key slot
under block b's MFMAs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    as_f32,
    bits,
    dequant8,
    global_i32_ptr,
    key_order4,
    maxf_nn,
    mfma_fp8_16x16x128,
    pack8,
    permlane_swap,
)

NW = 4
ROWS_PER_WAVE = 128
TILE_Q = NW * ROWS_PER_WAVE
BLK = 128  # tokens per block
D = 128  # index head dim
ROW_BYTES = D  # one e4m3 key row in LDS
BLOCK_BYTES = BLK * ROW_BYTES
RS_T = ROW_BYTES // 16  # 16 B tiles per LDS row
SLOT_T = BLK * RS_T  # 16 B tiles per key slot
N_SLOTS = 2
LDS_BYTES = N_SLOTS * BLOCK_BYTES
PAGE_CODES = BLK * D // 2  # page: [128 x 64 B] e2m1 codes, [128 x 8 B] e4m3 scales
PAGE_BYTES = PAGE_CODES + BLK * 8
GROUP = 4  # blocks per score store (16 B per row)
# SPAN groups of a row are stored back to back (the first SPAN - 1 wait in LDS):
# half-written 128 B score lines get evicted from L2 and cost extra write bytes.
SPAN = 4
STAGE_T = (SPAN - 1) * NW * 2 * 64  # 16 B LDS tiles: (group, wave, row half, lane)
NI = ROWS_PER_WAVE // 16  # N-tiles per wave
MT = BLK // 16  # M-tiles per block
SENTINEL_INIT = 1e30  # AITER's forced-selection scores
SENTINEL_LOCAL = 1e29
STAGE_W = 2048 + 256  # raw bytes per wave and block: 32 tokens' codes + scales
RAW_SLOTS = 2
# M-tile -> code dwords dequantized under its MFMAs.
DQ_AT = {1: (0,), 2: (1,), 3: (2,), 4: (3,), 5: (4,), 6: (5,), 7: (6, 7)}


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dma(rsrc, lds_addr, voff, size):
    """buffer_load ... lds: ``size`` bytes per lane into LDS at ``lds_addr``."""
    lds = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), _ir(fx.Int32(lds_addr)))
    fx.rocdl.raw_ptr_buffer_load_lds(
        rsrc,
        lds.result,
        fx.Int32(size),
        fx.Int32(voff),
        fx.Int32(0),
        fx.Int32(0),
        aux=0,
    )


def _xlane_max4_pair(x, y):
    """Max over lanes L, L^16, L^32, L^48 of two values at once, broadcast."""
    a, b = permlane_swap(_rocdl.permlane32_swap, bits(x), bits(y))
    m = bits(maxf_nn(as_f32(a), as_f32(b)))
    a, b = permlane_swap(_rocdl.permlane16_swap, m, m)
    big = bits(maxf_nn(as_f32(a), as_f32(b)))
    a, b = permlane_swap(_rocdl.permlane32_swap, big, big)
    return as_f32(a), as_f32(b)


def tile_rows(pid, i32_qt, i32_nseg, i32_nb, arg_cu, arg_seq, heads, tile_q):
    """Workgroup pid's request b, query tile and context segment.

    Returns (live, b, seg, seq_start, seq_len, prefix, q_rows, row0); padded
    workgroups read request 0 and are not live.
    """
    n_wg = i32_qt * i32_nseg * i32_nb
    q_tile = pid % i32_qt
    seg = (pid // i32_qt) % i32_nseg
    b = pid // (i32_qt * i32_nseg)
    bc = (pid < n_wg).select(b, fx.Int32(0))
    cu = global_i32_ptr(arg_cu)
    seq_start = fx.Int32(cu[bc])
    q_len = fx.Int32(cu[bc + 1]) - seq_start
    seq_len = fx.Int32(global_i32_ptr(arg_seq)[bc])
    prefix = seq_len - q_len
    q_rows = q_len * heads
    row0 = q_tile * tile_q
    live = (pid < n_wg) & (row0 < q_rows)
    return live, b, seg, seq_start, seq_len, prefix, q_rows, row0


def segment_blocks(prefix, row0, seq_len, seg, i32_nseg, heads, tile_q, align):
    """The tile's nblk causal blocks and its segment [blk0, blk1) of them."""
    hi_pos = prefix + (row0 + tile_q) // heads
    hi = (hi_pos < seq_len).select(hi_pos, seq_len)
    nblk = (hi + (BLK - 1)) // BLK
    seg_len = (nblk + i32_nseg - 1) // i32_nseg
    if align > 1:
        seg_len = (seg_len + (align - 1)) & ~(align - 1)
    blk0 = seg * seg_len
    blk1e = blk0 + seg_len
    blk1 = (blk1e < nblk).select(blk1e, nblk)
    return nblk, blk0, blk1


def score_resources(arg_q, arg_score, i32_total_q, i32_hs, i32_S, heads):
    """Buffer resources of the Q rows and the scores."""
    qr = buffer_ops.create_buffer_resource_from_addr(
        arg_q, num_records_bytes=fx.Int64(i32_total_q) * (heads * D)
    )
    sr = buffer_ops.create_buffer_resource_from_addr(
        arg_score,
        num_records_bytes=(
            fx.Int64(i32_hs) * (heads - 1) + fx.Int64(i32_total_q) * fx.Int64(i32_S)
        )
        * 4,
    )
    return qr, sr


def page_lookup(arg_bt, bt_row, end):
    """Page of a block, clamped to the last block before ``end``.

    A constant-space s_load: before an s_buffer_load LLVM would wait for the
    in-flight LDS DMAs.
    """
    bt = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<4>"), _ir(fx.Int64(arg_bt))).result

    def page_of(blk):
        bi = (blk < end).select(blk, end - 1)
        p = buffer_ops.get_element_ptr(
            bt, byte_offset=_ir((bt_row + bi) * 4), elem_type=T.i8
        )
        return fx.Int32(llvm.LoadOp(T.i32, p, alignment=4, invariant=True).result)

    return page_of


def with_sentinels(v, blk, nb, i32_init, i32_local):
    """AITER's init / local sentinels on block blk of a row of nb causal blocks."""
    is_local = (blk >= nb - i32_local) & (blk < nb)
    is_init = (blk < i32_init) & (blk < nb)
    return is_local.select(
        fx.Float32(SENTINEL_LOCAL), is_init.select(fx.Float32(SENTINEL_INIT), v)
    )


def compile_score_prefill(heads: int = 1):
    """Build the scorer for 1, 2 or 4 index heads (sharing one index-K head)."""
    assert heads in (1, 2, 4), heads
    H = heads
    name = f"m3_nvfp4_index_score_prefill_tq{TILE_Q}_h{H}"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # e4m3 key slots
        lut: fx.Array[fx.Int32, 512, 16]
        raw: fx.Array[fx.Uint8, RAW_SLOTS * NW * STAGE_W, 16]  # DMA'd pages
        stage: fx.Array[fx.Int32, STAGE_T * 4, 16]  # staged score groups

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [total_q, H, 128] e4m3
        arg_kv: fx.Int64,  # [num_blocks, 128, 72] uint8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] int64
        arg_score: fx.Int64,  # [H, total_q, S] fp32
        arg_bt: fx.Int64,  # [batch, >= max_blocks] int32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] int32
        arg_seq: fx.Int64,  # seq_lens [batch] int32
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,  # score head stride
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        live, b, seg, seq_start, seq_len, prefix, q_rows, row0 = tile_rows(
            pid, i32_qt, i32_nseg, i32_nb, arg_cu, arg_seq, H, TILE_Q
        )
        if live:
            nblk, blk0, blk1 = segment_blocks(
                prefix, row0, seq_len, seg, i32_nseg, H, TILE_Q, GROUP
            )
            if blk0 < blk1:
                qr, sr = score_resources(
                    arg_q, arg_score, i32_total_q, i32_hs, i32_S, H
                )
                page_of = page_lookup(arg_bt, b * i32_bt_stride, nblk)

                # Lane l of wave w owns token 32 w + l / 2, dims 64 (l % 2) ..
                # +63. The DMA writes LDS lane-linearly, so each lane reads back
                # its own code and scale bytes.
                raw_base_s = fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(smem.raw.ptr)))
                )
                code_voff = [wave * 2048 + lane * 32 + h * 16 for h in range(2)]
                scale_voff = PAGE_CODES + wave * 256 + lane * 4

                def dma_raw(page, st):
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page) * i64_page_stride,
                        num_records_bytes=PAGE_BYTES,
                    )
                    base = raw_base_s + (st * NW + wave) * STAGE_W
                    for h in range_constexpr(2):
                        _dma(kr, base + h * 1024, code_voff[h], 16)
                    _dma(kr, base + 2048, scale_voff, 4)

                raw_t = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.raw.ptr),
                        fx.make_layout(RAW_SLOTS * NW * STAGE_W // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                raw_d = smem.raw.ptr

                def read_raw(st):
                    """This lane's 8 code dwords and its scale dword."""
                    b4 = (st * NW + wave) * (STAGE_W // 4)
                    out = []
                    for h in range_constexpr(2):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(
                            fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32),
                            fx.slice(raw_t, (None, b4 // 4 + h * 64 + lane)),
                            r,
                        )
                        v = Vec(r.load())
                        out += [fx.Int32(v[i]) for i in range_constexpr(4)]
                    sc = fx.Int32(
                        fx.make_view(
                            fx.recast_iter(fx.Int32, raw_d),
                            fx.make_layout(RAW_SLOTS * NW * STAGE_W // 4, 1),
                        )[b4 + 512 + lane]
                    )
                    return out + [sc]

                pg = [page_of(blk0 + i) for i in range_constexpr(2)]
                for i in range_constexpr(2):
                    dma_raw(pg[i], i)
                pg_next = page_of(blk0 + 2)

                lut_g = global_i32_ptr(arg_lut)
                lut_l = smem.lut.ptr
                for i in range_constexpr(2):
                    lut_l[tx * 2 + i] = fx.Int32(lut_g[tx * 2 + i])

                # Rows past q_len * H read the next request's Q and are masked at
                # the store.
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r // H for r in rows]
                # Q^T in the key's dim order (even dims, then odd dims per 8).
                qf = []
                for ni in range_constexpr(NI):
                    qrow = (seq_start * H + rows[ni]) * ROW_BYTES
                    halves = []
                    for h in range_constexpr(2):
                        d = Vec(
                            buffer_ops.buffer_load(
                                qr,
                                (qrow + h * 64 + q16 * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        d = [fx.Int32(d[i]) for i in range_constexpr(4)]
                        halves.append(key_order4(d))
                    qf.append(pack8(halves[0], halves[1]))

                lds16 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.a.ptr),
                        fx.make_layout(LDS_BYTES // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
                lut8 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
                    ),
                    fx.make_layout(2, 1),
                )
                lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

                def lds_load16(tile):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
                    return r.load()

                def lds_store16(tile, vals):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(Vec.from_elements(vals, fx.Int32))
                    fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

                def lut_load(sbyte):
                    r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                    fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
                    v = Vec(r.load())
                    return fx.Int32(v[0]), fx.Int32(v[1])

                # 16 B column c of key row t sits at c ^ ((t >> 1) & 7).
                t_row = wave * 32 + (lane >> 1)
                hh = lane & 1
                wr_base = t_row * RS_T
                wr_sw = (t_row >> 1) & 7

                def dq_tables(raw):
                    sc = raw[8]
                    return [
                        lut_load((sc >> (8 * g)) & 0xFF) for g in range_constexpr(4)
                    ]

                def dq_step(raw, tabs, dslot, k, pend):
                    """Code dword k -> 8 keys; every second one stores 16 B."""
                    lo, hi = dequant8(raw[k], *tabs[k // 2])
                    if const_expr(k % 2 == 0):
                        return [lo, hi]
                    col = (hh * 4 + k // 2) ^ wr_sw
                    lds_store16(dslot * SLOT_T + wr_base + col, pend + [lo, hi])
                    return []

                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                rd_row = l16 * RS_T
                rd_col = [q16 ^ (l16 >> 1), (4 + q16) ^ (l16 >> 1)]

                def compute_block(slot, blk, masked, st, dslot):
                    """This wave's scores of ``blk``; dequantizes the next block's
                    raw page ``st`` into key slot ``dslot`` on the way."""
                    if const_expr(masked):
                        vis = [
                            qpos[ni] - (blk * BLK + q16 * 4)
                            for ni in range_constexpr(NI)
                        ]
                    run = [neg_inf for _ in range(NI)]
                    raw = read_raw(st)
                    tabs = dq_tables(raw)

                    def read_a(mt):
                        base = slot * SLOT_T + mt * 16 * RS_T + rd_row
                        return pack8(
                            lds_load16(base + rd_col[0]), lds_load16(base + rd_col[1])
                        )

                    accs = [[None] * NI, [None] * NI]

                    def issue(mt, a):
                        acc = accs[mt % 2]
                        for ni in range_constexpr(NI):
                            acc[ni] = mfma_fp8_16x16x128(a, qf[ni], zero4)

                    def reduce(mt):
                        acc = accs[mt % 2]
                        for ni in range_constexpr(NI):
                            v = acc[ni]
                            for i in range_constexpr(4):
                                x = v[i]
                                if const_expr(masked):
                                    x = (vis[ni] >= mt * 16 + i).select(x, neg_inf)
                                run[ni] = maxf_nn(run[ni], x)

                    pend = []
                    a_next = read_a(0)
                    for mt in range_constexpr(MT):
                        a = a_next
                        if const_expr(mt + 1 < MT):
                            a_next = read_a(mt + 1)
                        issue(mt, a)
                        if const_expr(mt >= 1):
                            reduce(mt - 1)
                        ks = DQ_AT.get(mt, ())
                        for k in ks:
                            pend = dq_step(raw, tabs, dslot, k, pend)
                        n_valu = (16 if not masked else 48) + 10 * len(ks)
                        n_dsw = sum(k % 2 for k in ks)
                        for i in range_constexpr(NI):
                            fx.rocdl.sched_group_barrier(0x008, 1, 0)
                            fx.rocdl.sched_group_barrier(
                                0x002, (n_valu + NI - 1) // NI, 0
                            )
                            if const_expr(i < 2):
                                fx.rocdl.sched_group_barrier(0x100, 1, 0)
                            if const_expr(i == 4 and n_dsw):
                                fx.rocdl.sched_group_barrier(0x200, n_dsw, 0)
                        fx.rocdl.sched_barrier(0)
                    reduce(MT - 1)
                    out = []
                    for p in range_constexpr(NI // 2):
                        x, y = _xlane_max4_pair(run[2 * p], run[2 * p + 1])
                        out += [x, y]
                    return out

                stage16 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.stage.ptr),
                        fx.make_layout(STAGE_T * 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )

                def stage_tile(k, h):
                    return ((k * NW + wave) * 2 + h) * 64 + lane

                def store_group(g0, sc):
                    """Scores sc[j][ni] of blocks g0 .. g0 + 3, sentinels applied.

                    Lane q16 = k stores rows 2k and 2k + 1. A group waits in LDS
                    unless it ends its SPAN piece or the segment; that one stores
                    the piece's staged groups and itself back to back.
                    """
                    full = g0 + GROUP <= blk1
                    pos = (g0 >> 2) & (SPAN - 1)
                    piece0 = g0 & ~(SPAN * GROUP - 1)
                    first = (piece0 < blk0).select(
                        (blk0 >> 2) & (SPAN - 1), fx.Int32(0)
                    )
                    flush = (pos == SPAN - 1) | (g0 + GROUP >= blk1)

                    def store_now(vals, off, valid):
                        """16 B per row, single dwords for a partial tail."""
                        if full:
                            buffer_ops.buffer_store(
                                fx.Vector.from_elements(vals, fx.Float32),
                                sr,
                                off,
                                mask=valid,
                            )
                        else:
                            for j in range_constexpr(GROUP):
                                buffer_ops.buffer_store(
                                    vals[j], sr, off + j, mask=valid & (g0 + j < blk1)
                                )

                    def flush_piece(h, vals, off, valid):
                        for k in range_constexpr(SPAN - 1):
                            t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(
                                lds_atom, fx.slice(stage16, (None, stage_tile(k, h))), t
                            )
                            buffer_ops.buffer_store(
                                Vec(t.load()),
                                sr,
                                off - (pos - k) * GROUP,
                                mask=valid & (first <= k) & (pos > k),
                            )
                        store_now(vals, off, valid)

                    def stage_group(h, vals):
                        t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        t.store(Vec.from_elements([bits(x) for x in vals], fx.Int32))
                        fx.copy(
                            lds_atom, t, fx.slice(stage16, (None, stage_tile(pos, h)))
                        )

                    for h in range_constexpr(2):
                        row = rows[h]
                        vals = [sc[j][h] for j in range_constexpr(GROUP)]
                        for k in range_constexpr(1, NW):
                            is_k = q16 == k
                            row = is_k.select(rows[2 * k + h], row)
                            vals = [
                                is_k.select(sc[j][2 * k + h], vals[j])
                                for j in range_constexpr(GROUP)
                            ]
                        valid = row < q_rows
                        tok = row // H
                        nb_row = (prefix + tok) // BLK + 1
                        for j in range_constexpr(GROUP):
                            blk = g0 + j
                            vals[j] = with_sentinels(
                                vals[j], blk, nb_row, i32_init, i32_local
                            )
                        off = (row % H) * i32_hs + (seq_start + tok) * i32_S + g0
                        if flush:
                            flush_piece(h, vals, off, valid)
                        else:
                            stage_group(h, vals)

                n_groups = (blk1 - blk0 + (GROUP - 1)) // GROUP
                # Groups every row of the tile sees whole need no causal mask.
                v = prefix + row0 // H - (BLK - 1)
                last_full = (v < 0).select(fx.Int32(-1), v // BLK)
                d = last_full - (GROUP - 1) - blk0
                nf = (d < 0).select(fx.Int32(0), d // GROUP + 1)
                n_full = (nf < n_groups).select(nf, n_groups)

                # Prologue: block blk0 dequantized into slot 0.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()
                raw0 = read_raw(0)
                tabs0 = dq_tables(raw0)
                pend = []
                for k in range_constexpr(8):
                    pend = dq_step(raw0, tabs0, 0, k, pend)
                fx.gpu.barrier()

                def group(g0, pg_next, masked):
                    """Blocks g0 .. g0 + 3: per block b, DMA block b + 2's page,
                    score b while dequantizing b + 1, then wait and barrier."""
                    sc = []
                    for j in range_constexpr(GROUP):
                        blk = g0 + j
                        dma_raw(pg_next, (j + 2) % RAW_SLOTS)
                        pg_next = page_of(blk + 3)
                        sc.append(
                            compute_block(
                                j % N_SLOTS,
                                blk,
                                masked,
                                (j + 1) % RAW_SLOTS,
                                (j + 1) % N_SLOTS,
                            )
                        )
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        fx.gpu.barrier()
                    store_group(g0, sc)
                    return pg_next

                def one(st):
                    return fx.Int32(st[0] if isinstance(st, (list, tuple)) else st)

                for g, st in range(
                    fx.Index(0), fx.Index(n_full), fx.Index(1), init=[_ir(pg_next)]
                ):
                    pgn = group(blk0 + fx.Int32(g) * GROUP, one(st), False)
                    res = yield [_ir(pgn)]
                for g, st in range(
                    fx.Index(n_full),
                    fx.Index(n_groups),
                    fx.Index(1),
                    init=[_ir(one(res))],
                ):
                    pgn = group(blk0 + fx.Int32(g) * GROUP, one(st), True)
                    res = yield [_ir(pgn)]

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_hs: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_cu,
            arg_seq,
            i64_page_stride,
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i32_nb,
            i32_init,
            i32_local,
            i32_hs,
            # Without the hint LLVM picks 264 VGPRs, one wave per SIMD.
            value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
