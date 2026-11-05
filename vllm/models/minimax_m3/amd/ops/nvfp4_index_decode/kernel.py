# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 decode index scorer for MiniMax-M3 (FlyDSL, gfx950).

Per (head, query row, block): the max of q . k over the row's visible tokens,
with AITER's init / local sentinels, on e4m3_rn_satfinite(e2m1 x scale) keys. A
workgroup takes a chunk of up to 64 requests (lane r holds request r) and every
splits-th block position. The distinct pages at a position (a shared prefix page
is one page for the whole chunk) go round robin over the 4 waves; a wave
dequantizes its page into the A fragments of v_mfma_f32_16x16x128_f8f6f4, issuing
the next page's loads into the freed registers, and scores it against every
16-column group of Q rows holding the page.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    NT,
    as_f32,
    bits,
    dequant8,
    global_i32_ptr,
    key_order4,
    maxf_nn,
    mfma_fp8_16x16x128,
    pack8,
    xlane4,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_prefill.kernel import with_sentinels

NW = 4
MAX_REQS = 64  # requests per chunk, one per lane
BLK = 128  # tokens per block
D = 128  # index head dim
PAGE_CODES = BLK * D // 2  # page: [128 x 64 B] e2m1 codes, [128 x 8 B] e4m3 scales
PAGE_BYTES = PAGE_CODES + BLK * 8
MT = BLK // 16  # token tiles per page
UMAX = 8  # max consecutive blocks per LDS-buffered unit
MAX_ROWS = 32  # H * query_len


def padded_rows(rows: int) -> int:
    """Columns a request takes in the 16-column groups: rows rounded up to a
    power of two."""
    return 1 << (rows - 1).bit_length()


def chunk_max(rows: int) -> int:
    """Requests per chunk: one per lane, and 256 Q rows (32 KB) in LDS."""
    return min(MAX_REQS, 256 // rows)


def block_sentinels(mx, visible, b, i32_init, i32_local):
    """AITER's sentinels on block b of a row seeing ``visible`` tokens."""
    nb = ((visible < 0).select(fx.Int32(0), visible) + (BLK - 1)) // BLK
    return with_sentinels(mx, b, nb, i32_init, i32_local)


def compile_score_decode(rows: int, heads: int, buffered: bool = False):
    """Build the scorer.

    Args:
        rows: H * query_len (1..32).
        heads: index heads (1, 2 or 4).
        buffered: gather a unit's scores in LDS and store whole rows (large
            score footprints); otherwise store them directly.

    """
    assert 1 <= rows <= MAX_ROWS and heads in (1, 2, 4) and rows % heads == 0
    H = heads
    QL = rows // H
    R = padded_rows(rows)  # a request's columns; column j = query row j // H
    RPC = max(1, 16 // R)  # requests per 16-column group
    CPR = max(1, R // 16)  # 16-column groups per request
    GMASK = (1 << RPC) - 1
    QROWS = chunk_max(R) * R
    QROW_BYTES = D
    name = f"m3_nvfp4_index_score_decode_r{rows}h{H}" + ("_buf" if buffered else "")
    SB_STRIDE = UMAX + 1  # padded against LDS bank conflicts

    @fx.struct
    class Shared:
        q: fx.Array[fx.Uint8, QROWS * QROW_BYTES, 16]
        lut: fx.Array[fx.Int32, 512, 16]
        seq: fx.Array[fx.Int32, MAX_REQS]
        # The unit's scores as f32 bits, plus 64 dummy slots.
        sbuf: fx.Array[fx.Int32, (QROWS * SB_STRIDE + 64) if buffered else 16, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [num_reqs * QL, H, 128] e4m3
        arg_kv: fx.Int64,  # [num_blocks, 128, 72] uint8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] int64
        arg_score: fx.Int64,  # [H, num_reqs * QL, S] fp32
        arg_bt: fx.Int64,  # [num_reqs, >= max_blocks] int32
        arg_seq: fx.Int64,  # [num_reqs] int32
        i64_page_stride: fx.Int64,
        i32_q_st: fx.Int32,  # q strides (bytes)
        i32_q_sh: fx.Int32,
        i64_q_bytes: fx.Int64,
        i32_s_sh: fx.Int32,  # score strides (elements)
        i32_s_st: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_chunk: fx.Int32,  # requests per chunk
        i32_nch: fx.Int32,
        i32_splits: fx.Int32,  # workgroups per chunk
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_per_wave: fx.Int32,  # 1: each wave walks its own positions
        i32_unit: fx.Int32,  # consecutive positions per unit, 0: none
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        col, kl = lane % 16, lane // 16
        ch = pid % i32_nch
        split = pid // i32_nch
        req0 = ch * i32_chunk
        left = i32_num_reqs - req0
        nr = (left < i32_chunk).select(left, i32_chunk)

        # Lane r holds request req0 + r.
        r_ok = lane < nr
        seq_r = fx.Int32(global_i32_ptr(arg_seq)[r_ok.select(req0 + lane, fx.Int32(0))])
        seq_r = r_ok.select(seq_r, fx.Int32(0))
        nblk_r = (seq_r + (BLK - 1)) // BLK
        first_row = seq_r - (QL - 1)
        # Every row sees blocks before rbf_r whole: init / local ones there only
        # need their sentinel.
        rbf_r = ((first_row < 0).select(fx.Int32(0), first_row) + (BLK - 1)) // BLK
        ll = nblk_r - i32_local
        loc_last_r = (ll < 0).select(fx.Int32(0), ll)
        fm_r = seq_r - QL  # tokens past this are masked for some row

        btr = buffer_ops.create_buffer_resource_from_addr(
            arg_bt, num_records_bytes=i64_bt_bytes
        )
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_score, num_records_bytes=i64_s_bytes
        )
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=i64_q_bytes
        )

        q16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.q.ptr),
                fx.make_layout(QROWS * QROW_BYTES // 4, 1),
            ),
            fx.make_layout(4, 1),
        )
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        seq_l = smem.seq.ptr
        scb = smem.sbuf.ptr
        per_wave = i32_per_wave != fx.Int32(0)
        if const_expr(buffered):
            direct = per_wave | (i32_unit == fx.Int32(0))
            to_lds = (i32_per_wave == fx.Int32(0)) & (i32_unit != fx.Int32(0))
            usz = (i32_unit == fx.Int32(0)).select(fx.Int32(1), i32_unit)
        else:
            direct = i32_unit >= fx.Int32(0)
            to_lds = i32_unit < fx.Int32(0)
            usz = fx.Int32(1)

        def prologue():
            """Q rows, the e4m3 table and the request lengths -> LDS.

            Row i = request i // R, column j = i % R (query row j // H, head
            j % H); 16 B column c8 sits at c8 ^ (i & 7), in the key's dim order.
            """
            for k in range_constexpr(QROWS * 8 // (64 * NW)):
                u = tx + k * 64 * NW
                i = u // 8
                c8 = u % 8
                rr = i // R
                j = i % R
                # Missing requests' and padding rows stay unwritten; their
                # scores are never stored.
                if (rr < nr) & (j // H < QL):
                    tq = (req0 + rr) * QL + j // H
                    off = (tq * i32_q_st + (j % H) * i32_q_sh + c8 * 16) // 4
                    d = Vec(
                        buffer_ops.buffer_load(
                            qr, off, vec_width=4, dtype=fx.Int32, mask=rr < nr
                        )
                    )
                    d = [fx.Int32(d[e]) for e in range_constexpr(4)]
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(key_order4(d))
                    fx.copy(atom16, r, fx.slice(q16, (None, i * 8 + (c8 ^ (i & 7)))))
            lut_g = global_i32_ptr(arg_lut)
            lut_l = smem.lut.ptr
            for e in range_constexpr(2):
                lut_l[tx * 2 + e] = fx.Int32(lut_g[tx * 2 + e])
            seq_l[lane] = seq_r

        prologue()
        fx.gpu.barrier()

        lut8 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
            ),
            fx.make_layout(2, 1),
        )
        lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

        def lut_load(sbyte):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])

        def q_frag(cg):
            """B operand of column group cg (dims 32 kl .. +31 of Q row cg * 16 +
            col)."""
            i = cg * 16 + col
            halves = []
            for h in range_constexpr(2):
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(
                    atom16, fx.slice(q16, (None, i * 8 + ((2 * kl + h) ^ (i & 7)))), r
                )
                halves.append(r.load())
            return pack8(halves[0], halves[1])

        code_off = (col * 64 + kl * 16) // 4  # dwords
        scale_off = col * 4 + kl  # u16: groups 2 kl, 2 kl + 1 of token col

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t of a page: this lane's 16 B of codes (token 16 t + col) and
            its two group scales; no memory access when not ``ok``."""
            c = Vec(
                buffer_ops.buffer_load(
                    kr,
                    code_off,
                    vec_width=4,
                    dtype=fx.Int32,
                    mask=ok,
                    soffset_bytes=t * 1024,
                    cache_modifier=NT,
                )
            )
            sc = fx.Int32(
                fx.Int16(
                    buffer_ops.buffer_load(
                        kr,
                        scale_off,
                        vec_width=1,
                        dtype=fx.Int16,
                        mask=ok,
                        soffset_bytes=PAGE_CODES + t * 128,
                        cache_modifier=NT,
                    )
                ).to(fx.Int32)
            )
            return c, sc

        def dequant_tile(c, sc):
            """The A fragment of a tile (e4m3 keys in the key's dim order)."""
            t0 = lut_load(sc & 0xFF)
            t1 = lut_load((sc >> 8) & 0xFF)
            f = []
            for e in range_constexpr(4):
                lo, hi = dequant8(fx.Int32(Vec(c)[e]), *(t0 if e < 2 else t1))
                f += [lo, hi]
            return Vec.from_elements(f, fx.Int32)

        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(float("-inf"))

        def frag_max(frags, qfs, masked, lims):
            """Per column group: the max of its column's scores over the page.

            Tile t's results are reduced under tile t + 1's MFMAs.
            """
            ng = len(qfs)
            runs = [neg_inf] * ng
            accs = [[None] * ng for _ in range(MT)]

            def reduce(t, runs):
                out = []
                for g in range_constexpr(ng):
                    run = runs[g]
                    v = accs[t][g]
                    for e in range_constexpr(4):
                        x = v[e]
                        if const_expr(masked):
                            x = (lims[g] >= t * 16 + e).select(x, neg_inf)
                        run = maxf_nn(run, x)
                    out.append(run)
                return out

            for t in range_constexpr(MT):
                for g in range_constexpr(ng):
                    accs[t][g] = mfma_fp8_16x16x128(frags[t], qfs[g], zero4)
                if const_expr(t >= 1):
                    runs = reduce(t - 1, runs)
            runs = reduce(MT - 1, runs)
            return [xlane4(r, maxf_nn) for r in runs]

        def row_value(mx, seq, q_off, b):
            return block_sentinels(mx, seq - QL + q_off + 1, b, i32_init, i32_local)

        def put(i_row, jpos, v, cond, addr):
            """Score v of Q row i_row at unit position jpos, to memory or to the
            LDS buffer; lanes with nothing to buffer write a dummy slot."""
            if const_expr(not buffered):
                buffer_ops.buffer_store(v, sr, addr, mask=cond)
            else:
                buffer_ops.buffer_store(v, sr, addr, mask=cond & direct)
                idx = (cond & to_lds).select(
                    i_row * SB_STRIDE + jpos, QROWS * SB_STRIDE + lane
                )
                (scb + idx).store(bits(v))  # `scb[idx] = v` would rebind scb

        def flush(ub):
            """Store the unit at block ub: 4 consecutive blocks per thread and
            pass."""
            NCH = UMAX // 4
            k = fx.Int32(0)
            while k < fx.Int32(QROWS * NCH // (64 * NW)):
                item = tx + k * (64 * NW)
                i = item // NCH
                c = item % NCH
                rr = i // R
                jc = i % R
                qoff = jc // H
                nb_c = (fx.Int32(seq_l[rr]) + (BLK - 1)) // BLK
                b0c = ub + c * 4
                ok = (rr < nr) & (qoff < QL) & (c * 4 < i32_unit)
                addr = (jc % H) * i32_s_sh + ((req0 + rr) * QL + qoff) * i32_s_st + b0c
                for e in range_constexpr(4):
                    v = as_f32(fx.Int32(scb[i * SB_STRIDE + c * 4 + e]))
                    buffer_ops.buffer_store(
                        v,
                        sr,
                        addr + e,
                        mask=ok & (b0c + e < nb_c) & (c * 4 + e < i32_unit),
                    )
                k = k + 1

        def score_cgs(frags, hm, masked_any, cgs, b, jpos):
            """Scores of column groups cgs, masked where a request is at its last
            blocks, with sentinels."""
            info = []
            for cg in cgs:
                i = cg * 16 + col
                crr = i // R
                st_j = i % R
                info.append((cg, crr, st_j // H, st_j % H))
            qfs = [q_frag(cg) for cg, _, _, _ in info]
            seqs = [fx.Int32(seq_l[crr]) for _, crr, _, _ in info]
            mxs = [neg_inf for _ in info]
            if masked_any:
                lims = [
                    seq_c - QL + qoff - b * BLK - kl * 4
                    for seq_c, (_, _, qoff, _) in zip(seqs, info)
                ]
                mxs = frag_max(frags, qfs, True, lims)
            else:
                mxs = frag_max(frags, qfs, False, None)
            for mx, seq_c, (cg, crr, qoff, head) in zip(mxs, seqs, info):
                hit = ((hm >> fx.Int64(crr)) & fx.Int64(1)) != fx.Int64(0)
                ov = row_value(mx, seq_c, qoff, b)
                otq = (req0 + crr) * QL + qoff
                put(
                    cg * 16 + col,
                    jpos,
                    ov,
                    hit & (kl == 0) & (qoff < QL),
                    head * i32_s_sh + otq * i32_s_st + b,
                )

        def score_groups(frags, hm, mmask, b, jpos):
            """Every column group holding the page (requests hm)."""
            left_m = hm
            while left_m != fx.Int64(0):
                lead_req = fx.Int32(fx.math.cttz(left_m).to(fx.Int32))
                if const_expr(CPR == 1):
                    cg = lead_req // RPC
                    gbits = fx.Int64(GMASK) << fx.Int64(cg * RPC)
                    left_m = left_m & ~gbits
                    score_cgs(frags, hm, (mmask & gbits) != fx.Int64(0), [cg], b, jpos)
                else:
                    gbits = fx.Int64(1) << fx.Int64(lead_req)
                    left_m = left_m & ~gbits
                    masked_any = (mmask & gbits) != fx.Int64(0)
                    score_cgs(
                        frags,
                        hm,
                        masked_any,
                        [lead_req * CPR + g for g in range(CPR)],
                        b,
                        jpos,
                    )

        # The waves share positions in units of i32_unit blocks; page k of
        # position `it` goes to wave (k + it) % NW. In per-wave mode (fewer
        # requests than waves) wave w walks its own units instead.
        u_step = per_wave.select(i32_splits * NW, i32_splits)
        ub = per_wave.select(split * NW + wave, split) * usz
        it = fx.Int32(0)
        while fx.Int64(fx.rocdl.ballot(T.i64, ub < nblk_r)) != fx.Int64(0):
            b = ub
            jpos = fx.Int32(0)
            while (jpos < usz) & (
                fx.Int64(fx.rocdl.ballot(T.i64, b < nblk_r)) != fx.Int64(0)
            ):
                in_r = b < nblk_r
                skip = (b < rbf_r) & ((b < i32_init) | (b >= loc_last_r))
                need_r = in_r & ~skip
                bid = fx.Int32(
                    buffer_ops.buffer_load(
                        btr,
                        need_r.select((req0 + lane) * i32_bt_stride + b, fx.Int32(0)),
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                bid = need_r.select(bid, fx.Int32(-1))
                # Sentinel-only blocks, stored by wave 0.
                if (fx.Int64(fx.rocdl.ballot(T.i64, in_r & skip)) != fx.Int64(0)) & (
                    (wave == 0) | per_wave
                ):
                    # A runtime loop: unrolled, all store addresses stay live.
                    jr = fx.Int32(0)
                    while jr < fx.Int32(QL * H):
                        sv = row_value(neg_inf, seq_r, jr // H, b)
                        stq = (req0 + lane) * QL + jr // H
                        put(
                            lane * R + jr,
                            jpos,
                            sv,
                            in_r & skip,
                            (jr % H) * i32_s_sh + stq * i32_s_st + b,
                        )
                        jr = jr + 1
                pend = fx.Int64(fx.rocdl.ballot(T.i64, need_r))
                mmask = fx.Int64(
                    fx.rocdl.ballot(T.i64, need_r & (b * BLK + (BLK - 1) > fm_r))
                )
                # This wave's pages, as the mask of their first lanes.
                mine = fx.Int64(0)
                k = it
                while pend != fx.Int64(0):
                    lead = fx.Int32(fx.math.cttz(pend).to(fx.Int32))
                    page = fx.Int32(fx.rocdl.readlane(T.i32, bid, lead))
                    pend = pend & ~fx.Int64(fx.rocdl.ballot(T.i64, bid == page))
                    mine = mine | (((k % NW) == wave) | per_wave).select(
                        fx.Int64(1) << fx.Int64(lead), fx.Int64(0)
                    )
                    k = k + 1
                ok0 = mine != fx.Int64(0)
                l0 = fx.Int32(fx.math.cttz(mine).to(fx.Int32))
                kr0 = page_rsrc(
                    fx.Int32(fx.rocdl.readlane(T.i32, bid, ok0.select(l0, fx.Int32(0))))
                )
                tiles = [load_tile(kr0, ok0, t) for t in range_constexpr(MT)]
                while mine != fx.Int64(0):
                    lead = fx.Int32(fx.math.cttz(mine).to(fx.Int32))
                    page = fx.Int32(fx.rocdl.readlane(T.i32, bid, lead))
                    hm = fx.Int64(fx.rocdl.ballot(T.i64, bid == page))
                    mine = mine & ~(fx.Int64(1) << fx.Int64(lead))
                    okn = mine != fx.Int64(0)
                    ln = fx.Int32(fx.math.cttz(mine).to(fx.Int32))
                    krn = page_rsrc(
                        fx.Int32(
                            fx.rocdl.readlane(T.i32, bid, okn.select(ln, fx.Int32(0)))
                        )
                    )
                    # Tile t of the next page loads into the registers tile t of
                    # this page frees; the barriers keep LLVM from hoisting the
                    # loads.
                    frags, nxt = [], []
                    for t in range_constexpr(MT):
                        frags.append(dequant_tile(*tiles[t]))
                        nxt.append(load_tile(krn, okn, t))
                        fx.rocdl.sched_barrier(0)
                    tiles = nxt
                    score_groups(frags, hm, mmask, b, jpos)
                b = b + 1
                jpos = jpos + 1
                it = it + 1
            if const_expr(buffered):  # noqa: SIM102 (the outer if is static)
                if to_lds:
                    fx.gpu.barrier()
                    flush(ub)
                    fx.gpu.barrier()
            ub = ub + u_step * usz

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_q_st: fx.Int32,
        i32_q_sh: fx.Int32,
        i64_q_bytes: fx.Int64,
        i32_s_sh: fx.Int32,
        i32_s_st: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_chunk: fx.Int32,
        i32_nch: fx.Int32,
        i32_splits: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_per_wave: fx.Int32,
        i32_unit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_seq,
            i64_page_stride,
            i32_q_st,
            i32_q_sh,
            i64_q_bytes,
            i32_s_sh,
            i32_s_st,
            i64_s_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_chunk,
            i32_nch,
            i32_splits,
            i32_init,
            i32_local,
            i32_per_wave,
            i32_unit,
        ).launch(
            grid=(fx.Int64(i32_nch * i32_splits), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
