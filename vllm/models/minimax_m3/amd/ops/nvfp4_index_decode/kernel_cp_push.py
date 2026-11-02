# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context-parallel decode index scorer for MiniMax-M3 (FlyDSL, gfx950, TP4).

Rank ``me`` scores blocks b with (b // CP_CH) % 4 == me for all four index heads
(bit for bit as ``kernel.py``) and stores head h's scores into rank h's buffer
(``index_cp.IndexCP``); it returns once every rank's scores have arrived.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.flydsl_utils import (
    NT,
    SC0,
    SC1,
    bits,
    dequant8,
    global_i32_ptr,
    key_order4,
    maxf_nn,
    mfma_fp8_16x16x128,
    pack8,
    perm,
    xlane4,
)
from vllm.models.minimax_m3.amd.ops.nvfp4_index_decode.kernel import block_sentinels

WORLD = 4  # TP ranks = index heads, one per rank
BLK = 128  # tokens per block
D = 128
CP_CH = 16  # blocks per run of one rank
PAGE_CODES = BLK * D // 2  # e2m1 codes, then [128 x 8 B] e4m3 group scales
PAGE_BYTES = PAGE_CODES + BLK * 8
MT = BLK // 16  # token tiles per page
SYS = SC0 | SC1
MAX_ROWS = 1024  # decode rows (max_num_seqs x query_len) the receive buffers hold
# Flag words per rank: [SC_FLAGS + p] source p's scores arrived, [SC_DONE + w]
# workgroup w done, [SC_IQ + p * SC_MAX_REQS + r] source p's index-q of request r.
SC_FLAGS = 64
SC_DONE = 1024
SC_MAX_WG = 1024
SC_IQ = 2048
SC_MAX_REQS = 256
FLAG_BYTES = 12288
# Device counters (int32); [CNT_SC] holds the call count.
CNT_SC = 128
CNT_WORDS = 512


def _flag_read(addr):
    """An atomic add of 0 at system scope (never hoisted out of a spin loop)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            global_i32_ptr(addr).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def _sel4(p, a0, a1, a2, a3):
    return (p == 0).select(a0, (p == 1).select(a1, (p == 2).select(a2, a3)))


GRID_SMALL, GRID = 512, 768
GRID_SMALL_REQS = 32
NW = 4  # waves per workgroup: quarter w of every run
QD = CP_CH // NW  # blocks per quarter
WIN = 64  # pages per block-table window (one VGPR)
MAX_REQS = 64 * NW  # one request per thread in the prefix sum
QL = 4  # DSpark rows per request
HEADS = 4


def _uni(x):
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(x)))


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def _wave_scan(x, lane):
    """Inclusive prefix sum over the wave's 64 lanes."""
    for d in range_constexpr(6):
        y = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane - (1 << d)) * 4, x))
        x = x + (lane >= (1 << d)).select(y, fx.Int32(0))
    return x


def compile_score_push():
    name = "m3_nvfp4_index_score_push_cp"

    @fx.struct
    class Shared:
        lut: fx.Array[fx.Int32, 512, 16]
        stage: fx.Array[fx.Int32, 2 * 16 * NW * QD, 16]  # [run % 2][col][quarter]
        incl: fx.Array[fx.Int32, MAX_REQS]  # inclusive prefix of the run counts
        seq: fx.Array[fx.Int32, MAX_REQS]
        wsum: fx.Array[fx.Int32, NW]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_iq: fx.Int64,  # this rank's head: [num_reqs * QL, 128] e4m3
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] i64
        arg_d0: fx.Int64,  # head h's scores [rows, S] fp32 (rank h's receive buffer)
        arg_d1: fx.Int64,
        arg_d2: fx.Int64,
        arg_d3: fx.Int64,
        arg_bt: fx.Int64,  # block table [num_reqs, >= max_blocks] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        arg_cnt: fx.Int64,  # device counters (CNT_*)
        arg_f0: fx.Int64,  # rank p's flag words
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        arg_q0: fx.Int64,  # rank p's index-q receive buffer [4, rmax, 128] e4m3
        arg_q1: fx.Int64,
        arg_q2: fx.Int64,
        arg_q3: fx.Int64,
        i32_rmax: fx.Int32,
        i64_page_stride: fx.Int64,
        i32_s: fx.Int32,  # score row stride (elements)
        i64_d_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_me: fx.Int32,
        i32_nwg: fx.Int32,  # the grid
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = _uni(tx // 64)
        col, kl = lane % 16, lane // 16
        st_qoff = col // HEADS  # this lane's column: row st_qoff of head st_head
        st_head = col % HEADS
        incl_l, seq_l = smem.incl.ptr, smem.seq.ptr

        cr = buffer_ops.create_buffer_resource_from_addr(
            arg_cnt, num_records_bytes=CNT_WORDS * 4
        )
        calls = (
            _uni(
                buffer_ops.buffer_load(
                    cr, CNT_SC, vec_width=1, dtype=fx.Int32, cache_modifier=SC1
                )
            )
            + 1
        )
        btr = buffer_ops.create_buffer_resource_from_addr(
            arg_bt, num_records_bytes=i64_bt_bytes
        )
        qbytes = fx.Int64(i32_rmax) * (HEADS * D)
        qs = (arg_q0, arg_q1, arg_q2, arg_q3)
        fs = (arg_f0, arg_f1, arg_f2, arg_f3)
        mine = _sel4(i32_me, arg_f0, arg_f1, arg_f2, arg_f3)
        qr = buffer_ops.create_buffer_resource_from_addr(
            _sel4(i32_me, arg_q0, arg_q1, arg_q2, arg_q3), num_records_bytes=qbytes
        )
        drs = [
            buffer_ops.create_buffer_resource_from_addr(
                d, num_records_bytes=i64_d_bytes
            )
            for d in (arg_d0, arg_d1, arg_d2, arg_d3)
        ]

        # LUT to LDS; per request this rank's run count, prefix-summed.
        lut_g = global_i32_ptr(arg_lut)
        lut_l = smem.lut.ptr
        for i in range_constexpr(2):
            lut_l[tx * 2 + i] = fx.Int32(lut_g[tx * 2 + i])
        r_ok = tx < i32_num_reqs
        seq_t = fx.Int32(global_i32_ptr(arg_seq)[r_ok.select(tx, fx.Int32(0))])
        seq_t = r_ok.select(seq_t, fx.Int32(0))
        # This rank's index-q rows of requests pid, pid + nwg, ... to slot me of
        # every rank; their flags follow below.
        ir = buffer_ops.create_buffer_resource_from_addr(
            arg_iq, num_records_bytes=fx.Int64(i32_num_reqs) * (QL * D)
        )
        rs = pid
        while (wave == 0) & (rs < i32_num_reqs):
            v = buffer_ops.buffer_load(
                ir,
                (rs * QL * D + lane * 16) // 4,
                vec_width=4,
                dtype=fx.Int32,
                mask=lane < QL * D // 16,
            )
            for p in range_constexpr(WORLD):
                buffer_ops.buffer_store(
                    v,
                    buffer_ops.create_buffer_resource_from_addr(
                        qs[p], num_records_bytes=qbytes
                    ),
                    ((i32_me * i32_rmax + rs * QL) * D + lane * 16) // 4,
                    mask=lane < QL * D // 16,
                    cache_modifier=SYS,
                )
            rs = rs + i32_nwg
        nb_t = (seq_t + (BLK - 1)) // BLK
        runs_t = (nb_t // (WORLD * CP_CH)) + (
            nb_t % (WORLD * CP_CH) > i32_me * CP_CH
        ).select(fx.Int32(1), fx.Int32(0))
        x = _wave_scan(runs_t, lane)
        if lane == 63:
            smem.wsum.ptr[wave] = x
        seq_l[tx] = seq_t
        fx.gpu.barrier()
        for w in range_constexpr(NW - 1):
            x = x + (wave > w).select(fx.Int32(smem.wsum.ptr[w]), fx.Int32(0))
        incl_l[tx] = x
        fx.gpu.barrier()

        # Runs pid, pid + grid, ...: the busy workgroups stay one per CU.
        total = _uni(incl_l[MAX_REQS - 1])
        nruns = (pid < total).select((total - pid + i32_nwg - 1) // i32_nwg, 0)
        nslots = nruns * QD  # pages of this wave

        lut8 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)
            ),
            fx.make_layout(2, 1),
        )
        lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        stage16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.stage.ptr),
                fx.make_layout(2 * 16 * NW * QD, 1),
            ),
            fx.make_layout(4, 1),
        )
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def lut_load(sbyte):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])

        code_off = (col * 64 + kl * 16) // 4  # dwords
        # Scales 2 kl, 2 kl + 1 of token col as a dword: a u16 load's sign
        # extension made the loop head wait for the whole next page.
        scale_off = col * 2 + kl // 2  # dwords
        # v_perm selectors: scale byte 2 (kl % 2) (+ 1) of the dword, zeros above
        scale_sel = (kl % 2) * 2 + 0x0C0C0C00

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t: this lane's codes and scales; no memory access unless ok."""
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
                buffer_ops.buffer_load(
                    kr,
                    scale_off,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=ok,
                    soffset_bytes=PAGE_CODES + t * 128,
                    cache_modifier=NT,
                )
            )
            return c, sc

        def dequant_tile(c, sc):
            """The A fragment of a tile (e4m3 keys, even dims then odd dims per 8)"""
            t0 = lut_load(perm(sc, sc, scale_sel))
            t1 = lut_load(perm(sc, sc, scale_sel + 1))
            f = []
            for e in range_constexpr(4):
                lo, hi = dequant8(fx.Int32(Vec(c)[e]), *(t0 if e < 2 else t1))
                f += [lo, hi]
            return Vec.from_elements(f, fx.Int32)

        def q_load(r, ok):
            """This lane's 32 B of Q: row r * QL + st_qoff, head st_head."""
            off = ((st_head * i32_rmax + r * QL + st_qoff) * D + kl * 32) // 4
            a = Vec(
                buffer_ops.buffer_load(qr, off, vec_width=4, dtype=fx.Int32, mask=ok)
            )
            b = Vec(
                buffer_ops.buffer_load(
                    qr, off + 4, vec_width=4, dtype=fx.Int32, mask=ok
                )
            )
            return a, b

        def q_frag(a, b):
            """B operand: the 32 B in the key's dim order."""
            halves = []
            for v in (a, b):
                d = [fx.Int32(Vec(v)[e]) for e in range_constexpr(4)]
                halves.append(key_order4(d))
            return pack8(halves[0], halves[1])

        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(float("-inf"))

        def reduce(v, t, run, masked, lim):
            for e in range_constexpr(4):
                x = v[e]
                if const_expr(masked):
                    x = (lim >= t * 16 + e).select(x, neg_inf)
                run = maxf_nn(run, x)
            return run

        def page_pass(tiles, qf, krn, okn, masked, lim):
            """Max of this lane's column over the page; tile t of the next page
            loads into tile t's registers (sched barriers stop LLVM hoisting)."""
            accs = []
            nxt = []
            for t in range_constexpr(MT):
                f = dequant_tile(*tiles[t])
                nxt.append(load_tile(krn, okn, t))
                accs.append(mfma_fp8_16x16x128(f, qf, zero4))
                fx.rocdl.sched_barrier(0)
            mx = neg_inf
            if masked:
                run_m = neg_inf
                for t in range_constexpr(MT):
                    run_m = reduce(accs[t], t, run_m, True, lim)
                mx = xlane4(run_m, maxf_nn)
            else:
                run_u = neg_inf
                for t in range_constexpr(MT):
                    run_u = reduce(accs[t], t, run_u, False, None)
                mx = xlane4(run_u, maxf_nn)
            return mx, nxt

        def row_value(mx, seq, q_off, b):
            return block_sentinels(mx, seq - QL + q_off + 1, b, i32_init, i32_local)

        fx.rocdl.s_waitcnt(vmcnt=0)
        rs = pid
        while (wave == 0) & (rs < i32_num_reqs):
            for p in range_constexpr(WORLD):
                buffer_ops.buffer_store(
                    calls,
                    buffer_ops.create_buffer_resource_from_addr(
                        fs[p], num_records_bytes=FLAG_BYTES
                    ),
                    SC_IQ + i32_me * SC_MAX_REQS + rs,
                    mask=lane == 0,
                    cache_modifier=SYS,
                )
            rs = rs + i32_nwg

        w0 = fx.Int32(0)
        while w0 < nslots:
            # One page per lane: page s is block 4 w + s % 4 of run pid + (s // 4)
            # * grid, of request rj (binary search in the prefix).
            s = w0 + lane
            k = pid + (s // QD) * i32_nwg
            rj = fx.Int32(0)
            for bs in range_constexpr(8):
                step = MAX_REQS >> (bs + 1)
                rj = (fx.Int32(incl_l[rj + step - 1]) <= k).select(rj + step, rj)
            valid = s < nslots
            rj = valid.select(rj, fx.Int32(0))
            excl = (rj > 0).select(fx.Int32(incl_l[_max(rj - 1, 0)]), fx.Int32(0))
            seqj = fx.Int32(seq_l[rj])
            nblkj = (seqj + (BLK - 1)) // BLK
            bj = ((k - excl) * WORLD + i32_me) * CP_CH + wave * QD + s % QD
            frow = seqj - (QL - 1)
            rbf = ((frow < 0).select(fx.Int32(0), frow) + (BLK - 1)) // BLK
            ll = nblkj - i32_local
            loc_last = (ll < 0).select(fx.Int32(0), ll)
            # Init / local blocks every row sees whole only need their sentinel.
            skip = (bj < rbf) & ((bj < i32_init) | (bj >= loc_last))
            need = valid & (bj < nblkj) & ~skip
            pgj = fx.Int32(
                buffer_ops.buffer_load(
                    btr,
                    need.select(rj * i32_bt_stride + bj, fx.Int32(0)),
                    vec_width=1,
                    dtype=fx.Int32,
                )
            )
            pgj = need.select(pgj, fx.Int32(-1))
            n = _min(nslots - w0, WIN)

            p_0 = fx.Int32(fx.rocdl.readlane(T.i32, pgj, 0))
            r_0 = fx.Int32(fx.rocdl.readlane(T.i32, rj, 0))
            ok0 = p_0 >= 0
            kr0 = page_rsrc(ok0.select(p_0, fx.Int32(0)))
            fx.rocdl.sched_barrier(0)
            tiles = []
            for t in range_constexpr(MT):
                tiles.append(load_tile(kr0, ok0, t))
                fx.rocdl.sched_barrier(0)
            # Wait for the index-q rows; lane i polls source i % 4 of its request.
            word = mine + fx.Int64((SC_IQ + (lane % WORLD) * SC_MAX_REQS + rj) * 4)
            want = lane < n
            f = calls
            if want:
                f = _flag_read(word)
            while fx.Int64(fx.rocdl.ballot(T.i64, want & (f < calls))) != 0:
                if want & (f < calls):
                    f = _flag_read(word)
            qa, qb = q_load(r_0, n > 0)
            # Nothing pending where the loop head merges with the back edge.
            fx.rocdl.s_waitcnt(vmcnt=0)
            qf = q_frag(qa, qb)
            qna, qnb = qa, qb
            sw = n < 0  # false: Q of the next page already loaded
            vq = [neg_inf] * QD
            j = fx.Int32(0)
            while j < n:
                r = fx.Int32(fx.rocdl.readlane(T.i32, rj, j))
                b = fx.Int32(fx.rocdl.readlane(T.i32, bj, j))
                seq = fx.Int32(fx.rocdl.readlane(T.i32, seqj, j))
                if sw:
                    qf = q_frag(qna, qnb)
                jn = j + 1
                okj = jn < n
                jc = okj.select(jn, fx.Int32(0))
                pgn = fx.Int32(fx.rocdl.readlane(T.i32, pgj, jc))
                rn = fx.Int32(fx.rocdl.readlane(T.i32, rj, jc))
                okn = okj & (pgn >= 0)
                krn = page_rsrc(okn.select(pgn, fx.Int32(0)))
                sw = okj & (rn != r)
                qna, qnb = q_load(rn, sw)
                fx.rocdl.sched_barrier(0)
                mx, tiles = page_pass(
                    tiles,
                    qf,
                    krn,
                    okn,
                    b * BLK + (BLK - 1) > seq - QL,
                    seq - QL + st_qoff - b * BLK - kl * 4,
                )
                ov = row_value(mx, seq, st_qoff, b)
                e = (w0 + j) % QD
                vq = [(e == i).select(ov, v) for i, v in enumerate(vq)]
                if e == QD - 1:
                    # The run's quarters meet in LDS; wave m % NW stores it.
                    m = (w0 + j) // QD
                    slot = m % 2
                    val = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    val.store(Vec.from_elements([bits(v) for v in vq], fx.Int32))
                    if kl == 0:
                        fx.copy(
                            atom16,
                            val,
                            fx.slice(stage16, (None, (slot * 16 + col) * NW + wave)),
                        )
                    fx.gpu.barrier()
                    if wave == m % NW:
                        got = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(
                            atom16,
                            fx.slice(stage16, (None, (slot * 16 + col) * NW + kl)),
                            got,
                        )
                        off = (r * QL + st_qoff) * i32_s + (b - b % CP_CH) + kl * QD
                        for h in range_constexpr(HEADS):
                            buffer_ops.buffer_store(
                                got.load(),
                                drs[h],
                                off,
                                mask=st_head == h,
                                cache_modifier=SYS,
                            )
                j = jn
            w0 = w0 + WIN

        # sc0 sc1 stores are system-visible once vmcnt drains; a release fence
        # per workgroup would write back L2. The last workgroup (dispatched last,
        # so its spin cannot starve the others) collects the done words, signals
        # every rank and waits for all four.
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        own = buffer_ops.create_buffer_resource_from_addr(
            mine, num_records_bytes=FLAG_BYTES
        )
        buffer_ops.buffer_store(
            calls, own, SC_DONE + pid, mask=tx == 0, cache_modifier=SYS
        )
        if pid == i32_nwg - 1:
            # Thread t polls done words t, t + 256, ...
            def done_min():
                lo = calls
                for i in range_constexpr(SC_MAX_WG // (64 * NW)):
                    w = tx + i * 64 * NW
                    f = _flag_read(
                        mine + fx.Int64((SC_DONE + _min(w, SC_MAX_WG - 1)) * 4)
                    )
                    lo = _min(lo, (w < i32_nwg).select(f, calls))
                return lo

            lo = done_min()
            while fx.Int64(fx.rocdl.ballot(T.i64, lo < calls)) != 0:
                lo = done_min()
            fx.gpu.barrier()
            _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
            for p in range_constexpr(WORLD):
                fr = buffer_ops.create_buffer_resource_from_addr(
                    fs[p], num_records_bytes=FLAG_BYTES
                )
                buffer_ops.buffer_store(
                    calls, fr, SC_FLAGS + i32_me, mask=tx == 0, cache_modifier=SYS
                )
            buffer_ops.buffer_store(calls, cr, CNT_SC, mask=tx == 0, cache_modifier=SYS)
            if tx < WORLD:
                f = _flag_read(mine + fx.Int64((SC_FLAGS + tx) * 4))
                while f < calls:
                    f = _flag_read(mine + fx.Int64((SC_FLAGS + tx) * 4))
            fx.gpu.barrier()
            _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")

    @flyc.jit
    def launch(
        arg_iq: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_d0: fx.Int64,
        arg_d1: fx.Int64,
        arg_d2: fx.Int64,
        arg_d3: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_cnt: fx.Int64,
        arg_f0: fx.Int64,
        arg_f1: fx.Int64,
        arg_f2: fx.Int64,
        arg_f3: fx.Int64,
        arg_q0: fx.Int64,
        arg_q1: fx.Int64,
        arg_q2: fx.Int64,
        arg_q3: fx.Int64,
        i32_rmax: fx.Int32,
        i64_page_stride: fx.Int64,
        i32_s: fx.Int32,
        i64_d_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_me: fx.Int32,
        i32_nwg: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_iq,
            arg_kv,
            arg_lut,
            arg_d0,
            arg_d1,
            arg_d2,
            arg_d3,
            arg_bt,
            arg_seq,
            arg_cnt,
            arg_f0,
            arg_f1,
            arg_f2,
            arg_f3,
            arg_q0,
            arg_q1,
            arg_q2,
            arg_q3,
            i32_rmax,
            i64_page_stride,
            i32_s,
            i64_d_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_init,
            i32_local,
            i32_me,
            i32_nwg,
        ).launch(grid=(fx.Int64(i32_nwg), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch


assert CNT_SC < CNT_WORDS and SC_DONE + SC_MAX_WG <= FLAG_BYTES // 4
