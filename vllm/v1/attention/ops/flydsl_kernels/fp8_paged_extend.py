# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL causal extend attention over a paged fp8 KV cache (gfx950).

Transposed form: one workgroup (4 waves) takes a 16-token query block and the 8
query heads of one KV head (128 query columns) and computes S^T = K Q^T on
32x32x64 scaled fp8 MFMAs, with Q^T resident in registers. Each lane owns 16 keys
of one column, so the running max is lane-local and the rescale only runs when a
tile exceeds it. P is packed to fp8 for the PV MFMA, the row sums come from a
ones x P^T MFMA, and K/V stream through a 2-stage LDS DMA pipeline with per-page
bounded descriptors.

Q is either fp8 with one descale per token (`q_fused=False`), or bf16 quantized
in the kernel with one scale per (token, KV head) (`q_fused=True`).
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

GQA = 8
HD = 64
BQ = 16  # query tokens per workgroup
NW = 4  # waves per workgroup
TILE = 128  # keys per tile
PAGE = 16  # keys per cache page
NBUF = 2  # LDS stages
STAGE = 2 * TILE * HD  # K + V bytes per stage
LOG2E = 1.4426950408889634
BIAS = math.log2(448.0)  # P = 448 at the running max: exact in e4m3
M_INIT = -1.0e30
WAVES_PER_EU = 2


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _i32c(v):
    return arith.constant(int(v), type=T.i32)


def _vec(raw, n, dt=fx.Float32):
    return fx.Vector(_to_raw(raw), shape=n, dtype=dt)


def _cat(a, b, n, dt=fx.Int32):
    return _to_raw(_vec(a, n, dt).shuffle(_vec(b, n, dt), list(range(2 * n))))


def _tree_max(vals):
    vals = list(vals)
    while len(vals) > 1:
        nxt = [arith.maximumf(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def _max_with_lane_xor32(x_f32):
    i32 = ir.IntegerType.get_signless(32)
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = arith.bitcast(T.i32, _to_raw(x_f32))
    sw = _rocdl.permlane32_swap(pair, raw, raw, False, True)
    a = arith.bitcast(T.f32, llvm.extractvalue(i32, sw, [0]))
    b = arith.bitcast(T.f32, llvm.extractvalue(i32, sw, [1]))
    return arith.maximumf(a, b)


def _sclamp16(x):
    """clamp(x, 0, 16) of a wave-uniform i32 on the scalar unit."""
    return llvm.InlineAsmOp(
        T.i32,
        [_to_raw(fx.Int32(x))],
        "s_max_i32 $0, $1, 0\n\ts_min_i32 $0, $0, 16",
        "=&s,s,~{scc}",
    ).result


def _sload(rsrc, i):
    return fx.Int32(
        buffer_ops.buffer_load(rsrc, i, vec_width=1, dtype=T.i32, is_scalar=True)
    )


def _lds_f32(ptr, i):
    return llvm.load(T.f32, fx.to_llvm_ptr(ptr + fx.Int64(i * 4)))


def _uniform(x):
    return fx.Int32(_rocdl.readfirstlane(T.i32, _to_raw(fx.Int32(x))))


def _scope_kwargs(scopes, sc):
    if sc is None:
        return {}
    return dict(
        alias_scopes=ir.ArrayAttr.get([scopes[sc]]),
        noalias_scopes=ir.ArrayAttr.get([s for i, s in enumerate(scopes) if i != sc]),
    )


@functools.lru_cache(maxsize=64)
def build_launcher(hk, kv_s0, kv_hs, kv_row, exp_scale, split, window, q_fused):
    """Return the host launcher for one kernel configuration.

    Args:
        hk: KV heads.
        kv_s0: Cache stride between pages, in bytes.
        kv_hs: Cache stride between KV heads, in bytes.
        kv_row: Cache stride between tokens of a page, in bytes.
        exp_scale: `softmax_scale * log2(e)`.
        split: Build the split-KV variant, which reads a per-request plan and
            writes partials for the merge kernel instead of the output.
        window: Sliding-window size in tokens, 0 for full attention.
        q_fused: Q is bf16 and quantized in the kernel.

    """
    SPLIT = split
    WIN = window
    Q_FUSED = q_fused

    if Q_FUSED:

        @fx.struct
        class Lds:
            kv: fx.Array[fx.Int8, NBUF * STAGE, 16]
            qs: fx.Array[fx.Int8, NW * 64 * 4, 4]  # per-lane |q| max
    else:

        @fx.struct
        class Lds:  # type: ignore[no-redef]
            kv: fx.Array[fx.Int8, NBUF * STAGE, 16]

    @flyc.kernel(
        name="flydsl_fp8_paged_extend_qf" if Q_FUSED else "flydsl_fp8_paged_extend"
    )
    def kernel(
        q: fx.Int64,
        k: fx.Int64,
        v: fx.Int64,
        out: fx.Int64,
        cuq: fx.Tensor,
        seqlens: fx.Tensor,
        table: fx.Tensor,
        qd: fx.Tensor,
        kd: fx.Tensor,
        vd: fx.Tensor,
        parto: fx.Int64,
        partl: fx.Int64,
        plan: fx.Tensor,
        nreq: fx.Int32,
        qblk: fx.Int32,
        bt_s0: fx.Int32,
        q_s0: fx.Int32,
        o_s0: fx.Int32,
    ):
        # Unsplit grid: (q block, request, kv head), q block fastest so a
        # request's blocks share its KV stream in L2. Split grid: the plan's
        # units (sum over requests of q blocks x splits) x kv heads.
        pid = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = _uniform(tid // 64)
        lane = tid % 64
        col = lane % 32  # query column within the wave's 32
        hl = lane // 32  # dim half
        if fx.const_expr(not SPLIT):
            kvh = fx.Int32(pid // (qblk * nreq))
            rem = pid % (qblk * nreq)
            seq = rem // qblk
            qb = rem % qblk
            sp = fx.Int32(0)
            nsp = fx.Int32(1)
            tsb = fx.Int32(0)
        else:
            kvh = fx.Int32(pid % hk)
            unit = pid // hk
            pl_r = buffer_ops.create_buffer_resource(plan, max_size=True)
            # binary search for the request whose unit range holds this
            # workgroup (12 rounds cover nreq <= 4096); mid is clamped before
            # the load because the plan's resource is unbounded
            blo = fx.Int32(0)
            bhi = nreq
            for _ in range_constexpr(12):
                mid = (blo + bhi) // 2
                go = blo < bhi
                midc = (mid < nreq).select(mid, nreq - 1)
                ub_m = _sload(pl_r, midc * 4)
                ub_m = (mid < nreq).select(ub_m, fx.Int32(1 << 30))
                blo = (go & (ub_m <= unit)).select(mid + 1, blo)
                bhi = (go & (ub_m > unit)).select(mid, bhi)
            seq = blo - 1
            seq = (seq > 0).select(seq, fx.Int32(0))
            ub = _sload(pl_r, seq * 4)
            tsb = _sload(pl_r, seq * 4 + 1)
            nsp = _sload(pl_r, seq * 4 + 3)
            u = unit - ub
            qb = u // nsp
            sp = u % nsp

        cu_r = buffer_ops.create_buffer_resource(cuq, max_size=True)
        sl_r = buffer_ops.create_buffer_resource(seqlens, max_size=True)
        bt_r = buffer_ops.create_buffer_resource(table, max_size=True)
        qd_r = buffer_ops.create_buffer_resource(qd, max_size=True)
        kd_r = buffer_ops.create_buffer_resource(kd, max_size=True)
        vd_r = buffer_ops.create_buffer_resource(vd, max_size=True)

        q_start = _sload(cu_r, seq)
        q_len = _sload(cu_r, seq + 1) - q_start
        seq_len = _sload(sl_r, seq)
        ctx = seq_len - q_len
        t0 = qb * BQ
        # whole blocks past the request's tokens: no tiles, no stores
        valid = t0 < q_len
        k_descale = arith.bitcast(T.f32, _to_raw(_sload(kd_r, 0)))
        v_descale = arith.bitcast(T.f32, _to_raw(_sload(vd_r, 0)))

        blk_end = fx.min(t0 + BQ, q_len)
        n_tiles = (ctx + blk_end + TILE - 1) // TILE
        n_tiles = valid.select(n_tiles, fx.Int32(0))
        # tiles below `interior` are visible to every query of the block
        interior = (ctx + t0 + 1) // TILE
        interior = (interior < n_tiles).select(interior, n_tiles)
        last_page = (seq_len - 1) // PAGE
        bt_row = seq * bt_s0
        if fx.const_expr(not SPLIT):
            lo = fx.Int32(0)
            hi = interior
            if fx.const_expr(WIN > 0):
                # skip tiles fully below the window; the rest runs masked
                lo = fx.max((ctx + t0 - (WIN - 1)) // TILE, fx.Int32(0))
                interior = lo
                hi = lo
        else:
            # split the interior tiles over the request's nsp workgroups; the
            # masked tail belongs to the last split
            per = (interior + nsp - 1) // nsp
            lo = fx.min(sp * per, interior)
            hi = fx.min(lo + per, interior)

        lds = fx.SharedAllocator().allocate(Lds).peek()
        lptr = lds.kv.ptr

        # Q^T B fragment, resident for the whole kernel: lane (col, hl) holds
        # token t0 + 4 * wave + col // 8, head kvh * 8 + col % 8, dims
        # 32 * hl .. +31. The descriptor bound zero-fills tokens past q_len.
        tok = t0 + wave * 4 + col // GQA
        head = kvh * GQA + col % GQA
        if fx.const_expr(Q_FUSED):
            q_rs = buffer_ops.create_buffer_resource_from_addr(
                _to_raw(q + fx.Int64(q_start) * fx.Int64(q_s0) * 2),
                num_records_bytes=_to_raw(fx.Int64(q_len) * fx.Int64(q_s0) * 2),
            )
            qb_off = (tok * q_s0 * 2 + head * (HD * 2) + hl * 64) // 4
            qw = [
                buffer_ops.buffer_load(q_rs, qb_off + 4 * u, vec_width=4, dtype=T.i32)
                for u in range_constexpr(4)
            ]
            xs = []
            for u in range_constexpr(4):
                wv = fx.Vector(_to_raw(qw[u]), shape=4, dtype=fx.Int32)
                for e2 in range_constexpr(4):
                    wi = _to_raw(wv[e2])
                    xs.append(arith.bitcast(T.f32, arith.shli(wi, _i32c(16))))
                    xs.append(arith.bitcast(T.f32, arith.andi(wi, _i32c(0xFFFF0000))))
            mxl = _tree_max([arith.maximumf(x, arith.subf(_f32(0.0), x)) for x in xs])
            # amax over the 16 lanes of one (token, kv head): 8 columns x 2
            # dim halves, exchanged through LDS
            llvm.store(_to_raw(mxl), fx.to_llvm_ptr(lds.qs.ptr + fx.Int64(tid * 4)))
            _rocdl.s_barrier()
            m0 = wave * 64 + (col // 8) * 8
            mx = _tree_max(
                [_lds_f32(lds.qs.ptr, m0 + i) for i in range_constexpr(8)]
                + [_lds_f32(lds.qs.ptr, m0 + 32 + i) for i in range_constexpr(8)]
            )
            q_descale = arith.mulf(arith.maximumf(mx, _f32(1e-12)), _f32(1.0 / 448.0))
            inv_sc = arith.divf(_f32(1.0), q_descale)
            words = []
            for w4 in range_constexpr(8):
                y = [arith.mulf(xs[4 * w4 + i], inv_sc) for i in range_constexpr(4)]
                wlo = _rocdl.cvt_pk_fp8_f32(
                    T.i32, _to_raw(y[0]), _to_raw(y[1]), llvm.mlir_poison(T.i32), False
                )
                words.append(
                    _rocdl.cvt_pk_fp8_f32(
                        T.i32, _to_raw(y[2]), _to_raw(y[3]), wlo, True
                    )
                )
            qf = _to_raw(
                fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32)
            )
        else:
            q_rs = buffer_ops.create_buffer_resource_from_addr(
                _to_raw(q + fx.Int64(q_start) * fx.Int64(q_s0)),
                num_records_bytes=_to_raw(fx.Int64(q_len) * fx.Int64(q_s0)),
            )
            q_off = (tok * q_s0 + head * HD + hl * 32) // 4
            qf = _cat(
                buffer_ops.buffer_load(q_rs, q_off, vec_width=4, dtype=T.i32),
                buffer_ops.buffer_load(q_rs, q_off + 4, vec_width=4, dtype=T.i32),
                4,
            )
            # the descale buffer spans every request's tokens and its resource
            # is unbounded: index it call-absolute, and point lanes past q_len
            # (unused) at row 0
            qd_row = (tok < q_len).select(q_start + tok, fx.Int32(0))
            q_descale = arith.bitcast(
                T.f32,
                _to_raw(buffer_ops.buffer_load(qd_r, qd_row, vec_width=1, dtype=T.i32)),
            )
        c_s = arith.mulf(arith.mulf(q_descale, k_descale), _f32(exp_scale))
        inv_c = arith.divf(_f32(1.0), c_s)
        c_pair = _to_raw(
            fx.Vector.from_elements([fx.Float32(c_s), fx.Float32(c_s)], fx.Float32)
        )
        qpos = ctx + tok

        # one alias scope per LDS stage: reads of stage s need not wait on the
        # DMA into the other stage
        _scopes = ir.Attribute.parse(
            "["
            + ", ".join(
                f"#llvm.alias_scope<id = distinct[{i + 1}]<>, "
                "domain = #llvm.alias_scope_domain<id = distinct[0]<>>>"
                for i in range(NBUF)
            )
            + "]"
        )
        _sc = [_scopes[i] for i in range(NBUF)]

        def _scope_kw(sc):
            return _scope_kwargs(_sc, sc)

        # LDS tile layout: [key 0..127][64 B] per tensor (K at +0, V at
        # +TILE * HD); 16-B chunk c of a row lands at chunk c ^ swz(row % 16).
        swz_w = ((lane // 8) + 2 * (lane // 32)) % 4  # swz of write row lane // 4
        lc_w = fx.Int32(arith.xori(_to_raw(lane % 4), _to_raw(swz_w)))
        # global token rows sit kv_row bytes apart; LDS rows are 64 B
        voff_b = (lane // 4) * kv_row + lc_w * 16
        wave_lds = _uniform(wave * 32 * HD)  # this wave's 32-row slice
        # K reader: lane (col, hl) reads key col of the 32-key row tile,
        # logical chunks 2 * hl and 2 * hl + 1
        swz_k = ((col // 2) + 2 * ((col // 8) % 2)) % 4
        k_rd = [
            col * HD
            + fx.Int32(arith.xori(_to_raw(fx.Int32(2) * hl + u), _to_raw(swz_k))) * 16
            for u in range_constexpr(2)
        ]
        # V reader (transposed 8-B reads): lane (gq, li) covers key key0 of the
        # 16-key sub-tile, byte 8 * (li % 2), dim chunks 2 * dt + gq % 2
        gq = lane // 16
        li = lane % 16
        kk = li // 2
        key0 = 8 * (kk // 4) + 4 * hl + kk % 4
        swz_v = (2 * hl + (kk % 4) // 2 + 2 * (kk // 4)) % 4
        v_rd = [
            TILE * HD
            + key0 * HD
            + fx.Int32(arith.xori(_to_raw(2 * dt + gq % 2), _to_raw(swz_v))) * 16
            + (li % 2) * 8
            for dt in range_constexpr(2)
        ]

        def pages(tile):
            # ids of this wave's two pages of the tile, clamped into the row
            ids = []
            for n in range_constexpr(2):
                pi = tile * (TILE // PAGE) + wave * 2 + n
                pc = (pi < last_page).select(pi, last_page)
                ids.append(
                    _to_raw(
                        buffer_ops.buffer_load(
                            bt_r, bt_row + pc, vec_width=1, dtype=T.i32
                        )
                    )
                )
            return ids

        def issue_dma(tile, stage_off, ids, sc):
            # one descriptor per (tensor, page) bounded to the page's valid
            # rows, so rows past seq_len read as zero
            for n in range_constexpr(2):
                nv = seq_len - (tile * (TILE // PAGE) + wave * 2 + n) * PAGE
                # wave-uniform, but VGPR-resident in the split build
                nrec = fx.Int32(_sclamp16(_uniform(nv))) * kv_row
                for ui, base in enumerate((k, v)):
                    rs = buffer_ops.create_buffer_resource_from_addr(
                        _to_raw(
                            base
                            + fx.Int64(_uniform(ids[n])) * fx.Int64(kv_s0)
                            + fx.Int64(kvh * kv_hs)
                        ),
                        num_records_bytes=_to_raw(fx.Int64(nrec)),
                    )
                    dst = wave_lds + (stage_off + ui * (TILE * HD) + n * (PAGE * HD))
                    _rocdl.raw_ptr_buffer_load_lds(
                        rs,
                        fx.to_llvm_ptr(lptr + fx.Int64(dst)),
                        _i32c(16),
                        _to_raw(voff_b),
                        _i32c(0),
                        _i32c(0),
                        **_scope_kw(sc),
                    )

        def k_frag(stage_off, mt, sc):
            parts = [
                llvm.load(
                    T.vec(4, T.i32),
                    fx.to_llvm_ptr(lptr + fx.Int64(stage_off + mt * 32 * HD + k_rd[u])),
                    alignment=16,
                    **_scope_kw(sc),
                )
                for u in range_constexpr(2)
            ]
            return _cat(parts[0], parts[1], 4)

        s127 = _i32c(127)  # e8m0 scale 1.0 for the scaled MFMAs
        zero16 = _to_raw(fx.full(16, 0.0, fx.Float32))
        ones8 = _to_raw(fx.full(8, 0x38383838, fx.Int32))  # e4m3 1.0 x 32

        def mfma(a, b, acc):
            return fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                T.vec(16, T.f32), [a, b, acc, 0, 0, 0, s127, 0, s127]
            )

        def softmax_tile(j, stage_off, o_acc, negm, thr, l_run, masked, sc):
            kfr = [k_frag(stage_off, mt, sc) for mt in range_constexpr(4)]
            s = [mfma(kfr[mt], qf, zero16) for mt in range_constexpr(4)]
            ninf = fx.Float32(float("-inf"))
            flat = []
            for mt in range_constexpr(4):
                x = _vec(s[mt], 16)
                for r in range_constexpr(16):
                    # accumulator r of lane (col, hl) is key
                    # 32 * mt + 8 * (r // 4) + 4 * hl + r % 4
                    if fx.const_expr(masked):
                        key = fx.Int32(32 * mt + 8 * (r // 4) + r % 4)
                        key_ok = key <= qpos - j * TILE - 4 * hl
                        if fx.const_expr(WIN > 0):
                            key_ok = key_ok & (key > qpos - WIN - j * TILE - 4 * hl)
                        flat.append(_to_raw(key_ok.select(x[r], ninf)))
                    else:
                        flat.append(_to_raw(x[r]))
            mx = _tree_max(flat)
            need = arith.cmpf(arith.CmpFPredicate.OGT, mx, _to_raw(thr))
            vote = fx.Int64(_rocdl.ballot(T.i64, _to_raw(need)))
            # rare: llvm.expect(false) moves the rescale out of line
            hit = fx.Boolean(
                llvm.intr_expect(
                    _to_raw(vote != fx.Int64(0)),
                    arith.constant(0, type=ir.IntegerType.get_signless(1)),
                )
            )
            if hit:
                mxa = _max_with_lane_xor32(mx)
                m_old = arith.subf(_f32(BIAS), _to_raw(_vec(negm, 2)[0]))
                mn = arith.maximumf(m_old, arith.mulf(mxa, c_s))
                alpha = fx.rocdl.exp2(T.f32, arith.subf(m_old, mn))
                nm = arith.subf(_f32(BIAS), mn)
                negm = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Float32(nm), fx.Float32(nm)], fx.Float32
                    )
                )
                thr = arith.mulf(mn, inv_c)
                l_run = _to_raw(_vec(l_run, 2) * fx.Float32(alpha))
                o_acc = [_to_raw(_vec(o, 16) * fx.Float32(alpha)) for o in o_acc]
            # all 64 exponent arguments s * c + negm before the exps:
            # interleaving them serializes on the transcendental unit
            zs = []
            for i in range_constexpr(32):
                pr = fx.Vector.from_elements(
                    [fx.Float32(flat[2 * i]), fx.Float32(flat[2 * i + 1])], fx.Float32
                )
                zs.append(_vec(fmath.fma(_to_raw(pr), c_pair, _to_raw(negm)), 2))
            fx.rocdl.sched_barrier(0)
            words = []
            for mt in range_constexpr(4):
                ps = []
                for r in range_constexpr(0, 16, 2):
                    z = zs[mt * 8 + r // 2]
                    ps.append(_to_raw(fx.rocdl.exp2(T.f32, _to_raw(z[0]))))
                    ps.append(_to_raw(fx.rocdl.exp2(T.f32, _to_raw(z[1]))))
                for w4 in range_constexpr(4):
                    lo_w = _rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        ps[4 * w4],
                        ps[4 * w4 + 1],
                        llvm.mlir_poison(T.i32),
                        False,
                    )
                    words.append(
                        _rocdl.cvt_pk_fp8_f32(
                            T.i32, ps[4 * w4 + 2], ps[4 * w4 + 3], lo_w, True
                        )
                    )
            # PV per 64-key step ks: P^T words 8ks..8ks+7 are the B operand as
            # packed (S^T accumulator key order)
            o_new = list(o_acc)
            lt = None
            for ks in range_constexpr(2):
                pb = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in words[8 * ks : 8 * ks + 8]], fx.Int32
                    )
                )
                # row sums first so their consumer sits after the PV MFMAs
                lt = mfma(ones8, pb, zero16 if ks == 0 else lt)
                for dt in range_constexpr(2):
                    # V^T A fragment of dim half dt: 4 transposed 8-B reads
                    rd = []
                    for r in range_constexpr(4):
                        off = (
                            stage_off
                            + v_rd[dt]
                            + (64 * ks + 32 * (r // 2) + 16 * (r % 2)) * HD
                        )
                        rd.append(
                            _rocdl.ds_read_tr8_b64(
                                T.vec(2, T.i32),
                                fx.to_llvm_ptr(lptr + fx.Int64(off)),
                                **_scope_kw(sc),
                            ).result
                        )
                    va = _cat(_cat(rd[0], rd[1], 2), _cat(rd[2], rd[3], 2), 4)
                    o_new[dt] = mfma(va, pb, o_new[dt])
            l_run = _to_raw(
                _vec(l_run, 2)
                + fx.Vector.from_elements(
                    [_vec(lt, 16)[0], fx.Float32(0.0)], fx.Float32
                )
            )
            return o_new, negm, thr, l_run

        def tile_barrier():
            # the only VMEM ops in flight are tile j's DMAs and tile j + 1's
            # page-id loads: wait for all of them, then one s_barrier
            fx.rocdl.sched_barrier(0)
            fx.rocdl.s_waitcnt(vmcnt=0)
            _rocdl.s_barrier()
            fx.rocdl.sched_barrier(0)

        def step(j, st, stage, masked):
            # stage: static LDS stage (int, alias-scoped) or None (dynamic)
            o_acc = list(st[0:2])
            negm, thr, l_run = st[2], st[3], st[4]
            ids = list(st[5:7])
            tile_barrier()
            nids = pages(j + NBUF)
            fx.rocdl.sched_barrier(0)
            if fx.const_expr(stage is None):
                # stages count from this split's first tile lo
                issue_dma(j + NBUF - 1, ((j + NBUF - 1 - lo) % NBUF) * STAGE, ids, None)
                stage_off = ((j - lo) % NBUF) * STAGE
                sc = None
            else:
                nst = (stage + NBUF - 1) % NBUF
                issue_dma(j + NBUF - 1, nst * STAGE, ids, nst)
                stage_off = stage * STAGE
                sc = stage
            fx.rocdl.sched_barrier(0)
            o_acc, negm, thr, l_run = softmax_tile(
                j, stage_off, o_acc, negm, thr, l_run, masked, sc
            )
            return (
                [_to_raw(x) for x in o_acc]
                + [_to_raw(negm), _to_raw(thr), _to_raw(l_run)]
                + [_to_raw(x) for x in nids]
            )

        # prologue: tile lo in flight, page ids of tile lo + 1 staged
        issue_dma(lo, 0, pages(lo), 0)
        ids0 = pages(lo + 1)
        nm0 = _to_raw(
            fx.Vector.from_elements(
                [fx.Float32(BIAS - M_INIT), fx.Float32(BIAS - M_INIT)], fx.Float32
            )
        )
        thr0 = arith.mulf(_f32(M_INIT), inv_c)
        l0 = _to_raw(fx.full(2, 0.0, fx.Float32))
        init = [zero16, zero16, nm0, thr0, l0] + ids0
        # unmasked tiles in groups of NBUF with static stages, the remainder
        # with dynamic stages, then the masked tail
        ngr = (hi - lo) // NBUF
        for gi, st in range(0, ngr, 1, init=init):  # type: ignore[call-overload]
            j0 = lo + fx.Int32(arith.index_cast(T.i32, _to_raw(gi))) * NBUF
            for u in range_constexpr(NBUF - 1):
                st = step(j0 + u, st, u, False)
            fin0 = yield step(j0 + NBUF - 1, st, NBUF - 1, False)
        for jj, st in range(ngr * NBUF, hi - lo, 1, init=list(fin0)):  # type: ignore[call-overload]
            j = lo + fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
            fin = yield step(j, st, None, False)
        if fx.const_expr(not SPLIT):
            n_tail = n_tiles - interior
        else:
            n_tail = (sp == nsp - 1).select(n_tiles - interior, fx.Int32(0))
        for jj, st in range(0, n_tail, 1, init=list(fin)):  # type: ignore[call-overload]
            j = interior + fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
            fin2 = yield step(j, st, None, True)
        o_acc = [_vec(fin2[0], 16), _vec(fin2[1], 16)]
        l_run = _vec(fin2[4], 2)
        lsum = arith.addf(_to_raw(l_run[0]), _to_raw(l_run[1]))

        if fx.const_expr(not SPLIT):
            inv = arith.divf(v_descale, lsum)
            o_rs = buffer_ops.create_buffer_resource_from_addr(
                _to_raw(out + fx.Int64(q_start) * fx.Int64(o_s0 * 2)),
                num_records_bytes=_to_raw(fx.Int64(q_len) * fx.Int64(o_s0 * 2)),
            )
            # accumulator r of o_acc[dt] is dim 32 * dt + 8 * (r // 4) + 4 * hl
            # + r % 4 of column col
            o_row = tok * o_s0 + head * HD + hl * 4
            for dt in range_constexpr(2):
                ov = _vec(_to_raw(o_acc[dt]), 16) * fx.Float32(inv)
                for i in range_constexpr(4):
                    seg = ov.shuffle(ov, [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]).to(
                        fx.BFloat16
                    )
                    buffer_ops.buffer_store(_to_raw(seg), o_rs, o_row + 32 * dt + 8 * i)
        else:
            # PartO [slot][H][64] fp32 holds acc / l without the V descale;
            # PartL [slot][H] holds m + log2(l), -inf when the split saw no
            # key. slot = tsb + tok * nsp + sp. Descriptors rooted at the
            # request's first slot drop the stores of tokens past q_len.
            live = lsum > fx.Float32(0.0)
            lsafe = live.select(lsum, fx.Float32(1.0))
            inv = arith.divf(_f32(1.0), lsafe)
            m_used = arith.subf(_f32(BIAS), _to_raw(_vec(fin2[2], 2)[0]))
            lse = live.select(
                arith.addf(m_used, fx.rocdl.log(T.f32, _to_raw(lsafe))),
                fx.Float32(float("-inf")),
            )
            HH = hk * GQA
            po_rs = buffer_ops.create_buffer_resource_from_addr(
                _to_raw(parto + fx.Int64(tsb) * fx.Int64(HH * HD * 4)),
                num_records_bytes=_to_raw(
                    fx.Int64(q_len * nsp) * fx.Int64(HH * HD * 4)
                ),
            )
            pl_rs = buffer_ops.create_buffer_resource_from_addr(
                _to_raw(partl + fx.Int64(tsb) * fx.Int64(HH * 4)),
                num_records_bytes=_to_raw(fx.Int64(q_len * nsp) * fx.Int64(HH * 4)),
            )
            p_row = ((tok * nsp + sp) * HH + head) * HD + hl * 4
            l_row = (tok * nsp + sp) * HH + head
            for dt in range_constexpr(2):
                ov = _vec(_to_raw(o_acc[dt]), 16) * fx.Float32(inv)
                for i in range_constexpr(4):
                    seg = ov.shuffle(ov, [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3])
                    buffer_ops.buffer_store(
                        _to_raw(seg), po_rs, p_row + 32 * dt + 8 * i
                    )
            if hl == 0:
                buffer_ops.buffer_store(_to_raw(fx.Float32(lse)), pl_rs, l_row)

    @flyc.jit
    def launch(
        q: fx.Int64,
        k: fx.Int64,
        v: fx.Int64,
        out: fx.Int64,
        cuq: fx.Tensor,
        seqlens: fx.Tensor,
        table: fx.Tensor,
        qd: fx.Tensor,
        kd: fx.Tensor,
        vd: fx.Tensor,
        parto: fx.Int64,
        partl: fx.Int64,
        plan: fx.Tensor,
        nreq: fx.Int32,
        qblk: fx.Int32,
        bt_s0: fx.Int32,
        q_s0: fx.Int32,
        o_s0: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            k,
            v,
            out,
            cuq,
            seqlens,
            table,
            qd,
            kd,
            vd,
            parto,
            partl,
            plan,
            nreq,
            qblk,
            bt_s0,
            q_s0,
            o_s0,
            value_attrs={"rocdl.waves_per_eu": WAVES_PER_EU},
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(NW * 64, 1, 1), stream=stream)

    return launch
