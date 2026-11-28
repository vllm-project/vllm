# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/sgl-project/sglang/pull/49704
"""Causal FP8 extend (prefill) attention on gfx950 in FlyDSL: head size 64, GQA 8.

Query `[tokens, 8 * kv_heads, 64]` over the paged FP8 e4m3 KV cache in the
`ROCM_AITER_FA` layout, causal, optionally with a sliding window; BF16 output. A
request's cached prefix and new tokens form one key range (the new tokens' K/V are
already in the cache), so there is no prefix/new split and no merge.

One workgroup is 16 query tokens x one KV head (128 query columns) on 4 waves. The
kernel works in transposed form on 32x32x64 scaled FP8 MFMAs: S^T = K Q^T per 128-key
tile, with Q^T held in registers. P is FP8 e4m3, encoded against the running row
maximum m with P(m) = 2^bias, and feeds the PV MFMA directly; the row sums come from
the same P bytes, so keys whose P rounds to zero drop out of numerator and denominator.

P is either e4m3(exp2(s c + bias - m)) from an FP8 query (exact), or, with
`exp_free`, the e4m3 code 8 z + 56 for z = log2(P), computed by the QK MFMA from a BF16
query cast in-kernel (no exponential or FP8 conversion per element). With `exp_free`,
codes below 8 are e4m3 subnormals and codes below 0.5 round to zero.

K/V tiles are LDS-DMA'd one 16-row block per instruction through per-block buffer
descriptors, so rows past a request read as zero. Two LDS stages, with 16-byte chunks
XOR-swizzled by row for conflict-free K and transposed V reads.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw, buf_base_i64
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T

from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

GQA = 8
HD = 64
BQ = 16  # query tokens per workgroup
NW = 4  # waves per workgroup
TILE = 128  # keys per tile
PAGE = 16  # rows per LDS-DMA block
NBUF = 2  # LDS stages
STAGE = 2 * TILE * HD  # K + V bytes of one LDS stage
LOG2E = 1.4426950408889634
LOG2_448 = math.log2(448.0)  # P = 448 at the running maximum: the largest e4m3
CODE_PER_OCTAVE = 8.0
CODE_ONE = 56.0  # e4m3 code of 1.0
MAX_CODE_BIAS = (126.0 - CODE_ONE) / CODE_PER_OCTAVE  # code 126 = 448
NO_REF = -1.0e30  # reference of a column that has only seen masked keys


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _i32c(v):
    return arith.constant(int(v), type=T.i32)


def _vec(raw, n, dt=fx.Float32):
    return fx.Vector(_to_raw(raw), shape=n, dtype=dt)


def _cat(a, b, n, dt=fx.Int32):
    return _to_raw(_vec(a, n, dt).shuffle(_vec(b, n, dt), list(range(2 * n))))


def _allreduce32(x_f32):
    """Max of x over lanes l and l ^ 32: permlane32_swap(x, x) returns {own, partner}
    in some order per lane."""
    i32 = ir.IntegerType.get_signless(32)
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = arith.bitcast(T.i32, _to_raw(x_f32))
    sw = _rocdl.permlane32_swap(pair, raw, raw, False, True)
    a = arith.bitcast(T.f32, llvm.extractvalue(i32, sw, [0]))
    b = arith.bitcast(T.f32, llvm.extractvalue(i32, sw, [1]))
    return arith.maximumf(a, b)


def _cvt_u8(x_f32, byte, old):
    """v_cvt_pk_u8_f32: byte `byte` of old <- saturate_u8(round(x))."""
    return llvm.call_intrinsic(
        T.i32, "llvm.amdgcn.cvt.pk.u8.f32", [x_f32, _i32c(byte), old], [], []
    )


def _mul_legacy(a, b):
    """v_mul_legacy_f32: a target intrinsic, so it is not packed into v_pk_mul_f32,
    and the hazard recognizer sees it (it may read fresh MFMA accumulators)."""
    return llvm.call_intrinsic(T.f32, "llvm.amdgcn.fmul.legacy", [a, b], [], [])


def _scale_np(vec_raw, n, a):
    """Vec * a as n non-packed multiplies."""
    v = _vec(vec_raw, n)
    return _to_raw(
        fx.Vector.from_elements(
            [fx.Float32(_mul_legacy(_to_raw(v[i]), a)) for i in range(n)], fx.Float32
        )
    )


def _tree_max(vals):
    """Max of a list of raw f32 (pairwise; the compiler folds it into v_max3)."""
    vals = list(vals)
    while len(vals) > 1:
        nxt = [arith.maximumf(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def _scope_kwargs(scopes, sc):
    """Alias / noalias scopes for an access to LDS stage sc (None: no metadata)."""
    if sc is None:
        return {}
    return dict(
        alias_scopes=ir.ArrayAttr.get([scopes[sc]]),
        noalias_scopes=ir.ArrayAttr.get(
            [scopes[i] for i in range(len(scopes)) if i != sc]
        ),
    )


def _sclamp16(x):
    """clamp(x, 0, 16) for a wave-uniform i32 on the scalar unit."""
    return llvm.InlineAsmOp(
        T.i32,
        [_to_raw(fx.Int32(x))],
        "s_max_i32 $0, $1, 0\n\ts_min_i32 $0, $0, 16",
        "=&s,s,~{scc}",
    ).result


def _uniform(x):
    return fx.Int32(_rocdl.readfirstlane(T.i32, _to_raw(fx.Int32(x))))


@functools.lru_cache(maxsize=64)
def _launcher(hk, kv_s0, kv_s1, kv_sh, scale, logb, bias, ppb, window, exp_free):
    # hk: KV heads; kv_s0 / kv_s1 / kv_sh: page, row and head strides of the cache
    # (bytes); logb: binary-search depth over the requests; ppb: 16-row blocks per page;
    # window: sliding window (0 = none), query p sees keys p - window + 1 .. p;
    # exp_free: Mitchell codes (BF16 query) instead of exp2 + RNE e4m3 (FP8 query).
    # exp_free spends fewer VALU cycles per key, so it runs 64-key softmax steps at 4
    # waves / SIMD; the exact softmax runs 128-key steps at 3.
    sub = 64 if exp_free else TILE  # keys per softmax step
    mts = sub // 32  # 32-key MFMA row tiles per step
    nsub = TILE // sub
    waves_per_eu = 4 if exp_free else 3
    s1w = kv_s1 // 4
    nid = 2 if ppb == 1 else 1  # page-id loads per wave and tile
    # Both modes run S in "code" units out of the QK MFMA (query pre-scaled, column
    # offsets as the accumulator input): exp_free codes are 8 log2(P) + 56; the exp2
    # mode's are log2(P) itself, the exponent argument.
    cpo = CODE_PER_OCTAVE if exp_free else 1.0
    code_ref = cpo * bias + (CODE_ONE if exp_free else 0.0)  # running maximum
    code_scale = cpo * float(scale) * LOG2E

    @fx.struct
    class Lds:
        kv: fx.Array[fx.Int8, NBUF * STAGE, 16]

    @flyc.kernel(name="fp8_extend_attention")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        out: fx.Tensor,
        cuq: fx.Tensor,
        seqlens: fx.Tensor,
        table: fx.Tensor,
        qd: fx.Tensor,
        kd: fx.Tensor,
        vd: fx.Tensor,
        num_seqs: fx.Int32,
        nqb: fx.Int32,
        bt_s0: fx.Int32,
        q_s0: fx.Int32,
        o_s0: fx.Int32,
    ):
        pid = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = _uniform(tid // 64)
        lane = tid % 64
        col = lane % 32
        hl = lane // 32
        h = pid % hk
        # longest blocks (last of a request) first
        qb = nqb - 1 - pid // hk

        cu_r = buffer_ops.create_buffer_resource(cuq, max_size=True)
        sl_r = buffer_ops.create_buffer_resource(seqlens, max_size=True)
        bt_r = buffer_ops.create_buffer_resource(table, max_size=True)
        sc_r = [
            buffer_ops.create_buffer_resource(x, max_size=True) for x in (qd, kd, vd)
        ]

        def sload(r, i):
            return fx.Int32(
                buffer_ops.buffer_load(r, i, vec_width=1, dtype=T.i32, is_scalar=True)
            )

        # request owning q-block qb in the grid numbering cu_q[s] // BQ + s
        lo = fx.Int32(0)
        hi = num_seqs
        for _ in range_constexpr(logb):
            mid = (lo + hi) // 2
            go = lo < hi
            val = sload(cu_r, mid) // BQ + mid
            lo = (go & (val <= qb)).select(mid + 1, lo)
            hi = (go & (val > qb)).select(mid, hi)
        seq = lo - 1
        seq = (seq > 0).select(seq, fx.Int32(0))
        q_start = sload(cu_r, seq)
        q_len = sload(cu_r, seq + 1) - q_start
        local = qb - (q_start // BQ + seq)
        valid = (local * BQ < q_len) & (local >= 0)
        seq_len = sload(sl_r, seq)
        ctx = seq_len - q_len
        q_scale, k_descale, v_descale = [
            arith.bitcast(T.f32, _to_raw(sload(r, 0))) for r in sc_r
        ]

        blk_end = (local * BQ + BQ < q_len).select(local * BQ + BQ, q_len)
        n_tiles = (ctx + blk_end + TILE - 1) // TILE
        n_tiles = valid.select(n_tiles, fx.Int32(0))
        # tiles below `interior` are visible to every query of the block: no masks
        interior = (ctx + local * BQ + 1) // TILE
        interior = (interior < n_tiles).select(interior, n_tiles)
        last_page = (seq_len - 1) // PAGE
        last_page = (last_page > 0).select(last_page, fx.Int32(0))
        if fx.const_expr(window):
            # first key any query of the block sees, its tile t0 (the loops count
            # tiles from t0), and the first tile every query of the block sees in
            # full (the window bound is masked below it)
            lo_key = ctx + local * BQ - (window - 1)
            lo_key = (lo_key > 0).select(lo_key, fx.Int32(0))
            t0 = valid.select(lo_key // TILE, fx.Int32(0))
            lo_page = lo_key // PAGE
            lo_full = ctx + blk_end - window
            lo_full = (lo_full > 0).select(lo_full, fx.Int32(0))
            lo_full = (lo_full + TILE - 1) // TILE
        else:
            t0 = fx.Int32(0)
            lo_full = fx.Int32(0)
        bt_row = seq * bt_s0
        s127 = _i32c(127)  # E8M0 2^0

        # Q^T B operand: column col -> token local * 16 + 4 wave + col // 8, head
        # h * 8 + col % 8; lane half hl holds dims 32 hl .. 32 hl + 31. The block's
        # first token is in the 64-bit descriptor base and the bound covers the
        # block's tokens of the request: rows past it read as zero, and the 32-bit
        # offsets stay within one block.
        blk_rows = q_len - local * BQ
        blk_rows = (blk_rows < BQ).select(blk_rows, fx.Int32(BQ))
        blk_rows = (blk_rows > 0).select(blk_rows, fx.Int32(0))
        blk_tok = fx.Int64(q_start) + fx.Int64(local * BQ)
        q_base = buf_base_i64(q)
        q_rs = buffer_ops.create_buffer_resource_from_addr(
            _to_raw(q_base + blk_tok * fx.Int64(q_s0)),
            num_records_bytes=_to_raw(fx.Int64(blk_rows) * fx.Int64(q_s0)),
        )
        tok_b = wave * 4 + col // GQA  # token within the block
        tok = local * BQ + tok_b
        head = h * GQA + col % GQA
        q_off = (tok_b * q_s0 + (head * HD + hl * 32) * 2) // 4
        qd16: list = []
        for i4 in range_constexpr(4):
            v4 = _vec(
                buffer_ops.buffer_load(q_rs, q_off + 4 * i4, vec_width=4, dtype=T.i32),
                4,
                fx.Int32,
            )
            qd16.extend(_to_raw(v4[e]) for e in range(4))
        fq = []
        for d in qd16:
            fq.append(arith.bitcast(T.f32, arith.shli(d, _i32c(16))))
            fq.append(arith.bitcast(T.f32, arith.andi(d, _i32c(-65536))))
        # S in code units needs Q pre-multiplied by c = 8 log2(e) scale k_descale.
        # Q is cast as fp8(clamp(q c 2^-e)) with the smallest integer
        # e >= log2(c q_scale), so queries within the static q scale's FP8 range
        # (|q| <= 448 q_scale) are not clipped, and the QK MFMA's E8M0 B scale 2^e
        # restores c exactly.
        c_q = arith.mulf(k_descale, _f32(code_scale))
        t_bits = arith.bitcast(T.i32, arith.mulf(c_q, q_scale))
        t_exp = arith.subi(
            arith.andi(arith.shrui(t_bits, _i32c(23)), _i32c(255)), _i32c(127)
        )
        t_frac = arith.andi(t_bits, _i32c(0x7FFFFF))
        e_q = arith.select(
            arith.cmpi(arith.CmpIPredicate.ne, t_frac, _i32c(0)),
            arith.addi(t_exp, _i32c(1)),
            t_exp,
        )
        e_q = arith.maxsi(arith.minsi(e_q, _i32c(126)), _i32c(-126))
        inv_q = arith.mulf(
            c_q,
            arith.bitcast(T.f32, arith.shli(arith.subi(_i32c(127), e_q), _i32c(23))),
        )
        q_sb = arith.addi(_i32c(127), e_q)
        fq = [
            arith.maximumf(
                arith.minimumf(arith.mulf(x, inv_q), _f32(448.0)), _f32(-448.0)
            )
            for x in fq
        ]
        qw = []
        for w4 in range_constexpr(8):
            e0 = 4 * w4
            hw = _rocdl.cvt_pk_fp8_f32(
                T.i32, fq[e0], fq[e0 + 1], llvm.mlir_poison(T.i32), False
            )
            qw.append(_rocdl.cvt_pk_fp8_f32(T.i32, fq[e0 + 2], fq[e0 + 3], hw, True))
        qf = _to_raw(fx.Vector.from_elements([fx.Int32(w) for w in qw], fx.Int32))
        qpos = ctx + tok

        k_base = buf_base_i64(k)
        v_base = buf_base_i64(v)

        lds = fx.SharedAllocator().allocate(Lds).peek()
        lptr = lds.kv.ptr

        # one alias scope per LDS stage: reads of stage s wait only for the DMA into s
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

        # loader: wave w fills rows 16 (2 w + n) .. + 15 of a tile's K and V; lane ->
        # row 16 n + lane // 4, physical chunk lane % 4 holding logical chunk
        # (lane % 4) ^ swizzle(row)
        swz_w = ((lane // 8) + 2 * (lane // 32)) % 4
        lc_w = fx.Int32(arith.xori(_to_raw(lane % 4), _to_raw(swz_w)))
        # The head offset goes into the descriptor base: in the cache layout the head
        # stride exceeds the row stride, and the per-block bound covers rows only.
        voff_b = ((lane // 4) * s1w + lc_w * 4) * 4
        h_off = fx.Int64(h) * fx.Int64(kv_sh)

        def pages(tile):
            # Page ids of this wave's two 16-row blocks of `tile`, clamped to the
            # request; vector loads with a uniform address (a scalar load after LDS
            # DMAs gets a conservative vmcnt(0)). Blocks 2 k and 2 k + 1 share a page
            # when pages hold more than 16 rows.
            ids = []
            for n in range_constexpr(nid):
                pi = tile * 8 + wave * 2 + n
                pc = (pi < last_page).select(pi, last_page)
                if fx.const_expr(ppb > 1):
                    pc = pc // ppb
                ids.append(
                    _to_raw(
                        buffer_ops.buffer_load(
                            bt_r, bt_row + pc, vec_width=1, dtype=T.i32
                        )
                    )
                )
            return ids

        wave_lds = _uniform(wave * 2048)

        def issue_dma(tile, stage_off, ids, sc):
            # LDS DMA: lane l of instruction (K or V, block n) lands at M0 + 16 l.
            nrec = []
            sub_off = []
            if fx.const_expr(ppb > 1):
                # the 16-row block inside the page (clamped like the page id); block
                # 2 k + 1 is the next 16 rows of the same page
                pi = tile * 8 + wave * 2
                pc = (pi < last_page).select(pi, last_page)
                o0 = fx.Int64(_uniform(ids[0])) * fx.Int64(kv_s0) + fx.Int64(
                    (pc % ppb) * (PAGE * kv_s1)
                )
                sub_off = [o0, o0 + fx.Int64(PAGE * kv_s1)]
            for n in range_constexpr(2):
                nv = seq_len - (tile * 8 + wave * 2 + n) * PAGE
                if fx.const_expr(window):
                    # blocks wholly before the window are never read: they may be
                    # unmapped or reused
                    nv = (tile * 8 + wave * 2 + n >= lo_page).select(nv, fx.Int32(0))
                nrec.append(fx.Int32(_sclamp16(nv)) * kv_s1)
            for ui, base in enumerate((k_base, v_base)):
                for n in range_constexpr(2):
                    rs = buffer_ops.create_buffer_resource_from_addr(
                        _to_raw(
                            base
                            + h_off
                            + (
                                sub_off[n]
                                if ppb > 1
                                else fx.Int64(_uniform(ids[n])) * fx.Int64(kv_s0)
                            )
                        ),
                        num_records_bytes=_to_raw(fx.Int64(nrec[n])),
                    )
                    dst = wave_lds + (stage_off + ui * (TILE * HD) + n * 1024)
                    _rocdl.raw_ptr_buffer_load_lds(
                        rs,
                        fx.to_llvm_ptr(lptr + fx.Int64(dst)),
                        _i32c(16),
                        _to_raw(voff_b),
                        _i32c(0),
                        _i32c(0),
                        **_scope_kw(sc),
                    )

        # K reads: lane (col, hl) reads key 32 mt + col, logical chunks 2 hl, 2 hl + 1
        swz_k = ((col // 2) + 2 * ((col // 8) % 2)) % 4
        k_rd = [
            col * HD
            + fx.Int32(arith.xori(_to_raw(fx.Int32(2) * hl + u), _to_raw(swz_k))) * 16
            for u in range_constexpr(2)
        ]
        # transposed V reads: lane (G, i) of read r addresses key
        # 64 ks + 32 (r // 2) + 16 (r % 2) + key0, logical chunk 2 dt + G % 2,
        # byte 8 (i % 2)
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

        zero16 = _to_raw(fx.full(16, 0.0, fx.Float32))
        zero4 = _to_raw(fx.full(4, 0.0, fx.Float32))
        # Row sums on one 16x16x128 MFMA per 64 keys: lane l's 32 P bytes are read as
        # column l % 16, k-group l // 16, so the group parity is the column half; A row
        # i keeps (ones) only the groups of parity (i // 4) % 2, which puts column
        # l % 32's sum in the rows lane l receives.
        sel_ok = ((lane // 16) % 2) == (((lane % 16) // 4) % 2)
        ones_sel = _to_raw(
            fx.Vector.from_elements(
                [sel_ok.select(fx.Int32(0x38383838), fx.Int32(0))] * 8, fx.Int32
            )
        )

        def mfma16(a, b, acc):
            return fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.vec(4, T.f32), [a, b, acc, 0, 0, 0, s127, 0, s127]
            )

        def mfma(a, b, acc, sb=s127):
            return fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                T.vec(16, T.f32), [a, b, acc, 0, 0, 0, s127, 0, sb]
            )

        def softmax_step(j, stage_off, si, o_acc, mref, thr, l_run, masked, sc):
            # Keys sub * si .. of tile j: mts row tiles of 32 keys. State: mref = the
            # exponent offset bias - m as a pair (exact) or the code-offset tile
            # (exp_free); thr = the score above which the reference must move (raw S
            # units, or code units); l_run = the row sum (a pair).
            kfr = []
            for t in range_constexpr(mts):
                mt = si * mts + t
                parts = [
                    llvm.load(
                        T.vec(4, T.i32),
                        fx.to_llvm_ptr(
                            lptr + fx.Int64(stage_off + mt * 32 * HD + k_rd[u])
                        ),
                        alignment=16,
                        **_scope_kw(sc),
                    )
                    for u in range_constexpr(2)
                ]
                kfr.append(_cat(parts[0], parts[1], 4))
            # codes: S in code units plus the column's offset (accumulator input)
            s = [mfma(kfr[t], qf, mref, q_sb) for t in range_constexpr(mts)]
            ninf = fx.Float32(float("-inf"))
            flat = []
            for t in range_constexpr(mts):
                mt = si * mts + t
                x = _vec(s[t], 16)
                if fx.const_expr(masked):
                    lim = qpos - j * TILE - 4 * hl
                    for r in range_constexpr(16):
                        kr = fx.Int32(32 * mt + 8 * (r // 4) + r % 4)
                        ok = kr <= lim
                        if fx.const_expr(window):
                            ok = ok & (kr >= lim - (window - 1))
                        flat.append(_to_raw(ok.select(x[r], ninf)))
                else:
                    flat.extend(_to_raw(x[r]) for r in range(16))
            mx = _tree_max(flat)
            need = arith.cmpf(arith.CmpFPredicate.OGT, mx, _to_raw(thr))
            vote = fx.Int64(_rocdl.ballot(T.i64, need))
            # rare: llvm.expect(false) lets block placement move the rescale out of
            # line, so the common path falls through
            hit = fx.Boolean(
                llvm.intr_expect(
                    _to_raw(vote != fx.Int64(0)),
                    arith.constant(0, type=ir.IntegerType.get_signless(1)),
                )
            )
            if hit:
                # Shift by d = new max - reference (code units). A column that
                # has only seen masked keys (thr = NO_REF) takes its first finite
                # maximum as the reference, in either direction.
                mxa = _allreduce32(_to_raw(mx))
                ini = arith.cmpf(arith.CmpFPredicate.OGT, _to_raw(thr), _f32(-1.0e29))
                fin = arith.cmpf(arith.CmpFPredicate.OGT, mxa, _f32(-1.0e29))
                d8 = arith.subf(mxa, _f32(code_ref))
                d8 = arith.select(
                    ini,
                    arith.maximumf(d8, _f32(0.0)),
                    arith.select(fin, d8, _f32(0.0)),
                )
                alpha = arith.select(
                    ini,
                    fx.rocdl.exp2(T.f32, arith.mulf(d8, _f32(-1.0 / cpo))),
                    _f32(1.0),
                )
                thr = arith.select(arith.ori(ini, fin), _f32(code_ref), _f32(NO_REF))
                mref = _to_raw(_vec(mref, 16) - fx.Float32(d8))
                flat = [arith.subf(_to_raw(x), d8) for x in flat]
                l_run = _scale_np(l_run, 2, alpha)
                o_acc = [_scale_np(o, 16, alpha) for o in o_acc]
            words = []
            if fx.const_expr(exp_free):
                fx.rocdl.sched_barrier(0)
                for w in range_constexpr(mts * 4):
                    acc_w = _i32c(0)
                    for b in range_constexpr(4):
                        acc_w = _cvt_u8(_to_raw(flat[4 * w + b]), b, acc_w)
                    words.append(acc_w)
            else:
                zs = [
                    [flat[2 * i], flat[2 * i + 1]]
                    for i in range_constexpr(len(flat) // 2)
                ]
                fx.rocdl.sched_barrier(0)
                for t in range_constexpr(mts):
                    ps = []
                    for r in range_constexpr(0, 16, 2):
                        z = zs[t * 8 + r // 2]
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
            o_new = list(o_acc)
            lt = zero4
            for kq in range_constexpr(mts // 2):
                # P^T B operand of 64 keys (row tiles 2 ks, 2 ks + 1)
                ks = si * (mts // 2) + kq
                pb = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in words[8 * kq : 8 * kq + 8]], fx.Int32
                    )
                )
                # row sums first, so their consumer sits after the PV MFMAs
                lt = mfma16(ones_sel, pb, lt)
                for dt in range_constexpr(2):
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
            # The row sum is carried as a pair (element 1 stays 0): as a lone scalar
            # in the loop state, register allocation spills at 4 waves / SIMD.
            lv = _vec(l_run, 2)
            l_new = arith.addf(_to_raw(lv[0]), _to_raw(_vec(lt, 4)[0]))
            l_run = _to_raw(
                fx.Vector.from_elements([fx.Float32(l_new), lv[1]], fx.Float32)
            )
            return o_new, mref, thr, l_run

        def tile_barrier():
            # Tile j's DMAs were issued one iteration ago and are the oldest VMEM ops
            # in flight: wait for them, then a bare s_barrier (gpu.barrier's release
            # fence would also wait for the DMAs just issued).
            fx.rocdl.sched_barrier(0)
            fx.rocdl.s_waitcnt(vmcnt=0)
            _rocdl.s_barrier()
            fx.rocdl.sched_barrier(0)

        def step(j, st, stage, masked):
            # stage: Python int (static LDS stage, scoped reads) or None (dynamic)
            o_acc = list(st[0:2])
            mref, thr, l_run = st[2], st[3], st[4]
            ids = list(st[5 : 5 + nid])
            tile_barrier()
            ja = j + t0  # absolute tile (j counts from the block's first tile)
            nids = pages(ja + NBUF)
            fx.rocdl.sched_barrier(0)
            if fx.const_expr(stage is None):
                issue_dma(ja + NBUF - 1, ((j + NBUF - 1) % NBUF) * STAGE, ids, None)
                stage_off = (j % NBUF) * STAGE
                sc = None
            else:
                nst = (stage + NBUF - 1) % NBUF
                issue_dma(ja + NBUF - 1, nst * STAGE, ids, nst)
                stage_off = stage * STAGE
                sc = stage
            fx.rocdl.sched_barrier(0)
            for si in range_constexpr(nsub):
                o_acc, mref, thr, l_run = softmax_step(
                    ja, stage_off, si, o_acc, mref, thr, l_run, masked, sc
                )
            return (
                [_to_raw(x) for x in o_acc]
                + [_to_raw(mref), _to_raw(thr), _to_raw(l_run)]
                + [_to_raw(x) for x in nids]
            )

        # prologue: tiles 0 .. NBUF - 2 in flight, page ids of tile NBUF - 1
        for t in range_constexpr(NBUF - 1):
            issue_dma(t0 + t, t * STAGE, pages(t0 + t), t)
        ids0 = pages(t0 + (NBUF - 1))
        # code offset 0 until a column's first finite maximum sets its reference
        init = [zero16, zero16, zero16, _f32(NO_REF)]
        init = init + [_to_raw(fx.full(2, 0.0, fx.Float32))] + ids0
        # tiles counted from t0: [0, nlo) window-masked (rounded up to whole stage
        # groups), [nlo, n_in) unmasked in groups of NBUF with static LDS stages then
        # the rest with dynamic stages, [n_in, nt) causal-masked
        nt = n_tiles - t0
        n_in = interior - t0
        n_in = (n_in < nt).select(n_in, nt)
        n_in = (n_in > 0).select(n_in, fx.Int32(0))
        if fx.const_expr(window):
            nlo = lo_full - t0
            nlo = (nlo > 0).select(nlo, fx.Int32(0))
            nlo = (nlo + NBUF - 1) // NBUF * NBUF
            nlo = (nlo < nt).select(nlo, nt)
            n_in = (n_in > nlo).select(n_in, nlo)
            for jj, st in range(0, nlo, 1, init=init):  # type: ignore[call-overload]
                j = fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
                fin_lo = yield step(j, st, None, True)
            init = list(fin_lo)
        else:
            nlo = fx.Int32(0)
        ngr = (n_in - nlo) // NBUF
        for gi, st in range(0, ngr, 1, init=init):  # type: ignore[call-overload]
            j0 = nlo + fx.Int32(arith.index_cast(T.i32, _to_raw(gi))) * NBUF
            for u in range_constexpr(NBUF - 1):
                st = step(j0 + u, st, u, False)
            fin0 = yield step(j0 + NBUF - 1, st, NBUF - 1, False)
        for jj, st in range(nlo + ngr * NBUF, n_in, 1, init=list(fin0)):  # type: ignore[call-overload]
            j = fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
            fin = yield step(j, st, None, False)
        for jj, st in range(0, nt - n_in, 1, init=list(fin)):  # type: ignore[call-overload]
            j = n_in + fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
            fin2 = yield step(j, st, None, True)
        o_acc = list(fin2[0:2])
        lsum = _vec(fin2[4], 2)
        inv = arith.divf(v_descale, arith.addf(_to_raw(lsum[0]), _to_raw(lsum[1])))

        o_base = buf_base_i64(out)
        o_rs = buffer_ops.create_buffer_resource_from_addr(
            _to_raw(o_base + blk_tok * fx.Int64(o_s0 * 2)),
            num_records_bytes=_to_raw(fx.Int64(blk_rows) * fx.Int64(o_s0 * 2)),
        )
        o_row = tok_b * o_s0 + head * HD + hl * 4
        for dt in range_constexpr(2):
            ov = _vec(o_acc[dt], 16) * fx.Float32(inv)
            for i in range_constexpr(4):
                seg = ov.shuffle(ov, [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]).to(
                    fx.BFloat16
                )
                buffer_ops.buffer_store(_to_raw(seg), o_rs, o_row + 32 * dt + 8 * i)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        out: fx.Tensor,
        cuq: fx.Tensor,
        seqlens: fx.Tensor,
        table: fx.Tensor,
        qd: fx.Tensor,
        kd: fx.Tensor,
        vd: fx.Tensor,
        num_seqs: fx.Int32,
        nqb: fx.Int32,
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
            num_seqs,
            nqb,
            bt_s0,
            q_s0,
            o_s0,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(NW * 64, 1, 1), stream=stream)

    return launch


def fp8_extend_attention(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    seq_lens,
    block_table,
    q_scale,
    k_scale,
    v_scale,
    scale: float,
    window: int = 0,
    bias: float = 0.0,
    exp_free: bool = False,
) -> None:
    """Causal extend attention: `out [tokens, 8 * kv_heads, 64]` (BF16) over the
    paged FP8 cache (`k_cache` / `v_cache` `[pages, page, kv_heads, 64]` views).
    `q` is FP8 e4m3 quantized with `q_scale` (exact softmax) or BF16 (`exp_free`).
    `seq_lens` counts each request's cached plus new tokens; `q_scale`, `k_scale`,
    `v_scale` are one-element float32 tensors; `window` > 0 limits each query to its
    last `window` keys. P at the running maximum is 2^bias (bias <= log2(448), or
    <= 8.75 for exp_free codes). int32 metadata."""
    import torch

    assert 0.0 <= bias <= (MAX_CODE_BIAS if exp_free else LOG2_448)
    n = q.shape[0]
    hk = k_cache.shape[2]
    bs = seq_lens.numel()
    nqb = n // BQ + bs
    _run_compiled(
        _launcher(
            hk,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            float(scale),
            max(8, bs.bit_length()),  # one compile for up to 255 requests
            float(bias),
            k_cache.shape[1] // PAGE,
            int(window),
            bool(exp_free),
        ),
        q,
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        out,
        cu_seqlens_q,
        seq_lens,
        block_table,
        q_scale,
        k_scale,
        v_scale,
        int(bs),
        int(nqb),
        int(block_table.stride(0)),
        int(q.stride(0)) * q.element_size(),
        int(out.stride(0)),
        int(nqb * hk),
        fx.Stream(torch.cuda.current_stream(q.device).cuda_stream),
    )
