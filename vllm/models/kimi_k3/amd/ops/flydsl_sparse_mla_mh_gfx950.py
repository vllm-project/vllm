# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
# mypy: ignore-errors

"""FlyDSL sparse MLA forward over fp8_ds_mla records for many heads, gfx950.

The DCP branch of Kimi-K3.1's sparse MLA attends all 96 gathered heads of a
query row over this rank's shard of the cache (about topk / dcp_world_size
keys per row). ``flydsl_sparse_mla_gfx950`` serves at most 16 heads, one
workgroup per row; run once per 16-head slice, it would gather every KV tile
six times. This kernel gathers each tile into LDS once for all head groups.
The host wrapper and its support predicate live in ``flydsl_sparse_mla.py``.

Same math as ``flydsl_sparse_mla_gfx950`` (Q split into fp8 hi + lo, scaled
fp8 MFMA on the raw latent with the record's pow2 scale as E8M0, bf16 MFMA for
rope, fp16 PV on tr8-read raw fp8 with the V scale folded into P), laid out
for many heads:

  * Workgroup = (query row, head split). It serves HG groups of 16 heads with
    2 waves per group; NSPLIT = H / (16 HG) workgroups share a row, placed on
    one XCD so they share its L2.
  * The two waves of a group split the 512 latent columns: wave half c owns
    columns [256c, 256c + 256) (record scale blocks 2c, 2c + 1) and rope dims
    [32c, 32c + 32). It holds only its share of Q and accumulates only
    O[:, 256c : 256c + 256].
  * QK: each wave computes partial S over its columns for all TT tokens of the
    tile; the pair swaps the partials through LDS, so both hold the full S and
    run the same softmax (no P exchange).
  * Softmax: P = exp2(s - m + 4), with m allowed to lag the running max by up
    to 4 (the O rescale is skipped while it does), so P <= 2^8 as in the
    16-head kernel.
  * PV: the S register layout is the PV B operand when chunk c's K order is
    k = 8g + i <-> token 32c + 16(i // 4) + 4g + i % 4, which the tr8 V^T
    reads use.
  * KV tiles of TT tokens (latent 2 x [TT][256 B] XOR-swizzled, rope
    [TT][128 B], scales [TT][16 B]) are gathered into NS LDS stages by
    global_load ... lds on 64-bit per-lane addresses, so the cache may span
    more than 4 GiB. The DMAs carry an alias scope that every LDS access is
    declared not to alias, so the compiler does not wait for the in-flight
    tile before each ds_read; the explicit vmcnt wait + barrier at the top of
    each tile orders them.

A row with no key gives out 0 and lse -inf.

Preconditions the host cannot check (as for ``flydsl_sparse_mla_gfx950``):
  * Every per-128 KV scale is a power of two <= 2^7 (vLLM's fp8_ds_mla writer
    stores pow2 scales, <= 2^7 iff |latent| <= 57344). Unlike the 16-head
    kernel this one does not clamp a larger scale.
  * Every index in [indptr[0], indptr[T]) is a valid cache row: there is no
    has_invalid masking. Positions past a row's end are masked; they read the
    row's first key, so the output does not depend on any other cache row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import GTensor
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.kimi_k3.amd.ops.flydsl_sparse_mla_gfx950 import (
    KV_LORA_RANK,
    RECORD_BYTES,
    _BoundedI32,
    _cat,
    _cvt_v8,
    _e8m0_for,
    _fmax,
    _guarded,
    _lds_llvm_ptr,
    _mfma_bf16,
    _mfma_f16,
    _mfma_fp8,
    _pack_fp8x4,
    _readfirstlane,
    _xmax,
    _xsum,
)

_LN2 = 0.6931471805599453
# P = exp2(s - m + _P_SHIFT) with m allowed to lag the running max by up to
# _TAU (lazy O rescale), so P <= 2^(_TAU + _P_SHIFT) and P' = P * scale <= 2^15
# fits fp16 for KV scales <= 2^7.
_P_SHIFT = 4.0
_TAU = 4.0
_NEG = -1.0e30


def _dma_scope():
    """Alias scope of the KV DMAs (an ArrayAttr; built while the kernel is traced)."""
    dom = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "mh_lds">'
    dma = ir.Attribute.parse(
        f'#llvm.alias_scope<id = distinct[1]<>, domain = {dom}, description = "kv_dma">'
    )
    return ir.ArrayAttr.get([dma])


def _ld(sc, elem, n, addr, align):
    """LDS load of n x elem that does not alias an in-flight KV DMA (scope sc)."""
    ty = T.vec(n, elem) if n > 1 else elem
    v = llvm.load(ty, _lds_llvm_ptr(addr), alignment=align, noalias_scopes=sc)
    return Vec(v) if n > 1 else v


def _st(sc, vec, addr, align):
    llvm.store(
        Vec(vec).ir_value(), _lds_llvm_ptr(addr), alignment=align, noalias_scopes=sc
    )


def _tr8(sc, addr):
    return Vec(
        rocdl.ds_read_tr8_b64_(T.vec(2, T.i32), _lds_llvm_ptr(addr), noalias_scopes=sc)
    )


def _fsw(tok):
    """16-B chunk swizzle of a 256-B latent half row: conflict-free QK (16
    consecutive tokens) and tr8 reads."""
    return (tok & 15) ^ (((tok >> 4) & 1) << 3)


def _rsw(tok):
    """16-B chunk swizzle of a 128-B rope row."""
    return (tok >> 1) & 7


def _f32(x):
    return fx.Int32(x).bitcast(fx.Float32)


def _amax_bf16(words):
    """The max |x| over bf16 pairs (i32 words), as f32: a packed u16 max of the
    magnitude bits."""
    from flydsl._mlir.dialects import arith

    v2 = T.vec(2, T.i16)
    m = None
    for w in words:
        mw = llvm.bitcast(v2, _raw(fx.Int32(fx.Int32(w) & 0x7FFF7FFF).ir_value()))
        m = mw if m is None else arith.maxui(m, mw)
    mi = fx.Int32(llvm.bitcast(T.i32, m))
    lo16 = mi & 0xFFFF
    hi16 = (mi >> 16) & 0xFFFF
    return ((lo16 > hi16).select(lo16, hi16) << 16).bitcast(fx.Float32)


def _quant_bf16(words, inv):
    """bf16 pairs * inv -> fp8 hi = the value with its mantissa truncated to 3
    bits (exact in e4m3 for |x * inv| >= 2^-6; smaller values are negligible
    next to the block's amax in [128, 256)) and fp8 lo = the remainder * 16
    (E8M0 of hi minus 4; the remainder is below one hi ulp, so * 16 stays
    < 256)."""
    hv, lv = [], []
    for w in words:
        w = fx.Int32(w)
        t = w & fx.Int32(-0x000F0010)
        x = [
            (w << 16).bitcast(fx.Float32),
            (w & fx.Int32(-0x10000)).bitcast(fx.Float32),
        ]
        h = [
            (t << 16).bitcast(fx.Float32),
            (t & fx.Int32(-0x10000)).bitcast(fx.Float32),
        ]
        for k in range(2):
            hv.append(h[k] * inv)
            lv.append((x[k] - h[k]) * (inv * 16.0))
    return (
        [_pack_fp8x4(hv[4 * i : 4 * i + 4]) for i in range(len(hv) // 4)],
        [_pack_fp8x4(lv[4 * i : 4 * i + 4]) for i in range(len(lv) // 4)],
    )


def _barrier():
    """s_barrier without the workgroup fence gpu.barrier adds: that release
    fence would wait (vmcnt) for the in-flight KV DMA. Callers wait lgkmcnt /
    vmcnt themselves; the memory clobber keeps LDS accesses in place."""
    llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)


def _order(pin=()):
    """Compiler-only fence: LDS accesses are not moved across it, and the
    values in `pin` (v4f32 MFMA accumulators) are produced before it (no-op asm
    with tied operands). Returns the pinned values."""
    pin = list(pin)
    if not pin:
        llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
        rocdl.sched_barrier(0)
        return []
    irs = [Vec(v).ir_value() for v in pin]
    if len(irs) == 1:
        r = llvm.inline_asm(
            irs[0].type, irs, "", "=v,0,~{memory}", has_side_effects=True
        )
        rocdl.sched_barrier(0)
        return [Vec(r)]
    ty = ir.Type.parse(f"!llvm.struct<({', '.join(['vector<4xf32>'] * len(irs))})>")
    cons = ",".join(
        ["=v"] * len(irs) + [str(i) for i in range(len(irs))] + ["~{memory}"]
    )
    ret = llvm.inline_asm(ty, irs, "", cons, has_side_effects=True)
    rocdl.sched_barrier(0)
    return [Vec(llvm.extractvalue(irs[i].type, ret, [i])) for i in range(len(irs))]


def build_sparse_mla_fp8ds_mh_module(num_heads, hg, tt, ns, waves_per_eu, return_lse):
    """Launcher for H = num_heads heads: hg 16-head groups per workgroup, tt-token
    KV tiles in ns LDS stages, waves_per_eu as the compile hint."""
    H = num_heads
    assert H % 16 == 0 and (H // 16) % hg == 0, (H, hg)
    assert tt in (32, 64)
    nsplit = H // 16 // hg
    W = 2 * hg  # waves
    NTH = 64 * W
    NS16 = tt // 16  # QK subtiles per tile
    NC = tt // 32  # PV chunks per tile
    ROPE_OFF = tt * 512
    SC_OFF = ROPE_OFF + tt * 128
    STAGE = SC_OFF + 1024
    NS = ns  # KV stages: tile i + NS - 1 is gathered while tile i is computed
    assert NS & (NS - 1) == 0 and NS >= 2
    XCH_OFF = NS * STAGE
    # Epilogue: a wave's O tile as bf16 [16 heads][256 dims], row pitch 512 + 16 B.
    OP = 528
    LDS = max(XCH_OFF + hg * 2 * NS16 * 1024, W * 16 * OP)
    assert LDS <= 160 * 1024, LDS
    NL, NR = tt // 2, tt // 8  # latent / rope DMA jobs of 1 KB; + 1 scale job
    NJ = NL + NR + 1
    J = (NJ + W - 1) // W  # DMA instructions per wave per tile

    name = (
        f"sparse_mla_fp8ds_mh_h{H}_g{hg}_t{tt}_s{ns}_w{waves_per_eu}_l{int(return_lse)}"
    )

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, LDS, 1024]

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def mh_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        IDX: fx.Tensor,
        IPTR: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        stride_q: fx.Int32,
        stride_qh: fx.Int32,
        stride_o: fx.Int32,
        stride_oh: fx.Int32,
        c_log2: fx.Float32,
        n_q: fx.Int32,
        idx_bytes: fx.Int32,
    ):
        sc = _dma_scope()
        tid = fx.Int32(fx.thread_idx.x)
        w = _readfirstlane(tid // 64)
        lane = tid % 64
        l16 = lane % 16
        g = lane // 16
        grp = w // 2
        half = w % 2
        bid = fx.Int32(fx.block_idx.x)
        if const_expr(nsplit > 1):
            # Workgroups go round-robin over the 8 XCDs: give each XCD a
            # contiguous run of (row, split) so a row's splits share one L2.
            per = fx.Int32(n_q * nsplit) // 8
            sw = (bid % 8) * per + bid // 8
            bid = (bid < per * 8).select(sw, bid)
            sp = bid % nsplit
            qi = bid // nsplit
        else:
            sp = fx.Int32(0)
            qi = bid
        gi = sp * hg + grp  # 16-head group
        head = gi * 16 + l16

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_t = GTensor(KV, dtype=T.i32, shape=(-1,))
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))
        kv_base = kv_t.base_i64
        kv_gptr = fx.PointerType.get(T.i8, fx.AddressSpace.Global, 1)
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        # ---- this row's CSR slice and the indices of tiles 0 .. NS - 1 (issued
        # before Q: vmcnt is in order)
        ip_rsrc = ip_t.rsrc
        r_beg = fx.Int32(
            buffer_ops.buffer_load(ip_rsrc, qi, vec_width=1, is_scalar=True)
        )
        r_end = fx.Int32(
            buffer_ops.buffer_load(ip_rsrc, qi + 1, vec_width=1, is_scalar=True)
        )
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)
        nt = _readfirstlane((r_len + (tt - 1)) // tt)
        idxv = [fx.Int32(idx_t[r_beg + tt * t + lane]) for t in range(NS)]
        # Padded tile positions gather this row's first key: a real record, so
        # their masked P' = 0 contributes exactly 0 (row 0 could hold NaN bytes).
        # An empty row computes no tile; its tile-0 gather reads row 0, never
        # indices[r_beg], which lies past the row (possibly past the valid part
        # of the index buffer).
        pad_ix = (r_len > 0).select(_readfirstlane(idxv[0]), fx.Int32(0))

        # ---- DMA jobs of this wave (1 KB each; lane L writes LDS bytes
        # 16L .. 16L + 15 of the job):
        #   latent n < NL: half n // (tt/4), tokens 4(n % (tt/4)) + L // 16,
        #     swizzled 16-B chunk L % 16;
        #   rope n - NL: tokens 8(n - NL) + L // 8, swizzled chunk L % 8;
        #   scale: token L % tt.
        # Slots past NJ redo job n - NJ (same bytes, harmless).
        def job(k, lane):
            """-> (token, source byte in the record, stage-relative LDS byte) of
            this wave's DMA slot k."""
            n = w + W * k
            n = (n < NJ).select(n, n - NJ)
            is_l = n < NL
            is_r = n < NL + NR
            lh = n // (tt // 4)
            lt0 = 4 * (n % (tt // 4))
            ltok = lt0 + lane // 16
            lsrc = 256 * lh + 16 * ((lane % 16) ^ _fsw(ltok))
            ldst = lh * (tt * 256) + lt0 * 256
            rj = n - NL
            rtok = 8 * rj + lane // 8
            rsrc = 528 + 16 * ((lane % 8) ^ _rsw(rtok))
            rdst = ROPE_OFF + 1024 * rj
            stok = lane % tt
            return (
                is_l.select(ltok, is_r.select(rtok, stok)),
                is_l.select(lsrc, is_r.select(rsrc, fx.Int32(512))),
                _readfirstlane(is_l.select(ldst, is_r.select(rdst, fx.Int32(SC_OFF)))),
            )

        def dma(idxv, tb, stage, lane=lane):
            """Gather tile rows [tb, tb + tt) into stage: idxv lane l holds
            indices[r_beg + tb + l]. Always J DMAs per wave (positions past
            the end read pad_ix), so the vmcnt waits below are static. The row
            offset is
            64-bit: a buffer descriptor's 32-bit offset cannot reach past 4 GiB,
            and its wave-uniform base cannot follow a per-lane gather."""
            for k in range_constexpr(J):
                tok, src, dst = job(k, lane)
                ix = fx.Int32(
                    rocdl.ds_bpermute(T.i32, _raw(tok * 4), _raw(fx.Int32(idxv)))
                )
                ix = (tb + tok < r_len).select(ix, pad_ix)
                addr = kv_base + fx.Int64(ix) * fx.Int64(RECORD_BYTES) + fx.Int64(src)
                rocdl.global_load_lds(
                    _raw(fx.to_llvm_ptr(fx.inttoptr(kv_gptr, addr))),
                    _lds_llvm_ptr(base + stage * STAGE + dst),
                    16,
                    0,
                    alias_scopes=sc,
                )

        # Tile 0 first (positions past the end read pad_ix, so no guard), then Q,
        # then tiles 1 .. NS - 2: the first tile's wait (vmcnt <= (NS - 2) J)
        # then covers exactly tile 0 and Q.
        dma(idxv[0], fx.Int32(0), fx.Int32(0))
        _order()
        # ---- Q: lane (h = l16, d = g) holds, for scale block 2 half + j, dims
        # 32d .. 32d + 31 as fp8 hi / lo (one E8M0 per head and block), and rope
        # dims 32 half + 8d .. + 7 (bf16).
        q_row = (qi * stride_q + head * stride_qh) // 2  # i32 units
        qv = []
        for j in range_constexpr(2):
            d0 = (256 * half + 128 * j + 32 * g) // 2
            parts = [Vec(q_t.load(q_row + d0 + 4 * k, vec_size=4)) for k in range(4)]
            qv.append([fx.Int32(p[i]) for p in parts for i in range(4)])  # 32 bf16
        qr = Vec(q_t.load(q_row + (KV_LORA_RANK + 32 * half + 8 * g) // 2, vec_size=4))

        _order()
        for t in range_constexpr(1, NS - 1):
            _guarded(
                fx.Int32(t) < nt,
                lambda _v=idxv[t], _t=t: dma(_v, fx.Int32(tt * _t), fx.Int32(_t)),
            )

        # ---- quantize Q (waits for the Q loads only; the DMA was issued after them)
        q8, q8l, qs, qsl = [], [], [], []
        for j in range_constexpr(2):
            a = _amax_bf16(qv[j])
            a = _xmax(a)  # amax over the head's 128 dims of block 2 half + j
            s, inv = _e8m0_for(a)
            hi, lo = _quant_bf16(qv[j], inv)
            q8.append(Vec.from_elements(hi, fx.Int32))
            q8l.append(Vec.from_elements(lo, fx.Int32))
            qs.append(s)
            ls = fx.Int32(s) - 4
            qsl.append((ls < 0).select(fx.Int32(0), ls))

        def lane_offsets(lane):
            """Per-lane LDS offsets (stage-relative). Swizzles only touch address
            bits 4-7, so a subtile / chunk / block offset is an XOR on those bits
            plus an add above them (an immediate)."""
            l16 = lane % 16
            g = lane // 16
            hb = half * (tt * 256)
            qk0 = (
                hb + l16 * 256 + 16 * ((2 * g) ^ l16)
            )  # + 4096 s, ^ 128 ((s & 1) ^ j) ^ 16 e
            rp0 = ROPE_OFF + l16 * 128 + 16 * ((4 * half + g) ^ _rsw(l16))  # + 2048 s
            sc0 = SC_OFF + l16 * 16 + 8 * half  # + 256 s
            ps0 = SC_OFF + 64 * g + 8 * half  # + 16 (32 c + 16 s2 + r)
            i8 = l16 // 2
            tok0 = 16 * (i8 // 4) + 4 * g + i8 % 4
            pv0 = hb + tok0 * 256 + 8 * (l16 % 2) + 16 * _fsw(tok0)  # ^ 16 d, + 8192 c
            x_wr = (
                base + XCH_OFF + (grp * 2 + half) * (NS16 * 1024) + lane * 16
            )  # + 1024 s
            x_rd = base + XCH_OFF + (grp * 2 + 1 - half) * (NS16 * 1024) + lane * 16
            return g, qk0, rp0, sc0, ps0, pv0, x_wr, x_rd

        # ---- tile phases (closures over the per-lane offsets)
        def qk_phase(sb, offs):
            """Partial S^T over this wave's columns of the tile in stage sb (lane
            (h = l16, g) holds tokens 16s + 4g + r), written to this wave's
            exchange slot."""
            qk0, rp0, sc0, x_wr = offs[1], offs[2], offs[3], offs[6]
            qkb = sb + qk0

            def qk_read(s, j):
                if const_expr(j == 2):
                    return _ld(sc, T.i32, 4, sb + rp0 + 2048 * s, 16)
                x = 128 * ((s & 1) ^ j)
                return _cat(
                    _ld(sc, T.i32, 4, (qkb ^ x) + 4096 * s, 16),
                    _ld(sc, T.i32, 4, (qkb ^ (x + 16)) + 4096 * s, 16),
                    4,
                )

            # Block j = 0 for every subtile, then j = 1, then rope; operands are
            # read one step ahead.
            steps = [(s_, j_) for j_ in range(3) for s_ in range(NS16)]
            scp = [_ld(sc, T.i32, 2, sb + sc0 + 256 * s_, 8) for s_ in range(NS16)]
            ops = [qk_read(*steps[0])]
            qh = q8[0]
            acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range(NS16)]
            for k in range_constexpr(len(steps)):
                s, j = steps[k]
                qcur = qh
                if const_expr(k + 1 < len(steps)):
                    ops.append(qk_read(*steps[k + 1]))
                    if const_expr(steps[k + 1][0] == 0 and steps[k + 1][1] < 2):
                        qh = q8[steps[k + 1][1]]
                _order()
                if const_expr(j < 2):
                    se = (fx.Int32(scp[s][j]) >> 23) & 0xFF
                    acc[s] = _mfma_fp8(ops[k], qcur, acc[s], se, qs[j])
                    acc[s] = _mfma_fp8(ops[k], q8l[j], acc[s], se, qsl[j])
                else:
                    acc[s] = _mfma_bf16(ops[k], qr, acc[s])
                    _st(sc, acc[s], x_wr + 1024 * s, 16)
                ops[k] = None

        def read_s(offs):
            """Full S (own + partner partial) * log2(e) * softmax_scale."""
            x_wr, x_rd = offs[6], offs[7]
            sl = []
            for s in range_constexpr(NS16):
                own = _ld(sc, T.f32, 4, x_wr + 1024 * s, 16)
                oth = _ld(sc, T.f32, 4, x_rd + 1024 * s, 16)
                for r in range_constexpr(4):
                    sl.append((fx.Float32(own[r]) + fx.Float32(oth[r])) * c_log2)
            return sl

        def softmax(sl, tb, m_old, l_old, o, gl):
            """Online softmax step.

            -> (sl masked, m_new, l_old * alpha, m_cl, o rescaled).
            """

            # Only the last tile can hold positions >= r_len (wave-uniform branch).
            @flyc.jit
            def _mask(sl, tb=tb):
                if tb + tt > r_len:
                    sl = [
                        (tb + 16 * (k // 4) + 4 * gl + k % 4 < r_len).select(
                            sl[k], fx.Float32(_NEG)
                        )
                        for k in range(len(sl))
                    ]
                return sl

            sl = _mask(sl)
            tmax = fx.Float32(_NEG)
            for x in sl:
                tmax = _fmax(tmax, x)
            tmax = _xmax(tmax)
            # Lazy rescale: move m (and rescale O) only when the tile max exceeds
            # it by more than _TAU.
            m_cand = _fmax(m_old, tmax)
            upd = m_cand > m_old + _TAU
            m_new = upd.select(m_cand, m_old)
            alpha = (m_old - m_new).exp2(fastmath="afn")
            m_cl = _fmax(m_new - _P_SHIFT, fx.Float32(-1.0e20))
            any_upd = fx.Int64(rocdl.ballot(T.i64, _raw(upd.ir_value()))) != fx.Int64(0)

            @flyc.jit
            def _rescale(o, alpha=alpha, any_upd=any_upd):
                if any_upd:
                    o = [
                        Vec.from_elements(
                            [fx.Float32(Vec(o[i])[k]) * alpha for k in range(4)],
                            fx.Float32,
                        )
                        for i in range(len(o))
                    ]
                return o

            return sl, m_new, l_old * alpha, m_cl, _rescale(o)

        def pv_phase(sb, offs, sl, m_cl, l_new, o):
            """O^T[d, h] += V^T[d, tok] P'^T[tok, h], P' = P * (token's scale of
            the column block). Groups of 4 output blocks; group k + 1's tr8 (and
            scale) reads are issued before group k's MFMAs, and group k - 1's
            accumulators are pinned there so the MFMAs stay in place.
            -> (o, l_new)."""
            pvb = sb + offs[5]
            ps0 = offs[4]
            groups = [(c_, q_) for c_ in range(NC) for q_ in range(4)]

            def pv_reads(k):
                c, q = groups[k]
                rd = [_tr8(sc, (pvb ^ (16 * (4 * q + i))) + 8192 * c) for i in range(4)]
                psc = None
                if const_expr(q % 2 == 0):
                    # V scale of token 32c + 16 (i // 4) + 4g + i % 4,
                    # column block 2 half + q // 2.
                    psc = [
                        _ld(
                            sc,
                            T.i32,
                            1,
                            sb
                            + ps0
                            + 4 * (q // 2)
                            + 16 * (32 * c + 16 * (i // 4) + i % 4),
                            4,
                        )
                        for i in range(8)
                    ]
                return rd, psc

            nxt = pv_reads(0)
            pl = None
            pp = None
            for k in range_constexpr(len(groups)):
                c, q = groups[k]
                cur = nxt
                if const_expr(k + 1 < len(groups)):
                    nxt = pv_reads(k + 1)
                if const_expr(k > 0):
                    pq = groups[k - 1][1]
                    o[4 * pq : 4 * pq + 4] = _order(o[4 * pq : 4 * pq + 4])
                else:
                    _order()
                if const_expr(q == 0):
                    pl = [(sl[8 * c + i] - m_cl).exp2(fastmath="afn") for i in range(8)]
                    for x in pl:
                        l_new = l_new + x
                if const_expr(q % 2 == 0):
                    vals = [pl[i] * _f32(cur[1][i]) for i in range(8)]
                    pp = (
                        Vec.from_elements(vals, fx.Float32)
                        .to(fx.Float16)
                        .bitcast(fx.Int32)
                    )
                for i in range_constexpr(4):
                    d = 4 * q + i
                    o[d] = _mfma_f16(_cvt_v8(cur[0][i]), pp, o[d])
            o[12:16] = _order(o[12:16])
            return o, l_new

        offs = lane_offsets(lane)
        init = [_raw(fx.Float32(_NEG).ir_value()), _raw(fx.Float32(0.0).ir_value())]
        init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(16)]
        init += [_raw(idxv[NS - 1].ir_value())]
        for tile, st in range(0, nt, 1, init=init):
            m_old = fx.Float32(st[0])
            l_old = fx.Float32(st[1])
            o = [Vec(st[2 + i]) for i in range(16)]
            idxv_c = fx.Int32(st[18])
            tb = fx.Int32(tile) * tt
            stage = fx.Int32(tile) & (NS - 1)
            sb = base + stage * STAGE

            # Tile `tile` has landed (own DMAs: the NS - 2 later tiles' may still
            # be in flight, or none near the end, where they were not issued) and
            # is visible (barrier); everyone is done with tile - 1's stage, which
            # takes tile + NS - 1.
            if const_expr(NS > 2):
                _guarded(
                    fx.Int32(tile) + (NS - 2) >= nt, lambda: rocdl.s_waitcnt(vmcnt=0)
                )
            rocdl.s_waitcnt(vmcnt=(NS - 2) * J)
            _barrier()
            next_tb = tb + (NS - 1) * tt
            next_stage = (stage + (NS - 1)) & (NS - 1)
            _guarded(
                fx.Int32(tile) + (NS - 1) < nt,
                lambda _v=idxv_c, _tb=next_tb, _s=next_stage: dma(_v, _tb, _s, lane),
            )
            idxv_n = fx.Int32(idx_t[r_beg + tb + NS * tt + lane])
            qk_phase(sb, offs)
            rocdl.s_waitcnt(lgkmcnt=0)
            _barrier()
            sl = read_s(offs)
            sl, m_new, l_new, m_cl, o = softmax(sl, tb, m_old, l_old, o, offs[0])
            o, l_new = pv_phase(sb, offs, sl, m_cl, l_new, o)
            fin = yield [m_new, l_new] + o + [idxv_n]

        # ---- epilogue
        rocdl.s_waitcnt(vmcnt=0)
        l_tot = _xsum(fx.Float32(fin[1]))
        has_l = l_tot > fx.Float32(0.0)
        inv_l = has_l.select(fx.Float32(1.0) / l_tot, fx.Float32(0.0))
        # O goes through LDS so each store instruction writes whole 512-B row segments.
        _barrier()  # everyone is done with the KV stages and the exchange buffer
        ob = base + w * (16 * OP)
        for d in range_constexpr(16):
            ov = Vec(fin[2 + d])
            bf = Vec.from_elements(
                [fx.Float32(ov[k]) * inv_l for k in range(4)], fx.Float32
            ).to(fx.BFloat16)
            _st(sc, bf.bitcast(fx.Int32), ob + l16 * OP + 32 * d + 8 * g, 8)
        rocdl.s_waitcnt(lgkmcnt=0)
        for k in range_constexpr(8):
            e = lane + 64 * k
            row = e // 32
            ch = e % 32
            v = _ld(sc, T.i32, 4, ob + row * OP + 16 * ch, 16)
            o_t.store(
                (qi * stride_o + (gi * 16 + row) * stride_oh + 256 * half + 8 * ch)
                // 2,
                v,
                vec_size=4,
            )
        if const_expr(return_lse):
            m_fin = fx.Float32(fin[0])
            lse2 = (
                m_fin
                + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0))))
                - _P_SHIFT
            )
            lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
            lse_bits = lse.bitcast(fx.Int32)

            def _st_lse(_v=lse_bits, _off=qi * H + head):
                lse_t.store(_off, _v)

            _guarded((half + g) == 0, _st_lse)

    @flyc.jit
    def launch_mh(
        Q,
        KV,
        IDX,
        IPTR,
        OUT,
        LSE,
        n_q,
        stride_q,
        stride_qh,
        stride_o,
        stride_oh,
        c_log2,
        idx_bytes,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        mh_kernel(
            Q,
            KV,
            IDX,
            IPTR,
            OUT,
            LSE,
            stride_q,
            stride_qh,
            stride_o,
            stride_oh,
            c_log2,
            n_q,
            idx_bytes,
        ).launch(grid=(fx.Int64(n_q * nsplit), 1, 1), block=(NTH, 1, 1), stream=stream)

    launch_mh.compile_hints = {"waves_per_eu": waves_per_eu, "fast_fp_math": False}
    return launch_mh
