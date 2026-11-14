# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102
"""CuTe DSL device code of stage `front` (the host side is `front.py`).

A operand in shared memory at bucket 8, early x polling, Lamport sentinel hand-off, one weight phase. Every float
operation is spelled with an explicit rounding mode (`add.rn`, `fma.rn`, `*.ftz`, and FMA only where intended),
so the outputs do not depend on compiler contraction.

Grid: 4 * (heads + meta) CTAs, cluster 4, 512 threads, one CTA per SM (~226 KB shared memory). Per head cluster:

1. warp 15: TMEM allocation, weight tile TMA (12 tiles of 128 x 128 E4M3, 128-byte swizzle) and scale atoms, their
   scale copy into TMEM (A operand in TMEM at bucket 16);
2. clusters c < rows, warps 0..14: boundary row c: Lamport push of the partial into every rank's exchange (unless the
   producer pushed), poll every rank's partial, residual add, Gemma RMSNorm over the cluster (DSMEM), MXFP8 of the
   normed row into this rank's private hand-off buffer (two parities, sentinel-armed);
3. every CTA: re-arm the other parity, poll its K slice of every row's MXFP8 activation into shared memory, 48
   tcgen05 block-scaled MMAs (M 128 weight rows, N 8/16 rows, K 32), K-split reduce over the cluster (st.async);
4. warps 0..3: per-head Gemma RMSNorm + NeoX RoPE, FP8 Q / index-Q, FP8 K/V and index-K cache inserts;
5. optionally one more cluster writes the step metadata of the layer.

Compile-time parameters: the bucket (MAXT 8 / 16), rows (0 = runtime), the TP world, the per-rank head geometry
(nq, nkv, niq: TP4 16/1/1, TP8 8/1/1), the rotary dim, the page size and the exchange mode.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import dsl_user_op

H = 6144
CL = 4  # cluster = K split of the GEMV = boundary CTAs per row
THREADS = 512  # 16 warps; the boundary math runs on the first 192 threads
BTH = 192
KS = H // CL  # 1536 K columns per CTA
HD = 128
MAX_ROWS = 16
TILES = KS // 128  # 12 k-tiles of 128 K
SMW = (
    TILES * HD * 128
)  # 196608: weight tiles (also the K-split receive buffer at bucket 16)
SMWSF = (
    (TILES * 512 + 1023) // 1024 * 1024
)  # 6144: weight scale atoms (receive buffer at bucket 8)
XTILE = MAX_ROWS * 128  # 2048: 16 activation rows x 128 B, 128-byte swizzle
SMX = TILES * XTILE  # 24576
XATOM = 256  # activation scale atom: rows < 16 of a 128x4 atom
SMXSF = TILES * XATOM + 256  # 3328
SMEM = SMW + SMWSF + SMX + SMXSF  # 230656 (+ the small arrays below)
NACC = 4  # independent accumulators
XQ_BUF = MAX_ROWS * H  # one parity of the private MXFP8 hand-off
XSF_BUF = (H // 128) * XATOM + 512
CTR_LEN = 256  # per-CTA call counters
SYNC_LEN = 64
EMPTY = -(2**31)  # an empty exchange word (0x80000000)
SPIN_CAP = 1 << 31
PEER_TABLE = 16
BND_THREADS = THREADS - 32  # the boundary's named barrier: warps 0..14
TMEM_COLS_FULL = 512


def asl(maxt: int) -> dict:
    """TMEM layout: bucket 8 reads A (weights) straight from shared memory; bucket 16 stages A into TMEM."""
    on = maxt <= 8
    return dict(
        on=on,
        cols=128 if on else 512,
        sfa=32 if on else 64,
        sfb=80 if on else 112,
        slots=maxt // 4 if on else 4,
        ta=128,
    )


# ------------------------------------------------------------------------------------------------ inline PTX helpers
def _asm(ret, args, text, cons, side=True, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm

    return _llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _outs(out, n, cls, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    ty = _T.f32() if cls is cutlass.Float32 else _T.i32()
    return [cls(_llvm.extractvalue(ty, out, [i], loc=loc, ip=ip)) for i in range(n)]


def _struct(n, f32=False):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return _llvm.StructType.get_literal([(_T.f32() if f32 else _T.i32())] * n)


def _i32t():
    from cutlass._mlir.extras import types as _T

    return _T.i32()


def _f32t():
    from cutlass._mlir.extras import types as _T

    return _T.f32()


def _i64t():
    from cutlass._mlir.extras import types as _T

    return _T.i64()


I32, I64, F32 = cutlass.Int32, cutlass.Int64, cutlass.Float32


# ---- float arithmetic with explicit rounding (no contraction by the compiler)
@dsl_user_op
def fop(op, a, b, *, loc=None, ip=None):
    """`op`.f32 a, b, e.g. op = "add.rn", "mul.rn.ftz", "div.full.ftz", "max", "min"."""
    return F32(
        _asm(
            _f32t(),
            [F32(a), F32(b)],
            f"{op}.f32 $0, $1, $2;",
            "=f,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ffma(a, b, c, ftz=False, *, loc=None, ip=None):
    m = ".ftz" if ftz else ""
    return F32(
        _asm(
            _f32t(),
            [F32(a), F32(b), F32(c)],
            f"fma.rn{m}.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f1(op, a, *, loc=None, ip=None):
    """One-operand op: "rsqrt.approx.ftz", "rsqrt.approx", "rcp.approx.ftz", "abs"."""
    return F32(
        _asm(_f32t(), [F32(a)], f"{op}.f32 $0, $1;", "=f,f", side=False, loc=loc, ip=ip)
    )


@dsl_user_op
def fneg(a, *, loc=None, ip=None):
    return F32(
        _asm(_f32t(), [F32(a)], "neg.f32 $0, $1;", "=f,f", side=False, loc=loc, ip=ip)
    )


@dsl_user_op
def bf16_round(x, *, loc=None, ip=None):
    """float(bf16(x)) (RN)."""
    return F32(
        _asm(
            _f32t(),
            [F32(x)],
            "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}",
            "=f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None):
    """bf16(lo) | bf16(hi) << 16 (RN)."""
    return I32(
        _asm(
            _i32t(),
            [F32(lo), F32(hi)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


def bf_lo(w):
    return (w << I32(16)).bitcast(F32)


def bf_hi(w):
    return (w & I32(-65536)).bitcast(F32)


@dsl_user_op
def e4m3x2(lo, hi, *, loc=None, ip=None):
    """E4M3 (RN, satfinite) of lo (byte 0) and hi (byte 1), zero-extended."""
    return I32(
        _asm(
            _i32t(),
            [F32(lo), F32(hi)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.satfinite.e4m3x2.f32 h, $2, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0(x, *, loc=None, ip=None):
    """UE8M0 of x rounded up, satfinite (the low byte of cvt.rp.satfinite.ue8m0x2.f32 with a 0 partner)."""
    return I32(
        _asm(
            _i32t(),
            [F32(x)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 w;\n\tcvt.rp.satfinite.ue8m0x2.f32 h, 0f00000000, $1;\n\t"
            "cvt.u32.u16 w, h;\n\tand.b32 $0, w, 255;\n\t}",
            "=r,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_value(code, *, loc=None, ip=None):
    """The float value of a UE8M0 code (through bf16, as __nv_fp8_e8m0 -> float)."""
    return F32(
        _asm(
            _f32t(),
            [I32(code)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 r;\n\tcvt.u16.u32 h, $1;\n\tand.b16 h, h, 255;\n\t"
            "cvt.rn.bf16x2.ue8m0x2 r, h;\n\tshl.b32 $0, r, 16;\n\t}",
            "=f,r",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def zero_byte_mask(t, *, loc=None, ip=None):
    """0x80 in every byte of t that is zero, else 0 (exact per byte)."""
    return I32(
        _asm(
            _i32t(),
            [I32(t)],
            "{\n\t.reg .b32 z;\n\tand.b32 z, $1, 0x7F7F7F7F;\n\tadd.u32 z, z, 0x7F7F7F7F;\n\t"
            "or.b32 z, z, $1;\n\tnot.b32 z, z;\n\tand.b32 $0, z, 0x80808080;\n\t}",
            "=r,r",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_xor(v, off, *, loc=None, ip=None):
    return F32(
        _asm(
            _f32t(),
            [F32(v), I32(off)],
            "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


# ---- global memory (addresses are Int64 generic addresses of global memory)
@dsl_user_op
def ldg_v4(addr, pred, *, loc=None, ip=None):
    """ld.global.v4.b32, zeros when pred is 0."""
    out = _asm(
        _struct(4),
        [I64(addr), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
        "mov.b32 $3, 0;\n\t@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 4, I32, loc, ip)


@dsl_user_op
def ldg_v2(addr, pred, *, loc=None, ip=None):
    out = _asm(
        _struct(2),
        [I64(addr), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\t"
        "@p ld.global.v2.b32 {$0, $1}, [$2];\n\t}",
        "=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 2, I32, loc, ip)


@dsl_user_op
def ldg_b32(addr, pred, *, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [I64(addr), I32(pred)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t@p ld.global.b32 $0, [$1];\n\t}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_u16(addr, pred, *, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [I64(addr), I32(pred)],
            "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b16 h, 0;\n\t"
            "@p ld.global.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_b64(addr, pred, dflt=-1, *, loc=None, ip=None):
    return I64(
        _asm(
            _i64t(),
            [I64(addr), I32(pred), I64(dflt)],
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\tmov.b64 $0, $3;\n\t@p ld.global.b64 $0, [$1];\n\t}",
            "=l,l,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_nc_b32(addr, *, loc=None, ip=None):
    return I32(
        _asm(_i32t(), [I64(addr)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def ldv_v4(addr, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [I64(addr)],
        "ld.volatile.global.v4.s32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 4, I32, loc, ip)


@dsl_user_op
def ldv_v4_pred(addr, pred, *, loc=None, ip=None):
    """ld.volatile.global.v4 when pred, else four EMPTY words is not wanted: zeros (the caller masks the poll)."""
    out = _asm(
        _struct(4),
        [I64(addr), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
        "mov.b32 $3, 0;\n\t@p ld.volatile.global.v4.s32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 4, I32, loc, ip)


@dsl_user_op
def ldv_b32(addr, *, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [I64(addr)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def stg_v4(addr, w0, w1, w2, w3, pred, *, loc=None, ip=None):
    """st.global.v4.b32 (explicit .global: a generic store to an NVLS multicast address faults)."""
    _asm(
        None,
        [I64(addr), I32(w0), I32(w1), I32(w2), I32(w3), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $5, 0;\n\t@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n\t}",
        "l,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_v2(addr, w0, w1, pred, *, loc=None, ip=None):
    _asm(
        None,
        [I64(addr), I32(w0), I32(w1), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t@p st.global.v2.b32 [$0], {$1, $2};\n\t}",
        "l,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b32(addr, w, pred, *, loc=None, ip=None):
    _asm(
        None,
        [I64(addr), I32(w), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_u8(addr, w, pred, *, loc=None, ip=None):
    _asm(
        None,
        [I64(addr), I32(w), I32(pred)],
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tsetp.ne.b32 p, $2, 0;\n\tcvt.u16.u32 h, $1;\n\t"
        "@p st.global.b8 [$0], h;\n\t}",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


# ---- shared memory (Int32 shared::cta addresses)
@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return I32(
        _asm(_i32t(), [I32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def ldsv_b32(addr, *, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [I32(addr)],
            "ld.volatile.shared.b32 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return F32(
        _asm(_f32t(), [I32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def lds_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        _struct(4, True),
        [I32(addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 4, F32, loc, ip)


@dsl_user_op
def sts_b32(addr, w, *, loc=None, ip=None):
    _asm(None, [I32(addr), I32(w)], "st.shared.b32 [$0], $1;", "r,r", loc=loc, ip=ip)


# ---- synchronization
@dsl_user_op
def ptx(text, *, loc=None, ip=None):
    """A side-effecting PTX statement without operands."""
    _asm(None, [], text, "", loc=loc, ip=ip)


@dsl_user_op
def ptx_r(text, a, *, loc=None, ip=None):
    _asm(None, [I32(a)], text, "r", loc=loc, ip=ip)


@dsl_user_op
def ptx_rr(text, a, b, *, loc=None, ip=None):
    _asm(None, [I32(a), I32(b)], text, "r,r", loc=loc, ip=ip)


@dsl_user_op
def ptx_p(text, pred, *, loc=None, ip=None):
    """A PTX statement executed when pred != 0."""
    _asm(
        None,
        [I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $0, 0;\n\t@p " + text + "\n\t}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ptx_rp(text, a, pred, *, loc=None, ip=None):
    """A PTX statement with one register operand ($0), executed when pred != 0."""
    _asm(
        None,
        [I32(a), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\t@p " + text + "\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ptx_rrp(text, a, b, pred, *, loc=None, ip=None):
    _asm(
        None,
        [I32(a), I32(b), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p " + text + "\n\t}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_b32_pred(addr, w, pred, *, loc=None, ip=None):
    _asm(
        None,
        [I32(addr), I32(w), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.shared.b32 [$0], $1;\n\t}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def poll_empty(addr, w0, w1, w2, w3, pred, *, loc=None, ip=None):
    """Re-poll a 16-byte exchange item (first load w0..w3) while pred and any word is EMPTY; traps after SPIN_CAP."""
    out = _asm(
        _struct(4),
        [I64(addr), I32(w0), I32(w1), I32(w2), I32(w3), I32(pred)],
        "{\n\t.reg .pred p, q;\n\t.reg .b64 sp;\n\tmov.b32 $0, $5;\n\tmov.b32 $1, $6;\n\tmov.b32 $2, $7;\n\t"
        "mov.b32 $3, $8;\n\tsetp.ne.b32 q, $9, 0;\n\t@!q bra PED;\n\tmov.b64 sp, 0;\n\tPEL:\n\t"
        "setp.eq.b32 p, $0, 0x80000000;\n\tsetp.eq.or.b32 p, $1, 0x80000000, p;\n\t"
        "setp.eq.or.b32 p, $2, 0x80000000, p;\n\tsetp.eq.or.b32 p, $3, 0x80000000, p;\n\t@!p bra PED;\n\t"
        "ld.volatile.global.v4.s32 {$0, $1, $2, $3}, [$4];\n\tadd.s64 sp, sp, 1;\n\t"
        "setp.gt.s64 p, sp, 2147483648;\n\t@p trap;\n\tbra PEL;\n\tPED:\n\t}",
        "=r,=r,=r,=r,l,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 4, I32, loc, ip)


@dsl_user_op
def cluster_ctarank(*, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [],
            "mov.u32 $0, %cluster_ctarank;",
            "=r",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mapa(addr, rank, *, loc=None, ip=None):
    return I32(
        _asm(
            _i32t(),
            [I32(addr), I32(rank)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_async(dst, w, bar, pred=1, *, loc=None, ip=None):
    _asm(
        None,
        [I32(dst), I32(w), I32(bar), I32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $3, 0;\n\t"
        "@p st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [$0], $1, [$2];\n\t}",
        "r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait(bar, parity, sem="", *, loc=None, ip=None):
    """Spin on mbarrier.try_wait.parity (sem: "" or ".acquire.cluster")."""
    _asm(
        None,
        [I32(bar), I32(parity)],
        "{\n\t.reg .pred p;\n\tMBW:\n\t"
        f"mbarrier.try_wait.parity{sem}.shared::cta.b64 p, [$0], $1;\n\t@!p bra MBW;\n\t}}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_g2s(dst, src, nbytes, bar, *, loc=None, ip=None):
    _asm(
        None,
        [I32(dst), I64(src), I32(nbytes), I32(bar)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_2d(dst, tmap, c0, c1, bar, *, loc=None, ip=None):
    _asm(
        None,
        [I32(dst), I64(tmap), I32(c0), I32(c1), I32(bar)],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tc_cp(text, taddr, desc, *, loc=None, ip=None):
    _asm(None, [I32(taddr), I64(desc)], text, "r,l", loc=loc, ip=ip)


@dsl_user_op
def tc_mma_ss(d, da, db, idesc, acc, sfa, sfb, *, loc=None, ip=None):
    """Block-scaled MMA, A and B from shared memory descriptors."""
    _asm(
        None,
        [I32(d), I64(da), I64(db), I32(idesc), I32(acc), I32(sfa), I32(sfb)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$5], [$6], p;\n\t}",
        "r,l,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tc_mma_ts(d, ta, db, idesc, acc, sfa, sfb, *, loc=None, ip=None):
    """Block-scaled MMA, A from TMEM, B from a shared memory descriptor."""
    _asm(
        None,
        [I32(d), I32(ta), I64(db), I32(idesc), I32(acc), I32(sfa), I32(sfb)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [$0], [$1], $2, $3, [$5], [$6], p;\n\t}",
        "r,r,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tc_ld8(taddr, *, loc=None, ip=None):
    """tcgen05.ld 32x32b.x8 and its wait in one statement (the values exist only after the wait)."""
    out = _asm(
        _struct(8, True),
        [I32(taddr)],
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];\n\t"
        "tcgen05.wait::ld.sync.aligned;",
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _outs(out, 8, F32, loc, ip)


@dsl_user_op
def tc_ld16(taddr, *, loc=None, ip=None):
    """Two tcgen05.ld 32x32b.x8 (columns c, c + 8) and one wait."""
    out = _asm(
        _struct(16),
        [I32(taddr)],
        "{\n\t.reg .b32 t;\n\tadd.u32 t, $16, 8;\n\t"
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$16];\n\t"
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {$8, $9, $10, $11, $12, $13, $14, $15}, [t];\n\t"
        "tcgen05.wait::ld.sync.aligned;\n\t}",
        ",".join(["=r"] * 16) + ",r",
        loc=loc,
        ip=ip,
    )
    return [v.bitcast(F32) for v in _outs(out, 16, I32, loc, ip)]


@dsl_user_op
def poll_x(pend, g0, g1, g2, g3, gw, s0, s1, s2, s3, sw, *, loc=None, ip=None):
    """The Lamport hand-off poll of one thread: up to four 16-byte activation items (bits 0..3 of pend) and one scale
    word (bit 4), all loaded in one round per spin; an item is accepted once none of its bytes is the sentinel (0x80
    for activations, 0xFF for scales) and is stored to shared memory."""
    loads = []
    checks = []
    for q in range(4):
        loads.append(
            f"and.b32 t, pend, {1 << q};\n\tsetp.ne.b32 p, t, 0;\n\t"
            f"@p ld.volatile.global.v4.s32 {{x{q}0, x{q}1, x{q}2, x{q}3}}, [${1 + q}];\n\t"
        )
        chk = f"and.b32 t, pend, {1 << q};\n\tsetp.eq.b32 p, t, 0;\n\t@p bra SK{q};\n\tmov.b32 acc, 0;\n\t"
        for j in range(4):
            chk += (
                f"xor.b32 t, x{q}{j}, 0x80808080;\n\tand.b32 z, t, 0x7F7F7F7F;\n\tadd.u32 z, z, 0x7F7F7F7F;\n\t"
                "or.b32 z, z, t;\n\tnot.b32 z, z;\n\tand.b32 z, z, 0x80808080;\n\tor.b32 acc, acc, z;\n\t"
            )
        chk += (
            f"setp.ne.b32 p, acc, 0;\n\t@p bra SK{q};\n\t"
            f"st.shared.v4.b32 [${6 + q}], {{x{q}0, x{q}1, x{q}2, x{q}3}};\n\tand.b32 pend, pend, {~(1 << q) & 0xFFFFFFFF};\n\t"
            f"SK{q}:\n\t"
        )
        checks.append(chk)
    text = (
        "{\n\t.reg .pred p;\n\t.reg .b32 pend, t, z, acc, w;\n\t.reg .b64 sp;\n\t"
        + "".join(f".reg .b32 x{q}0, x{q}1, x{q}2, x{q}3;\n\t" for q in range(4))
        + "mov.b32 pend, $0;\n\tmov.b64 sp, 0;\n\tPXL:\n\tsetp.eq.b32 p, pend, 0;\n\t@p bra PXD;\n\t"
        + "".join(loads)
        + "and.b32 t, pend, 16;\n\tsetp.ne.b32 p, t, 0;\n\t@p ld.volatile.global.u32 w, [$5];\n\t"
        + "".join(checks)
        + "and.b32 t, pend, 16;\n\tsetp.eq.b32 p, t, 0;\n\t@p bra SKW;\n\tnot.b32 t, w;\n\t"
        "and.b32 z, t, 0x7F7F7F7F;\n\tadd.u32 z, z, 0x7F7F7F7F;\n\tor.b32 z, z, t;\n\tnot.b32 z, z;\n\t"
        "and.b32 z, z, 0x80808080;\n\tsetp.ne.b32 p, z, 0;\n\t@p bra SKW;\n\tst.shared.b32 [$10], w;\n\t"
        "and.b32 pend, pend, 0xFFFFFFEF;\n\tSKW:\n\t"
        "add.s64 sp, sp, 1;\n\tsetp.gt.s64 p, sp, 2147483648;\n\t@p trap;\n\tbra PXL;\n\tPXD:\n\t}"
    )
    _asm(
        None,
        [
            I32(pend),
            I64(g0),
            I64(g1),
            I64(g2),
            I64(g3),
            I64(gw),
            I32(s0),
            I32(s1),
            I32(s2),
            I32(s3),
            I32(sw),
        ],
        text,
        "r,l,l,l,l,l,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def trap_if(cond, *, loc=None, ip=None):
    _asm(
        None,
        [I32(cond)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $0, 0;\n\t@p trap;\n\t}",
        "r",
        loc=loc,
        ip=ip,
    )


def sdesc(saddr, lbo: int, sbo: int, layout: int):
    """tcgen05 shared memory descriptor of a 16-byte aligned shared address."""
    lo = I64((saddr >> I32(4)) & I32(0x3FFF))
    return lo | I64(
        ((lbo >> 4) & 0x3FFF) << 16
        | ((sbo >> 4) & 0x3FFF) << 32
        | (1 << 46)
        | (layout << 61)
    )


def b2i(b):
    return cutlass.select_(b, I32(1), I32(0))


# ------------------------------------------------------------------------------------------------ the kernel
@cute.kernel
def front_kernel(
    tmw: cutlass.GridConstant[
        cuda.TensorMap
    ],  # weight [N, K] u8, box 128 x 128, 128-byte swizzle
    x: cutlass.Int64,
    x2: cutlass.Int64,
    fc2: cutlass.Int64,
    perm: cutlass.Int64,
    scales: cutlass.Int64,
    residual: cutlass.Int64,
    norm_w: cutlass.Int64,
    out_residual: cutlass.Int64,
    ws_uc: cutlass.Int64,
    flags: cutlass.Int64,
    xq: cutlass.Int64,
    xsf: cutlass.Int64,
    ctr: cutlass.Int64,
    wsf: cutlass.Int64,
    q_w: cutlass.Int64,
    k_w: cutlass.Int64,
    iq_w: cutlass.Int64,
    ik_w: cutlass.Int64,
    cos_sin: cutlass.Int64,
    positions: cutlass.Int64,
    slot: cutlass.Int64,
    islot: cutlass.Int64,
    kv_cache: cutlass.Int64,
    index_cache: cutlass.Int64,
    q_out: cutlass.Int64,
    q_fp8: cutlass.Int64,
    index_q: cutlass.Int64,
    m_qsl: cutlass.Int64,
    m_seq: cutlass.Int64,
    m_bt: cutlass.Int64,
    m_nvalid: cutlass.Int64,
    m_sl: cutlass.Int64,
    m_pt: cutlass.Int64,
    p0: cutlass.Int64,
    p1: cutlass.Int64,
    p2: cutlass.Int64,
    p3: cutlass.Int64,
    p4: cutlass.Int64,
    p5: cutlass.Int64,
    p6: cutlass.Int64,
    p7: cutlass.Int64,
    p8: cutlass.Int64,
    p9: cutlass.Int64,
    p10: cutlass.Int64,
    p11: cutlass.Int64,
    p12: cutlass.Int64,
    p13: cutlass.Int64,
    p14: cutlass.Int64,
    p15: cutlass.Int64,
    rows: cutlass.Int32,
    rank: cutlass.Int32,
    npeers: cutlass.Int32,
    topk: cutlass.Int32,
    scale_fp32: cutlass.Int32,
    single_round: cutlass.Int32,
    eps: cutlass.Float32,
    peps: cutlass.Float32,
    q_inv: cutlass.Float32,
    kv_s_block: cutlass.Int64,
    kv_s_head: cutlass.Int64,
    kv_s_token: cutlass.Int64,
    bt_stride: cutlass.Int64,
    sl_stride_h: cutlass.Int64,
    sl_stride_t: cutlass.Int64,
    pt_stride: cutlass.Int64,
    max_pages: cutlass.Int32,
    num_reqs: cutlass.Int32,
    nd: cutlass.Int32,
    num_sl_heads: cutlass.Int32,
    pt_maxp: cutlass.Int32,
    MAXT: cutlass.Constexpr[int],
    ROWS: cutlass.Constexpr[int],
    WORLD: cutlass.Constexpr[int],
    NQ: cutlass.Constexpr[int],
    NKV: cutlass.Constexpr[int],
    NIQ: cutlass.Constexpr[int],
    ROT: cutlass.Constexpr[int],
    BS: cutlass.Constexpr[int],
    PREPUSHED: cutlass.Constexpr[bool],
    EARLY: cutlass.Constexpr[bool],
    META: cutlass.Constexpr[bool],
):
    L = asl(MAXT)
    HEADS = NQ + 2 * NKV + NIQ + 1
    NN = 8 if (MAXT <= 8 or 0 < ROWS <= 8) else 16  # the MMA's N
    assert HEADS * CL * THREADS >= XQ_BUF // 16, (
        "the re-arm stride covers a parity in one pass"
    )

    sm_arr = cutlass.Array(
        cutlass.Uint8, SMEM, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sm_misc = cutlass.Array(
        cutlass.Int32, 16, space=cutlass.AddressSpace.smem, alignment=16
    )
    sm_bar = cutlass.Array(
        cutlass.Int64, 6, space=cutlass.AddressSpace.smem, alignment=8
    )
    sm = I32(sm_arr.data_ptr().toint())
    s_w = sm
    s_wsf = sm + I32(SMW)
    s_x = s_wsf + I32(SMWSF)
    s_xsf = s_x + I32(SMX)
    misc = I32(sm_misc.data_ptr().toint())
    warp_sq = misc  # f32[8]
    cluster_sq = misc + I32(32)  # f32[4]
    tmem_base_s = misc + I32(48)
    s_call = misc + I32(52)
    bars = I32(sm_bar.data_ptr().toint())
    xbar, wbar, mbar, abar, rbar = (
        bars,
        bars + I32(8),
        bars + I32(16),
        bars + I32(24),
        bars + I32(32),
    )

    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    tid = I32(tid)
    bx = I32(bx)
    lane = tid & I32(31)
    warp = tid >> I32(5)
    crank = cluster_ctarank()
    clus = bx >> I32(2)

    if cutlass.const_expr(META):
        if clus >= I32(HEADS):
            # Step metadata (kernels/reference/front_ref.py step_meta): per decode row
            # t, its request r, position pos = seq_len - q_len + (t - q_begin), n_valid
            # = pos / 128 + 1 (0 when unresolved), sparse_len for every head and the
            # first block-table entries. One warp per row.
            mt = (bx - I32(HEADS * CL)) * I32(THREADS // 32) + warp
            ptx("griddepcontrol.wait;")
            while mt < nd:
                lo = I32(0)
                hi = num_reqs + I32(1)
                while lo < hi:
                    mid = (lo + hi) // I32(2)
                    qv = ldg_nc_b32(m_qsl + I64(mid) * I64(4))
                    lo = cutlass.select_(qv <= mt, mid + I32(1), lo)
                    hi = cutlass.select_(qv <= mt, hi, mid)
                mr = lo - I32(1)
                okr = (mr >= I32(0)) & (mr < num_reqs)
                rc = cutlass.select_(okr, mr, I32(0))
                q_begin = ldg_b32(m_qsl + I64(rc) * I64(4), b2i(okr))
                q_len = ldg_b32(m_qsl + I64(rc + I32(1)) * I64(4), b2i(okr)) - q_begin
                seq_len = ldg_b32(m_seq + I64(rc) * I64(4), b2i(okr))
                mpos = cutlass.select_(okr, seq_len - q_len + (mt - q_begin), I32(-1))
                resolved = okr & (seq_len >= q_len)
                nv = cutlass.select_(resolved, mpos // I32(128) + I32(1), I32(0))
                req = cutlass.select_(resolved, mr, I32(0))
                stg_b32(m_nvalid + I64(mt) * I64(4), nv, b2i(lane == I32(0)))
                n_sel = cutlass.select_(
                    mpos >= I32(0),
                    cutlass.min(mpos // I32(128) + I32(1), I32(16)),
                    I32(0),
                )
                ln = cutlass.select_(
                    n_sel > I32(0),
                    (n_sel - I32(1)) * I32(128) + mpos % I32(128) + I32(1),
                    I32(1),
                )
                if m_sl != I64(0):
                    hh = lane
                    while hh < num_sl_heads:
                        stg_b32(
                            m_sl
                            + (I64(hh) * sl_stride_h + I64(mt) * sl_stride_t) * I64(4),
                            ln,
                            I32(1),
                        )
                        hh = hh + I32(32)
                if (m_pt != I64(0)) & (nv > I32(0)):
                    npt = cutlass.min(nv, cutlass.min(pt_maxp, max_pages))
                    pi = lane
                    while pi < npt:
                        pv = ldg_nc_b32(
                            m_bt + (I64(req) * bt_stride + I64(pi)) * I64(4)
                        )
                        stg_b32(
                            m_pt + (I64(mt) * pt_stride + I64(pi)) * I64(4), pv, I32(1)
                        )
                        pi = pi + I32(32)
                mt = mt + I32(CL * (THREADS // 32))
            ptx("griddepcontrol.launch_dependents;")
    is_head = clus < I32(HEADS)
    if is_head:
        h = clus
        isQ = h < I32(NQ)
        isK = (h >= I32(NQ)) & (h < I32(NQ + NKV))
        isV = (h >= I32(NQ + NKV)) & (h < I32(NQ + 2 * NKV))
        isIQ = (h >= I32(NQ + 2 * NKV)) & (h < I32(NQ + 2 * NKV + NIQ))
        isIK = h == I32(NQ + 2 * NKV + NIQ)
        R = I32(ROWS) if cutlass.const_expr(ROWS > 0) else rows
        # Epilogue inputs that no kernel of the step writes (norm weight, RoPE table, position, slot): loaded now.
        er = crank + I32(4) * warp
        pre = (warp < I32(4)) & (er < R)
        d0 = lane * I32(4)
        nw = cutlass.select_(
            isQ, q_w, cutlass.select_(isK, k_w, cutlass.select_(isIQ, iq_w, ik_w))
        )
        wv = ldg_v2(nw + I64(d0 * I32(2)), b2i(pre & (isV == cutlass.Boolean(False))))
        one = F32(1.0)
        pw = [
            fop("add.rn", bf_lo(wv[0]), one),
            fop("add.rn", bf_hi(wv[0]), one),
            fop("add.rn", bf_lo(wv[1]), one),
            fop("add.rn", bf_hi(wv[1]), one),
        ]
        half = ROT // 2
        in_rope = d0 < I32(ROT)
        er_c = cutlass.select_(pre, er, I32(0))
        pos = ldg_b64(
            positions + I64(er_c) * I64(8),
            b2i(pre & (isV == cutlass.Boolean(False)) & in_rope),
            0,
        )
        ib = cutlass.select_(d0 < I32(half), d0, d0 - I32(half))
        cp = cos_sin + pos * I64(ROT * 2) + I64(ib * I32(2))
        rp = b2i(pre & (isV == cutlass.Boolean(False)) & in_rope)
        cw = ldg_v2(cp, rp)
        sw_ = ldg_v2(cp + I64(half * 2), rp)
        pc = [bf_lo(cw[0]), bf_hi(cw[0]), bf_lo(cw[1]), bf_hi(cw[1])]
        ps = [bf_lo(sw_[0]), bf_hi(sw_[0]), bf_lo(sw_[1]), bf_hi(sw_[1])]
        slot_src = cutlass.select_(isIK, islot, slot)
        pslot = ldg_b64(
            slot_src + I64(er_c) * I64(8), b2i(pre & (isK | isV | isIK)), -1
        )

        bnd = clus < R
        if tid == I32(0):
            ptx_r("mbarrier.init.shared::cta.b64 [$0], 1;", rbar)
            ptx("fence.mbarrier_init.release.cluster;")
        if warp == I32(15):
            # Warp 15: TMEM, barriers, weight TMA, then (while warps 0..14 run the boundary) scales (+ A) into TMEM.
            if bnd:
                ptx("barrier.cluster.arrive.relaxed.aligned;")
            ptx_rr(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], $1;",
                tmem_base_s,
                I32(L["cols"]),
            )
            ptx("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;")
            ptx("tcgen05.fence::before_thread_sync;")
            ptx("bar.warp.sync -1;")
            ptx("tcgen05.fence::after_thread_sync;")
            if lane == I32(0):
                ptx_r("mbarrier.init.shared::cta.b64 [$0], 1;", wbar)
                ptx_r("mbarrier.init.shared::cta.b64 [$0], 1;", mbar)
                ptx_r("mbarrier.init.shared::cta.b64 [$0], 1;", abar)
                ptx("fence.mbarrier_init.release.cluster;")
                tmem0 = lds_b32(tmem_base_s)
                tmap = I64(tmw.get_ptr().toint())
                wsrc = (
                    wsf + I64(h) * I64((H // 128) * 512) + I64(crank * I32(TILES * 512))
                )
                ptx_rr(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
                    wbar,
                    I32(SMW + TILES * 512),
                )
                for t in cutlass.range_constexpr(TILES):
                    tma_2d(
                        s_w + I32(t * HD * 128),
                        tmap,
                        crank * I32(KS) + I32(t * 128),
                        h * I32(HD),
                        wbar,
                    )
                bulk_g2s(s_wsf, wsrc, I32(TILES * 512), wbar)
                mbar_wait(wbar, I32(0))
                ptx("tcgen05.fence::after_thread_sync;")
                for t in cutlass.range_constexpr(TILES):
                    if cutlass.const_expr(not L["on"]):
                        da = sdesc(s_w + I32(t * HD * 128), 16, 1024, 2)
                        for kb in cutlass.range_constexpr(4):
                            tc_cp(
                                "tcgen05.cp.cta_group::1.128x256b [$0], $1;",
                                tmem0 + I32(L["ta"] + (t * 4 + kb) * 8),
                                da + I64(2 * kb),
                            )
                    tc_cp(
                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;",
                        tmem0 + I32(L["sfa"] + 4 * t),
                        sdesc(s_wsf + I32(t * 512), 16, 128, 0),
                    )
                ptx_r(
                    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
                    abar,
                )
                mbar_wait(
                    abar, I32(0)
                )  # sW is free once A is in TMEM: it becomes the K-split receive buffer
            ptx("bar.warp.sync -1;")
            if bnd:
                ptx("barrier.cluster.wait.aligned;")
            else:
                ptx("griddepcontrol.wait;")
            ptx("barrier.cluster.arrive.release.aligned;")  # #2: sW free, rbar live
        else:
            if bnd:
                _boundary(
                    MAXT,
                    WORLD,
                    PREPUSHED,
                    clus,
                    crank,
                    tid,
                    lane,
                    warp,
                    bx,
                    warp_sq,
                    cluster_sq,
                    xbar,
                    x,
                    x2,
                    fc2,
                    perm,
                    scales,
                    topk,
                    scale_fp32,
                    residual,
                    norm_w,
                    eps,
                    out_residual,
                    ws_uc,
                    flags,
                    xq,
                    xsf,
                    ctr,
                    rank,
                    npeers,
                    single_round,
                    [
                        p0,
                        p1,
                        p2,
                        p3,
                        p4,
                        p5,
                        p6,
                        p7,
                        p8,
                        p9,
                        p10,
                        p11,
                        p12,
                        p13,
                        p14,
                        p15,
                    ],
                )
            else:
                ptx("griddepcontrol.wait;")
        # PRE: warps 0..14 re-arm + poll x without waiting for warp 15's weight path
        if warp != I32(15):
            if tid == I32(0):
                sts_b32(s_call, ldg_b32(ctr + I64(bx * I32(4)), I32(1)))
            ptx("bar.sync 1, 480;")
        is15 = warp == I32(15)
        call_g = ldg_b32(ctr + I64(bx * I32(4)), b2i(is15))
        call_s = lds_b32(s_call)
        call0 = cutlass.select_(is15, call_g, call_s)
        # re-arm parity 1 - p: its readers (the previous call) have completed
        ri = bx * I32(THREADS) + tid
        rpar = I64((call0 & I32(1)) ^ I32(1))
        stg_v4(
            xq + rpar * I64(XQ_BUF) + I64(ri) * I64(16),
            -2139062144,
            -2139062144,
            -2139062144,
            -2139062144,
            b2i(ri < I32(XQ_BUF // 16)),
        )
        stg_v4(
            xsf + rpar * I64(XSF_BUF) + I64(ri) * I64(16),
            -1,
            -1,
            -1,
            -1,
            b2i(ri < I32(XSF_BUF // 16)),
        )
        if warp != I32(15):
            ptx("barrier.cluster.arrive.release.aligned;")  # #2
        if cutlass.const_expr(EARLY):
            ptx("griddepcontrol.launch_dependents;")

        # 3. Lamport hand-off: poll this slice's rows from the boundary's parity buffer into the B tiles and scale atoms
        par = I64(call0 & I32(1))
        xqp = xq + par * I64(XQ_BUF)
        xsfp = xsf + par * I64(XSF_BUF)
        PT = THREADS - 32
        KQ = (MAX_ROWS * TILES * 8 + PT - 1) // PT
        assert KQ == 4
        nitems = R * I32(TILES * 8)
        pend = I32(0)
        gq = []
        sq_ = []
        for q in cutlass.range_constexpr(KQ):
            i = tid + I32(q * PT)
            ok = (i < nitems) & (warp != I32(15))
            pend = pend | (b2i(ok) << I32(q))
            r = i // I32(TILES * 8)
            ti = (i // I32(8)) % I32(TILES)
            c = i % I32(8)
            sw = r * I32(128) + ((c ^ (r & I32(7))) * I32(16))
            gq.append(xqp + I64((crank * I32(TILES) + ti) * I32(XTILE) + sw))
            sq_.append(s_x + ti * I32(XTILE) + sw)
        okw = tid < R * I32(TILES)
        pend = pend | (b2i(okw) << I32(4))
        wr = tid // I32(TILES)
        wt = tid % I32(TILES)
        gw = xsfp + I64((crank * I32(TILES) + wt) * I32(XATOM) + wr * I32(16))
        swa = s_xsf + wt * I32(XATOM) + wr * I32(16)
        poll_x(
            pend, gq[0], gq[1], gq[2], gq[3], gw, sq_[0], sq_[1], sq_[2], sq_[3], swa
        )
        ptx("fence.proxy.async.shared::cta;")  # generic smem writes -> tcgen05 reads
        ptx("bar.sync 0;")

        # 4. MMA: M = 128 weight rows, N = NN activation rows (B, smem), K = 32 per instruction, UE8M0 scales in TMEM
        tmem = lds_b32(tmem_base_s)
        if tid == I32(THREADS - 32):
            ptx("tcgen05.fence::after_thread_sync;")
            idesc0 = ((NN >> 3) << 17) | (1 << 23) | (8 << 24)
            if cutlass.const_expr(L["on"]):
                for i in cutlass.range_constexpr(
                    TILES
                ):  # all 12 SFB atoms in one burst ahead of the MMAs
                    tc_cp(
                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;",
                        tmem + I32(L["sfb"] + 4 * i),
                        sdesc(s_xsf + I32(i * XATOM), 16, 128, 0),
                    )
            for i in cutlass.range_constexpr(TILES):
                db = sdesc(s_x + I32(i * XTILE), 16, 1024, 2)
                if cutlass.const_expr(L["on"]):
                    da = sdesc(s_w + I32(i * HD * 128), 16, 1024, 2)
                    for kb in cutlass.range_constexpr(4):
                        j = i * 4 + kb
                        tc_mma_ss(
                            tmem + I32((j % NACC) * NN),
                            da + I64(2 * kb),
                            db + I64(2 * kb),
                            I32(idesc0 | (kb << 4) | (kb << 29)),
                            I32(1 if j >= NACC else 0),
                            tmem + I32(L["sfa"] + 4 * i),
                            tmem + I32(L["sfb"] + 4 * i),
                        )
                else:
                    tc_cp(
                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [$0], $1;",
                        tmem + I32(L["sfb"]),
                        sdesc(s_xsf + I32(i * XATOM), 16, 128, 0),
                    )
                    for kb in cutlass.range_constexpr(4):
                        j = i * 4 + kb
                        tc_mma_ts(
                            tmem + I32((j % NACC) * NN),
                            tmem + I32(L["ta"] + j * 8),
                            db + I64(2 * kb),
                            I32(idesc0 | (kb << 4) | (kb << 29)),
                            I32(1 if j >= NACC else 0),
                            tmem + I32(L["sfa"] + 4 * i),
                            tmem + I32(L["sfb"]),
                        )
            ptx_r(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
                mbar,
            )
        ptx("bar.warp.sync -1;")
        mbar_wait(mbar, I32(0))
        ptx("tcgen05.fence::after_thread_sync;")
        rb = s_wsf if cutlass.const_expr(L["on"]) else s_w
        # K-split reduce: row n belongs to CTA n % 4; every CTA st.async's its fp32
        # partial of that row straight into the owner's receive buffer and the owner's
        # rbar counts the bytes.
        if warp < I32(4):
            ta = ldsv_b32(tmem_base_s) + (warp * I32(32) << I32(16))
            acc = [F32(0.0)] * 16
            for q in cutlass.range_constexpr(NACC):
                if cutlass.const_expr(NN == 16):
                    tv = tc_ld16(ta + I32(q * NN))
                else:
                    tv = tc_ld8(ta + I32(q * NN)) + [F32(0.0)] * 8
                acc = [fop("add.rn", acc[n], tv[n]) for n in range(16)]
            ptx("tcgen05.fence::before_thread_sync;")
            ptx("bar.arrive 2, 160;")  # TMEM reads done (warp 15 deallocates)
            cnt = cutlass.select_(crank < R, (R - crank + I32(3)) // I32(4), I32(0))
            if (tid == I32(0)) & (cnt > I32(0)):
                ptx_rr(
                    "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
                    rbar,
                    I32(CL * HD * 4) * cnt,
                )
            ptx("barrier.cluster.wait.acquire.aligned;")  # #2: every peer's sW is free
            m = warp * I32(32) + lane
            for n in cutlass.range_constexpr(min(16, MAXT)):
                if cutlass.const_expr(ROWS == 0 or n < ROWS):
                    roff = rb + (
                        ((crank * I32(L["slots"]) + I32(n >> 2)) * I32(HD) + m)
                        << I32(2)
                    )
                    st_async(
                        mapa(roff, I32(n & 3)),
                        acc[n].bitcast(I32),
                        mapa(rbar, I32(n & 3)),
                        b2i(I32(n) < R) if cutlass.const_expr(ROWS == 0) else I32(1),
                    )
            if cnt > I32(0):
                mbar_wait(rbar, I32(0))
        if warp == I32(15):
            ptx("bar.sync 2, 160;")
            ptx("tcgen05.fence::after_thread_sync;")
            ptx_rr(
                "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;",
                tmem,
                I32(L["cols"]),
            )

        # 5. K-split reduce (fixed slice order) + producer epilogue: CTA `crank` owns rows crank, crank + 4, ...
        if warp < I32(4):
            if er < R:
                rbuf = rb + (warp * I32(HD) + d0) * I32(4)
                s4 = lds_v4_f32(rbuf)
                for cc in cutlass.range_constexpr(1, CL):
                    rv4 = lds_v4_f32(rbuf + I32(cc * L["slots"] * HD * 4))
                    s4 = [fop("add.rn", s4[z], rv4[z]) for z in range(4)]
                e = [bf16_round(s4[z]) for z in range(4)]
                hq = I64(h * I32(HD) + d0)
                # normAndRopePre (every warp computes it; the V head keeps e)
                ss = ffma(e[0], e[0], F32(0.0))
                for z in cutlass.range_constexpr(1, 4):
                    ss = ffma(e[z], e[z], ss)
                for off in cutlass.range_constexpr(5):
                    ss = fop("add.rn", ss, shfl_xor(ss, I32(16 >> off)))
                rms = f1("rsqrt.approx", ffma(ss, F32(0.0078125), peps))
                en = [fop("mul.rn", fop("mul.rn", rms, e[z]), pw[z]) for z in range(4)]
                part = [shfl_xor(en[z], I32(half // 4)) for z in range(4)]
                first = d0 < I32(half)
                er_ = []
                for z in cutlass.range_constexpr(4):
                    ps_ = fop("mul.rn", ps[z], part[z])
                    rot_v = ffma(en[z], pc[z], cutlass.select_(first, fneg(ps_), ps_))
                    er_.append(cutlass.select_(in_rope, rot_v, en[z]))
                e = [cutlass.select_(isV, e[z], er_[z]) for z in range(4)]
                if isIQ:
                    iq = [
                        fop(
                            "min", fop("max", bf16_round(e[z]), F32(-448.0)), F32(448.0)
                        )
                        for z in range(4)
                    ]
                    stg_b32(
                        index_q
                        + I64(er) * I64(NIQ * HD)
                        + I64((h - I32(NQ + 2 * NKV)) * I32(HD) + d0),
                        e4m3x2(iq[0], iq[1]) | (e4m3x2(iq[2], iq[3]) << I32(16)),
                        I32(1),
                    )
                if isQ:
                    if q_out != I64(0):
                        stg_v2(
                            q_out + (I64(er) * I64(NQ * HD) + hq) * I64(2),
                            bf16x2(e[0], e[1]),
                            bf16x2(e[2], e[3]),
                            I32(1),
                        )
                    sc = [
                        fop(
                            "min",
                            fop(
                                "max",
                                fop("mul.rn", bf16_round(e[z]), q_inv),
                                F32(-448.0),
                            ),
                            F32(448.0),
                        )
                        for z in range(4)
                    ]
                    stg_b32(
                        q_fp8 + I64(er) * I64(NQ * HD) + hq,
                        e4m3x2(sc[0], sc[1]) | (e4m3x2(sc[2], sc[3]) << I32(16)),
                        I32(1),
                    )
                if (isK | isV) & (pslot >= I64(0)):
                    head = cutlass.select_(isK, h - I32(NQ), h - I32(NQ + NKV))
                    blk = pslot // I64(BS)
                    tt = pslot % I64(BS)
                    kv = [e4m3x2(e[z], F32(0.0)) for z in range(4)]
                    word = (
                        kv[0]
                        | (kv[1] << I32(8))
                        | (kv[2] << I32(16))
                        | (kv[3] << I32(24))
                    )
                    stg_b32(
                        kv_cache
                        + blk * kv_s_block
                        + I64(head) * kv_s_head
                        + tt * kv_s_token
                        + I64(cutlass.select_(isK, I32(0), I32(HD)) + d0),
                        word,
                        I32(1),
                    )
                if isIK & (pslot >= I64(0)):
                    ik = [
                        fop("min", fop("max", e[z], F32(-448.0)), F32(448.0))
                        for z in range(4)
                    ]
                    stg_b32(
                        index_cache + pslot * I64(HD) + I64(d0),
                        e4m3x2(ik[0], ik[1]) | (e4m3x2(ik[2], ik[3]) << I32(16)),
                        I32(1),
                    )
        if tid == I32(0):
            stg_b32(ctr + I64(bx * I32(4)), lds_b32(s_call) + I32(1), I32(1))
        if cutlass.const_expr(not EARLY):
            ptx("griddepcontrol.launch_dependents;")


def _boundary(
    MAXT,
    WORLD,
    PREPUSHED,
    t,
    crank,
    tid,
    lane,
    warp,
    bx,
    warp_sq,
    cluster_sq,
    xbar,
    x,
    x2,
    fc2,
    perm,
    scales,
    topk,
    scale_fp32,
    residual,
    norm_w,
    eps,
    out_residual,
    ws_uc,
    flags,
    xq,
    xsf,
    ctr,
    rank,
    npeers,
    single_round,
    peers,
):
    """Boundary row t (FlashInfer one-shot all-reduce + norm rounding), on warps 0..14; 192 threads do the math."""
    t0 = b2i(tid == I32(0))
    ptx_rp("mbarrier.init.shared::cta.b64 [$0], 1;", xbar, t0)
    ptx_p("fence.mbarrier_init.release.cluster;", t0)
    ptx("barrier.cluster.arrive.relaxed.aligned;")
    c0 = crank * I32(KS) + tid * I32(8)
    active = tid < I32(BTH)
    act = b2i(active)
    c0c = cutlass.select_(active, c0, I32(0))
    half_words = MAXT * WORLD * (H // 2)
    fl = t * I32(CL) + crank
    if not PREPUSHED:
        ptx("griddepcontrol.wait;")
    parity = ldv_b32(flags + I64(fl * I32(4))) & I32(1)
    call = ldg_b32(ctr + I64(bx * I32(4)), I32(1))
    xqo = xq + I64(call & I32(1)) * I64(XQ_BUF)
    xsfo = xsf + I64(call & I32(1)) * I64(XSF_BUF)
    row = I64(t) * I64(H) + I64(c0c)
    rraw = ldg_v4(residual + row * I64(2), act)
    wraw = ldg_v4(norm_w + I64(c0c) * I64(2), act)
    if not PREPUSHED:
        has_fin = fc2 != I64(0)
        has_x2 = x2 != I64(0)
        # finalize: x = sum_k scales[t, k] * fc2[perm[t, k]] (+ x2), fp32, rounded once
        f = [F32(0.0)] * 8
        for k in range(8):
            kin = active & has_fin & (I32(k) < topk)
            e_ = t * topk + I32(k)
            prow = ldg_b32(perm + I64(e_) * I64(4), b2i(kin))
            sc32 = ldg_b32(scales + I64(e_) * I64(4), b2i(kin & (scale_fp32 != I32(0))))
            sc16 = ldg_u16(scales + I64(e_) * I64(2), b2i(kin & (scale_fp32 == I32(0))))
            sc = cutlass.select_(
                scale_fp32 != I32(0), sc32.bitcast(F32), (sc16 << I32(16)).bitcast(F32)
            )
            valid = kin & (prow >= I32(0))
            rv = ldg_v4(fc2 + (I64(prow) * I64(H) + I64(c0c)) * I64(2), b2i(valid))
            for q in range(4):
                f[2 * q] = cutlass.select_(
                    valid, ffma(bf_lo(rv[q]), sc, f[2 * q]), f[2 * q]
                )
                f[2 * q + 1] = cutlass.select_(
                    valid, ffma(bf_hi(rv[q]), sc, f[2 * q + 1]), f[2 * q + 1]
                )
        xv2 = ldg_v4(x2 + row * I64(2), b2i(active & has_x2))
        fin_w = []
        for q in range(4):
            a0 = cutlass.select_(
                has_x2, fop("add.rn", bf_lo(xv2[q]), f[2 * q]), f[2 * q]
            )
            a1 = cutlass.select_(
                has_x2, fop("add.rn", bf_hi(xv2[q]), f[2 * q + 1]), f[2 * q + 1]
            )
            fin_w.append(bf16x2(a0, a1))
        xv = ldg_v4(x + row * I64(2), b2i(active & (has_fin == cutlass.Boolean(False))))
        x_w = []
        for q in range(4):
            s0 = bf16x2(
                fop("add.rn", bf_lo(xv[q]), bf_lo(xv2[q])),
                fop("add.rn", bf_hi(xv[q]), bf_hi(xv2[q])),
            )
            x_w.append(cutlass.select_(has_x2, s0, xv[q]))
        raw = [cutlass.select_(has_fin, fin_w[q], x_w[q]) for q in range(4)]
        # canonicalise -0 halves to +0 (0x80000000 marks an empty exchange word)
        val = []
        for q in range(4):
            w = raw[q]
            w = cutlass.select_((w & I32(0xFFFF)) == I32(0x8000), w & I32(-65536), w)
            w = cutlass.select_(
                (w & I32(-65536)) == I32(-2147483648), w & I32(0xFFFF), w
            )
            val.append(w)
        slot = (
            I64(parity) * I64(half_words)
            + (I64(t) * I64(WORLD) + I64(rank)) * I64(H // 2)
            + I64(c0c // I32(2))
        )
        for r in range(PEER_TABLE):
            if r < WORLD:
                stg_v4(
                    peers[r] + slot * I64(4),
                    val[0],
                    val[1],
                    val[2],
                    val[3],
                    b2i(active & (I32(r) < npeers)),
                )
    # poll every rank's partial of this row
    p0 = ws_uc + (
        I64(parity) * I64(half_words)
        + I64(t) * I64(WORLD * (H // 2))
        + I64(c0c // I32(2))
    ) * I64(4)
    E = I32(EMPTY)
    first = [ldv_v4_pred(p0 + I64(r * (H // 2) * 4), act) for r in range(WORLD)]
    s = [F32(0.0)] * 8
    for r in range(WORLD):
        a = p0 + I64(r * (H // 2) * 4)
        ww = poll_empty(a, first[r][0], first[r][1], first[r][2], first[r][3], act)
        stg_v4(a, E, E, E, E, act)
        for q in range(4):
            if r == 0:
                s[2 * q] = bf_lo(ww[q])
                s[2 * q + 1] = bf_hi(ww[q])
            else:
                s[2 * q] = fop("add.rn.ftz", s[2 * q], bf_lo(ww[q]))
                s[2 * q + 1] = fop("add.rn.ftz", s[2 * q + 1], bf_hi(ww[q]))
    sq = F32(0.0)
    upd = []
    for q in range(4):
        rx, ry = bf_lo(rraw[q]), bf_hi(rraw[q])
        u1 = bf16x2(
            fop("add.rn.ftz", s[2 * q], rx), fop("add.rn.ftz", s[2 * q + 1], ry)
        )
        u2 = bf16x2(
            fop("add.rn.ftz", bf16_round(s[2 * q]), rx),
            fop("add.rn.ftz", bf16_round(s[2 * q + 1]), ry),
        )
        u = cutlass.select_(single_round != I32(0), u1, u2)
        upd.append(u)
        s[2 * q] = bf_lo(u)
        s[2 * q + 1] = bf_hi(u)
        sq = ffma(s[2 * q + 1], s[2 * q + 1], ffma(s[2 * q], s[2 * q], sq, True), True)
    stg_v4(out_residual + row * I64(2), upd[0], upd[1], upd[2], upd[3], act)
    for off in range(5):
        sq = fop("add.rn.ftz", sq, shfl_xor(sq, I32(16 >> off)))
    sts_b32_pred(
        warp_sq + (warp & I32(7)) * I32(4),
        sq.bitcast(I32),
        b2i((lane == I32(0)) & active),
    )
    ptx("bar.sync 1, 480;")
    stg_b32(
        flags + I64(fl * I32(4)), parity ^ I32(1), t0
    )  # after the barrier: every boundary warp read the parity
    vv = cutlass.select_(
        tid < I32(BTH // 32), lds_f32(warp_sq + (tid & I32(7)) * I32(4)), F32(0.0)
    )
    for off in range(5):
        vv = fop("add.rn.ftz", vv, shfl_xor(vv, I32(16 >> off)))
    ptx("barrier.cluster.wait.aligned;")
    ptx_rrp(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", xbar, I32(CL * 4), t0
    )
    for c in range(CL):
        st_async(
            mapa(cluster_sq + crank * I32(4), I32(c)),
            vv.bitcast(I32),
            mapa(xbar, I32(c)),
            t0,
        )
    mbar_wait(xbar, I32(0), ".acquire.cluster")
    total = F32(0.0)
    for c in range(CL):
        total = fop("add.rn.ftz", total, lds_f32(cluster_sq + I32(4 * c)))
    inv = f1(
        "rsqrt.approx.ftz",
        fop("add.rn.ftz", fop("div.full.ftz", total, F32(float(H))), eps),
    )
    for q in range(4):
        wx = fop("add.rn.ftz", F32(1.0), bf_lo(wraw[q]))
        wy = fop("add.rn.ftz", F32(1.0), bf_hi(wraw[q]))
        n = bf16x2(
            fop("mul.rn.ftz", fop("mul.rn.ftz", s[2 * q], inv), wx),
            fop("mul.rn.ftz", fop("mul.rn.ftz", s[2 * q + 1], inv), wy),
        )
        s[2 * q] = bf_lo(n)
        s[2 * q + 1] = bf_hi(n)
    # MXFP8 of the normed row (trtllm mxfp8_quantize order)
    amax = F32(0.0)
    for z in range(8):
        amax = fop("max", amax, f1("abs", s[z]))
    amax = fop("max", amax, shfl_xor(amax, I32(1)))
    amax = fop("max", amax, shfl_xor(amax, I32(2)))
    code = ue8m0(fop("mul.rn", amax, f1("rcp.approx.ftz", F32(448.0))))
    out_scale = cutlass.select_(
        amax != F32(0.0), f1("rcp.approx.ftz", ue8m0_value(code)), F32(0.0)
    )
    packed = []
    for hh in range(2):
        lo = e4m3x2(
            fop("mul.rn", s[4 * hh], out_scale), fop("mul.rn", s[4 * hh + 1], out_scale)
        )
        hi = e4m3x2(
            fop("mul.rn", s[4 * hh + 2], out_scale),
            fop("mul.rn", s[4 * hh + 3], out_scale),
        )
        pk = lo | (hi << I32(16))
        pk = pk & (
            zero_byte_mask(pk ^ I32(-2139062144)) ^ I32(-1)
        )  # -0 (0x80) is the sentinel: store +0
        packed.append(pk)
    tile = c0c // I32(128)
    ch = (c0c % I32(128)) // I32(16)
    xo = xqo + I64(
        (tile * I32(MAX_ROWS) + t) * I32(128)
        + ((ch ^ (t & I32(7))) * I32(16))
        + (c0c % I32(16))
    )
    stg_v2(xo, packed[0], packed[1], act)
    stg_u8(
        xsfo
        + I64((c0c // I32(128)) * I32(XATOM) + t * I32(16) + (c0c // I32(32)) % I32(4)),
        code,
        b2i(active & ((tid & I32(3)) == I32(0))),
    )
    ptx("bar.sync 1, 480;")


@cute.jit
def front_launch(
    w_addr: cutlass.Int64,
    x: cutlass.Int64,
    x2: cutlass.Int64,
    fc2: cutlass.Int64,
    perm: cutlass.Int64,
    scales: cutlass.Int64,
    residual: cutlass.Int64,
    norm_w: cutlass.Int64,
    out_residual: cutlass.Int64,
    ws_uc: cutlass.Int64,
    flags: cutlass.Int64,
    xq: cutlass.Int64,
    xsf: cutlass.Int64,
    ctr: cutlass.Int64,
    wsf: cutlass.Int64,
    q_w: cutlass.Int64,
    k_w: cutlass.Int64,
    iq_w: cutlass.Int64,
    ik_w: cutlass.Int64,
    cos_sin: cutlass.Int64,
    positions: cutlass.Int64,
    slot: cutlass.Int64,
    islot: cutlass.Int64,
    kv_cache: cutlass.Int64,
    index_cache: cutlass.Int64,
    q_out: cutlass.Int64,
    q_fp8: cutlass.Int64,
    index_q: cutlass.Int64,
    m_qsl: cutlass.Int64,
    m_seq: cutlass.Int64,
    m_bt: cutlass.Int64,
    m_nvalid: cutlass.Int64,
    m_sl: cutlass.Int64,
    m_pt: cutlass.Int64,
    p0: cutlass.Int64,
    p1: cutlass.Int64,
    p2: cutlass.Int64,
    p3: cutlass.Int64,
    p4: cutlass.Int64,
    p5: cutlass.Int64,
    p6: cutlass.Int64,
    p7: cutlass.Int64,
    p8: cutlass.Int64,
    p9: cutlass.Int64,
    p10: cutlass.Int64,
    p11: cutlass.Int64,
    p12: cutlass.Int64,
    p13: cutlass.Int64,
    p14: cutlass.Int64,
    p15: cutlass.Int64,
    rows: cutlass.Int32,
    rank: cutlass.Int32,
    npeers: cutlass.Int32,
    topk: cutlass.Int32,
    scale_fp32: cutlass.Int32,
    single_round: cutlass.Int32,
    eps: cutlass.Float32,
    peps: cutlass.Float32,
    q_inv: cutlass.Float32,
    kv_s_block: cutlass.Int64,
    kv_s_head: cutlass.Int64,
    kv_s_token: cutlass.Int64,
    bt_stride: cutlass.Int64,
    sl_stride_h: cutlass.Int64,
    sl_stride_t: cutlass.Int64,
    pt_stride: cutlass.Int64,
    max_pages: cutlass.Int32,
    num_reqs: cutlass.Int32,
    nd: cutlass.Int32,
    num_sl_heads: cutlass.Int32,
    pt_maxp: cutlass.Int32,
    stream: cuda_driver.CUstream,
    MAXT: cutlass.Constexpr[int],
    ROWS: cutlass.Constexpr[int],
    WORLD: cutlass.Constexpr[int],
    NQ: cutlass.Constexpr[int],
    NKV: cutlass.Constexpr[int],
    NIQ: cutlass.Constexpr[int],
    ROT: cutlass.Constexpr[int],
    BS: cutlass.Constexpr[int],
    PREPUSHED: cutlass.Constexpr[bool],
    EARLY: cutlass.Constexpr[bool],
    META: cutlass.Constexpr[bool],
    PDL: cutlass.Constexpr[bool],
):
    heads = NQ + 2 * NKV + NIQ + 1
    tmw = cuda.create_tensor_map_tiled(
        global_address=w_addr,
        dtype=cutlass.Uint8,
        global_dims=[H, heads * HD],
        global_strides=[H // 16],
        box_dims=[128, 128],
        swizzle=cuda.TensorMapSwizzle.s128b,
        l2_promotion=cuda.TensorMapL2Promotion.l2_256b,
    )
    front_kernel(
        tmw,
        x,
        x2,
        fc2,
        perm,
        scales,
        residual,
        norm_w,
        out_residual,
        ws_uc,
        flags,
        xq,
        xsf,
        ctr,
        wsf,
        q_w,
        k_w,
        iq_w,
        ik_w,
        cos_sin,
        positions,
        slot,
        islot,
        kv_cache,
        index_cache,
        q_out,
        q_fp8,
        index_q,
        m_qsl,
        m_seq,
        m_bt,
        m_nvalid,
        m_sl,
        m_pt,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        rows,
        rank,
        npeers,
        topk,
        scale_fp32,
        single_round,
        eps,
        peps,
        q_inv,
        kv_s_block,
        kv_s_head,
        kv_s_token,
        bt_stride,
        sl_stride_h,
        sl_stride_t,
        pt_stride,
        max_pages,
        num_reqs,
        nd,
        num_sl_heads,
        pt_maxp,
        MAXT,
        ROWS,
        WORLD,
        NQ,
        NKV,
        NIQ,
        ROT,
        BS,
        PREPUSHED,
        EARLY,
        META,
    ).launch(
        grid=((heads + (1 if META else 0)) * CL, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(CL, 1, 1),
        stream=stream,
        use_pdl=PDL,
    )
