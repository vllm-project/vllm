# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102
"""CuTe DSL kernels of the post family (the attention -> MoE boundary of a sparse layer).

Every arithmetic op is spelled as an explicit PTX instruction (ftz / rn / approx variants, no contraction), so the
outputs do not depend on compiler contraction. Cross-rank protocols: the prepushed two-parity exchange with the
0x80000000 empty sentinel, tagged rx router partials, tagged route lines, PDL trigger points.

Kernels:

- `post_route_kernel` (allreduce_norm stage): one cluster of CL CTAs per row. Row warps land the W ranks' bf16
  partials from this rank's exchange (poll + re-arm), add the residual, Gemma RMSNorm via a cluster st.async
  exchange of the row's sum of squares (and the |normed| argmax candidate), then the mode-Q per-token NVFP4 tail
  (and optional MXFP8). One extra routing warp (crank 0) polls the o_proj router CTAs' tagged partial logits,
  scales them by 1/rms and publishes the MiniMax2 top-4 as tagged route lines (optionally the fp32 logits).
- `mxfp8_rows_kernel`: MXFP8 (UE8M0, 128x4-swizzled scales) of bf16 rows.

All buffers are passed as raw 64-bit device addresses (the host checks shapes, dtypes and alignment).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, dsl_user_op
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T

H = 6144
E = 128
TOPK = 4
LINE_U64 = 16  # one 128-byte route line per (replica, row)
RCTR_STRIDE = 32  # int32 words per row counter
SF_K_TILES = H // 32 // 4  # 48: k-tiles of the 128x4-swizzled MXFP8 scales
K_EMPTY = 0x80000000  # empty exchange word
SPIN_CAP = 1 << 31  # bounded spin of the exchange poll (then trap)
HALF_NS = 384  # delay of the second rx poll round after the first (ns)
ROUTE_SCALE_BITS = 0x40000000  # 2.0f
SUM_EPS_BITS = 0x1E3CE508  # 1e-20f
INV_2688_BITS = 0x39C30C31  # fp32(1.0 / 2688.0) (double quotient rounded to float)
RQ_TH = 256  # threads per mxfp8_rows CTA


# ---------------------------------------------------------------------------------------------------------------------
# inline PTX helpers (explicit instructions)
# ---------------------------------------------------------------------------------------------------------------------
def _asm(ret, args, text, cons, side=False, loc=None, ip=None):
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


def _unpack(out, ty, n, cls, loc, ip):
    return [cls(_llvm.extractvalue(ty, out, [i], loc=loc, ip=ip)) for i in range(n)]


def _fop2(op):
    @dsl_user_op
    def f(a, b, *, loc=None, ip=None):
        return Float32(
            _asm(
                _T.f32(),
                [Float32(a), Float32(b)],
                f"{op} $0, $1, $2;",
                "=f,f,f",
                loc=loc,
                ip=ip,
            )
        )

    return f


def _fop1(op):
    @dsl_user_op
    def f(a, *, loc=None, ip=None):
        return Float32(
            _asm(_T.f32(), [Float32(a)], f"{op} $0, $1;", "=f,f", loc=loc, ip=ip)
        )

    return f


# ftz spellings
add_ftz = _fop2("add.rn.ftz.f32")
mul_ftz = _fop2("mul.rn.ftz.f32")
div_full_ftz = _fop2("div.full.ftz.f32")
rsqrt_ftz = _fop1("rsqrt.approx.ftz.f32")
rcp_ftz = _fop1("rcp.approx.ftz.f32")
# route_warp (IEEE, no ftz) spellings
add_rn = _fop2("add.rn.f32")
sub_rn = _fop2("sub.rn.f32")
mul_rn = _fop2("mul.rn.f32")
div_rn = _fop2("div.rn.f32")
div_full = _fop2("div.full.f32")
fmax = _fop2("max.f32")
fabs = _fop1("abs.f32")


@dsl_user_op
def fma_ftz(a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            _T.f32(),
            [Float32(a), Float32(b), Float32(c)],
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            _T.f32(),
            [Float32(a), Float32(b), Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fimm(bits: int, *, loc=None, ip=None):
    """An fp32 constant with exactly these bits (kept out of the optimizer's constant folding)."""
    return Float32(
        _asm(_T.f32(), [], f"mov.b32 $0, 0f{bits:08X};", "=f", loc=loc, ip=ip)
    )


@dsl_user_op
def bf16r(a, *, loc=None, ip=None):
    """__bfloat162float(__float2bfloat16_rn(a))"""
    return Float32(
        _asm(
            _T.f32(),
            [Float32(a)],
            "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2(a, b, *, loc=None, ip=None):
    """__floats2bfloat162_rn(a, b) as a word (a in the low half)."""
    return Int32(
        _asm(
            _T.i32(),
            [Float32(a), Float32(b)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lo_f(w, *, loc=None, ip=None):
    """The low bf16 of a word as fp32."""
    return Float32(
        _asm(_T.f32(), [Int32(w)], "shl.b32 $0, $1, 16;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def hi_f(w, *, loc=None, ip=None):
    """The high bf16 of a word as fp32."""
    return Float32(
        _asm(
            _T.f32(), [Int32(w)], "and.b32 $0, $1, 0xFFFF0000;", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def fbits(x, *, loc=None, ip=None):
    return Int32(
        _asm(_T.i32(), [Float32(x)], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip)
    )


@dsl_user_op
def ibits(x, *, loc=None, ip=None):
    return Float32(
        _asm(_T.f32(), [Int32(x)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def twiddle(x, *, loc=None, ip=None):
    """Order-preserving uint32 key of an fp32 (as Int32 bits; compare with ugt)."""
    return Int32(
        _asm(
            _T.i32(),
            [Float32(x)],
            "{\n\t.reg .b32 b, n;\n\t.reg .pred p;\n\tmov.b32 b, $1;\n\tsetp.lt.s32 p, b, 0;\n\t"
            "not.b32 n, b;\n\tor.b32 b, b, 0x80000000;\n\tselp.b32 $0, n, b, p;\n\t}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3_bits(x, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Float32(x)],
            "{\n\t.reg .b16 t;\n\t.reg .b32 u;\n\tcvt.rn.satfinite.e4m3x2.f32 t, 0f00000000, $1;\n\t"
            "cvt.u32.u16 u, t;\n\tand.b32 $0, u, 255;\n\t}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3_to_f32(b, *, loc=None, ip=None):
    return Float32(
        _asm(
            _T.f32(),
            [Int32(b)],
            "{\n\t.reg .b16 t, l, u;\n\t.reg .b32 h;\n\tcvt.u16.u32 t, $1;\n\tcvt.rn.f16x2.e4m3x2 h, t;\n\t"
            "mov.b32 {l, u}, h;\n\tcvt.f32.f16 $0, l;\n\t}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e2m1x2(hi, lo, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Float32(hi), Float32(lo)],
            "{\n\t.reg .b8 t;\n\t.reg .b32 u;\n\tcvt.rn.satfinite.e2m1x2.f32 t, $1, $2;\n\t"
            "cvt.u32.u8 u, t;\n\tand.b32 $0, u, 255;\n\t}",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None):
    """Four fp32 to E4M3 (RN, satfinite), v0 in the low byte (two __nv_fp8x2_e4m3 halves)."""
    return Int32(
        _asm(
            _T.i32(),
            [Float32(v) for v in (v0, v1, v2, v3)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_rp(x, *, loc=None, ip=None):
    """__nv_cvt_float_to_e8m0(x, __NV_SATFINITE, cudaRoundPosInf)"""
    return Int32(
        _asm(
            _T.i32(),
            [Float32(x)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 w;\n\tcvt.rp.satfinite.ue8m0x2.f32 h, 0f00000000, $1;\n\t"
            "cvt.u32.u16 w, h;\n\tand.b32 $0, w, 255;\n\t}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_to_f32(c, *, loc=None, ip=None):
    """float(__nv_fp8_e8m0): 2^(c - 127); c = 0 -> 2^-127 (0x00400000), c = 255 -> NaN."""
    return Float32(
        _asm(
            _T.f32(),
            [Int32(c)],
            "{\n\t.reg .b32 b;\n\t.reg .pred z, n;\n\tshl.b32 b, $1, 23;\n\tsetp.eq.u32 z, $1, 0;\n\t"
            "setp.eq.u32 n, $1, 255;\n\tselp.b32 b, 0x00400000, b, z;\n\tselp.b32 $0, 0x7FFFFFFF, b, n;\n\t}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ugt(a, b, *, loc=None, ip=None):
    """A > b as uint32 -> Int32 0 / 1."""
    return Int32(
        _asm(
            _T.i32(),
            [Int32(a), Int32(b)],
            "{\n\t.reg .pred p;\n\tsetp.gt.u32 p, $1, $2;\n\tselp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


def better(h1, l1, h0, l0):
    """(h1, l1) > (h0, l0) lexicographically, uint32 -> Boolean."""
    return (
        ugt(h1, h0) | (cutlass.select_(h1 == h0, Int32(1), Int32(0)) & ugt(l1, l0))
    ) != 0


# shuffles / warp votes
@dsl_user_op
def shfl_bfly_f(x, off: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            _T.f32(),
            [Float32(x)],
            f"shfl.sync.bfly.b32 $0, $1, {off}, 0x1f, 0xffffffff;",
            "=f,f",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx_f(x, src, *, loc=None, ip=None):
    return Float32(
        _asm(
            _T.f32(),
            [Float32(x), Int32(src)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx_i(x, src, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int32(x), Int32(src)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def redux_max(v, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int32(v)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ballot(pred, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int32(pred)],
            "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $1, 0;\n\tvote.sync.ballot.b32 $0, p, 0xffffffff;\n\t}",
            "=r,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ctz(x, *, loc=None, ip=None):
    """__ffs(x) - 1 for x != 0."""
    return Int32(
        _asm(
            _T.i32(),
            [Int32(x)],
            "{\n\t.reg .b32 r;\n\tbrev.b32 r, $1;\n\tclz.b32 $0, r;\n\t}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


# memory
@dsl_user_op
def ld_v4(addr, pred, *, loc=None, ip=None):
    """16 bytes (zero when pred == 0), ordered where written (the post's residual may alias its residual_out)."""
    out = _asm(
        _llvm.StructType.get_literal([_T.i32()] * 4),
        [Int64(addr), Int32(pred)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\tmov.b32 $3, 0;\n\t"
        "setp.ne.u32 p, $5, 0;\n\t@p ld.global.v4.u32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=r,=r,=r,=r,l,r",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, _T.i32(), 4, Int32, loc, ip)


@dsl_user_op
def ld_nc_v4(addr, *, loc=None, ip=None):
    """Read-only 16 bytes (ld.global.nc), ordered where written."""
    out = _asm(
        _llvm.StructType.get_literal([_T.i32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, _T.i32(), 4, Int32, loc, ip)


@dsl_user_op
def ld_v4f_if(addr, pred, *, loc=None, ip=None):
    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [Int64(addr), Int32(pred)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0f00000000;\n\tmov.b32 $1, 0f00000000;\n\tmov.b32 $2, 0f00000000;\n\t"
        "mov.b32 $3, 0f00000000;\n\tsetp.ne.u32 p, $5, 0;\n\t@p ld.global.v4.f32 {$0, $1, $2, $3}, [$4];\n\t}",
        "=f,=f,=f,=f,l,r",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, _T.f32(), 4, Float32, loc, ip)


@dsl_user_op
def ld_s32_if(addr, pred, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int64(addr), Int32(pred)],
            "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tsetp.ne.u32 p, $2, 0;\n\t@p ld.global.s32 $0, [$1];\n\t}",
            "=r,l,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_s32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int64(addr)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_volatile_s32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int64(addr)],
            "ld.volatile.global.s32 $0, [$1];",
            "=r,l",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_volatile_v2u64(addr, *, loc=None, ip=None):
    out = _asm(
        _llvm.StructType.get_literal([_T.i64()] * 2),
        [Int64(addr)],
        "ld.volatile.global.v2.u64 {$0, $1}, [$2];",
        "=l,=l,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, _T.i64(), 2, Int64, loc, ip)


@dsl_user_op
def st_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Int32(w0), Int32(w1), Int32(w2), Int32(w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_v4f(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_b32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Int32(v)],
        "st.global.b32 [$0], $1;",
        "l,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Float32(v)],
        "st.global.f32 [$0], $1;",
        "l,f",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_v2_if(addr, w0, w1, pred, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Int32(w0), Int32(w1), Int32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $3, 0;\n\t@p st.global.v2.b32 [$0], {$1, $2};\n\t}",
        "l,r,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_b32_if(addr, v, pred, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Int32(v), Int32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}",
        "l,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_u8_if(addr, v, pred, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Int32(v), Int32(pred)],
        "{\n\t.reg .b16 h;\n\t.reg .pred p;\n\tsetp.ne.u32 p, $2, 0;\n\tcvt.u16.u32 h, $1;\n\t"
        "@p st.global.u8 [$0], h;\n\t}",
        "l,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_relaxed_v2u64_if(addr, a, b, pred, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Int64(a), Int64(b), Int32(pred)],
        "{\n\t.reg .pred p;\n\tsetp.ne.u32 p, $3, 0;\n\t@p st.relaxed.gpu.global.v2.u64 [$0], {$1, $2};\n\t}",
        "l,l,l,r",
        side=True,
        loc=loc,
        ip=ip,
    )


def _b(cond):
    """Boolean -> Int32 0/1 (a store predicate)."""
    return cutlass.select_(cond, Int32(1), Int32(0))


@dsl_user_op
def ld_shared_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            _T.f32(),
            [Int32(addr)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr), Float32(v)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        side=True,
        loc=loc,
        ip=ip,
    )


# control / sync
def _stmt(text, cons="~{memory}"):
    @dsl_user_op
    def f(*, loc=None, ip=None):
        _asm(None, [], text, cons, side=True, loc=loc, ip=ip)

    return f


trap = _stmt("trap;", "")
threadfence = _stmt("membar.gl;")
syncthreads = _stmt("bar.sync 0;")
syncwarp = _stmt("bar.warp.sync 0xffffffff;")
gdc_launch = _stmt("griddepcontrol.launch_dependents;")
gdc_wait = _stmt("griddepcontrol.wait;")
cluster_arrive_relaxed = _stmt("barrier.cluster.arrive.relaxed.aligned;")
cluster_wait = _stmt("barrier.cluster.wait.aligned;")


@dsl_user_op
def bar_named(nthreads: int, *, loc=None, ip=None):
    _asm(None, [], f"bar.sync 1, {nthreads};", "~{memory}", side=True, loc=loc, ip=ip)


@dsl_user_op
def globaltimer(*, loc=None, ip=None):
    return Int64(
        _asm(_T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", side=True, loc=loc, ip=ip)
    )


@dsl_user_op
def cluster_ctarank(*, loc=None, ip=None):
    return Int32(
        _asm(_T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc=loc, ip=ip)
    )


@dsl_user_op
def mbar_init_cluster(bar, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar)],
        "mbarrier.init.shared::cta.b64 [$0], 1;\n\tfence.mbarrier_init.release.cluster;",
        "r,~{memory}",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_expect_tx(bar, nbytes: int, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};",
        "r,~{memory}",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait_cluster0(bar, *, loc=None, ip=None):
    """Spin until phase 0 of the CTA's barrier completes (acquire, cluster scope)."""
    _asm(
        None,
        [Int32(bar)],
        "{\n\t.reg .pred q;\n\tMBW:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 q, [$0], 0;\n\t"
        "@!q bra MBW;\n\t}",
        "r,~{memory}",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mapa(addr, peer, *, loc=None, ip=None):
    return Int32(
        _asm(
            _T.i32(),
            [Int32(addr), Int32(peer)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_async_v4(dst, w0, w1, w2, w3, bar, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(dst), Float32(w0), Float32(w1), Float32(w2), Float32(w3), Int32(bar)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,f,f,f,f,r,~{memory}",
        side=True,
        loc=loc,
        ip=ip,
    )


def _poll_exchange(world: int):
    """The post's exchange poll: re-load every rank's 16 bytes at p0 + r * 12288 that still holds the empty sentinel
    in any word, until none does (bounded spin, then trap). Returns the 4 * world words."""
    n = 4 * world

    @dsl_user_op
    def poll(p0, *, loc=None, ip=None):
        lds, chk = [], []
        for r in range(world):
            o = ", ".join(f"${4 * r + i}" for i in range(4))
            lds.append(
                f"\tand.b32 m, pend, {1 << r};\n\tsetp.ne.u32 q, m, 0;\n\t"
                f"@q ld.volatile.global.v4.s32 {{{o}}}, [${n}+{r * (H // 2) * 4}];\n"
            )
            ws = [f"${4 * r + i}" for i in range(4)]
            chk.append(
                f"\tand.b32 m, pend, {1 << r};\n\tsetp.ne.u32 q, m, 0;\n\t"
                f"setp.ne.and.s32 q, {ws[0]}, {K_EMPTY - (1 << 32)}, q;\n\t"
                f"setp.ne.and.s32 q, {ws[1]}, {K_EMPTY - (1 << 32)}, q;\n\t"
                f"setp.ne.and.s32 q, {ws[2]}, {K_EMPTY - (1 << 32)}, q;\n\t"
                f"setp.ne.and.s32 q, {ws[3]}, {K_EMPTY - (1 << 32)}, q;\n\t"
                f"@q and.b32 pend, pend, {~(1 << r) & 0xFFFFFFFF};\n"
            )
        text = (
            "{\n\t.reg .pred q;\n\t.reg .b32 pend, m;\n\t.reg .b64 spins;\n\t"
            f"mov.b32 pend, {(1 << world) - 1};\n\tmov.u64 spins, 0;\n\t"
            + "".join(f"\tmov.b32 ${i}, 0;\n" for i in range(n))
            + "PXL:\n"
            + "".join(lds)
            + "".join(chk)
            + f"\tadd.u64 spins, spins, 1;\n\tsetp.gt.s64 q, spins, {SPIN_CAP};\n\t@q trap;\n\t"
            "setp.ne.u32 q, pend, 0;\n\t@q bra PXL;\n\t}"
        )
        out = _asm(
            _llvm.StructType.get_literal([_T.i32()] * n),
            [Int64(p0)],
            text,
            ",".join(["=r"] * n) + ",l",
            side=True,
            loc=loc,
            ip=ip,
        )
        return _unpack(out, _T.i32(), n, Int32, loc, ip)

    return poll


# ---------------------------------------------------------------------------------------------------------------------
# shared math (boundary device functions)
# ---------------------------------------------------------------------------------------------------------------------
def bf16_words_to_f32(w):
    """4 words (8 bf16) -> 8 fp32 (low half first)."""
    out = []
    for q in range(4):
        out += [lo_f(w[q]), hi_f(w[q])]
    return out


def inv_rms(total, eps):
    return rsqrt_ftz(add_ftz(div_full_ftz(total, fimm(0x45C00000)), eps))  # 6144.0f


def gemma_w(gw, q):
    """(1 + w) of the gamma pair q (fp32, ftz add)."""
    one = fimm(0x3F800000)
    return add_ftz(one, lo_f(gw[q])), add_ftz(one, hi_f(gw[q]))


def mxfp8_8(s, active, t, c0, out_q, out_sf, tid):
    """lb::mxfp8_8: 4 lanes per 32-column block, warp-synchronous (stores predicated on active)."""
    amax = Float32(0.0)
    for e in range(8):
        amax = fmax(amax, fabs(s[e]))
    amax = cutlass.select_(active != 0, amax, Float32(0.0))
    amax = fmax(amax, shfl_bfly_f(amax, 1))
    amax = fmax(amax, shfl_bfly_f(amax, 2))
    code = ue8m0_rp(mul_rn(amax, rcp_ftz(fimm(0x43E00000))))  # 448
    osc = cutlass.select_(
        amax != Float32(0.0), rcp_ftz(ue8m0_to_f32(code)), Float32(0.0)
    )
    p0 = e4m3x4(
        mul_rn(s[0], osc), mul_rn(s[1], osc), mul_rn(s[2], osc), mul_rn(s[3], osc)
    )
    p1 = e4m3x4(
        mul_rn(s[4], osc), mul_rn(s[5], osc), mul_rn(s[6], osc), mul_rn(s[7], osc)
    )
    st_v2_if(out_q + Int64(t) * Int64(H) + Int64(c0), p0, p1, active)
    kb = c0 // 32
    off = (
        (t // 128) * (SF_K_TILES * 512)
        + (kb // 4) * 512
        + (t % 32) * 16
        + ((t % 128) // 32) * 4
        + kb % 4
    )
    st_u8_if(out_sf + Int64(off), code, active & _b((tid & 3) == 0))


def x4_quant8(xn, active, t, c0, inv_of, xq, xsf, tid, sfb):
    """Shared tail of the NVFP4 16-column block quantizers: codes of xn * inv, sf byte from even lanes (stores
    predicated on active)."""
    w = Int32(0)
    for z in range(4):
        w = w | (
            e2m1x2(mul_rn(xn[2 * z + 1], inv_of), mul_rn(xn[2 * z], inv_of)) << (8 * z)
        )
    st_b32_if(xq + Int64(t) * Int64(H // 2) + Int64(c0 // 2), w, active)
    st_u8_if(
        xsf + Int64(t) * Int64(H // 16) + Int64(c0 // 16),
        sfb,
        active & _b((tid & 1) == 0),
    )


@dsl_user_op
def tanh_engine(x, *, loc=None, ip=None):
    """Tanh as an explicit instruction sequence, so the routing keys are reproducible bit for bit (cute.math.tanh
    lowers to a sequence with different bits)."""
    return Float32(
        _asm(
            _T.f32(),
            [Float32(x)],
            "{\n\t.reg .f32 a, b, c, d, e, f, g, h, i, j, k, l, m, n, o;\n\t.reg .pred p1, p2;\n\t"
            "abs.f32 a, $1;\n\tmul.f32 b, a, 0f4038AA3B;\n\tex2.approx.ftz.f32 c, b;\n\t"
            "add.f32 d, c, 0f3F800000;\n\trcp.approx.ftz.f32 e, d;\n\t"
            "fma.rn.f32 f, e, 0fC0000000, 0f3F800000;\n\tsetp.ge.f32 p1, a, 0f41102CB4;\n\t"
            "selp.f32 g, 0f3F800000, f, p1;\n\tcopysign.f32 h, $1, g;\n\tmul.f32 i, $1, $1;\n\t"
            "fma.rn.f32 j, i, 0f3C80F082, 0fBD563CAE;\n\tfma.rn.f32 k, j, i, 0f3E085941;\n\t"
            "fma.rn.f32 l, k, i, 0fBEAAA9ED;\n\tfma.rn.f32 m, l, i, 0f00000000;\n\t"
            "fma.rn.f32 n, m, $1, $1;\n\tsetp.ge.f32 p2, a, 0f3F19999A;\n\tselp.f32 $0, h, n, p2;\n\t}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


def rx_ld(row, lane, nb, buf):
    """rx_ld: the lane's two 16-byte loads (experts 4 lane .. +3) of every rx block (unconditional)."""
    for i in range(nb):
        a = row + Int64(i * E * 8) + Int64(lane) * Int64(32)
        v0 = ld_volatile_v2u64(a)
        v1 = ld_volatile_v2u64(a + Int64(16))
        buf[4 * i + 0] = v0[0]
        buf[4 * i + 1] = v0[1]
        buf[4 * i + 2] = v1[0]
        buf[4 * i + 3] = v1[1]


def rx_take(tag, nb, buf, pending, acc):
    """rx_take: accept blocks whose four words carry tag (returns the new pending mask)."""
    tg = Int64(tag) & Int64(0xFFFFFFFF)
    for i in range(nb):
        ok = (pending >> i) & 1
        for j in range(4):
            ok = ok & cutlass.select_(
                ((buf[4 * i + j] >> 32) & Int64(0xFFFFFFFF)) == tg, Int32(1), Int32(0)
            )
        for j in range(4):
            acc[4 * i + j] = cutlass.select_(
                ok != 0,
                ibits(Int32(buf[4 * i + j] & Int64(0xFFFFFFFF))),
                acc[4 * i + j],
            )
        pending = cutlass.select_(ok != 0, pending & Int32(~(1 << i)), pending)
    return pending


def rx_stale(nb, buf, pending):
    st = Int32(0)
    for i in range(nb - 1, -1, -1):
        st = cutlass.select_(
            ((pending >> i) & 1) != 0, Int32((buf[4 * i] >> 32) & Int64(0xFFFFFFFF)), st
        )
    return st


@cute.jit
def rx_poll_sum(
    row: Int64,
    lane: Int32,
    tag: Int32,
    deadline: Int64,
    nb: cutlass.Constexpr[int],
    rx2: cutlass.Constexpr[bool],
    S,
):
    """rx_poll_sum: poll the lane's rx words (exact tag match, globaltimer deadline), then S_e in rank order.
    Returns (late, stale)."""
    pending = Int32((1 << nb) - 1)
    acc = cute.make_rmem_tensor(cute.make_layout((4 * nb,)), Float32)
    a = cute.make_rmem_tensor(cute.make_layout((4 * nb,)), Int64)
    b = cute.make_rmem_tensor(cute.make_layout((4 * nb,)), Int64)
    for i in cutlass.range_constexpr(4 * nb):
        acc[i] = Float32(0.0)
    late = Int32(0)
    stale = Int32(0)
    t0 = Int64(0)
    go = Int32(1)
    rx_ld(row, lane, nb, a)
    if cutlass.const_expr(rx2):
        ti = globaltimer()
        while globaltimer() - ti < Int64(HALF_NS):
            pass
        while go != 0:
            rx_ld(row, lane, nb, b)
            pending = rx_take(tag, nb, a, pending, acc)
            if pending == 0:
                go = Int32(0)
            else:
                now = globaltimer()
                if t0 == Int64(0):
                    t0 = now
                elif now - t0 > deadline:
                    late = Int32(1)
                    stale = rx_stale(nb, a, pending)
                    go = Int32(0)
                if go != 0:
                    rx_ld(row, lane, nb, a)
                    pending = rx_take(tag, nb, b, pending, acc)
                    if pending == 0:
                        go = Int32(0)
                    else:
                        now = globaltimer()
                        if now - t0 > deadline:
                            late = Int32(1)
                            stale = rx_stale(nb, b, pending)
                            go = Int32(0)
    else:
        while go != 0:
            pending = rx_take(tag, nb, a, pending, acc)
            if pending == 0:
                go = Int32(0)
            else:
                now = globaltimer()
                if t0 == Int64(0):
                    t0 = now
                elif now - t0 > deadline:
                    late = Int32(1)
                    stale = rx_stale(nb, a, pending)
                    go = Int32(0)
                if go != 0:
                    rx_ld(row, lane, nb, a)
    if (
        late != 0
    ):  # keep whatever the last read held (rx_finish's garbage-tolerant values)
        for i in cutlass.range_constexpr(nb):
            if ((pending >> i) & 1) != 0:
                for j in cutlass.range_constexpr(4):
                    acc[4 * i + j] = ibits(Int32(a[4 * i + j] & Int64(0xFFFFFFFF)))
    for e in cutlass.range_constexpr(4):
        x = acc[e]
        for i in cutlass.range_constexpr(1, nb):
            x = add_rn(x, acc[4 * i + e])
        S[e] = x
    return late, stale


def _cex(a, b):
    """Compare-exchange on (hi, lo, sg) triples: a gets the larger."""
    s = better(b[0], b[1], a[0], a[1])
    x = tuple(cutlass.select_(s, b[i], a[i]) for i in range(3))
    y = tuple(cutlass.select_(s, a[i], b[i]) for i in range(3))
    return x, y


def route_pub(route, repl, rmaxt, t, lane, tag, inv, S, bb):
    """route_pub: top-4 of the hoisted logits, then lane l stores words 2 (l & 1), +1 of replica l >> 1."""
    c = []
    for i in range(4):
        x = mul_rn(inv, S[i])
        kk = add_rn(
            fma_rn(
                fimm(0x3F000000),
                tanh_engine(mul_rn(fimm(0x3F000000), x)),
                fimm(0x3F000000),
            ),
            bb[i],
        )
        c.append((twiddle(kk), Int32(65535) - (4 * lane + i), sub_rn(kk, bb[i])))
    c[0], c[1] = _cex(c[0], c[1])
    c[2], c[3] = _cex(c[2], c[3])
    c[0], c[2] = _cex(c[0], c[2])
    c[1], c[3] = _cex(c[1], c[3])
    c[1], c[2] = _cex(c[1], c[2])
    ek, sk = [], []
    zero = (Int32(0), Int32(0), Float32(0.0))
    for k in range(TOPK):
        mh = redux_max(c[0][0])
        ml = redux_max(cutlass.select_(c[0][0] == mh, c[0][1], Int32(0)))
        ek.append(Int32(65535) - ml)
        sk.append(shfl_idx_f(c[0][2], ek[k] >> 2))
        if k < TOPK - 1:  # the last shift is dead
            sh = c[0][1] == ml
            nc = [
                tuple(cutlass.select_(sh, c[j + 1][i], c[j][i]) for i in range(3))
                for j in range(3)
            ]
            nc.append(tuple(cutlass.select_(sh, zero[i], c[3][i]) for i in range(3)))
            c = nc
    total = add_rn(add_rn(sk[0], sk[2]), add_rn(sk[1], sk[3]))
    den = add_rn(total, fimm(SUM_EPS_BITS))
    h = (lane & 1) != 0
    s0 = cutlass.select_(h, sk[2], sk[0])
    s1 = cutlass.select_(h, sk[3], sk[1])
    e0 = cutlass.select_(h, ek[2], ek[0])
    e1 = cutlass.select_(h, ek[3], ek[1])
    w0 = bf16r(div_rn(mul_rn(s0, fimm(ROUTE_SCALE_BITS)), den))
    w1 = bf16r(div_rn(mul_rn(s1, fimm(ROUTE_SCALE_BITS)), den))
    hi = Int64(tag) << 32
    d0 = hi | (Int64(e0) << 16) | ((Int64(fbits(w0)) >> 16) & Int64(0xFFFF))
    d1 = hi | (Int64(e1) << 16) | ((Int64(fbits(w1)) >> 16) & Int64(0xFFFF))
    r0 = lane >> 1
    hoff = Int64((lane & 1) * 2)
    st_relaxed_v2u64_if(
        route
        + ((Int64(r0) * Int64(rmaxt) + Int64(t)) * Int64(LINE_U64) + hoff) * Int64(8),
        d0,
        d1,
        _b(r0 < repl),
    )
    st_relaxed_v2u64_if(
        route
        + ((Int64(r0 + 16) * Int64(rmaxt) + Int64(t)) * Int64(LINE_U64) + hoff)
        * Int64(8),
        d0,
        d1,
        _b(r0 + 16 < repl),
    )


# ---------------------------------------------------------------------------------------------------------------------
# post_route: prepushed AR + residual + Gemma RMSNorm + mode-Q NVFP4 tail (+ MXFP8) +
# routing warp
# ---------------------------------------------------------------------------------------------------------------------
@cute.kernel
def post_route_kernel(
    ex: Int64,
    flags: Int64,
    maxt: Int32,
    residual: Int64,
    gamma: Int64,
    out_res: Int64,
    out_norm: Int64,
    out_q: Int64,
    out_sf: Int64,
    eps: Float32,
    rows: Int32,
    rx: Int64,
    rxc: Int64,
    bias: Int64,
    route: Int64,
    rctr: Int64,
    repl: Int32,
    rmaxt: Int32,
    deadline: Int64,
    hang_dbg: Int32,
    x4_q: Int64,
    x4_sf: Int64,
    x4_pts: Int64,
    lg_out: Int64,
    CL: cutlass.Constexpr[int],
    TH_A: cutlass.Constexpr[int],
    RX2: cutlass.Constexpr[bool],
    W: cutlass.Constexpr[int],
    MX: cutlass.Constexpr[bool],
    LG: cutlass.Constexpr[bool],
):
    NWA = TH_A // 32
    XW = 4  # exchange words per CTA: sq, argmax |s w|, its s, its w
    cols = H // CL
    tid, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    lane = tid & 31
    sm_wsq = cutlass.Array(Float32, NWA, space=cutlass.AddressSpace.smem, alignment=16)
    sm_csq = cutlass.Array(
        Float32, CL * XW, space=cutlass.AddressSpace.smem, alignment=16
    )
    sm_wc = cutlass.Array(
        Float32, 3 * NWA, space=cutlass.AddressSpace.smem, alignment=16
    )
    sm_bar = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    a_wsq = Int32(sm_wsq.data_ptr().toint())
    a_csq = Int32(sm_csq.data_ptr().toint())
    a_wc = Int32(sm_wc.data_ptr().toint())
    a_bar = Int32(sm_bar.data_ptr().toint())
    if tid == 0:
        mbar_init_cluster(a_bar)
    cluster_arrive_relaxed()
    crank = cluster_ctarank()
    t = bx // CL
    rwarp = tid >= TH_A
    c0 = crank * cols + tid * 8
    active = cutlass.select_(
        (tid < TH_A) & (tid * 8 < cols) & (t < rows), Int32(1), Int32(0)
    )
    router = cutlass.select_(rwarp & (crank == 0) & (t < rows), Int32(1), Int32(0))
    fidx = t * 4 + crank // (CL // 4)
    gw = ld_v4(gamma + Int64(c0) * Int64(2), active)
    rr = ld_v4(residual + (Int64(t) * Int64(H) + Int64(c0)) * Int64(2), active)
    bb = ld_v4f_if(bias + Int64(16) * Int64(lane), router)
    lane0r = cutlass.select_((router != 0) & (lane == 0), Int32(1), Int32(0))
    prev_r = ld_s32_if(rctr + Int64(t * RCTR_STRIDE) * Int64(4), lane0r)
    prev_x = ld_s32_if(rxc + Int64(t * RCTR_STRIDE) * Int64(4), lane0r)
    gdc_launch()
    s = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
    S = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
    for e in cutlass.range_constexpr(8):
        s[e] = Float32(0.0)
    for e in cutlass.range_constexpr(4):
        S[e] = Float32(0.0)
    parity = Int32(0)
    if tid < TH_A:
        parity = ld_volatile_s32(flags + Int64(fidx) * Int64(4)) & 1
        half = Int64(maxt) * Int64(W * (H // 2))
        p0 = ex + (
            Int64(parity) * half + Int64(t) * Int64(W * (H // 2)) + Int64(c0 // 2)
        ) * Int64(4)
        sq = Float32(0.0)
        if active != 0:
            v = _poll_exchange(W)(p0)
            for r in cutlass.range_constexpr(W):
                k = K_EMPTY - (1 << 32)
                st_v4(
                    p0 + Int64(r * (H // 2) * 4), Int32(k), Int32(k), Int32(k), Int32(k)
                )
                for q in cutlass.range_constexpr(4):
                    lo = lo_f(v[4 * r + q])
                    hi = hi_f(v[4 * r + q])
                    if cutlass.const_expr(r == 0):
                        s[2 * q] = lo
                        s[2 * q + 1] = hi
                    else:
                        s[2 * q] = add_ftz(s[2 * q], lo)
                        s[2 * q + 1] = add_ftz(s[2 * q + 1], hi)
            upd = []
            for q in cutlass.range_constexpr(4):  # residual_add8
                u = bf16x2(
                    add_ftz(bf16r(s[2 * q]), lo_f(rr[q])),
                    add_ftz(bf16r(s[2 * q + 1]), hi_f(rr[q])),
                )
                upd.append(u)
                ux = lo_f(u)
                uy = hi_f(u)
                s[2 * q] = ux
                s[2 * q + 1] = uy
                sq = fma_ftz(uy, uy, fma_ftz(ux, ux, sq))
            st_v4(
                out_res + (Int64(t) * Int64(H) + Int64(c0)) * Int64(2),
                upd[0],
                upd[1],
                upd[2],
                upd[3],
            )
        for o in cutlass.range_constexpr(5):
            sq = add_ftz(sq, shfl_bfly_f(sq, 16 >> o))
        if lane == 0:
            st_shared_f32(a_wsq + (tid // 32) * 4, sq)
        # amax_cand8: argmax |s * w| of the thread's 8 columns (bp = -1: none)
        bp = Float32(-1.0)
        bs = Float32(0.0)
        bw = Float32(0.0)
        for q in cutlass.range_constexpr(4):
            wx, wy = gemma_w(gw, q)
            pa = fabs(mul_rn(s[2 * q], wx))
            pb = fabs(mul_rn(s[2 * q + 1], wy))
            ta = pa > bp
            bs = cutlass.select_(ta, s[2 * q], bs)
            bw = cutlass.select_(ta, wx, bw)
            bp = cutlass.select_(ta, pa, bp)
            tb = pb > bp
            bs = cutlass.select_(tb, s[2 * q + 1], bs)
            bw = cutlass.select_(tb, wy, bw)
            bp = cutlass.select_(tb, pb, bp)
        bp = cutlass.select_(active != 0, bp, Float32(-1.0))
        bs = cutlass.select_(active != 0, bs, Float32(0.0))
        bw = cutlass.select_(active != 0, bw, Float32(0.0))
        for o in cutlass.range_constexpr(5):
            p_ = shfl_bfly_f(bp, 16 >> o)
            a_ = shfl_bfly_f(bs, 16 >> o)
            w_ = shfl_bfly_f(bw, 16 >> o)
            tk = p_ > bp
            bp = cutlass.select_(tk, p_, bp)
            bs = cutlass.select_(tk, a_, bs)
            bw = cutlass.select_(tk, w_, bw)
        if lane == 0:
            wi = tid // 32
            st_shared_f32(a_wc + wi * 12, bp)
            st_shared_f32(a_wc + wi * 12 + 4, bs)
            st_shared_f32(a_wc + wi * 12 + 8, bw)
        bar_named(TH_A)
    elif router != 0:
        rtag = shfl_idx_i(prev_x, 0) + 1
        # the slot's tag-parity half (rx maxt == the exchange bucket maxt:
        # PostRouteStage checks it)
        rxrow = rx + (Int64(rtag & 1) * Int64(maxt) + Int64(t)) * Int64(W * E * 8)
        late, stale = rx_poll_sum(rxrow, lane, rtag, deadline, W, RX2, S)
        lm = ballot(late)
        if (lm != 0) & (hang_dbg == 0):
            trap()
        xr = rxc + Int64(t * RCTR_STRIDE) * Int64(4)
        if lane == 0:
            st_b32(xr, rtag)
            if lm != 0:
                st_b32(xr + Int64(4), ld_s32(xr + Int64(4)) + 1)
                st_b32(xr + Int64(8), rtag)
        sl = shfl_idx_i(stale, cutlass.select_(lm != 0, ctz(lm), Int32(0)))
        if (lane == 0) & (lm != 0):
            st_b32(xr + Int64(12), sl)
    cluster_wait()
    if tid == 0:
        tot = ld_shared_f32(a_wsq)
        for w in cutlass.range_constexpr(1, NWA):
            tot = add_ftz(tot, ld_shared_f32(a_wsq + w * 4))
        cbp = ld_shared_f32(a_wc)
        cbs = ld_shared_f32(a_wc + 4)
        cbw = ld_shared_f32(a_wc + 8)
        for w in cutlass.range_constexpr(1, NWA):
            cp_ = ld_shared_f32(a_wc + w * 12)
            ctk = cp_ > cbp
            cbs = cutlass.select_(ctk, ld_shared_f32(a_wc + w * 12 + 4), cbs)
            cbw = cutlass.select_(ctk, ld_shared_f32(a_wc + w * 12 + 8), cbw)
            cbp = cutlass.select_(ctk, cp_, cbp)
        d = a_csq + crank * (XW * 4)
        mbar_expect_tx(a_bar, CL * 4 * XW)
        for c in cutlass.range_constexpr(CL):
            st_async_v4(mapa(d, c), tot, cbp, cbs, cbw, mapa(a_bar, c))
    if (tid < TH_A) | (router != 0):
        mbar_wait_cluster0(a_bar)
        if (tid == 0) & (crank % (CL // 4) == 0) & (t < rows):
            st_b32(flags + Int64(fidx) * Int64(4), parity ^ 1)
        total = ld_shared_f32(a_csq)
        for c in cutlass.range_constexpr(1, CL):
            total = add_ftz(total, ld_shared_f32(a_csq + c * (XW * 4)))
        inv = inv_rms(total, eps)
        if rwarp:
            tag = shfl_idx_i(prev_r, 0) + 1
            tag = cutlass.select_(
                (tag == 0) | (tag < 0), Int32(1), tag
            )  # pack_route's 0 < tag < 2^31
            if cutlass.const_expr(LG):
                st_v4f(
                    lg_out + (Int64(t) * Int64(E) + Int64(4 * lane)) * Int64(4),
                    mul_rn(inv, S[0]),
                    mul_rn(inv, S[1]),
                    mul_rn(inv, S[2]),
                    mul_rn(inv, S[3]),
                )
            route_pub(
                route, repl, rmaxt, t, lane, tag, inv, [S[0], S[1], S[2], S[3]], bb
            )
            if lane == 0:
                st_b32(rctr + Int64(t * RCTR_STRIDE) * Int64(4), tag)
        else:
            if active != 0:  # norm8
                o = []
                for q in cutlass.range_constexpr(4):
                    wx, wy = gemma_w(gw, q)
                    n = bf16x2(
                        mul_ftz(mul_ftz(s[2 * q], inv), wx),
                        mul_ftz(mul_ftz(s[2 * q + 1], inv), wy),
                    )
                    o.append(n)
                    s[2 * q] = lo_f(n)
                    s[2 * q + 1] = hi_f(n)
                st_v4(
                    out_norm + (Int64(t) * Int64(H) + Int64(c0)) * Int64(2),
                    o[0],
                    o[1],
                    o[2],
                    o[3],
                )
            # X4 == 2: row amax over the CTAs' candidates of |bf16((s * inv) * w)|, then x4_8_qat
            ra = Float32(0.0)
            for c in cutlass.range_constexpr(CL):
                cb = a_csq + c * (XW * 4)
                cand = fabs(
                    bf16r(
                        mul_ftz(
                            mul_ftz(ld_shared_f32(cb + 8), inv), ld_shared_f32(cb + 12)
                        )
                    )
                )
                ra = cutlass.select_(
                    ld_shared_f32(cb + 4) >= Float32(0.0), fmax(ra, cand), ra
                )
            ra = fmax(ra, fimm(0x2B8CBCCC))  # 1e-12f
            enc = div_full(fimm(0x45280000), ra)  # 2688
            xn = []
            am = Float32(0.0)
            for e in cutlass.range_constexpr(8):
                xe = cutlass.select_(
                    active != 0, bf16r(mul_rn(s[e], enc)), Float32(0.0)
                )
                xn.append(xe)
                am = fmax(am, fabs(xe))
            bm = fmax(am, shfl_bfly_f(am, 1))
            sfb = e4m3_bits(mul_rn(bm, rcp_ftz(fimm(0x40C00000))))  # 6
            sff = e4m3_to_f32(sfb)
            iv = cutlass.select_(sff == Float32(0.0), Float32(0.0), rcp_ftz(sff))
            x4_quant8(xn, active, t, c0, iv, x4_q, x4_sf, tid, sfb)
            if (tid == 0) & (crank == 0) & (t < rows):
                st_f32(
                    x4_pts + Int64(t) * Int64(4), bf16r(div_full(ra, fimm(0x45280000)))
                )
            if cutlass.const_expr(MX):
                mxfp8_8([s[e] for e in range(8)], active, t, c0, out_q, out_sf, tid)


@cute.jit
def post_route(
    ex: Int64,
    flags: Int64,
    maxt: Int32,
    residual: Int64,
    gamma: Int64,
    out_res: Int64,
    out_norm: Int64,
    out_q: Int64,
    out_sf: Int64,
    eps: Float32,
    rows: Int32,
    nrow_blocks: Int32,
    rx: Int64,
    rxc: Int64,
    bias: Int64,
    route: Int64,
    rctr: Int64,
    repl: Int32,
    rmaxt: Int32,
    deadline: Int64,
    hang_dbg: Int32,
    x4_q: Int64,
    x4_sf: Int64,
    x4_pts: Int64,
    lg_out: Int64,
    CL: cutlass.Constexpr[int],
    TH_A: cutlass.Constexpr[int],
    RX2: cutlass.Constexpr[bool],
    W: cutlass.Constexpr[int],
    MX: cutlass.Constexpr[bool],
    LG: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
):
    post_route_kernel(
        ex,
        flags,
        maxt,
        residual,
        gamma,
        out_res,
        out_norm,
        out_q,
        out_sf,
        eps,
        rows,
        rx,
        rxc,
        bias,
        route,
        rctr,
        repl,
        rmaxt,
        deadline,
        hang_dbg,
        x4_q,
        x4_sf,
        x4_pts,
        lg_out,
        CL,
        TH_A,
        RX2,
        W,
        MX,
        LG,
    ).launch(
        grid=(nrow_blocks * CL, 1, 1),
        block=(TH_A + 32, 1, 1),
        cluster=(CL, 1, 1),
        stream=stream,
        use_pdl=True,
        min_blocks_per_mp=1,
    )


# ---------------------------------------------------------------------------------------------------------------------
# mxfp8_rows (the wide front's MXFP8 input)
# ---------------------------------------------------------------------------------------------------------------------
@cute.kernel
def mxfp8_rows_kernel(x: Int64, q: Int64, sf: Int64):
    tid, _, _ = cute.arch.thread_idx()
    t, _, _ = cute.arch.block_idx()
    vs = []
    for ps in cutlass.range_constexpr(
        H // (8 * RQ_TH)
    ):  # every load first (non-coherent loads)
        vs.append(
            ld_nc_v4(
                x + (Int64(t) * Int64(H) + Int64((ps * RQ_TH + tid) * 8)) * Int64(2)
            )
        )
    for ps in cutlass.range_constexpr(H // (8 * RQ_TH)):
        mxfp8_8(
            bf16_words_to_f32(vs[ps]), Int32(1), t, (ps * RQ_TH + tid) * 8, q, sf, tid
        )


@cute.jit
def mxfp8_rows(
    x: Int64, q: Int64, sf: Int64, rows: Int32, stream: cuda_driver.CUstream
):
    mxfp8_rows_kernel(x, q, sf).launch(
        grid=(rows, 1, 1), block=(RQ_TH, 1, 1), stream=stream
    )
