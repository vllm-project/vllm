# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PTX building blocks of the low-batch attention kernels.

NVFP4 pages decode as the stock kernels read them: an E2M1 code times its
E4M3 block scale, rounded to E4M3 (saturating). The product is exact in F16
(at most 6 significant bits, 2^-10 to 2688), so ``cvt`` to F16, ``mul`` and
``cvt`` to E4M3 give that rounding exactly.

The warp MMA is ``mma.sync`` m16n8k16 on F16 operands (E4M3 values are exact
in F16; SM100 runs E4M3 ``mma.sync`` through F16 anyway) with FP32
accumulation. With ``g = lane // 4`` and ``t = lane % 4``, an A register
holds row ``g`` (or ``g + 8``), columns ``2t, 2t + 1`` (or ``8 + 2t ..``); a
B register holds column ``g``, rows ``2t, 2t + 1`` (or ``8 + 2t ..``); the
accumulator holds rows ``g`` and ``g + 8``, columns ``2t`` and ``2t + 1``.
"""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(result, operands, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(value, loc, ip):
    return value.ir_value(loc=loc, ip=ip)


def _words(result, count, loc, ip, ty=None):
    ty = ty or T.i32()
    return [llvm.extractvalue(ty, result, [i], loc=loc, ip=ip) for i in range(count)]


def _struct(count, ty=None):
    return llvm.StructType.get_literal([ty or T.i32()] * count)


# Loads of data earlier kernels wrote are side-effecting, so they stay after
# the grid dependency wait.
@dsl_user_op
def ld_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    text = "ld.global.b32 $0, [$1];"
    return Uint32(
        _asm(T.i32(), [_v(Int64(address), loc, ip)], text, "=r,l", loc, ip, True)
    )


@dsl_user_op
def ld_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    text = "ld.global.b32 $0, [$1];"
    return Int32(
        _asm(T.i32(), [_v(Int64(address), loc, ip)], text, "=r,l", loc, ip, True)
    )


@dsl_user_op
def ld_u16(address: Int64, *, loc=None, ip=None) -> Uint32:
    text = "ld.global.u16 $0, [$1];"
    return Uint32(
        _asm(T.i32(), [_v(Int64(address), loc, ip)], text, "=r,l", loc, ip, True)
    )


@dsl_user_op
def ld_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    text = "ld.global.f32 $0, [$1];"
    return Float32(
        _asm(T.f32(), [_v(Int64(address), loc, ip)], text, "=f,l", loc, ip, True)
    )


@dsl_user_op
def ld_u32x2(address: Int64, *, loc=None, ip=None):
    text = "ld.global.v2.b32 {$0, $1}, [$2];"
    res = _asm(
        _struct(2), [_v(Int64(address), loc, ip)], text, "=r,=r,l", loc, ip, True
    )
    return [Uint32(w) for w in _words(res, 2, loc, ip)]


@dsl_user_op
def ld_u32x4(address: Int64, *, loc=None, ip=None):
    text = "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];"
    res = _asm(
        _struct(4), [_v(Int64(address), loc, ip)], text, "=r,=r,=r,=r,l", loc, ip, True
    )
    return [Uint32(w) for w in _words(res, 4, loc, ip)]


@dsl_user_op
def ld_f32_s32(address: Int64, *, loc=None, ip=None):
    """A float and an int32 stored side by side."""
    text = "ld.global.v2.b32 {$0, $1}, [$2];"
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [_v(Int64(address), loc, ip)],
        text,
        "=f,=r,l",
        loc,
        ip,
        True,
    )
    value = llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)
    other = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Float32(value), Int32(other)


@dsl_user_op
def ld_s32x2(address: Int64, *, loc=None, ip=None):
    text = "ld.global.v2.b32 {$0, $1}, [$2];"
    res = _asm(
        _struct(2), [_v(Int64(address), loc, ip)], text, "=r,=r,l", loc, ip, True
    )
    return [Int32(w) for w in _words(res, 2, loc, ip)]


@dsl_user_op
def st_f32_s32(address: Int64, value: Float32, other: Int32, *, loc=None, ip=None):
    operands = [_v(Int64(address), loc, ip), _v(Float32(value), loc, ip)]
    operands.append(_v(Int32(other), loc, ip))
    _asm(None, operands, "st.global.v2.b32 [$0], {$1, $2};", "l,f,r", loc, ip, True)


@dsl_user_op
def st_s32x2(address: Int64, a: Int32, b: Int32, *, loc=None, ip=None) -> None:
    operands = [_v(Int64(address), loc, ip), _v(Int32(a), loc, ip)]
    operands.append(_v(Int32(b), loc, ip))
    _asm(None, operands, "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, True)


@dsl_user_op
def bulk_copy(
    smem: Int32, address: Int64, size: Int32, mbar: Int32, *, loc=None, ip=None
):
    """Copy ``size`` bytes (16-byte aligned) from global to shared memory with
    the TMA unit, completing on the shared memory barrier ``mbar``."""
    text = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];"
    )
    operands = [_v(Int32(smem), loc, ip), _v(Int64(address), loc, ip)]
    operands += [_v(Int32(size), loc, ip), _v(Int32(mbar), loc, ip)]
    _asm(None, operands, text, "r,l,r,r", loc, ip, True)


@dsl_user_op
def ld_shared_u32x4(smem: Int32, *, loc=None, ip=None):
    text = "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];"
    res = _asm(
        _struct(4), [_v(Int32(smem), loc, ip)], text, "=r,=r,=r,=r,r", loc, ip, True
    )
    return [Uint32(w) for w in _words(res, 4, loc, ip)]


@dsl_user_op
def st_f32x4(address: Int64, a, b, c, d, *, loc=None, ip=None) -> None:
    operands = [_v(Int64(address), loc, ip)] + [
        _v(Float32(x), loc, ip) for x in (a, b, c, d)
    ]
    text = "st.global.v4.f32 [$0], {$1, $2, $3, $4};"
    _asm(None, operands, text, "l,f,f,f,f", loc, ip, True)


@dsl_user_op
def st_shared_f32(smem: Int32, value: Float32, *, loc=None, ip=None) -> None:
    operands = [_v(Int32(smem), loc, ip), _v(Float32(value), loc, ip)]
    _asm(None, operands, "st.shared.f32 [$0], $1;", "r,f", loc, ip, True)


@dsl_user_op
def st_shared_f32x4(smem: Int32, a, b, c, d, *, loc=None, ip=None) -> None:
    operands = [_v(Int32(smem), loc, ip)] + [
        _v(Float32(x), loc, ip) for x in (a, b, c, d)
    ]
    text = "st.shared.v4.f32 [$0], {$1, $2, $3, $4};"
    _asm(None, operands, text, "r,f,f,f,f", loc, ip, True)


@dsl_user_op
def ld_shared_f32x4(smem: Int32, *, loc=None, ip=None):
    text = "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];"
    res = _asm(
        _struct(4, T.f32()),
        [_v(Int32(smem), loc, ip)],
        text,
        "=f,=f,=f,=f,r",
        loc,
        ip,
        True,
    )
    return [Float32(w) for w in _words(res, 4, loc, ip, T.f32())]


@dsl_user_op
def ld_shared_u32(smem: Int32, *, loc=None, ip=None) -> Uint32:
    text = "ld.shared.b32 $0, [$1];"
    return Uint32(
        _asm(T.i32(), [_v(Int32(smem), loc, ip)], text, "=r,r", loc, ip, True)
    )


@dsl_user_op
def st_f32(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    operands = [_v(Int64(address), loc, ip), _v(Float32(value), loc, ip)]
    _asm(None, operands, "st.global.f32 [$0], $1;", "l,f", loc, ip, True)


@dsl_user_op
def st_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    operands = [_v(Int64(address), loc, ip), _v(Uint32(value), loc, ip)]
    _asm(None, operands, "st.global.b32 [$0], $1;", "l,r", loc, ip, True)


@dsl_user_op
def st_u8(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    operands = [_v(Int64(address), loc, ip), _v(Uint32(value), loc, ip)]
    text = "{\n.reg .b16 b;\ncvt.u16.u32 b, $1;\nst.global.u8 [$0], b;\n}"
    _asm(None, operands, text, "l,r", loc, ip, True)


@dsl_user_op
def e4m3_byte(value: Float32, *, loc=None, ip=None) -> Uint32:
    """``value`` as an E4M3 byte (round to nearest, saturating)."""
    text = (
        "{\n.reg .b16 b;\ncvt.rn.satfinite.e4m3x2.f32 b, 0f00000000, $1;\n"
        "cvt.u32.u16 $0, b;\nand.b32 $0, $0, 255;\n}"
    )
    return Uint32(_asm(T.i32(), [_v(Float32(value), loc, ip)], text, "=r,f", loc, ip))


@dsl_user_op
def arrive_count(address: Int64, *, loc=None, ip=None) -> Int32:
    """Add 1 to the int32 at ``address`` (acquire-release, GPU scope);
    returns the old value."""
    text = "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;"
    return Int32(
        _asm(T.i32(), [_v(Int64(address), loc, ip)], text, "=r,l", loc, ip, True)
    )


@dsl_user_op
def ld_cg_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """A float other CTAs of this grid wrote (from L2)."""
    text = "ld.global.cg.f32 $0, [$1];"
    return Float32(
        _asm(T.f32(), [_v(Int64(address), loc, ip)], text, "=f,l", loc, ip, True)
    )


@dsl_user_op
def ld_cg_f32x4(address: Int64, *, loc=None, ip=None):
    text = "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];"
    res = _asm(
        _struct(4, T.f32()),
        [_v(Int64(address), loc, ip)],
        text,
        "=f,=f,=f,=f,l",
        loc,
        ip,
        True,
    )
    return [Float32(w) for w in _words(res, 4, loc, ip, T.f32())]


@dsl_user_op
def scale_pairs(scales: Uint32, *, loc=None, ip=None):
    """The two E4M3 scales in the low 16 bits of ``scales`` as the F16 pairs
    (s0, s0) and (s1, s1)."""
    text = (
        "{\n.reg .b16 b, lo, hi;\n.reg .b32 h;\ncvt.u16.u32 b, $2;\n"
        "cvt.rn.f16x2.e4m3x2 h, b;\nmov.b32 {lo, hi}, h;\n"
        "mov.b32 $0, {lo, lo};\nmov.b32 $1, {hi, hi};\n}"
    )
    res = _asm(_struct(2), [_v(Uint32(scales), loc, ip)], text, "=r,=r,r", loc, ip)
    return [Uint32(w) for w in _words(res, 2, loc, ip)]


@dsl_user_op
def scale_quad(scales: Uint32, *, loc=None, ip=None):
    """The four E4M3 scales of ``scales`` as F16 pairs (s_i, s_i)."""
    text = (
        "{\n.reg .b16 b0, b1, l0, h0, l1, h1;\n.reg .b32 x, y;\n"
        "mov.b32 {b0, b1}, $4;\ncvt.rn.f16x2.e4m3x2 x, b0;\n"
        "cvt.rn.f16x2.e4m3x2 y, b1;\nmov.b32 {l0, h0}, x;\nmov.b32 {l1, h1}, y;\n"
        "mov.b32 $0, {l0, l0};\nmov.b32 $1, {h0, h0};\n"
        "mov.b32 $2, {l1, l1};\nmov.b32 $3, {h1, h1};\n}"
    )
    res = _asm(
        _struct(4), [_v(Uint32(scales), loc, ip)], text, "=r,=r,=r,=r,r", loc, ip
    )
    return [Uint32(w) for w in _words(res, 4, loc, ip)]


def _decode(code: str, scale: str) -> str:
    """PTX of eight E2M1 codes (``code``) times the F16 pair ``scale`` as the
    E4M3 pairs e0 .. e3."""
    return (
        ".reg .b8 c0, c1, c2, c3;\n.reg .b32 h0, h1, h2, h3;\n"
        ".reg .b16 e0, e1, e2, e3;\n"
        f"mov.b32 {{c0, c1, c2, c3}}, {code};\n"
        "cvt.rn.f16x2.e2m1x2 h0, c0;\ncvt.rn.f16x2.e2m1x2 h1, c1;\n"
        "cvt.rn.f16x2.e2m1x2 h2, c2;\ncvt.rn.f16x2.e2m1x2 h3, c3;\n"
        f"mul.rn.f16x2 h0, h0, {scale};\nmul.rn.f16x2 h1, h1, {scale};\n"
        f"mul.rn.f16x2 h2, h2, {scale};\nmul.rn.f16x2 h3, h3, {scale};\n"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\ncvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
        "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\ncvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
    )


@dsl_user_op
def nvfp4_f16x8(code: Uint32, scale: Uint32, *, loc=None, ip=None):
    """Eight E2M1 codes times the F16 pair ``scale``, rounded to E4M3, as four
    F16 pairs (codes 2i and 2i + 1 in word i)."""
    text = (
        "{\n"
        + _decode("$4", "$5")
        + "cvt.rn.f16x2.e4m3x2 $0, e0;\ncvt.rn.f16x2.e4m3x2 $1, e1;\n"
        "cvt.rn.f16x2.e4m3x2 $2, e2;\ncvt.rn.f16x2.e4m3x2 $3, e3;\n}"
    )
    operands = [_v(Uint32(code), loc, ip), _v(Uint32(scale), loc, ip)]
    res = _asm(_struct(4), operands, text, "=r,=r,=r,=r,r,r", loc, ip)
    return [Uint32(w) for w in _words(res, 4, loc, ip)]


@dsl_user_op
def e4m3x4_f16(word: Uint32, *, loc=None, ip=None):
    """Four E4M3 bytes as two F16 pairs (bytes 0-1, then 2-3)."""
    text = (
        "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $2;\n"
        "cvt.rn.f16x2.e4m3x2 $0, lo;\ncvt.rn.f16x2.e4m3x2 $1, hi;\n}"
    )
    res = _asm(_struct(2), [_v(Uint32(word), loc, ip)], text, "=r,=r,r", loc, ip)
    return [Uint32(w) for w in _words(res, 2, loc, ip)]


@dsl_user_op
def prmt(a: Uint32, b: Uint32, selector: int, *, loc=None, ip=None) -> Uint32:
    operands = [_v(Uint32(a), loc, ip), _v(Uint32(b), loc, ip)]
    text = f"prmt.b32 $0, $1, $2, {selector:#x};"
    return Uint32(_asm(T.i32(), operands, text, "=r,r,r", loc, ip))


@dsl_user_op
def f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    operands = [_v(Float32(lo), loc, ip), _v(Float32(hi), loc, ip)]
    text = "cvt.rn.f16x2.f32 $0, $2, $1;"
    return Uint32(_asm(T.i32(), operands, text, "=r,f,f", loc, ip))


def _mma(shape, a, b, c, loc, ip):
    operands = [_v(Uint32(x), loc, ip) for x in (*a, *b)]
    operands += [_v(Float32(x), loc, ip) for x in c]
    text = (
        f"mma.sync.aligned.{shape} {{$0, $1, $2, $3}}, "
        "{$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    res = _asm(
        _struct(4, T.f32()),
        operands,
        text,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc,
        ip,
    )
    return [Float32(x) for x in _words(res, 4, loc, ip, T.f32())]


@dsl_user_op
def mma_f16(a, b, c, *, loc=None, ip=None):
    """D = A B + C, m16n8k16, F16 operands."""
    return _mma("m16n8k16.row.col.f32.f16.f16.f32", a, b, c, loc, ip)
