# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the routed-expert front (`front.py`).

The front reproduces two stock kernels bit for bit, so each arithmetic helper
names the instruction the stock kernel's step compiles to:

* FlashInfer's TRT-LLM routing kernels are built with `-use_fast_math`:
  `expf(x)` is `ex2.approx.ftz(x * log2(e))`, `a / b` is
  `div.approx.ftz` (`__fdividef`), `1.f / a` is `rcp.approx.ftz`, and
  additions and multiplications flush denormals (`.ftz`).
* vLLM's NVFP4 activation quantization (`cvt_warp_fp16_to_fp4`) spells its
  reciprocals as `rcp.approx.ftz` and its conversions as
  `cvt.rn.satfinite` to E4M3 and E2M1; its multiplications round to
  nearest.

No helper leaves a floating-point operation to the compiler, which could fuse
a multiply with an add. Memory accesses are side-effecting assembly so that
they stay where the kernel issues them (after the grid-dependency wait, or
between the barriers that order them). Global addresses are 64-bit integers,
shared-memory addresses 32-bit.
"""

from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import _asm

Floats8 = tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]


def _f32(value: Float32, loc, ip):
    """IR value of a Float32."""
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unary(text: str, a: Float32, loc, ip) -> Float32:
    """One FP32 result of one FP32 operand."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip)],
            text,
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def _binary(text: str, a: Float32, b: Float32, loc, ip) -> Float32:
    """One FP32 result of two FP32 operands."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            text,
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b` under fast math."""
    return _binary("add.ftz.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def sub_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a - b` under fast math."""
    return _binary("sub.ftz.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b` rounded to nearest (never fused with an add)."""
    return _binary("mul.rn.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def max_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`fmaxf(a, b)` under fast math."""
    return _binary("max.ftz.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def div_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a / b` under fast math (`__fdividef`)."""
    return _binary("div.approx.ftz.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def rcp_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """`1.f / a` under fast math."""
    return _unary("rcp.approx.ftz.f32 $0, $1;", a, loc, ip)


@dsl_user_op
def exp_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """`expf(a)` under fast math (`__expf`): `2^(a log2(e))`."""
    return _unary(
        "{\n\t.reg .f32 t;\n\tmul.ftz.f32 t, $1, 0f3FB8AA3B;\n\t"
        "ex2.approx.ftz.f32 $0, t;\n\t}",
        a,
        loc,
        ip,
    )


@dsl_user_op
def bits_f32(a: Float32, *, loc=None, ip=None) -> Uint32:
    """The bits of an FP32 value."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_bits(a: Uint32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of 32 bits."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ordered_f32(a: Float32, *, loc=None, ip=None) -> Uint32:
    """CUB's `TwiddleIn` of an FP32 value: bits whose unsigned order is the
    values' order (sign bit set for non-negative values, all bits flipped
    for negative ones)."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip)],
            "{\n\t.reg .b32 b, m;\n\tmov.b32 b, $1;\n\tshr.s32 m, b, 31;\n\t"
            "or.b32 m, m, 0x80000000;\n\txor.b32 $0, b, m;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unordered_f32(a: Uint32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of `ordered_f32` bits (CUB's `TwiddleOut`)."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 m;\n\tnot.b32 m, $1;\n\tshr.s32 m, m, 31;\n\t"
            "or.b32 m, m, 0x80000000;\n\txor.b32 m, $1, m;\n\tmov.b32 $0, m;\n\t}",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_u64(a: Uint64, b: Uint64, *, loc=None, ip=None) -> Uint64:
    """The larger of two 64-bit keys."""
    return Uint64(
        _asm(
            T.i64(),
            [Uint64(a).ir_value(loc=loc, ip=ip), Uint64(b).ir_value(loc=loc, ip=ip)],
            "max.u64 $0, $1, $2;",
            "=l,l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def clear_if_equal_u64(a: Uint64, b: Uint64, *, loc=None, ip=None) -> Uint64:
    """`0` if `a == b`, else `a`."""
    return Uint64(
        _asm(
            T.i64(),
            [Uint64(a).ir_value(loc=loc, ip=ip), Uint64(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.u64 p, $1, $2;\n\t"
            "selp.b64 $0, 0, $1, p;\n\t}",
            "=l,l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def warp_max_u32(a: Uint32, *, loc=None, ip=None) -> Uint32:
    """The largest of the warp's values (`redux.sync.max.u32`)."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_bf16(address: Int64, *, loc=None, ip=None) -> Float32:
    """One BF16 value an earlier kernel wrote, widened to FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w;\n\tld.global.u16 w, [$1];\n\tshl.b32 w, w, 16;\n\t"
            "mov.b32 $0, w;\n\t}",
            "=f,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_bf16x8(address: Int64, *, loc=None, ip=None) -> Floats8:
    """Eight consecutive BF16 values an earlier kernel wrote (16 bytes),
    widened to FP32."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 w0, w1, w2, w3;\n\t"
        "ld.global.v4.b32 {w0, w1, w2, w3}, [$8];\n\t"
        "shl.b32 $0, w0, 16;\n\tand.b32 $1, w0, 0xffff0000;\n\t"
        "shl.b32 $2, w1, 16;\n\tand.b32 $3, w1, 0xffff0000;\n\t"
        "shl.b32 $4, w2, 16;\n\tand.b32 $5, w2, 0xffff0000;\n\t"
        "shl.b32 $6, w3, 16;\n\tand.b32 $7, w3, 0xffff0000;\n\t}",
        "=f,=f,=f,=f,=f,=f,=f,=f,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Float32(llvm.extractvalue(T.f32(), loaded, [index], loc=loc, ip=ip))
        for index in range(8)
    )


@dsl_user_op
def load_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """One FP32 value."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3_scale(value: Float32, *, loc=None, ip=None) -> tuple[Uint32, Float32]:
    """`value` rounded to the nearest E4M3 with saturation: its byte and
    its FP32 value (`__nv_fp8_e4m3(value)` and back)."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_f32(value, loc, ip)],
        "{\n\t.reg .b16 q, h0, h1;\n\t.reg .b32 pair;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 q, 0f00000000, $2;\n\t"
        "cvt.u32.u16 $0, q;\n\tand.b32 $0, $0, 0xff;\n\t"
        "cvt.rn.f16x2.e4m3x2 pair, q;\n\tmov.b32 {h0, h1}, pair;\n\t"
        "cvt.f32.f16 $1, h0;\n\t}",
        "=r,=f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def e2m1x8(values: Floats8, *, loc=None, ip=None) -> Uint32:
    """Eight values rounded to the nearest E2M1 with saturation, packed two
    per byte (the even value in the low nibble), as vLLM packs them."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(v, loc, ip) for v in values],
            "{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n\t"
            "mov.b32 $0, {b0, b1, b2, b3};\n\t}",
            "=r,f,f,f,f,f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def _store(text: str, constraints: str, operands: list, loc, ip) -> None:
    """One side-effecting store."""
    _asm(None, operands, text, constraints, side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def store_u32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """Store one 32-bit word."""
    _store(
        "st.global.b32 [$0], $1;",
        "l,r",
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        loc,
        ip,
    )


@dsl_user_op
def store_u8(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store the low byte of `value`."""
    _store(
        "st.global.b8 [$0], $1;",
        "l,r",
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        loc,
        ip,
    )


@dsl_user_op
def store_u32x2(address: Int64, lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> None:
    """Store two 32-bit words (8 bytes, `lo` first)."""
    _store(
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(lo).ir_value(loc=loc, ip=ip),
            Uint32(hi).ir_value(loc=loc, ip=ip),
        ],
        loc,
        ip,
    )


@dsl_user_op
def store_bf16(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    """Store `value` rounded to the nearest BF16."""
    _store(
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f",
        [Int64(address).ir_value(loc=loc, ip=ip), _f32(value, loc, ip)],
        loc,
        ip,
    )


@dsl_user_op
def store_shared_u32(address: Int32, value: Int32, *, loc=None, ip=None) -> None:
    """Store one 32-bit word in this CTA's shared memory."""
    _store(
        "st.shared.b32 [$0], $1;",
        "r,r",
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        loc,
        ip,
    )


@dsl_user_op
def load_shared_u32(address: Int32, *, loc=None, ip=None) -> Int32:
    """One 32-bit word of this CTA's shared memory."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(address).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def or_shared_u32(address: Int32, value: Uint32, *, loc=None, ip=None) -> None:
    """Atomically OR `value` into a word of this CTA's shared memory."""
    _store(
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        loc,
        ip,
    )


@dsl_user_op
def store_cluster_u32(
    address: Int32, cta_rank: Int32, value: Int32, *, loc=None, ip=None
) -> None:
    """Store one 32-bit word at `address` in the shared memory of the
    cluster's CTA `cta_rank` (distributed shared memory)."""
    _store(
        "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $0, $1;\n\t"
        "st.shared::cluster.b32 [a], $2;\n\t}",
        "r,r,r",
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Int32(cta_rank).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        loc,
        ip,
    )


@dsl_user_op
def nanosleep(nanoseconds: Int32, *, loc=None, ip=None) -> None:
    """Suspend the thread for about `nanoseconds` (`nanosleep`, at most
    one millisecond)."""
    _store(
        "nanosleep.u32 $0;",
        "r",
        [Int32(nanoseconds).ir_value(loc=loc, ip=ip)],
        loc,
        ip,
    )
