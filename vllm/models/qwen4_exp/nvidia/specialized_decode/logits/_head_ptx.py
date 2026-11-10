# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the FP8-prefiltered draft head (`draft_head.py`).

The error bound's arithmetic is written instruction by instruction with an
explicit rounding mode: upper bounds round up (`.rp`), lower bounds round
down (`.rm`), and the quantization steps that must be exact use `.rn`
operations that the compiler may not contract or reorder. Memory accesses
whose position matters are side-effecting assembly, as in `_ptx.py`.
"""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import _asm

Floats2 = tuple[Float32, Float32]


def _f(value: Float32, loc, ip):
    """The IR value of an FP32 operand."""
    return Float32(value).ir_value(loc=loc, ip=ip)


def _float_op(text: str, operands: list[Float32], loc, ip) -> Float32:
    """One FP32 instruction `text` over `operands` ($1, $2, ...)."""
    return Float32(
        _asm(
            T.f32(),
            [_f(x, loc, ip) for x in operands],
            text,
            "=f" + ",f" * len(operands),
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_ru(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b` rounded up."""
    return _float_op("add.rp.f32 $0, $1, $2;", [a, b], loc, ip)


@dsl_user_op
def mul_ru(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b` rounded up."""
    return _float_op("mul.rp.f32 $0, $1, $2;", [a, b], loc, ip)


@dsl_user_op
def fma_ru(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b + c` with one rounding up."""
    return _float_op("fma.rp.f32 $0, $1, $2, $3;", [a, b, c], loc, ip)


@dsl_user_op
def sub_rd(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a - b` rounded down."""
    return _float_op("sub.rm.f32 $0, $1, $2;", [a, b], loc, ip)


@dsl_user_op
def sqrt_ru(a: Float32, *, loc=None, ip=None) -> Float32:
    """The square root of `a >= 0` rounded up."""
    return _float_op("sqrt.rp.f32 $0, $1;", [a], loc, ip)


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b` rounded to nearest, never contracted into a multiply-add."""
    return _float_op("mul.rn.f32 $0, $1, $2;", [a, b], loc, ip)


@dsl_user_op
def sub_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a - b` rounded to nearest, never contracted into a multiply-add."""
    return _float_op("sub.rn.f32 $0, $1, $2;", [a, b], loc, ip)


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b` rounded to nearest, never contracted into a multiply-add."""
    return _float_op("add.rn.f32 $0, $1, $2;", [a, b], loc, ip)


@dsl_user_op
def code_scale(amax: Float32, *, loc=None, ip=None) -> Float32:
    """The scale of one row's E4M3 codes: the smallest power of two not below
    `amax / 448` (`amax` positive, finite and normal), so every value over
    the scale lies within the E4M3 range. `amax / 448` is rounded up, then a
    value with a non-zero significand goes up to the next power of two."""
    return _float_op(
        "{\n\t.reg .f32 q;\n\t.reg .b32 b, m, e;\n\t.reg .pred p;\n\t"
        "div.rp.f32 q, $1, 0f43E00000;\n\t"
        "mov.b32 b, q;\n\tand.b32 m, b, 0x007FFFFF;\n\t"
        "and.b32 e, b, 0x7F800000;\n\tsetp.ne.u32 p, m, 0;\n\t"
        "@p add.u32 e, e, 0x00800000;\n\tmov.b32 $0, e;\n\t}",
        [amax],
        loc,
        ip,
    )


@dsl_user_op
def quantize_e4m3x2(
    a: Float32, b: Float32, scale: Float32, *, loc=None, ip=None
) -> tuple[Uint32, Float32, Float32]:
    """E4M3 codes of `a / scale` and `b / scale` (round to nearest even,
    saturating; `scale` a normal power of two, so its reciprocal and the
    products are exact) packed in
    the low 16 bits of a word, `a`'s code in the low byte, and the values
    the codes stand for times `scale` (exact)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
        [_f(a, loc, ip), _f(b, loc, ip), _f(scale, loc, ip)],
        "{\n\t.reg .f32 x, y, u, v;\n\t.reg .b16 c;\n\t.reg .b32 h;\n\t"
        ".reg .f16 l0, l1;\n\t"
        "rcp.rn.f32 u, $5;\n\tmul.rn.f32 x, $3, u;\n\tmul.rn.f32 y, $4, u;\n\t"
        # The second source operand lands in the low byte.
        "cvt.rn.satfinite.e4m3x2.f32 c, y, x;\n\t"
        "cvt.u32.u16 $0, c;\n\t"
        "cvt.rn.f16x2.e4m3x2 h, c;\n\t"
        "mov.b32 {l0, l1}, h;\n\t"
        "cvt.f32.f16 u, l0;\n\tcvt.f32.f16 v, l1;\n\t"
        "mul.rn.f32 $1, u, $5;\n\tmul.rn.f32 $2, v, $5;\n\t}",
        "=r,=f,=f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), loaded, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), loaded, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), loaded, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def widen_bf16x2(word: Uint32, *, loc=None, ip=None) -> Floats2:
    """The two BF16 values of a word as FP32 (the low half first)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $2, 16;\n\tmov.b32 $0, t;\n\t"
        "and.b32 t, $2, 0xFFFF0000;\n\tmov.b32 $1, t;\n\t}",
        "=f,=f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), loaded, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), loaded, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def round_bf16(value: Float32, *, loc=None, ip=None) -> Float32:
    """`value` rounded to nearest BF16, as FP32 (monotone in `value`)."""
    return _float_op(
        "{\n\t.reg .b16 h;\n\t.reg .b32 t;\n\tcvt.rn.bf16.f32 h, $1;\n\t"
        "cvt.u32.u16 t, h;\n\tshl.b32 t, t, 16;\n\tmov.b32 $0, t;\n\t}",
        [value],
        loc,
        ip,
    )


@dsl_user_op
def bf16_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    """The bits of `value` rounded to nearest BF16, in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [_f(value, loc, ip)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def abs_or_inf(value: Float32, *, loc=None, ip=None) -> Float32:
    """`|value|`, or +inf for a NaN or infinite `value`."""
    return _float_op(
        "{\n\t.reg .f32 t;\n\t.reg .pred p;\n\tabs.f32 t, $1;\n\t"
        "setp.le.f32 p, t, 0f7F7FFFFF;\n\tselp.f32 $0, t, 0f7F800000, p;\n\t}",
        [value],
        loc,
        ip,
    )


@dsl_user_op
def not_finite(value: Float32, *, loc=None, ip=None) -> Float32:
    """1.0 for a NaN or infinite `value`, else 0.0."""
    return _float_op(
        "{\n\t.reg .f32 t;\n\t.reg .pred p;\n\tabs.f32 t, $1;\n\t"
        "setp.le.f32 p, t, 0f7F7FFFFF;\n\tselp.f32 $0, 0f00000000, 0f3F800000, p;\n\t}",
        [value],
        loc,
        ip,
    )


@dsl_user_op
def ordered_f32(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Order-preserving bits of a non-NaN FP32 value: unsigned comparison of
    the results orders the values (-0 just below +0); every result is
    positive, so 0 is below all of them."""
    return Uint32(
        _asm(
            T.i32(),
            [_f(value, loc, ip)],
            "{\n\t.reg .b32 b, s;\n\tmov.b32 b, $1;\n\tshr.s32 s, b, 31;\n\t"
            "or.b32 s, s, 0x80000000;\n\txor.b32 $0, b, s;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def from_ordered_f32(bits: Uint32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of `ordered_f32` bits (a non-zero word)."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(bits).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 b, s;\n\t.reg .pred p;\n\t"
            "setp.lt.s32 p, $1, 0;\n\t"
            "xor.b32 s, $1, 0x80000000;\n\tnot.b32 b, $1;\n\t"
            "selp.b32 b, s, b, p;\n\tmov.b32 $0, b;\n\t}",
            "=f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def logit_key(total: Float32, index: Uint32, *, loc=None, ip=None) -> Int64:
    """The token key of an FP32 logit sum at vocabulary `index`, as the
    token choice orders tokens (`choice.py`): the order-preserving bits of
    the sum rounded to nearest BF16 (NaN as -inf, -0 as +0), shifted by 24,
    then `0xFFFFFF - index`."""
    return Int64(
        _asm(
            T.i64(),
            [_f(total, loc, ip), Uint32(index).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 v, t;\n\t.reg .b64 k, i;\n\t"
            ".reg .pred p;\n\t"
            "cvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 v, h;\n\t"
            "and.b32 t, v, 0x7FFF;\n\tsetp.gt.u32 p, t, 0x7F80;\n\t"
            "@p mov.b32 v, 0xFF80;\n\t"
            "setp.eq.u32 p, v, 0x8000;\n\t@p mov.b32 v, 0;\n\t"
            "and.b32 t, v, 0x8000;\n\tsetp.ne.u32 p, t, 0;\n\t"
            "xor.b32 t, v, 0xFFFF;\n\tor.b32 v, v, 0x8000;\n\tselp.b32 v, t, v, p;\n\t"
            "cvt.u64.u32 k, v;\n\tshl.b64 k, k, 24;\n\t"
            "mov.b32 t, 0xFFFFFF;\n\tsub.u32 t, t, $2;\n\tcvt.u64.u32 i, t;\n\t"
            "or.b64 $0, k, i;\n\t}",
            "=l,f,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """One FP32 value an earlier kernel wrote."""
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
def load_f32_l2(address: Int64, *, loc=None, ip=None) -> Float32:
    """One FP32 value from L2, never from this SM's L1 (`.cg`): for reads
    before a grid-dependency wait, which no L1 invalidation may precede."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_f32x2(address: Int64, *, loc=None, ip=None) -> Floats2:
    """Two FP32 values an earlier kernel wrote (8-byte aligned)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), loaded, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), loaded, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def load_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """One 32-bit word an earlier kernel wrote."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_u32x2(address: Int64, *, loc=None, ip=None) -> tuple:
    """Two 32-bit words an earlier kernel wrote (8-byte aligned)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), loaded, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def load_shared_x2(address: Int32, *, loc=None, ip=None) -> tuple:
    """8 bytes from an 8-byte aligned shared-memory address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), loaded, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def store_shared_x4(address: Int32, words: tuple, *, loc=None, ip=None) -> None:
    """Store 16 bytes at a 16-byte aligned shared-memory address."""
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in words],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_shared_f32(address: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """Store one FP32 value in shared memory."""
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip), _f(value, loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32(address: Int32, *, loc=None, ip=None) -> Float32:
    """One FP32 value from shared memory."""
    return Float32(
        _asm(
            T.f32(),
            [Int32(address).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_shared_u64(address: Int32, value: Int64, *, loc=None, ip=None) -> None:
    """Store one 64-bit word in shared memory."""
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Int64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u64 [$0], $1;",
        "r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_u64(address: Int32, *, loc=None, ip=None) -> Int64:
    """One 64-bit word from shared memory."""
    return Int64(
        _asm(
            T.i64(),
            [Int32(address).ir_value(loc=loc, ip=ip)],
            "ld.shared.u64 $0, [$1];",
            "=l,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_shared_s32(address: Int32, value: Int32, *, loc=None, ip=None) -> None:
    """Store one 32-bit integer in shared memory."""
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.s32 [$0], $1;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_s32(address: Int32, *, loc=None, ip=None) -> Int32:
    """One 32-bit integer from shared memory."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(address).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shared_add_s32(address: Int32, value: Int32, *, loc=None, ip=None) -> Int32:
    """Add to a shared-memory counter; returns the value before."""
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(address).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_max_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """`max` into a GPU-scope 32-bit word (relaxed)."""
    _asm(
        T.i32(),
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "atom.relaxed.gpu.global.max.u32 $0, [$1], $2;",
        "=r,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add to a GPU-scope 32-bit word (relaxed); returns the word before."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_x2(address: Int64, low: Uint32, high: Uint32, *, loc=None, ip=None) -> None:
    """Store 8 bytes (8-byte aligned)."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(low).ir_value(loc=loc, ip=ip),
            Uint32(high).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_f32(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    """Store one FP32 value."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), _f(value, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_u16(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store the low 16 bits of a word."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u16 [$0], h;\n\t}",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    """Abort the kernel: the launch fails with a CUDA error (a broken
    invariant of the draft head's protocol)."""
    _asm(None, [], "trap;", "", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def prefetch_l1(address: Int64, *, loc=None, ip=None) -> None:
    """Bring the global-memory line holding `address` into L1."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L1 [$0];",
        "l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def copy_async_commit(*, loc=None, ip=None) -> None:
    """Close this thread's open group of `cp.async` copies."""
    _asm(None, [], "cp.async.commit_group;", "", side_effects=True, loc=loc, ip=ip)


@dsl_user_op
def copy_async_wait_group(pending: int, *, loc=None, ip=None) -> None:
    """Wait until at most `pending` (a compile-time count) of this thread's
    `cp.async` groups are outstanding."""
    _asm(
        None,
        [],
        f"cp.async.wait_group {int(pending)};",
        "",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
