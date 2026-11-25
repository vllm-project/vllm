# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL device helpers of the SM80 kernels: global loads by address and
FP8/FP4 decoding with integer operations.

SM80 has no instruction that converts E4M3 or E2M1 to a wider type (they
arrive with SM89 and SM100), so these decoders place a code's bits in an FP32
word and scale it by a power of two:

* E4M3 `s eeee mmm`: the FP32 word `s | eeeemmm << 20` has exponent field
  `eeee` and mantissa `mmm`; times `2^120` it is the E4M3 value, the
  subnormals (`eeee = 0`) included, which are FP32 subnormals before the
  multiplication. FP32 arithmetic keeps subnormals unless a kernel opts into
  flush-to-zero, which these kernels do not.
* E2M1 `s eem`: the FP32 word `s | eem << 22` times `2^126`.

Every product is a power-of-two scaling of a value with at most four
significant bits, so it is exact, and so is its BF16 rounding.

Kernels that only need BF16 decode two codes at a time into the halves of a
32-bit word (`e4m3x2_bf16`, `e2m1x2_bf16`): the same bit placement in
BF16, whose exponent field is FP32's, then the power of two and the block
scale applied with two `fma.rn.bf16x2`. Each product is exact, so the
result is the bits the FP32 decoders round to.
"""

import cutlass.cute as cute
from cutlass import Constexpr, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# 2^120 and 2^126: the E4M3 and E2M1 decoding scales.
E4M3_SCALE = 1.329227995784916e36
E2M1_SCALE = 8.507059173023462e37
# The same scales, and -0.0 (the FMA addend that keeps a product's sign), in
# both halves of a BF16x2 word.
E4M3_SCALE_BF16X2 = 0x7B807B80
E2M1_SCALE_BF16X2 = 0x7E807E80
NEGATIVE_ZERO_BF16X2 = 0x80008000


def _asm(result, operands, text, constraints, *, loc, ip):
    """A side-effect-free inline PTX statement."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """One 32-bit global load."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """One 32-bit global load of a signed index."""
    return Int32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.s32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


def _issue(kind: str, address: Int64, *, loc, ip):
    """A 32-bit global load that stays where it is written.

    `load_u32` is free to move: the compiler may sink it into the branch
    that uses its value, behind the loads of earlier branches. A kernel that
    issues a batch of these before decoding them keeps the whole batch in
    flight at once.
    """
    return llvm.inline_asm(
        T.i32(),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        f"ld.global.nc.{kind} $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def issue_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    """`load_u32` issued where it is written (see `_issue`)."""
    return Uint32(_issue("u32", address, loc=loc, ip=ip))


@dsl_user_op
def issue_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """`load_s32` issued where it is written (see `_issue`)."""
    return Int32(_issue("s32", address, loc=loc, ip=ip))


def _store(width: str, address: Int64, value: Uint32, *, loc, ip) -> None:
    """One global store of the low `width` bits of `value`."""
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        f"st.global.{width} [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_u8(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store one byte."""
    _store("u8", address, value, loc=loc, ip=ip)


@dsl_user_op
def store_u16(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store two bytes (2-byte aligned)."""
    _store("u16", address, value, loc=loc, ip=ip)


@dsl_user_op
def store_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    """Store four bytes (4-byte aligned)."""
    _store("u32", address, value, loc=loc, ip=ip)


@dsl_user_op
def f32_from_bits(bits: Uint32, *, loc=None, ip=None) -> Float32:
    """The FP32 value of a 32-bit word."""
    return Float32(
        llvm.bitcast(T.f32(), Uint32(bits).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def bits_from_f32(value: Float32, *, loc=None, ip=None) -> Uint32:
    """The 32-bit word of an FP32 value."""
    return Uint32(
        llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@cute.jit
def ue8m0(code: Uint32) -> Float32:
    """A UE8M0 scale byte as FP32 (the byte is the exponent field)."""
    return f32_from_bits(code << 23)


@cute.jit
def e4m3(code: Uint32) -> Float32:
    """An E4M3 byte (the low 8 bits of `code`) as FP32."""
    bits = ((code & 0x80) << 24) | ((code & 0x7F) << 20)
    return f32_from_bits(bits) * Float32(E4M3_SCALE)


@cute.jit
def e2m1(code: Uint32) -> Float32:
    """An E2M1 nibble (the low 4 bits of `code`) as FP32."""
    bits = ((code & 0x8) << 28) | ((code & 0x7) << 22)
    return f32_from_bits(bits) * Float32(E2M1_SCALE)


@cute.jit
def exp2_to_f32(exponent: Int32) -> Float32:
    """`2^exponent` for a normal FP32 exponent."""
    return f32_from_bits(Uint32(exponent + 127) << 23)


@cute.jit
def ceil_log2(x: Float32) -> Int32:
    """`ceil(log2(x))` of a positive normal FP32 value, exactly."""
    bits = bits_from_f32(x)
    exponent = Int32((bits >> 23) & 0xFF) - 127
    if (bits & 0x7FFFFF) != 0:
        exponent = exponent + 1
    return exponent


@cute.jit
def to_e2m1(x: Float32) -> Uint32:
    """FP32 to an E2M1 nibble, nearest-even and saturating: the magnitude code
    counts the rounding thresholds below `|x|` (at 0.75, 1.75 and 3.5 the
    even code is the upper one)."""
    a = cute.math.absf(x)
    code = Uint32(0)
    if a > 0.25:
        code = code + 1
    if a >= 0.75:
        code = code + 1
    if a > 1.25:
        code = code + 1
    if a >= 1.75:
        code = code + 1
    if a > 2.5:
        code = code + 1
    if a >= 3.5:
        code = code + 1
    if a > 5.0:
        code = code + 1
    return code | ((bits_from_f32(x) >> 28) & 0x8)


@cute.jit
def to_e4m3(x: Float32) -> Uint32:
    """A positive FP32 value in `[2^-9, 448]` to an E4M3 byte, nearest-even.

    Normal codes round the FP32 exponent-and-mantissa field to three mantissa
    bits (a carry moves to the exponent); below `2^-6` the code is the
    multiple of `2^-9`, rounded by the FP32 magic-number addition.
    """
    code = Uint32(0)
    if x >= 0.015625:
        bits = bits_from_f32(x)
        code = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - ((127 - 7) << 3)
    else:
        magic = Float32(12582912.0)  # 1.5 * 2^23
        code = bits_from_f32(x * 512.0 + magic) - bits_from_f32(magic)
    return code


@dsl_user_op
def byte_perm(a: Uint32, b: Uint32, selector: int, *, loc=None, ip=None) -> Uint32:
    """`prmt.b32`: byte `i` of the result is the byte of `(a, b)`
    (bytes 0-3 of `a`, then 4-7 of `b`) named by nibble `i` of
    `selector`."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, $2, {selector:#x};",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_bf16x2(a: Uint32, b: Uint32, c: Uint32, *, loc=None, ip=None) -> Uint32:
    """`a * b + c` on both BF16 halves of 32-bit words, rounded to
    nearest; subnormals are kept."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Uint32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _scale_bf16x2(x: Uint32, power: Constexpr[int], scale: Uint32) -> Uint32:
    """BF16x2 `x * power * scale`, both products exact."""
    negative_zero = Uint32(NEGATIVE_ZERO_BF16X2)
    x = fma_bf16x2(x, Uint32(power), negative_zero)
    return fma_bf16x2(x, scale, negative_zero)


@cute.jit
def e4m3x2_bf16(word: Uint32, pair: Constexpr[int], scale: Uint32) -> Uint32:
    """E4M3 bytes `2 pair` and `2 pair + 1` of `word` times the BF16x2
    `scale`, as BF16x2: code `s eeee mmm` becomes the BF16
    `s | eeeemmm << 4` (the value times `2^-120`, subnormals included)."""
    x = byte_perm(word, Uint32(0), 0x4140 + 0x0202 * pair)
    x = ((x << 8) & 0x80008000) | ((x << 4) & 0x07F007F0)
    return _scale_bf16x2(x, E4M3_SCALE_BF16X2, scale)


@cute.jit
def e2m1x2_bf16(word: Uint32, byte: Constexpr[int], scale: Uint32) -> Uint32:
    """E2M1 nibbles `2 byte` and `2 byte + 1` of `word` times the
    BF16x2 `scale`, as BF16x2: code `s eem` becomes the BF16
    `s | eem << 6` (the value times `2^-126`)."""
    x = byte_perm(word, Uint32(0), 0x4440 + byte)
    # The byte at bits 0-7 and 12-19: the low nibble of each half is a code.
    x = x * Uint32(4097)
    x = ((x << 12) & 0x80008000) | ((x << 6) & 0x01C001C0)
    return _scale_bf16x2(x, E2M1_SCALE_BF16X2, scale)


@cute.jit
def ue8m0_bf16x2(code: Uint32) -> Uint32:
    """A UE8M0 scale byte as BF16 in both halves (the byte is the exponent
    field)."""
    return code * Uint32(0x00800080)


@cute.jit
def e4m3_bf16x2(code: Uint32) -> Uint32:
    """An E4M3 byte as BF16 in both halves."""
    x = (((code & 0x80) << 8) | ((code & 0x7F) << 4)) * Uint32(0x00010001)
    return fma_bf16x2(x, Uint32(E4M3_SCALE_BF16X2), Uint32(NEGATIVE_ZERO_BF16X2))
