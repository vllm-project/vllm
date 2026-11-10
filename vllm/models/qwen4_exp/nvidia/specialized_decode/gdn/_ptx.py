# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the GDN decode core.

The arithmetic helpers name the exact PTX instruction each step of the stock
decode path compiles to, so the fused kernel reproduces its rounding:

* the convolution (Triton `_causal_conv1d_update_kernel`): BF16 products,
  FP32 sums without flush-to-zero, `ex2.approx.f32` and `div.full.f32` in
  its SiLU;
* the recurrence, the normalizations and the gates (the CUDA
  `gdn_decode_post_conv_mtp` kernel, built with `--use_fast_math`): FP32
  with flush-to-zero, contracted multiply-adds, `ex2.approx.ftz`,
  `rcp.approx.ftz` and `rsqrt.approx.ftz`; the state updates use the packed
  `.f32x2` multiply and multiply-add (SM100+), which round each element as
  the scalar instructions do.

Memory accesses are side-effecting assembly, so the compiler keeps them where
the kernel issues them relative to the grid-dependency wait. Addresses are
64-bit global or 32-bit shared-memory integers.
"""

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import _asm

Floats2 = tuple[Float32, Float32]
Floats4 = tuple[Float32, Float32, Float32, Float32]

# log2(e) as the FP32 constant nvcc's __expf and Triton's exp multiply by.
LOG2E = 1.4426950408889634


def _f32(value: Float32, loc, ip):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unary(text: str, a: Float32, loc, ip) -> Float32:
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
def fma_ftz(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b + c`, rounded once, subnormals flushed."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip), _f32(c, loc, ip)],
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a * b` with subnormals flushed."""
    return _binary("mul.rn.ftz.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def add_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b` with subnormals flushed."""
    return _binary("add.rn.ftz.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def sub_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a - b` with subnormals flushed."""
    return _binary("sub.rn.ftz.f32 $0, $1, $2;", a, b, loc, ip)


def _packed(op: str, pairs: list[Floats2], loc, ip) -> Floats2:
    """A packed `.f32x2` instruction (SM100+) on pairs of FP32 values: each
    element is rounded as the scalar instruction rounds it."""
    names = [f"a{i}" for i in range(len(pairs))]
    moves = "".join(
        f"mov.b64 {name}, {{${2 + 2 * i}, ${3 + 2 * i}}};\n\t"
        for i, name in enumerate(names)
    )
    text = (
        f"{{\n\t.reg .b64 d, {', '.join(names)};\n\t{moves}"
        f"{op} d, {', '.join(names)};\n\tmov.b64 {{$0, $1}}, d;\n\t}}"
    )
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_f32(value, loc, ip) for pair in pairs for value in pair],
        text,
        "=f,=f," + ",".join(["f"] * (2 * len(pairs))),
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Float32(llvm.extractvalue(T.f32(), result, [index], loc=loc, ip=ip))
        for index in range(2)
    )


@dsl_user_op
def fma_ftz_x2(a: Floats2, b: Floats2, c: Floats2, *, loc=None, ip=None) -> Floats2:
    """`fma_ftz` of two pairs of operands in one instruction."""
    return _packed("fma.rn.ftz.f32x2", [a, b, c], loc, ip)


@dsl_user_op
def mul_ftz_x2(a: Floats2, b: Floats2, *, loc=None, ip=None) -> Floats2:
    """`mul_ftz` of two pairs of operands in one instruction."""
    return _packed("mul.rn.ftz.f32x2", [a, b], loc, ip)


@dsl_user_op
def rsqrt_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """`rsqrtf(a)` under fast math."""
    return _unary("rsqrt.approx.ftz.f32 $0, $1;", a, loc, ip)


@dsl_user_op
def rcp_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """`1.0f / a` under fast math."""
    return _unary("rcp.approx.ftz.f32 $0, $1;", a, loc, ip)


@dsl_user_op
def exp_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """`__expf(a)`: `2^(a log2(e))` with flush-to-zero."""
    return _unary(
        "{\n\t.reg .f32 t;\n\tmul.rn.ftz.f32 t, $1, 0f3FB8AA3B;\n\t"
        "ex2.approx.ftz.f32 $0, t;\n\t}",
        a,
        loc,
        ip,
    )


@dsl_user_op
def conv_silu(acc: Float32, *, loc=None, ip=None) -> Float32:
    """`acc / (1 + exp(-acc))` as the Triton convolution computes it."""
    return _unary(
        "{\n\t.reg .f32 t, e;\n\tmul.rn.f32 t, $1, 0fBFB8AA3B;\n\t"
        "ex2.approx.f32 e, t;\n\tadd.rn.f32 e, e, 0f3F800000;\n\t"
        "div.full.f32 $0, $1, e;\n\t}",
        acc,
        loc,
        ip,
    )


@dsl_user_op
def softplus_select(x: Float32, smooth: Float32, *, loc=None, ip=None) -> Float32:
    """`x > 20 ? x : smooth`: the stock softplus's linear branch."""
    return _binary(
        "{\n\t.reg .pred p;\n\tsetp.gt.ftz.f32 p, $1, 0f41A00000;\n\t"
        "selp.f32 $0, $1, $2, p;\n\t}",
        x,
        smooth,
        loc,
        ip,
    )


@dsl_user_op
def select_f32(flag: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`flag != 0 ? a : b`."""
    return Float32(
        _asm(
            T.f32(),
            [Int32(flag).ir_value(loc=loc, ip=ip), _f32(a, loc, ip), _f32(b, loc, ip)],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\t"
            "selp.f32 $0, $2, $3, p;\n\t}",
            "=f,r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_s32(flag: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    """`flag != 0 ? a : b`."""
    return Int32(
        _asm(
            T.i32(),
            [
                Int32(flag).ir_value(loc=loc, ip=ip),
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\t"
            "selp.b32 $0, $2, $3, p;\n\t}",
            "=r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """`a + b` (IEEE, subnormals kept)."""
    return _binary("add.rn.f32 $0, $1, $2;", a, b, loc, ip)


@dsl_user_op
def mul_bf16(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """The BF16 product of two BF16 values held as FP32, widened back."""
    return _binary(
        "{\n\t.reg .f32 p;\n\t.reg .b16 h;\n\tmul.rn.f32 p, $1, $2;\n\t"
        "cvt.rn.bf16.f32 h, p;\n\tcvt.f32.bf16 $0, h;\n\t}",
        a,
        b,
        loc,
        ip,
    )


@dsl_user_op
def round_bf16(a: Float32, *, loc=None, ip=None) -> Float32:
    """`a` rounded to nearest BF16, widened back."""
    return _unary(
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.f32.bf16 $0, h;\n\t}",
        a,
        loc,
        ip,
    )


@dsl_user_op
def load_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """One int32 an earlier kernel or copy wrote."""
    return Int32(
        _asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
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


def _floats4(struct, loc, ip) -> Floats4:
    return tuple(  # type: ignore[return-value]
        Float32(llvm.extractvalue(T.f32(), struct, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def load_state_x4(address: Int64, *, loc=None, ip=None) -> Floats4:
    """Four FP32 state values (16 bytes) without L1 allocation: the state is
    read once per step."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.L1::no_allocate.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _floats4(loaded, loc, ip)


@dsl_user_op
def load_state_x4_if(flag: Int32, address: Int64, *, loc=None, ip=None) -> Floats4:
    """Four FP32 values (16 bytes, no L1 allocation) when `flag != 0`, else
    zeros without a load."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(flag).ir_value(loc=loc, ip=ip), Int64(address).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "mov.f32 $0, 0f00000000;\n\tmov.f32 $1, 0f00000000;\n\t"
        "mov.f32 $2, 0f00000000;\n\tmov.f32 $3, 0f00000000;\n\t"
        "@p ld.global.L1::no_allocate.v4.f32 {$0, $1, $2, $3}, [$5];\n\t}",
        "=f,=f,=f,=f,r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _floats4(loaded, loc, ip)


@dsl_user_op
def load_f32_if(flag: Int32, address: Int64, *, loc=None, ip=None) -> Float32:
    """One FP32 value when `flag != 0`, else zero without a load."""
    return Float32(
        _asm(
            T.f32(),
            [
                Int32(flag).ir_value(loc=loc, ip=ip),
                Int64(address).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\t"
            "mov.f32 $0, 0f00000000;\n\t@p ld.global.f32 $0, [$2];\n\t}",
            "=f,r,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_state_x4(address: Int64, values: Floats4, *, loc=None, ip=None) -> None:
    """Store four FP32 state values (16 bytes)."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)] + [_f32(v, loc, ip) for v in values],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_stream_x4(address: Int64, values: Floats4, *, loc=None, ip=None) -> None:
    """Store four FP32 values (16 bytes) as streaming data (`st.global.cs`:
    evict-first in L1 and L2)."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)] + [_f32(v, loc, ip) for v in values],
        "st.global.cs.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_bf16x4(address: Int64, *, loc=None, ip=None) -> Floats4:
    """Four consecutive BF16 weights (8 bytes, non-coherent), widened."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 lo, hi, w;\n\tld.global.nc.v2.b32 {lo, hi}, [$4];\n\t"
        "shl.b32 w, lo, 16;\n\tmov.b32 $0, w;\n\tand.b32 w, lo, 0xffff0000;\n\t"
        "mov.b32 $1, w;\n\tshl.b32 w, hi, 16;\n\tmov.b32 $2, w;\n\t"
        "and.b32 w, hi, 0xffff0000;\n\tmov.b32 $3, w;\n\t}",
        "=f,=f,=f,=f,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _floats4(loaded, loc, ip)


@dsl_user_op
def load_shared_f32x4(address: Int32, *, loc=None, ip=None) -> Floats4:
    """Four FP32 values at a 16-byte aligned shared-memory address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _floats4(loaded, loc, ip)


@dsl_user_op
def load_s64(address: Int64, *, loc=None, ip=None) -> Int64:
    """One int64 (an entry of an address table written before the launch)."""
    return Int64(
        _asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.s64 $0, [$1];",
            "=l,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    """Store one int32."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_f32(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    """Store one FP32 value."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), _f32(value, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_stream_f32(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    """Store one FP32 value as streaming data (`st.global.cs`)."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), _f32(value, loc, ip)],
        "st.global.cs.f32 [$0], $1;",
        "l,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_l2(address: Int64, size: Int32, *, loc=None, ip=None) -> None:
    """Prefetch `size` bytes at `address` into L2 (`cp.async.bulk.prefetch`,
    SM90+; both multiples of 16): nothing reaches the thread."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip), Int32(size).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    """Abort the kernel: the launch fails with a CUDA error (a broken
    invariant of the state protocol)."""
    _asm(None, [], "trap;", "", side_effects=True, loc=loc, ip=ip)
