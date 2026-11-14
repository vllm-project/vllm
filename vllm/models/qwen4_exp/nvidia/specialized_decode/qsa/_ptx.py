# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the QSA row attention.

Memory accesses are side-effecting assembly, so the compiler keeps them where
the kernel issues them relative to the grid-dependency wait and the barriers
that order them. Addresses are 64-bit global or 32-bit shared-memory
integers. Registers holding BF16 data are 32-bit words of two values (the
lower half is the lower address or index).

The arithmetic helpers name the instructions the stock Triton kernels compile
to where the row attention keeps their rounding: `ex2.approx` for the
softmax exponentials and Triton's `tl.sigmoid` sequence for the output gate.
"""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..hc._ptx import Floats4, Words4, _asm, _words

Floats2 = tuple[Float32, Float32]


@dsl_user_op
def copy_async_16_fill(
    shared: Int32, address: Int64, size: Int32, *, loc=None, ip=None
) -> None:
    """Copy `size` (0 or 16) bytes from global memory through L2 to 16 bytes
    of shared memory and fill the rest with zeros (`cp.async.cg` with a
    source size; a size of 0 reads nothing)."""
    _asm(
        None,
        [
            Int32(shared).ir_value(loc=loc, ip=ip),
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_matrix_x4_trans(address: Int32, *, loc=None, ip=None) -> Words4:
    """Four transposed 8 x 8 BF16 matrices from shared memory
    (`ldmatrix.x4.trans`): lanes `8 i .. 8 i + 7` give the addresses of
    the 16-byte rows of matrix `i`; word `i` of lane `(g, t)` holds
    elements `(2 t, g)` and `(2 t + 1, g)` of matrix `i`."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)


@dsl_user_op
def mma_bf16_1688(
    acc: Floats4, a: tuple[Uint32, Uint32], b: Uint32, *, loc=None, ip=None
) -> Floats4:
    """`acc + A B` by one `mma.sync.aligned.m16n8k8` (BF16 inputs, FP32
    accumulator): lane `(g, t)` gives A rows `g` (word 0) and `g + 8`
    (word 1) at columns `2 t, 2 t + 1`, B rows `2 t, 2 t + 1` at column
    `g`, and holds D at rows `g, g + 8` and columns `2 t, 2 t + 1`."""
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in acc]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in a]
        + [Uint32(b).ir_value(loc=loc, ip=ip)],
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$8, $9}, {$10}, {$4, $5, $6, $7};",
        "=f,=f,=f,=f,f,f,f,f,r,r,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Float32(llvm.extractvalue(T.f32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def exp2_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """`2^a` by `ex2.approx.ftz.f32` (`2^-inf` is 0)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sigmoid(x: Float32, *, loc=None, ip=None) -> Float32:
    """`tl.sigmoid(x)` in the instruction sequence Triton emits for FP32:
    `1 / (1 + 2^(x * -log2(e)))` with `ex2.approx` and a full-range
    division."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 e, d;\n\t"
            "mul.f32 e, $1, 0fBFB8AA3B;\n\t"
            "ex2.approx.f32 e, e;\n\t"
            "add.f32 d, e, 0f3F800000;\n\t"
            "div.full.f32 $0, 0f3F800000, d;\n\t}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def vote_any(flag: Int32, *, loc=None, ip=None) -> Int32:
    """1 when `flag` is non-zero in any lane of the warp, else 0 (every
    lane must execute it)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(flag).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p, q;\n\tsetp.ne.s32 p, $1, 0;\n\t"
            "vote.sync.any.pred q, p, 0xffffffff;\n\tselp.s32 $0, 1, 0, q;\n\t}",
            "=r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_s32_if(address: Int64, flag: Int32, *, loc=None, ip=None) -> Int32:
    """The 32-bit integer an earlier kernel wrote at `address` when `flag`
    is non-zero, else -1 (no access)."""
    return Int32(
        _asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Int32(flag).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.b32 $0, 0xFFFFFFFF;\n\t"
            "@p ld.global.s32 $0, [$1];\n\t}",
            "=r,l,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_shared_s32x2(address: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32]:
    """Two 32-bit integers from an 8-byte aligned shared-memory address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.s32 {$0, $1}, [$2];",
        "=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), loaded, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), loaded, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def store_shared_f32x2(address: Int32, values: Floats2, *, loc=None, ip=None) -> None:
    """Store two FP32 values at an 8-byte aligned shared-memory address."""
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32x2(address: Int32, *, loc=None, ip=None) -> Floats2:
    """Two FP32 values from an 8-byte aligned shared-memory address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), loaded, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), loaded, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def load_shared_s32x4(
    address: Int32, *, loc=None, ip=None
) -> tuple[Int32, Int32, Int32, Int32]:
    """Four 32-bit integers from a 16-byte aligned shared-memory address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.s32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Int32(llvm.extractvalue(T.i32(), loaded, [index], loc=loc, ip=ip))
        for index in range(4)
    )
