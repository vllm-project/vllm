# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX used by the hyper-connection read kernels.

Memory accesses whose position matters are side-effecting assembly, so the
compiler keeps them where the kernel issues them: weight loads before the
grid-dependency wait (they stream while the previous kernel runs), activation
loads after it (an earlier kernel of the graph wrote them), shared-memory
accesses between the barriers that order them.

Addresses are plain integers: 64-bit global addresses, 32-bit shared-memory
addresses. Registers holding BF16 data are 32-bit words of two values (the
lower half is the lower address).
"""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

Words4 = tuple[Uint32, Uint32, Uint32, Uint32]
Floats4 = tuple[Float32, Float32, Float32, Float32]


def _asm(
    result: ir.Type | None,
    operands: list[ir.Value],
    text: str,
    constraints: str,
    *,
    side_effects: bool,
    loc: ir.Location | None,
    ip: ir.InsertionPoint | None,
) -> ir.Value:
    """One PTX inline assembly operation."""
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


def _words(struct: ir.Value, loc, ip) -> Words4:
    """The four 32-bit members of an inline-assembly result struct."""
    return tuple(  # type: ignore[return-value]
        Uint32(llvm.extractvalue(T.i32(), struct, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def evict_first_policy(*, loc=None, ip=None) -> Int64:
    """An L2 cache policy under which the lines a load fills are evicted first
    (for weights read once per forward)."""
    return Int64(
        _asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def load_weight_x4(address: Int64, policy: Int64, *, loc=None, ip=None) -> Words4:
    """16 bytes of an immutable weight: non-coherent, without L1 allocation,
    with the L2 `policy`. Issued where it is written, so a weight load placed
    before the grid-dependency wait overlaps the previous kernel."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Int64(policy).ir_value(loc=loc, ip=ip),
        ],
        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 "
        "{$0, $1, $2, $3}, [$4], $5;",
        "=r,=r,=r,=r,l,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)


@dsl_user_op
def load_x4(address: Int64, *, loc=None, ip=None) -> Words4:
    """16 bytes an earlier kernel wrote (a coherent load, kept after the
    grid-dependency wait)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)


@dsl_user_op
def load_bf16(address: Int64, *, loc=None, ip=None) -> Float32:
    """One BF16 value an earlier kernel wrote, widened to FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w;\n\tld.global.u16 w, [$1];\n\t"
            "shl.b32 w, w, 16;\n\tmov.b32 $0, w;\n\t}",
            "=f,l",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_bf16(address: Int64, value: Float32, *, loc=None, ip=None) -> None:
    """Store `value` rounded to nearest BF16."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16_16816(
    acc: Floats4, a: Words4, b: tuple[Uint32, Uint32], *, loc=None, ip=None
) -> Floats4:
    """`acc + A B` by one `mma.sync.aligned.m16n8k16` (BF16 inputs, FP32
    accumulator) in the standard fragment layouts."""
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in acc]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in a]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in b],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};",
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(  # type: ignore[return-value]
        Float32(llvm.extractvalue(T.f32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


def mma_chunk(acc: Floats4, lo: Words4, hi: Words4, x: Words4) -> Floats4:
    """Accumulate one 32-wide k chunk of a 16 x 8 tile with two m16n8k16 MMAs.

    Lane `(g, t) = (lane / 4, lane % 4)` holds, as four 32-bit words, the
    eight consecutive BF16 values at `k = 8 t .. 8 t + 7` of the chunk: of
    A rows `g` (`lo`) and `g + 8` (`hi`) and of B column `g` (`x`).
    Words 0-1 feed the first MMA and words 2-3 the second. A and B undergo the
    same permutation of k, so every dot product is unchanged, and each operand
    comes from one contiguous 16-byte load.
    """
    acc = mma_bf16_16816(acc, (lo[0], hi[0], lo[1], hi[1]), (x[0], x[1]))
    return mma_bf16_16816(acc, (lo[2], hi[2], lo[3], hi[3]), (x[2], x[3]))


@dsl_user_op
def store_shared_f32x4(address: Int32, values: Floats4, *, loc=None, ip=None) -> None:
    """Store four FP32 values at a 16-byte aligned shared-memory address."""
    _asm(
        None,
        [Int32(address).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_shared_f32(address: Int32, value: Float32, *, loc=None, ip=None) -> None:
    """Store one FP32 value in this CTA's shared memory."""
    _asm(
        None,
        [
            Int32(address).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_shared_f32(address: Int32, *, loc=None, ip=None) -> Float32:
    """One FP32 value from this CTA's shared memory."""
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
def load_shared_x4(address: Int32, *, loc=None, ip=None) -> Words4:
    """16 bytes from a 16-byte aligned shared-memory address."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(address).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )
    return _words(loaded, loc, ip)


@dsl_user_op
def load_cluster_f32(address: Int32, cta_rank: Int32, *, loc=None, ip=None) -> Float32:
    """One FP32 value at `address` in the shared memory of the cluster's CTA
    `cta_rank` (distributed shared memory)."""
    return Float32(
        _asm(
            T.f32(),
            [
                Int32(address).ir_value(loc=loc, ip=ip),
                Int32(cta_rank).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 a;\n\tmapa.shared::cluster.u32 a, $1, $2;\n\t"
            "ld.shared::cluster.f32 $0, [a];\n\t}",
            "=f,r,r",
            side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def copy_async_16(shared: Int32, address: Int64, *, loc=None, ip=None) -> None:
    """Copy 16 bytes from global to shared memory through L2 (`cp.async.cg`,
    coherent: the source may come from an earlier kernel)."""
    _asm(
        None,
        [
            Int32(shared).ir_value(loc=loc, ip=ip),
            Int64(address).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def copy_async_wait_all(*, loc=None, ip=None) -> None:
    """Wait for this thread's outstanding `cp.async` copies."""
    _asm(None, [], "cp.async.wait_all;", "", side_effects=True, loc=loc, ip=ip)
