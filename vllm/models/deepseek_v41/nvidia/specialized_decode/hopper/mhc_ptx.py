# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the Hopper mHC mixes kernel (`mhc_cute`): cp.async, shared
and distributed shared memory, tf32 `mma.sync`."""

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


def _v(x, cls, loc, ip):
    return cls(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def smem_u32(ptr, *, loc=None, ip=None) -> Int32:
    """The 32-bit shared-memory address of a smem pointer."""
    return Int32(ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def cp_async_16(dst, src, *, loc=None, ip=None):
    """16-byte cp.async global -> shared, L2 only."""
    _asm(
        None,
        [_v(dst, Int32, loc, ip), _v(src, Int64, loc, ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None):
    _asm(None, [], "cp.async.wait_all;", "", loc, ip, side_effects=True)


@dsl_user_op
def sts128_f32(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, Int32, loc, ip)] + [_v(t, Float32, loc, ip) for t in (a, b, c, d)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def cluster_sync(*, loc=None, ip=None):
    """barrier.cluster arrive (release) + wait (acquire), all threads."""
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def cluster_rank(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc, ip))


@dsl_user_op
def mma_tf32_1688(c, a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    """D = A B + C, m16n8k8, TF32 inputs (FP32 bit patterns), FP32
    accumulators (4 floats)."""
    r = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_v(t, Float32, loc, ip) for t in c]
        + [_v(t, Uint32, loc, ip) for t in (a0, a1, a2, a3, b0, b1)],
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};",
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        loc,
        ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def u32_as_f32(v, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_v(v, Uint32, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip)
    )


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "barrier.cluster.arrive.relaxed.aligned;",
        "",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.aligned;", "", loc, ip, side_effects=True)


@dsl_user_op
def tf32_rna(v, *, loc=None, ip=None) -> Uint32:
    """FP32 bits rounded to TF32 (nearest), as the TF32 GEMMs that round their
    FP32 operands (DeepGEMM's prenorm GEMM) see them; `mma.sync` itself
    ignores the low mantissa bits."""
    return Uint32(
        _asm(
            T.i32(),
            [_v(v, Uint32, loc, ip)],
            "cvt.rna.tf32.f32 $0, $1;",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", loc, ip, side_effects=True)
