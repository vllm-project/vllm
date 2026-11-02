# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL device primitives for Flint's small-M BF16 x NVFP4/BF16 GEMV.

Shared with other Flint kernels; keep signatures stable. A-fragment layout
(``mma.sync`` m16n8k16, swap-AB: weight rows are the MMA M dimension): lane
``(g, t) = (lane // 4, lane % 4)`` of k-tile ``j`` within a 64-wide K chunk
covers ``k = 64c + 16t + 4j + {0..3}`` of rows ``g`` and ``g + 8``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


def ptx(code, constraints, operands, results=None, *, effects=True, loc=None, ip=None):
    if results is None:
        ty = None
    elif isinstance(results, tuple):
        ty = llvm.StructType.get_literal([r.mlir_type for r in results])
    else:
        ty = results.mlir_type
    value = llvm.inline_asm(
        ty,
        [op.ir_value(loc=loc, ip=ip) for op in operands],
        code,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if results is None:
        return None
    if isinstance(results, tuple):
        return tuple(
            r(llvm.extractvalue(r.mlir_type, value, [i], loc=loc, ip=ip))
            for i, r in enumerate(results)
        )
    return results(value)


@dsl_user_op
def smem_addr(ptr, *, loc=None, ip=None):
    return Uint32(ptr.toint(loc=loc, ip=ip))


# Loads are volatile so that every weight load is issued up front, in
# program order, before the dependent math.
@dsl_user_op
def ld_w8(addr: Int64, *, loc=None, ip=None):
    return list(
        ptx(
            "ld.global.nc.L1::no_allocate.v2.u32 {$0, $1}, [$2];",
            "=r,=r,l",
            [addr],
            (Uint32,) * 2,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_w8p(addr: Int64, *, loc=None, ip=None):
    # ``ld_w8`` with a 128-byte L2 prefetch: a lane's later chunks of the same
    # row follow in 32-byte steps.
    return list(
        ptx(
            "ld.global.nc.L1::no_allocate.L2::128B.v2.u32 {$0, $1}, [$2];",
            "=r,=r,l",
            [addr],
            (Uint32,) * 2,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_w16(addr: Int64, *, loc=None, ip=None):
    return list(
        ptx(
            "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
            "=r,=r,=r,=r,l",
            [addr],
            (Uint32,) * 4,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_u8(addr: Int64, *, loc=None, ip=None):
    return ptx(
        "{\n.reg .b16 h;\nld.global.nc.u8 h, [$1];\ncvt.u32.u16 $0, h;\n}",
        "=r,l",
        [addr],
        Uint32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_x16(addr: Int64, pred: Int32, *, loc=None, ip=None):
    # 16 bf16 activations, or zeros for padded tokens.
    return list(
        ptx(
            "{\n.reg .pred p;\nsetp.ne.b32 p, $9, 0;\n"
            "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;\n"
            "mov.b32 $4, 0; mov.b32 $5, 0; mov.b32 $6, 0; mov.b32 $7, 0;\n"
            "@p ld.global.v4.u32 {$0, $1, $2, $3}, [$8];\n"
            "@p ld.global.v4.u32 {$4, $5, $6, $7}, [$8+16];\n}",
            "=r,=r,=r,=r,=r,=r,=r,=r,l,r",
            [addr, pred],
            (Uint32,) * 8,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def scale_pair(sc: Uint32, *, loc=None, ip=None):
    # E4M3 byte -> BF16x2 (s, s); exact.
    return ptx(
        "{\n.reg .b16 h;\n.reg .b8 b;\ncvt.u16.u32 h, $1;\n"
        "mov.b16 {b, _}, h;\nmov.b16 h, {b, b};\n"
        "cvt.rn.bf16x2.e4m3x2 $0, h;\n}",
        "=r,r",
        [sc],
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def decode(
    lo: Uint32,
    hi: Uint32,
    j: cutlass.Constexpr,
    sg: Uint32,
    sg8: Uint32,
    *,
    loc=None,
    ip=None,
):
    # A fragment of k-tile j: bytes (2j, 2j+1) % 4 of the row-g word `lo`
    # and the row-(g+8) word `hi`, scaled exactly in BF16.
    b = (2 * j) % 4
    return list(
        ptx(
            "{\n.reg .b8 b0, b1, b2, b3, c0, c1, c2, c3;\n"
            "mov.b32 {b0, b1, b2, b3}, $4;\nmov.b32 {c0, c1, c2, c3}, $5;\n"
            f"cvt.rn.bf16x2.e2m1x2 $0, b{b};\n"
            f"cvt.rn.bf16x2.e2m1x2 $1, c{b};\n"
            f"cvt.rn.bf16x2.e2m1x2 $2, b{b + 1};\n"
            f"cvt.rn.bf16x2.e2m1x2 $3, c{b + 1};\n"
            "mul.rn.bf16x2 $0, $0, $6;\nmul.rn.bf16x2 $1, $1, $7;\n"
            "mul.rn.bf16x2 $2, $2, $6;\nmul.rn.bf16x2 $3, $3, $7;\n}",
            "=r,=r,=r,=r,r,r,r,r",
            [lo, hi, sg, sg8],
            (Uint32,) * 4,
            effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mma(c, a, b0: Uint32, b1: Uint32, *, loc=None, ip=None):
    return list(
        ptx(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};",
            "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
            [*c, *a, b0, b1],
            (Float32,) * 4,
            effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_out(addr: Int64, v: Float32, pred: Int32, *, loc=None, ip=None):
    ptx(
        "{\n.reg .pred p;\n.reg .b16 h;\nsetp.ne.b32 p, $2, 0;\n"
        "cvt.rn.bf16.f32 h, $1;\n@p st.global.b16 [$0], h;\n}",
        "l,f,r",
        [addr, v, pred],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_dsmem_f32(smem_ptr, v: Float32, *, loc=None, ip=None):
    remote = cute.arch.map_dsmem_ptr(smem_ptr, Int32(0), loc=loc, ip=ip)
    addr = Int32(remote.toint(loc=loc, ip=ip))
    ptx("st.shared::cluster.f32 [$0], $1;", "r,f", [addr, v], loc=loc, ip=ip)


@dsl_user_op
def silu_mul(g: Float32, u: Float32, *, loc=None, ip=None):
    # g / (1 + expf(-g)) * u in fp32 (rounded once by the caller's store), with
    # CUDA's precise expf (libdevice __nv_expf) and an IEEE division.
    return ptx(
        "{\n.reg .b32 x, t, j, f, e, sc, d, a;\n"
        "neg.f32 x, $1;\n"
        "fma.rn.ftz.f32 t, x, 0f3BBB989D, 0f3F000000;\n"
        "cvt.ftz.sat.f32.f32 t, t;\n"
        "fma.rm.ftz.f32 j, t, 0f437C0000, 0f4B400001;\n"
        "add.f32 f, j, 0fCB40007F;\n"
        "neg.f32 f, f;\n"
        "fma.rn.ftz.f32 f, x, 0f3FB8AA3B, f;\n"
        "fma.rn.ftz.f32 f, x, 0f32A57060, f;\n"
        "shl.b32 sc, j, 23;\n"
        "ex2.approx.ftz.f32 e, f;\n"
        "fma.rn.f32 d, e, sc, 0f3F800000;\n"
        "div.rn.f32 a, $1, d;\n"
        "mul.f32 $0, a, $2;\n}",
        "=f,f,f",
        [g, u],
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def round_bf16(v: Float32, *, loc=None, ip=None):
    return ptx(
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}",
        "=f,f",
        [v],
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    ptx(
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        [Uint32(addr), Float32(a)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return ptx(
        "ld.shared.f32 $0, [$1];",
        "=f,r,~{memory}",
        [Uint32(addr)],
        Float32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_bf16(addr, a, *, loc=None, ip=None):
    ptx(
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; st.global.b16 [$0], h; }",
        "l,f,~{memory}",
        [Int64(addr), Float32(a)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_u32x4(addr, a, b, c, d, *, loc=None, ip=None):
    ptx(
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};",
        "l,r,r,r,r,~{memory}",
        [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bf16x2(lo, hi, *, loc=None, ip=None):
    return ptx(
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        [Float32(hi), Float32(lo)],
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fma(a, b, c, *, loc=None, ip=None):
    return ptx(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [Float32(a), Float32(b), Float32(c)],
        Float32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@cute.jit
def load_rows(
    w_base,
    s_base,
    row,
    KC: cutlass.Constexpr,
    CPW: cutlass.Constexpr,
    c_begin,
    t,
    qs,
    ss,
    q_off: cutlass.Constexpr,
    s_off: cutlass.Constexpr,
    PREFETCH: cutlass.Constexpr = False,
):
    """NVFP4 weight words / scale bytes of rows (row, row + 8) for CPW chunks
    from ``c_begin`` (ModelOpt weight, 128x4-swizzled scales)."""
    kb = KC * 32
    wbase = w_base + Int64(row) * kb + c_begin * 32 + t * 8
    row8 = row + 8
    sb_g = (
        s_base
        + Int64(row // 128) * (KC * 512)
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + t
        + c_begin * 512
    )
    sb_g8 = (
        s_base
        + Int64(row8 // 128) * (KC * 512)
        + (row8 % 32) * 16
        + ((row8 % 128) // 32) * 4
        + t
        + c_begin * 512
    )
    for c in cutlass.range_constexpr(CPW):
        if cutlass.const_expr(PREFETCH):
            qa = ld_w8p(wbase + c * 32)
            qb = ld_w8p(wbase + 8 * kb + c * 32)
        else:
            qa = ld_w8(wbase + c * 32)
            qb = ld_w8(wbase + 8 * kb + c * 32)
        q0 = q_off + c * 4
        qs[q0 + 0] = qa[0]
        qs[q0 + 1] = qa[1]
        qs[q0 + 2] = qb[0]
        qs[q0 + 3] = qb[1]
        ss[s_off + c * 2] = ld_u8(sb_g + c * 512)
        ss[s_off + c * 2 + 1] = ld_u8(sb_g8 + c * 512)


def alloc_reduce(smem, NA, WARPS, SPLIT):
    red = smem.allocate_tensor(
        Float32,
        cute.make_layout((WARPS, NA, 32), stride=(NA * 32, 32, 1)),
        byte_alignment=16,
    )
    part = smem.allocate_tensor(
        Float32,
        cute.make_layout((SPLIT, NA, 32), stride=(NA * 32, 32, 1)),
        byte_alignment=16,
    )
    return red, part


@cute.jit
def cta_reduce(
    acc,
    NA: cutlass.Constexpr,
    WARPS: cutlass.Constexpr,
    SPLIT: cutlass.Constexpr,
    warp,
    lane,
    rank,
    red,
    part,
):
    """Sum each lane's NA accumulators over the CTA's warps and the cluster's
    CTAs (fixed order) into warp 0 of cluster rank 0. With SPLIT > 1 the
    caller must have issued ``cluster_arrive_relaxed``."""
    for i in cutlass.range_constexpr(NA):
        red[warp, i, lane] = acc[i]
    cute.arch.sync_threads()
    if cutlass.const_expr(SPLIT > 1):
        cute.arch.cluster_wait()
    if warp == 0:
        for i in cutlass.range_constexpr(NA):
            v = acc[i]
            for w2 in cutlass.range_constexpr(1, WARPS):
                v = v + red[w2, i, lane]
            acc[i] = v
        if cutlass.const_expr(SPLIT > 1):
            for i in cutlass.range_constexpr(NA):
                st_dsmem_f32(part.iterator + (rank * NA + i) * 32 + lane, acc[i])
    if cutlass.const_expr(SPLIT > 1):
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp == 0 and rank == 0:
            for i in cutlass.range_constexpr(NA):
                v = part[0, i, lane]
                for r2 in cutlass.range_constexpr(1, SPLIT):
                    v = v + part[r2, i, lane]
                acc[i] = v
