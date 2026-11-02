# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PTX building blocks of the Hopper low-batch MoE kernel (``moe_cute``)."""

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


def _u32(v, loc, ip):
    return Uint32(v).ir_value(loc=loc, ip=ip)


def _i32(v, loc, ip):
    return Int32(v).ir_value(loc=loc, ip=ip)


def _i64(v, loc, ip):
    return Int64(v).ir_value(loc=loc, ip=ip)


def _f32(v, loc, ip):
    return Float32(v).ir_value(loc=loc, ip=ip)


def _struct(n, ty):
    return llvm.StructType.get_literal([ty] * n)


def _unpack(res, n, ty, cls, loc, ip):
    return [cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n)]


@dsl_user_op
def smem_addr(ptr, *, loc=None, ip=None) -> Int32:
    """32-bit shared-window address of a shared-memory cute Pointer."""
    return Int32(ptr.toint(loc=loc, ip=ip))


@dsl_user_op
def lds128(addr, *, loc=None, ip=None):
    r = _asm(
        _struct(4, T.i32()),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def lds32(addr, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i32(addr, loc, ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def lds_u16(addr, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip)],
            "{\n.reg .u16 h;\nld.shared.u16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def sts32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _u32(v, loc, ip)],
        "st.shared.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def sts_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _f32(v, loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def ldg128(addr, *, loc=None, ip=None):
    """16-byte global load through L1 (data written by an earlier kernel)."""
    r = _asm(
        _struct(4, T.i32()),
        [_i64(addr, loc, ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i64(addr, loc, ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def ldg_cg_i32(addr, *, loc=None, ip=None) -> Int32:
    """L2 load (no L1), for data other CTAs of this kernel wrote."""
    return Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip)],
            "ld.global.cg.s32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def ldg_cg_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i64(addr, loc, ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def stg_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _f32(v, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def stg_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _u32(v, loc, ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def stg_i32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "st.global.s32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def stg_u32x4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip)] + [_u32(v, loc, ip) for v in (a, b, c, d)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def stg_u16(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _u32(v, loc, ip)],
        "{\n.reg .u16 h;\ncvt.u16.u32 h, $1;\nst.global.u16 [$0], h;\n}",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def atom_add_acqrel_gpu(addr, v, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip), _i32(v, loc, ip)],
            "atom.add.acq_rel.gpu.s32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def fence_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip, side_effects=True)


@dsl_user_op
def mbar_init(addr, count, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(count, loc, ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def fence_mbar_init(*, loc=None, ip=None):
    _asm(
        None, [], "fence.mbarrier_init.release.cluster;", "", loc, ip, side_effects=True
    )


@dsl_user_op
def mbar_arrive_expect_tx(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(nbytes, loc, ip)],
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def mbar_arrive(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip)],
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
        "r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def mbar_wait(addr, phase, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(phase, loc, ip)],
        "{\n.reg .pred p;\nWAIT_${:uid}:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT_${:uid};\n}",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def bulk_g2s(dst, src, nbytes, mbar, policy, *, loc=None, ip=None):
    """cp.async.bulk global->shared, completion on ``mbar``; L2 cache hint."""
    _asm(
        None,
        [
            _i32(dst, loc, ip),
            _i64(src, loc, ip),
            _i32(nbytes, loc, ip),
            _i32(mbar, loc, ip),
            _i64(policy, loc, ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        " [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def evict_first_policy(*, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def named_barrier(bar_id, nthreads, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(bar_id, loc, ip), _i32(nthreads, loc, ip)],
        "bar.sync $0, $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def dequant_e2m1x8(qb, eo, *, loc=None, ip=None):
    """Humming fused MXFP4 -> E4M3: 8 codes of word ``qb`` (sign bits
    interleaved, see hlayout.py) with exponent offset ``eo`` -> 2 words of
    E4M3 (elements 0-3, 4-7) = e2m1 * 2^(eo - 6)."""
    r = _asm(
        _struct(2, T.i32()),
        [_u32(qb, loc, ip), _u32(eo, loc, ip)],
        "{\n.reg .b32 b1, b2, s, m0, m1, t;\n"
        "mad.lo.u32 b1, $3, 0x08080800, 0x0C080000;\n"
        "mad.lo.u32 b2, $3, 0x08080808, 0x1C181410;\n"
        "and.b32 s, $2, 0x77777777;\n"
        "prmt.b32 m0, b1, b2, s;\n"
        "shr.u32 s, s, 16;\n"
        "prmt.b32 m1, b1, b2, s;\n"
        "shl.b32 t, $2, 4;\n"
        "lop3.b32 $0, t, 0x80808080, m0, 0xEA;\n"
        "lop3.b32 $1, $2, 0x80808080, m1, 0xEA;\n}",
        "=r,=r,r,r",
        loc,
        ip,
    )
    return _unpack(r, 2, T.i32(), Uint32, loc, ip)


@dsl_user_op
def cvt_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """Four f32 -> four saturating E4M3 bytes (a lowest)."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(v, loc, ip) for v in (a, b, c, d)],
            "{\n.reg .b16 lo, hi;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
            "mov.b32 $0, {lo, hi};\n}",
            "=r,f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def cvt_bf16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Two f32 -> bf16x2 (a in the low half)."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def swiglu(gate, up, limit, *, loc=None, ip=None) -> Float32:
    """min(gate, limit) * sigmoid(min(gate, limit)) * clamp(up, -limit, limit)."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(gate, loc, ip), _f32(up, loc, ip), _f32(limit, loc, ip)],
            "{\n.reg .f32 g, u, e, nl;\n"
            "min.f32 g, $1, $3;\n"
            "neg.f32 nl, $3;\n"
            "min.f32 u, $2, $3;\nmax.f32 u, u, nl;\n"
            "mul.f32 e, g, 0fBFB8AA3B;\n"
            "ex2.approx.ftz.f32 e, e;\n"
            "add.f32 e, e, 0f3F800000;\n"
            "rcp.approx.ftz.f32 e, e;\n"
            "mul.f32 e, e, g;\n"
            "mul.f32 $0, e, u;\n}",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fabs(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_f32(x, loc, ip)], "abs.f32 $0, $1;", "=f,f", loc, ip)
    )


@dsl_user_op
def fmax(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def rcp(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_f32(x, loc, ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc, ip)
    )


@dsl_user_op
def shfl_xor_f32(x, m, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(x, loc, ip), _i32(m, loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def griddep_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc, ip, side_effects=True)


@dsl_user_op
def griddep_launch(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc, ip, side_effects=True)


@dsl_user_op
def wgmma_fence(*, loc=None, ip=None):
    _asm(None, [], "wgmma.fence.sync.aligned;", "", loc, ip, side_effects=True)


@dsl_user_op
def wgmma_commit(*, loc=None, ip=None):
    _asm(None, [], "wgmma.commit_group.sync.aligned;", "", loc, ip, side_effects=True)


@dsl_user_op
def fence_proxy_async_smem(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip, side_effects=True)


def _fence_regs_impl(vals, loc, ip):
    n = len(vals)
    cons = ",".join(["=f"] * n + [str(i) for i in range(n)])
    r = _asm(
        _struct(n, T.f32()),
        [_f32(v, loc, ip) for v in vals],
        "",
        cons,
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, n, T.f32(), Float32, loc, ip)


@dsl_user_op
def fence_regs(vals, *, loc=None, ip=None):
    """Pin accumulator registers (ordering point after wgmma wait)."""
    return _fence_regs_impl(list(vals), loc, ip)


@dsl_user_op
def wgmma_rs(n, acc, a0, a1, a2, a3, desc, scale_d, *, loc=None, ip=None):
    """Wgmma m64n{n}k32 e4m3 x e4m3 -> f32, A from registers, B by descriptor.
    ``acc``: n/2 floats; returns the n/2 updated accumulators."""
    nr = n // 2
    outs = ",".join(f"${i}" for i in range(nr))
    a = ",".join(f"${nr + i}" for i in range(4))
    d = f"${nr + 4}"
    sc = f"${nr + 5}"
    text = (
        "{\n.reg .pred p;\n"
        f"setp.ne.b32 p, {sc}, 0;\n"
        f"wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 "
        f"{{{outs}}}, {{{a}}}, {d}, p, 1, 1;\n}}"
    )
    cons = ",".join(["=f"] * nr + ["r"] * 4 + ["l", "r"] + [str(i) for i in range(nr)])
    ops = (
        [_u32(v, loc, ip) for v in (a0, a1, a2, a3)]
        + [_i64(desc, loc, ip), _i32(scale_d, loc, ip)]
        + [_f32(v, loc, ip) for v in acc]
    )
    r = _asm(_struct(nr, T.f32()), ops, text, cons, loc, ip, side_effects=True)
    return _unpack(r, nr, T.f32(), Float32, loc, ip)


@dsl_user_op
def sts128(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip)] + [_u32(v, loc, ip) for v in (a, b, c, d)],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


def make_desc_nosw(addr, lbo, sbo):
    """Wgmma smem descriptor, no swizzle (layout type 0)."""
    return (
        Int64((addr >> 4) & 0x3FFF)
        | (Int64((lbo >> 4) & 0x3FFF) << 16)
        | (Int64((sbo >> 4) & 0x3FFF) << 32)
    )


@dsl_user_op
def tma_load_3d(dst, tmap, c0, c1, c2, mbar, policy, *, loc=None, ip=None):
    _asm(
        None,
        [
            _i32(dst, loc, ip),
            _i64(tmap, loc, ip),
            _i32(c0, loc, ip),
            _i32(c1, loc, ip),
            _i32(c2, loc, ip),
            _i32(mbar, loc, ip),
            _i64(policy, loc, ip),
        ],
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [$1, {$2, $3, $4}], [$5], $6;",
        "r,l,r,r,r,r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def tma_load_4d(dst, tmap, c0, c1, c2, c3, mbar, policy, *, loc=None, ip=None):
    _asm(
        None,
        [
            _i32(dst, loc, ip),
            _i64(tmap, loc, ip),
            _i32(c0, loc, ip),
            _i32(c1, loc, ip),
            _i32(c2, loc, ip),
            _i32(c3, loc, ip),
            _i32(mbar, loc, ip),
            _i64(policy, loc, ip),
        ],
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [$1, {$2, $3, $4, $5}], [$6], $7;",
        "r,l,r,r,r,r,r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def prefetch_tmap(tmap, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(tmap, loc, ip)],
        "prefetch.tensormap [$0];",
        "l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def wgmma_wait_n(n, *, loc=None, ip=None):
    _asm(
        None, [], f"wgmma.wait_group.sync.aligned {n};", "", loc, ip, side_effects=True
    )


@dsl_user_op
def ld_acquire_i32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip)],
            "ld.acquire.gpu.global.s32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _asm(
        None, [_i32(ns, loc, ip)], "nanosleep.u32 $0;", "r", loc, ip, side_effects=True
    )


@dsl_user_op
def ldg_cg_f32x2(addr, *, loc=None, ip=None):
    r = _asm(
        _struct(2, T.f32()),
        [_i64(addr, loc, ip)],
        "ld.global.cg.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, 2, T.f32(), Float32, loc, ip)


@dsl_user_op
def atom_add_relaxed(addr, v, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip), _i32(v, loc, ip)],
            "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def shfl_up_i32(x, d, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip), _i32(d, loc, ip)],
            "shfl.sync.up.b32 $0, $1, $2, 0x0, 0xffffffff;",
            "=r,r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def bf16x2_to_f32x2(v, *, loc=None, ip=None):
    r = _asm(
        _struct(2, T.f32()),
        [_u32(v, loc, ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xFFFF0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc,
        ip,
    )
    return _unpack(r, 2, T.f32(), Float32, loc, ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.global;", "", loc, ip, side_effects=True)


@dsl_user_op
def lds64(addr, *, loc=None, ip=None):
    r = _asm(
        _struct(2, T.i32()),
        [_i32(addr, loc, ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, 2, T.i32(), Uint32, loc, ip)


@dsl_user_op
def prmt(a, b, sel, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip), _u32(sel, loc, ip)],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def e8m0_to_f32(e, *, loc=None, ip=None) -> Float32:
    """2^(e - 127) for a UE8M0 byte e in the low bits (e = 0 -> 0)."""
    return Float32(
        _asm(
            T.f32(),
            [_u32(e, loc, ip)],
            "{\n.reg .b32 t;\nand.b32 t, $1, 0xFF;\nshl.b32 t, t, 23;\n"
            "mov.b32 $0, t;\n}",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def lds_acquire(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip)],
            "ld.acquire.cta.shared::cta.u32 $0, [$1];",
            "=r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def smem_add_release(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(v, loc, ip)],
        "red.release.cta.shared::cta.add.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def sts_release(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(v, loc, ip)],
        "st.release.cta.shared::cta.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def tma_load_2d(dst, tmap, c0, c1, mbar, policy, *, loc=None, ip=None):
    _asm(
        None,
        [
            _i32(dst, loc, ip),
            _i64(tmap, loc, ip),
            _i32(c0, loc, ip),
            _i32(c1, loc, ip),
            _i32(mbar, loc, ip),
            _i64(policy, loc, ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [$1, {$2, $3}], [$4], $5;",
        "r,l,r,r,r,l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def prefetch_l2_bulk(addr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(nbytes, loc, ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def ldg_i64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_i64(addr, loc, ip)],
            "ld.global.nc.s64 $0, [$1];",
            "=l,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def red_add_gpu(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def cvt_e4m3_word(w, *, loc=None, ip=None):
    """Humming W8A16 word [k0,k8,k1,k9] (e4m3) -> (f16x2{k0,k1}, f16x2{k8,k9}), exact."""  # noqa: E501
    r = _asm(
        _struct(2, T.i32()),
        [_u32(w, loc, ip)],
        "{\n.reg .b32 p;\n.reg .b16 l, h;\nprmt.b32 p, $2, $2, 0x3120;\n"
        "mov.b32 {l, h}, p;\ncvt.rn.f16x2.e4m3x2 $0, l;\ncvt.rn.f16x2.e4m3x2 $1, h;\n}",
        "=r,=r,r",
        loc,
        ip,
    )
    return _unpack(r, 2, T.i32(), Uint32, loc, ip)


@dsl_user_op
def f16x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Two f32 -> f16x2 (a in the low half)."""
    return Uint32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def bf16x2_to_f16x2(v, *, loc=None, ip=None) -> Uint32:
    """bf16x2 -> f16x2 (via f32; values of a post-RMSNorm activation fit f16)."""
    return Uint32(
        _asm(
            T.i32(),
            [_u32(v, loc, ip)],
            "{\n.reg .b32 lo, hi;\n.reg .f32 a, b;\nshl.b32 lo, $1, 16;\n"
            "and.b32 hi, $1, 0xFFFF0000;\nmov.b32 a, lo;\nmov.b32 b, hi;\n"
            "cvt.rn.f16x2.f32 $0, b, a;\n}",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def round_bf16(x, *, loc=None, ip=None) -> Float32:
    """f32 rounded to bf16 precision (RNE), returned as f32."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(x, loc, ip)],
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def wgmma_rs_f16(n, acc, a0, a1, a2, a3, desc, scale_d, *, loc=None, ip=None):
    """Wgmma m64n{n}k16 f16 x f16 -> f32, A from registers, B (K-major) by descriptor."""  # noqa: E501
    nr = n // 2
    outs = ",".join(f"${i}" for i in range(nr))
    a = ",".join(f"${nr + i}" for i in range(4))
    d = f"${nr + 4}"
    sc = f"${nr + 5}"
    text = (
        "{\n.reg .pred p;\n"
        f"setp.ne.b32 p, {sc}, 0;\n"
        f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.f16.f16 "
        f"{{{outs}}}, {{{a}}}, {d}, p, 1, 1, 0;\n}}"
    )
    cons = ",".join(["=f"] * nr + ["r"] * 4 + ["l", "r"] + [str(i) for i in range(nr)])
    ops = (
        [_u32(v, loc, ip) for v in (a0, a1, a2, a3)]
        + [_i64(desc, loc, ip), _i32(scale_d, loc, ip)]
        + [_f32(v, loc, ip) for v in acc]
    )
    r = _asm(_struct(nr, T.f32()), ops, text, cons, loc, ip, side_effects=True)
    return _unpack(r, nr, T.f32(), Float32, loc, ip)


@dsl_user_op
def mapa(addr, rank, *, loc=None, ip=None) -> Int32:
    """shared::cta address -> the same offset in CTA ``rank`` of the cluster."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip), _i32(rank, loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_cluster_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _f32(v, loc, ip)],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def mbar_arrive_cluster(addr, *, loc=None, ip=None):
    """Arrive (release, cluster scope) on an mbarrier given by a shared::cluster
    address (possibly in a peer CTA)."""
    _asm(
        None,
        [_i32(addr, loc, ip)],
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def mbar_wait_cluster(addr, phase, *, loc=None, ip=None):
    """Parity wait with cluster-scope acquire (arrivals came from a peer CTA)."""
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(phase, loc, ip)],
        "{\n.reg .pred p;\nWAITC_${:uid}:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAITC_${:uid};\n}",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def fence_mbar_init_cluster(*, loc=None, ip=None):
    _asm(
        None, [], "fence.mbarrier_init.release.cluster;", "", loc, ip, side_effects=True
    )


@dsl_user_op
def cluster_sync_all(*, loc=None, ip=None):
    """barrier.cluster arrive (release) + wait (acquire), every thread."""
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;",
        "",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def score_of(logit, *, loc=None, ip=None) -> Float32:
    """sqrt(softplus(x)), softplus(x) = x for x > 20 else log(1 + exp(x))."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(logit).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 e, l;\n.reg .pred p;\n"
            "mul.f32 e, $1, 0f3FB8AA3B;\nex2.approx.ftz.f32 e, e;\n"
            "add.f32 e, e, 0f3F800000;\nlg2.approx.ftz.f32 l, e;\n"
            "mul.f32 l, l, 0f3F317218;\n"
            "setp.gt.f32 p, $1, 0f41A00000;\nselp.f32 l, $1, l, p;\n"
            "sqrt.approx.ftz.f32 $0, l;\n}",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def shfl_xor_i32(x, m, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int32(x).ir_value(loc=loc, ip=ip), Int32(m).ir_value(loc=loc, ip=ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def shfl_idx_f32(x, src, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Int32(src).ir_value(loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def smem_atom_add(addr, v, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def smem_atom_or(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(v).ir_value(loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def popc(x, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            loc,
            ip,
        )
    )
