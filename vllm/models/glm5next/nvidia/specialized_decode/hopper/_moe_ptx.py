# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX helpers of the Hopper MoE kernels (`moe_router`, `moe_experts`).

Addresses are plain integers: 64-bit global addresses ("l") and 32-bit shared
window addresses ("r", see `smem_addr`)."""

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
def ldg128_cg(addr, *, loc=None, ip=None):
    """16-byte global load from L2, for data an earlier kernel writes. Under
    PDL that kernel can still be writing while this one starts, so the load
    must not use the non-coherent path (`.nc`), which assumes the data stays
    read-only for this kernel's whole lifetime."""
    r = _asm(
        _struct(4, T.i32()),
        [_i64(addr, loc, ip)],
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
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
    """cp.async.bulk global->shared, completion on `mbar`; L2 cache hint."""
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
    """Waits for the PDL primary grids; a memory clobber, so no load of their
    outputs moves above it."""
    _asm(None, [], "griddepcontrol.wait;", "~{memory}", loc, ip, side_effects=True)


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
def fence_regs(vals, *, loc=None, ip=None):
    """Pin accumulator registers (ordering point after wgmma wait)."""
    vals = list(vals)
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
def wgmma_rs(n, acc, a0, a1, a2, a3, desc, scale_d, *, loc=None, ip=None):
    """Wgmma m64n{n}k32 e4m3 x e4m3 -> f32, A from registers, B by descriptor.
    `acc`: n/2 floats; returns the n/2 updated accumulators."""
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
def mapa(addr, rank, *, loc=None, ip=None) -> Int32:
    """shared::cta address -> the same offset in CTA `rank` of the cluster."""
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
def smem_atom_or(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _u32(v, loc, ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def red_max_shared_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _u32(v, loc, ip)],
        "red.shared.max.u32 [$0], $1;",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def stg_u8(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _u32(v, loc, ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def f32_bits(x, *, loc=None, ip=None) -> Uint32:
    return Uint32(_asm(T.i32(), [_f32(x, loc, ip)], "mov.b32 $0, $1;", "=r,f", loc, ip))


@dsl_user_op
def bits_f32(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_u32(x, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip)
    )


@dsl_user_op
def sts_u8(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _u32(v, loc, ip)],
        "{\n.reg .u16 h;\ncvt.u16.u32 h, $1;\nst.shared.u8 [$0], h;\n}",
        "r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def lds_u8(addr, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip)],
            "{\n.reg .u16 h;\nld.shared.u8 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def ldg_cg_f32x4(addr, *, loc=None, ip=None):
    r = _asm(
        _struct(4, T.f32()),
        [_i64(addr, loc, ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def stg_u32x2(addr, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _u32(a, loc, ip), _u32(b, loc, ip)],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def mma_bf16(d, a, b, *, loc=None, ip=None):
    """`mma.sync` m16n8k16 BF16 x BF16 -> FP32: d 4 floats, a 4 words (A row),
    b 2 words (B column); returns 4 floats."""
    text = (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    cons = "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3"
    ops = (
        [_u32(v, loc, ip) for v in a]
        + [_u32(v, loc, ip) for v in b]
        + [_f32(v, loc, ip) for v in d]
    )
    r = _asm(_struct(4, T.f32()), ops, text, cons, loc, ip, side_effects=True)
    return _unpack(r, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def ldg128_na(addr, *, loc=None, ip=None):
    """16-byte read-only global load, no L1 allocation (streamed weights)."""
    r = _asm(
        _struct(4, T.i32()),
        [_i64(addr, loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
        side_effects=True,
    )
    return _unpack(r, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def ld_cluster_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i32(addr, loc, ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def cluster_rank(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc, ip))


@dsl_user_op
def redux_max_u32(x, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [_u32(x, loc, ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def sel_u32(c, a, b, *, loc=None, ip=None) -> Uint32:
    """`c != 0 ? a : b`."""
    return Uint32(
        _asm(
            T.i32(),
            [_i32(c, loc, ip), _u32(a, loc, ip), _u32(b, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def sel_f32(c, a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_i32(c, loc, ip), _f32(a, loc, ip), _f32(b, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def eq_u32(a, b, *, loc=None, ip=None) -> Int32:
    """1 if a == b else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_u32(a, loc, ip), _u32(b, loc, ip)],
            "{\n.reg .pred p;\nsetp.eq.u32 p, $1, $2;\nselp.s32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def score_key(logit, bias, *, loc=None, ip=None):
    """The `single_group_topk` scores of a sigmoid router: (the unbiased
    sigmoid, the twiddled key of sigmoid + bias). A non-finite logit or bias
    gives 0 and a -inf key; -0 keys as +0. The sigmoid is
    1 / (1 + 2^(-x log2 e)) (`rcp.rn`)."""
    r = _asm(
        _struct(2, T.i32()),
        [_f32(logit, loc, ip), _f32(bias, loc, ip)],
        "{\n.reg .pred fx, fb, ok, ng, z;\n.reg .f32 e, u, s;\n.reg .b32 bs, nb, k;\n"
        "testp.finite.f32 fx, $2;\ntestp.finite.f32 fb, $3;\nand.pred ok, fx, fb;\n"
        "mul.f32 e, $2, 0fBFB8AA3B;\nex2.approx.f32 e, e;\n"
        "add.f32 e, e, 0f3F800000;\nrcp.rn.f32 u, e;\n"
        "add.f32 s, u, $3;\nsetp.eq.f32 z, s, 0f00000000;\n@z mov.f32 s, 0f00000000;\n"
        "@!ok mov.f32 u, 0f00000000;\n@!ok mov.f32 s, 0fFF800000;\n"
        "mov.b32 bs, s;\nsetp.lt.s32 ng, bs, 0;\nnot.b32 nb, bs;\n"
        "or.b32 k, bs, 0x80000000;\n@ng mov.b32 k, nb;\n"
        "mov.b32 $0, u;\nmov.b32 $1, k;\n}",
        "=r,=r,f,f",
        loc,
        ip,
    )
    return _unpack(r, 2, T.i32(), Uint32, loc, ip)


@dsl_user_op
def div_rn(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def mul_rn(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def prefetch_l2(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip)],
        "prefetch.global.L2 [$0];",
        "l",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def minmax_key(ah, al, bh, bl, *, loc=None, ip=None):
    """Lexicographic unsigned (hi, lo) max and min: (maxh, maxl, minh, minl)."""
    r = _asm(
        _struct(4, T.i32()),
        [_u32(ah, loc, ip), _u32(al, loc, ip), _u32(bh, loc, ip), _u32(bl, loc, ip)],
        "{\n.reg .b64 a, b, x, n;\nmov.b64 a, {$5, $4};\nmov.b64 b, {$7, $6};\n"
        "max.u64 x, a, b;\nmin.u64 n, a, b;\n"
        "mov.b64 {$1, $0}, x;\nmov.b64 {$3, $2}, n;\n}",
        "=r,=r,=r,=r,r,r,r,r",
        loc,
        ip,
    )
    return _unpack(r, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def red_release_add(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc,
        ip,
        side_effects=True,
    )
