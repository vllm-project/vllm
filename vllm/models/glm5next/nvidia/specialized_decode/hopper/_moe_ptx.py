# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline PTX of the Hopper MoE kernels (`moe_router`, `moe_experts`) that
`..cutedsl.primitives` does not have, written with its `asm` decorator.

Addresses are plain integers: 64-bit global addresses ("l") and 32-bit shared
window addresses ("r", see `smem_addr`)."""

from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import dsl_user_op

from ..cutedsl.primitives import asm


@dsl_user_op
def smem_addr(ptr, *, loc=None, ip=None) -> Int32:
    """32-bit shared-window address of a shared-memory cute Pointer."""
    return Int32(ptr.toint(loc=loc, ip=ip))


# ------------------------------------------------------------ shared memory
@asm(Uint32)
def lds32(addr):
    return "ld.shared.u32 $0, [$1];", "=r,r", (Int32(addr),)


@asm(Float32)
def lds_f32(addr):
    return "ld.shared.f32 $0, [$1];", "=f,r", (Int32(addr),)


@asm(Uint32)
def lds_u8(addr):
    code = "{\n.reg .u16 h;\nld.shared.u8 h, [$1];\ncvt.u32.u16 $0, h;\n}"
    return code, "=r,r", (Int32(addr),)


@asm()
def sts32(addr, v):
    return "st.shared.u32 [$0], $1;", "r,r", (Int32(addr), Uint32(v))


@asm()
def sts_f32(addr, v):
    return "st.shared.f32 [$0], $1;", "r,f", (Int32(addr), Float32(v))


@asm()
def sts_u8(addr, v):
    code = "{\n.reg .u16 h;\ncvt.u16.u32 h, $1;\nst.shared.u8 [$0], h;\n}"
    return code, "r,r", (Int32(addr), Uint32(v))


@asm()
def smem_atom_or(addr, v):
    return "red.shared.or.b32 [$0], $1;", "r,r", (Int32(addr), Uint32(v))


@asm()
def red_max_shared_u32(addr, v):
    return "red.shared.max.u32 [$0], $1;", "r,r", (Int32(addr), Uint32(v))


@asm(Int32, pure=True)
def mapa(addr, rank):
    """shared::cta address -> the same offset in CTA `rank` of the cluster."""
    return "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", (Int32(addr), Int32(rank))


@asm(Float32)
def ld_cluster_f32(addr):
    return "ld.shared::cluster.f32 $0, [$1];", "=f,r", (Int32(addr),)


@asm(Int32, pure=True)
def cluster_rank():
    return "mov.u32 $0, %cluster_ctarank;", "=r", ()


# ------------------------------------------------------------ global memory
@asm((Uint32,) * 4)
def ldg128_cg(addr):
    """16-byte global load from L2, for data an earlier kernel writes. Under
    PDL that kernel can still be writing while this one starts, so the load
    must not use the non-coherent path (`.nc`), which assumes the data stays
    read-only for this kernel's whole lifetime."""
    return (
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        (Int64(addr),),
    )


@asm((Uint32,) * 4)
def ldg128_na(addr):
    """16-byte read-only global load, no L1 allocation (streamed weights)."""
    code = "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];"
    return code, "=r,=r,=r,=r,l", (Int64(addr),)


@asm(Float32)
def ldg_f32(addr):
    return "ld.global.nc.f32 $0, [$1];", "=f,l", (Int64(addr),)


@asm(Int32)
def ldg_cg_i32(addr):
    """L2 load (no L1), for data an earlier kernel or another CTA wrote."""
    return "ld.global.cg.s32 $0, [$1];", "=r,l", (Int64(addr),)


@asm(Float32)
def ldg_cg_f32(addr):
    return "ld.global.cg.f32 $0, [$1];", "=f,l", (Int64(addr),)


@asm((Float32,) * 4)
def ldg_cg_f32x4(addr):
    return (
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        (Int64(addr),),
    )


@asm()
def stg_f32(addr, v):
    return "st.global.f32 [$0], $1;", "l,f", (Int64(addr), Float32(v))


@asm()
def stg_i32(addr, v):
    return "st.global.s32 [$0], $1;", "l,r", (Int64(addr), Int32(v))


@asm()
def stg_u8(addr, v):
    return "st.global.u8 [$0], $1;", "l,r", (Int64(addr), Uint32(v))


@asm()
def stg_u32x2(addr, a, b):
    code = "st.global.v2.u32 [$0], {$1, $2};"
    return code, "l,r,r", (Int64(addr), Uint32(a), Uint32(b))


@asm()
def stg_u32x4(addr, a, b, c, d):
    code = "st.global.v4.u32 [$0], {$1, $2, $3, $4};"
    return code, "l,r,r,r,r", (Int64(addr), *(Uint32(v) for v in (a, b, c, d)))


@asm(Int32)
def ld_acquire_i32(addr):
    return "ld.acquire.gpu.global.s32 $0, [$1];", "=r,l", (Int64(addr),)


@asm(Int32)
def atom_add_acqrel_gpu(addr, v):
    code = "atom.add.acq_rel.gpu.s32 $0, [$1], $2;"
    return code, "=r,l,r", (Int64(addr), Int32(v))


@asm(Int32)
def atom_add_relaxed(addr, v):
    code = "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;"
    return code, "=r,l,r", (Int64(addr), Int32(v))


@asm()
def red_add_gpu(addr, v):
    return "red.relaxed.gpu.global.add.s32 [$0], $1;", "l,r", (Int64(addr), Int32(v))


@asm()
def red_release_add(addr, v):
    return "red.release.gpu.global.add.s32 [$0], $1;", "l,r", (Int64(addr), Int32(v))


# ----------------------------------------------------- fences and barriers
@asm()
def fence_gpu():
    return "fence.acq_rel.gpu;", "", ()


@asm()
def fence_proxy_async_global():
    return "fence.proxy.async.global;", "", ()


@asm()
def fence_proxy_async_shared():
    return "fence.proxy.async.shared::cta;", "", ()


@asm()
def griddep_wait():
    """Waits for the PDL primary grids; a memory clobber, so no load of their
    outputs moves above it."""
    return "griddepcontrol.wait;", "~{memory}", ()


@asm()
def griddep_launch():
    return "griddepcontrol.launch_dependents;", "", ()


@asm()
def cluster_sync_all():
    """barrier.cluster arrive (release) + wait (acquire), every thread."""
    return "barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;", "", ()


@asm()
def mbar_init(addr, count):
    code = "mbarrier.init.shared::cta.b64 [$0], $1;"
    return code, "r,r", (Int32(addr), Int32(count))


@asm()
def fence_mbar_init():
    return "fence.mbarrier_init.release.cluster;", "", ()


@asm()
def mbar_arrive_expect_tx(addr, nbytes):
    code = "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;"
    return code, "r,r", (Int32(addr), Int32(nbytes))


@asm()
def mbar_arrive(addr):
    return "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];", "r", (Int32(addr),)


@asm()
def mbar_wait(addr, phase):
    code = (
        "{\n.reg .pred p;\nWAIT_${:uid}:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT_${:uid};\n}"
    )
    return code, "r,r", (Int32(addr), Int32(phase))


# -------------------------------------------------------------- TMA, bulk
@asm(Int64)
def evict_first_policy():
    return "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", ()


@asm()
def bulk_g2s(dst, src, nbytes, mbar, policy):
    """cp.async.bulk global->shared, completion on `mbar`; L2 cache hint."""
    code = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [$1], $2, [$3], $4;"
    )
    operands = (Int32(dst), Int64(src), Int32(nbytes), Int32(mbar), Int64(policy))
    return code, "r,l,r,r,l", operands


@asm()
def tma_load_3d(dst, tmap, c0, c1, c2, mbar, policy):
    code = (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx"
        "::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4}], [$5], $6;"
    )
    coords = (Int32(c) for c in (c0, c1, c2))
    operands = (Int32(dst), Int64(tmap), *coords, Int32(mbar), Int64(policy))
    return code, "r,l,r,r,r,r,l", operands


@asm()
def tma_load_4d(dst, tmap, c0, c1, c2, c3, mbar, policy):
    code = (
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx"
        "::bytes.L2::cache_hint [$0], [$1, {$2, $3, $4, $5}], [$6], $7;"
    )
    coords = (Int32(c) for c in (c0, c1, c2, c3))
    operands = (Int32(dst), Int64(tmap), *coords, Int32(mbar), Int64(policy))
    return code, "r,l,r,r,r,r,r,l", operands


@asm()
def prefetch_tmap(tmap):
    return "prefetch.tensormap [$0];", "l", (Int64(tmap),)


@asm()
def tma_prefetch_3d(tmap, c0, c1, c2):
    """L2 prefetch of one TMA box (no shared memory, no barrier); unlike
    `primitives.prefetch_map`, the first coordinate is not fixed at 0."""
    code = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [$0, {$1, $2, $3}];"
    return code, "l,r,r,r", (Int64(tmap), Int32(c0), Int32(c1), Int32(c2))


# ------------------------------------------------------------------ wgmma
@asm()
def wgmma_fence():
    return "wgmma.fence.sync.aligned;", "", ()


@asm()
def wgmma_commit():
    return "wgmma.commit_group.sync.aligned;", "", ()


@asm()
def wgmma_wait_n(n):
    return f"wgmma.wait_group.sync.aligned {n};", "", ()


@asm()
def fence_regs(vals):
    """Pin accumulator registers (ordering point after wgmma wait)."""
    n = len(vals)
    constraints = ",".join(["=f"] * n + [str(i) for i in range(n)])
    return "", constraints, tuple(Float32(v) for v in vals), (Float32,) * n


def make_desc_nosw(addr, lbo, sbo):
    """Wgmma smem descriptor, no swizzle (layout type 0)."""
    return (
        Int64((addr >> 4) & 0x3FFF)
        | (Int64((lbo >> 4) & 0x3FFF) << 16)
        | (Int64((sbo >> 4) & 0x3FFF) << 32)
    )


def make_desc_sw128(addr):
    """Wgmma smem descriptor of a K-major tile of 128-byte rows in TMA's 128B
    swizzle (layout type 1, 1024 bytes per 8 rows; the tile is 1024-aligned and
    `addr` may step into it by K). Swizzled K-major ignores the LBO (set to 1)."""
    return (
        Int64((addr >> 4) & 0x3FFF)
        | (Int64(1) << 16)
        | (Int64(1024 >> 4) << 32)
        | (Int64(1) << 62)
    )


@asm()
def wgmma_ss(n, acc, adesc, bdesc, scale_d):
    """Wgmma m64n{n}k32 e4m3 x e4m3 -> f32, A and B by descriptor; `acc`: n/2
    floats, ignored where `scale_d` is 0; returns the n/2 accumulators."""
    nr = n // 2
    outs = ",".join(f"${i}" for i in range(nr))
    code = (
        "{\n.reg .pred p;\n"
        f"setp.ne.b32 p, ${nr + 2}, 0;\n"
        f"wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 "
        f"{{{outs}}}, ${nr}, ${nr + 1}, p, 1, 1;\n}}"
    )
    constraints = ",".join(["=f"] * nr + ["l", "l", "r"] + [str(i) for i in range(nr)])
    operands = (Int64(adesc), Int64(bdesc), Int32(scale_d), *(Float32(v) for v in acc))
    return code, constraints, operands, (Float32,) * nr


# ------------------------------------------------------------- warp, math
@asm(Float32)
def shfl_xor_f32(x, m):
    code = "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;"
    return code, "=f,f,r", (Float32(x), Int32(m))


@asm(Int32)
def shfl_up_i32(x, d):
    return (
        "shfl.sync.up.b32 $0, $1, $2, 0x0, 0xffffffff;",
        "=r,r,r",
        (Int32(x), Int32(d)),
    )


@asm(Float32, pure=True)
def swiglu(gate, up, limit):
    """min(gate, limit) * sigmoid(min(gate, limit)) * clamp(up, -limit, limit)."""
    code = (
        "{\n.reg .f32 g, u, e, nl;\n"
        "min.f32 g, $1, $3;\n"
        "neg.f32 nl, $3;\n"
        "min.f32 u, $2, $3;\nmax.f32 u, u, nl;\n"
        "mul.f32 e, g, 0fBFB8AA3B;\n"
        "ex2.approx.ftz.f32 e, e;\n"
        "add.f32 e, e, 0f3F800000;\n"
        "rcp.approx.ftz.f32 e, e;\n"
        "mul.f32 e, e, g;\n"
        "mul.f32 $0, e, u;\n}"
    )
    return code, "=f,f,f,f", (Float32(gate), Float32(up), Float32(limit))


@asm(Float32, pure=True)
def fabs(x):
    return "abs.f32 $0, $1;", "=f,f", (Float32(x),)


@asm(Float32, pure=True)
def fmax(a, b):
    return "max.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def rcp(x):
    return "rcp.approx.ftz.f32 $0, $1;", "=f,f", (Float32(x),)


@asm(Float32, pure=True)
def div_rn(a, b):
    return "div.rn.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Uint32, pure=True)
def sel_u32(c, a, b):
    """`c != 0 ? a : b`."""
    code = "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}"
    return code, "=r,r,r,r", (Int32(c), Uint32(a), Uint32(b))


@asm(Float32, pure=True)
def sel_f32(c, a, b):
    code = "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}"
    return code, "=f,r,f,f", (Int32(c), Float32(a), Float32(b))


@asm(Int32, pure=True)
def eq_u32(a, b):
    """1 if a == b else 0."""
    code = "{\n.reg .pred p;\nsetp.eq.u32 p, $1, $2;\nselp.s32 $0, 1, 0, p;\n}"
    return code, "=r,r,r", (Uint32(a), Uint32(b))


@asm((Uint32, Uint32), pure=True)
def score_key(logit, bias):
    """The `single_group_topk` scores of a sigmoid router: (the unbiased
    sigmoid, the twiddled key of sigmoid + bias). A non-finite logit or bias
    gives 0 and a -inf key; -0 keys as +0. The sigmoid is
    1 / (1 + 2^(-x log2 e)) (`rcp.rn`)."""
    code = (
        "{\n.reg .pred fx, fb, ok, ng, z;\n.reg .f32 e, u, s;\n.reg .b32 bs, nb, k;\n"
        "testp.finite.f32 fx, $2;\ntestp.finite.f32 fb, $3;\nand.pred ok, fx, fb;\n"
        "mul.f32 e, $2, 0fBFB8AA3B;\nex2.approx.f32 e, e;\n"
        "add.f32 e, e, 0f3F800000;\nrcp.rn.f32 u, e;\n"
        "add.f32 s, u, $3;\nsetp.eq.f32 z, s, 0f00000000;\n@z mov.f32 s, 0f00000000;\n"
        "@!ok mov.f32 u, 0f00000000;\n@!ok mov.f32 s, 0fFF800000;\n"
        "mov.b32 bs, s;\nsetp.lt.s32 ng, bs, 0;\nnot.b32 nb, bs;\n"
        "or.b32 k, bs, 0x80000000;\n@ng mov.b32 k, nb;\n"
        "mov.b32 $0, u;\nmov.b32 $1, k;\n}"
    )
    return code, "=r,=r,f,f", (Float32(logit), Float32(bias))


@asm((Uint32,) * 4, pure=True)
def minmax_key(ah, al, bh, bl):
    """Lexicographic unsigned (hi, lo) max and min: (maxh, maxl, minh, minl)."""
    code = (
        "{\n.reg .b64 a, b, x, n;\nmov.b64 a, {$5, $4};\nmov.b64 b, {$7, $6};\n"
        "max.u64 x, a, b;\nmin.u64 n, a, b;\n"
        "mov.b64 {$1, $0}, x;\nmov.b64 {$3, $2}, n;\n}"
    )
    return code, "=r,=r,=r,=r,r,r,r,r", tuple(Uint32(v) for v in (ah, al, bh, bl))
