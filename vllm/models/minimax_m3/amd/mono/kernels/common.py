# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side building blocks shared by the mono kernels.

Ported from the FlyDSL GLM-5 layer kernel (ROCm/FlyDSL branch
``glm5-mla-moe-layer-monokernel`` @ 1e28fd8a,
``kernels/mla_moe_layer/shared_reuse_moe_kernel.py``): wave reductions, tagged
mailboxes and the MFMA GEMV loop. Only published flydsl APIs and aiter's vendored
``buffer_ops`` / ``dpp_utils`` are used, so nothing here needs the FlyDSL repo.

Every function but ``kernel_symbol`` is traced inside a ``@flyc.kernel`` body.

Raw boundaries the kernels keep, for want of an equivalent on the pinned FlyDSL:
``buffer_ops`` loads / stores (the kernels take raw Int64 addresses, no tensors
for ``make_buffer_tensor``); raw MFMA intrinsics (hand-laid operands: permlane
transposes, fp8 packing, MXFP4 scales); the LLVM intrinsics here (``expect``,
``s_memrealtime``, the approximate rsq / rcp / exp2) and the struct unpacking of
``permlane*_swap``; ``inttoptr`` for the timeline's global stores.
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, as_ir_value

from vllm.models.minimax_m3.amd.mono.config import LAYER_SLOTS, WAVES

# gfx95x cache policy bits (LLVM CPol): SC0 = 1, SC1 = 16. SC1 alone is device
# scope (past the per-XCD caches); SC0 | SC1 is system scope (peers over XGMI).
CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)

# Namespace these kernels' symbols appear under in a profile.
KERNEL_NAMESPACE = "flydsl"


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``flydsl::<stem>_<params>``. The
    amdhsa assembler rejects ``::`` in a symbol, so it is Itanium-mangled, as
    aiter's ``_ZN5aiter...E`` kernels are.

    Not ``vllm``: that is vLLM's own C++ kernel namespace, so these would be
    indistinguishable from its native kernels in a profile.
    """
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN{len(KERNEL_NAMESPACE)}{KERNEL_NAMESPACE}{len(name)}{name}E"


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def permlane_swap(off, x, y):
    """v_permlane{32,16}_swap on registers (x, y): off 32 trades x's upper-half
    lanes with y's lower-half lanes; off 16 trades x's odd 16-lane rows with y's
    even rows (each within its half). -> (x', y')."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v, a source lane a lane (ds_bpermute)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def readlane(v, src_lane):
    """Lane ``src_lane``'s value of v (wave-uniform)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def memrealtime():
    """s_memrealtime: the 100 MHz device clock (no FlyDSL wrapper)."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def unlikely(cond):
    """``cond`` marked cold (llvm.expect): LLVM lays the branch out of the hot
    path. A rare path's code inline among the hot stages costs them even when it
    never runs (the long-context selection: S = 16, 3k context +1.4 us)."""
    return fx.Boolean(
        llvm.call_intrinsic(
            ir.IntegerType.get_signless(1),
            "llvm.expect.i1",
            [as_ir_value(cond), as_ir_value(fx.Boolean(False))],
            [],
            [],
        )
    )


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def kv_cache_scale(ptr):
    """An FP8 KV cache's scale: one fp32 for the whole tensor, wave-uniform.

    vLLM keeps a single scale per cache (the attention layer's ``_k_scale`` /
    ``_v_scale``), fixed once the checkpoint is loaded, so K and V quantize and
    dequantize against a scalar rather than against a scale per token. It stays a
    pointer rather than a build constant to keep the kernel cache key independent
    of the loaded value.
    """
    bits = uniform(bo.buffer_load(rsrc(ptr, 4), 0, vec_width=1, dtype=T.i32))
    return bits.bitcast(fx.Float32)


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp, no libm range fixups)."""
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def div_rn(x, d, r):
    """``x / d`` rounded to nearest, bit-identical to the IEEE division, given
    ``r = 1.0 / d`` from one true division per divisor (Markstein: a quotient
    within 1 ulp plus its exact fma residual times RN(1/d) rounds correctly)."""
    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


def xshfl(v, off):
    """Value of lane ``lane ^ off``: offsets 32 / 16 via v_permlane*_swap, in-row
    offsets via DPP (VALU latency, not LDS)."""
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def xred(v, off, op):
    """op(v, lane ^ off) for a symmetric op; 32 / 16 take both halves of one swap."""
    if off < 16:
        return op(v, xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    a, b = permlane_swap(off, x, x)
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over the given xor offsets, in the given order.

    The order is the numerical contract: the aiter per-head norm reduces 32
    lanes with offsets 16, 8, 4, 2, 1, and matching it keeps rstd bit-exact."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def traced(fn):
    """Apply flydsl's AST rewriting to a helper, so its ``while`` / ``if`` over
    traced values lower to scf ops. Kernels rewrite only their own body."""
    return ASTRewriter.transform(fn)


def wave_max(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1), fx.max)


@traced
def block_max(v, lane, wave, red):
    """Max of v over the CTA's WAVES waves; ``red`` needs WAVES words."""
    wv = wave_max(v)
    gpu.barrier()
    if lane == 0:
        fx.ptr_store(wv, red + wave)
    gpu.barrier()
    t = fx.ptr_load(red + 0)
    for w in range_constexpr(1, WAVES):
        t = fx.max(t, fx.ptr_load(red + w))
    gpu.barrier()
    return t


def mfma_bf16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def fp8x8_bf16_pk(words):
    """Eight fp8 of two dwords -> bf16 MFMA operand (exact) by
    v_cvt_scalef32_pk_bf16_fp8 (scale 1), two values an instruction."""
    out = []
    for w in words:
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, as_ir_value(fx.Int32(w)), as_ir_value(fx.Float32(1.0)), hi
            )
            out.append(fx.Vector(p).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32).bitcast(fx.BFloat16)


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)


def fp8_pack4(a, b, c, d):
    """Four f32 (already in range) -> one dword of four E4M3 bytes, RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


@traced
def _spin(load_all, pending):
    """Re-issue ``load_all`` until ``pending`` clears; a side-effecting op in the
    loop keeps the loads from being hoisted."""
    v = load_all()
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


class Mailbox:
    """Tagged-pair hand-off between CTAs (device scope) and GPUs (system scope).

    Every 32-bit value is stored next to this launch's epoch tag; a consumer
    polls the payload until all tags match, so a hand-off costs one round trip
    (no store drain, no separate flag). The epoch is ``step * LAYER_SLOTS +
    layer + 1`` with ``step`` a device counter, so the sequence is graph-safe.
    """

    def __init__(self, scratch, step_ptr, layer):
        self.scratch = scratch
        self.tag = (
            uniform(bo.buffer_load(rsrc(step_ptr), 0, vec_width=1, dtype=T.i32))
            * LAYER_SLOTS
            + layer
            + 1
        )

    def addr(self, offset):
        return self.scratch + fx.Int64(offset)

    def put(self, base_addr, i, v, cm=CM_DEV):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(base_addr),
            i * 2,
            cache_modifier=cm,
        )

    def put_bf(self, base_addr, i, vs, cm=CM_DEV):
        """Elements i .. i+len(vs) (2 or 4, i aligned) as packed bf16 pairs, in
        one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(base_addr),
            i,
            cache_modifier=cm,
        )

    def put_words(self, base_addr, i, words, cm=CM_DEV):
        """Raw 32-bit words (1 or 2) at pairs i.. as (word, tag). Integer payloads
        (fp8 words) must come this way: through an f32 value a NaN bit pattern is
        not kept."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(base_addr),
            i * 2,
            cache_modifier=cm,
        )

    def poll(self, specs, scope="agent", batch=POLL_MAX):
        """Batched poll of pairs ``specs`` = [(base_addr, pair index, npairs in
        {1, 2})].

        All pairs are re-loaded together while any tag is stale, so a batch costs
        one round trip after its last producer lands. ``scope``: "agent" for pairs
        another CTA of this GPU wrote, "system" for pairs a peer GPU wrote. Returns
        the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], scope, batch) + self.poll(
                specs[batch:], scope, batch
            )
        assert scope in ("agent", "system"), scope
        cm = CM_DEV if const_expr(scope == "agent") else CM_SYS
        tag = self.tag

        def load_all():
            words = []
            for b, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


def ar_pack_sumsq(vals):
    """One logical thread of the custom all-reduce epilogue: its 8 elements'
    squares accumulated in order (hipcc contracts ``acc += v * v`` into fma)."""
    acc = fx.Float32(0.0)
    for v in vals:
        acc = fmath.fma(v, v, acc)
    return acc


@traced
def ar_block_sum(p0, p1, lane, wave, red):
    """Sum of the 768 per-pack partials of a 6144-wide bf16 row in the order the
    1-stage fused all-reduce + RMSNorm reduces them: 24 warps of 32, a 16..1
    butterfly per warp, then the 24 warp sums butterflied again in one warp.

    Logical warp ``3 * wave + lane // 32`` holds ``p0``; logical warp
    ``3 * wave + 2`` holds ``p1`` in lanes < 32 (``WAVES`` = 8). ``red`` needs
    24 words. Matching this order keeps rstd bit-exact with the original path.
    """
    return ar_block_sums([(p0, p1)], lane, wave, red)[0]


@traced
def ar_block_sums(parts, lane, wave, red):
    """``ar_block_sum`` of several rows ``parts = [(p0, p1), ...]`` behind one set
    of barriers; ``red`` needs 24 words per row."""
    bs = [
        (butterfly(p0, (16, 8, 4, 2, 1)), butterfly(p1, (16, 8, 4, 2, 1)))
        for p0, p1 in parts
    ]
    gpu.barrier()
    for k in range_constexpr(len(bs)):
        b0, b1 = bs[k]
        if lane % 32 == 0:
            fx.ptr_store(b0, red + (24 * k + wave * 3 + lane // 32))
        if lane == 0:
            fx.ptr_store(b1, red + (24 * k + wave * 3 + 2))
    gpu.barrier()
    li = lane % 32
    tots = [
        butterfly(
            (li < 24).select(
                fx.ptr_load(red + (24 * k + fx.min(li, 23))), fx.Float32(0.0)
            ),
            (16, 8, 4, 2, 1),
        )
        for k in range(len(parts))
    ]
    gpu.barrier()
    return tots
