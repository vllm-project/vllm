# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
# ruff: noqa: E501, F841

# Vendored from AMD's Muse Spark FlyDSL monokernel kernels (mk/dev.py):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Device-side primitives for the Muse FlyDSL monokernel (FlyDSL 0.3.2).

Most of this is adapted from ATOM's ``atom/mono/device`` (ops.py, sync.py,
stamps.py; MIT, AMD), trimmed to what the MoE block needs and changed where the
installed FlyDSL 0.3.2 / aiter differ (no ``kernel_signature``). Differences
from ATOM that matter:

* Mailbox tags are epoch based (``tag = epoch * 64 + slot + 1`` with the epoch a
  device word bumped once per forward inside the CUDA graph) instead of ATOM's
  zero-at-step-start discipline: nothing is ever zeroed.
* Every spin is bounded by a wall-clock timeout (s_memrealtime, 100 MHz) that
  writes a diagnostic record and sets a sticky FAILED word, so a bug turns into
  a report instead of a hang (brief rule).
"""

from __future__ import annotations

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

# gfx950 cache-policy bits (LLVM CPol): SC0 = 1, SC1 = 16, NT = 2.
CM_DEV = 16  # device scope: past the per-XCD L2
CM_SYS = 17  # system scope
CM_NT = 2  # nontemporal: weights streamed once


def traced(fn):
    """FlyDSL AST rewriting for a helper (``if`` / ``while`` on traced values)."""
    return ASTRewriter.transform(fn)


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def memrealtime():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def readlane(v, src_lane):
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    r = fx.Int32(rocdl.readlane(T.i32, bits, fx.Int32(src_lane)))
    return r.bitcast(fx.Float32) if isinstance(v, fx.Float32) else r


def fresh(v):
    """An opaque v_mov: stops LLVM hoisting / CSE of lane math into long-lived
    registers (ATOM ops.fresh)."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(v).ir_value()], "v_mov_b32 $0, $1", "=v,v",
            has_side_effects=True,
        )
    )  # fmt: skip


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rsq(x):
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def hw_sqrt(x):
    return _hw_f32("llvm.amdgcn.sqrt.f32", x)


def permlane_swap(off, x, y):
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def xshfl(v, off):
    assert off in (1, 2, 4, 8, 16, 32), off
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
    for off in offsets:
        v = xred(v, off, op)
    return v


def wave_sum(v):
    return butterfly(v, (32, 16, 8, 4, 2, 1))


def ballot(pred):
    return fx.Int64(rocdl.ballot(T.i64, as_ir_value(pred)))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    return popcount(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))


def _xpartner(v, off, lane):
    if off < 16:
        return xshfl(v, off)
    a, b = permlane_swap(off, v, v)
    return ((lane & off) != 0).select(a, b)


def wave_incl_scan(v, lane):
    blk = v
    for off in (1, 2, 4, 8, 16, 32):
        o = _xpartner(blk, off, lane)
        v = v + ((lane & off) != 0).select(o, 0)
        blk = blk + o
    return v


def bpermute(v, src_lane):
    """Lane ``src_lane``'s i32 v (ds_bpermute: the LDS crossbar, no LDS memory)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def mfma_bf16(a, b, c):
    """16x16x32 bf16 MFMA: a, b bf16x8 vectors, c f32x4."""
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one i32 word holding (bf16(a) low, bf16(b) high), RNE."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def fp4x8_bf16(word, scale_f32):
    """One dword of 8 E2M1 (low nibble first) times ``scale_f32`` -> bf16x8
    (exact: v_cvt_scalef32_pk_bf16_fp4, the aiter Muse a16w4 decode)."""
    parts = []
    for sel in range_constexpr(4):
        pair = rocdl.cvt_scalef32_pk_bf16_fp4(
            T.vec(2, T.bf16), as_ir_value(fx.Int32(word)), as_ir_value(scale_f32), sel
        )
        parts.append(fx.Vector(pair).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(parts, fx.Int32).bitcast(fx.BFloat16)


def e8m0_f32(code):
    """E8M0 biased exponent (an i32 holding 1..254) -> the f32 power of two."""
    return (fx.Int32(code) << 23).bitcast(fx.Float32)


def ld_i32(ptr, i, cm=0):
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


def ld_f32(ptr, i, cm=0):
    return fx.Float32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32, cache_modifier=cm)
    )


def ld_vec(ptr, i, words, cm=0, nbytes=None):
    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr, nbytes), i, vec_width=words, dtype=T.i32, cache_modifier=cm
        )
    )


OOB_DW = 0x3FFFFFFF  # a dword index past any buffer's num_records: the load returns 0, no memory access
OOB_BOUND = 0x80000000  # num_records of a mailbox resource: every real offset is below it, OOB_DW*4 above


def ld_vec_if(ok, ptr, i, words, nbytes, cm=0):
    """``words`` dwords at dword ``i``, or zeros without a memory access when not ``ok``
    (out-of-range buffer offset; ``nbytes`` must bound the buffer)."""
    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr, nbytes),
            ok.select(i, fx.Int32(OOB_DW)),
            vec_width=words,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )


def ld_i32_if(ok, ptr, i, nbytes, cm=0):
    return fx.Int32(
        bo.buffer_load(
            rsrc(ptr, nbytes),
            ok.select(i, fx.Int32(OOB_DW)),
            vec_width=1,
            dtype=T.i32,
            cache_modifier=cm,
        )
    )


def st_i32(v, ptr, i, cm=0):
    bo.buffer_store(fx.Int32(v), rsrc(ptr), i, cache_modifier=cm)


# ------------------------------------------------------------------ mailbox
DIAG_WORDS = 8
TIMEOUT_TICKS = 2 * 100_000_000  # 2 s at 100 MHz


@traced
def _spin_bounded(load_all, pending, health, region_id, index, tag):
    """Re-load until ``pending`` clears or TIMEOUT_TICKS pass; on a timeout the
    CTA's first record goes to ``health`` (word 0 = sticky FAILED, words
    8 + 8 bid .. = record) and later waits of this launch exit at once."""
    t0 = memrealtime()
    v = load_all()
    late = fx.Int32(0)
    failed = ld_i32(health, 0, CM_DEV)
    while pending(v) & (late == 0) & (failed == 0):
        rocdl.s_sleep(1)
        v = load_all()
        late = (memrealtime() - t0 > TIMEOUT_TICKS).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        bid = fx.block_idx.x
        at = fx.Int32(memrealtime() & 0x7FFFFFFF)
        record = [1, region_id, index, tag, v[1], bid, at, 0]
        for h in range_constexpr(2):
            bo.buffer_store(
                fx.Vector.from_elements(
                    [fx.Int32(w) for w in record[4 * h : 4 * h + 4]], fx.Int32
                ),
                rsrc(health),
                DIAG_WORDS + bid * DIAG_WORDS + 4 * h,
                cache_modifier=CM_DEV,
            )
        bo.buffer_store(fx.Int32(1), rsrc(health), 0, cache_modifier=CM_DEV)
    return v


# Mailbox helpers are plain functions, not methods: FlyDSL 0.3.2's region
# analyzer carries every local whose method is called inside an ``if`` / ``while``
# as scf state, which an object of Python callables cannot be.
#
# ``tag`` is a traced Int32 (epoch-based, unique per launch and stage), so a pair
# from an earlier launch never matches; no region is ever zeroed.


def mb_put(tag, base, i, v):
    """Pair i := (v, tag), device scope."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(
        fx.Vector.from_elements([bits, tag], fx.Int32),
        rsrc(base),
        i * 2,
        cache_modifier=CM_DEV,
    )


def mb_put_sys(tag, base, i, v):
    """Pair i := (v, tag), system scope (a peer GPU's buffer)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(
        fx.Vector.from_elements([bits, tag], fx.Int32),
        rsrc(base),
        i * 2,
        cache_modifier=CM_SYS,
    )


def mb_put2(tag, base, i, w0, w1):
    """Pairs i, i+1 := (w0, tag), (w1, tag): one 16 B store (i even)."""
    bo.buffer_store(
        fx.Vector.from_elements([fx.Int32(w0), tag, fx.Int32(w1), tag], fx.Int32),
        rsrc(base),
        i * 2,
        cache_modifier=CM_DEV,
    )


@traced
def _spin_bounded_from(v, load_all, pending, health, region_id, index, tag):
    """``_spin_bounded`` whose first round (``v``) was issued by the caller."""
    t0 = memrealtime()
    late = fx.Int32(0)
    failed = ld_i32(health, 0, CM_DEV)
    while pending(v) & (late == 0) & (failed == 0):
        rocdl.s_sleep(1)
        v = load_all()
        late = (memrealtime() - t0 > TIMEOUT_TICKS).select(fx.Int32(1), fx.Int32(0))
    if late != 0:
        bo.buffer_store(fx.Int32(1), rsrc(health), 0, cache_modifier=CM_DEV)
    return v


def mb_poll_split(tag, health, specs, region_id=0):
    """A poll whose first round is issued now and checked later:
    ``first, finish = mb_poll_split(...)``; other loads may be issued between,
    then ``finish(first)`` -> words as ``mb_poll``. One batch (len(specs) <= 32)."""

    def load_all():
        words = []
        for b, i, n in specs:
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(b),
                    fx.Int32(i) * 2,
                    vec_width=2 * n,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            words += [w[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    nw = sum(2 * n for _, _, n in specs)

    def pending(v):
        bad = v[1] != tag
        for e in range(3, nw, 2):
            bad = bad | (v[e] != tag)
        return bad

    def finish(v):
        v = _spin_bounded_from(
            v, load_all, pending, health, region_id, specs[0][1], tag
        )
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs

    return load_all(), finish


def mb_poll(tag, health, specs, region_id=0, batch=16, oks=None, cm=CM_DEV):
    """[(base, pair index, npairs in {1, 2})] -> value words, all re-loaded
    together until every tag matches (one round trip after the last lands).
    ``oks`` (optional, one traced bool a spec): a false spec is not loaded (an
    out-of-range buffer offset, no memory access), returns zeros, and is not
    waited for."""
    if len(specs) == 0:
        return []
    if len(specs) > batch:
        o1 = None if oks is None else oks[:batch]
        o2 = None if oks is None else oks[batch:]
        return mb_poll(tag, health, specs[:batch], region_id, batch, o1, cm) + mb_poll(
            tag, health, specs[batch:], region_id, batch, o2, cm
        )

    def load_all():
        words = []
        for si, (b, i, n) in enumerate(specs):
            idx = fx.Int32(i) * 2
            if oks is not None:
                idx = oks[si].select(idx, fx.Int32(OOB_DW))
            w = fx.Vector(
                bo.buffer_load(
                    rsrc(b, OOB_BOUND),
                    idx,
                    vec_width=2 * n,
                    dtype=T.i32,
                    cache_modifier=cm,
                )
            )
            words += [w[e] for e in range(2 * n)]
        return fx.Vector.from_elements(words, fx.Int32)

    nw = sum(2 * n for _, _, n in specs)

    def pending(v):
        bad = None
        e = 0
        for si, (_, _, n) in enumerate(specs):
            for q in range(n):
                b1 = v[e + 2 * q + 1] != tag
                if oks is not None:
                    b1 = b1 & oks[si]
                bad = b1 if bad is None else (bad | b1)
            e += 2 * n
        return bad

    v = _spin_bounded(load_all, pending, health, region_id, specs[0][1], tag)
    outs, e = [], 0
    for _, _, n in specs:
        outs.append([v[e + 2 * q] for q in range(n)])
        e += 2 * n
    return outs


# ------------------------------------------------------------------ timeline
@traced
def stamp(on, tls, tid, k):
    if const_expr(on):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(memrealtime(), tls + k)


def memrealtime_after(dep):
    """s_memrealtime ordered after ``dep`` (an i32 the asm takes as an input) and
    waited for: an in-kernel timestamp that the scheduler cannot hoist."""
    return fx.Int64(
        llvm.inline_asm(
            T.i64,
            [fx.Int32(dep).ir_value()],
            "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
            "=s,v",
            has_side_effects=True,
        )
    )


@traced
def stamp_dep(on, tls, tid, k, dep):
    """A stamp at point k ordered after ``dep`` (low 32 bits of the clock)."""
    if const_expr(on):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(memrealtime_after(dep), tls + k)


@traced
def stamp_flush(on, tls, tl, tid, bid, points):
    if const_expr(on):  # noqa: SIM102
        if tid < points:
            v = fx.ptr_load(tls + tid)
            lo = fx.Int32(v & 0xFFFFFFFF)
            hi = fx.Int32(v >> 32)
            bo.buffer_store(
                fx.Vector.from_elements([lo, hi], fx.Int32),
                rsrc(tl),
                (bid * points + tid) * 2,
            )


def fp8x2(a, b):
    """Two f32 -> two E4M3 bytes in the low 16 bits (v_cvt_pk_fp8_f32, RNE)."""
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False)) & 0xFFFF


def cache_inv(system=False):
    """Acquire-side invalidate (gfx94x/95x memory model): buffer_inv sc1 (device scope) or
    sc0 sc1 (system scope), so the normal (cached) loads that follow see data other CUs or
    other GPUs released before their flag. Side-effecting: no memory op moves across it."""
    asm = "buffer_inv sc0 sc1" if system else "buffer_inv sc1"
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [], asm + "\n\tv_mov_b32 $0, 0", "=v", has_side_effects=True
        )
    )


def st_if(ok, v, base, i, cm=0):
    """buffer_store of v (i32 or i32 vector) at dword i, dropped (out-of-range offset, no
    branch) when not ``ok``: keeps the waitcnt pass from draining at a join."""
    bo.buffer_store(
        v,
        rsrc(base, OOB_BOUND),
        ok.select(fx.Int32(i), fx.Int32(OOB_DW)),
        cache_modifier=cm,
    )


def mb_put_if(ok, tag, base, i, v, cm=CM_DEV):
    """Pair i := (v, tag) when ``ok`` (branch-free, see st_if)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    st_if(ok, fx.Vector.from_elements([bits, tag], fx.Int32), base, fx.Int32(i) * 2, cm)


def min_tree(vs):
    """Pairwise minimum of a power-of-two list of traced ints (plain Python: not AST-rewritten)."""
    vs = list(vs)
    while len(vs) > 1:
        vs = [fx.min(vs[2 * j], vs[2 * j + 1]) for j in range(len(vs) // 2)]
    return vs[0]
