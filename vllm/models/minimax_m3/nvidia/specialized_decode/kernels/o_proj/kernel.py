# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, SIM102
"""MXFP8 decode GEMV on tcgen05: `y[M, N] = x[M, K] @ W[N, K]^T` for M <= 8, block-scaled E4M3 x E4M3.

The weight is the MMA's A operand (128 output features per CTA), the activation is B (8 token columns),
fp32 accumulation in TMEM. Both operands come in their MXFP8 storage: E4M3 values (K contiguous) and UE8M0
scales in the 128x4-swizzled layout, whose 512-byte atom per (128 rows, 128 K) is exactly the source of a
`tcgen05.cp` 32x128b.warpx4 scale copy, so both scale tensors are bulk-copied unchanged. For an activation of at
most 32 rows the atom of k-tile `k` sits at byte `512 k`.

A cluster of SPLIT CTAs shares one 128-row weight tile; rank r streams the interleaved k-tiles r, r + SPLIT, ...
through a RING-stage ring. Before the grid dependency each CTA issues its first RING weight tiles (and their scale
atoms) and prefetches the rest into L2; after it, the activation slice of the CTA (values and scale atoms) lands in
one buffer. Rank r owns rows [r * 128 / SPLIT, (r + 1) * 128 / SPLIT) of the tile: the other ranks store their fp32
partials of those rows into its shared memory and arrive on its barrier (release, cluster scope); the owner adds the
SPLIT partials in rank order and stores bf16.

Warps: 0 weight TMA, 1 activation TMA, 2 TMEM allocation + scale copies + MMA, 3 idle, 4-7 epilogue (warp 4 + i
reads TMEM lanes 32 i .. 32 i + 31).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims

CTA_M = 128  # weight rows per CTA = the MMA's M
MMA_N = 8  # token columns
CTA_K = 128  # one k-tile: 128 E4M3 bytes per row = one 128-byte swizzle row
MMA_INST_K = 32
K_BLOCKS = CTA_K // MMA_INST_K  # 4 MMAs per k-tile, one per scale byte of the atom
THREADS = 256
SF_ATOM_BYTES = 512
SF_ATOM_U16 = SF_ATOM_BYTES // 2
SF_COLS = 4  # TMEM columns of one scale atom
W_TILE_BYTES = CTA_M * CTA_K
X_TILE_BYTES = MMA_N * CTA_K
EVICT_FIRST = 0x12F0000000000000
A_COLS_PER_MMA = MMA_INST_K // 4  # TMEM columns of one 128 x K32 E4M3 granule of A

a_dtype = cutlass.Float8E4M3FN
b_dtype = cutlass.Float8E4M3FN


def num_k_tiles(k_in: int) -> int:
    return k_in // CTA_K


def acc_cols(nacc: int, mma_n: int = MMA_N) -> int:
    # nacc independent accumulators of MMA_N columns: MMA j of the CTA accumulates into
    # j % nacc, so the M128 x N8 MMAs do not form one serial dependency chain; the
    # epilogue adds them.
    return max(16, nacc * mma_n)


def a_col0(nacc: int, my_tiles: int, mma_n: int = MMA_N) -> int:
    # a_tmem: [acc | SFA per k-tile | one SFB slot (reused in issue order) | A: 32 columns (128 x 128 E4M3) per k-tile]
    return (acc_cols(nacc, mma_n) + (my_tiles + 1) * SF_COLS + 31) // 32 * 32


def tmem_cols(
    ring: int, nacc: int, a_tmem: bool = False, my_tiles: int = 0, mma_n: int = MMA_N
) -> int:
    need = acc_cols(nacc, mma_n) + 2 * ring * SF_COLS
    if a_tmem:
        need = a_col0(nacc, my_tiles, mma_n) + my_tiles * K_BLOCKS * A_COLS_PER_MMA
    cols = 32
    while cols < need:
        cols *= 2
    return cols


@dsl_user_op
def _mapa_u32(smem_ptr, peer, *, loc=None, ip=None):
    """The shared::cluster address of this CTA's shared-memory location in cluster CTA `peer`."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [smem_ptr.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_async_v4(dst, w0, w1, w2, w3, mbar, *, loc=None, ip=None):
    """st.async of four int32 words to a shared::cluster address, completing `mbar` there by 16 bytes."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w1).ir_value(loc=loc, ip=ip), cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
         cutlass.Int32(w3).ir_value(loc=loc, ip=ip), cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];", "r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _ld_shared_v4_f32(smem_ptr, *, loc=None, ip=None):
    """ld.shared.v4.f32 of four fp32 at a 16-byte-aligned shared-memory pointer."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _llvm.inline_asm(
        _llvm.StructType.get_literal([_T.f32()] * 4), [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip
    return [
        cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _test_wait_cluster(mbar, parity, *, loc=None, ip=None):
    """mbarrier.test_wait.parity.acquire.cluster on a barrier of this CTA completed by other CTAs' st.async."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    done = cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [mbar.toint(loc=loc, ip=ip).ir_value(), cutlass.Int32(parity).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}", "=r,r,r,~{memory}", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip
    return done != cutlass.Int32(0)


@dsl_user_op
def _absmax(acc, v, *, loc=None, ip=None):
    """max(acc, |v|)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [cutlass.Float32(acc).ir_value(loc=loc, ip=ip), cutlass.Float32(v).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 a;\n\tabs.f32 a, $2;\n\tmax.f32 $0, $1, a;\n\t}", "=f,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _llvm.inline_asm(
            _T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _e4m3x4(v0, v1, v2, v3, *, loc=None, ip=None):
    """Four f32 to E4M3 (RN, satfinite), packed little-endian into one int32 word (v0 in the low byte)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\tmov.b32 $0, {lo, hi};\n\t}", "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ld_volatile_s32(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.volatile.global.s32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _ld_global_s64(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int64(
        _llvm.inline_asm(
            _T.i64(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)], "ld.global.s64 $0, [$1];", "=l,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;", "l,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _bf16x2_word(lo, hi, *, loc=None, ip=None):
    """bf16(lo) | bf16(hi) << 16 (RN), with -0.0 halves turned into +0.0 (0x80000000 marks an empty exchange word)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(hi).ir_value(loc=loc, ip=ip), cutlass.Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 w, l, h;\n\t.reg .pred pl, ph;\n\tcvt.rn.bf16x2.f32 w, $1, $2;\n\t"
            "and.b32 l, w, 0xFFFF;\n\tsetp.eq.u32 pl, l, 0x8000;\n\t@pl and.b32 w, w, 0xFFFF0000;\n\t"
            "shr.u32 h, w, 16;\n\tsetp.eq.u32 ph, h, 0x8000;\n\t@ph and.b32 w, w, 0x0000FFFF;\n\t"
            "mov.b32 $0, w;\n\t}", "=r,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    """bf16(x) (RN) as the fp32 bit pattern of the rounded value."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h, z;\n\tcvt.rn.bf16.f32 h, $1;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}", "=r,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _bf16_round(x):
    return _bf16_bits(x).bitcast(cutlass.Float32)


# ---- push mode 5: o_proj + Lamport AR + residual + Gemma norm + MXFP8 in the epilogue (post/kernel.py math) ----
def _asm(ret, args, text, cons, side=True, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm

    return _llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _p5_st_sys(addr, v, *, loc=None, ip=None):
    # weak store, as the boundary exchange / mode 4 push (Lamport tag carries visibility; .sys scope costs per store)
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _p5_spin(addr, tag, *, loc=None, ip=None):
    """Poll a Lamport word until its tag (high 16 bits) matches; returns the word."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr), cutlass.Int32(tag)],
            "{\n\t.reg .b32 w, g;\n\t.reg .pred p;\n\tP5SPIN:\n\tld.volatile.global.b32 w, [$1];\n\t"
            "shr.u32 g, w, 16;\n\tsetp.ne.u32 p, g, $2;\n\t@p bra P5SPIN;\n\tmov.b32 $0, w;\n\t}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_fix(addr, tag, w, active, *, loc=None, ip=None):
    """W was loaded from addr; while active and its tag is stale, re-poll (stragglers only)."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [
                cutlass.Int64(addr),
                cutlass.Int32(tag),
                cutlass.Int32(w),
                cutlass.Int32(active),
            ],
            "{\n\t.reg .b32 w, g;\n\t.reg .pred p, a;\n\tmov.b32 w, $3;\n\tsetp.ne.u32 a, $4, 0;\n\t"
            "@!a bra P5FD;\n\tP5FL:\n\tshr.u32 g, w, 16;\n\tsetp.ne.u32 p, g, $2;\n\t@!p bra P5FD;\n\t"
            "ld.volatile.global.b32 w, [$1];\n\tbra P5FL;\n\tP5FD:\n\tmov.b32 $0, w;\n\t}",
            "=r,l,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_ld64(addr, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int64(
        _asm(
            _T.i64(),
            [cutlass.Int64(addr)],
            "ld.relaxed.gpu.global.b64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_fix64(addr, tag, w, active, *, loc=None, ip=None):
    """Tagged rowsq word: re-poll while active and stale; returns the float payload."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [
                cutlass.Int64(addr),
                cutlass.Int32(tag),
                cutlass.Int64(w),
                cutlass.Int32(active),
            ],
            "{\n\t.reg .b64 w;\n\t.reg .b32 lo, hi;\n\t.reg .pred p, a;\n\tmov.b64 w, $3;\n\t"
            "setp.ne.u32 a, $4, 0;\n\tP5GL:\n\tmov.b64 {lo, hi}, w;\n\t@!a bra P5GD;\n\t"
            "setp.ne.u32 p, hi, $2;\n\t@!p bra P5GD;\n\tld.relaxed.gpu.global.b64 w, [$1];\n\tbra P5GL;\n\t"
            "P5GD:\n\tmov.b32 $0, lo;\n\t}",
            "=f,l,r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _px_poll(addr, *, loc=None, ip=None):
    """Poll mode: a bf16-pair word of the decode output, re-polled while it is the 0xFFFFFFFF sentinel."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "{\n\t.reg .b32 w;\n\t.reg .pred p;\n\tPXPL:\n\tld.volatile.global.b32 w, [$1];\n\t"
            "setp.eq.u32 p, w, 0xFFFFFFFF;\n\t@p bra PXPL;\n\tmov.b32 $0, w;\n\t}",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _px_bcast0(v, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int32(v)],
            "shfl.sync.idx.b32 $0, $1, 0, 0x1f, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_ld_bf16(addr, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int64(addr)],
            "{\n\t.reg .b16 h;\n\t.reg .b16 z;\n\tld.global.b16 h, [$1];\n\tmov.b16 z, 0;\n\t"
            "mov.b32 $0, {z, h};\n\t}",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_st_bf16(addr, x, *, loc=None, ip=None):
    """bf16(x) RN stored to addr (x is already bf16-exact here)."""
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Float32(x)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _p5_st_u8(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u8 [$0], h;\n\t}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _p5_f(op, a, b, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            f"{op}.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_rsqrt(a, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_rowsq_pub(addr, tag, x, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(tag), cutlass.Float32(x)],
        "{\n\t.reg .b32 f;\n\t.reg .b64 w;\n\tmov.b32 f, $2;\n\tmov.b64 w, {f, $1};\n\t"
        "st.relaxed.gpu.global.b64 [$0], w;\n\t}",
        "l,r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _p5_rowsq_spin(addr, tag, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int64(addr), cutlass.Int32(tag)],
            "{\n\t.reg .b64 w;\n\t.reg .b32 lo, hi;\n\t.reg .pred p;\n\tP5RSQ:\n\t"
            "ld.relaxed.gpu.global.b64 w, [$1];\n\tmov.b64 {lo, hi}, w;\n\t"
            "setp.ne.u32 p, hi, $2;\n\t@p bra P5RSQ;\n\tmov.b32 $0, lo;\n\t}",
            "=f,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_e8m0(x, *, loc=None, ip=None):
    """UE8M0 of x, rounded up, satfinite (trtllm mxfp8_quantize scale rule)."""
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Float32(x)],
            "{\n\t.reg .b16 h;\n\t.reg .b32 w;\n\tcvt.rp.satfinite.ue8m0x2.f32 h, 0f00000000, $1;\n\t"
            "cvt.u32.u16 w, h;\n\tand.b32 $0, w, 255;\n\t}",
            "=r,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_atom_add(addr, v, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr), cutlass.Int32(v)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _p5_st_s32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int64(addr), cutlass.Int32(v)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rope_bits(c, a, s, b, *, loc=None, ip=None):
    """bf16(c * a + s * b) with separately rounded products (no FMA), as fp32 bits: the stock RoPE arithmetic."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (c, a, s, b)],
            "{\n\t.reg .f32 p, q, r;\n\t.reg .b16 h, z;\n\tmul.rn.f32 p, $1, $2;\n\tmul.rn.f32 q, $3, $4;\n\t"
            "add.rn.f32 r, p, q;\n\tcvt.rn.bf16.f32 h, r;\n\tmov.b16 z, 0;\n\tmov.b32 $0, {z, h};\n\t}", "=r,f,f,f,f",
            has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _din_load4(table, hidden, hidden_half, word_idx):
    """Four int32 words at word_idx of the hidden rows (hidden_half == 1) or the embedding table: one CTA-uniform
    base-address select instead of a branch."""
    addr = cutlass.select_(
        hidden_half == cutlass.Int32(1),
        hidden.data_ptr(0).toint(),
        table.data_ptr(0).toint(),
    )
    ptr = cutlass.inttoptr(
        addr + cutlass.Int64(word_idx) * cutlass.Int64(4),
        mem_space=1,
        dtype=cutlass.Int32,
    )
    return ptr.load(count=4, alignment=16)


def _k_tile(rank, i: int, split: int, my_tiles: int, contig: bool):
    # contig: rank r streams k-tiles [r * my_tiles, (r + 1) * my_tiles); else the interleaved r, r + split, ...
    if contig:
        return rank * cutlass.Int32(my_tiles) + cutlass.Int32(i)
    return rank + cutlass.Int32(i * split)


# ---- router CTAs: `ra` dedicated router CTAs at bx 0 .. ra - 1 of the grid ----
RA_TAB_WORDS = 8  # ra_tab int64: blob_a, blob_r, residual_mid, xq, xsf, rx bases (int64[world]), oep, rxc
RA_E = 8  # experts per router CTA
# TP4 shapes. a-part K = 2048 (8 E4M3 per thread, 2 chunks of 1024); r-part 1536
# residual columns per rank, padded to 2048 (2 chunks of 1024; columns >= 1536 have zero
# weights and a clamped, finite residual load); one rx part per source rank (a + r
# summed in the router CTA, add.rn), so the post polls world * 1 = 4 blocks.
RA_KO = 2048  # o_proj input width per rank (a-part columns)
RA_HK = 1536  # residual columns per rank (r-part)
RA_HKP = 2048  # padded r-part columns (zero weights beyond RA_HK)
RA_KM = RA_KO // 1024  # a-part chunks of 256 threads x 4
RA_H = 6144
RA_A_BYTES = RA_E * RA_KO * 4  # W'' slice of one router CTA in smem (64 KB)
RA_R_BYTES = RA_E * RA_HKP * 4  # padded W' slice (64 KB)
RA_NPART = 1  # rx parts per source rank: part 0 = a + r


def _f4(out, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return [
        cutlass.Float32(_llvm.extractvalue(_T.f32(), out, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _ra_ld4f_nc(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int64(addr)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _f4(out, loc, ip)


@dsl_user_op
def _ra_ld4b_cg(addr, *, loc=None, ip=None):
    """4 words at L2 (the decode's output: never an L1 copy)"""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.i32()] * 4),
        [cutlass.Int64(addr)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [
        cutlass.Int32(_llvm.extractvalue(_T.i32(), out, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _ra_ld2bf_cg(addr, *, loc=None, ip=None):
    """4 bf16 at addr (8 B) as 4 fp32"""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int64(addr)],
        "{\n\t.reg .b32 a, b, c, d;\n\tld.global.cg.v2.b32 {a, c}, [$4];\n\tshl.b32 $0, a, 16;\n\t"
        "and.b32 $1, a, 0xFFFF0000;\n\tshl.b32 $2, c, 16;\n\tand.b32 $3, c, 0xFFFF0000;\n\t}",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _f4(out, loc, ip)


@dsl_user_op
def _ra_scale_cg(addr, *, loc=None, ip=None):
    """UE8M0 byte at addr -> fp32 2^(s - 127) (s = 0: 2^-127)"""
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int64(addr)],
            "{\n\t.reg .b32 s, b;\n\t.reg .pred z;\n\tld.global.cg.u8 s, [$1];\n\t"
            "shl.b32 b, s, 23;\n\tsetp.eq.u32 z, s, 0;\n\tselp.b32 $0, 0x00400000, b, z;\n\t}",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra_e4m3x4(w, *, loc=None, ip=None):
    """4 packed E4M3 (byte 0 first) -> 4 fp32 (exact)"""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int32(w)],
        "{\n\t.reg .b16 lo, hi, a, b, c, d;\n\t.reg .b32 h0, h1;\n\tmov.b32 {lo, hi}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 h0, lo;\n\tcvt.rn.f16x2.e4m3x2 h1, hi;\n\tmov.b32 {a, b}, h0;\n\t"
        "mov.b32 {c, d}, h1;\n\tcvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\tcvt.f32.f16 $2, c;\n\t"
        "cvt.f32.f16 $3, d;\n\t}",
        "=f,=f,=f,=f,r",
        side=False,
        loc=loc,
        ip=ip,
    )
    return _f4(out, loc, ip)


@dsl_user_op
def _ra_f(op, a, b, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            f"{op}.f32 $0, $1, $2;",
            "=f,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra_fma(a, b, c, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            side=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra_lds4(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int32(addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _f4(out, loc, ip)


@dsl_user_op
def _ra_lds(addr, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Int32(addr)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra_sts(addr, x, *, loc=None, ip=None):
    _asm(
        None,
        [cutlass.Int32(addr), cutlass.Float32(x)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ra_bar(*, loc=None, ip=None):
    _asm(None, [], "barrier.sync 1, 256;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ra_bfly(x, off, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Float32(
        _asm(
            _T.f32(),
            [cutlass.Float32(x), cutlass.Int32(off)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra_entry(tid, mb0, mb1, dst, src_a, src_r, *, loc=None, ip=None):
    """Thread 0: init both W'' barriers, the first half (experts 0-3, 64 KB) by 4 bulk copies on mb0; L2 prefetch of the
    CTA's r blob"""
    cp = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$3+{16384 * i}], "
        f"[$4+{16384 * i}], 16384, [$1];\n"
        for i in range(RA_A_BYTES // 2 // 16384)
    )
    _asm(
        None,
        [
            cutlass.Int32(tid),
            cutlass.Int32(mb0),
            cutlass.Int32(mb1),
            cutlass.Int32(dst),
            cutlass.Int64(src_a),
            cutlass.Int64(src_r),
        ],
        "{\n\t.reg .pred p;\n\tsetp.eq.u32 p, $0, 0;\n\t@p mbarrier.init.shared::cta.b64 [$1], 1;\n\t"
        "@p mbarrier.init.shared::cta.b64 [$2], 1;\n\t"
        "@p fence.mbarrier_init.release.cluster;\n\t@p fence.proxy.async.shared::cta;\n\t"
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], {RA_A_BYTES // 2};\n"
        + cp
        + f"\t@p cp.async.bulk.prefetch.L2.global [$5], {RA_R_BYTES};\n\t}}",
        "r,r,r,r,l,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ra_load_a1(tid, mb1, dst, src_a, *, loc=None, ip=None):
    """Thread 0, after the CTA barrier that ends the generic-proxy use of the region: the second W'' half on mb1"""
    cp = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$2+{16384 * i}], "
        f"[$3+{16384 * i}], 16384, [$1];\n"
        for i in range(RA_A_BYTES // 2 // 16384)
    )
    _asm(
        None,
        [
            cutlass.Int32(tid),
            cutlass.Int32(mb1),
            cutlass.Int32(dst),
            cutlass.Int64(src_a),
        ],
        "{\n\t.reg .pred p;\n\tsetp.eq.u32 p, $0, 0;\n\t@p fence.proxy.async.shared::cta;\n\t"
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], {RA_A_BYTES // 2};\n"
        + cp
        + "\t}",
        "r,r,r,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ra_st_tag2(addr, tag, za, zr, on, *, loc=None, ip=None):
    """Rx words (part 0 = a, part 1 = r) of one (token, src, expert): tag << 32 | fp32, relaxed sys (peer memory)"""
    _asm(
        None,
        [
            cutlass.Int64(addr),
            cutlass.Int32(tag),
            cutlass.Float32(za),
            cutlass.Float32(zr),
            cutlass.Int32(on),
        ],
        "{\n\t.reg .pred p;\n\t.reg .b32 f, g;\n\t.reg .b64 q, q1;\n\tsetp.ne.u32 p, $4, 0;\n\tmov.b32 f, $2;\n\t"
        "mov.b32 g, $3;\n\tmov.b64 q, {f, $1};\n\tmov.b64 q1, {g, $1};\n\t"
        f"@p st.relaxed.sys.global.b64 [$0], q;\n\t@p st.relaxed.sys.global.b64 [$0+{8 * 128}], q1;\n\t}}",
        "l,r,f,f,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ra_st_tag1(addr, tag, z, on, *, loc=None, ip=None):
    """TP4: one rx word (part 0 = a + r) of one (token, src, expert): tag << 32 | fp32, relaxed sys"""
    _asm(
        None,
        [
            cutlass.Int64(addr),
            cutlass.Int32(tag),
            cutlass.Float32(z),
            cutlass.Int32(on),
        ],
        "{\n\t.reg .pred p;\n\t.reg .b32 f;\n\t.reg .b64 q;\n\tsetp.ne.u32 p, $3, 0;\n\tmov.b32 f, $2;\n\t"
        "mov.b64 q, {f, $1};\n\t@p st.relaxed.sys.global.b64 [$0], q;\n\t}",
        "l,r,f,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ra_ld_relaxed(addr, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int64(addr)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra_trap(*, loc=None, ip=None):
    _asm(None, [], "trap;", "", loc=loc, ip=ip)


@dsl_user_op
def _ra_exit(*, loc=None, ip=None):
    _asm(None, [], "exit;", "~{memory}", loc=loc, ip=ip)


# ---- router CTAs hold W'' and W' in smem (one bulk burst at entry), one FMA pass each ----
RA_R_OFF = RA_A_BYTES  # W' slice after the W'' slice
RA_W_BYTES = RA_A_BYTES + RA_R_BYTES  # 229376: all router weights of one router CTA
RA_RG = 2  # r-part experts per pass above 4 rows


def _al(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _ra_ext_bytes(
    ring: int,
    my_tiles: int,
    x_tile_bytes: int,
    split: int,
    rows_per_rank: int,
    mb_stride: int,
) -> int:
    # pad after the GEMV arrays (allocation order of the kernel) so the router's W'' | W' | 2 mbarriers fit from smem_a
    off = ring * W_TILE_BYTES
    off = _al(off, 1024) + my_tiles * x_tile_bytes
    off = _al(off, 128) + ring * SF_ATOM_BYTES
    off = _al(off, 128) + my_tiles * SF_ATOM_BYTES
    off = _al(off, 8) + 8 * (2 * ring + 3)
    off = _al(off, 4) + 4
    off = _al(off, 16) + split * rows_per_rank * mb_stride * 4
    return max(_al(RA_W_BYTES + 16 - _al(off, 16), 16), 16)


@dsl_user_op
def _ra3_entry(tid, dst, mb_r, mb_a, src_r, src_a, *, loc=None, ip=None):
    """Thread 0: init both barriers, then W' (mb_r) and W'' (mb_a) by 16 KB bulk copies in one burst"""
    cpr = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1+{RA_R_OFF + 16384 * i}], "
        f"[$4+{16384 * i}], 16384, [$2];\n"
        for i in range(RA_R_BYTES // 16384)
    )
    cpa = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1+{16384 * i}], "
        f"[$5+{16384 * i}], 16384, [$3];\n"
        for i in range(RA_A_BYTES // 16384)
    )
    _asm(
        None,
        [
            cutlass.Int32(tid),
            cutlass.Int32(dst),
            cutlass.Int32(mb_r),
            cutlass.Int32(mb_a),
            cutlass.Int64(src_r),
            cutlass.Int64(src_a),
        ],
        "{\n\t.reg .pred p;\n\tsetp.eq.u32 p, $0, 0;\n\t@p mbarrier.init.shared::cta.b64 [$2], 1;\n\t"
        "@p mbarrier.init.shared::cta.b64 [$3], 1;\n\t@p fence.mbarrier_init.release.cluster;\n\t"
        "@p fence.proxy.async.shared::cta;\n\t"
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$2], {RA_R_BYTES};\n"
        + cpr
        + f"\t@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$3], {RA_A_BYTES};\n"
        + cpa
        + "\t}",
        "r,r,r,r,l,l,~{memory}",
        loc=loc,
        ip=ip,
    )


RC_A0 = 8  # TP4: all 8 W'' experts loaded in the entry burst (beside W' in the ring); the rest after the r-part, over W'
RC_E = RA_A_BYTES // RA_E  # 8 KB per expert (TP4)


def _rc_a_base(e: int) -> int:
    # smem offset of W'' expert e: experts < RC_A0 after W', the others over the consumed W'
    return RA_R_BYTES + e * RC_E if e < RC_A0 else (e - RC_A0) * RC_E


@dsl_user_op
def _rc_entry(tid, dst, mb0, mb1, mb2, src_r, src_a, *, loc=None, ip=None):
    """Thread 0: init 3 barriers; W' (96 KB, mb0) at 0 and W'' experts 0..RC_A0-1 (mb1) after it, one burst"""
    cpr = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1+{16384 * i}], "
        f"[$5+{16384 * i}], 16384, [$2];\n"
        for i in range(RA_R_BYTES // 16384)
    )
    cpa = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1+{RA_R_BYTES + 16384 * i}], "
        f"[$6+{16384 * i}], 16384, [$3];\n"
        for i in range(RC_A0 * RC_E // 16384)
    )
    _asm(
        None,
        [
            cutlass.Int32(tid),
            cutlass.Int32(dst),
            cutlass.Int32(mb0),
            cutlass.Int32(mb1),
            cutlass.Int32(mb2),
            cutlass.Int64(src_r),
            cutlass.Int64(src_a),
        ],
        "{\n\t.reg .pred p;\n\tsetp.eq.u32 p, $0, 0;\n\t@p mbarrier.init.shared::cta.b64 [$2], 1;\n\t"
        "@p mbarrier.init.shared::cta.b64 [$3], 1;\n\t@p mbarrier.init.shared::cta.b64 [$4], 1;\n\t"
        "@p fence.mbarrier_init.release.cluster;\n\t@p fence.proxy.async.shared::cta;\n\t"
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$2], {RA_R_BYTES};\n"
        + cpr
        + f"\t@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$3], {RC_A0 * RC_E};\n"
        + cpa
        + "\t}",
        "r,r,r,r,r,l,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rc_load_a(tid, dst, mb2, src_a, *, loc=None, ip=None):
    """Thread 0, after the CTA barrier that ends the generic reads of W': W'' experts RC_A0..7 over it (mb2)"""
    n = (RA_E - RC_A0) * RC_E
    cp = "".join(
        f"\t@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1+{16384 * i}], "
        f"[$3+{RC_A0 * RC_E + 16384 * i}], 16384, [$2];\n"
        for i in range(n // 16384)
    )
    _asm(
        None,
        [
            cutlass.Int32(tid),
            cutlass.Int32(dst),
            cutlass.Int32(mb2),
            cutlass.Int64(src_a),
        ],
        "{\n\t.reg .pred p;\n\tsetp.eq.u32 p, $0, 0;\n\t@p fence.proxy.async.shared::cta;\n\t"
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$2], {n};\n" + cp + "\t}",
        "r,r,r,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ra3_try(mb, *, loc=None, ip=None):
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _asm(
            _T.i32(),
            [cutlass.Int32(mb)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], 0;\n\t"
            "selp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ra3_ld2_cg(addr, *, loc=None, ip=None):
    """4 bf16 (8 B) at L2, kept packed"""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.i32()] * 2),
        [cutlass.Int64(addr)],
        "ld.global.cg.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [
        cutlass.Int32(_llvm.extractvalue(_T.i32(), out, [i], loc=loc, ip=ip))
        for i in range(2)
    ]


@dsl_user_op
def _ra3_bf4(p, *, loc=None, ip=None):
    """2 packed bf16 words -> 4 fp32 (exact)"""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int32(p[0]), cutlass.Int32(p[1])],
        "{\n\tshl.b32 $0, $4, 16;\n\tand.b32 $1, $4, 0xFFFF0000;\n\tshl.b32 $2, $5, 16;\n\t"
        "and.b32 $3, $5, 0xFFFF0000;\n\t}",
        "=f,=f,=f,=f,r,r",
        loc=loc,
        ip=ip,
    )  # side: no CSE across passes
    return _f4(out, loc, ip)


@dsl_user_op
def _ra3_e4m3x4(w, *, loc=None, ip=None):
    """_ra_e4m3x4 kept per use (side effects: a CSE across the halves would hold 16 T fp32 live)"""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    out = _asm(
        _llvm.StructType.get_literal([_T.f32()] * 4),
        [cutlass.Int32(w)],
        "{\n\t.reg .b16 lo, hi, a, b, c, d;\n\t.reg .b32 h0, h1;\n\tmov.b32 {lo, hi}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 h0, lo;\n\tcvt.rn.f16x2.e4m3x2 h1, hi;\n\tmov.b32 {a, b}, h0;\n\t"
        "mov.b32 {c, d}, h1;\n\tcvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\tcvt.f32.f16 $2, c;\n\t"
        "cvt.f32.f16 $3, d;\n\t}",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _f4(out, loc, ip)


def _ra3_wred(v, lane):
    """Warp reduce-scatter of len(v) (power of 2, <= 32) per-lane values: returns the lane's sum of output
    lane >> (5 - log2 len(v)) over all 32 lanes (halving exchange, then plain butterflies)"""
    d = 16
    while len(v) > 1:
        half = len(v) // 2
        up = (lane & cutlass.Int32(d)) != cutlass.Int32(0)
        nv = []
        for i in range(half):
            snd = cutlass.select_(up, v[i], v[i + half])
            kp = cutlass.select_(up, v[i + half], v[i])
            nv.append(_ra_f("add.rn", kp, _ra_bfly(snd, d)))
        v = nv
        d //= 2
    x = v[0]
    while d >= 1:
        x = _ra_f("add.rn", x, _ra_bfly(x, d))
        d //= 2
    return x


@cute.kernel
def m3_mx_gemv_kernel(
    tma_desc_w: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W [N, K] E4M3 as bytes, box 128 x 128
    tma_desc_wsf: cutlass.GridConstant[
        cuda.TensorMap
    ],  # W scale atoms [N / 128 * K / 128, 256] u16
    tma_desc_x: cutlass.GridConstant[
        cuda.TensorMap
    ],  # x [M, K] E4M3 as bytes, box 128 x 8
    tma_desc_xsf: cutlass.GridConstant[
        cuda.TensorMap
    ],  # x scale atoms [K / 128, 256] u16
    y: cutlass.Array,  # [M * N] bf16, token-major
    x_words: cutlass.Array,  # x_bf16: the bf16 activation [M, K] as int32 words
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    contig: cutlass.Constexpr[bool],
    nacc: cutlass.Constexpr[int],
    a_tmem: cutlass.Constexpr[bool],
    st_async: cutlass.Constexpr[bool],
    x_bf16: cutlass.Constexpr[bool],
    ar_ws: cutlass.Int64,  # ar_push: the all-reduce workspace's multicast mapping (int32 words)
    ar_flags: cutlass.Int64,  # ar_push: its call flags (parity of the next call = flags[0] & 1)
    ar_rank: cutlass.Int32,
    ar_push: cutlass.Constexpr[
        int
    ],  # 1: push the output rows into the next all-reduce; 2: and trigger after it; 3: unicast push into the
    # boundary's slots (ar_ws = device int64[world] of each rank's buffer base, ar_flags
    # = its Flags, whose word 0 is the call index; slot = base + ((call % 3 * world +
    # rank) * ar_maxt + token) * 6144), trigger after it
    ar_world: cutlass.Constexpr[int],
    ar_maxt: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[
        int
    ],  # token columns of the MMA (8, 16, 32, 48 or 64): M <= mma_n
    din_i: cutlass.Array,  # din: int32 token ids [M]
    din_table: cutlass.Array,  # din: embedding [V, K / 2] bf16 as int32 words
    din_hidden: cutlass.Array,  # din: hidden rows [M, K / 2] bf16 as int32 words
    din_w: cutlass.Array,  # din: [embedding norm | hidden norm] weights [K] bf16 as int32 words
    din_f: cutlass.Array,  # din: fp32 [eps_e, eps_h]
    din_vocab: cutlass.Int32,
    din_g: cutlass.Array,  # din, flag 1: int64 hidden row of each token, as int32 words
    din_res: cutlass.Array,  # din, flag 2: the gathered hidden rows out [M, K / 2] bf16 as int32 words
    din_flags: cutlass.Int32,  # din: 1 = gather the hidden rows, 2 = write them to din_res
    rope_cs: cutlass.Array,  # rope: fp32 (cos, sin) [positions][64][2]
    rope_i: cutlass.Array,  # rope: int32 cache lengths [M / T]
    rope_bt: cutlass.Array,  # rope: int32 K / V page tables [M / T][2][max_blocks]
    rope_q: cutlass.Array,  # rope: E4M3 Q out [M][hq][128] as uint8
    rope_pool: cutlass.Array,  # rope: paged E4M3 cache [pages][hkv][tokens per page][128] as uint8
    rope_scale: cutlass.Array,  # rope: fp32 [1] E4M3 quantization scale
    rope_dims: cutlass.Array,  # rope: int32 [max_blocks, tokens-per-page log2, hq, hkv, T]
    din: cutlass.Constexpr[bool],
    rope: cutlass.Constexpr[bool],
    stagger: cutlass.Constexpr[
        int
    ] = 0,  # rotate the bf16-quant item order by n_tile * stagger
    early_w: cutlass.Constexpr[
        bool
    ] = False,  # weight TMA at CTA entry (MXFP8-input path)
    qfirst: cutlass.Constexpr[
        int
    ] = 0,  # bf16 path: qfirst weight tiles before the quant lands
    burst: cutlass.Constexpr[
        int
    ] = 0,  # 1 tensor-box / 2 per-row 1-D L2 prefetch of the slice (warp 3)
    w_addr: cutlass.Int64 = 0,  # burst 2: weight / weight-scale base addresses
    wsf_addr: cutlass.Int64 = 0,
    pfx: cutlass.Constexpr[
        bool
    ] = False,  # split-K push mode 4 loads its parity flags / peer bases before the MMA wait
    ra: cutlass.Constexpr[
        int
    ] = 0,  # router CTAs at bx 0 .. ra - 1 (GEMV CTAs at bx >= ra); 0 = none
    ra_tab: cutlass.Int64 = 0,  # device int64[RA_TAB_WORDS]
    ra_tok: cutlass.Constexpr[int] = 0,  # token rows of the router (= the call's M)
    ra_ck: cutlass.Constexpr[int] = 0,  # 1 traps on oep != rxc[0]
):
    # Poll mode (ar_push = 10 + base mode): the bf16 input is polled for the producer's
    # writes (sentinel words 0xFFFFFFFF, re-armed by the last of the k-range's n_out/128
    # CTAs), so the GEMV can launch before its producer.
    ar_poll = ar_push >= 10
    ar_push = ar_push % 10
    k_tiles = num_k_tiles(k_in)
    x_tile_bytes = mma_n * CTA_K
    my_tiles = k_tiles // split
    rows_per_rank = CTA_M // split
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    bx_raw = bx
    if cutlass.const_expr(ra):
        bx = bx - cutlass.Int32(ra)
    warp_id = cute.arch.warp_idx()
    rank = cutlass.Int32(0)
    if cutlass.const_expr(split > 1):
        rank = cute.arch.block_idx_in_cluster()
    tma_ptr_w = tma_desc_w.get_ptr()
    tma_ptr_wsf = tma_desc_wsf.get_ptr()
    tma_ptr_x = tma_desc_x.get_ptr()
    tma_ptr_xsf = tma_desc_xsf.get_ptr()
    n_tile = bx // cutlass.Int32(split)
    m_offset = n_tile * cutlass.Int32(CTA_M)

    smem_a = cutlass.Array(
        cutlass.Uint8,
        ring * W_TILE_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    smem_b = cutlass.Array(
        cutlass.Uint8,
        my_tiles * x_tile_bytes,
        space=cutlass.AddressSpace.smem,
        alignment=1024,
    )
    smem_sfa = cutlass.Array(
        cutlass.Uint8,
        ring * SF_ATOM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    smem_sfb = cutlass.Array(
        cutlass.Uint8,
        my_tiles * SF_ATOM_BYTES,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    w_full = cutlass.Array(
        cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8
    )
    mma_done = cutlass.Array(
        cutlass.Int64, ring, space=cutlass.AddressSpace.smem, alignment=8
    )
    act_full = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_done = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    reduce_ready = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    # Partials of the rows this rank owns: [source rank][row][token] fp32 (its own slot
    # stays unused). Rows are padded to mma_n + 4 words: the owner's 16-byte row reads
    # (lane = row) then hit all 32 banks once per 8 lanes; at a stride of mma_n they
    # were 8- to 16-way conflicted (4.3 us of the 48-token QKV tail).
    mb_stride = mma_n + 4
    mailbox = cutlass.Array(
        cutlass.Float32,
        split * rows_per_rank * mb_stride,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    # din: per-token RMS factor of the half of x this CTA's k-tiles lie in.
    din_rs = None
    if cutlass.const_expr(din):
        din_rs = cutlass.Array(
            cutlass.Float32, 8, space=cutlass.AddressSpace.smem, alignment=16
        )

    if cutlass.const_expr(ra):
        assert (
            ra % split == 0
            and ra * RA_E == 128
            and ring * W_TILE_BYTES >= RA_A_BYTES + RA_R_BYTES
        )
        assert (
            k_in == RA_KO
            and n_out == RA_H
            and not x_bf16
            and not din
            and not rope
            and ar_world in (2, 4)
        )
        assert 1 <= ra_tok <= 8 and ra_tok & (ra_tok - 1) == 0
        if bx_raw < cutlass.Int32(ra):
            # router CTA r = bx_raw: experts 8r .. 8r + 7 of this rank's a = W''_k
            # deq(a_k) and r = W'[:, H_k] r[H_k]. ring (128 KB): W' at 0 + W'' experts
            # 0-1 at 96 KB (entry burst); W'' experts 2-7 over W' after the r-part.
            ra_r = bx_raw
            ra_sb = smem_a.data_ptr().toint()
            ra_mb = (
                smem_sfa.data_ptr().toint()
            )  # 3 mbarriers in the (router-unused) GEMV scale ring
            ra_sx = (
                smem_b.data_ptr().toint()
            )  # warp partials (r at 0, a at 4 KB) in the GEMV x stage
            ra_ba = _ld_global_s64(ra_tab) + cutlass.Int64(RA_A_BYTES) * cutlass.Int64(
                ra_r
            )
            ra_br = _ld_global_s64(ra_tab + cutlass.Int64(8)) + cutlass.Int64(
                RA_R_BYTES
            ) * cutlass.Int64(ra_r)
            _rc_entry(
                tx,
                ra_sb,
                ra_mb,
                ra_mb + cutlass.Int32(8),
                ra_mb + cutlass.Int32(16),
                ra_br,
                ra_ba,
            )
            # residual_mid (final at launch): 4 bf16 per (token, m), all loads in one round
            ra_res = _ld_global_s64(ra_tab + cutlass.Int64(16))
            ra_rv = []
            for t in cutlass.range_constexpr(ra_tok):
                ra_tt = cutlass.select_(
                    cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                )
                ra_rv.append(
                    [
                        _ra3_ld2_cg(
                            ra_res
                            + cutlass.Int64(2)
                            * (
                                cutlass.Int64(ra_tt) * cutlass.Int64(RA_H)
                                + cutlass.Int64(
                                    ar_rank * cutlass.Int32(RA_HK)
                                    + cutlass.select_(
                                        cutlass.Int32(ra_m * 1024)
                                        + cutlass.Int32(4) * tx
                                        < cutlass.Int32(RA_HK),
                                        cutlass.Int32(ra_m * 1024)
                                        + cutlass.Int32(4) * tx,
                                        cutlass.Int32(4) * tx,
                                    )
                                )
                            )
                        )
                        for ra_m in range(RA_HKP // 1024)
                    ]
                )
            ra_oep = _ld_global_s64(ra_tab + cutlass.Int64(48)) + cutlass.Int64(
                128
            ) * cutlass.Int64(ra_r)
            ra_tag = _ra_ld_relaxed(ra_oep) + cutlass.Int32(1)
            ra_nv = 4 * ra_tok  # values per half and lane: (expert of the half, token)
            ra_sh = 5 - (ra_nv.bit_length() - 1)
            ra_lane = tx % cutlass.Int32(32)
            ra_wp = tx // cutlass.Int32(32)
            ra_i = ra_lane >> cutlass.Int32(ra_sh)
            ra_rep = (ra_lane & cutlass.Int32((1 << ra_sh) - 1)) == cutlass.Int32(0)
            ra_o = cutlass.select_(
                tx < cutlass.Int32(8 * ra_tok), tx, cutlass.Int32(0)
            )  # output e * T + t of thread tx
            _ra_bar()  # barrier inits visible to all threads
            while _ra3_try(ra_mb) == cutlass.Int32(0):
                pass
            ra_z = []
            ra_g = (
                RA_RG if ra_tok > 4 else 4
            )  # experts per r pass (above 4 rows: fewer accumulators live beside the residual words)
            for ra_h in cutlass.range_constexpr(2):
                ra_hv = []
                for ra_p in cutlass.range_constexpr(4 // ra_g):
                    ra_acc = [[cutlass.Float32(0.0)] * ra_tok for _ in range(ra_g)]
                    for ra_m in cutlass.range_constexpr(RA_HKP // 1024):
                        ra_w = [
                            _ra_lds4(
                                ra_sb
                                + cutlass.Int32(16)
                                * (
                                    cutlass.Int32(
                                        (
                                            (ra_h * 4 + ra_p * ra_g + e)
                                            * (RA_HKP // 1024)
                                            + ra_m
                                        )
                                        * THREADS
                                    )
                                    + tx
                                )
                            )
                            for e in range(ra_g)
                        ]
                        for t in cutlass.range_constexpr(ra_tok):
                            ra_x = _ra3_bf4(ra_rv[t][ra_m])
                            for e in cutlass.range_constexpr(ra_g):
                                for j in cutlass.range_constexpr(4):
                                    ra_acc[e][t] = _ra_fma(
                                        ra_w[e][j], ra_x[j], ra_acc[e][t]
                                    )
                    ra_hv += [ra_acc[e][t] for e in range(ra_g) for t in range(ra_tok)]
                ra_z.append(_ra3_wred(ra_hv, ra_lane))
            _ra_bar()  # W' consumed: W'' experts RC_A0.. go over it
            if cutlass.const_expr(RC_A0 < RA_E):
                _rc_load_a(tx, ra_sb, ra_mb + cutlass.Int32(16), ra_ba)
            ra_s = ra_sx
            if ra_rep:
                for ra_h in cutlass.range_constexpr(2):
                    _ra_sts(
                        ra_s
                        + cutlass.Int32(4)
                        * (
                            ra_wp * cutlass.Int32(8 * ra_tok)
                            + cutlass.Int32(ra_h * ra_nv)
                            + ra_i
                        ),
                        ra_z[ra_h],
                    )
            _ra_bar()
            ra_zr = _ra_lds(ra_s + cutlass.Int32(4) * ra_o)
            for ra_k in cutlass.range_constexpr(1, THREADS // 32):
                ra_zr = _ra_f(
                    "add.rn",
                    ra_zr,
                    _ra_lds(
                        ra_s
                        + cutlass.Int32(4) * (cutlass.Int32(ra_k * 8 * ra_tok) + ra_o)
                    ),
                )
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            if cutlass.const_expr(ra_ck):
                if tx == 0:
                    # pairing check: this o_proj's calls == its post's consumes (else an unpaired call left words the post accepts)
                    ra_rxc = _ra_ld_relaxed(_ld_global_s64(ra_tab + cutlass.Int64(56)))
                    if ra_rxc != ra_tag - cutlass.Int32(1):
                        _ra_trap()
            # a_k (the decode's MXFP8 output, final at the wait): 16 E4M3 per token, one UE8M0 per token (block tid / 2)
            ra_xq = _ld_global_s64(ra_tab + cutlass.Int64(24))
            ra_xsf = _ld_global_s64(ra_tab + cutlass.Int64(32))
            ra_xw = []
            ra_sc = []
            for t in cutlass.range_constexpr(ra_tok):
                ra_tt = cutlass.select_(
                    cutlass.Int32(t) < num_tokens, cutlass.Int32(t), cutlass.Int32(0)
                )
                ra_xw.append(
                    _ra3_ld2_cg(
                        ra_xq
                        + cutlass.Int64(
                            ra_tt * cutlass.Int32(RA_KO) + cutlass.Int32(8) * tx
                        )
                    )
                )
                ra_sc.append(
                    _ra_scale_cg(
                        ra_xsf
                        + cutlass.Int64(
                            (tx // cutlass.Int32(16)) * cutlass.Int32(SF_ATOM_BYTES)
                            + cutlass.Int32(16) * ra_tt
                            + (tx // cutlass.Int32(4)) % cutlass.Int32(4)
                        )
                    )
                )
            while _ra3_try(ra_mb + cutlass.Int32(8)) == cutlass.Int32(0):
                pass
            if cutlass.const_expr(RC_A0 < RA_E):
                while _ra3_try(ra_mb + cutlass.Int32(16)) == cutlass.Int32(0):
                    pass
            # token passes of <= 4 (8 rows: two passes, W'' read twice from smem; keeps the <= 4-row register shape, no spill)
            ra_tp = min(ra_tok, 4)
            ra_ip = ra_lane >> cutlass.Int32(5 - ((4 * ra_tp).bit_length() - 1))
            ra_rp = (
                ra_lane & cutlass.Int32((1 << (5 - ((4 * ra_tp).bit_length() - 1))) - 1)
            ) == cutlass.Int32(0)
            ra_s = ra_sx + cutlass.Int32(
                4096
            )  # past the r partials (their readers may still be on them)
            for ra_pp in cutlass.range_constexpr(ra_tok // ra_tp):
                for ra_h in cutlass.range_constexpr(2):
                    ra_acc = [[cutlass.Float32(0.0)] * ra_tp for _ in range(4)]
                    for ra_m in cutlass.range_constexpr(RA_KM):
                        ra_w = [
                            _ra_lds4(
                                ra_sb
                                + cutlass.Int32(_rc_a_base(ra_h * 4 + e))
                                + cutlass.Int32(16)
                                * (cutlass.Int32(ra_m * THREADS) + tx)
                            )
                            for e in range(4)
                        ]
                        for tq in cutlass.range_constexpr(ra_tp):
                            t = ra_pp * ra_tp + tq
                            ra_x = [
                                _ra_f("mul.rn", v, ra_sc[t])
                                for v in _ra3_e4m3x4(ra_xw[t][ra_m])
                            ]
                            for e in cutlass.range_constexpr(4):
                                for j in cutlass.range_constexpr(4):
                                    ra_acc[e][tq] = _ra_fma(
                                        ra_w[e][j], ra_x[j], ra_acc[e][tq]
                                    )
                    ra_zz = _ra3_wred(
                        [ra_acc[e][tq] for e in range(4) for tq in range(ra_tp)],
                        ra_lane,
                    )
                    # lane value i = e4 * TP + t' -> output (4 h + e4) * T + pass * TP + t'
                    ra_oo = (
                        (cutlass.Int32(4 * ra_h) + ra_ip // cutlass.Int32(ra_tp))
                        * cutlass.Int32(ra_tok)
                        + cutlass.Int32(ra_pp * ra_tp)
                        + ra_ip % cutlass.Int32(ra_tp)
                    )
                    if ra_rp:
                        _ra_sts(
                            ra_s
                            + cutlass.Int32(4)
                            * (ra_wp * cutlass.Int32(8 * ra_tok) + ra_oo),
                            ra_zz,
                        )
            _ra_bar()
            ra_za = _ra_lds(ra_s + cutlass.Int32(4) * ra_o)
            for ra_k in cutlass.range_constexpr(1, THREADS // 32):
                ra_za = _ra_f(
                    "add.rn",
                    ra_za,
                    _ra_lds(
                        ra_s
                        + cutlass.Int32(4) * (cutlass.Int32(ra_k * 8 * ra_tok) + ra_o)
                    ),
                )
            ra_t = ra_o % cutlass.Int32(ra_tok)
            ra_e = ra_r * cutlass.Int32(RA_E) + ra_o // cutlass.Int32(ra_tok)
            ra_on = cutlass.select_(
                (tx < cutlass.Int32(8 * ra_tok)) & (ra_t < num_tokens),
                cutlass.Int32(1),
                cutlass.Int32(0),
            )
            ra_rx = _ld_global_s64(ra_tab + cutlass.Int64(40))
            ra_w0 = cutlass.Int64(8) * (
                (
                    (
                        cutlass.Int64(ra_t) * cutlass.Int64(ar_world)
                        + cutlass.Int64(ar_rank)
                    )
                    * cutlass.Int64(RA_NPART)
                )
                * cutlass.Int64(128)
                + cutlass.Int64(ra_e)
            )
            # the slot's tag-parity half (allreduce_norm.RouterExchangeSlot:
            # rx maxt == the exchange bucket ar_maxt)
            ra_w0 = ra_w0 + cutlass.Int64(ra_tag & cutlass.Int32(1)) * cutlass.Int64(
                8 * ar_maxt * ar_world * RA_NPART * 128
            )
            ra_zs = _ra_f("add.rn", ra_za, ra_zr)
            for ra_i2 in cutlass.range_constexpr(ar_world):
                _ra_st_tag1(
                    _ld_global_s64(ra_rx + cutlass.Int64(8 * ra_i2)) + ra_w0,
                    ra_tag,
                    ra_zs,
                    ra_on,
                )
            if tx == 0:
                _p5_st_s32(ra_oep, ra_tag)
            _ra_exit()

    if cutlass.const_expr(burst == 2):
        # per-row 1-D bulk prefetch: each of the CTA's 128 weight rows has its k-slice
        # contiguous (contig), one request per row instead of 128-byte box rows; the
        # scale atoms of the slice are one contiguous run
        if warp_id == 3:
            bl = tx % cutlass.Int32(32)
            for br in range(bl, CTA_M, 32):
                _asm(
                    None,
                    [
                        w_addr
                        + cutlass.Int64(m_offset + br) * cutlass.Int64(k_in)
                        + cutlass.Int64(rank * cutlass.Int32(my_tiles * CTA_K)),
                        cutlass.Int32(my_tiles * CTA_K),
                    ],
                    "cp.async.bulk.prefetch.L2.global [$0], $1;",
                    "l,r",
                )
            if bl == cutlass.Int32(0):
                _asm(
                    None,
                    [
                        wsf_addr
                        + cutlass.Int64(
                            n_tile * cutlass.Int32(k_tiles)
                            + rank * cutlass.Int32(my_tiles)
                        )
                        * cutlass.Int64(SF_ATOM_BYTES),
                        cutlass.Int32(my_tiles * SF_ATOM_BYTES),
                    ],
                    "cp.async.bulk.prefetch.L2.global [$0], $1;",
                    "l,r",
                )
    if cutlass.const_expr(burst == 1):
        if warp_id == 3:
            if prims.elect_sync():
                for i in cutlass.range_constexpr(my_tiles):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_ptr_w, [k_global * cutlass.Int32(CTA_K), m_offset], []
                    )
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_ptr_wsf,
                        [cutlass.Int32(0), n_tile * cutlass.Int32(k_tiles) + k_global],
                        [],
                    )
    if warp_id == 0:
        prims.prefetch_tensormap(tma_ptr_w)
        prims.prefetch_tensormap(tma_ptr_wsf)
        prims.prefetch_tensormap(tma_ptr_x)
        prims.prefetch_tensormap(tma_ptr_xsf)
        if prims.elect_sync():
            for s in cutlass.range_constexpr(ring):
                prims.mbarrier_init(w_full.subview(s), 1)
                prims.mbarrier_init(mma_done.subview(s), 1)
            prims.mbarrier_init(act_full, 1)
            prims.mbarrier_init(acc_done, 1)
            if cutlass.const_expr(split > 1 and st_async):
                # Completed by the peers' st.async bytes; this CTA's own arrival carries the expected count.
                prims.mbarrier_init(reduce_ready, 1)
                prims.mbarrier_arrive_expect_tx(
                    reduce_ready, (split - 1) * rows_per_rank * mma_n * 4
                )
            elif cutlass.const_expr(split > 1):
                prims.mbarrier_init(reduce_ready, (split - 1) * rows_per_rank)
            if cutlass.const_expr(early_w and not x_bf16):
                prims.fence_mbarrier_init()
                for i in cutlass.range_constexpr(ring):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
                    prims.mbarrier_arrive_expect_tx(
                        w_full.subview(i), W_TILE_BYTES + SF_ATOM_BYTES
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(i * W_TILE_BYTES),
                        tma_ptr_w,
                        (k_global * cutlass.Int32(CTA_K), m_offset),
                        w_full.subview(i),
                        l2_cache_hint=EVICT_FIRST,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfa.subview(i * SF_ATOM_BYTES),
                        tma_ptr_wsf,
                        (cutlass.Int32(0), n_tile * cutlass.Int32(k_tiles) + k_global),
                        w_full.subview(i),
                        l2_cache_hint=EVICT_FIRST,
                    )
    if warp_id == 2:
        prims.tcgen05_alloc(
            tmem_ptr_i32, tmem_cols(ring, nacc, a_tmem, my_tiles, mma_n)
        )
        prims.tcgen05_relinquish_alloc_permit()
    prims.fence_mbarrier_init()
    if cutlass.const_expr(split > 1):
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)

    if warp_id == 0:
        if prims.elect_sync():
            for i in cutlass.range_constexpr(ring, my_tiles):
                k_global = _k_tile(rank, i, split, my_tiles, contig)
                prims.cp_async_bulk_tensor_prefetch(
                    tma_ptr_w, [k_global * cutlass.Int32(CTA_K), m_offset], []
                )
            if cutlass.const_expr((n_out // CTA_M) * split >= 80):
                # Wide grids launch in part only after the predecessor (e.g. the fused
                # attention) frees its SMs, which loses their pre-wait weight prefetch:
                # the CTAs of the first half also stage their partner's (bx + grid / 2)
                # weights into L2.
                if bx < cutlass.Int32((n_out // CTA_M) * split // 2):
                    pb = bx + cutlass.Int32((n_out // CTA_M) * split // 2)
                    p_rank = pb % cutlass.Int32(split)
                    p_m_offset = (pb // cutlass.Int32(split)) * cutlass.Int32(CTA_M)
                    for i in cutlass.range_constexpr(my_tiles):
                        p_k = _k_tile(p_rank, i, split, my_tiles, contig)
                        prims.cp_async_bulk_tensor_prefetch(
                            tma_ptr_w, [p_k * cutlass.Int32(CTA_K), p_m_offset], []
                        )
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                if cutlass.const_expr(early_w and not x_bf16 and i < ring):
                    continue
                if cutlass.const_expr(x_bf16 and qfirst > 0 and i == qfirst):
                    while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                        pass
                if cutlass.const_expr(i >= ring):
                    while not cute.arch.mbarrier_try_wait(
                        mma_done.subview(stage).data_ptr(), (i // ring - 1) % 2
                    ):
                        pass
                k_global = _k_tile(rank, i, split, my_tiles, contig)
                prims.mbarrier_arrive_expect_tx(
                    w_full.subview(stage), W_TILE_BYTES + SF_ATOM_BYTES
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(stage * W_TILE_BYTES),
                    tma_ptr_w,
                    (k_global * cutlass.Int32(CTA_K), m_offset),
                    w_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_sfa.subview(stage * SF_ATOM_BYTES),
                    tma_ptr_wsf,
                    (cutlass.Int32(0), n_tile * cutlass.Int32(k_tiles) + k_global),
                    w_full.subview(stage),
                    l2_cache_hint=EVICT_FIRST,
                )
    elif warp_id == 1:
        prims.griddepcontrol(prims.GridDepAction.WAIT)
        if cutlass.const_expr(trigger_early):
            # Dependents launch once this grid's own dependency is satisfied (not at
            # entry): consumers that read step metadata or caches before their own grid
            # wait rely on every grid ahead of them having passed its wait, so the chain
            # of releases never runs ahead of a producer still in flight.
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if cutlass.const_expr(not x_bf16):
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    act_full, my_tiles * (x_tile_bytes + SF_ATOM_BYTES)
                )
                for i in cutlass.range_constexpr(my_tiles):
                    k_global = _k_tile(rank, i, split, my_tiles, contig)
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_b.subview(i * x_tile_bytes),
                        tma_ptr_x,
                        (k_global * cutlass.Int32(CTA_K), cutlass.Int32(0)),
                        act_full,
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_sfb.subview(i * SF_ATOM_BYTES),
                        tma_ptr_xsf,
                        (cutlass.Int32(0), k_global),
                        act_full,
                    )
    elif warp_id == 2:
        tmem_raw = tmem_ptr_i32.load()
        base_col = tmem_raw & 0xFFFF
        base_row = tmem_raw >> 16
        sfa_col0 = base_col + acc_cols(nacc, mma_n)
        sfb_col0 = sfa_col0 + ring * SF_COLS
        idesc = prims.Tcgen05MxInstrDesc.build(
            a_dtype=a_dtype, b_dtype=b_dtype, scale_format=1, n_dim=mma_n, m_dim=CTA_M
        )
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        if cutlass.const_expr(a_tmem):
            # A staged in TMEM: before the grid wait every weight k-tile and its scale
            # atom go on into TMEM as they land (the copies of a stage refilled later
            # are committed to its mma_done); once the activation lands, its scale atoms
            # and the MMAs, A read from TMEM.
            base_row = tmem_raw >> 16
            base_col = tmem_raw & 0xFFFF
            sfa_col0 = base_col + acc_cols(nacc, mma_n)
            sfb_col0 = sfa_col0 + my_tiles * SF_COLS
            a_base = tmem_raw + cutlass.Int32(a_col0(nacc, my_tiles, mma_n))
            s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                while not cute.arch.mbarrier_try_wait(
                    w_full.subview(stage).data_ptr(), (i // ring) % 2
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                desc_a = prims.Tcgen05SmemDesc.build(
                    smem_a.subview(stage * W_TILE_BYTES), leading_byte_offset=16, stride_byte_offset=1024,
                    base_offset=0, layout=2,
                )  # fmt: skip
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    smem_sfa.subview(stage * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                if prims.elect_sync():
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        prims.tcgen05_cp(
                            prims.Tcgen05CpShape.SHAPE_128X256B,
                            cutlass.inttoptr(
                                a_base
                                + cutlass.Int32((i * K_BLOCKS + kb) * A_COLS_PER_MMA),
                                6,
                                cutlass.Int32,
                            ),
                            desc_a + ((MMA_INST_K >> 4) * kb),
                            group=prims.CTAGroup.CTA_1,
                        )
                    prims.tcgen05_cp(
                        s2t_shape,
                        cutlass.inttoptr((base_row << 16) | (sfa_col0 + i * SF_COLS), 6, cutlass.Int32),
                        desc_sfa, multicast=s2t_multicast,
                    )  # fmt: skip
                    if cutlass.const_expr(i + ring < my_tiles):
                        prims.tcgen05_commit(mma_done.subview(stage))
            if cutlass.const_expr(x_bf16):
                # Completed by a thread's arrive: a suspended try_wait wakes late on those.
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                    pass
            else:
                while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                    pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                for i in cutlass.range_constexpr(my_tiles):
                    desc_sfb = prims.Tcgen05SmemDesc.build(
                        smem_sfb.subview(i * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                        base_offset=0, layout=0,
                    )  # fmt: skip
                    desc_b = prims.Tcgen05SmemDesc.build(
                        smem_b.subview(i * x_tile_bytes), leading_byte_offset=16, stride_byte_offset=1024,
                        base_offset=0, layout=2,
                    )  # fmt: skip
                    sfa_tmem = cutlass.inttoptr(
                        (base_row << 16) | (sfa_col0 + i * SF_COLS), 6, cutlass.Int32
                    )
                    sfb_tmem = cutlass.inttoptr(
                        (base_row << 16) | sfb_col0, 6, cutlass.Int32
                    )
                    prims.tcgen05_cp(
                        s2t_shape, sfb_tmem, desc_sfb, multicast=s2t_multicast
                    )
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw, 6, cutlass.Float32),
                            cutlass.inttoptr(
                                a_base + cutlass.Int32((i * K_BLOCKS + kb) * A_COLS_PER_MMA), 6, cutlass.Int32
                            ),
                            desc_b + ((MMA_INST_K >> 4) * kb), idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb),
                            i > 0 or kb > 0,
                            sfa_tmem, sfb_tmem,
                        )  # fmt: skip
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)
        else:
            if cutlass.const_expr(x_bf16):
                # Completed by a thread's arrive: a suspended try_wait wakes late on those.
                while not cute.arch.mbarrier_test_wait(act_full.data_ptr(), 0):
                    pass
            else:
                while not cute.arch.mbarrier_try_wait(act_full.data_ptr(), 0):
                    pass
            for i in cutlass.range_constexpr(my_tiles):
                stage = i % ring
                while not cute.arch.mbarrier_try_wait(
                    w_full.subview(stage).data_ptr(), (i // ring) % 2
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                sfa_tmem = (base_row << 16) | (sfa_col0 + stage * SF_COLS)
                sfb_tmem = (base_row << 16) | (sfb_col0 + stage * SF_COLS)
                desc_sfa = prims.Tcgen05SmemDesc.build(
                    smem_sfa.subview(stage * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                desc_sfb = prims.Tcgen05SmemDesc.build(
                    smem_sfb.subview(i * SF_ATOM_BYTES), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                desc_a = prims.Tcgen05SmemDesc.build(
                    smem_a.subview(stage * W_TILE_BYTES), leading_byte_offset=16, stride_byte_offset=1024,
                    base_offset=0, layout=2,
                )  # fmt: skip
                desc_b = prims.Tcgen05SmemDesc.build(
                    smem_b.subview(i * x_tile_bytes), leading_byte_offset=16, stride_byte_offset=1024,
                    base_offset=0, layout=2,
                )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_cp(s2t_shape, cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32), desc_sfa,
                                     multicast=s2t_multicast)  # fmt: skip
                    prims.tcgen05_cp(s2t_shape, cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32), desc_sfb,
                                     multicast=s2t_multicast)  # fmt: skip
                    for kb in cutlass.range_constexpr(K_BLOCKS):
                        inc = ((MMA_INST_K * 8 // 8) >> 4) * kb
                        j = i * K_BLOCKS + kb
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1,
                            cutlass.inttoptr(tmem_raw + cutlass.Int32((j % nacc) * mma_n), 6, cutlass.Float32),
                            desc_a + inc, desc_b + inc, idesc.set_sf_ids(a_sf_id=kb, b_sf_id=kb), j >= nacc,
                            cutlass.inttoptr(sfa_tmem, 6, cutlass.Int32), cutlass.inttoptr(sfb_tmem, 6, cutlass.Int32),
                        )  # fmt: skip
                    prims.tcgen05_commit(mma_done.subview(stage))
            if prims.elect_sync():
                prims.tcgen05_commit(acc_done)
    elif warp_id >= 4:
        lane = tx % 32
        row = (warp_id % 4) * 32 + lane
        if cutlass.const_expr(x_bf16):
            # The bf16 activation of this CTA's k-tiles to MXFP8 in shared memory, as
            # torch.ops.trtllm.mxfp8_quantize(x, True) does (UE8M0 = round-up(qz_amax *
            # rcp(448)), E4M3 RN-satfinite): one 32-element block (token qz_t, k-tile
            # qz_i, block qz_kb) per thread and round, the E4M3 bytes into the
            # 128-byte-swizzled B tile, the qz_scale byte into the tile's qz_scale atom.
            qz_tid = tx - cutlass.Int32(128)
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            din_half_k = k_in // 2
            din_hidden_half = cutlass.Int32(0)
            if cutlass.const_expr(din):
                # Draft input x = [rmsnorm(table[ids]) * w_e, rmsnorm(hidden) * w_h] (an
                # out-of-range id embeds to zero): this CTA's k-tiles lie in one half,
                # so it needs that half's statistic of every token. 16 threads per
                # token, 16-byte loads, all of a thread's loads in flight at once.
                din_hidden_half = cutlass.select_(
                    rank * cutlass.Int32(my_tiles * CTA_K) >= cutlass.Int32(din_half_k),
                    cutlass.Int32(1),
                    cutlass.Int32(0),
                )
                din_tok = qz_tid // cutlass.Int32(16)
                din_part = qz_tid % cutlass.Int32(16)
                din_tok_ld = cutlass.select_(
                    din_tok < num_tokens, din_tok, cutlass.Int32(0)
                )
                din_id = din_i.load(idx=din_tok_ld)
                din_valid = (din_id >= cutlass.Int32(0)) & (din_id < din_vocab)
                din_hrow = din_tok_ld
                if (din_flags & cutlass.Int32(1)) != cutlass.Int32(0):
                    din_hrow = din_g.load(idx=din_tok_ld * cutlass.Int32(2))
                din_row = cutlass.select_(
                    din_hidden_half == cutlass.Int32(1),
                    din_hrow,
                    cutlass.select_(din_valid, din_id, cutlass.Int32(0)),
                )
                # The first hidden-half CTA of tile 0 reads whole hidden rows here: it also writes them out.
                din_copy = (
                    ((din_flags & cutlass.Int32(2)) != cutlass.Int32(0))
                    & (bx == cutlass.Int32(split // 2))
                    & (din_hidden_half == cutlass.Int32(1))
                    & (din_tok < num_tokens)
                )
                din_base = din_row * cutlass.Int32(din_half_k // 2)
                din_ss = cutlass.Float32(0.0)
                din_vecs_per_part = din_half_k // 8 // 16
                for din_round in cutlass.range_constexpr(0, din_vecs_per_part, 16):
                    din_n = min(16, din_vecs_per_part - din_round)
                    din_loaded = []
                    for din_j in cutlass.range_constexpr(din_n):
                        din_widx = din_base + (
                            din_part + cutlass.Int32((din_round + din_j) * 16)
                        ) * cutlass.Int32(4)
                        din_loaded.append(
                            _din_load4(din_table, din_hidden, din_hidden_half, din_widx)
                        )
                    if din_copy:
                        for din_j in cutlass.range_constexpr(din_n):
                            din_v = din_loaded[din_j]
                            din_res.store(
                                (din_v[0], din_v[1], din_v[2], din_v[3]),
                                idx=din_tok * cutlass.Int32(din_half_k // 2)
                                + (din_part + cutlass.Int32((din_round + din_j) * 16))
                                * cutlass.Int32(4),
                                alignment=16,
                            )
                    for din_j in cutlass.range_constexpr(din_n):
                        for din_e in cutlass.range_constexpr(4):
                            din_word = cutlass.Int32(din_loaded[din_j][din_e])
                            din_lo = (din_word << cutlass.Int32(16)).bitcast(
                                cutlass.Float32
                            )
                            din_hi = (din_word & cutlass.Int32(-65536)).bitcast(
                                cutlass.Float32
                            )
                            din_ss = din_ss + din_lo * din_lo + din_hi * din_hi
                for din_o in cutlass.range_constexpr(4):
                    din_ss = din_ss + cute.arch.shuffle_sync_bfly(
                        din_ss, offset=1 << din_o
                    )
                if din_part == cutlass.Int32(0):
                    din_eps = din_f.load(idx=din_hidden_half)
                    din_r = cute.math.rsqrt(
                        din_ss / cutlass.Float32(din_half_k) + din_eps, fastmath=True
                    )
                    din_keep = din_valid | (din_hidden_half == cutlass.Int32(1))
                    din_rs.store(  # type: ignore[union-attr]
                        cutlass.select_(din_keep, din_r, cutlass.Float32(0.0)),
                        idx=din_tok,
                    )
                prims.barrier_cta_sync(3, thread_count=128)
            qz_b_words = cutlass.Array(smem_b.data_ptr(0), shape=(my_tiles * x_tile_bytes // 4,), dtype=cutlass.Int32,
                                    alignment=16)  # fmt: skip
            qz_items = my_tiles * K_BLOCKS * mma_n
            qz_rounds = (qz_items + 127) // 128
            # A group's loads first (clamped to a valid row), then its math: one L2
            # round trip per group of QZ_GROUP rounds instead of one per round, without
            # holding every round's data in registers.
            QZ_GROUP = qz_rounds if qz_rounds <= 8 else 4
            qz_loaded = []
            qz_stag = stagger > 0 and qz_items % 128 == 0
            qz_off = cutlass.Int32(0)
            if cutlass.const_expr(qz_stag):
                qz_off = (n_tile * cutlass.Int32(stagger)) % cutlass.Int32(qz_items)
            for qz_r in cutlass.range_constexpr(qz_rounds):
                qz_item = qz_tid + cutlass.Int32(qz_r * 128)
                if cutlass.const_expr(qz_stag):
                    qz_item = (qz_item + qz_off) % cutlass.Int32(qz_items)
                qz_t = qz_item % cutlass.Int32(mma_n)
                qz_kb = (qz_item // cutlass.Int32(mma_n)) % cutlass.Int32(K_BLOCKS)
                qz_i = qz_item // cutlass.Int32(mma_n * K_BLOCKS)
                if cutlass.const_expr(qz_r * 128 + 127 >= qz_items):
                    qz_i = cutlass.select_(
                        qz_item < cutlass.Int32(qz_items), qz_i, cutlass.Int32(0)
                    )
                qz_k_global = rank + qz_i * cutlass.Int32(split)
                if cutlass.const_expr(contig):
                    qz_k_global = rank * cutlass.Int32(my_tiles) + qz_i
                qz_t_ld = cutlass.select_(qz_t < num_tokens, qz_t, cutlass.Int32(0))
                qz_x_base = (
                    qz_t_ld * cutlass.Int32(k_in // 2)
                    + qz_k_global * cutlass.Int32(CTA_K // 2)
                    + qz_kb * cutlass.Int32(MMA_INST_K // 2)
                )
                qz_wvecs = None
                if cutlass.const_expr(din):
                    # The draft input's raw values (table row of the id / hidden row) and its norm weights.
                    qz_col = qz_k_global * cutlass.Int32(CTA_K) + qz_kb * cutlass.Int32(
                        MMA_INST_K
                    )
                    qz_hcol = qz_col - din_hidden_half * cutlass.Int32(din_half_k)
                    qz_id = din_i.load(idx=qz_t_ld)
                    qz_hrow = qz_t_ld
                    if (din_flags & cutlass.Int32(1)) != cutlass.Int32(0):
                        qz_hrow = din_g.load(idx=qz_t_ld * cutlass.Int32(2))
                    qz_row = cutlass.select_(
                        din_hidden_half == cutlass.Int32(1),
                        qz_hrow,
                        cutlass.select_(
                            (qz_id >= cutlass.Int32(0)) & (qz_id < din_vocab),
                            qz_id,
                            cutlass.Int32(0),
                        ),
                    )
                    qz_src_w = qz_row * cutlass.Int32(
                        din_half_k // 2
                    ) + qz_hcol // cutlass.Int32(2)
                    qz_vecs = [
                        _din_load4(
                            din_table,
                            din_hidden,
                            din_hidden_half,
                            qz_src_w + cutlass.Int32(qz_v * 4),
                        )
                        for qz_v in range(4)
                    ]
                    qz_wvecs = [
                        din_w.load(
                            idx=qz_col // cutlass.Int32(2) + cutlass.Int32(qz_v * 4),
                            vector_size=4,
                            alignment=16,
                        )
                        for qz_v in range(4)
                    ]
                elif cutlass.const_expr(ar_poll):
                    qz_xa = x_words.data_ptr(0).toint() + cutlass.Int64(
                        4
                    ) * cutlass.Int64(qz_x_base)
                    qz_vecs = [
                        [
                            _px_poll(qz_xa + cutlass.Int64(4 * (qz_v * 4 + qz_e)))
                            for qz_e in range(4)
                        ]
                        for qz_v in range(4)
                    ]
                else:
                    qz_vecs = [
                        x_words.load(
                            idx=qz_x_base + cutlass.Int32(qz_v * 4),
                            vector_size=4,
                            alignment=16,
                        )
                        for qz_v in range(4)
                    ]
                qz_loaded.append((qz_item, qz_t, qz_kb, qz_i, qz_vecs, qz_wvecs))
                if cutlass.const_expr(
                    (qz_r + 1) % QZ_GROUP == 0 or qz_r + 1 == qz_rounds
                ):
                    for qz_g in cutlass.range_constexpr(len(qz_loaded)):
                        qz_item, qz_t, qz_kb, qz_i, qz_vecs, qz_wvecs = qz_loaded[qz_g]
                        qz_vals = []
                        qz_amax = cutlass.Float32(0.0)
                        if cutlass.const_expr(din):
                            qz_rs = din_rs.load(  # type: ignore[union-attr]
                                idx=cutlass.select_(
                                    qz_t < num_tokens, qz_t, cutlass.Int32(0)
                                )
                            )
                        for qz_v in cutlass.range_constexpr(4):
                            for qz_e4 in cutlass.range_constexpr(4):
                                qz_word = cutlass.Int32(qz_vecs[qz_v][qz_e4])
                                qz_lo = (qz_word << cutlass.Int32(16)).bitcast(
                                    cutlass.Float32
                                )
                                qz_hi = (qz_word & cutlass.Int32(-65536)).bitcast(
                                    cutlass.Float32
                                )
                                if cutlass.const_expr(din):
                                    # bf16(x * rs * w), the draft-input kernel's rounding.
                                    qz_wword = cutlass.Int32(qz_wvecs[qz_v][qz_e4])
                                    qz_lo = _bf16_round(
                                        qz_lo
                                        * qz_rs
                                        * (qz_wword << cutlass.Int32(16)).bitcast(
                                            cutlass.Float32
                                        )
                                    )
                                    qz_hi = _bf16_round(
                                        qz_hi
                                        * qz_rs
                                        * (qz_wword & cutlass.Int32(-65536)).bitcast(
                                            cutlass.Float32
                                        )
                                    )
                                qz_vals.append(qz_lo)
                                qz_vals.append(qz_hi)
                                qz_amax = _absmax(_absmax(qz_amax, qz_lo), qz_hi)
                        qz_bits = (
                            qz_amax * _rcp_approx_ftz(cutlass.Float32(448.0))
                        ).bitcast(cutlass.Int32)
                        qz_e = (
                            (qz_bits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
                        ) + cutlass.select_(
                            (qz_bits & cutlass.Int32(0x7FFFFF)) != cutlass.Int32(0),
                            cutlass.Int32(1),
                            cutlass.Int32(0),
                        )
                        qz_e = cutlass.select_(
                            qz_e > cutlass.Int32(254), cutlass.Int32(254), qz_e
                        )
                        qz_scale = (
                            (cutlass.Int32(254) - qz_e) << cutlass.Int32(23)
                        ).bitcast(cutlass.Float32)
                        qz_scale = cutlass.select_(
                            qz_amax != cutlass.Float32(0.0),
                            qz_scale,
                            cutlass.Float32(0.0),
                        )
                        qz_q = [_e4m3x4(qz_vals[4 * qz_j] * qz_scale, qz_vals[4 * qz_j + 1] * qz_scale,
                                        qz_vals[4 * qz_j + 2] * qz_scale, qz_vals[4 * qz_j + 3] * qz_scale)
                                for qz_j in range(8)]  # fmt: skip
                        if qz_item < cutlass.Int32(qz_items):
                            if qz_t < num_tokens:
                                qz_row_w = qz_i * cutlass.Int32(
                                    x_tile_bytes // 4
                                ) + qz_t * cutlass.Int32(CTA_K // 4)
                                qz_c0 = qz_kb * cutlass.Int32(2)
                                qz_b_words.store(
                                    (qz_q[0], qz_q[1], qz_q[2], qz_q[3]),
                                    idx=qz_row_w
                                    + (qz_c0 ^ (qz_t & cutlass.Int32(7)))
                                    * cutlass.Int32(4),
                                    alignment=16,
                                )
                                qz_b_words.store(
                                    (qz_q[4], qz_q[5], qz_q[6], qz_q[7]),
                                    idx=qz_row_w
                                    + (
                                        (qz_c0 + cutlass.Int32(1))
                                        ^ (qz_t & cutlass.Int32(7))
                                    )
                                    * cutlass.Int32(4),
                                    alignment=16,
                                )
                                smem_sfb.store(
                                    cutlass.Uint8(qz_e),
                                    idx=qz_i * cutlass.Int32(SF_ATOM_BYTES)
                                    + (qz_t % cutlass.Int32(32)) * cutlass.Int32(16)
                                    + (qz_t // cutlass.Int32(32)) * cutlass.Int32(4)
                                    + qz_kb,
                                )
                    qz_loaded = []
            # Generic-proxy stores made visible to the MMA's async-proxy reads, then one arrival for all.
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(2, thread_count=128)
            if qz_tid == cutlass.Int32(0):
                prims.mbarrier_arrive(act_full)
        if cutlass.const_expr(pfx and ar_push == 4 and split > 1):
            # the push's parity flags and peer bases resolve while the MMA runs, not after the split-K reduce
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            pfx_n = m_offset + row
            pfx_par = [
                _ld_volatile_s32(
                    ar_flags
                    + cutlass.Int64(4)
                    * (
                        cutlass.Int64(t * 4)
                        + cutlass.Int64(pfx_n // cutlass.Int32(n_out // 4))
                    )
                )
                & cutlass.Int32(1)
                for t in range(mma_n)
            ]
            pfx_base = [
                _ld_global_s64(ar_ws + cutlass.Int64(8 * r)) for r in range(ar_world)
            ]
        if cutlass.const_expr(st_async):
            while not cute.arch.mbarrier_test_wait(acc_done.data_ptr(), 0):
                pass
        else:
            while not cute.arch.mbarrier_try_wait(acc_done.data_ptr(), 0):
                pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        # tcgen05.ld widths are powers of two: the accumulator columns in descending power-of-two pieces.
        acc_all = []  # type: ignore[var-annotated]
        ld_col = 0
        for ld_w in (128, 64, 32, 16, 8):
            if cutlass.const_expr((mma_n * nacc - ld_col) >= ld_w):
                piece = prims.tcgen05_ld(
                    "32x32b",
                    cutlass.inttoptr(
                        tmem_ptr_i32.load() + cutlass.Int32(ld_col), 6, cutlass.Float32
                    ),
                    num=ld_w,
                )
                acc_all = acc_all + [piece[c] for c in range(ld_w)]
                ld_col += ld_w
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        acc = []
        for t in cutlass.range_constexpr(mma_n):
            v = cutlass.Float32(acc_all[t])
            for a in cutlass.range_constexpr(1, nacc):
                v = v + cutlass.Float32(acc_all[a * mma_n + t])
            acc.append(v)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        n = m_offset + row
        if cutlass.const_expr(split == 1):
            for t in cutlass.range_constexpr(mma_n):
                if cutlass.Int32(t) < num_tokens:
                    y.store(
                        cutlass.BFloat16(cutlass.Float32(acc[t])),
                        idx=cutlass.Int32(t * n_out) + n,
                    )
            if cutlass.const_expr(ar_push == 3):
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                g3_buf = _ld_volatile_s32(ar_flags) % cutlass.Int32(3)
                g3_base = [
                    _ld_global_s64(ar_ws + cutlass.Int64(8 * r))
                    for r in range(ar_world)
                ]
                for t in cutlass.range_constexpr(mma_n):
                    g3_v = cutlass.Float32(acc[t])
                    g3_hi = cute.arch.shuffle_sync_bfly(g3_v, offset=1)
                    if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                        if cutlass.Int32(t) < num_tokens:
                            g3_off = (
                                (
                                    cutlass.Int64(g3_buf) * cutlass.Int64(ar_world)
                                    + cutlass.Int64(ar_rank)
                                )
                                * cutlass.Int64(ar_maxt)
                                + cutlass.Int64(t)
                            ) * cutlass.Int64(n_out) + cutlass.Int64(n)
                            g3_word = _bf16x2_word(g3_v, g3_hi)
                            for r in cutlass.range_constexpr(ar_world):
                                _st_global_b32(
                                    g3_base[r] + g3_off * cutlass.Int64(2), g3_word
                                )
            if cutlass.const_expr(ar_push == 6):
                # mode 6: mode 4 for an N-slice GEMV (column halves): ar_rank = rank |
                # (column offset << 8), row stride and slices of the full 6144-wide
                # boundary row.
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                p6_rk = ar_rank & cutlass.Int32(0xFF)
                p6_n = n + (ar_rank >> cutlass.Int32(8))
                p6_par = [
                    _ld_volatile_s32(
                        ar_flags
                        + cutlass.Int64(4)
                        * (
                            cutlass.Int64(t * 4)
                            + cutlass.Int64(p6_n // cutlass.Int32(1536))
                        )
                    )
                    & cutlass.Int32(1)
                    for t in range(mma_n)
                ]
                p6_base = [
                    _ld_global_s64(ar_ws + cutlass.Int64(8 * r))
                    for r in range(ar_world)
                ]
                for t in cutlass.range_constexpr(mma_n):
                    p6_v = cutlass.Float32(acc[t])
                    p6_hi = cute.arch.shuffle_sync_bfly(p6_v, offset=1)
                    if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                        if cutlass.Int32(t) < num_tokens:
                            p6_off = (
                                cutlass.Int64(p6_par[t])
                                * cutlass.Int64(ar_maxt * ar_world * 3072)
                                + (cutlass.Int64(t * ar_world) + cutlass.Int64(p6_rk))
                                * cutlass.Int64(3072)
                                + (cutlass.Int64(p6_n) // cutlass.Int64(2))
                            )
                            p6_word = _bf16x2_word(p6_v, p6_hi)
                            for r in cutlass.range_constexpr(ar_world):
                                _st_global_b32(
                                    p6_base[r] + p6_off * cutlass.Int64(4), p6_word
                                )
            if cutlass.const_expr(ar_push == 4):
                # the boundary exchange slots (exact mode, 4 column slices per row), unicast into every rank's exchange.
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                p4_par = [
                    _ld_volatile_s32(
                        ar_flags
                        + cutlass.Int64(4)
                        * (
                            cutlass.Int64(t * 4)
                            + cutlass.Int64(n // cutlass.Int32(n_out // 4))
                        )
                    )
                    & cutlass.Int32(1)
                    for t in range(mma_n)
                ]
                p4_base = [
                    _ld_global_s64(ar_ws + cutlass.Int64(8 * r))
                    for r in range(ar_world)
                ]
                for t in cutlass.range_constexpr(mma_n):
                    p4_v = cutlass.Float32(acc[t])
                    p4_hi = cute.arch.shuffle_sync_bfly(p4_v, offset=1)
                    if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                        if cutlass.Int32(t) < num_tokens:
                            p4_off = (
                                cutlass.Int64(p4_par[t])
                                * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                + (cutlass.Int64(t * ar_world) + cutlass.Int64(ar_rank))
                                * cutlass.Int64(n_out // 2)
                                + (cutlass.Int64(n) // cutlass.Int64(2))
                            )
                            p4_word = _bf16x2_word(p4_v, p4_hi)
                            for r in cutlass.range_constexpr(ar_world):
                                _st_global_b32(
                                    p4_base[r] + p4_off * cutlass.Int64(4), p4_word
                                )
            if cutlass.const_expr(ar_push > 0 and ar_push < 3):
                prims.griddepcontrol(prims.GridDepAction.WAIT)
                # The parity of each (token, column slice) slot: the all-reduce's CTA t
                # * 8 + slice flag. All loads are issued before the first store so they
                # overlap (one L2 round trip, not one per token).
                ar_par = [
                    _ld_volatile_s32(
                        ar_flags
                        + cutlass.Int64(4)
                        * (
                            cutlass.Int64(t * 8)
                            + cutlass.Int64(n // cutlass.Int32(n_out // 8))
                        )
                    )
                    & cutlass.Int32(1)
                    for t in range(mma_n)
                ]
                for t in cutlass.range_constexpr(mma_n):
                    ar_v = cutlass.Float32(acc[t])
                    ar_hi = cute.arch.shuffle_sync_bfly(ar_v, offset=1)
                    if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                        if cutlass.Int32(t) < num_tokens:
                            ar_off = (
                                cutlass.Int64(ar_par[t])
                                * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                + (cutlass.Int64(t * ar_world) + cutlass.Int64(ar_rank))
                                * cutlass.Int64(n_out // 2)
                                + (cutlass.Int64(n) // cutlass.Int64(2))
                            )
                            _st_global_b32(
                                ar_ws + ar_off * cutlass.Int64(4),
                                _bf16x2_word(ar_v, ar_hi),
                            )
        else:
            owner = row // cutlass.Int32(rows_per_rank)
            local = row % cutlass.Int32(rows_per_rank)
            if cutlass.const_expr(rope):
                # Rank r owns head dims [r * P, (r + 1) * P) and their rotation partners 64 + ..., P = rows / 2.
                pairs = rows_per_rank // 2
                owner = (row % cutlass.Int32(64)) // cutlass.Int32(pairs)
                local = (row // cutlass.Int32(64)) * cutlass.Int32(pairs) + (
                    row % cutlass.Int32(64)
                ) % cutlass.Int32(pairs)
            if owner != rank:
                base = (rank * cutlass.Int32(rows_per_rank) + local) * cutlass.Int32(
                    mb_stride
                )
                if cutlass.const_expr(st_async):
                    dst = _mapa_u32(mailbox.subview(base).data_ptr(), owner)
                    mb = _mapa_u32(reduce_ready.data_ptr(), owner)
                    words = [
                        cutlass.Float32(acc[t]).bitcast(cutlass.Int32)
                        for t in range(mma_n)
                    ]
                    for g in cutlass.range_constexpr(mma_n // 4):
                        _st_async_v4(
                            dst + cutlass.Int32(16 * g),
                            words[4 * g],
                            words[4 * g + 1],
                            words[4 * g + 2],
                            words[4 * g + 3],
                            mb,
                        )
                else:
                    for t in cutlass.range_constexpr(mma_n):
                        prims.mapa(
                            mailbox.subview(base + cutlass.Int32(t)), owner
                        ).store(cutlass.Float32(acc[t]))
                    prims.mbarrier_arrive(
                        prims.mapa(reduce_ready, owner), scope=prims.MemScope.CLUSTER
                    )
            else:
                if cutlass.const_expr(st_async):
                    while not _test_wait_cluster(reduce_ready.data_ptr(), 0):
                        pass
                else:
                    while not prims.mbarrier_try_wait_parity(
                        reduce_ready, 0, scope=prims.MBarrierScope.CLUSTER
                    ):
                        pass
                totals = []
                for g in cutlass.range_constexpr(mma_n // 4):
                    sums = [cutlass.Float32(0.0)] * 4
                    for q in cutlass.range_constexpr(split):
                        parts = _ld_shared_v4_f32(
                            mailbox.subview(
                                (cutlass.Int32(q * rows_per_rank) + local)
                                * cutlass.Int32(mb_stride)
                                + cutlass.Int32(4 * g)
                            ).data_ptr()
                        )
                        sums = [sums[j] + cutlass.select_(rank == cutlass.Int32(q), cutlass.Float32(acc[4 * g + j]),
                                                          parts[j]) for j in range(4)]  # fmt: skip
                    for j in cutlass.range_constexpr(4):
                        totals.append(sums[j])
                        if cutlass.const_expr(rope):
                            # The own-rank mailbox slot is unused: it carries the totals to the RoPE stage.
                            mailbox.store(
                                sums[j],
                                idx=(rank * cutlass.Int32(rows_per_rank) + local)
                                * cutlass.Int32(mb_stride)
                                + cutlass.Int32(4 * g + j),
                            )
                        elif cutlass.Int32(4 * g + j) < num_tokens:
                            y.store(
                                cutlass.BFloat16(sums[j]),
                                idx=cutlass.Int32((4 * g + j) * n_out) + n,
                            )
                if cutlass.const_expr(ar_push == 3):
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    g3_buf = _ld_volatile_s32(ar_flags) % cutlass.Int32(3)
                    g3_base = [
                        _ld_global_s64(ar_ws + cutlass.Int64(8 * r))
                        for r in range(ar_world)
                    ]
                    for t in cutlass.range_constexpr(mma_n):
                        g3_hi = cute.arch.shuffle_sync_bfly(totals[t], offset=1)
                        if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                            if cutlass.Int32(t) < num_tokens:
                                g3_off = (
                                    (
                                        cutlass.Int64(g3_buf) * cutlass.Int64(ar_world)
                                        + cutlass.Int64(ar_rank)
                                    )
                                    * cutlass.Int64(ar_maxt)
                                    + cutlass.Int64(t)
                                ) * cutlass.Int64(n_out) + cutlass.Int64(n)
                                g3_word = _bf16x2_word(totals[t], g3_hi)
                                for r in cutlass.range_constexpr(ar_world):
                                    _st_global_b32(
                                        g3_base[r] + g3_off * cutlass.Int64(2), g3_word
                                    )
                if cutlass.const_expr(ar_push == 5):
                    # mode 5 (tagged-word wire format): ar_ws = descriptor [recv_local,
                    # peer[0..7], ctr, rowsq, exit, res_in, res_out, gamma, out_q,
                    # out_sf, normed, eps_bits, grid] (int64). Owner thread = one column
                    # n for every token; an owner warp = one 32-column MXFP8 block /
                    # rowsq tile.
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    p5_peer = [
                        _ld_global_s64(ar_ws + cutlass.Int64(8 * (1 + r)))
                        for r in range(ar_world)
                    ]
                    p5_recv = _ld_global_s64(ar_ws)
                    p5_round = _ld_volatile_s32(
                        _ld_global_s64(ar_ws + cutlass.Int64(72))
                    )
                    p5_rsq = _ld_global_s64(ar_ws + cutlass.Int64(80))
                    p5_res_in = _ld_global_s64(ar_ws + cutlass.Int64(96))
                    p5_res_out = _ld_global_s64(ar_ws + cutlass.Int64(104))
                    p5_gamma = _ld_global_s64(ar_ws + cutlass.Int64(112))
                    p5_q = _ld_global_s64(ar_ws + cutlass.Int64(120))
                    p5_sf = _ld_global_s64(ar_ws + cutlass.Int64(128))
                    p5_nrm = _ld_global_s64(ar_ws + cutlass.Int64(136))
                    p5_eps = cutlass.Int32(
                        _ld_global_s64(ar_ws + cutlass.Int64(144))
                    ).bitcast(cutlass.Float32)
                    p5_tag = p5_round % cutlass.Int32(65535) + cutlass.Int32(1)
                    p5_tile = n // cutlass.Int32(32)
                    # 1. push this rank's bf16 partial (own area too) as (tag << 16) | bf16
                    for t in cutlass.range_constexpr(mma_n):
                        if cutlass.Int32(t) < num_tokens:
                            p5_w = (p5_tag << cutlass.Int32(16)) | (
                                (_bf16_bits(totals[t]) >> cutlass.Int32(16))
                                & cutlass.Int32(0xFFFF)
                            )
                            p5_off = (
                                cutlass.Int64(
                                    ar_rank * cutlass.Int32(16) + cutlass.Int32(t)
                                )
                                * cutlass.Int64(n_out)
                                + cutlass.Int64(n)
                            ) * cutlass.Int64(4)
                            for r in cutlass.range_constexpr(ar_world):
                                _p5_st_sys(p5_peer[r] + p5_off, p5_w)
                    p5_gw = _p5_ld_bf16(p5_gamma + cutlass.Int64(2) * cutlass.Int64(n))
                    p5_g1 = _p5_f("add.rn", cutlass.Float32(1.0), p5_gw)
                    p5_act = [
                        cutlass.select_(
                            cutlass.Int32(t) < num_tokens,
                            cutlass.Int32(1),
                            cutlass.Int32(0),
                        )
                        for t in range(mma_n)
                    ]
                    p5_eo = [
                        cutlass.Int64(t) * cutlass.Int64(n_out) + cutlass.Int64(n)
                        for t in range(mma_n)
                    ]
                    # 2. all ranks' words of all tokens: one batch of loads, then re-poll stragglers only
                    p5_adr = [
                        [
                            p5_recv
                            + (
                                cutlass.Int64(r * 16 + t) * cutlass.Int64(n_out)
                                + cutlass.Int64(n)
                            )
                            * cutlass.Int64(4)
                            for r in range(ar_world)
                        ]
                        for t in range(mma_n)
                    ]
                    p5_raw = [
                        [_ld_volatile_s32(p5_adr[t][r]) for r in range(ar_world)]
                        for t in range(mma_n)
                    ]
                    p5_res = [
                        _p5_ld_bf16(p5_res_in + cutlass.Int64(2) * p5_eo[t])
                        for t in range(mma_n)
                    ]
                    p5_u = []
                    p5_sq = []
                    for t in cutlass.range_constexpr(mma_n):
                        p5_s = cutlass.Float32(0.0)
                        for r in cutlass.range_constexpr(ar_world):
                            p5_x = _p5_fix(
                                p5_adr[t][r],
                                p5_tag,
                                p5_raw[t][r],
                                p5_act[t],
                            )
                            p5_f = (p5_x << cutlass.Int32(16)).bitcast(cutlass.Float32)
                            p5_s = p5_f if r == 0 else _p5_f("add.rn", p5_s, p5_f)
                        # 3. + residual (stock: bf16(bf16(sum) + residual))
                        u = _bf16_round(_p5_f("add.rn", _bf16_round(p5_s), p5_res[t]))
                        p5_u.append(u)
                        q = _p5_f("mul.rn", u, u)
                        for o in (16, 8, 4, 2, 1):
                            q = _p5_f(
                                "add.rn", q, cute.arch.shuffle_sync_bfly(q, offset=o)
                            )
                        p5_sq.append(q)
                    for t in cutlass.range_constexpr(mma_n):
                        if cutlass.Int32(t) < num_tokens:
                            _p5_st_bf16(
                                p5_res_out + cutlass.Int64(2) * p5_eo[t], p5_u[t]
                            )
                            if lane == cutlass.Int32(0):
                                _p5_rowsq_pub(
                                    p5_rsq
                                    + cutlass.Int64(8)
                                    * (cutlass.Int64(t * 192) + cutlass.Int64(p5_tile)),
                                    p5_tag,
                                    p5_sq[t],
                                )
                    # 4. row sums of squares: lane l gathers tiles l + 32k of every token in one batch, then warp sum
                    p5_ga = [
                        [
                            p5_rsq
                            + cutlass.Int64(8)
                            * (cutlass.Int64(t * 192 + k * 32) + cutlass.Int64(lane))
                            for k in range(6)
                        ]
                        for t in range(mma_n)
                    ]
                    p5_gw2 = [
                        [_p5_ld64(p5_ga[t][k]) for k in range(6)] for t in range(mma_n)
                    ]
                    for t in cutlass.range_constexpr(mma_n):
                        tot = cutlass.Float32(0.0)
                        for k in cutlass.range_constexpr(6):
                            tot = _p5_f(
                                "add.rn",
                                tot,
                                _p5_fix64(
                                    p5_ga[t][k],
                                    p5_tag,
                                    p5_gw2[t][k],
                                    p5_act[t],
                                ),
                            )
                        for o in (16, 8, 4, 2, 1):
                            tot = _p5_f(
                                "add.rn",
                                tot,
                                cute.arch.shuffle_sync_bfly(tot, offset=o),
                            )
                        # 5. Gemma norm
                        inv = _p5_rsqrt(
                            _p5_f(
                                "add.rn",
                                _p5_f("div.full", tot, cutlass.Float32(6144.0)),
                                p5_eps,
                            )
                        )
                        nv = _bf16_round(
                            _p5_f("mul.rn", _p5_f("mul.rn", p5_u[t], inv), p5_g1)
                        )
                        # 6. MXFP8 over the warp's 32 columns
                        am = _absmax(cutlass.Float32(0.0), nv)
                        for o in (16, 8, 4, 2, 1):
                            am = _absmax(am, cute.arch.shuffle_sync_bfly(am, offset=o))
                        code = _p5_e8m0(am * _rcp_approx_ftz(cutlass.Float32(448.0)))
                        osc = cutlass.select_(
                            am != cutlass.Float32(0.0),
                            _rcp_approx_ftz(
                                (code << cutlass.Int32(23)).bitcast(cutlass.Float32)
                            ),
                            cutlass.Float32(0.0),
                        )
                        qb = _e4m3x4(
                            nv * osc,
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                        )
                        if cutlass.Int32(t) < num_tokens:
                            if p5_nrm != cutlass.Int64(0):
                                _p5_st_bf16(p5_nrm + cutlass.Int64(2) * p5_eo[t], nv)
                            if p5_q != cutlass.Int64(0):
                                _p5_st_u8(p5_q + p5_eo[t], qb & cutlass.Int32(0xFF))
                                if lane == cutlass.Int32(0):
                                    p5_soff = (
                                        cutlass.Int64(t // 128)
                                        * cutlass.Int64(48 * 512)
                                        + cutlass.Int64(p5_tile // 4)
                                        * cutlass.Int64(512)
                                        + cutlass.Int64(
                                            (t % 32) * 16 + ((t % 128) // 32) * 4
                                        )
                                        + cutlass.Int64(p5_tile % 4)
                                    )
                                    _p5_st_u8(p5_sf + p5_soff, code)
                if cutlass.const_expr(ar_push == 6):
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    p6_rk = ar_rank & cutlass.Int32(0xFF)
                    p6_n = n + (ar_rank >> cutlass.Int32(8))
                    p6_par = [
                        _ld_volatile_s32(
                            ar_flags
                            + cutlass.Int64(4)
                            * (
                                cutlass.Int64(t * 4)
                                + cutlass.Int64(p6_n // cutlass.Int32(1536))
                            )
                        )
                        & cutlass.Int32(1)
                        for t in range(mma_n)
                    ]
                    p6_base = [
                        _ld_global_s64(ar_ws + cutlass.Int64(8 * r))
                        for r in range(ar_world)
                    ]
                    for t in cutlass.range_constexpr(mma_n):
                        p6_hi = cute.arch.shuffle_sync_bfly(totals[t], offset=1)
                        if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                            if cutlass.Int32(t) < num_tokens:
                                p6_off = (
                                    cutlass.Int64(p6_par[t])
                                    * cutlass.Int64(ar_maxt * ar_world * 3072)
                                    + (
                                        cutlass.Int64(t * ar_world)
                                        + cutlass.Int64(p6_rk)
                                    )
                                    * cutlass.Int64(3072)
                                    + (cutlass.Int64(p6_n) // cutlass.Int64(2))
                                )
                                p6_word = _bf16x2_word(totals[t], p6_hi)
                                for r in cutlass.range_constexpr(ar_world):
                                    _st_global_b32(
                                        p6_base[r] + p6_off * cutlass.Int64(4), p6_word
                                    )
                if cutlass.const_expr(ar_push == 4):
                    if cutlass.const_expr(pfx):
                        p4_par, p4_base = pfx_par, pfx_base
                    else:
                        prims.griddepcontrol(prims.GridDepAction.WAIT)
                        p4_par = [
                            _ld_volatile_s32(
                                ar_flags
                                + cutlass.Int64(4)
                                * (
                                    cutlass.Int64(t * 4)
                                    + cutlass.Int64(n // cutlass.Int32(n_out // 4))
                                )
                            )
                            & cutlass.Int32(1)
                            for t in range(mma_n)
                        ]
                        p4_base = [
                            _ld_global_s64(ar_ws + cutlass.Int64(8 * r))
                            for r in range(ar_world)
                        ]
                    for t in cutlass.range_constexpr(mma_n):
                        p4_hi = cute.arch.shuffle_sync_bfly(totals[t], offset=1)
                        if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                            if cutlass.Int32(t) < num_tokens:
                                p4_off = (
                                    cutlass.Int64(p4_par[t])
                                    * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                    + (
                                        cutlass.Int64(t * ar_world)
                                        + cutlass.Int64(ar_rank)
                                    )
                                    * cutlass.Int64(n_out // 2)
                                    + (cutlass.Int64(n) // cutlass.Int64(2))
                                )
                                p4_word = _bf16x2_word(totals[t], p4_hi)
                                for r in cutlass.range_constexpr(ar_world):
                                    _st_global_b32(
                                        p4_base[r] + p4_off * cutlass.Int64(4), p4_word
                                    )
                if cutlass.const_expr(ar_push > 0 and ar_push < 3):
                    # Every lane of an owner warp is an owner (rows_per_rank >= 32), so lane pairs hold row pairs.
                    prims.griddepcontrol(prims.GridDepAction.WAIT)
                    ar_par = [
                        _ld_volatile_s32(
                            ar_flags
                            + cutlass.Int64(4)
                            * (
                                cutlass.Int64(t * 8)
                                + cutlass.Int64(n // cutlass.Int32(n_out // 8))
                            )
                        )
                        & cutlass.Int32(1)
                        for t in range(mma_n)
                    ]
                    for t in cutlass.range_constexpr(mma_n):
                        ar_hi = cute.arch.shuffle_sync_bfly(totals[t], offset=1)
                        if (lane % cutlass.Int32(2)) == cutlass.Int32(0):
                            if cutlass.Int32(t) < num_tokens:
                                ar_off = (
                                    cutlass.Int64(ar_par[t])
                                    * cutlass.Int64(ar_maxt * ar_world * (n_out // 2))
                                    + (
                                        cutlass.Int64(t * ar_world)
                                        + cutlass.Int64(ar_rank)
                                    )
                                    * cutlass.Int64(n_out // 2)
                                    + (cutlass.Int64(n) // cutlass.Int64(2))
                                )
                                _st_global_b32(
                                    ar_ws + ar_off * cutlass.Int64(4),
                                    _bf16x2_word(totals[t], ar_hi),
                                )
        prims.barrier_cta_sync(1, thread_count=128)
        if cutlass.const_expr(ar_poll):
            # every reader of this k-range is past its loads: the last of the n_out/128 CTAs re-arms it (warp 4)
            if warp_id == 4:
                px_lane = tx % cutlass.Int32(32)
                px_ctr = _ld_global_s64(
                    ar_ws + cutlass.Int64(8 * ar_world)
                ) + cutlass.Int64(4) * cutlass.Int64(rank)
                px_old = cutlass.Int32(-1)
                if px_lane == cutlass.Int32(0):
                    px_old = _p5_atom_add(px_ctr, cutlass.Int32(1))
                px_old = _px_bcast0(px_old)
                if px_old == cutlass.Int32(n_out // CTA_M - 1):
                    px_words = my_tiles * CTA_K // 2
                    px_base = x_words.data_ptr(0).toint() + cutlass.Int64(
                        4 * (rank * my_tiles * CTA_K // 2)
                    )
                    for px_i in range(
                        px_lane, num_tokens * cutlass.Int32(px_words), 32
                    ):
                        px_t = px_i // cutlass.Int32(px_words)
                        px_j = px_i - px_t * cutlass.Int32(px_words)
                        _p5_st_s32(
                            px_base
                            + cutlass.Int64(4)
                            * (
                                cutlass.Int64(px_t) * cutlass.Int64(k_in // 2)
                                + cutlass.Int64(px_j)
                            ),
                            cutlass.Int32(-1),
                        )
                    if px_lane == cutlass.Int32(0):
                        _p5_st_s32(px_ctr, cutlass.Int32(0))
        if cutlass.const_expr(rope):
            # Token t of request t / T sits at cache position kv_len - T + t % T. Q / K
            # heads: GPT-NeoX RoPE over the 128-wide head, bf16(cos * x -/+ sin * pair)
            # on the bf16 projection; then E4M3 of x * scale, Q to rope_q, K / V into
            # the page of that position.
            pairs = rows_per_rank // 2
            rp_e = tx - cutlass.Int32(128)
            rp_p = rp_e % cutlass.Int32(pairs)
            rp_t = rp_e // cutlass.Int32(pairs)
            if rp_t < num_tokens:
                rp_max_blocks = rope_dims.load(idx=0)
                rp_tpb_log2 = rope_dims.load(idx=1)
                rp_hq = rope_dims.load(idx=2)
                rp_hkv = rope_dims.load(idx=3)
                rp_T = rope_dims.load(idx=4)
                rp_d = rank * cutlass.Int32(pairs) + rp_p
                rp_a = _bf16_round(
                    mailbox.load(
                        idx=(rank * cutlass.Int32(rows_per_rank) + rp_p)
                        * cutlass.Int32(mb_stride)
                        + rp_t
                    )
                )
                rp_b = _bf16_round(
                    mailbox.load(
                        idx=(
                            rank * cutlass.Int32(rows_per_rank)
                            + cutlass.Int32(pairs)
                            + rp_p
                        )
                        * cutlass.Int32(mb_stride)
                        + rp_t
                    )
                )
                rp_req = rp_t // rp_T
                rp_tok = rope_i.load(idx=rp_req) - rp_T + (rp_t - rp_req * rp_T)
                rp_head = n_tile
                if rp_head < rp_hq + rp_hkv:
                    rp_c = rope_cs.load(
                        idx=(rp_tok * cutlass.Int32(64) + rp_d) * cutlass.Int32(2)
                    )
                    rp_s = rope_cs.load(
                        idx=(rp_tok * cutlass.Int32(64) + rp_d) * cutlass.Int32(2)
                        + cutlass.Int32(1)
                    )
                    rp_ra = _rope_bits(rp_c, rp_a, -rp_s, rp_b).bitcast(cutlass.Float32)
                    rp_rb = _rope_bits(rp_c, rp_b, rp_s, rp_a).bitcast(cutlass.Float32)
                    rp_a = rp_ra
                    rp_b = rp_rb
                rp_qs = rope_scale.load(idx=0)
                rp_w = _e4m3x4(
                    rp_a * rp_qs,
                    rp_b * rp_qs,
                    cutlass.Float32(0.0),
                    cutlass.Float32(0.0),
                )
                rp_lo = cutlass.Uint8(rp_w & cutlass.Int32(0xFF))
                rp_hi = cutlass.Uint8((rp_w >> cutlass.Int32(8)) & cutlass.Int32(0xFF))
                if rp_head < rp_hq:
                    rp_q = (rp_t * rp_hq + rp_head) * cutlass.Int32(128) + rp_d
                    rope_q.store(rp_lo, idx=rp_q)
                    rope_q.store(rp_hi, idx=rp_q + cutlass.Int32(64))
                else:
                    rp_kv = cutlass.select_(
                        rp_head < rp_hq + rp_hkv, cutlass.Int32(0), cutlass.Int32(1)
                    )
                    rp_kh = rp_head - rp_hq - rp_kv * rp_hkv
                    rp_page = rope_bt.load(
                        idx=(rp_req * cutlass.Int32(2) + rp_kv) * rp_max_blocks
                        + (rp_tok >> rp_tpb_log2)
                    )
                    # 64-bit byte offset: the pool view can exceed 2^31 bytes.
                    rp_dst = (
                        (
                            cutlass.Int64(rp_page) * cutlass.Int64(rp_hkv)
                            + cutlass.Int64(rp_kh)
                        )
                        << cutlass.Int64(rp_tpb_log2 + cutlass.Int32(7))
                    ) + cutlass.Int64(
                        (
                            rp_tok
                            & ((cutlass.Int32(1) << rp_tpb_log2) - cutlass.Int32(1))
                        )
                        * cutlass.Int32(128)
                        + rp_d
                    )
                    rp_addr = rope_pool.data_ptr(0).toint() + rp_dst
                    cutlass.inttoptr(rp_addr, mem_space=1, dtype=cutlass.Uint8).store(
                        rp_lo
                    )
                    cutlass.inttoptr(
                        rp_addr + cutlass.Int64(64), mem_space=1, dtype=cutlass.Uint8
                    ).store(rp_hi)
        if tx == 128:
            if cutlass.const_expr(ar_push == 5):
                # last CTA out advances the site's round (every CTA read it before its exit increment)
                p5_ctr = _ld_global_s64(ar_ws + cutlass.Int64(72))
                p5_exit = _ld_global_s64(ar_ws + cutlass.Int64(88))
                p5_grid = cutlass.Int32(_ld_global_s64(ar_ws + cutlass.Int64(152)))
                p5_old = _p5_atom_add(p5_exit, cutlass.Int32(1))
                if p5_old == p5_grid - cutlass.Int32(1):
                    _p5_st_s32(p5_ctr, _ld_volatile_s32(p5_ctr) + cutlass.Int32(1))
                    _p5_st_s32(p5_exit, cutlass.Int32(0))
            if cutlass.const_expr(ar_push >= 2):
                # The all-reduce polls the pushed words; it launches after this CTA's rows are out, not at grid end.
                prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        if warp_id == 4:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Int32),
                tmem_cols(ring, nacc, a_tmem, my_tiles, mma_n),
            )


def _bytes_map(t, cols, rows, box_rows, swizzle):
    return cuda.create_tensor_map_tiled(
        global_address=t.iterator.toint(),
        dtype=cutlass.Uint8,
        global_dims=[cols, rows],
        global_strides=[cols // 16],
        box_dims=[CTA_K, box_rows],
        swizzle=swizzle,
    )


def _atom_map(t, atoms):
    return cuda.create_tensor_map_tiled(
        global_address=t.iterator.toint(),
        dtype=cutlass.Uint16,
        global_dims=[SF_ATOM_U16, atoms],
        global_strides=[SF_ATOM_BYTES // 16],
        box_dims=[SF_ATOM_U16, 1],
        swizzle=cuda.TensorMapSwizzle.none,
    )


RA_MINB = 2  # min_blocks_per_mp for T > 4 (0: uncapped)


def _ra_launch(ra, ra_tok):
    # router: max smem carveout (224 KB dynamic); above 4 rows: cap regs so a post CTA still co-resides with a GEMV CTA
    if not ra:
        return {}
    return (
        dict(min_blocks_per_mp=RA_MINB, preferred_smem_carveout=100)
        if ra_tok > 4 and RA_MINB
        else {}
    )


@cute.jit
def m3_mx_gemv(
    w: cute.Tensor,  # [N, K] E4M3 as uint8, K contiguous
    wsf: cute.Tensor,  # weight scales, 128x4-swizzled, as uint8
    x: cute.Tensor,  # [M, K] E4M3 as uint8, M <= 8
    xsf: cute.Tensor,  # activation scales, 128x4-swizzled (atom k at byte 512 k), as uint8
    y: cute.Tensor,  # [M * N] bf16
    x_words: cute.Tensor,  # x_bf16: bf16 x [M, K] as int32 words (x / xsf are then unused placeholders)
    num_tokens: cutlass.Int32,
    n_out: cutlass.Constexpr[int],
    k_in: cutlass.Constexpr[int],
    split: cutlass.Constexpr[int],
    ring: cutlass.Constexpr[int],
    trigger_early: cutlass.Constexpr[bool],
    contig: cutlass.Constexpr[bool],
    nacc: cutlass.Constexpr[int],
    a_tmem: cutlass.Constexpr[bool],
    st_async: cutlass.Constexpr[bool],
    x_bf16: cutlass.Constexpr[bool],
    ar_ws: cutlass.Int64,
    ar_flags: cutlass.Int64,
    ar_rank: cutlass.Int32,
    ar_push: cutlass.Constexpr[int],
    ar_world: cutlass.Constexpr[int],
    ar_maxt: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
    din_i: cute.Tensor,  # din / rope inputs: see m3_mx_gemv_kernel (placeholders when off)
    din_table: cute.Tensor,
    din_hidden: cute.Tensor,
    din_w: cute.Tensor,
    din_f: cute.Tensor,
    din_vocab: cutlass.Int32,
    din_g: cute.Tensor,
    din_res: cute.Tensor,
    din_flags: cutlass.Int32,
    rope_cs: cute.Tensor,
    rope_i: cute.Tensor,
    rope_bt: cute.Tensor,
    rope_q: cute.Tensor,
    rope_pool: cute.Tensor,
    rope_scale: cute.Tensor,
    rope_dims: cute.Tensor,
    din: cutlass.Constexpr[bool],
    rope: cutlass.Constexpr[bool],
    stagger: cutlass.Constexpr[int] = 0,
    early_w: cutlass.Constexpr[bool] = False,
    qfirst: cutlass.Constexpr[int] = 0,
    burst: cutlass.Constexpr[int] = 0,
    pfx: cutlass.Constexpr[bool] = False,
    ra: cutlass.Constexpr[int] = 0,
    ra_tab: cutlass.Int64 = 0,
    ra_tok: cutlass.Constexpr[int] = 0,
    ra_ck: cutlass.Constexpr[int] = 0,
) -> None:
    k_tiles = num_k_tiles(k_in)
    tma_desc_w = _bytes_map(w, k_in, n_out, CTA_M, cuda.TensorMapSwizzle.s128b)
    tma_desc_wsf = _atom_map(wsf, (n_out // CTA_M) * k_tiles)
    tma_desc_x = _bytes_map(x, k_in, num_tokens, mma_n, cuda.TensorMapSwizzle.s128b)
    tma_desc_xsf = _atom_map(xsf, k_tiles)
    m3_mx_gemv_kernel(
        tma_desc_w, tma_desc_wsf, tma_desc_x, tma_desc_xsf, y, x_words, num_tokens, n_out, k_in, split, ring,
        trigger_early, contig, nacc, a_tmem, st_async, x_bf16, ar_ws, ar_flags, ar_rank, ar_push, ar_world,
        ar_maxt, mma_n,
        din_i, din_table, din_hidden, din_w, din_f, din_vocab, din_g, din_res, din_flags, rope_cs,
        rope_i, rope_bt, rope_q, rope_pool, rope_scale, rope_dims, din, rope, stagger, early_w, qfirst, burst, w.iterator.toint(), wsf.iterator.toint(), pfx, ra, ra_tab, ra_tok, ra_ck,
    ).launch(
        grid=((n_out // CTA_M) * split + ra, 1, 1),
        block=(THREADS, 1, 1),
        cluster=(split, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
        **_ra_launch(ra, ra_tok),
    )  # fmt: skip
