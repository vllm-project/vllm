# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from TensorRT-LLM (tensorrt_llm/_torch/kernels/blackwell/mhc/mhc_boundary.py),
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
"""DeepSeek-V4.1 shifted mHC boundary for decode-size batches (CuTe DSL).

The hidden dimension is split into 16 slices. For every token, a slice's CTA updates its
part of the residual stream, collapses it with the lagged pre-mix, and multiplies the
updated residual with its slice of the mixer weight on BF16 tensor cores. The weight is
rounded to TF32 and split into BF16 high and low parts that sum to it exactly
(``prepare_boundary_weight``), so every product equals the TF32 product of DeepGEMM's
Mega mHC. A 16-CTA cluster owns a tile of tokens and its CTAs push their partial mixes
and sums of squares to the CTA that owns the token through distributed shared memory, so
up to ``MAX_TOKENS`` tokens run in one wave of one kernel. Every reduction runs in a
fixed order, so results are deterministic.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

Int32, Int64, Uint32, Float32 = (
    cutlass.Int32,
    cutlass.Int64,
    cutlass.Uint32,
    cutlass.Float32,
)

CLUSTER = 16
THREADS = 640
MMA_WARPS = 10
MULT = 4
MIXES = 24
# floats per pushed token partial: 24 mixes, sum of squares of residual_cur, padding
RECV = 28
TILE = 28
# Seven 16-CTA clusters with this kernel's shared memory are co-resident on GB300: one
# wave of tiles.
CLUSTERS = 7
MAX_TOKENS = CLUSTERS * TILE
# Tile sizes the single-pass kernel is compiled for; a batch takes the smallest that
# fills one wave.
TILES = (14, 18, 21, 24, TILE)

_ATT = llvm.AsmDialect.AD_ATT


def _asm(result, args, body, constraints, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        result,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        body,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=_ATT,
        loc=loc,
        ip=ip,
    )


def _fields(out, n, ty, wrap, loc, ip):
    return tuple(
        wrap(llvm.extractvalue(ty, out, [i], loc=loc, ip=ip)) for i in range(n)
    )


@dsl_user_op
def lds_v4_u32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def ldsm_x4(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr), Float32(value)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v2_f32(addr, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr), Float32(a), Float32(b)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v4_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr)] + [Uint32(v) for v in w],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None, [Int32(addr), Uint32(w)], "st.shared.u32 [$0], $1;", "r,r", loc=loc, ip=ip
    )


@dsl_user_op
def ldg_v4_u32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def ldg_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def stg_v4_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr)] + [Uint32(v) for v in w],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Float32(value)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_v2_u32(addr, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Uint32(a), Uint32(b)],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Uint32(value)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mxfp8_block(amax, vals, *, loc=None, ip=None):
    """E4M3 bytes and UE8M0 scale of 8 values of a 32-element MXFP8 block with absolute
    maximum ``amax``, as ``cvt_warp_fp16_to_mxfp8`` computes them (``exp2f_rcp``
    scale, IEEE products that keep denormals): two words of E4M3 bytes and the scale
    in the low byte of a third."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 3),
        [Float32(amax)] + [Float32(v) for v in vals],
        "{\n.reg .f32 r, sf, os, f<8>;\n.reg .b16 e, h<4>;\n.reg .b32 sb;\n.reg .pred z, p;\n"  # noqa: E501
        "rcp.approx.ftz.f32 r, 0f43E00000;\nmul.rn.f32 sf, $3, r;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 e, 0f00000000, sf;\n"
        # exp2f_rcp: 2^(127 - e), but 1 for e == 0.
        "cvt.u32.u16 sb, e;\nand.b32 sb, sb, 255;\ncvt.rn.f32.u32 os, sb;\n"
        "sub.f32 os, 0f42FE0000, os;\nex2.approx.f32 os, os;\n"
        "setp.eq.u32 p, sb, 0;\nselp.f32 os, 0f3F800000, os, p;\n"
        "setp.neu.f32 z, $3, 0f00000000;\nselp.f32 os, os, 0f00000000, z;\n"
        "mul.rn.f32 f0, $4, os;\nmul.rn.f32 f1, $5, os;\nmul.rn.f32 f2, $6, os;\n"
        "mul.rn.f32 f3, $7, os;\nmul.rn.f32 f4, $8, os;\nmul.rn.f32 f5, $9, os;\n"
        "mul.rn.f32 f6, $10, os;\nmul.rn.f32 f7, $11, os;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, f1, f0;\ncvt.rn.satfinite.e4m3x2.f32 h1, f3, f2;\n"  # noqa: E501
        "cvt.rn.satfinite.e4m3x2.f32 h2, f5, f4;\ncvt.rn.satfinite.e4m3x2.f32 h3, f7, f6;\n"  # noqa: E501
        "mov.b32 $0, {h0, h1};\nmov.b32 $1, {h2, h3};\ncvt.u32.u16 $2, e;\n}",
        "=r,=r,=r,f,f,f,f,f,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 3, T.i32(), Uint32, loc, ip)


@dsl_user_op
def bulk_g2s(dst, src, size, bar, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(dst), Int64(src), Int32(size), Int32(bar)],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",  # noqa: E501
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_async_v4(addr, w, bar, rank, *, loc=None, ip=None):
    """Store four words at the same shared-memory offset of CTA ``rank``, completing on
    its ``bar``."""
    _asm(
        None,
        [Int32(addr)] + [Uint32(v) for v in w] + [Int32(bar), Int32(rank)],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $6;\nmapa.shared::cluster.u32 rb, $5, $6;\n"  # noqa: E501
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$1, $2, $3, $4}, [rb];\n}",  # noqa: E501
        "r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_init(bar, count, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar), Int32(count)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_fence_init(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_expect_tx(bar, size, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar), Int32(size)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait(bar, phase, *, scope="cta", loc=None, ip=None):
    """Wait for ``phase``; ``scope="cluster"`` acquires data that peers delivered with
    st.async."""
    sem = ".acquire.cluster" if scope == "cluster" else ""
    _asm(
        None,
        [Int32(bar), Int32(phase)],
        "{\n.reg .pred p;\nWAIT:\n"
        f"mbarrier.try_wait.parity{sem}.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16(a, b, c, *, loc=None, ip=None):
    """mma.sync m16n8k16 BF16 with FP32 accumulation."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Uint32(v) for v in a] + [Uint32(v) for v in b] + [Float32(v) for v in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, "  # noqa: E501
        "{$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def shfl_xor_f32(value, mask, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [Float32(value), Int32(mask)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp(x, *, loc=None, ip=None):
    """IEEE reciprocal of a normal input: the fast path of rcp.rn.f32 without its
    out-of-range branch."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x)],
            "{\n.reg .f32 r, e;\nrcp.approx.ftz.f32 r, $1;\nfma.rn.f32 e, $1, r, 0fBF800000;\n"  # noqa: E501
            "neg.f32 e, e;\nfma.rn.f32 $0, r, e, r;\n}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_lo(word, *, loc=None, ip=None):
    return Float32(
        llvm.bitcast(
            T.f32(), (Uint32(word) << 16).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def bf16x2_hi(word, *, loc=None, ip=None):
    high = Uint32(word) & Uint32(0xFFFF0000)
    return Float32(llvm.bitcast(T.f32(), high.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def f32_bits(value, *, loc=None, ip=None):
    return Uint32(
        llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two floats to BF16 (nearest even); ``lo`` goes to the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(hi), Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def sigmoid(v):
    """``1 / (1 + exp(-v))``; the denominator is clamped below 2^126 so the reciprocal
    is exact."""
    return rcp(cute.arch.fmin(Float32(1.0) + cute.math.exp(-v), Float32(2.0**126)))


def residual_update(pw, cm, xw, rw):
    """BF16 pairs of ``bf16(post[s] * x + sum_j comb[j][s] * residual[j])`` for 8 hidden
    elements per stream ``s``, ``mhc_fused_hc``'s flush-to-zero FMA chain."""
    xf = [(bf16x2_lo(xw[p]), bf16x2_hi(xw[p])) for p in range(4)]
    rf = [
        [(bf16x2_lo(rw[j][p]), bf16x2_hi(rw[j][p])) for p in range(4)]
        for j in range(MULT)
    ]
    cur = []
    for s in range(MULT):
        words = []
        for p in range(4):
            nu = cute.arch.mul_packed_f32x2((pw[s], pw[s]), xf[p], ftz=True)
            for j in range(MULT):
                nu = cute.arch.fma_packed_f32x2(
                    (cm[j][s], cm[j][s]), rf[j][p], nu, ftz=True
                )
            words.append(pack_bf16x2(nu[0], nu[1]))
        cur.append(words)
    return cur


def lagged_collapse(cur, pre, ssq_rc, ssq_li):
    """BF16 pairs of ``li = bf16(sum_j pre[j] * residual_cur[j])`` for 8 hidden
    elements, with the running sums of squares of residual_cur and li."""
    li = []
    for p in range(4):
        acc = (Float32(0.0), Float32(0.0))
        for j in range(MULT):
            v = (bf16x2_lo(cur[j][p]), bf16x2_hi(cur[j][p]))
            ssq_rc = cute.arch.fma_packed_f32x2(v, v, ssq_rc)
            acc = cute.arch.fma_packed_f32x2((pre[j], pre[j]), v, acc)
        word = pack_bf16x2(acc[0], acc[1])
        li.append(word)
        v = (bf16x2_lo(word), bf16x2_hi(word))
        ssq_li = cute.arch.fma_packed_f32x2(v, v, ssq_li)
    return li, ssq_rc, ssq_li


class MhcBoundaryKernel:
    """``tile`` (at most ``TILE``) tokens per 16-CTA cluster; CTA ``r`` owns hidden ``[r
    * slice, (r + 1) * slice)``. Each token's arithmetic is the same at every tile
    size.

    Mixer GEMM per CTA: ``[48, K slice] x [K slice, tokens]`` with rows 0-23 the high
    and rows 24-47 the low BF16 part of the weight. Warp ``w < MMA_WARPS`` takes the k16
    steps ``[w * steps_per_warp, (w + 1) * steps_per_warp)``.
    """

    def __init__(
        self,
        hidden: int,
        with_post: bool,
        sinkhorn_iters: int,
        mxfp8: bool = False,
        tile: int = TILE,
    ):
        self.tile = tile
        self.hidden = hidden
        self.with_post = with_post
        self.sinkhorn_iters = sinkhorn_iters
        # Also write layer_input as torch.ops.trtllm.mxfp8_quantize(layer_input, True)
        # does: E4M3 [B, hidden] and UE8M0 scales in the 128x4 swizzled layout (padding
        # rows' not written).
        self.mxfp8 = mxfp8
        self.slice = hidden // CLUSTER
        self.chunks = self.slice // 8
        self.parts = (
            self.chunks // 2
        )  # elementwise threads per token, two 8-wide chunks each
        self.k_steps = MULT * self.slice // 16
        self.steps_per_warp = self.k_steps // MMA_WARPS
        self.n_tiles = (self.tile + 7) // 8
        self.cols = 40  # padded token columns of the per-warp partials
        self.owned = (
            self.tile + CLUSTER - 1
        ) // CLUSTER  # at most 2: warp 0 runs 16 lanes per owned token
        self.li_words = (self.tile + 3) // 4 * 4
        self.w_bytes = self.k_steps * 32 * 48
        # residual_cur rows in k16 block order; 16 bytes of padding per 32-wide chunk
        # and per row keep the elementwise stores and the ldmatrix rows on distinct
        # banks.
        self.r_chunk = 8 * 32 + 16
        self.r_row = self.slice // 32 * self.r_chunk + 16
        self.x_row = self.slice * 2
        off = 0

        def take(size, align=16):
            nonlocal off
            off = (off + align - 1) // align * align
            start = off
            off += size
            return start

        # The weight fragments are dead after the mixer GEMM; its per-warp partials
        # reuse the space.
        self.o_w = take(max(self.w_bytes, MMA_WARPS * MIXES * self.cols * 4), 128)
        self.o_r = take(self.tile * self.r_row)
        self.o_x = take(self.tile * self.x_row)
        self.o_prc = take(self.tile * self.parts * 4)
        self.o_pli = take(self.tile * self.parts * 4)
        self.o_cta = take(self.tile * RECV * 4)
        self.o_cli = take(self.li_words * 4)
        self.o_rmix = take(self.owned * CLUSTER * RECV * 4)
        self.o_rli = take(CLUSTER * self.li_words * 4)
        self.o_tot = take(2 * RECV * 4)
        self.o_rows = take(2 * MULT * MULT * 4)
        self.o_rstd = take(self.li_words * 4)
        self.o_hcw = take(32 * 4)
        self.o_bar = take(16, 8)
        self.smem_bytes = off

    @staticmethod
    def supports(hidden: int) -> bool:
        slice_ = hidden // CLUSTER
        return (
            hidden % 512 == 0
            and slice_ % 32 == 0
            and (MULT * slice_ // 16) % MMA_WARPS == 0
            and TILE * (slice_ // 16) <= THREADS
        )

    @cute.jit
    def __call__(
        self,
        x,
        residual,
        post,
        comb,
        pre_lag,
        weight,
        scale,
        base,
        norm_w,
        residual_out,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        mxfp8_out: Int64,
        mxfp8_scale: Int64,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
        stream,
    ):
        tiles = (residual.shape[0] + self.tile - 1) // self.tile
        self.kernel(
            x,
            residual,
            post,
            comb,
            pre_lag,
            weight,
            scale,
            base,
            norm_w,
            residual_out,
            layer_input,
            pre_out,
            post_out,
            comb_out,
            mxfp8_out,
            mxfp8_scale,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        ).launch(
            grid=[tiles * CLUSTER, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        x,
        residual,
        post,
        comb,
        pre_lag,
        weight,
        scale,
        base,
        norm_w,
        residual_out,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        mxfp8_out: Int64,
        mxfp8_scale: Int64,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
    ):
        H = self.hidden
        S = self.slice
        PARTS = self.parts
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        rank = bid % CLUSTER
        t0 = (bid // CLUSTER) * self.tile
        valid = cutlass.min(self.tile, residual.shape[0] - t0)

        smem = SmemAllocator()
        sbase = smem.allocate(self.smem_bytes, 128).toint()
        s_w = sbase + self.o_w
        s_r = sbase + self.o_r
        s_x = sbase + self.o_x
        s_hcw = sbase + self.o_hcw
        bar_w = sbase + self.o_bar
        bar_x = bar_w + 8
        hslice = Int64(rank * S)
        owned_here = (self.tile - rank + CLUSTER - 1) // CLUSTER

        w_chunks = 10
        if tid == 0:
            mbar_init(bar_w, 1)
            mbar_init(bar_x, 1)
            mbar_fence_init()
            mbar_expect_tx(bar_w, self.w_bytes)
            mbar_expect_tx(
                bar_x, owned_here * CLUSTER * RECV * 4 + CLUSTER * self.li_words * 4
            )
        cute.arch.sync_threads()
        # Peers push into bar_x: every CTA's barriers must be initialized before the
        # first push.
        cute.arch.cluster_arrive_relaxed()

        # ---- the weight slice in a few bulk copies; hc scale and base to shared memory
        jw = tid - (THREADS - w_chunks)
        if jw >= 0:
            size = self.w_bytes // w_chunks
            bulk_g2s(
                s_w + jw * size,
                weight.iterator.toint() + Int64(rank) * self.w_bytes + jw * size,
                size,
                bar_w,
            )
        elif (tid >= 256) & (tid < 256 + 3 + MIXES):
            i = tid - 256
            if i < 3:
                sts_f32(s_hcw + i * 4, ldg_f32(scale.iterator.toint() + i * 4))
            else:
                sts_f32(s_hcw + i * 4, ldg_f32(base.iterator.toint() + (i - 3) * 4))

        # Everything above reads only weights; the activations come from the previous
        # kernel.
        cute.arch.griddepcontrol_wait()

        # ---- elementwise pass: residual update, lagged collapse li, sums of squares.
        # Thread (t, part)
        # handles the 8-wide chunks part and part + PARTS of token t, loaded straight to
        # registers.
        t = tid // PARTS
        part = tid % PARTS
        if t < valid:
            gt = Int64(t0 + t)
            pre = ldg_v4_f32(pre_lag.iterator.toint() + gt * 16)
            pw = pre
            cm = [pre] * MULT
            if cutlass.const_expr(self.with_post):
                pw = ldg_v4_f32(post.iterator.toint() + gt * 16)
                cm = [
                    ldg_v4_f32(comb.iterator.toint() + gt * 64 + j * 16)
                    for j in range(MULT)
                ]
            g_row = residual.iterator.toint() + (gt * MULT * H + hslice) * 2
            g_xrow = x.iterator.toint() + (gt * H + hslice) * 2
            rws = []
            xws = []
            for half in cutlass.range_constexpr(2):
                c = part + half * PARTS
                rws.append(
                    [ldg_v4_u32(g_row + j * H * 2 + c * 16) for j in range(MULT)]
                )
                if cutlass.const_expr(self.with_post):
                    xws.append(ldg_v4_u32(g_xrow + c * 16))
            ssq_rc = (Float32(0.0), Float32(0.0))
            ssq_li = (Float32(0.0), Float32(0.0))
            for half in cutlass.range_constexpr(2):
                c = part + half * PARTS
                r_addr = s_r + t * self.r_row + (c // 4) * self.r_chunk + (c % 4) * 4
                cur = rws[half]
                if cutlass.const_expr(self.with_post):
                    cur = residual_update(pw, cm, xws[half], rws[half])
                    for s in cutlass.range_constexpr(MULT):
                        stg_v4_u32(
                            residual_out.iterator.toint()
                            + ((gt * MULT + s) * H + hslice + c * 8) * 2,
                            cur[s],
                        )
                # Word w of stream s lands in k16 block 2 * s + w // 2, fragment column
                # half w % 2.
                for s in cutlass.range_constexpr(MULT):
                    for w in cutlass.range_constexpr(4):
                        sts_u32(
                            r_addr + (2 * s + w // 2) * 32 + (w % 2) * 16, cur[s][w]
                        )
                li, ssq_rc, ssq_li = lagged_collapse(cur, pre, ssq_rc, ssq_li)
                sts_v4_u32(s_x + t * self.x_row + c * 16, li)
            sts_f32(
                sbase + self.o_prc + (part * self.tile + t) * 4, ssq_rc[0] + ssq_rc[1]
            )
            sts_f32(
                sbase + self.o_pli + (part * self.tile + t) * 4, ssq_li[0] + ssq_li[1]
            )
        cute.arch.sync_threads()

        # ---- mixer GEMM on m16n8k16 BF16: A = weight fragments, B = residual_cur
        # tokens
        mbar_wait(bar_w, 0)
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        q = lane % 4
        # ldmatrix rows: lanes 0-7 / 8-15 / 16-23 / 24-31 = tokens 0-7 at k, 0-7 at k +
        # 8, 8-15 at k, 8-15 at k + 8
        mrow = (lane % 8) + (lane // 16) * 8
        mk = ((lane // 8) % 2) * 8
        if warp < MMA_WARPS:
            acc = [[[Float32(0.0)] * 4 for _ in range(self.n_tiles)] for _ in range(3)]
            for step in cutlass.range_constexpr(self.steps_per_warp):
                kk = warp * self.steps_per_warp + step
                frag = s_w + (kk * 32 + lane) * 48
                a = [lds_v4_u32(frag + 16 * mt) for mt in range(3)]
                b = []
                for pair in cutlass.range_constexpr((self.n_tiles + 1) // 2):
                    tok = cutlass.min(pair * 16 + mrow, self.tile - 1)
                    m4 = ldsm_x4(
                        s_r
                        + tok * self.r_row
                        + kk // 8 * self.r_chunk
                        + (kk % 8) * 32
                        + mk * 2
                    )
                    b.append((m4[0], m4[1]))
                    b.append((m4[2], m4[3]))
                for nt in cutlass.range_constexpr(self.n_tiles):
                    for mt in cutlass.range_constexpr(3):
                        acc[mt][nt] = list(mma_bf16(a[mt], b[nt], acc[mt][nt]))
            # Rows g / g + 8 of the three m tiles are high g, high g + 8 | high g + 16,
            # low g | low g + 8, low g + 16, so each thread holds both parts of mixes g,
            # g + 8 and g + 16.
            cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
            for nt in cutlass.range_constexpr(self.n_tiles):
                col = nt * 8 + 2 * q
                for mi in cutlass.range_constexpr(3):
                    hi_t, hi_r = ((0, 0), (0, 2), (1, 0))[mi]
                    lo_t, lo_r = ((1, 2), (2, 0), (2, 2))[mi]
                    sts_v2_f32(
                        s_w + ((warp * MIXES + mi * 8 + g) * self.cols + col) * 4,
                        acc[hi_t][nt][hi_r] + acc[lo_t][nt][lo_r],
                        acc[hi_t][nt][hi_r + 1] + acc[lo_t][nt][lo_r + 1],
                    )
        cute.arch.sync_threads()

        # ---- CTA partial per token: 24 mixes (warps in order), sums of squares of
        # residual_cur and li
        s_cta = sbase + self.o_cta
        s_cli = sbase + self.o_cli
        for k in cutlass.range_constexpr((self.tile * 26 + THREADS - 1) // THREADS):
            i = tid + k * THREADS
            if i < self.tile * MIXES:
                n = i // self.tile
                tt = i % self.tile
                parts = [
                    lds_f32(s_w + ((w * MIXES + n) * self.cols + tt) * 4)
                    for w in range(MMA_WARPS)
                ]
                v = Float32(0.0)
                for w in cutlass.range_constexpr(MMA_WARPS):
                    v = v + parts[w]
                sts_f32(s_cta + (tt * RECV + n) * 4, v)
            elif i < self.tile * 26:
                ii = i - self.tile * MIXES
                kind = ii // self.tile
                tt = ii % self.tile
                src = sbase + self.o_prc + kind * (self.o_pli - self.o_prc) + tt * 4
                parts = [lds_f32(src + j * self.tile * 4) for j in range(PARTS)]
                v = Float32(0.0)
                for j in cutlass.range_constexpr(PARTS):
                    v = v + parts[j]
                if kind == 0:
                    sts_f32(s_cta + (tt * RECV + MIXES) * 4, v)
                else:
                    sts_f32(s_cli + tt * 4, v)
        cute.arch.sync_threads()

        # ---- push token t's partial to its owner (rank t % 16, slot t // 16) and the
        # li sums to all
        cute.arch.cluster_wait()
        quads = RECV // 4
        li_quads = self.li_words // 4
        if tid < self.tile * quads:
            tt = tid // quads
            j = tid % quads
            dst = (
                sbase
                + self.o_rmix
                + (((tt // CLUSTER) * CLUSTER + rank) * RECV + j * 4) * 4
            )
            st_async_v4(
                dst, lds_v4_u32(s_cta + (tt * RECV + j * 4) * 4), bar_x, tt % CLUSTER
            )
        elif tid < self.tile * quads + CLUSTER * li_quads:
            i = tid - self.tile * quads
            j = i % li_quads
            dst = sbase + self.o_rli + (rank * self.li_words + j * 4) * 4
            st_async_v4(dst, lds_v4_u32(s_cli + j * 16), bar_x, i // li_quads)
        mbar_wait(bar_x, 0, scope="cluster")
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()

        # ---- totals in rank order: li rstd of every token, the 24 mixes and sum of
        # squares of owned ones
        s_tot = sbase + self.o_tot
        s_rstd = sbase + self.o_rstd
        if tid < self.tile:
            parts = [
                lds_f32(sbase + self.o_rli + (p * self.li_words + tid) * 4)
                for p in range(CLUSTER)
            ]
            v = Float32(0.0)
            for p in cutlass.range_constexpr(CLUSTER):
                v = v + parts[p]
            sts_f32(s_rstd + tid * 4, cute.math.rsqrt(v / Float32(H) + norm_eps))
        item = tid - 32
        if (item >= 0) & (item < self.owned * 25):
            o = item // 25
            vi = item % 25
            parts = [
                lds_f32(sbase + self.o_rmix + ((o * CLUSTER + p) * RECV + vi) * 4)
                for p in range(CLUSTER)
            ]
            v = Float32(0.0)
            for p in cutlass.range_constexpr(CLUSTER):
                v = v + parts[p]
            sts_f32(s_tot + (o * RECV + vi) * 4, v)
        cute.arch.sync_threads()

        if tid < 32:
            # 16 lanes per owned token
            self.coefficients(
                s_tot + (lane // 16) * RECV * 4,
                s_hcw,
                sbase + self.o_rows + (lane // 16) * 64,
                rank + (lane // 16) * CLUSTER,
                lane % 16,
                t0,
                valid,
                pre_out,
                post_out,
                comb_out,
                rms_eps,
                hc_eps,
                sinkhorn_eps,
                post_mult,
            )
        else:
            # layer_input = bf16(li * rstd * norm_w) for this CTA's hidden slice
            for k in cutlass.range_constexpr(
                (self.tile * self.chunks + THREADS - 33) // (THREADS - 32)
            ):
                u = tid - 32 + k * (THREADS - 32)
                tt = u // self.chunks
                c = u % self.chunks
                if cutlass.const_expr(self.mxfp8):
                    # The 4 lanes of a 32-wide MXFP8 block share its maximum by
                    # shuffles, so every lane computes (past the last token on the
                    # tile's last row) and valid ones store.
                    ts = cutlass.min(tt, self.tile - 1)
                    out = self.layer_input_words(s_x, s_rstd, norm_w, hslice, ts, c)
                    vals = []
                    for p in cutlass.range_constexpr(4):
                        vals.append(bf16x2_lo(out[p]))
                        vals.append(bf16x2_hi(out[p]))
                    amax = Float32(0.0)
                    for v in vals:
                        amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
                    amax = cute.arch.fmax(amax, shfl_xor_f32(amax, 1))
                    amax = cute.arch.fmax(amax, shfl_xor_f32(amax, 2))
                    if tt < valid:
                        row = Int64(t0 + tt)
                        stg_v4_u32(
                            layer_input.iterator.toint()
                            + (row * H + hslice + c * 8) * 2,
                            out,
                        )
                        w0, w1, sf = mxfp8_block(amax, vals)
                        stg_v2_u32(mxfp8_out + row * H + hslice + c * 8, w0, w1)
                        if c % 4 == 0:
                            kb = (hslice + c * 8) // 32
                            stg_u8(
                                mxfp8_scale
                                + (row // 128) * (H // 128 * 512)
                                + (kb // 4) * 512
                                + (row % 32) * 16
                                + (row % 128) // 32 * 4
                                + kb % 4,
                                sf,
                            )
                elif tt < valid:
                    out = self.layer_input_words(s_x, s_rstd, norm_w, hslice, tt, c)
                    stg_v4_u32(
                        layer_input.iterator.toint()
                        + (Int64(t0 + tt) * H + hslice + c * 8) * 2,
                        out,
                    )
        # No CTA leaves while its peers may still read its barriers.
        cute.arch.cluster_wait()

    @cute.jit
    def layer_input_words(self, s_x, s_rstd, norm_w, hslice, tt, c):
        """BF16 pairs of ``li * rstd * norm_w`` for 8-wide chunk ``c`` of tile token
        ``tt``."""
        li = lds_v4_u32(s_x + tt * self.x_row + c * 16)
        nw = ldg_v4_u32(norm_w.iterator.toint() + hslice * 2 + c * 16)
        rstd = lds_f32(s_rstd + tt * 4)
        out = []
        for p in cutlass.range_constexpr(4):
            lo = (bf16x2_lo(li[p]) * rstd) * bf16x2_lo(nw[p])
            hi = (bf16x2_hi(li[p]) * rstd) * bf16x2_hi(nw[p])
            out.append(pack_bf16x2(lo, hi))
        return out

    @cute.jit
    def coefficients(
        self,
        tot,
        s_hcw,
        s_rows,
        tt,
        r,
        t0,
        valid,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        eps: Float32,
        post_mult: Float32,
    ):
        """Pre / post / Sinkhorn comb of token ``tt`` from its mixes, in
        ``mhc_fused_hc``'s order.

        Lane ``r`` of the token's 16: 0-3 pre, 4-7 post, 8-11 the softmax of comb row
        ``r - 8``; lane 12 then runs the Sinkhorn iterations on the four rows. With
        ``sinkhorn_iters == 0`` lanes 8-11 store the softmax rows to ``comb_out`` for
        ``MhcSinkhornKernel``.
        """
        rstd = cute.math.rsqrt(
            lds_f32(tot + MIXES * 4) / Float32(MULT * self.hidden) + rms_eps
        )
        gt = Int64(t0 + tt)
        if r < 2 * MULT:
            v = cute.math.fma(
                lds_f32(tot + r * 4) * rstd,
                lds_f32(s_hcw + (r // MULT) * 4),
                lds_f32(s_hcw + (3 + r) * 4),
            )
            if r < MULT:
                if tt < valid:
                    stg_f32(
                        pre_out.iterator.toint() + (gt * MULT + r) * 4,
                        sigmoid(v) + hc_eps,
                    )
            elif tt < valid:
                stg_f32(
                    post_out.iterator.toint() + (gt * MULT + r - MULT) * 4,
                    sigmoid(v) * post_mult,
                )
        elif r < 3 * MULT:
            i = r - 2 * MULT
            s2 = lds_f32(s_hcw + 2 * 4)
            row = [
                cute.math.fma(
                    lds_f32(tot + (8 + 4 * i + k) * 4) * rstd,
                    s2,
                    lds_f32(s_hcw + (11 + 4 * i + k) * 4),
                )
                for k in range(MULT)
            ]
            row_max = cute.arch.fmax(
                cute.arch.fmax(row[0], row[1]), cute.arch.fmax(row[2], row[3])
            )
            row = [cute.math.exp(v - row_max) for v in row]
            inv = rcp(row[0] + row[1] + row[2] + row[3])
            row = [cute.math.fma(v, inv, eps) for v in row]
            if cutlass.const_expr(self.sinkhorn_iters == 0):
                if tt < valid:
                    stg_v4_u32(
                        comb_out.iterator.toint() + (gt * 16 + i * 4) * 4,
                        [f32_bits(v) for v in row],
                    )
            else:
                for k in cutlass.range_constexpr(MULT):
                    sts_f32(s_rows + (4 * i + k) * 4, row[k])
        if cutlass.const_expr(self.sinkhorn_iters > 0):
            cute.arch.sync_warp()
            if r == 3 * MULT:
                out = self.sinkhorn(
                    [lds_f32(s_rows + k * 4) for k in range(MULT * MULT)], eps
                )
                if tt < valid:
                    for i in cutlass.range_constexpr(MULT):
                        stg_v4_u32(
                            comb_out.iterator.toint() + (gt * 16 + i * 4) * 4,
                            [f32_bits(v) for v in out[4 * i : 4 * i + 4]],
                        )

    @cute.jit
    def sinkhorn(self, c, eps: Float32):
        """A column normalization, then ``sinkhorn_iters - 1`` row and column
        normalizations."""
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = (
            self.col_normalize(c, eps)
        )
        for _ in cutlass.range(self.sinkhorn_iters - 1, unroll=1):
            cc = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
            rr = []
            for i in cutlass.range_constexpr(MULT):
                rinv = rcp(
                    cc[4 * i] + cc[4 * i + 1] + cc[4 * i + 2] + cc[4 * i + 3] + eps
                )
                rr.extend([cc[4 * i + k] * rinv for k in range(MULT)])
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = (
                self.col_normalize(rr, eps)
            )
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]

    @staticmethod
    def col_normalize(c, eps):
        """Column sums (r0 + r1) + (r2 + r3), ``mhc_fused_hc``'s shuffle order."""
        out = list(c)
        for k in range(MULT):
            inv = rcp((c[k] + c[4 + k]) + (c[8 + k] + c[12 + k]) + eps)
            for i in range(MULT):
                out[4 * i + k] = c[4 * i + k] * inv
        return out


class MhcSinkhornKernel(MhcBoundaryKernel):
    """The Sinkhorn iterations of the comb rows that a boundary run with
    ``sinkhorn_iters = 0`` wrote, in place, one token per thread."""

    def __init__(self, sinkhorn_iters: int):
        self.sinkhorn_iters = sinkhorn_iters

    @cute.jit
    def __call__(self, comb, sinkhorn_eps: Float32, stream):
        self.kernel(comb, sinkhorn_eps).launch(
            grid=[(comb.shape[0] + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, comb, sinkhorn_eps: Float32):
        t = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if t < comb.shape[0]:
            row = comb.iterator.toint() + Int64(t) * 64
            c = [v for i in range(MULT) for v in ldg_v4_f32(row + i * 16)]
            out = self.sinkhorn(c, sinkhorn_eps)
            for i in cutlass.range_constexpr(MULT):
                stg_v4_u32(row + i * 16, [f32_bits(v) for v in out[4 * i : 4 * i + 4]])


def prepare_boundary_weight(fn: torch.Tensor) -> torch.Tensor:
    """``fn [24, 4 * hidden]`` -> int32 mma A fragments ``[16 slices, k16 steps, 32
    lanes, 12]``.

    The weight is rounded to TF32 (nearest, ties away, as the TF32 tensor-core path) and
    split into BF16 high and low parts whose sum is exact; the 12 words of a lane are
    its A fragments of the three m16 tiles of the stacked ``[high; low]`` rows. k16 step
    ``8 c + kb`` of a slice is chunk ``c`` (hidden ``[32 c, 32 c + 32)``), stream ``kb
    // 2``; fragment columns ``2q, 2q + 1`` and ``2q + 8, 2q + 9`` hold hidden ``8 q + 4
    (kb % 2) + {0, 1}`` and ``+ {2, 3}`` of the chunk.
    """
    hidden = fn.shape[1] // MULT
    S = hidden // CLUSTER
    bits = fn.contiguous().view(torch.int32)
    rounded = ((bits + 0x1000) & -0x2000).view(torch.float32)
    hi = rounded.to(torch.bfloat16)
    lo = (rounded - hi.float()).to(torch.bfloat16)
    pairs = torch.cat([hi, lo]).view(torch.int32)
    dev = fn.device
    rank = torch.arange(CLUSTER, device=dev).view(-1, 1, 1, 1)
    kk = torch.arange(MULT * S // 16, device=dev).view(1, -1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, -1, 1)
    g, q = lane // 4, lane % 4
    c, kb = kk // 8, kk % 8
    base = (kb // 2) * hidden + rank * S + c * 32 + q * 8 + (kb % 2) * 4
    rows = torch.tensor([0, 8, 0, 8, 16, 24, 16, 24, 32, 40, 32, 40], device=dev).view(
        1, 1, 1, -1
    )
    upper = torch.tensor([0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2], device=dev).view(
        1, 1, 1, -1
    )
    return pairs[rows + g, (base + upper) // 2].contiguous()


_COMPILED = set()


def compiled(hidden: int, num_sinkhorn_iters: int, tokens: int) -> bool:
    """Whether ``mhc_boundary`` for this batch is compiled (compiling inside graph
    capture is not allowed)."""
    return (
        0 < tokens <= MAX_TOKENS
        and (hidden, True, num_sinkhorn_iters, False, boundary_tile(tokens))
        in _COMPILED
    )


def _mat(dtype, *shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16
    )


def boundary_tile(tokens: int) -> int:
    """Tokens per cluster of the single-pass kernel: the smallest of ``TILES`` that runs
    ``tokens`` in one wave, so small batches spread over every co-resident cluster."""
    return next(tile for tile in TILES if tile * CLUSTERS >= tokens)


@functools.cache
def _compile_tile(
    hidden: int, with_post: bool, sinkhorn_iters: int, mxfp8: bool, tile: int
):
    tokens = cute.sym_int()
    bf16 = cutlass.BFloat16
    vec = (tokens, MULT)
    fn = cute.compile(
        MhcBoundaryKernel(hidden, with_post, sinkhorn_iters, mxfp8, tile),
        _mat(bf16, tokens, hidden),
        _mat(bf16, tokens, MULT, hidden),
        _mat(Float32, *vec),
        _mat(Float32, tokens, MULT * MULT),
        _mat(Float32, *vec),
        _mat(Int32, CLUSTER, MULT * (hidden // CLUSTER) // 16, 32, 12),
        _mat(Float32, 3),
        _mat(Float32, MIXES),
        _mat(bf16, hidden),
        _mat(bf16, tokens, MULT, hidden),
        _mat(bf16, tokens, hidden),
        _mat(Float32, *vec),
        _mat(Float32, *vec),
        _mat(Float32, tokens, MULT * MULT),
        Int64(0),
        Int64(0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add((hidden, with_post, sinkhorn_iters, mxfp8, tile))
    return fn


@functools.cache
def _compile_sinkhorn(sinkhorn_iters: int):
    fn = cute.compile(
        MhcSinkhornKernel(sinkhorn_iters),
        _mat(Float32, cute.sym_int(), MULT * MULT),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add(("sinkhorn", sinkhorn_iters))
    return fn


def mhc_boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_shifted_post_pre_deep_gemm`` for up to ``MAX_TOKENS`` tokens, with the same
    inputs except ``weight = prepare_boundary_weight(fn)``. With ``num_sinkhorn_iters
    = 0`` the returned comb holds the row softmax for ``mhc_sinkhorn``."""
    tokens, mult, hidden = residual.shape
    assert 0 < tokens <= MAX_TOKENS and mult == MULT, residual.shape
    new_residual = torch.empty_like(residual)
    layer_input = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=x.device)
    new_pre, new_post = torch.empty(
        2, tokens, mult, dtype=torch.float32, device=x.device
    )
    new_comb = torch.empty(tokens, mult, mult, dtype=torch.float32, device=x.device)
    _compile_tile(hidden, True, num_sinkhorn_iters, False, boundary_tile(tokens))(
        x,
        residual,
        post_mix.view(tokens, mult),
        comb_res_mix.view(tokens, mult * mult),
        pre_mix,
        weight,
        scale,
        base,
        norm_weight,
        new_residual,
        layer_input,
        new_pre,
        new_post,
        new_comb.view(tokens, mult * mult),
        0,
        0,
        hc_norm_eps,
        hc_pre_eps,
        sinkhorn_eps,
        hc_post_scale,
        norm_eps,
    )
    return new_residual, new_post.unsqueeze(-1), new_comb, layer_input, new_pre


def mhc_sinkhorn(
    comb: torch.Tensor, sinkhorn_eps: float, num_sinkhorn_iters: int
) -> None:
    """Run the Sinkhorn iterations of a comb ``mhc_boundary(..., num_sinkhorn_iters=0)``
    returned, in place."""
    _compile_sinkhorn(num_sinkhorn_iters)(comb.view(-1, MULT * MULT), sinkhorn_eps)


def mhc_shifted_post_pre_boundary(
    x: torch.Tensor,
    residual: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``mhc_shifted_post_pre_deep_gemm`` on the CuTe DSL boundary kernel, or None when
    the batch is not decode-sized or the kernel is not ready (it compiles on eager
    runs before capture)."""
    tokens, mult, hidden = residual.shape
    capturing = torch.cuda.is_current_stream_capturing()
    if (
        mult != MULT
        or not MhcBoundaryKernel.supports(hidden)
        or not 0 < tokens <= MAX_TOKENS
        or (capturing and not compiled(hidden, num_sinkhorn_iters, tokens))
    ):
        return None
    # The prepared weight follows in-place updates of `fn` (weight reloads) by its
    # version.
    cached = getattr(fn, "_mhc_boundary_weight", None)
    if cached is None or cached[0] != fn._version:
        if capturing:
            return None
        cached = (fn._version, prepare_boundary_weight(fn))
        fn._mhc_boundary_weight = cached
    inputs = (x, residual, pre_mix, post_mix, comb_res_mix)
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in inputs):
        return None
    return mhc_boundary(
        x,
        residual,
        pre_mix,
        post_mix,
        comb_res_mix,
        cached[1],
        scale,
        base,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        num_sinkhorn_iters,
        norm_weight,
        norm_eps,
    )
