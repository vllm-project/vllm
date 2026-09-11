# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 prefill MoE final reduce over gemm2's mxfp8 token-major rows
(``out_mode="fp8"``; the a4w4 lab's ``reduce_fp8.py``):

    y[tok, :] = sum_k w[tok, k] * dequant(out[tok*topk + k, :])      -> bf16

One wave per (token, 1024-column chunk): lane L owns columns ``chunk*1024 + L*16
.. +16`` of the token's ``topk`` rows (16 fp8 = one dwordx4 per row, so every
load instruction of the wave covers 1 KB contiguous and every store 2 KB). Per
row: one dwordx4 + one scale byte, ``v_cvt_pk_f32_fp8`` and one fma per value
with (2^(e8m0-127) * w). The partials are read once (non-temporal loads); the
output is stored normally so the layer's next op finds it in cache. 4 waves per
CTA (32768 tokens, standalone: 277 -> 243 us against the previous
one-wave-per-token layout; ``chunks_per_wave`` / ``waves_per_cta`` /
``nt_store`` keep the other variants for the lab).

Layouts (bytes):
  OUT     [n_tokens*topk, H]      fp8 e4m3, token-major (gemm2)
  OUT_sc  [n_tokens*topk, H/32]   e8m0
  W       [n_tokens, topk]        f32 routing weights (shared expert = 1)
  Y       [n_tokens, H]           bf16
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.loaders import _as_f32


def _i1(v: bool):
    return fx.Boolean(v).ir_value()


WAVES_PER_CTA = 4
CHUNKS_PER_WAVE = 1  # dwordx4 per lane per row
_NT = 2  # non-temporal cache policy
_CHUNK = 1024  # fp8 columns per (wave, dwordx4)


def _cvt_pk_f32_fp8(dword, hi: bool):
    """v_cvt_pk_f32_fp8: 2 fp8 (low / high word of ``dword``) -> 2 f32"""
    v2f32 = _ir.VectorType.get([2], _T.f32)
    res = _llvm.call_intrinsic(
        v2f32, "llvm.amdgcn.cvt.pk.f32.fp8", [fx.as_ir_value(dword), _i1(hi)], [], []
    )
    return Vec(res)


def _fma(a, b, c):
    return fx.Float32(
        _llvm.call_intrinsic(
            _T.f32,
            "llvm.fma.f32",
            [fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(c)],
            [],
            [],
        )
    )


def _pack_bf16x2(a, b):
    """two f32 -> one i32 of 2 bf16 (RNE, v_cvt_pk_bf16_f32 on gfx950)"""
    v2f32 = _ir.VectorType.get([2], _T.f32)
    v2bf16 = _ir.VectorType.get([2], _ir.BF16Type.get())
    v = _vector.FromElementsOp(v2f32, [fx.as_ir_value(a), fx.as_ir_value(b)]).result
    t = _arith.TruncFOp(v2bf16, v).result
    return _llvm.bitcast(_T.i32, t)


def _v4i32(vals):
    ty = _ir.VectorType.get([4], _T.i32)
    return _vector.FromElementsOp(ty, [fx.as_ir_value(v) for v in vals]).result


def compile_moe_reduce_fp8(
    *,
    H: int,
    topk: int,
    chunks_per_wave: int = CHUNKS_PER_WAVE,
    waves_per_cta: int = WAVES_PER_CTA,
    nt_load: bool = True,
    nt_store: bool = False,
):
    CPW = chunks_per_wave
    WPC = waves_per_cta
    assert H % (_CHUNK * CPW) == 0
    SEGS = H // (_CHUNK * CPW)  # waves per token
    SC_COLS = H // 32
    ROW_DW = H // 4
    CHUNK_DW = _CHUNK // 4
    CHUNK_SC = _CHUNK // 32
    LD = _NT if nt_load else 0
    ST = _NT if nt_store else 0
    name = f"m3_reduce_fp8_h{H}_k{topk}_c{CPW}_w{WPC}_l{LD}_s{ST}"

    @flyc.kernel(name=name, known_block_size=[64 * WPC, 1, 1])
    def kernel_reduce_fp8(
        OUT: fx.Tensor,
        OUT_sc: fx.Tensor,
        W: fx.Tensor,
        Y: fx.Tensor,
        n_tokens: fx.Int32,
    ):
        lane = fx.thread_idx.x % 64
        gw = fx.block_idx.x * WPC + fx.thread_idx.x // 64
        tok = gw // SEGS
        seg = gw % SEGS
        out_rsrc = buffer_ops.create_buffer_resource(
            OUT, max_size=False, num_records_bytes=n_tokens * (topk * H)
        )
        sc_rsrc = buffer_ops.create_buffer_resource(
            OUT_sc, max_size=False, num_records_bytes=n_tokens * (topk * SC_COLS)
        )
        w_rsrc = buffer_ops.create_buffer_resource(
            W, max_size=False, num_records_bytes=n_tokens * (topk * 4)
        )
        y_rsrc = buffer_ops.create_buffer_resource(
            Y, max_size=False, num_records_bytes=n_tokens * (H * 2)
        )
        if tok < n_tokens:
            col_dw = seg * (CHUNK_DW * CPW) + lane * 4
            col_sc = seg * (CHUNK_SC * CPW) + lane // 2
            row0 = tok * topk
            ws = [
                fx.Float32(
                    buffer_ops.buffer_load(
                        w_rsrc, row0 + k, vec_width=1, dtype=fx.Float32
                    )
                )
                for k in range(topk)
            ]
            data = [
                [
                    Vec(
                        buffer_ops.buffer_load(
                            out_rsrc,
                            (row0 + k) * ROW_DW + col_dw + i * CHUNK_DW,
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=LD,
                        )
                    )
                    for i in range(CPW)
                ]
                for k in range(topk)
            ]
            e8s = [
                [
                    fx.Int32(
                        buffer_ops.buffer_load(
                            sc_rsrc,
                            (row0 + k) * SC_COLS + col_sc + i * CHUNK_SC,
                            vec_width=1,
                            dtype=fx.Int8,
                            cache_modifier=LD,
                        )
                    )
                    for i in range(CPW)
                ]
                for k in range(topk)
            ]
            acc = [[fx.Float32(0.0) for _ in range(16)] for _ in range(CPW)]
            for k in range_constexpr(topk):
                for i in range_constexpr(CPW):
                    sw = _as_f32((e8s[k][i] & fx.Int32(0xFF)) << 23) * ws[k]
                    for d in range_constexpr(4):
                        dw = fx.Int32(data[k][i][d])
                        lo = _cvt_pk_f32_fp8(dw, False)
                        hi = _cvt_pk_f32_fp8(dw, True)
                        vals = [
                            fx.Float32(lo[0]),
                            fx.Float32(lo[1]),
                            fx.Float32(hi[0]),
                            fx.Float32(hi[1]),
                        ]
                        for q in range_constexpr(4):
                            acc[i][4 * d + q] = _fma(vals[q], sw, acc[i][4 * d + q])
            y_dw = (
                tok * (H // 2) + seg * (CHUNK_DW * 2 * CPW) + lane * 8
            )  # bf16 row in dwords
            for i in range_constexpr(CPW):
                packed = [
                    _pack_bf16x2(acc[i][2 * j], acc[i][2 * j + 1]) for j in range(8)
                ]
                base = y_dw + i * (CHUNK_DW * 2)
                buffer_ops.buffer_store(
                    _v4i32(packed[0:4]), y_rsrc, base, cache_modifier=ST
                )
                buffer_ops.buffer_store(
                    _v4i32(packed[4:8]), y_rsrc, base + 4, cache_modifier=ST
                )

    @flyc.jit
    def launch_reduce_fp8(
        OUT: fx.Tensor,
        OUT_sc: fx.Tensor,
        W: fx.Tensor,
        Y: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        grid = (n_tokens * SEGS + (WPC - 1)) // WPC
        kernel_reduce_fp8(OUT, OUT_sc, W, Y, n_tokens).launch(
            grid=(grid, 1, 1), block=(64 * WPC, 1, 1), stream=stream
        )

    launch_reduce_fp8.kernel_name = name
    return launch_reduce_fp8
