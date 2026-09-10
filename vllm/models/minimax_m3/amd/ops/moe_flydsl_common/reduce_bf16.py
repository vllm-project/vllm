# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 prefill MoE final reduce over gemm2's bf16 token-major partials
(the "bf16" output mode of the a4w4 and a8w8 gemm2):

    y[tok, :] = sum_k out[tok*topk + k, :]      -> bf16 (f32 accumulate)

gemm2 already multiplied every partial row by its routing weight, so this is a
plain top-k sum in k order -- the arithmetic of aiter's ``moe_reduction_kernel``
it replaces, bit-identical output (32768 tokens, standalone: 451 -> 404 us; the
bytes are the same, the gain is the non-temporal loads and one wave
per 1 KB chunk of every row).

One wave per (token, 512-column chunk): lane L owns columns ``chunk*512 + L*8
.. +8`` of the token's ``topk`` rows -- one dwordx4 per row, so every load /
store instruction of the wave covers 1 KB contiguous. The partials are read once
(non-temporal loads); the output is stored normally so the layer's next op finds
it in cache (a non-temporal store only pays off standalone at 32768 tokens, and
loses at 8192). One wave per CTA measured fastest (4 waves: +3%, 2 waves: +10%);
``chunks_per_wave`` / ``waves_per_cta`` / ``nt_store`` keep the other variants
for the lab.

Layouts (bytes):
  OUT  [n_tokens*topk, H]   bf16, token-major (gemm2, routing weights applied)
  Y    [n_tokens, H]        bf16
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.epilogue import _bf16x2
from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.loaders import _as_f32

WAVES_PER_CTA = 1
CHUNKS_PER_WAVE = 1  # dwordx4 per lane per row
_CHUNK = 512  # bf16 columns per (wave, dwordx4)
_NT = 2  # non-temporal cache policy


def compile_moe_reduce_bf16(
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
    ROW_DW = H // 2
    CHUNK_DW = _CHUNK // 2
    LD = _NT if nt_load else 0
    ST = _NT if nt_store else 0
    name = f"m3_reduce_bf16_h{H}_k{topk}_c{CPW}_w{WPC}_l{LD}_s{ST}"

    @flyc.kernel(name=name, known_block_size=[64 * WPC, 1, 1])
    def kernel_reduce_bf16(OUT: fx.Tensor, Y: fx.Tensor, n_tokens: fx.Int32):
        lane = fx.thread_idx.x % 64
        gw = fx.block_idx.x * WPC + fx.thread_idx.x // 64
        tok = gw // SEGS
        seg = gw % SEGS
        out_rsrc = buffer_ops.create_buffer_resource(
            OUT, max_size=False, num_records_bytes=n_tokens * (topk * H * 2)
        )
        y_rsrc = buffer_ops.create_buffer_resource(
            Y, max_size=False, num_records_bytes=n_tokens * (H * 2)
        )
        if tok < n_tokens:
            col_dw = seg * (CHUNK_DW * CPW) + lane * 4
            row_dw = tok * (topk * ROW_DW) + col_dw
            data = [
                [
                    Vec(
                        buffer_ops.buffer_load(
                            out_rsrc,
                            row_dw + k * ROW_DW + j * CHUNK_DW,
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=LD,
                        )
                    )
                    for j in range(CPW)
                ]
                for k in range(topk)
            ]
            acc = [[fx.Float32(0.0) for _ in range(8)] for _ in range(CPW)]
            for k in range_constexpr(topk):
                for j in range_constexpr(CPW):
                    for d in range_constexpr(4):
                        dw = fx.Int32(data[k][j][d])
                        acc[j][2 * d] = acc[j][2 * d] + _as_f32(dw << 16)
                        acc[j][2 * d + 1] = acc[j][2 * d + 1] + _as_f32(
                            dw & fx.Int32(-65536)
                        )
            y_dw = tok * ROW_DW + col_dw
            for j in range_constexpr(CPW):
                packed = Vec.from_elements(
                    [_bf16x2(acc[j][2 * i], acc[j][2 * i + 1]) for i in range(4)],
                    fx.Int32,
                ).ir_value()
                buffer_ops.buffer_store(
                    packed, y_rsrc, y_dw + j * CHUNK_DW, cache_modifier=ST
                )

    @flyc.jit
    def launch_reduce_bf16(
        OUT: fx.Tensor, Y: fx.Tensor, n_tokens: fx.Int32, stream: fx.Stream
    ):
        grid = (n_tokens * SEGS + (WPC - 1)) // WPC
        kernel_reduce_bf16(OUT, Y, n_tokens).launch(
            grid=(grid, 1, 1), block=(64 * WPC, 1, 1), stream=stream
        )

    launch_reduce_bf16.kernel_name = name
    return launch_reduce_bf16
