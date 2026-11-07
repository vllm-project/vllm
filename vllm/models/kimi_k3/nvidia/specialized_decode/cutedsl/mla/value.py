# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA value projection (W_UV) and sigmoid gate of the BF16 attention latent."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint32

from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    copy_g2s,
    launch,
    load_words,
    mma_into,
    sigmoid,
    store_words,
    unpack_bf16,
    zeros,
)


class MlaValue:
    """W_UV and the sigmoid gate, [B, H, 512] BF16 latent -> [B, H*128]. One
    128-thread CTA per (32 outputs, head, 16 rows); warp w owns outputs
    8w..8w+7 over the whole latent (BF16 MMA, FP32 accumulation). W_UV must have a
    contiguous latent axis (the absorbed kv_b_proj view). Rows are tokens, ``q`` per
    request (``lengths`` per request)."""

    def __init__(self, heads: int, batch: int, q: int = 1):
        self.heads, self.batch, self.q = heads, batch, q
        self.grid = (4, heads, (batch + 15) // 16)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 128)

    @cute.kernel
    def kernel(self, args):
        latent, w_uv, gate, lengths, output = args
        rows = alloc(cutlass.utils.SmemAllocator(), BFloat16, (16, 544), 128)
        tid, _, _ = cute.arch.thread_idx()
        chunk, head, tile = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0, v = tile * 16, chunk * 32 + warp * 8 + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        for c in range(tid, 16 * 64, 128):
            r, j = c // 64, c % 64
            if t0 + r < self.batch:
                src = at(latent, t0 + r, head, j * 8)
                copy_g2s(rows.iterator + r * 544 + j * 8, src)
        cute.arch.cp_async_commit_group()
        col = head * 128 + chunk * 32 + warp * 8 + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < self.batch:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    rows.iterator + r * 544 + 32 * c + 8 * quarter, count=4, shared=True
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            t = t0 + group + 8 * r
            if t < self.batch:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if lengths[t // self.q] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e])) * sigmoid(gates[r][e], False)
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))


# Rows from which the value projection of many local heads (TP1 / DEP: 96) runs as
# one batched GEMM over the heads (cuBLAS, the stock decode's W_UV bmm: FP32
# accumulation, one BF16 rounding) and MlaGate, instead of MlaValue, whose CTAs (32
# outputs, head, 16 rows) each stage their W_UV slice (32 KB) again for every
# 16-row tile and run mma.sync. GB300 TP1 DSA layer, us (32K / AgentX): 40 rows
# 152.4 / 165.1 -> 153.2 / 166.4, 80 rows 192.1 / 206.3 -> 186.2 / 200.2, 320 rows
# 408.8 / 460.9 -> 375.0 / 426.9 (MlaValue 16.1 / 50.0 us CUPTI at 80 / 320 rows,
# MlaGate 5.1 / 7.4 plus the bmm). At TP8 (12 heads) MlaValue is 4-9 us and the
# layer ties (80-320 rows within 0.5 us): MlaValue stays.
VALUE_BMM_FROM = 64
VALUE_BMM_HEADS = 64


def value_bmm(heads: int, batch: int) -> bool:
    """Whether ``batch`` rows of ``heads`` local heads take the batched GEMM."""
    return heads >= VALUE_BMM_HEADS and batch >= VALUE_BMM_FROM


class MlaGate:
    """The sigmoid gate of the W_UV output rows: ``values`` [H, B, 128] BF16 (the
    batched GEMM's), ``gate`` [B, H * 128] -> ``output`` [B, H * 128], as MlaValue
    rounds and gates them (BF16 value times the FP32 sigmoid, rounded to BF16;
    rows of empty requests zero). One 256-thread CTA per (16 rows, head): each
    thread 8 outputs of one row."""

    def __init__(self, heads: int, batch: int, q: int = 1):
        self.heads, self.batch, self.q = heads, batch, q
        self.grid = ((batch + 15) // 16, heads, 1)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, 256)

    @cute.kernel
    def kernel(self, args):
        values, gate, lengths, output = args
        tid, _, _ = cute.arch.thread_idx()
        tile, head, _ = cute.arch.block_idx()
        t, chunk = tile * 16 + tid // 16, tid % 16
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if t < self.batch:
            col = head * 128 + chunk * 8
            words = load_words(at(values, head, t, chunk * 8), count=4, mutable=True)
            gates = load_words(at(gate, t, col), count=4, mutable=True)
            out = []
            for e in cutlass.range_constexpr(4):
                v, g = unpack_bf16(words[e]), unpack_bf16(gates[e])
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if lengths[t // self.q] > 0:
                    lo, hi = (v[k] * sigmoid(g[k], False) for k in range(2))
                out.append(bf16x2(lo, hi))
            store_words(at(output, t, col), tuple(out))
