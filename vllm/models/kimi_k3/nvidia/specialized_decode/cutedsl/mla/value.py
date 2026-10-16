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
