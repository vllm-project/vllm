# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA RecoverSSM state commit: one CTA per (head chunk, request, layer).

Folds the n accepted tokens of a verify window into the checkpoint:

    S_n = S_0 * e^{L_n} + sum_{t<n} u_t (k_t * e^{L_n - L_t})

(columns scale per key channel; ``u_t`` are the verify's corrections, ``k_t`` the
normalized recorded keys and ``L_t`` the log decay of tokens 0..t inclusive). With
prefix caching (``align``), a window that crosses a block boundary after its first
m tokens also stores S_m to the boundary block's slot.

Each CTA walks up to ``MAX_HEADS_PER_CTA`` heads of one (request, layer). A head's
32 KB state page streams in with one bulk copy into a double-buffered shared tile
(the next head's page loads while this head is folded); thread c rebuilds key
channel c's decays and scaled keys; 16 lanes per state row (8 keys each) fold the
tile in place, and one bulk store writes it to the destination slot (the boundary
state of ``align`` is stored row by row from registers). Layer tensors are addressed
through the per-layer base-address tables of ``KDARecoverSSMCommitContext``.
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint64

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    bf16,
    bf16x2,
    copy_g2s,
    init_barriers,
    launch,
    load_words,
    sigmoid,
    store_words,
    wait,
    warp_sum,
)

K = 128
THREADS = 128
LANES_PER_ROW = 16
KEYS_PER_LANE = K // LANES_PER_ROW  # 8
ROWS_PER_PASS = THREADS // LANES_PER_ROW  # 8
PASSES = K // ROWS_PER_PASS  # 16
PAGE = K * K  # BF16 elements of one head's state
PAGE_BYTES = PAGE * 2
MAX_HEADS_PER_CTA = 8


@asm()
def store_bulk(dst, src, size):
    """Bulk store of ``size`` bytes from shared memory, in this thread's bulk group."""
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2; "
    code += "cp.async.bulk.commit_group;"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def bulk_read_done():
    """Wait until every committed bulk store has read its shared source."""
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


@asm()
def bulk_done():
    """Wait until every committed bulk store has completed."""
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm()
def proxy_fence():
    """Order this thread's shared writes before later async-proxy (bulk) reads."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


class KdaCommit:
    """Args: state / correction / key-gate base addresses [L], their slot / row
    strides [L] (elements), a_log [L, H], dt_bias [L, H, K], source slots [B],
    commit lengths [B], destination slots [B], boundary slots [B] and lengths [B]
    (read with ``align``), lower bound, null slot, stream. Per slot the state is
    [H, V, K] BF16; request b's records are row b of the corrections
    [rows, H, Q, V] FP32 and key / gate records [rows, H, Q, 2K] BF16, all
    contiguous."""

    def __init__(self, heads: int, window: int, align: bool = False):
        assert 1 <= window <= 8
        self.heads, self.window, self.align = heads, window, align
        self.heads_per_cta = math.gcd(heads, MAX_HEADS_PER_CTA)

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads // self.heads_per_cta, args[8].shape[0], args[0].shape[0])
        launch(self.kernel, args, grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        state_addrs, corr_addrs, kg_addrs = args[:3]
        state_strides, corr_strides, kg_strides, a_log, dt_bias = args[3:8]
        sources, commit_lens, destinations, boundaries, boundary_lens = args[8:13]
        lower_bound, null_slot = args[13:15]
        Q: cutlass.Constexpr = self.window
        ALIGN: cutlass.Constexpr = self.align
        HEADS_PER_CTA: cutlass.Constexpr = self.heads_per_cta
        tid, _, _ = cute.arch.thread_idx()
        chunk_id, request, layer = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        group, sub = tid // LANES_PER_ROW, tid % LANES_PER_ROW
        col = sub * KEYS_PER_LANE
        smem = cutlass.utils.SmemAllocator()
        tiles = alloc(smem, cutlass.BFloat16, (2, PAGE), align=128)
        bars = alloc(smem, Uint64, (2,), align=8)
        scaled = alloc(smem, Float32, (Q + 1, K))  # k_t e^{L_n - L_t}; row Q: e^{L_n}
        scaled_b = alloc(smem, Float32, (Q + 1, K))  # the same for S_m (align)
        norms = alloc(smem, Float32, (4, Q))
        corrections = alloc(smem, Float32, (Q, K))  # u_t of the current head
        kv = cute.make_rmem_tensor((Q,), Float32)
        logs = cute.make_rmem_tensor((Q,), Float32)
        coef = cute.make_rmem_tensor((Q + 1, KEYS_PER_LANE), Float32)
        coef_b = cute.make_rmem_tensor((Q + 1, KEYS_PER_LANE), Float32)
        kn = cute.make_rmem_tensor((Q,), Float32)
        u = cute.make_rmem_tensor((Q,), Float32)

        if tid == 0:
            init_barriers(bars.iterator, 2, async_proxy=True)
        cute.arch.barrier()
        cute.arch.griddepcontrol_wait()
        n = commit_lens[request]
        source = Int64(sources[request])
        destination = Int64(destinations[request])
        live = (n > 0) & (source > Int64(null_slot)) & (destination > Int64(null_slot))
        m = Int32(0)
        boundary = Int64(null_slot)
        if cutlass.const_expr(ALIGN):
            m = boundary_lens[request]
            boundary = Int64(boundaries[request])
        boundary_live = (m > 0) & (boundary > Int64(null_slot))
        if live:
            V: cutlass.Constexpr = K
            first_head = chunk_id * HEADS_PER_CTA
            state_base = Int64(state_addrs[layer])
            stride = Int64(state_strides[layer]) * 2  # bytes per slot
            src_base = state_base + source * stride
            dst_base = state_base + destination * stride
            boundary_base = state_base + boundary * stride
            corr_ptr = cute.make_ptr(
                Float32, corr_addrs[layer], cute.AddressSpace.gmem, assumed_align=16
            )
            kg_ptr = cute.make_ptr(
                cutlass.BFloat16,
                kg_addrs[layer],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            corr_slot = corr_ptr + Int64(request) * Int64(corr_strides[layer])
            kg_slot = kg_ptr + Int64(request) * Int64(kg_strides[layer])

            if tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator, PAGE_BYTES)
                page0 = cute.make_ptr(
                    cutlass.BFloat16,
                    src_base + Int64(first_head) * PAGE_BYTES,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                copy_g2s(tiles.iterator, page0, size=PAGE_BYTES, barrier=bars.iterator)

            for h in cutlass.range(HEADS_PER_CTA):
                head = first_head + h
                buf = h % 2
                tile = tiles.iterator + buf * PAGE
                # ---- The next head's page streams in while this head folds. ----
                if tid == 0:  # noqa: SIM102 (traced: `and` does not combine traced values)
                    if h + 1 < HEADS_PER_CTA:
                        # Its buffer's last bulk store must have read it out.
                        bulk_read_done()
                        nxt = (h + 1) % 2
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            bars.iterator + nxt, PAGE_BYTES
                        )
                        page = cute.make_ptr(
                            cutlass.BFloat16,
                            src_base + Int64(head + 1) * PAGE_BYTES,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        copy_g2s(
                            tiles.iterator + nxt * PAGE,
                            page,
                            size=PAGE_BYTES,
                            barrier=bars.iterator + nxt,
                        )

                corr = cute.make_tensor(
                    corr_slot + head * Q * V, cute.make_layout((Q, V), stride=(V, 1))
                )
                kg = cute.make_tensor(
                    kg_slot + head * Q * 2 * K,
                    cute.make_layout((Q, 2 * K), stride=(2 * K, 1)),
                )
                for t in cutlass.range_constexpr(Q):
                    value = Float32(0)
                    if t < n:
                        value = corr[t, tid]
                    corrections[t, tid] = value

                # ---- Thread c: key channel c's decays and scaled keys. ----
                bias = dt_bias[layer, head, tid].to(Float32)
                a = cute.math.exp(a_log[layer, head].to(Float32))
                cum = Float32(0)
                for t in cutlass.range_constexpr(Q):
                    kv[t] = Float32(0)
                    if t < n:
                        kv[t] = kg[t, tid].to(Float32)
                        cum = cum + lower_bound * sigmoid(
                            a * (kg[t, K + tid].to(Float32) + bias), True
                        )
                    logs[t] = cum
                for t in cutlass.range_constexpr(Q):
                    s = warp_sum(kv[t] * kv[t])
                    if lane == 0:
                        norms[warp, t] = s
                cute.arch.barrier()
                for t in cutlass.range_constexpr(Q):
                    total = norms[0, t] + norms[1, t] + norms[2, t] + norms[3, t]
                    kn[t] = kv[t] * cute.math.rsqrt(total + Float32(1e-6))
                    # Token t's update decays by the tokens after it: e^{L_n - L_t}.
                    scaled[t, tid] = kn[t] * cute.math.exp(cum - logs[t])
                scaled[Q, tid] = cute.math.exp(cum)
                if cutlass.const_expr(ALIGN):
                    # S_m: the first m tokens (m <= n), decayed to token m - 1.
                    cum_b = Float32(0)
                    for t in cutlass.range_constexpr(Q):
                        if t < m:
                            cum_b = logs[t]
                    for t in cutlass.range_constexpr(Q):
                        value = Float32(0)
                        if t < m:
                            value = kn[t] * cute.math.exp(cum_b - logs[t])
                        scaled_b[t, tid] = value
                    scaled_b[Q, tid] = cute.math.exp(cum_b)
                cute.arch.barrier()
                for t in cutlass.range_constexpr(Q + 1):
                    for j in cutlass.range_constexpr(KEYS_PER_LANE):
                        coef[t, j] = scaled[t, col + j]
                        if cutlass.const_expr(ALIGN):
                            coef_b[t, j] = scaled_b[t, col + j]

                # ---- Rows: 16 lanes per row, 8 keys per lane, folded in place. ----
                wait(bars.iterator + buf, (h // 2) % 2, scope="cta")
                for it in cutlass.range_constexpr(PASSES):
                    r = it * ROWS_PER_PASS + group
                    for t in cutlass.range_constexpr(Q):
                        u[t] = corrections[t, r]
                    pointer = tile + r * K + col
                    words = load_words(pointer, count=4, shared=True, mutable=True)
                    out = []
                    out_b = []
                    for w in cutlass.range_constexpr(4):
                        pair = []
                        pair_b = []
                        for half in cutlass.range_constexpr(2):
                            j = 2 * w + half
                            initial = bf16(words[w], half == 1)
                            value = initial * coef[Q, j]
                            for t in cutlass.range_constexpr(Q):
                                value = value + u[t] * coef[t, j]
                            pair.append(value)
                            if cutlass.const_expr(ALIGN):
                                value_b = initial * coef_b[Q, j]
                                for t in cutlass.range_constexpr(Q):
                                    value_b = value_b + u[t] * coef_b[t, j]
                                pair_b.append(value_b)
                        out.append(bf16x2(pair[0], pair[1]))
                        if cutlass.const_expr(ALIGN):
                            out_b.append(bf16x2(pair_b[0], pair_b[1]))
                    store_words(pointer, tuple(out), shared=True)
                    if cutlass.const_expr(ALIGN):  # noqa: SIM102 (a compile-time branch)
                        if boundary_live:
                            row = Int64(head) * PAGE_BYTES + Int64(r * K + col) * 2
                            store_words(boundary_base + row, tuple(out_b))
                proxy_fence()
                cute.arch.barrier()
                if tid == 0:
                    store_bulk(dst_base + Int64(head) * PAGE_BYTES, tile, PAGE_BYTES)
            if tid == 0:
                bulk_done()
        cute.arch.griddepcontrol_launch_dependents()
