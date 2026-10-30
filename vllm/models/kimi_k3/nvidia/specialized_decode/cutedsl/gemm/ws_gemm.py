# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-stationary skinny GEMM for the MoE front: y[M, N] = x[M, K] @ W[N, K]^T.

Each CTA owns ``rt`` 16-row N tiles; its ``rt * kq`` warps split K into ``kq`` ranges.
Weights enter registers and cp.async SMEM stages before the PDL wait. The epilogue
publishes the first ``valid`` of ``split`` rows (router shard) as FP32 words at BF16
column 3584 + 2 * (``slot`` + row) of a multicast [1, capacity, width] mailbox and the
rest (latent down shard) as sanitized BF16 at column ``base``.

``tg`` 0 is the single pass: warps load x from global and the cluster leader sums the
K partials over DSMEM. ``tg`` > 0 streams tokens in groups of ``tg``: warp 0 bulk-copies
the CTA's K slice of x once (one mbarrier per group) for all row tiles, and after each
group's MMAs every warp pushes its 8-token x 16-row partial blocks to their owner CTAs
(block % ks) with st.async; owners sum the ks * kq parts in K order and publish one
group behind, so only the last group's blocks follow the MMA loop.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    copy_g2s,
    float_bits,
    launch,
    load_words,
    mma_into,
    multicast_words,
    push_float,
    smem_words,
    wait,
)


@asm(Uint32, pure=True)
def positive_zero(low, high):
    """BF16 (low, high) with -0.0 halves made +0.0, so it never equals the sentinel."""
    code = "{ .reg .b32 t; .reg .pred p,q; and.b32 t,$1,0xFFFF; "
    code += "setp.eq.u32 p,t,0x8000; shr.u32 t,$1,16; setp.eq.u32 q,t,0x8000; "
    code += "mov.b32 $0,$1; @p and.b32 $0,$0,0xFFFF0000; @q and.b32 $0,$0,0x0000FFFF; }"
    return code, "=r,r", (Uint32(bf16x2(low, high)),)


@asm(Float32)
def peer_load(pointer, peer):
    """FP32 at the same shared-memory offset in cluster CTA ``peer``."""
    code = "{ .reg .b32 a; mapa.shared::cluster.u32 a,$1,$2; "
    return code + "ld.shared::cluster.f32 $0,[a]; }", "=f,r,r", (pointer, Int32(peer))


@asm(Uint32, pure=True)
def positive_zero32(value):
    """FP32 bits of ``value`` with -0.0 made +0.0, so it never equals the sentinel."""
    code = "{ .reg .pred p; setp.eq.u32 p,$1,0x80000000; selp.b32 $0,0,$1,p; }"
    return code, "=r,r", (Uint32(float_bits(value)),)


class WsGemm:
    def __init__(
        self, n, k, m, rt, kq, staged, split, valid, ks=1, tg=0, ch=0
    ):  # fmt: skip
        self.n, self.k, self.m, self.rt, self.kq = n, k, m, rt, kq
        # ks: a cluster of ks CTAs splits K, so each CTA preloads 1/ks of the tile's
        # weights; ch: accumulator chains per 8-token tile (0: 4 / tiles per group).
        self.split, self.valid, self.ks = split, valid, ks
        self.staged, self.kb = staged, k // (32 * kq * ks)
        self.rb, self.nt = self.kb - staged, (m + 7) // 8
        self.grouped, self.ng = tg > 0, min(self.nt, tg // 8) if tg else self.nt
        if self.grouped and ks <= 1:  # token groups push partials to K-split peers
            raise ValueError(f"WsGemm token groups (tg={tg}) need a K-split cluster")
        self.groups, self.chains = -(-self.nt // self.ng), ch or max(1, 4 // self.ng)
        self.grid, self.threads = n // (16 * rt) * ks, 32 * rt * kq
        self.blocks, self.parts = self.nt * rt, ks * kq
        self.slots = -(-self.blocks // ks)  # owned blocks s * ks + kpart
        # Bulk x rows: the CTA's K slice plus 64 B, so token pairs hit disjoint banks.
        self.slice = k // ks
        self.pitch = self.slice // 2 + 16

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, self.threads, self.ks, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, w = args[0:2]
        smem = cutlass.utils.SmemAllocator()
        block, _, _ = cute.arch.block_idx()
        cta, kpart = block // self.ks, block % self.ks
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group, part = lane // 4, lane % 4
        tile, quarter = warp // self.kq, warp % self.kq
        row = (cta * self.rt + tile) * 16 + group
        kofs = (kpart * self.kq + quarter) * self.kb * 32 + part * 8
        shape = (self.nt, self.rt, self.kq, 8, 17)
        red = smem.allocate_tensor(Float32, cute.make_layout(shape), 16)
        stage = alloc(smem, Uint32, (self.rt * self.kq, max(self.staged, 1), 2, 32, 4))
        xs = recv = None
        bars: Any = None
        owned: Any = None
        if cutlass.const_expr(self.grouped):
            xs = alloc(smem, Uint32, (self.nt * 8, self.pitch))
            bars = alloc(smem, Uint64, (self.groups,), 8)
            recv = alloc(smem, Uint32, (self.slots, self.parts, 32, 4))
            owned = alloc(smem, Uint64, (self.slots,), 8)
            if tid == 0:
                for g in cutlass.range_constexpr(self.groups):
                    cute.arch.mbarrier_init(bars.iterator + g, 1)
                for s in cutlass.range_constexpr(self.slots):
                    if s * self.ks + kpart < self.blocks:
                        cute.arch.mbarrier_init(owned.iterator + s, 1)
                        nbytes = self.parts * 512
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            owned.iterator + s, nbytes
                        )
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
        wp = w.iterator + row * self.k + kofs
        for j in cutlass.range_constexpr(self.staged):
            for h in cutlass.range_constexpr(2):
                dst = at(stage, warp, j, h, lane, 0)
                copy_g2s(dst, wp + h * 8 * self.k + (self.rb + j) * 32)
        cute.arch.cp_async_commit_group()
        wr = {}
        for j in cutlass.range_constexpr(self.rb):
            for h in cutlass.range_constexpr(2):
                src = wp + h * 8 * self.k + j * 32
                wr[j, h] = load_words(src, count=4, evict_first=True)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.grouped):
            if warp == 0:
                self.stage_x(x, xs, bars, kpart, lane)
            cute.arch.cp_async_wait_group(0)
        else:
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        for g in cutlass.range_constexpr(self.groups):
            acc = cute.make_rmem_tensor((self.ng, self.chains, 4), Float32)
            acc.fill(0)
            if cutlass.const_expr(self.grouped):
                wait(bars.iterator + g)
            for j in cutlass.range_constexpr(self.kb):
                if cutlass.const_expr(j < self.rb):
                    wa, wb = wr[j, 0], wr[j, 1]
                else:
                    sa = at(stage, warp, j - self.rb, 0, lane, 0)
                    wa, wb = smem_words(sa), smem_words(sa + 128)
                for t in cutlass.range_constexpr(self.tiles(g)):
                    token = (g * self.ng + t) * 8 + group
                    xv = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    if cutlass.const_expr(self.grouped):
                        col = (quarter * self.kb + j) * 16 + part * 4
                        xv = smem_words(at(xs, token, col))
                    else:
                        xg = x.iterator + token * x.stride[0] + kofs + j * 32
                        if (g * self.ng + t) * 8 + 7 < self.m or token < self.m:
                            xv = load_words(xg, count=4, mutable=True)
                    for h in cutlass.range_constexpr(2):
                        c = (j * 2 + h) % self.chains
                        a = (wa[2 * h], wb[2 * h], wa[2 * h + 1], wb[2 * h + 1])
                        b = (xv[2 * h], xv[2 * h + 1])
                        mma_into(acc[t, c, None], a, b, bf16=True)
            if cutlass.const_expr(self.grouped and g == 0):
                cute.arch.cluster_wait()  # owners' barriers are armed
            for t in cutlass.range_constexpr(self.tiles(g)):
                values = []
                for v in cutlass.range_constexpr(4):
                    value = acc[t, 0, v]
                    for c in cutlass.range_constexpr(1, self.chains):
                        value = value + acc[t, c, v]
                    tt, col = g * self.ng + t, group + (v // 2) * 8
                    if cutlass.const_expr(self.grouped):
                        values.append(value)
                    else:
                        red[tt, tile, quarter, 2 * part + v % 2, col] = value
                if cutlass.const_expr(self.grouped):
                    blk = (g * self.ng + t) * self.rt + tile
                    s, peer = blk // self.ks, blk % self.ks
                    dst = at(recv, s, kpart * self.kq + quarter, lane, 0)
                    push_float(dst, tuple(values), owned.iterator + s, peer)
            if cutlass.const_expr(self.grouped and g > 0):
                self.settle(args, red, recv, owned, cta, kpart, warp, lane, g - 1)
        if cutlass.const_expr(self.grouped):
            self.settle(args, red, recv, owned, cta, kpart, warp, lane, self.groups - 1)
        else:
            cute.arch.sync_threads()
            if cutlass.const_expr(self.ks > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
            self.gather(args, red, cta, kpart, tid, lane)
            if cutlass.const_expr(self.ks > 1):
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()

    def tiles(self, g):
        """8-token tiles in token group ``g``."""
        return min(self.ng, self.nt - g * self.ng)

    @cute.jit
    def stage_x(self, x, xs, bars, kpart, lane):
        """Warp 0 bulk-copies every token's K slice; one barrier per token group."""
        if lane == 0:
            for g in cutlass.range_constexpr(self.groups):
                tokens = min(self.m, (g + 1) * self.ng * 8) - g * self.ng * 8
                nbytes = tokens * self.slice * 2
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + g, nbytes)
        cute.arch.sync_warp()
        for i in cutlass.range_constexpr(-(-self.m // 32)):
            token = i * 32 + lane
            if i * 32 + 31 < self.m or token < self.m:
                src = x.iterator + token * x.stride[0] + kpart * self.slice
                bar = bars.iterator + token // (8 * self.ng)
                copy_g2s(at(xs, token, 0), src, self.slice * 2, bar)

    @cute.jit
    def settle(
        self, args, red, recv, owned, cta, kpart, warp, lane, g: cutlass.Constexpr
    ):
        """Owned blocks of token group ``g``: a warp per block waits for every K part,
        sums them in K order, transposes through ``red`` and publishes."""
        for t in cutlass.range_constexpr(self.tiles(g)):
            for tile in cutlass.range_constexpr(self.rt):
                blk = (g * self.ng + t) * self.rt + tile
                s = blk // self.ks
                if (kpart == blk % self.ks) & (warp == s % (self.rt * self.kq)):
                    wait(owned.iterator + s)
                    total = cute.make_rmem_tensor(4, Float32)
                    for p in cutlass.range_constexpr(self.parts):
                        words = smem_words(at(recv, s, p, lane, 0))
                        for v in cutlass.range_constexpr(4):
                            value = bits_float(words[v])
                            total[v] = value if p == 0 else total[v] + value
                    tt = g * self.ng + t
                    for v in cutlass.range_constexpr(4):
                        col = lane // 4 + v // 2 * 8
                        red[tt, tile, 0, 2 * (lane % 4) + v % 2, col] = total[v]
                    cute.arch.sync_warp()
                    token = tt * 8 + lane // 2
                    if (lane < 16) & (token < self.m):
                        half = lane % 2 * 8
                        values = [
                            red[tt, tile, 0, lane // 2, half + j] for j in range(8)
                        ]
                        col = (cta * self.rt + tile) * 16 + half
                        self.epilogue(args, token, col, values)

    @cute.jit
    def gather(self, args, red, cta, kpart, tid, lane):
        """Each thread sums one column's K parts in order (the leader also its peers');
        each eight-lane group then gathers the sums for its first lane's 16 B store."""
        total = self.m * self.rt * 16
        for start in range(0, total, self.threads):
            item = start + tid
            v8, half = item % 8, (item // 8) % 2
            rtile, token = (item // 16) % self.rt, item // (16 * self.rt)
            mine = Float32(0)
            if item < total:
                mine = red[token // 8, rtile, 0, token % 8, half * 8 + v8]
                for q in cutlass.range_constexpr(1, self.kq):
                    mine = mine + red[token // 8, rtile, q, token % 8, half * 8 + v8]
                if kpart == 0:
                    for peer in cutlass.range_constexpr(1, self.ks):
                        for q in cutlass.range_constexpr(self.kq):
                            idx = (token // 8, rtile, q, token % 8, half * 8 + v8)
                            mine = mine + peer_load(at(red, *idx), peer)
            values = [cute.arch.shuffle_sync(mine, (lane & 24) + v) for v in range(8)]
            if (item < total) & (v8 == 0) & (kpart == 0):
                col = (cta * self.rt + rtile) * 16 + half * 8
                self.epilogue(args, token, col, values)

    @cute.jit
    def epilogue(self, args, token, col, values):
        out, mc, slot, base = args[2:6]
        if col < self.valid:
            for v in cutlass.range_constexpr(2):
                words = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                offset = token * out.shape[2] + 3584 + 2 * (slot + col + 4 * v)
                multicast_words(mc + Int64(offset) * 2, words, kind="bf16x2")
        elif col >= self.split:
            column = base + col - self.split
            words = tuple(positive_zero(*values[i : i + 2]) for i in range(0, 8, 2))
            offset = token * out.shape[2] + column
            multicast_words(mc + Int64(offset) * 2, words, kind="bf16x2")
