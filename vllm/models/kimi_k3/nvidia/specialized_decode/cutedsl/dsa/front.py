# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSA indexer front: the q_a RMSNorm, the wq_b projection (N 4096, K 1536) and
DsaIndex in one cluster kernel, the indexer path's only step before the scoring when
the MLA query is forked off it (overlap.py "front").

A cluster of 4 CTAs owns one indexer head (128 wq_b rows); CTA ``part`` holds their
K quarter (384 columns) in shared memory, bulk-loaded before the PDL wait. After it:

- the batch's q_a quarter; each row's sum of squares of the quarter goes to every
  CTA of the cluster, which add the four in the same order (the RMSNorm in place,
  BF16, as MlaQuery's scalar form computes it);
- BF16 MMAs (FP32 accumulation): warp w owns head columns 16w..16w+15;
- the partial [B, 128] sums go (one bulk copy per owner) to the CTA owning each
  row (a quarter of the rows each), which adds the four in order, rounds to BF16
  (the projection's output) and quantizes the head as DsaIndex does: FP8 with a
  power-of-two scale, folded into the BF16 head weight;
- CTA head * 4 + part (< B) also inserts that row's indexer key (DsaIndex's code)."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint32

from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    copy_evict,
    copy_g2s,
    copy_s2cluster,
    float_bits,
    fma,
    launch,
    load_bf16x8,
    load_words,
    mma_into,
    pack_fp8,
    query_norm_rsqrt,
    smem_words,
    store_shared_words,
    store_words,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)
from ..mla.query import pow2_ceil
from .index import (
    DIM,
    FP4_WEIGHT,
    HEADS,
    insert_key,
    is_fp4_cache,
    quantize_fp4_query,
)

CLUSTER = 4
KQ = 1536 // CLUSTER  # K columns per CTA
KS = KQ + 32  # padded shared row (elements): 832 B, conflict-free 16 B fragments
CHUNKS = KQ // 8  # 16 B chunks per row quarter


class IndexFront:
    """``batch`` rows; ``key``/``weight``: the wk and wv column offsets in the
    packed projection row; ``mx``: key scales as packed UE8M0 (MX) bytes."""

    def __init__(self, batch: int, key: int, weight: int, mx: bool = False):
        self.batch, self.key, self.weight, self.mx = batch, key, weight, mx
        self.mt = -(-batch // 16)  # M16 tiles
        self.rows = 16 * self.mt
        self.owned = -(-batch // CLUSTER)  # most rows a CTA owns

    @cute.jit
    def __call__(self, *args):
        options = dict(cluster=CLUSTER, preferred_smem_carveout=100)
        launch(self.kernel, args, (CLUSTER, HEADS, 1), 256, **options)

    @cute.kernel
    def kernel(self, args):
        projected, w_qa, wq, norm_w, norm_b, cache, slots = args[0:7]
        q_out, w_out, eps, key_eps = args[7:11]
        tid, _, _ = cute.arch.thread_idx()
        part, head, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        B, RP, MT, OW = self.batch, self.rows, self.mt, self.owned
        smem = cutlass.utils.SmemAllocator()
        w = alloc(smem, BFloat16, (128, KS), 128)
        x = alloc(smem, BFloat16, (RP, KS), 128)
        red = alloc(smem, Float32, (CLUSTER, OW, 128), 128)
        ss = alloc(smem, Float32, (CLUSTER, RP))
        rrs = alloc(smem, Float32, RP)
        bars = [smem.allocate(cutlass.Uint64, 8) for _ in range(4)]
        bar_w, bar_x, bar_ss, bar_red = bars
        # The partial sums, after the MMAs, in place of x: [RP, 128] FP32.
        stage = view(x.iterator, 0, Float32, (RP, 128), (128, 1))
        own = cutlass.max(0, cutlass.min(OW, B - part * OW))  # rows part * OW, ...
        if tid == 0:
            for bar in bars:
                cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(bar_w, 128 * KQ * 2)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_x, B * KQ * 2)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_ss, CLUSTER * B * 4)
            cute.arch.mbarrier_arrive_and_expect_tx(bar_red, CLUSTER * own * 512)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()
        k0 = part * KQ
        if tid < 128:  # this head's rows, their K quarter (bulk copies)
            source = at(wq, head * 128 + tid, k0)
            copy_evict(w.iterator + tid * KS, source, KQ * 2, bar_w)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tid < B:
            copy_g2s(x.iterator + tid * KS, at(projected, tid, k0), KQ * 2, bar_x)
        for c in range(tid, (RP - B) * CHUNKS, 256):  # the M tile's padding rows
            b, j = B + c // CHUNKS, c % CHUNKS
            store_shared_words(x.iterator + b * KS + j * 8, (Uint32(0),) * 4)
        wait(bar_x)
        cute.arch.cluster_wait()  # the peers' barriers are initialized
        # Each row's sum of squares of this quarter, one row per warp, to every CTA.
        for b in range(warp, B, 8):
            total = Float32(0)
            for h in cutlass.range_constexpr(2):
                j = lane + 32 * h
                if j < CHUNKS:
                    for word in smem_words(x.iterator + b * KS + j * 8):
                        lo, hi = unpack_bf16(word)
                        total = fma(hi, hi, fma(lo, lo, total))
            total = warp_sum(total)
            if lane < CLUSTER:
                destination = ss.iterator + part * RP + b
                cute.arch.store_async_dsmem(
                    destination, float_bits(total), bar_ss, lane
                )
        wait(bar_ss)
        if tid < B:
            total = ((ss[0, tid] + ss[1, tid]) + ss[2, tid]) + ss[3, tid]
            rrs[tid] = query_norm_rsqrt(total, eps)
        cute.arch.sync_threads()
        for c in range(tid, B * CHUNKS, 256):  # the RMSNorm in place
            b, j = c // CHUNKS, c % CHUNKS
            rr = rrs[b]
            gamma = load_bf16x8(w_qa.iterator + k0 + j * 8)
            source = x.iterator + b * KS + j * 8
            values: list = []
            for word in smem_words(source):
                values += list(unpack_bf16(word))
            words = [
                bf16x2(values[i] * rr * gamma[i], values[i + 1] * rr * gamma[i + 1])
                for i in range(0, 8, 2)
            ]
            store_shared_words(source, tuple(words))
        wait(bar_w)
        cute.arch.sync_threads()
        # [RP, 384] x [384, 16] per warp. The lanes' K fragments are 8 contiguous
        # columns; A and B use the same K permutation (as MlaQuery.tiled).
        acc = zeros((MT * 2, 4), Float32)
        for kc in cutlass.range_constexpr(KQ // 32):
            k = kc * 32 + 8 * quarter
            z = [
                load_words(
                    w.iterator + (16 * warp + 8 * n + group) * KS + k,
                    count=4,
                    shared=True,
                )
                for n in range(2)
            ]
            for m in cutlass.range_constexpr(MT):
                xa, ya = (
                    load_words(x.iterator + r * KS + k, count=4, shared=True)
                    for r in (16 * m + group, 16 * m + group + 8)
                )
                for n in cutlass.range_constexpr(2):
                    for s in cutlass.range_constexpr(2):
                        a = (xa[2 * s], ya[2 * s], xa[2 * s + 1], ya[2 * s + 1])
                        bb = (z[n][2 * s], z[n][2 * s + 1])
                        mma_into(acc[m * 2 + n, None], a, bb, bf16=True)
        cute.arch.sync_threads()  # x is consumed: the partials take its place
        for m in cutlass.range_constexpr(MT):
            for n in cutlass.range_constexpr(2):
                for h in cutlass.range_constexpr(2):
                    row = 16 * m + group + 8 * h
                    col = 16 * warp + 8 * n + 2 * quarter
                    pair = acc[m * 2 + n, 2 * h], acc[m * 2 + n, 2 * h + 1]
                    words = tuple(float_bits(v) for v in pair)
                    store_shared_words(at(stage, row, col), words)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        for r in cutlass.range_constexpr(CLUSTER):  # each owner's rows, one copy
            rows = max(0, min(OW, B - r * OW))
            if cutlass.const_expr(rows > 0):  # noqa: SIM102
                if tid == r:
                    copy_s2cluster(
                        red.iterator + part * OW * 128,
                        at(stage, r * OW, 0),
                        bar_red,
                        r,
                        size=rows * 512,
                    )
        wait(bar_red)
        cute.arch.cluster_arrive_relaxed()  # the copies into this CTA have landed
        # The owned rows: one per warp, 4 columns per lane (the head's 128).
        for local in range(warp, own, 8):
            b = part * OW + local
            sums = [
                load_words(
                    red.iterator + (r * OW + local) * 128 + 4 * lane,
                    count=4,
                    shared=True,
                    result_type=Float32,
                    word_type="f32",
                )
                for r in range(CLUSTER)
            ]
            values = [
                Float32(BFloat16(((sums[0][i] + sums[1][i]) + sums[2][i]) + sums[3][i]))
                for i in range(4)
            ]
            if cutlass.const_expr(is_fp4_cache(cache)):  # MXFP4 (index.py)
                quantize_fp4_query(values, 4, at(q_out, b, 0, 0), head, lane, lane)
                if lane == 0:
                    wv = Float32(projected[b, self.weight + head])
                    w_out[b, head] = BFloat16(wv * Float32(FP4_WEIGHT))
            else:
                self.fp8_head(values, projected, q_out, w_out, b, head, lane)
        token = head * CLUSTER + part
        if (warp == 0) & (token < B):
            key = projected, self.key, norm_w, norm_b, cache, slots, token, key_eps
            insert_key(*key, self.mx, lane)
        cute.arch.cluster_wait()  # no CTA exits while a peer may read or write it

    @cute.jit
    def fp8_head(self, values, projected, q_out, w_out, b, head, lane):
        """FP8 of one head's 4 columns per lane (a power-of-two head scale) and its
        folded head weight."""
        amax = Float32(1e-10)
        for i in cutlass.range_constexpr(4):
            amax = cute.arch.fmax(amax, abs(values[i]))
        for shift in cutlass.range_constexpr(4, -1, -1):
            amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, 1 << shift))
        scale = pow2_ceil(amax / Float32(448))  # amax >= 1e-10
        inverse = Float32(1) / scale
        word = pack_fp8(tuple(values[i] * inverse for i in range(4)))
        store_words(at(q_out, b, head, 4 * lane), (word,))
        if lane == 0:
            wv = Float32(projected[b, self.weight + head])
            fold = (wv * scale) * Float32((HEADS * DIM) ** -0.5)
            w_out[b, head] = BFloat16(fold)
