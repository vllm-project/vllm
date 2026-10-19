# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cluster MLA query RMSNorm, Q_B projection, W_UK absorption and paged FP8 KV insert.
Static weights load before the PDL wait; current-token data is consumed after it."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int64

from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    copy_evict,
    copy_g2s,
    copy_s2cluster,
    float_bits,
    fma,
    fma2,
    fp8_bits,
    launch,
    ldmatrix,
    load_bf16x8,
    load_volatile,
    load_words,
    mma_into,
    mul2,
    multicast_words,
    prefetch_l2,
    published,
    query_norm_rsqrt,
    smem_words,
    store_fp8_pair,
    store_shared_words,
    unpack_bf16,
    view,
    wait,
    warp_sum,
    zeros,
)

KP = 1536 + 32  # padded Q_B / x row (elements): 3136 B, conflict-free 16 B fragments
# The smallest batch on the tiled kernel (per head count):
# its 16-request tiles beat FP32 FMAs from B4 at 12 heads and from B6 at 6 heads.
TILED_FROM = {12: 4, 6: 6}
Q_SENT = 0xFFFFFFFF  # q mailbox sentinel: four FP8 0xFF bytes, never written


class MlaQuery:
    """A cluster splits one head's 192 Q_B rows; each CTA then absorbs its share of the
    512 latent columns. Small batches (below TILED_FROM) are served by one CTA per
    cluster part with FP32 FMAs; wider batches tile 16-64 requests per CTA (one wave
    of 96 CTAs at B64 for both head counts) and use BF16 MMAs (FP32 accumulation).

    ``dcp`` (decode context parallel size; args then end with this parity's local q
    mailbox, its multicast address and the rank): each CTA also multicasts the query
    bytes it wrote into head block ``rank`` of every rank's Lamport q mailbox
    (mla.dcp.DcpBox), then polls its share of the gathered mailbox until no word is
    the sentinel, so the next kernel reads the gathered query."""

    def __init__(
        self,
        heads: int,
        batch: int,
        *,
        q_b_cp_async: bool = False,
        dcp: int = 0,
    ):
        tiled = TILED_FROM[heads]
        self.heads, self.batch, self.wide = heads, batch, batch >= tiled
        self.dcp = dcp
        self.cluster = 16 if heads == 6 and not self.wide else 8
        self.rows, self.latent = 192 // self.cluster, 512 // self.cluster
        self.groups, self.p_per_group = 256 // self.latent, self.latent // 2
        self.tokens = 1 if batch == 2 else min(batch, 8)
        if self.wide:
            self.tokens = 16 if batch <= 16 else 32 if batch <= 32 or heads < 8 else 64
        self.grid = (self.cluster, heads, (batch + self.tokens - 1) // self.tokens)
        carveout = 100 if batch > 1 else None
        self.options = dict(cluster=self.cluster, preferred_smem_carveout=carveout)
        self.q_b_cp_async = q_b_cp_async

    @cute.jit
    def __call__(self, *args):
        kernel = self.tiled if self.wide else self.kernel
        launch(kernel, args, self.grid, 256, **self.options)

    @cute.jit
    def live(self, token):
        return cutlass.const_expr(not self.wide) or token < self.batch

    @cute.jit
    def publish(self, args, tok0, tid):
        """DCP: this CTA's query bytes (latent columns L*part.. as 16 B, the rope rows
        of Q_B rows R*part.. as 4 B multicast stores) -> every rank's q mailbox (FP8
        bytes are never 0xFF: no word is the sentinel)."""
        output, (box_mc, rank) = args[10], args[12:14]
        part, head, _ = cute.arch.block_idx()
        R, L, D = self.rows, self.latent, self.dcp
        chunks: cutlass.Constexpr = L // 16
        items: cutlass.Constexpr = chunks + R // 4
        cute.arch.sync_threads()
        for i in range(tid, self.tokens * items, 256):
            t, w = i // items, i % items
            p = R * part + 4 * (w - chunks)  # Q_B row of a rope word
            row = ((tok0 + t) * D + rank) * self.heads + head
            if w < chunks:
                byte = L * part + 16 * w
                if self.live(tok0 + t):
                    words = load_words(
                        at(output, tok0 + t, head, byte), count=4, mutable=True
                    )
                    multicast_words(box_mc + Int64(row) * 576 + byte, words)
            elif self.live(tok0 + t) & (p >= 128):
                byte = 512 + p - 128
                word = load_words(
                    at(output, tok0 + t, head, byte), count=1, mutable=True
                )
                multicast_words(box_mc + Int64(row) * 576 + byte, word, kind="b32")

    @cute.jit
    def gathered(self, args, tid):
        """DCP: poll this CTA's share of the gathered mailbox (16 B chunks of the
        B x DCP*H x 576 bytes) until every rank's bytes have landed."""
        mailbox = args[11]
        part, head, tile = cute.arch.block_idx()
        threads: cutlass.Constexpr = self.grid[0] * self.grid[1] * self.grid[2] * 256
        chunks: cutlass.Constexpr = self.batch * self.dcp * self.heads * 36
        first = ((tile * self.heads + head) * self.cluster + part) * 256 + tid
        last = chunks - 1  # threads past the end re-poll it
        for i0 in cutlass.range_constexpr(0, -(-chunks // threads), 8):
            n: cutlass.Constexpr = min(8, -(-chunks // threads) - i0)
            sources = [
                mailbox.iterator + min(first + (i0 + i) * threads, last) * 16
                for i in range(n)
            ]
            loads = [load_volatile(source, count=4) for source in sources]
            for i in cutlass.range_constexpr(n):  # the loads above are in flight
                words = loads[i]
                while not published(words, sentinel=Q_SENT):
                    words = load_volatile(sources[i], count=4)

    @cute.kernel
    def kernel(self, args):
        qkv, w_qa, w_kva, w_qb, w_uk, cache, slots, q_scale_inv = args[0:8]
        k_scale_inv, eps, output = args[8:11]
        tid, _, _ = cute.arch.thread_idx()
        part, head, tok0 = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        B, R, L, G = self.tokens, self.rows, self.latent, self.groups
        smem = cutlass.utils.SmemAllocator()
        qb = alloc(smem, BFloat16, (R, 1536), 128)
        uk = alloc(smem, BFloat16, (128, L))
        qa = alloc(smem, BFloat16, 1536)
        xq = alloc(smem, BFloat16, (B, 1536))
        qrow = alloc(smem, Float32, (R, B))
        qn = alloc(smem, Float32, (B, 128))
        red = alloc(smem, Float32, (G, B, L))
        ss = alloc(smem, Float32, (8, max(B, 2)))
        if cutlass.const_expr(not self.q_b_cp_async):
            barw = smem.allocate(cutlass.Uint64, 8)
        barq = smem.allocate(cutlass.Uint64, 8)
        if tid == 0:
            if cutlass.const_expr(not self.q_b_cp_async):
                cute.arch.mbarrier_init(barw, 1)
            cute.arch.mbarrier_init(barq, 1)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(not self.q_b_cp_async):
                cute.arch.mbarrier_arrive_and_expect_tx(barw, R * 1536 * 2)
                copy_evict(
                    qb.iterator,
                    w_qb.iterator + (head * 192 + part * R) * 1536,
                    size=R * 1536 * 2,
                    barrier=barw,
                )
            cute.arch.mbarrier_arrive_and_expect_tx(barq, B * 128 * 4)
        cute.arch.cluster_arrive_relaxed()
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            copy_evict(uk.iterator + p * L + j * 8, at(w_uk, head, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, w_qa.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.q_b_cp_async):
            for c in range(tid, R * 1536 // 8, 256):
                dst = qb.iterator + c * 8
                src = w_qb.iterator + (head * 192 + part * R) * 1536 + c * 8
                copy_evict(dst, src)
            cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        qsi = q_scale_inv[0]
        xv = cute.make_rmem_tensor((B, 8), Float32)
        for b in cutlass.range_constexpr(B):
            total = Float32(0)
            if (tid < 192) & self.live(tok0 + b):
                values = load_bf16x8(at(qkv, tok0 + b, tid * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    xv[b, i] = values[i]
                    total = fma(values[i], values[i], total)
            total = warp_sum(total)
            if lane == 0:
                ss[warp, b] = total
        cute.arch.cp_async_wait_group(1 if self.q_b_cp_async else 0)
        cute.arch.sync_threads()
        if tid < 192:
            weights = load_bf16x8(qa.iterator + tid * 8, shared=True)
            for b in cutlass.range_constexpr(B):
                total = Float32(0)
                for w in cutlass.range_constexpr(8):
                    total += ss[w, b]
                rr = query_norm_rsqrt(total, eps)
                for i in cutlass.range_constexpr(8):
                    xq[b, tid * 8 + i] = BFloat16(xv[b, i] * rr * weights[i])
        cute.arch.sync_threads()
        if cutlass.const_expr(self.q_b_cp_async):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            wait(barw)
        for row in range(warp, R, 8):
            acc = zeros((B, 2), Float32)
            for chunk in cutlass.range_constexpr(6):
                k = (chunk * 32 + lane) * 8
                weights = load_bf16x8(qb.iterator + row * 1536 + k, shared=True)
                for b in cutlass.range_constexpr(B):
                    values = load_bf16x8(xq.iterator + b * 1536 + k, shared=True)
                    for i in cutlass.range_constexpr(8):
                        acc[b, i % 2] = fma(weights[i], values[i], acc[b, i % 2])
            for b in cutlass.range_constexpr(B):
                value = warp_sum(acc[b, 0] + acc[b, 1])
                if lane == 0:
                    qrow[row, b] = Float32(BFloat16(value))
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        nrows = min(R, max(0, 128 - R * part))
        for i in range(tid, nrows * B * self.cluster, 256):
            dst, jb = i % self.cluster, i // self.cluster
            j, b = jb % nrows, jb // nrows
            cute.arch.store_async_dsmem(
                qn.iterator + b * 128 + R * part + j, float_bits(qrow[j, b]), barq, dst
            )
        for i in range(tid, R * B, 256):
            j, b = i % R, i // R
            p = R * part + j
            if (p >= 128) & self.live(tok0 + b):
                output[tok0 + b, head, 512 + p - 128] = fp8_bits(qrow[j, b] * qsi)
        wait(barq)
        latent_col, group = tid % L, tid // L
        acc = zeros((B, 2), Float32)
        for i in cutlass.range_constexpr(self.p_per_group):
            p = group * self.p_per_group + i
            w = Float32(uk[p, latent_col])
            for b in cutlass.range_constexpr(B):
                acc[b, i % 2] = fma(qn[b, p], w, acc[b, i % 2])
        for b in cutlass.range_constexpr(B):
            red[group, b, latent_col] = acc[b, 0] + acc[b, 1]
        cute.arch.sync_threads()
        for i in range(tid, L * B, 256):
            b, ll = i // L, i % L
            value = Float32(0)
            for g in cutlass.range_constexpr(G):
                value += red[g, b, ll]
            if self.live(tok0 + b):
                rounded = Float32(BFloat16(value))
                output[tok0 + b, head, L * part + ll] = fp8_bits(rounded * qsi)
        if cutlass.const_expr(self.dcp):
            self.publish(args, tok0, tid)
        valid = (head == 0) & (part < B) & self.live(tok0 + part)
        self.insert(args, ss, tok0, part, valid, tid)
        if cutlass.const_expr(self.dcp):
            self.gathered(args, tid)

    @cute.jit
    def insert(self, args, ss, tok0, j, valid, tid):
        """RMSNorm the current latent KV row of request tok0 + j and write it, with
        k_pe, to its FP8 cache slot."""
        qkv, w_kva, (cache, slots), (k_scale_inv, eps) = (
            args[0],
            args[2],
            args[5:7],
            args[8:10],
        )
        lane, warp = tid % 32, tid // 32
        slot = cutlass.Int64(-1)
        if valid:
            slot = slots[tok0 + j]
        if slot >= 0:
            ksi = k_scale_inv[0]
            xx = cute.make_rmem_tensor(8, Float32)
            total = Float32(0)
            if tid < 64:
                values = load_bf16x8(at(qkv, tok0 + j, 1536 + tid * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    xx[i] = values[i]
                    total = fma(values[i], values[i], total)
            total = warp_sum(total)
            cute.arch.sync_threads()
            if (lane == 0) & (warp < 2):
                ss[warp, 0] = total
            cute.arch.sync_threads()
            page, offset = slot // cache.shape[1], slot % cache.shape[1]
            if tid < 64:
                mean = fma(ss[0, 0] + ss[1, 0], Float32(1 / 512), eps)
                rr = cute.math.rsqrt(mean, fastmath=True)
                weights = load_bf16x8(w_kva.iterator + tid * 8)
                for i in cutlass.range_constexpr(8):
                    normed = Float32(BFloat16(xx[i] * rr * weights[i]))
                    cache[page, offset, tid * 8 + i] = fp8_bits(normed * ksi)
            elif tid < 72:
                c = tid - 64
                values = load_bf16x8(at(qkv, tok0 + j, 2048 + c * 8), mutable=True)
                for i in cutlass.range_constexpr(8):
                    cache[page, offset, 512 + c * 8 + i] = fp8_bits(values[i] * ksi)

    @cute.kernel
    def tiled(self, args):
        """B>8: T requests per CTA. Q_B fragments load to registers before the PDL
        wait; in-place RMSNorm of the staged rows, split-K Q_B MMAs over the warps,
        nope rows gathered from the six producing CTAs by bulk DSMEM copies (into the
        dead x rows), then W_UK MMAs (warp w owns latent columns 8w..8w+7)."""
        qkv, w_qa, w_qb, w_uk, q_scale_inv = (args[i] for i in (0, 1, 3, 4, 7))
        eps, output = args[9], args[10]
        tid, _, _ = cute.arch.thread_idx()
        part, head, tok0 = cute.arch.block_idx()
        tok0 *= self.tokens
        smem = cutlass.utils.SmemAllocator()
        T, R, L = self.tokens, self.rows, self.latent
        MT, senders = T // 16, (128 + R - 1) // R
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        xq = alloc(smem, BFloat16, (T, KP), 128)
        uk = alloc(smem, BFloat16, (128, L + 8), 128)  # 144 B rows for ldmatrix
        qa = alloc(smem, BFloat16, 1536)
        ss = alloc(smem, Float32, (8, 2))
        barq = smem.allocate(cutlass.Uint64, 8)
        # After the Q_B MMAs the x rows hold the split-K partials, then our nope rows
        # and the six producers' (peers write them only after our cluster arrival).
        red = view(xq.iterator, 0, Float32, (8, T * R // 32, 32), (T * R, 32, 1))
        qblk = view(xq.iterator, 8 * T * R * 4, BFloat16, (T, R), (R, 1))
        qn = view(xq.iterator, 9 * T * R * 4, BFloat16, (senders, T, R), (T * R, R, 1))
        if tid == 0:
            cute.arch.mbarrier_init(barq, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(barq, senders * T * R * 2)
        # Split-K Q_B: warp w owns k = 192w..192w+191 (8 values per lane and 32-wide
        # chunk; the x fragments use the same permutation). L2 now, registers later.
        rows = head * 192 + part * R + group
        qb_at = [
            [
                at(w_qb, rows + 8 * n, (6 * warp + cc) * 32 + 8 * quarter)
                for n in range(3)
            ]
            for cc in range(6)
        ]
        for chunk in qb_at:
            for address in chunk:
                prefetch_l2(address)
        for c in range(tid, 128 * L // 8, 256):
            p, j = c // (L // 8), c % (L // 8)
            dst = uk.iterator + p * (L + 8) + j * 8
            copy_evict(dst, at(w_uk, head, p, part * L + j * 8))
        if tid < 192:
            copy_evict(qa.iterator + tid * 8, w_qa.iterator + tid * 8)
        cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        for c in range(tid, T * 192, 256):
            t, j = c // 192, c % 192
            if self.live(tok0 + t):
                copy_g2s(xq.iterator + t * KP + j * 8, at(qkv, tok0 + t, j * 8))
        cute.arch.cp_async_commit_group()
        qsi = q_scale_inv[0]
        # The KV insert overlaps the staging of x (CTA j of the tile owns request j).
        j = head * self.cluster + part
        self.insert(args, ss, tok0, j, (j < T) & self.live(tok0 + j), tid)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # Packed FP32 pairs: the same IEEE ops in the same order as the scalar form.
        gamma = []
        for i in cutlass.range_constexpr(6):
            gamma += list(load_bf16x8(qa.iterator + 256 * i + 8 * lane, shared=True))
        for pair in cutlass.range_constexpr(T // 16):  # requests warp + 8k, two at once
            xrows = [xq.iterator + (warp + 16 * pair + 8 * h) * KP for h in range(2)]
            values: list = []
            rr: list = []
            for h in cutlass.range_constexpr(2):
                values.append([])
                for i in cutlass.range_constexpr(6):
                    for word in smem_words(xrows[h] + 256 * i + 8 * lane):
                        values[h] += list(unpack_bf16(word))
            for h in cutlass.range_constexpr(2):
                zero = Float32(0), Float32(0)
                chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
                for e in cutlass.range_constexpr(0, 48, 2):
                    x = values[h][e], values[h][e + 1]
                    chains[e % 4 // 2] = fma2(x, x, chains[e % 4 // 2])
                (c0, c1), (c2, c3) = chains
                rr.append(query_norm_rsqrt(warp_sum((c0 + c1) + (c2 + c3)), eps))
            for h in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(6):
                    words = []
                    for e in cutlass.range_constexpr(8 * i, 8 * i + 8, 2):
                        x = values[h][e], values[h][e + 1]
                        y = mul2(mul2(x, (rr[h], rr[h])), (gamma[e], gamma[e + 1]))
                        words.append(bf16x2(*y))
                    store_shared_words(xrows[h] + 256 * i + 8 * lane, tuple(words))
        qbr = [[load_words(address, count=4) for address in chunk] for chunk in qb_at]
        cute.arch.sync_threads()
        acc = zeros((MT * 3, 4), Float32)
        for cc in cutlass.range_constexpr(6):
            k = (6 * warp + cc) * 32 + 8 * quarter
            z = qbr[cc]
            for m in cutlass.range_constexpr(MT):
                x, y = (
                    load_words(xq.iterator + r * KP + k, count=4, shared=True)
                    for r in (16 * m + group, 16 * m + group + 8)
                )
                for n in cutlass.range_constexpr(3):
                    for s in cutlass.range_constexpr(2):
                        a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                        b = (z[n][2 * s], z[n][2 * s + 1])
                        mma_into(acc[m * 3 + n, None], a, b, bf16=True)
        cute.arch.sync_threads()
        cute.arch.cluster_arrive_relaxed()  # x is consumed: peers may fill qn
        for i in cutlass.range_constexpr(MT * 12):
            red[warp, i, lane] = acc[i // 4, i % 4]
        cute.arch.sync_threads()
        for e in range(tid, T * R, 256):
            i, ln = e // 32, e % 32
            value = Float32(0)
            for w in cutlass.range_constexpr(8):
                value += red[w, i, ln]
            value = Float32(BFloat16(value))
            tile, c = i // 4, i % 4
            t = 16 * (tile // 3) + ln // 4 + 8 * (c // 2)
            r = 8 * (tile % 3) + 2 * (ln % 4) + c % 2
            p = R * part + r
            if p < 128:
                qblk[t, r] = BFloat16(value)
            elif self.live(tok0 + t):
                output[tok0 + t, head, 512 + p - 128] = fp8_bits(value * qsi)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        cute.arch.cluster_wait()
        if (part < senders) & (tid < self.cluster):
            copy_s2cluster(
                qn.iterator + part * T * R, qblk.iterator, barq, tid, size=T * R * 2
            )
        wait(barq)
        cute.arch.cluster_arrive_relaxed()  # peers may exit once qblk is delivered
        b_frags = [
            ldmatrix(uk.iterator + (32 * kp + lane) * (L + 8) + 8 * warp, trans=True)
            for kp in range(4)
        ]
        for m in cutlass.range_constexpr(MT):
            out = zeros(4, Float32)
            t = 16 * m + lane % 8 + 8 * ((lane // 8) % 2)
            for ks in cutlass.range_constexpr(8):
                p = 16 * ks + 8 * (lane // 16)
                a = ldmatrix(qn.iterator + ((p // R) * T + t) * R + p % R)
                b = b_frags[ks // 2][2 * (ks % 2) : 2 * (ks % 2) + 2]
                mma_into(out, a, b, bf16=True)
            for h in cutlass.range_constexpr(2):  # rows group, group + 8: 2 bytes
                row = 16 * m + group + 8 * h
                if self.live(tok0 + row):
                    lo, hi = (Float32(BFloat16(out[2 * h + e])) * qsi for e in range(2))
                    col = L * part + 8 * warp + 2 * quarter
                    store_fp8_pair(at(output, tok0 + row, head, col), lo, hi)
        if cutlass.const_expr(self.dcp):
            self.publish(args, tok0, tid)
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.dcp):
            self.gathered(args, tid)
