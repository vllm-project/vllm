# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized sparse MLA decode of a DCP K3.1 DSA layer on a per-tensor FP8 cache
(`SparseDcpDecode`): every rank's gathered query heads (the q mailbox) over this
rank's compacted flat KV rows (the exact top-k merge's `rows` / `valid`).

Why: at the latency checkpoint (DCP8, 5 verify rows) each row attends ~256 local
keys. FlashInfer's prims_ts decode (split-KV over 80 CTAs plus its reduction,
launched only after the merge: it reads its inputs before its PDL wait) and
DcpValue's push of its rows hold the chain ~18 us at 3-25% DRAM for ~1 MB of KV.
Here a cluster of SPLIT CTAs per (16 heads, row) splits the row's keys, launches
early under PDL (reads nothing before its wait), merges its partials over DSMEM
and pushes the merged rows straight into the owners' DcpValue mailbox.

Per CTA (row b, heads 16 m .. 16 m + 15, its 1 / SPLIT of the row's keys), per
chunk of up to CHUNK keys:
  gather   the Q rows (first chunk) and the chunk's KV rows (cp.async.bulk, one
           per 576 B row, on one mbarrier); V rows past the count up to a 32-key
           step are zeroed
  QK       warp w < CHUNK / 8: keys 8 w .. 8 w + 7 x the 16 heads, FP8 mma.sync
           m16n8k32 (FP32 accumulation) over the 576 bytes
  softmax  the chunk's row max (shared memory), the running max, P =
           exp2(s * scale_log2 - max + log2(448)) quantized to E4M3 (satfinite,
           RN), FP32 sums of the unquantized P, the accumulators rescaled when the
           running max rises
  PV       warp w: latent columns 32 w .. 32 w + 31, E4M3 P x E4M3 V (V transposed
           in registers with byte permutes), FP32 accumulation
then the cluster merges (FP32): CTA s of the cluster takes columns s / SPLIT of the
16 heads, weights each CTA's partial by exp2(max_c - max), and writes
BF16(sum * k_scale / total) and LSE = max + log2(total / 448) (`push`: into the
owners' mailbox slot, base 2, as DcpValue's push; else the latent [B, Q, 512] and
natural-log LSE [B, Q] DcpValue reads).

Wide CTAs (`tiles` > 1, SPARSE_TILES): a CTA takes `tiles` such 16-head tiles
over the same keys. The KV rows it gathers feed every tile, QK runs on QK_WARPS
warps per tile (warps a one-tile CTA leaves idle in QK), each PV warp reuses its
transposed V fragments for every tile, and the CTA issues its q mailbox loads right
after the PDL wait. `bulk` (SPARSE_BULK): each CTA stages its partials of a
peer's columns in shared memory and moves them with one bulk shared::cluster copy
per peer instead of per-thread st.async stores. Per head and key the operations
and their order are unchanged (key split, chunks, QK chains, row maxima, P anchor
and rounding, sums in warp order, PV k-steps, the merge in CTA order) and the
partials move as the same FP32 bits, so both match the one-tile kernel bit for bit.

Precision: FlashInfer's sparse MLA decode (E4M3 query and cache, FP32 logits, E4M3
P at 448 of the running row max of its split, FP32 sums of the unquantized P, FP32
accumulation, one BF16 rounding of the output). FlashInfer's split-KV partials are
BF16 (merged by its reduction); here they merge in FP32, so only the order of the
sums, where the P scale anchors and the partials' rounding differ.

Rows of count 0 (padded rows; a rank owning none of a row's keys) write zeros and
LSE -inf (pushed: only the LSE EMPTY_LSE)."""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import fake, stream, to_cute
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    copy_g2s,
    copy_s2cluster,
    fence_tmem,
    float_bits,
    launch,
    ldmatrix,
    load_tmem,
    load_words,
    mma_into,
    named_barrier,
    no_neg_zero,
    pack_fp8,
    push_float,
    release_tmem,
    shared_address,
    store_relaxed,
    store_shared_words,
    store_words,
    wait,
    zeros,
)
from ..mla.dcp import (
    EMPTY_LSE,
    Q_SENT,
    TCS_ON,
    TCS_PAIR_OWN,
    TCS_ROWS,
    TCS_WIDE_ON,
    TCS_WIDE_ROWS,
)

ROW = 576  # FP8 bytes per cache / query row (512 latent + 64 rope)
LATENT = 512
SPLIT = 4  # CTAs per (16 heads, row): a cluster
CHUNK = 64  # keys per CTA per shared-memory chunk (DCP8: ~2048 / 8 / SPLIT)
PITCH = ROW + 16  # 37 x 16 B: the 8 rows of an ldmatrix hit distinct banks
P_PITCH = CHUNK + 16  # 5 x 16 B
HEADS = 16  # query heads per head tile (one m16 tile)
QK_WARPS = CHUNK // 8  # one key n-tile each
QK_CHAINS = 3  # QK accumulator chains (18 k-steps of 32 B)
THREADS = 512
WARPS = THREADS // 32
COLS = LATENT // SPLIT  # merged columns per CTA
# Decode rows (DCP) the sparse decode serves instead of FlashInfer (96 gathered
# heads: 24 CTAs, 6 clusters of 4, per row). DCP8 DSA layer microbenchmark at
# AgentX contexts, GB300, us FlashInfer -> sparse: 5 rows 60.7 -> 51.9, 8 rows
# (q1) 67.3 -> 63.7, 10 rows 65.6 -> 66.3, 16 rows 71.1 -> 80.6, 20 rows 77.4
# -> 85.2 (past ~one wave of clusters the split-KV FlashInfer decode wins).
SPARSE_DCP = "1-8:sparse"
# Head tiles per CTA (batch tables per DCP size; default 1) and the bulk cluster
# exchange (1: on) of the pushing decode. At 96 gathered heads one tile per CTA is
# 24 CTAs per row; with one 512-thread CTA per SM the clusters of 7-8 rows take two
# waves, two tiles per CTA one. Same results bit for bit. Set from a one-GPU
# microbenchmark of the decode alone (DCP16: 96 gathered heads, ~130 keys per row
# and rank, L2 flushed): two tiles + the bulk exchange ahead at 7-8 rows; at 4
# rows (one wave already) one tile stays ahead.
# DCP8 entries (TP8 + DCP8, DSpark k = 4, 1-3 requests = 5-15 rows): set by a
# decode-step A/B there.
SPARSE_TILES = {16: "7-8:2", 8: "5-15:2"}
SPARSE_BULK = {16: "7-8:1", 8: "5-15:1"}
# P-exchange form of the pushing decode, batch tables per DCP size (1: on): the
# cluster CTAs of a row exchange their split's E4M3 P and row statistics instead of
# the FP32 partials, and each CTA runs the PV of every split over its own merged
# columns. Same results bit for bit. DCP16 at 7-8 rows: set by a one-GPU A/B of the
# decode and a 16-rank decode-step A/B.
SPARSE_PX = {16: "7-8:1"}
PX_VPITCH = COLS + 16  # another split's V columns of this CTA, per key
LOG2_P_SCALE = math.log2(448.0)  # E4M3 P at 448 of the row max (FlashInfer's)
LN2 = math.log(2.0)


@asm(Uint32, pure=True)
def prmt(a, b, selector: int):
    return f"prmt.b32 $0,$1,$2,{selector};", "=r,r,r", (a, b)


@asm()
def store_fp8_pair_shared(pointer, low: Float32, high: Float32):
    """E4M3 (satfinite, RN) of `low`, `high` as two bytes of shared memory."""
    code = "{ .reg .b16 pair; cvt.rn.satfinite.e4m3x2.f32 pair,$2,$1; "
    code += "st.shared.b16 [$0],pair; }"
    return code, "r,f,f,~{memory}", (pointer, low, high)


def transpose4(w):
    """Bytes of four words -> four words of bytes: out[j] = (w0.j, w1.j, w2.j,
    w3.j), byte 0 the lowest."""
    lo01, hi01 = prmt(w[0], w[1], 0x5140), prmt(w[0], w[1], 0x7362)
    lo23, hi23 = prmt(w[2], w[3], 0x5140), prmt(w[2], w[3], 0x7362)
    return (
        prmt(lo01, lo23, 0x5410),
        prmt(lo01, lo23, 0x7632),
        prmt(hi01, hi23, 0x5410),
        prmt(hi01, hi23, 0x7632),
    )


class SparseDcpDecode:
    """`batch` rows, `heads` gathered query heads (a multiple of 16 `tiles`;
    the q mailbox and outputs hold `padded` rows per token), the merge's row width
    `width` (its K). `push` (owner heads H, DCP, mailbox rows): push the
    merged rows and base-2 LSEs into their owners' DcpValue "bf16" mailbox and
    re-arm the q mailbox rows read (DcpValue `pushed` then only merges).
    `tiles`: 16-head tiles per CTA; `bulk`: the cluster exchange as one bulk
    copy per peer CTA (see the module docstring)."""

    def __init__(self, batch, heads, padded, width, push=None, tiles=1, bulk=False):
        assert heads % (HEADS * tiles) == 0 and heads <= padded
        assert QK_WARPS * tiles <= WARPS, tiles
        self.batch, self.heads, self.padded, self.width = batch, heads, padded, width
        self.push, self.tiles, self.bulk = push, tiles, bulk
        self.cta_heads = HEADS * tiles
        self.grid = (heads // self.cta_heads * SPLIT, batch, 1)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(self.tiles > 1):
            # One CTA per SM: the 128 registers per thread of that 512-thread
            # launch (the tiles' accumulators would exceed it unbounded).
            launch(
                self.kernel,
                args,
                self.grid,
                THREADS,
                cluster=(SPLIT, 1, 1),
                min_blocks_per_mp=1,
            )
        else:
            launch(self.kernel, args, self.grid, THREADS, cluster=(SPLIT, 1, 1))

    @cute.kernel
    def kernel(self, args):
        query, kv, rows_in, valid_in = args[:4]
        scale_log2, out_scale = args[6:8]
        T: cutlass.Constexpr = self.tiles
        CH: cutlass.Constexpr = self.cta_heads
        smem = cutlass.utils.SmemAllocator()
        kvs = alloc(smem, Uint8, (CHUNK, PITCH), 128)
        qs = alloc(smem, Uint8, (CH, PITCH), 128)
        ps = alloc(smem, Uint8, (CH, P_PITCH), 128)
        red = alloc(smem, Float32, (2, QK_WARPS, CH))
        # The cluster's partials of this CTA's columns: rows and (max, sum) per
        # source CTA, pushed by the sources (st.async on ``inbound``).
        inbox = alloc(smem, Float32, (SPLIT, CH, COLS), 16)
        stats = alloc(smem, Float32, (SPLIT, 2, CH))
        if cutlass.const_expr(self.bulk):
            # This CTA's partials of the other CTAs' columns, one block per peer.
            staged = alloc(smem, Float32, (SPLIT - 1, CH, COLS), 128)
        bar = smem.allocate(cutlass.Uint64, 8)
        inbound = smem.allocate(cutlass.Uint64, 8)
        tid, _, _ = cute.arch.thread_idx()
        x, b, _ = cute.arch.block_idx()
        tile, split = x // SPLIT, x % SPLIT
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        # QK and P: warp w < QK_WARPS * T takes head tile qt = w // QK_WARPS and
        # keys 8 kw .. 8 kw + 7 (kw = w % QK_WARPS) of the chunk.
        qt, kw = 0, warp
        if cutlass.const_expr(T > 1):
            qt, kw = warp // QK_WARPS, warp % QK_WARPS
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(inbound, 1)
            cute.arch.mbarrier_init_fence()
            inbound_bytes = SPLIT * (CH * COLS + 2 * CH) * 4
            if cutlass.const_expr(self.bulk):  # its own columns: plain stores
                inbound_bytes = ((SPLIT - 1) * CH * COLS + SPLIT * 2 * CH) * 4
            cute.arch.mbarrier_arrive_and_expect_tx(inbound, inbound_bytes)
        cute.arch.sync_threads()
        # Every CTA's ``inbound`` is armed before any peer pushes (waited below).
        cute.arch.cluster_arrive_relaxed()
        # Nothing above reads the inputs: the kernel before may launch this early.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        chunks16 = ROW // 16
        q_loads: cutlass.Constexpr = -(-CH * chunks16 // THREADS)
        q_words = []
        if cutlass.const_expr(T > 1):
            # The q mailbox rows of every tile in one round right after the wait,
            # ahead of the counts and rows the KV gather waits on (a clamped index:
            # no branch); stored to shared memory with the first chunk. Loading
            # them after the gather instead spills more registers.
            for i in cutlass.range_constexpr(q_loads):
                h_flat = min(tid + THREADS * i, Int32(CH * chunks16 - 1))
                h_row, h_chunk = h_flat // chunks16, h_flat % chunks16
                h_src = at(query, b, tile * CH + h_row, 16 * h_chunk)
                q_words.append(load_words(h_src, count=4, mutable=True))
        n = valid_in[b]
        n = max(min(n, Int32(self.width)), Int32(0))
        # This CTA's keys: [first, first + mine).
        per = (n + SPLIT - 1) // SPLIT
        first = min(n, split * per)
        mine = min(n - first, per)
        kv_base = kv.iterator.toint()
        neg_inf = Float32(-math.inf)
        # Rows group, group + 8 of tile t at 2 t, 2 t + 1 (log2 units).
        m_run = zeros(2 * T, Float32)
        m_run.fill(neg_inf)
        l_run = zeros(2 * T, Float32)
        acc = zeros((T, 4, 4), Float32)  # tile t, latent n-tiles j: columns 4 g + j
        # ldmatrix lane addresses: A (16 heads x 32 B), B (8 keys x 2 x 32 B).
        a_row = lane % 8 + 8 * ((lane // 8) % 2)
        a_col = 16 * (lane // 16)
        b_row = 8 * kw + lane % 8
        b_col = 16 * (lane // 8)  # k-steps 2 i, 2 i + 1
        phase = Int32(0)
        chunks = (mine + CHUNK - 1) // CHUNK
        for c in cutlass.range(0, chunks, unroll=1):
            base = first + c * CHUNK
            count = min(first + mine - base, Int32(CHUNK))
            span = (count + 31) // 32 * 32  # keys the PV steps read
            if c > 0:
                cute.arch.sync_threads()  # the previous chunk's readers are done
            if tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(bar, count * ROW)
            if tid < count:
                src = kv_base + Int64(rows_in[b, base + tid]) * ROW
                copy_g2s(kvs.iterator + tid * PITCH, src, ROW, bar)
            elif tid < span:  # V past the count: zero (P is zero there)
                for k in cutlass.range_constexpr(LATENT // 16):
                    dst = kvs.iterator + tid * PITCH + 16 * k
                    store_words(dst, (Uint32(0),) * 4, shared=True)
            if c == 0:
                # The q mailbox rows: generic loads, as the multicast stores that
                # filled them and QueryWait's poll.
                for i in cutlass.range_constexpr(q_loads):
                    flat = tid + THREADS * i
                    if flat < CH * chunks16:
                        q_row, q_chunk = flat // chunks16, flat % chunks16
                        if cutlass.const_expr(T > 1):
                            words = q_words[i]
                        else:
                            src = at(query, b, tile * CH + q_row, 16 * q_chunk)
                            words = load_words(src, count=4, mutable=True)
                        dst = qs.iterator + q_row * PITCH + 16 * q_chunk
                        store_words(dst, words, shared=True)
            wait(bar, phase, scope="cta")
            phase = phase ^ 1
            cute.arch.sync_threads()  # the zeroed V rows
            # ---- QK: warp w < QK_WARPS * T, tile qt, keys 8 kw .. 8 kw + 7 -----------
            s = zeros(4, Float32)
            if (warp < QK_WARPS * T) & (8 * kw < span):
                parts = zeros((QK_CHAINS, 4), Float32)
                for kp in cutlass.range_constexpr(ROW // 64):
                    k = ldmatrix(kvs.iterator + b_row * PITCH + 64 * kp + b_col)
                    for h in cutlass.range_constexpr(2):
                        ks = 2 * kp + h
                        qk_row = 16 * qt + a_row  # this warp's tile
                        a = ldmatrix(qs.iterator + qk_row * PITCH + 32 * ks + a_col)
                        b_frag = (k[2 * h], k[2 * h + 1])
                        mma_into(parts[ks % QK_CHAINS, None], a, b_frag, fp8=True)
                for e in cutlass.range_constexpr(4):
                    total = parts[0, e]
                    for chain in cutlass.range_constexpr(1, QK_CHAINS):
                        total += parts[chain, e]
                    s[e] = total
            top = zeros(2, Float32)
            top.fill(neg_inf)
            for e in cutlass.range_constexpr(4):
                key = 8 * kw + 2 * quarter + e % 2
                value = neg_inf
                if key < count:
                    value = s[e] * scale_log2
                s[e] = value
                top[e // 2] = max(top[e // 2], value)
            for r in cutlass.range_constexpr(2):
                for shift in cutlass.range_constexpr(2):
                    other = cute.arch.shuffle_sync_bfly(top[r], 1 << shift)
                    top[r] = max(top[r], other)
            if (quarter == 0) & (warp < QK_WARPS * T):
                red[0, kw, 16 * qt + group] = top[0]
                red[0, kw, 16 * qt + group + 8] = top[1]
            cute.arch.sync_threads()
            alpha = zeros(2 * T, Float32)
            alpha.fill(1.0)
            m_new = zeros(2 * T, Float32)
            for t in cutlass.range_constexpr(T):
                for r in cutlass.range_constexpr(2):
                    i = 2 * t + r
                    chunk_max = red[0, 0, 16 * t + group + 8 * r]
                    for w in cutlass.range_constexpr(1, QK_WARPS):
                        chunk_max = max(chunk_max, red[0, w, 16 * t + group + 8 * r])
                    m_new[i] = max(m_run[i], chunk_max)
                    if m_run[i] != neg_inf:
                        alpha[i] = cute.math.exp2(m_run[i] - m_new[i], fastmath=True)
            own = m_new  # the running maxima of this warp's QK tile
            if cutlass.const_expr(T > 1):
                own = zeros(2, Float32)
                for t in cutlass.range_constexpr(T):
                    if qt == t:  # warp-uniform
                        for r in cutlass.range_constexpr(2):
                            own[r] = m_new[2 * t + r]
            # ---- P (E4M3 at 448 of the running max) and its FP32 sums -------------
            sums = zeros(2, Float32)
            if (warp < QK_WARPS * T) & (8 * kw < span):
                for r in cutlass.range_constexpr(2):
                    bias = own[r] - Float32(LOG2_P_SCALE)
                    p = []
                    for e in cutlass.range_constexpr(2):
                        value = s[2 * r + e]
                        prob = Float32(0)
                        if value != neg_inf:
                            prob = cute.math.exp2(value - bias, fastmath=True)
                        sums[r] += prob
                        p.append(prob)
                    key = 8 * kw + 2 * quarter
                    dst = ps.iterator + (16 * qt + group + 8 * r) * P_PITCH + key
                    store_fp8_pair_shared(dst, p[0], p[1])
            for r in cutlass.range_constexpr(2):
                for shift in cutlass.range_constexpr(2):
                    sums[r] += cute.arch.shuffle_sync_bfly(sums[r], 1 << shift)
            if (quarter == 0) & (warp < QK_WARPS * T):
                red[1, kw, 16 * qt + group] = sums[0]
                red[1, kw, 16 * qt + group + 8] = sums[1]
            cute.arch.sync_threads()
            for t in cutlass.range_constexpr(T):
                for r in cutlass.range_constexpr(2):
                    i = 2 * t + r
                    total = red[1, 0, 16 * t + group + 8 * r]
                    for w in cutlass.range_constexpr(1, QK_WARPS):
                        total += red[1, w, 16 * t + group + 8 * r]
                    l_run[i] = l_run[i] * alpha[i] + total
                    m_run[i] = m_new[i]
                    for j in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(2):
                            acc[t, j, 2 * r + e] *= alpha[i]
            # ---- PV: warp w, latent columns 32 w .. 32 w + 31, every tile ----------
            col = 32 * warp + 4 * group
            for ks in cutlass.range_constexpr(CHUNK // 32):
                if 32 * ks < span:
                    a = ldmatrix(ps.iterator + a_row * P_PITCH + 32 * ks + a_col)
                    halves = []
                    for half in cutlass.range_constexpr(2):
                        words = []
                        for i in cutlass.range_constexpr(4):
                            key = 32 * ks + 16 * half + 4 * quarter + i
                            src = kvs.iterator + key * PITCH + col
                            words.append(load_words(src, count=1, shared=True)[0])
                        halves.append(transpose4(words))
                    for t in cutlass.range_constexpr(T):
                        if cutlass.const_expr(t > 0):  # tile 0's P was read above
                            pv_row = 16 * t + a_row
                            pv_at = pv_row * P_PITCH + 32 * ks + a_col
                            a = ldmatrix(ps.iterator + pv_at)
                        for j in cutlass.range_constexpr(4):
                            b_frag = (halves[0][j], halves[1][j])
                            mma_into(acc[t, j, None], a, b_frag, fp8=True)
        # ---- push this CTA's partials to the column owners -----------------------
        if cutlass.const_expr(self.bulk):
            self.stage(inbox, staged, acc, split, warp, group, quarter)
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.bulk):  # noqa: SIM102 (compile time)
            if tid < SPLIT - 1:  # one copy per peer (staged block tid)
                dst_cta = tid
                if tid >= split:
                    dst_cta = tid + 1
                inbox_ptr, block_ptr = at(inbox, split, 0, 0), at(staged, tid, 0, 0)
                size = CH * COLS * 4
                copy_s2cluster(inbox_ptr, block_ptr, inbound, dst_cta, size=size)
        owner = warp // (COLS // 32)  # columns 32 w .. 32 w + 31
        for t in cutlass.range_constexpr(T):
            for r in cutlass.range_constexpr(2):
                row = 16 * t + group + 8 * r
                if cutlass.const_expr(not self.bulk):
                    for e in cutlass.range_constexpr(2):
                        # acc[t, j, 2 r + e]: column 32 w + 8 q + 4 e + j
                        values = tuple(acc[t, j, 2 * r + e] for j in range(4))
                        column = 32 * (warp % (COLS // 32)) + 8 * quarter + 4 * e
                        dst = at(inbox, split, row, column)
                        push_float(dst, values, inbound, owner)
                if (warp == 0) & (quarter == 0):
                    for peer in cutlass.range_constexpr(SPLIT):
                        push_float(
                            at(stats, split, 0, row), m_run[2 * t + r], inbound, peer
                        )
                        push_float(
                            at(stats, split, 1, row), l_run[2 * t + r], inbound, peer
                        )
        wait(inbound, 0)  # every source's partial of these columns landed
        if cutlass.const_expr(self.bulk):
            cute.arch.sync_threads()  # this CTA's own columns (plain stores)
            # The copies into this CTA (out of the peers' ``staged``) completed:
            # the peers may exit once every CTA has passed here.
            cute.arch.cluster_arrive_relaxed()
        for t in cutlass.range_constexpr(T):
            self.merge(args, inbox, stats, tile, split, b, tid, t)
        if cutlass.const_expr(self.push is not None):  # noqa: SIM102 (compile time)
            # Every CTA of the cluster has read the q rows (it pushed after PV).
            if split == 0:
                for i in range(tid, CH * chunks16, THREADS):
                    q_row, q_chunk = i // chunks16, i % chunks16
                    sentinel = (Uint32(Q_SENT),) * 4
                    q_dst = at(query, b, tile * CH + q_row, 16 * q_chunk)
                    store_words(q_dst, sentinel)
        if cutlass.const_expr(self.bulk):
            cute.arch.cluster_wait()  # this CTA's ``staged`` is no longer read

    @cute.jit
    def stage(self, inbox, staged, acc, split, warp, group, quarter):
        """`bulk`: this CTA's partials of every column (FP32 bits) into its own
        `inbox` block (its columns) or the `staged` block of their CTA (the
        peers in CTA order without this one), visible to the bulk copies."""
        owner = warp // (COLS // 32)  # columns 32 w .. 32 w + 31
        block = owner
        if owner > split:
            block = owner - 1
        for t in cutlass.range_constexpr(self.tiles):
            for r in cutlass.range_constexpr(2):
                row = 16 * t + group + 8 * r
                for e in cutlass.range_constexpr(2):
                    words = tuple(float_bits(acc[t, j, 2 * r + e]) for j in range(4))
                    column = 32 * (warp % (COLS // 32)) + 8 * quarter + 4 * e
                    if owner == split:  # warp-uniform
                        store_words(at(inbox, split, row, column), words, shared=True)
                    else:
                        store_words(at(staged, block, row, column), words, shared=True)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

    @cute.jit
    def merge(self, args, inbox, stats, tile, split, b, tid, t: cutlass.Constexpr[int]):
        """Columns split * COLS .. + COLS of head tile `t`'s 16 heads over the
        cluster's partials (FP32, in `inbox` / `stats`): thread i takes head
        16 t + i // 32, 4 columns."""
        latent, lse_out = args[4:6]
        out_scale = args[7]
        neg_inf = Float32(-math.inf)
        head, lane = 16 * t + tid // 32, tid % 32
        assert COLS == 128, COLS  # 32 lanes x 4 columns
        col = split * COLS + 4 * lane
        maxima = [stats[c, 0, head] for c in range(SPLIT)]
        sums = [stats[c, 1, head] for c in range(SPLIT)]
        rows = [
            load_words(at(inbox, c, head, 4 * lane), count=4, shared=True)
            for c in range(SPLIT)
        ]
        rows = [tuple(bits_float(w) for w in words) for words in rows]
        top = maxima[0]
        for c in cutlass.range_constexpr(1, SPLIT):
            top = max(top, maxima[c])
        total = Float32(0)
        merged = zeros(4, Float32)
        for c in cutlass.range_constexpr(SPLIT):
            weight = Float32(0)
            if maxima[c] != neg_inf:
                weight = cute.math.exp2(maxima[c] - top, fastmath=True)
            total += weight * sums[c]
            for e in cutlass.range_constexpr(4):
                merged[e] += weight * rows[c][e]
        lse2 = neg_inf
        inverse = Float32(0)
        if total > Float32(0):
            lse2 = top - Float32(LOG2_P_SCALE) + cute.math.log2(total, fastmath=False)
            inverse = out_scale / total
        gh = tile * self.cta_heads + head
        words = (
            bf16x2(merged[0] * inverse, merged[1] * inverse),
            bf16x2(merged[2] * inverse, merged[3] * inverse),
        )
        if cutlass.const_expr(self.push is None):
            store_words(at(latent, b, gh, col), words)
            if (split == 0) & (lane == 0):
                lse_out[b, gh] = lse2 * Float32(LN2)
        else:
            peers, lpeers, parity, rank = args[8:12]
            H, D, cap = self.push
            owner = gh // H
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            # As DcpValue.sent_lse: a row with nothing to merge sends EMPTY_LSE.
            value = lse2
            bad = (value != value) | (value <= Float32(EMPTY_LSE))
            if bad | (value >= Float32(-EMPTY_LSE)):
                value = Float32(EMPTY_LSE)
            if value > Float32(EMPTY_LSE):
                sent = tuple(no_neg_zero(w) for w in words)
                store_relaxed(peers[owner] + (Int64(slot) * LATENT + col) * 2, sent)
            if (split == 0) & (lane == 0):
                store_relaxed(lpeers[owner] + Int64(slot) * 4, (float_bits(value),))


def _split_chunk(n, per, s, c):
    """(first key of chunk `c` of split `s`, its key count >= 0) over `n` keys
    split `per` per split (SparseDcpDecode's partition)."""
    first = cutlass.min(n, Int32(s * per))
    mine = cutlass.min(n - first, per)
    base = first + c * CHUNK
    count = cutlass.min(first + mine - base, Int32(CHUNK))
    return base, cutlass.max(count, Int32(0))


class SparseDcpDecodePx(SparseDcpDecode):
    """The pushing two-tile SparseDcpDecode (SPARSE_PX) with the cluster exchange
    moved before PV: CTA `split` of a row's cluster runs QK and the softmax of its
    split as SparseDcpDecode, then sends its split's E4M3 P and (max, sum, rescale)
    rows to the other CTAs (one bulk copy each per peer), and every CTA runs the PV
    of all SPLIT splits over its own COLS merged columns (the other splits' keys
    gathered as just those columns). Each split's accumulators of these columns
    then sit in the local inbox, and the merge is SparseDcpDecode's. Per head and
    column the operations and their order are SparseDcpDecode's (the same P bytes,
    V bytes, PV k-steps, rescales and merge), so it matches bit for bit. Rows of
    more than one chunk per split exchange per chunk (double-buffered)."""

    def __init__(self, batch, heads, padded, width, push):
        assert push is not None and heads % (2 * HEADS) == 0 and heads <= padded
        assert QK_WARPS * 2 == WARPS and COLS == 128
        self.batch, self.heads, self.padded, self.width = batch, heads, padded, width
        self.push, self.tiles, self.cta_heads = push, 2, 2 * HEADS
        self.grid = (heads // self.cta_heads * SPLIT, batch, 1)

    @cute.jit
    def __call__(self, *args):
        launch(
            self.kernel,
            args,
            self.grid,
            THREADS,
            cluster=(SPLIT, 1, 1),
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, args):
        query, kv, rows_in, valid_in = args[:4]
        scale_log2 = args[6]
        T: cutlass.Constexpr = self.tiles
        CH: cutlass.Constexpr = self.cta_heads
        smem = cutlass.utils.SmemAllocator()
        kvs = alloc(smem, Uint8, (CHUNK, PITCH), 128)  # this split's rows
        vcols = alloc(smem, Uint8, (SPLIT - 1, CHUNK, PX_VPITCH), 128)
        qs = alloc(smem, Uint8, (CH, PITCH), 128)
        # Per chunk parity: every split's P and (max, sum, rescale) rows; this
        # CTA's split written here, the others by their CTAs' bulk copies.
        ps = alloc(smem, Uint8, (2, SPLIT, CH, P_PITCH), 128)
        xst = alloc(smem, Float32, (2, SPLIT, 3, CH), 128)
        red = alloc(smem, Float32, (2, QK_WARPS, CH))
        stats = alloc(smem, Float32, (SPLIT, 2, CH))
        # The splits' partials of this CTA's columns (rows padded: the PV warps'
        # row-strided stores spread over the banks).
        inbox = alloc(smem, Float32, (SPLIT, CH, COLS + 4), 16)
        bar = smem.allocate(cutlass.Uint64, 8)
        inbound = smem.allocate(cutlass.Uint64, 8)
        tid, _, _ = cute.arch.thread_idx()
        x, b, _ = cute.arch.block_idx()
        tile, split = x // SPLIT, x % SPLIT
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        qt, kw = warp // QK_WARPS, warp % QK_WARPS
        psplit = warp // (WARPS // SPLIT)  # PV: split, 32-column block
        pblock = warp % (WARPS // SPLIT)
        x_bytes: cutlass.Constexpr = (SPLIT - 1) * (CH * P_PITCH + 3 * CH * 4)
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(inbound, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.mbarrier_arrive_and_expect_tx(inbound, x_bytes)
        cute.arch.sync_threads()
        # Every CTA's barriers are armed before any peer copies (waited below).
        cute.arch.cluster_arrive_relaxed()
        # Nothing above reads the inputs: the kernel before may launch this early.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        chunks16 = ROW // 16
        q_loads: cutlass.Constexpr = -(-CH * chunks16 // THREADS)
        q_words = []
        for i in cutlass.range_constexpr(q_loads):
            h_flat = min(tid + THREADS * i, Int32(CH * chunks16 - 1))
            h_row, h_chunk = h_flat // chunks16, h_flat % chunks16
            h_src = at(query, b, tile * CH + h_row, 16 * h_chunk)
            q_words.append(load_words(h_src, count=4, mutable=True))
        n = valid_in[b]
        n = max(min(n, Int32(self.width)), Int32(0))
        per = (n + SPLIT - 1) // SPLIT
        kv_base = kv.iterator.toint()
        neg_inf = Float32(-math.inf)
        m_run = zeros(2 * T, Float32)
        m_run.fill(neg_inf)
        l_run = zeros(2 * T, Float32)
        acc = zeros((T, 4, 4), Float32)  # split psplit, this CTA's columns
        a_row = lane % 8 + 8 * ((lane // 8) % 2)
        a_col = 16 * (lane // 16)
        b_row = 8 * kw + lane % 8
        b_col = 16 * (lane // 8)
        phase = Int32(0)
        # Every CTA of the cluster runs split 0's chunk count (the most).
        chunks = (per + CHUNK - 1) // CHUNK
        for c in cutlass.range(0, chunks, unroll=1):
            par = c % 2
            if c > 0:
                cute.arch.sync_threads()  # the previous chunk's readers are done
            obase, count = _split_chunk(n, per, split, c)
            span = (count + 31) // 32 * 32
            if tid == 0:
                total = count * ROW
                for j in cutlass.range_constexpr(1, SPLIT):
                    total += _split_chunk(n, per, (split + j) % SPLIT, c)[1] * COLS
                cute.arch.mbarrier_arrive_and_expect_tx(bar, total)
            if tid < CHUNK:
                if tid < count:
                    src = kv_base + Int64(rows_in[b, obase + tid]) * ROW
                    copy_g2s(kvs.iterator + tid * PITCH, src, ROW, bar)
                elif tid < span:  # V past the count: zero (P is zero there)
                    for k in cutlass.range_constexpr(LATENT // 16):
                        dst = kvs.iterator + tid * PITCH + 16 * k
                        store_words(dst, (Uint32(0),) * 4, shared=True)
            elif tid < SPLIT * CHUNK:
                # This CTA's columns of the other splits' keys (slot j - 1).
                j = tid // CHUNK
                vk = tid % CHUNK
                vbase, vcount = _split_chunk(n, per, (split + j) % SPLIT, c)
                vspan = (vcount + 31) // 32 * 32
                vdst = at(vcols, j - 1, vk, 0)
                if vk < vcount:
                    src = kv_base + Int64(rows_in[b, vbase + vk]) * ROW + COLS * split
                    copy_g2s(vdst, src, COLS, bar)
                elif vk < vspan:
                    for k in cutlass.range_constexpr(COLS // 16):
                        store_words(vdst + 16 * k, (Uint32(0),) * 4, shared=True)
            if c == 0:
                for i in cutlass.range_constexpr(q_loads):
                    flat = tid + THREADS * i
                    if flat < CH * chunks16:
                        q_row, q_chunk = flat // chunks16, flat % chunks16
                        dst = qs.iterator + q_row * PITCH + 16 * q_chunk
                        store_words(dst, q_words[i], shared=True)
            wait(bar, phase, scope="cta")
            phase = phase ^ 1
            cute.arch.sync_threads()  # the zeroed V rows
            # ---- QK (this split): warp w, tile qt, keys 8 kw .. 8 kw + 7 ----------
            # (n-tiles past the count only hold masked keys: their scores stay 0
            # and are masked below, as SparseDcpDecode masks the ones it computes)
            s = zeros(4, Float32)
            if 8 * kw < count:
                parts = zeros((QK_CHAINS, 4), Float32)
                for kp in cutlass.range_constexpr(ROW // 64):
                    k = ldmatrix(kvs.iterator + b_row * PITCH + 64 * kp + b_col)
                    for h in cutlass.range_constexpr(2):
                        ks = 2 * kp + h
                        qk_row = 16 * qt + a_row
                        a = ldmatrix(qs.iterator + qk_row * PITCH + 32 * ks + a_col)
                        b_frag = (k[2 * h], k[2 * h + 1])
                        mma_into(parts[ks % QK_CHAINS, None], a, b_frag, fp8=True)
                for e in cutlass.range_constexpr(4):
                    total = parts[0, e]
                    for chain in cutlass.range_constexpr(1, QK_CHAINS):
                        total += parts[chain, e]
                    s[e] = total
            top = zeros(2, Float32)
            top.fill(neg_inf)
            for e in cutlass.range_constexpr(4):
                key = 8 * kw + 2 * quarter + e % 2
                value = neg_inf
                if key < count:
                    value = s[e] * scale_log2
                s[e] = value
                top[e // 2] = max(top[e // 2], value)
            for r in cutlass.range_constexpr(2):
                for shift in cutlass.range_constexpr(2):
                    other = cute.arch.shuffle_sync_bfly(top[r], 1 << shift)
                    top[r] = max(top[r], other)
            if quarter == 0:
                red[0, kw, 16 * qt + group] = top[0]
                red[0, kw, 16 * qt + group + 8] = top[1]
            cute.arch.sync_threads()
            alpha = zeros(2 * T, Float32)
            alpha.fill(1.0)
            m_new = zeros(2 * T, Float32)
            for t in cutlass.range_constexpr(T):
                for r in cutlass.range_constexpr(2):
                    i = 2 * t + r
                    chunk_max = red[0, 0, 16 * t + group + 8 * r]
                    for w in cutlass.range_constexpr(1, QK_WARPS):
                        chunk_max = max(chunk_max, red[0, w, 16 * t + group + 8 * r])
                    m_new[i] = max(m_run[i], chunk_max)
                    if m_run[i] != neg_inf:
                        alpha[i] = cute.math.exp2(m_run[i] - m_new[i], fastmath=True)
            own = zeros(2, Float32)  # the running maxima of this warp's QK tile
            for t in cutlass.range_constexpr(T):
                if qt == t:  # warp-uniform
                    for r in cutlass.range_constexpr(2):
                        own[r] = m_new[2 * t + r]
            # ---- P (E4M3 at 448 of the running max) and its FP32 sums -------------
            sums = zeros(2, Float32)
            if 8 * kw < span:
                for r in cutlass.range_constexpr(2):
                    bias = own[r] - Float32(LOG2_P_SCALE)
                    p = []
                    for e in cutlass.range_constexpr(2):
                        value = s[2 * r + e]
                        prob = Float32(0)
                        if value != neg_inf:
                            prob = cute.math.exp2(value - bias, fastmath=True)
                        sums[r] += prob
                        p.append(prob)
                    key = 8 * kw + 2 * quarter
                    p_at = at(ps, par, split, 16 * qt + group + 8 * r, key)
                    store_fp8_pair_shared(p_at, p[0], p[1])
            for r in cutlass.range_constexpr(2):
                for shift in cutlass.range_constexpr(2):
                    sums[r] += cute.arch.shuffle_sync_bfly(sums[r], 1 << shift)
            if quarter == 0:
                red[1, kw, 16 * qt + group] = sums[0]
                red[1, kw, 16 * qt + group + 8] = sums[1]
            cute.arch.sync_threads()
            for t in cutlass.range_constexpr(T):
                for r in cutlass.range_constexpr(2):
                    i = 2 * t + r
                    total = red[1, 0, 16 * t + group + 8 * r]
                    for w in cutlass.range_constexpr(1, QK_WARPS):
                        total += red[1, w, 16 * t + group + 8 * r]
                    l_run[i] = l_run[i] * alpha[i] + total
                    m_run[i] = m_new[i]
            if (warp == 0) & (quarter == 0):
                for t in cutlass.range_constexpr(T):
                    for r in cutlass.range_constexpr(2):
                        row = 16 * t + group + 8 * r
                        xst[par, split, 0, row] = m_run[2 * t + r]
                        xst[par, split, 1, row] = l_run[2 * t + r]
                        xst[par, split, 2, row] = alpha[2 * t + r]
            # ---- exchange this split's P and rows with the other CTAs --------------
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            # Peers' barriers armed (c = 0) / done reading this parity (c > 1).
            cute.arch.cluster_wait()
            if tid < SPLIT - 1:
                peer = tid
                if tid >= split:
                    peer = tid + 1
                p_blk = at(ps, par, split, 0, 0)
                copy_s2cluster(p_blk, p_blk, inbound, peer, size=CH * P_PITCH)
                x_blk = at(xst, par, split, 0, 0)
                copy_s2cluster(x_blk, x_blk, inbound, peer, size=3 * CH * 4)
            wait(inbound, c % 2)  # the other splits' P and rows of this chunk
            if (tid == 0) & (c + 1 < chunks):
                cute.arch.mbarrier_arrive_and_expect_tx(inbound, x_bytes)
            if c == 0:
                # Every CTA of the cluster read the q rows (their P landed here):
                # re-arm this CTA's quarter of them.
                self.rearm(query, tile, split, b, tid)
            # ---- PV: split psplit, this CTA's columns 32 pblock .. + 31 -------------
            for t in cutlass.range_constexpr(T):
                for r in cutlass.range_constexpr(2):
                    factor = xst[par, psplit, 2, 16 * t + group + 8 * r]
                    for j in cutlass.range_constexpr(4):
                        for e in cutlass.range_constexpr(2):
                            acc[t, j, 2 * r + e] *= factor
            pspan = (_split_chunk(n, per, psplit, c)[1] + 31) // 32 * 32
            slot = (psplit - split + SPLIT) % SPLIT  # 0: this CTA's split
            vcol = 32 * pblock + 4 * group
            for ks in cutlass.range_constexpr(CHUNK // 32):
                if 32 * ks < pspan:
                    halves = []
                    for hf in cutlass.range_constexpr(2):
                        words = []
                        for i in cutlass.range_constexpr(4):
                            key = 32 * ks + 16 * hf + 4 * quarter + i
                            word = Uint32(0)
                            if slot == 0:  # warp-uniform: this split's full rows
                                mine_v = (
                                    kvs.iterator + key * PITCH + COLS * split + vcol
                                )
                                word = load_words(mine_v, count=1, shared=True)[0]
                            else:
                                theirs = at(vcols, slot - 1, key, vcol)
                                word = load_words(theirs, count=1, shared=True)[0]
                            words.append(word)
                        halves.append(transpose4(words))
                    for t in cutlass.range_constexpr(T):
                        pv_at = at(ps, par, psplit, 16 * t + a_row, 32 * ks + a_col)
                        a = ldmatrix(pv_at)
                        for j in cutlass.range_constexpr(4):
                            b_frag = (halves[0][j], halves[1][j])
                            mma_into(acc[t, j, None], a, b_frag, fp8=True)
            # Done reading the peers' P and rows of this chunk.
            cute.arch.cluster_arrive_relaxed()
        # ---- the splits' statistics (last chunk's) and partials, then the merge -----
        last = (chunks + 1) % 2  # (chunks - 1) mod 2
        if tid < SPLIT * 2 * CH:
            sc, sk, sh = tid // (2 * CH), (tid // CH) % 2, tid % CH
            value = neg_inf
            if sk == 1:
                value = Float32(0)
            if chunks > 0:
                value = xst[last, sc, sk, sh]
            stats[sc, sk, sh] = value
        for t in cutlass.range_constexpr(T):
            for r in cutlass.range_constexpr(2):
                row = 16 * t + group + 8 * r
                for e in cutlass.range_constexpr(2):
                    p_words = tuple(float_bits(acc[t, j, 2 * r + e]) for j in range(4))
                    column = 32 * pblock + 8 * quarter + 4 * e
                    store_words(at(inbox, psplit, row, column), p_words, shared=True)
        cute.arch.sync_threads()
        for t in cutlass.range_constexpr(T):
            self.merge(args, inbox, stats, tile, split, b, tid, t)
        if chunks == 0:  # no q row read is used
            self.rearm(query, tile, split, b, tid)
        # No peer copies into this CTA or reads its blocks any more.
        cute.arch.cluster_wait()

    @cute.jit
    def rearm(self, query, tile, split, b, tid):
        """Q_SENT into this CTA's quarter of its tile's q mailbox rows."""
        chunks16 = ROW // 16
        part: cutlass.Constexpr = self.cta_heads * chunks16 // SPLIT
        for i in range(tid, part, THREADS):
            flat = split * part + i
            q_row, q_chunk = flat // chunks16, flat % chunks16
            sentinel = (Uint32(Q_SENT),) * 4
            q_dst = at(query, b, tile * self.cta_heads + q_row, 16 * q_chunk)
            store_words(q_dst, sentinel)


# ---- Tensor-core sparse decode (VLLM_KIMI_K3_DCP8_TC_SPARSE)
# --------------------------------------------- TcsDecode: a cluster of TCS_GROUP CTAs
# per row. CTA g owns the row's keys of split g (consecutive ranges of ceil(n / G) keys,
# as SparseDcpDecode) and latent columns 128 g .. 128 g + 127. On tcgen05 it computes S
# = Q K^T of its keys (M = 128 padded heads, N = its key count rounded to 16, K = 576
# FP8) into tensor memory, then E4M3 P at 448 of its keys' maximum per head (as
# SparseDcpDecode's split) and the FP32 sums; it sends its P block, maxima and sums to
# the other CTAs (DSMEM), and accumulates P_j V_j of every CTA j's keys for its 128
# columns (V: the 128-byte latent block of each key row) in tensor memory accumulator j.
# The four accumulators then merge in FP32 with weights 2^(m_j - m) (m_j: CTA j's
# maxima, m: the row's) into one BF16 row, as the split merge. Rows with more keys than
# a pass (TCS_PASS) merge each pass into the running rows the same way.
TCS_GROUP = 4  # CTAs per row (a cluster)
TCS_COLS = LATENT // TCS_GROUP  # latent columns per CTA
TCS_OWN = 96  # keys per CTA per pass (its S columns)
TCS_PASS = TCS_GROUP * TCS_OWN
TCS_OTHERS = (TCS_GROUP - 1) * TCS_OWN // 32  # V gather iterations (4 keys per warp)
TCS_THREADS = 288  # warps 0-7: loads, softmax, epilogue; warp 8: MMA issue
TCS_REGION = TCS_OWN * 128  # one 128-byte K block of 96 rows (8-row atoms of 1 KB)
TCS_STAGED = 4 * TCS_REGION + TCS_OWN * 64  # 96 staged FP8 rows of 576 bytes
# Shared memory (bytes from a 1024-aligned base): q rows (96 heads; the MMA's padded
# rows 96-127 read the next block), own key rows, the other CTAs' V blocks, the P
# blocks of the four CTAs, the exchanged maxima and sums, the pass's row indices and
# the owners' mailbox addresses.
TCS_Q = 0
TCS_K = TCS_STAGED
TCS_V = 2 * TCS_STAGED
TCS_P = TCS_V + (TCS_GROUP - 1) * TCS_REGION
TCS_X = TCS_P + TCS_GROUP * TCS_REGION + 4096
TCS_RW = TCS_X + 2 * TCS_GROUP * 128 * 4
TCS_PP = TCS_RW + TCS_PASS * 4
TCS_T = TCS_PP + 2 * 16 * 8  # the two warp halves' partial maxima and sums
TCS_SMEM = TCS_T + 4 * 128 * 4
TCS_SW128 = 2  # tcgen05 shared-memory descriptor layouts
TCS_INTERLEAVE = 0


def tcs_instruction(n, b_mn_major):
    """tcgen05 kind::f8f6f4 instruction descriptor: E4M3 A and B, FP32 D, M = 128."""
    return 1 << 4 | int(b_mn_major) << 16 | (n >> 3) << 17 | (128 >> 4) << 24


@cute.jit
def tcs_desc(
    address,
    lbo: cutlass.Constexpr[int],
    sbo: cutlass.Constexpr[int],
    layout: cutlass.Constexpr[int],
):
    """tcgen05 shared-memory matrix descriptor of the operand at `address`."""
    fixed: cutlass.Constexpr = (
        (lbo >> 4) << 16 | (sbo >> 4) << 32 | 1 << 46 | layout << 61
    )
    return Uint64((Uint32(address) & 262143) >> 4) | Uint64(fixed)


@asm()
def tcs_mma_ss(d, a, b, instruction, accumulate):
    """One tcgen05 FP8 MMA, A and B from shared memory (descriptors)."""
    code = "{ .reg .pred p; setp.ne.b32 p,$4,0; "
    code += "tcgen05.mma.cta_group::1.kind::f8f6f4 [$0],$1,$2,$3,p; }"
    ops = (Uint32(d), Uint64(a), Uint64(b), Uint32(instruction), Int32(accumulate))
    return code, "r,l,l,r,r", ops


@asm()
def tcs_commit(barrier):
    code = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];"
    return code, "r", (barrier,)


@asm()
def tcs_cp_commit_wait():
    """Wait for this thread's cp.async copies."""
    return "cp.async.commit_group; cp.async.wait_group 0;", "~{memory}", ()


@asm(Float32, pure=True)
def tcs_scaled(s, scale, i, n):
    """Scaled score: s * scale (log2 units) for key i < n, else -inf."""
    code = "{ .reg .pred p; .reg .f32 x; mul.f32 x,$1,$2; setp.lt.s32 p,$3,$4; "
    code += "selp.f32 $0,x,0fFF800000,p; }"
    return code, "=f,f,f,r,r", (Float32(s), Float32(scale), Int32(i), Int32(n))


@asm()
def tcs_bar_sync(barrier_id, count: int):
    """Named barrier `barrier_id` (a register) of `count` threads."""
    return f"bar.sync $0,{count};", "r,~{memory}", (Int32(barrier_id),)


@cute.jit
def tcs_key_address(base, key, chunk):
    """Shared address of 16-byte chunk `chunk` (0..35) of staged row `key` (< 96) of
    a [96, 576] FP8 operand: four 128-byte K blocks, K-major 128B-swizzled (8-row
    atoms of 1 KB), then the 64-byte rope block K-major without swizzle (8-row core
    matrices of 8 x 16 B: chunk c at 128 c, 8-row groups 512 B apart)."""
    block = chunk // 8
    address = base + block * TCS_REGION + key // 8 * 1024 + key % 8 * 128
    address = address + ((chunk % 8) ^ (key % 8)) * 16
    if block == 4:
        rope = base + 4 * TCS_REGION + key // 8 * 512 + (chunk - 32) * 128
        address = rope + key % 8 * 16
    return address


@cute.jit
def tcs_first(count, cta):
    """First key of CTA `cta` in a pass of `count` keys: the CTAs take consecutive
    ranges of ceil(count / G) keys (SparseDcpDecode's split of a row's keys)."""
    return min(count, cta * ((count + TCS_GROUP - 1) // TCS_GROUP))


@cute.jit
def tcs_owned(count, cta):
    """Keys of a pass of `count` keys owned by CTA `cta` (see tcs_first)."""
    per = (count + TCS_GROUP - 1) // TCS_GROUP
    return max(min(count - tcs_first(count, cta), per), Int32(0))


class TcsDecode:
    """Pushing sparse decode of `batch` rows on tcgen05 (see TCS_*): grid (G, rows),
    one cluster of G CTAs per row. `heads` gathered heads (96: the q mailbox holds
    `padded` rows per token), the merge's row width `width`. `push` (owner heads H,
    DCP, mailbox rows): the rows and base-2 LSEs into the owners' DcpValue "bf16"
    mailbox and the q mailbox rows read re-armed; else the latent [B, padded, 512]
    and natural-log LSE [B, padded]."""

    def __init__(
        self, batch, heads, padded, width, push=None, groups=TCS_GROUP, carry=False
    ):
        assert heads == TCS_OWN and heads <= padded and groups == TCS_GROUP
        assert push is None or push[1] <= 16
        self.batch, self.heads, self.padded, self.width = batch, heads, padded, width
        self.push, self.groups = push, groups
        # `carry`: from the second pass on, a CTA's P anchor (its exchanged maximum)
        # is the larger of its own maximum and the running maximum of the passes
        # before (the same value in every CTA of the row), as a running-max split
        # quantizes P; the first pass and single-pass rows are unchanged.
        self.carry = carry
        self.grid = (groups, batch, 1)

    @cute.jit
    def __call__(self, *args):
        launch(
            self.kernel,
            args,
            self.grid,
            TCS_THREADS,
            cluster=(self.groups, 1, 1),
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, args):
        query, valid_in = args[0], args[3]
        smem = cutlass.utils.SmemAllocator()
        buf = alloc(smem, Uint8, TCS_SMEM, 1024)
        # k_full (loads), s_full (QK), p_local (this CTA's P written), p_src[j] (CTA
        # j's P block, maximum and sum in; j != g), o_full (PV)
        bars = alloc(smem, cutlass.Uint64, 8, 8)
        tslot = alloc(smem, Uint32, 1)
        tid, _, _ = cute.arch.thread_idx()
        g, b, _ = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        bar = bars.iterator
        ba = Int32(shared_address(buf.iterator))
        if tid == 0:
            cute.arch.mbarrier_init(bar, 256)
            cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init(bar + 2, 256)
            for j in cutlass.range_constexpr(TCS_GROUP):
                cute.arch.mbarrier_init(bar + 3 + j, 1)
            cute.arch.mbarrier_init(bar + 7, 1)
            cute.arch.mbarrier_init_fence()
            p_bytes: cutlass.Constexpr = self.heads * (128 + 8)
            for j in cutlass.range_constexpr(TCS_GROUP):
                if j != g:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar + 3 + j, p_bytes)
        if warp == 8:
            cute.arch.alloc_tmem(512, tslot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(self.push is not None):
            # The owners' mailbox addresses (static: read before the PDL wait).
            peers, lpeers = args[8], args[9]
            if tid < self.push[1]:
                pp = ba + TCS_PP + 8 * tid
                for k, table in ((0, peers), (128, lpeers)):
                    word = table[tid]
                    halves = (Uint32(word & 0xFFFFFFFF), Uint32(word >> 32))
                    store_shared_words(pp + k, halves)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = Int32(tslot[0])
        # The barriers above are armed before any peer writes this CTA (waited
        # before the first DSMEM store).
        cute.arch.cluster_arrive_relaxed()
        # Nothing above reads the inputs: the kernel before may launch this early.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if warp < 8:  # the flat rows of the first pass, alongside the count
            self.load_rows(ba, args[2], b, 0, tid)
        n = valid_in[b]
        n = max(min(n, Int32(self.width)), Int32(0))
        passes = (n + TCS_PASS - 1) // TCS_PASS
        if warp == 8:
            self.issue(tmem, ba, bar, n, passes, g, lane == 0)
            cute.arch.sync_warp()
            cute.arch.cluster_wait()
        else:
            self.work(args, tmem, ba, bar, n, passes, g, b, tid)
        release_tmem(tmem, 512, warp, 8)
        # No peer copies into this CTA or out of its blocks any more (warps 0-7
        # arrived after the last exchange; warp 8 and empty rows arrive here).
        if (warp == 8) | (passes == 0):
            cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.push is not None):  # noqa: SIM102
            if (passes == 0) & (warp < 8):  # no q row was read
                self.rearm(query, g, tid)

    @cute.jit
    def rearm(self, query, g, tid):
        """Q_SENT into this CTA's share of the row's q mailbox rows."""
        _, b, _ = cute.arch.block_idx()
        part: cutlass.Constexpr = self.heads * (ROW // 16) // self.groups
        for i in range(tid, part, 256):
            flat = g * part + i
            q_row, q_chunk = flat // (ROW // 16), flat % (ROW // 16)
            sentinel = (Uint32(Q_SENT),) * 4
            store_words(at(query, b, q_row, 16 * q_chunk), sentinel)

    @cute.jit
    def issue(self, tmem, ba, bar, n, passes, g, leader):
        """Warp 8 (the `leader` lane issues): per pass QK of this CTA's keys into S
        (columns 0 ..), then, once every CTA's P is here, PV of CTA j's keys into
        accumulator j (columns 128 j .. 128 j + 127; S is consumed by then)."""
        base_qk: cutlass.Constexpr = tcs_instruction(16, False) & ~(63 << 17)
        pv: cutlass.Constexpr = tcs_instruction(TCS_COLS, True)
        for p in cutlass.range(passes, unroll=1):
            cnt = min(n - p * TCS_PASS, Int32(TCS_PASS))
            own = tcs_owned(cnt, g)
            wait(bar, p % 2, scope="cta")
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem()
            if own > 0:
                n16 = (own + 15) // 16 * 16
                qk = Uint32(base_qk) | (Uint32(n16) >> 3) << 17
                for ks in cutlass.range_constexpr(18):
                    if cutlass.const_expr(ks < 16):
                        at_q = ba + TCS_Q + ks // 4 * TCS_REGION + ks % 4 * 32
                        a = tcs_desc(at_q, 16, 1024, TCS_SW128)
                        at_k = ba + TCS_K + ks // 4 * TCS_REGION + ks % 4 * 32
                        bd = tcs_desc(at_k, 16, 1024, TCS_SW128)
                    else:
                        rope = 4 * TCS_REGION + (ks - 16) * 256
                        a = tcs_desc(ba + TCS_Q + rope, 128, 512, TCS_INTERLEAVE)
                        bd = tcs_desc(ba + TCS_K + rope, 128, 512, TCS_INTERLEAVE)
                    if leader:
                        tcs_mma_ss(tmem, a, bd, qk, Int32(ks > 0))
            if leader:
                tcs_commit(bar + 1)
            # PV of this CTA's keys once its P is written, then of each other CTA's
            # keys once its P block is here.
            for d in cutlass.range_constexpr(TCS_GROUP):
                j = (g + d) % TCS_GROUP
                if cutlass.const_expr(d == 0):
                    wait(bar + 2, p % 2, scope="cta")
                    v_at = ba + TCS_K + g * TCS_REGION
                else:
                    wait(bar + 3 + j, p % 2)
                    v_at = ba + TCS_V + (j - Int32(j > g)) * TCS_REGION
                fence_tmem()
                steps = (tcs_owned(cnt, j) + 31) // 32
                for kk in cutlass.range(steps, unroll=1):
                    p_at = ba + TCS_P + j * TCS_REGION + kk * 32
                    a = tcs_desc(p_at, 16, 1024, TCS_SW128)
                    bd = tcs_desc(v_at + kk * 4096, TCS_REGION, 1024, TCS_SW128)
                    if leader:
                        tcs_mma_ss(
                            tmem + TCS_COLS * j, a, bd, Uint32(pv), Int32(kk > 0)
                        )
            if leader:
                tcs_commit(bar + 7)

    @cute.jit
    def work(self, args, tmem, ba, bar, n, passes, g, b, tid):
        """Warps 0-7: per pass the loads, the softmax and exchange, then the fold of
        the four accumulators into this thread's 64 output columns (FP32 registers);
        then the epilogue."""
        query, kv, rows_in = args[0], args[1], args[2]
        warp, lane = tid // 32, tid % 32
        kv_base = Int64(kv.iterator.toint())
        # Head 32 (w % 4) + lane, columns 64 (w // 4) .. of this CTA's block.
        out = zeros(TCS_COLS // 2, Float32)
        m_run, l_run = Float32(-math.inf), Float32(0)
        lane_tmem = tmem + ((32 * (warp % 4)) << 16)
        if passes == 0:
            cute.arch.cluster_wait()
        for p in cutlass.range(passes, unroll=1):
            base = p * TCS_PASS
            cnt = min(n - base, Int32(TCS_PASS))
            if p > 0:
                self.load_rows(ba, rows_in, b, base, tid)
            if p == 0:
                # The q rows (warp w: heads w, w + 8, ..; lane l: 16-byte chunk l, and
                # 32 + l (the rope) on lanes 0-3).
                for j in cutlass.range_constexpr(self.heads // 8):
                    h = warp + 8 * j
                    src = at(query, b, h, 0).toint()
                    copy_g2s(tcs_key_address(ba + TCS_Q, h, lane), src + lane * 16)
                    if lane < 4:
                        dst = tcs_key_address(ba + TCS_Q, h, 32 + lane)
                        copy_g2s(dst, src + 512 + lane * 16)
            named_barrier(1, 256)  # the pass's row indices
            self.gather(ba, kv_base, cnt, g, warp, lane)
            tcs_cp_commit_wait()
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem(before=True)  # the last fold's TMEM loads precede this QK
            cute.arch.mbarrier_arrive(bar)
            if p == 0:
                cute.arch.cluster_wait()  # every peer's barriers are armed
            top, total = self.softmax(
                args, ba, bar, lane_tmem, p, passes, cnt, g, m_run
            )
            wait(bar + 7, p % 2, scope="cta")  # PV of this pass done
            cute.arch.sync_warp()
            fence_tmem()
            m_run, l_run = self.fold(
                ba, lane_tmem, out, cnt, g, top, total, m_run, l_run
            )
        self.epilogue(args, ba, out, m_run, l_run, passes, g, b, tid)

    @cute.jit
    def load_rows(self, ba, rows_in, b, base, tid):
        """The flat rows of a pass's TCS_PASS keys into shared memory (no dependence
        on the count; indices past the merge's width read its last entry)."""
        last = Int32(self.width - 1)
        for i in cutlass.range_constexpr(-(-TCS_PASS // 256)):
            k = tid + 256 * i
            if k < TCS_PASS:
                row = rows_in[b, min(base + k, last)]
                store_shared_words(ba + TCS_RW + 4 * k, (Uint32(row),))

    @cute.jit
    def gather(self, ba, kv_base, cnt, g, warp, lane):
        """cp.async of this pass: the CTA's own key rows (warp w: keys w, w + 8, ..),
        then the other CTAs' keys' 128-byte V blocks of this CTA's columns (4 keys
        per warp instruction); V rows past a count up to its 32-key step zeroed."""
        own = tcs_owned(cnt, g)
        first = tcs_first(cnt, g)
        span = (own + 31) // 32 * 32
        for j in cutlass.range_constexpr(TCS_OWN // 8):
            i = warp + 8 * j
            if i < own:
                key_at = ba + TCS_RW + 4 * (first + i)
                row = load_words(key_at, count=1, shared=True)[0]
                src = kv_base + Int64(row) * ROW
                copy_g2s(tcs_key_address(ba + TCS_K, i, lane), src + lane * 16)
                if lane < 4:
                    dst = tcs_key_address(ba + TCS_K, i, 32 + lane)
                    copy_g2s(dst, src + 512 + lane * 16)
            elif (i < span) & (lane // 8 == g):
                dst = tcs_key_address(ba + TCS_K, i, lane)
                store_shared_words(dst, (Uint32(0),) * 4)
        chunk = lane % 8
        for it in cutlass.range_constexpr(TCS_OTHERS):
            t = 4 * (warp + 8 * it) + lane // 8
            q = t // TCS_OWN
            i = t % TCS_OWN
            j = q + Int32(q >= g)
            nj = tcs_owned(cnt, j)
            dst = ba + TCS_V + q * TCS_REGION + i // 8 * 1024 + i % 8 * 128
            dst = dst + (chunk ^ (i % 8)) * 16
            if i < nj:
                key = tcs_first(cnt, j) + i
                row = load_words(ba + TCS_RW + 4 * key, count=1, shared=True)[0]
                src = kv_base + Int64(row) * ROW + g * TCS_COLS + chunk * 16
                copy_g2s(dst, src)
            elif i < (nj + 31) // 32 * 32:
                store_shared_words(dst, (Uint32(0),) * 4)

    @cute.jit
    def softmax(self, args, ba, bar, lane_tmem, p, passes, cnt, g, m_prev):
        """Warps 0-7, thread = head h = 32 (w % 4) + lane; warp w takes this CTA's S
        chunks w // 4 and w // 4 + 2 (32 keys each): the CTA's maximum, P of its keys
        (E4M3 at 448 of that maximum) into its P block and the FP32 sum; the P block
        (bulk copies, 32 heads per warp), maximum and sum (DSMEM stores) go to the
        other CTAs. -> (maximum, sum) of this CTA."""
        scale_log2 = args[6]
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        quarter, half = warp % 4, warp // 4
        h = 32 * quarter + lane
        live = h < self.heads
        own = tcs_owned(cnt, g)
        span = (own + 31) // 32 * 32
        wait(bar + 1, p % 2, scope="cta")
        cute.arch.sync_warp()
        fence_tmem()
        top = Float32(-math.inf)
        for i in cutlass.range_constexpr(2):
            c = half + 2 * i
            if 32 * c < own:
                s = load_tmem(lane_tmem + 32 * c, 32)
                for e in cutlass.range_constexpr(32):
                    top = max(top, tcs_scaled(s[e], scale_log2, 32 * c + e, own))
        tmp = ba + TCS_T + 4 * h
        store_shared_words(tmp + 512 * half, (float_bits(top),))
        named_barrier(1, 256)
        top = max(
            bits_float(load_words(tmp, count=1, shared=True)[0]),
            bits_float(load_words(tmp + 512, count=1, shared=True)[0]),
        )
        if cutlass.const_expr(self.carry):
            top = max(top, m_prev)  # -inf before the first fold
        bias = top - Float32(LOG2_P_SCALE)
        total = Float32(0)
        row_at = ba + TCS_P + g * TCS_REGION + h // 8 * 1024 + h % 8 * 128
        for i in cutlass.range_constexpr(2):
            c = half + 2 * i
            if 32 * c < span:
                s = load_tmem(lane_tmem + 32 * c, 32)
                for hf in cutlass.range_constexpr(2):
                    words = []
                    for q in cutlass.range_constexpr(4):
                        probs = []
                        for e in cutlass.range_constexpr(4):
                            k = 16 * hf + 4 * q + e
                            x = tcs_scaled(s[k], scale_log2, 32 * c + k, own)
                            prob = cute.math.exp2(x - bias, fastmath=True)
                            total += prob
                            probs.append(prob)
                        words.append(pack_fp8(tuple(probs)))
                    chunk = 2 * c + hf
                    if live:
                        store_shared_words(
                            row_at + (chunk ^ (h % 8)) * 16, tuple(words)
                        )
        store_shared_words(tmp + 1024 + 512 * half, (float_bits(total),))
        cute.arch.fence_proxy("async.shared", space="cta")
        tcs_bar_sync(3 + quarter, 64)  # warps w and w + 4: this quarter's P rows
        total = bits_float(load_words(tmp + 1024, count=1, shared=True)[0])
        total += bits_float(load_words(tmp + 1536, count=1, shared=True)[0])
        inbound = bar + 3 + g  # each receiver's barrier of this CTA's data
        if live & (half == 0):
            mine = ba + TCS_X + 4 * (g * 128 + h)
            for d in cutlass.range_constexpr(1, TCS_GROUP):
                peer = (g + d) % TCS_GROUP
                push_float(mine, top, inbound, peer)
                push_float(mine + TCS_GROUP * 512, total, inbound, peer)
            if lane == 0:  # these 32 heads' P rows (4 KB) to the other CTAs
                block = ba + TCS_P + g * TCS_REGION + 4096 * quarter
                for d in cutlass.range_constexpr(1, TCS_GROUP):
                    copy_s2cluster(
                        block, block, inbound, (g + d) % TCS_GROUP, size=4096
                    )
        cute.arch.mbarrier_arrive(bar + 2)  # this CTA's P is written (fenced above)
        for d in cutlass.range_constexpr(1, TCS_GROUP):
            wait(bar + 3 + (g + d) % TCS_GROUP, p % 2)
        if cutlass.const_expr(self.push is not None):  # noqa: SIM102
            if p == 0:
                # Every CTA of the row computed its QK: their q rows are read.
                self.rearm(args[0], g, tid)
        if p + 1 == passes:
            # Every peer's data is here: it may exit once the others have theirs.
            cute.arch.cluster_arrive_relaxed()
        if (tid == 0) & (p + 1 < passes):  # the next pass's exchange
            p_bytes: cutlass.Constexpr = self.heads * (128 + 8)
            for d in cutlass.range_constexpr(1, TCS_GROUP):
                source = bar + 3 + (g + d) % TCS_GROUP
                cute.arch.mbarrier_arrive_and_expect_tx(source, p_bytes)
        return top, total

    @cute.jit
    def fold(self, ba, lane_tmem, out, cnt, g, top, total, m_run, l_run):
        """Warps 0-7 (thread: head h, its 64 columns): this pass's accumulators j,
        weighted by 2^(m_j - m) (m_j: CTA j's maximum, m: the running maximum), into
        `out` (rescaled by 2^(m_old - m)); -> the running maximum and sum."""
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        quarter, half = warp % 4, warp // 4
        h = 32 * quarter + lane
        maxima, sums = [], []
        for j in cutlass.range_constexpr(TCS_GROUP):
            mj, lj = top, total
            if j != g:
                at_j = ba + TCS_X + 4 * (j * 128 + h)
                mj = bits_float(load_words(at_j, count=1, shared=True)[0])
                lj = bits_float(
                    load_words(at_j + TCS_GROUP * 512, count=1, shared=True)[0]
                )
            maxima.append(mj)
            sums.append(lj)
        m_new = m_run
        for j in cutlass.range_constexpr(TCS_GROUP):
            m_new = max(m_new, maxima[j])
        alpha = cute.math.exp2(m_run - m_new, fastmath=True)
        weights = [cute.math.exp2(maxima[j] - m_new, fastmath=True) for j in range(4)]
        l_new = l_run * alpha
        for j in cutlass.range_constexpr(TCS_GROUP):
            if tcs_owned(cnt, j) > 0:
                l_new += weights[j] * sums[j]
        if 32 * quarter < self.heads:
            if (
                tcs_owned(cnt, TCS_GROUP - 1) > 0
            ):  # every CTA owns keys (CTA G - 1 fewest)
                for e in cutlass.range_constexpr(TCS_COLS // 2):
                    out[e] = out[e] * alpha
                for j in cutlass.range_constexpr(TCS_GROUP):
                    for c0 in cutlass.range_constexpr(0, TCS_COLS // 2, 32):
                        o_at = lane_tmem + TCS_COLS * j + TCS_COLS // 2 * half + c0
                        o = load_tmem(o_at, 32)
                        for e in cutlass.range_constexpr(32):
                            out[c0 + e] = out[c0 + e] + weights[j] * o[e]
            else:  # the accumulators of CTAs with keys only
                for e in cutlass.range_constexpr(TCS_COLS // 2):
                    out[e] = out[e] * alpha
                for j in cutlass.range_constexpr(TCS_GROUP):
                    if tcs_owned(cnt, j) > 0:
                        for c0 in cutlass.range_constexpr(0, TCS_COLS // 2, 32):
                            o_at = lane_tmem + TCS_COLS * j + TCS_COLS // 2 * half + c0
                            o = load_tmem(o_at, 32)
                            for e in cutlass.range_constexpr(32):
                                out[c0 + e] = out[c0 + e] + weights[j] * o[e]
        return m_new, l_new

    @cute.jit
    def epilogue(self, args, ba, out, m_run, l_run, passes, g, b, tid):
        """Warps 0-7: the folded rows times out_scale / l as BF16 into shared memory
        (over the own key rows), then the rows and LSEs out (pushed to their owners,
        or the latent / LSE)."""
        latent, lse_out, out_scale = args[4], args[5], args[7]
        warp, lane = tid // 32, tid % 32
        h = 32 * (warp % 4) + lane
        half = warp // 4
        pitch: cutlass.Constexpr = 2 * TCS_COLS + 16
        stage = ba + TCS_K
        if (passes > 0) & (32 * (warp % 4) < self.heads):
            inverse = out_scale / l_run
            for q in cutlass.range_constexpr(TCS_COLS // 16):
                words = tuple(
                    bf16x2(
                        out[8 * q + 2 * e] * inverse, out[8 * q + 2 * e + 1] * inverse
                    )
                    for e in range(4)
                )
                if cutlass.const_expr(self.push is not None):
                    words = tuple(no_neg_zero(w) for w in words)
                at_byte = h * pitch + 2 * (TCS_COLS // 2 * half + 8 * q)
                store_shared_words(stage + at_byte, words)
        named_barrier(1, 256)
        lse2 = m_run - Float32(LOG2_P_SCALE) + cute.math.log2(l_run, fastmath=False)
        chunks: cutlass.Constexpr = TCS_COLS // 8  # 16-byte chunks per head row
        if cutlass.const_expr(self.push is None):
            for i in range(tid, self.heads * chunks, 256):
                hh, c = i // chunks, i % chunks
                words = (Uint32(0),) * 4
                if passes > 0:
                    words = load_words(
                        stage + hh * pitch + 16 * c, count=4, shared=True
                    )
                store_words(at(latent, b, hh, g * TCS_COLS + 8 * c), words)
            if (g == 0) & (tid < self.heads):
                value = Float32(-math.inf)
                if passes > 0:
                    value = lse2 * LN2
                lse_out[b, tid] = value
        else:
            parity, rank = args[10], args[11]
            H, D, cap = self.push
            if passes > 0:
                for i in cutlass.range_constexpr(self.heads * chunks // 256):
                    hh = (tid + 256 * i) // chunks
                    c = (tid + 256 * i) % chunks
                    words = load_words(
                        stage + hh * pitch + 16 * c, count=4, shared=True
                    )
                    slot = ((parity * D + rank) * cap + b) * H + hh % H
                    pw = load_words(ba + TCS_PP + 8 * (hh // H), count=2, shared=True)
                    owner = Int64(pw[0]) | (Int64(pw[1]) << 32)
                    address = owner + (Int64(slot) * LATENT + g * TCS_COLS + 8 * c) * 2
                    store_relaxed(address, words)
            if (g == 0) & (tid < self.heads):
                value = Float32(EMPTY_LSE)
                if passes > 0:
                    value = lse2
                    bad = (value != value) | (value <= Float32(EMPTY_LSE))
                    if bad | (value >= Float32(-EMPTY_LSE)):
                        value = Float32(EMPTY_LSE)
                slot = ((parity * D + rank) * cap + b) * H + tid % H
                lw = load_words(
                    ba + TCS_PP + 128 + 8 * (tid // H), count=2, shared=True
                )
                laddr = Int64(lw[0]) | (Int64(lw[1]) << 32)
                store_relaxed(laddr + Int64(slot) * 4, (float_bits(value),))


# Pair decode (TcsDecodePair): two CTAs per row, so a row's cluster is half as wide and
# up to twice as many rows run in one wave. Each CTA takes 256 latent columns and up to
# PAIR_OWN keys per pass; the two CTAs exchange their maxima before P, both quantize P
# at 448 of the same anchor (the pair's maximum, never below the passes before), so the
# PV of both CTAs' keys accumulates into one TMEM accumulator (columns 0-255; S in
# columns 256-383). When the anchor rises between passes the accumulator is rescaled
# in TMEM. The maxima exchange also orders the P blocks: a CTA's next P block reaches
# its peer only after the peer's PV of the pass before.
PAIR_GROUP = 2
PAIR_COLS = LATENT // PAIR_GROUP
PAIR_OWN = 160  # a row's keys on one DCP rank (about top-k / DCP) mostly in one pass
PAIR_PASS = PAIR_GROUP * PAIR_OWN
PAIR_KREG = PAIR_OWN * 128  # one 128-byte K block of 160 rows (8-row atoms of 1 KB)
PAIR_KSTAGED = 4 * PAIR_KREG + PAIR_OWN * 64
PAIR_TAIL = 128 * 32  # P keys 128-159 of 128 rows: K-major, no swizzle (8 x 16 B cores)
PAIR_Q = 0
PAIR_K = TCS_STAGED
PAIR_V = PAIR_K + PAIR_KSTAGED
PAIR_P = PAIR_V + 2 * PAIR_KREG  # P keys 0-127 (128B-swizzled), then the two tails
PAIR_PT = PAIR_P + PAIR_GROUP * TCS_REGION
PAIR_X = PAIR_PT + PAIR_GROUP * PAIR_TAIL
PAIR_RW = PAIR_X + 2 * PAIR_GROUP * 128 * 4
PAIR_PP = PAIR_RW + PAIR_PASS * 4
PAIR_T = PAIR_PP + 2 * 16 * 8
PAIR_SMEM = PAIR_T + 4 * 128 * 4
assert PAIR_SMEM + 1024 + 128 <= 232448, PAIR_SMEM
PAIR_S = 2 * 128  # TMEM column of S
PAIR_P_BYTES = TCS_OWN * (128 + 32 + 4)  # a P block, its tail and the sums (96 heads)


@asm()
def tcs_store_tmem(address, values):
    """Store len(values) FP32 TMEM columns of this lane and wait for the stores."""
    n = len(values)
    registers = ",".join(f"${i + 1}" for i in range(n))
    code = f"tcgen05.st.sync.aligned.32x32b.x{n}.b32 [$0],{{{registers}}}; "
    code += "tcgen05.wait::st.sync.aligned;"
    constraints = ",".join(["r"] + ["f"] * n + ["~{memory}"])
    return code, constraints, (Uint32(address), *[Float32(v) for v in values])


@cute.jit
def pair_key_address(base, key, chunk, kreg: cutlass.Constexpr[int]):
    """tcs_key_address of a staged [rows, 576] FP8 operand (K blocks `kreg` bytes)."""
    block = chunk // 8
    address = base + block * kreg + key // 8 * 1024 + key % 8 * 128
    address = address + ((chunk % 8) ^ (key % 8)) * 16
    if block == 4:
        rope = base + 4 * kreg + key // 8 * 512 + (chunk - 32) * 128
        address = rope + key % 8 * 16
    return address


@cute.jit
def pair_first(count, cta):
    """First key of CTA `cta` in a pass of `count` keys (consecutive halves)."""
    return min(count, cta * ((count + PAIR_GROUP - 1) // PAIR_GROUP))


@cute.jit
def pair_owned(count, cta):
    per = (count + PAIR_GROUP - 1) // PAIR_GROUP
    return max(min(count - pair_first(count, cta), per), Int32(0))


class TcsDecodePair:
    """TcsDecode with two CTAs per row (see PAIR_*): grid (2, rows), one cluster of two
    CTAs per row; same arguments and outputs as TcsDecode."""

    def __init__(self, batch, heads, padded, width, push=None, own=PAIR_OWN):
        assert heads == TCS_OWN and heads <= padded and own in (128, PAIR_OWN)
        assert push is None or push[1] <= 16
        self.batch, self.heads, self.padded, self.width = batch, heads, padded, width
        self.push, self.groups = push, PAIR_GROUP
        self.grid = (PAIR_GROUP, batch, 1)
        # `own` keys per CTA per pass (PAIR_*'s layout for that count).
        self.own, self.npass, self.kreg = own, PAIR_GROUP * own, own * 128
        self.kstaged = 4 * self.kreg + own * 64
        self.off_k = TCS_STAGED
        self.off_v = self.off_k + self.kstaged
        self.off_p = self.off_v + 2 * self.kreg
        self.off_pt = self.off_p + PAIR_GROUP * TCS_REGION
        self.off_x = self.off_pt + PAIR_GROUP * PAIR_TAIL
        self.off_rw = self.off_x + 2 * PAIR_GROUP * 128 * 4
        self.off_pp = self.off_rw + self.npass * 4
        self.off_t = self.off_pp + 2 * 16 * 8
        self.smem = self.off_t + 4 * 128 * 4

    @cute.jit
    def __call__(self, *args):
        launch(
            self.kernel,
            args,
            self.grid,
            TCS_THREADS,
            cluster=(PAIR_GROUP, 1, 1),
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, args):
        query, valid_in = args[0], args[3]
        smem = cutlass.utils.SmemAllocator()
        buf = alloc(smem, Uint8, self.smem, 1024)
        # k_full (loads), s_full (QK), p_local (this CTA's P written), p_src (the peer's
        # P block and sum in), m_src (the peer's maximum in), o_full (PV)
        bars = alloc(smem, cutlass.Uint64, 8, 8)
        tslot = alloc(smem, Uint32, 1)
        tid, _, _ = cute.arch.thread_idx()
        g, b, _ = cute.arch.block_idx()
        warp = tid // 32
        bar = bars.iterator
        ba = Int32(shared_address(buf.iterator))
        if tid == 0:
            cute.arch.mbarrier_init(bar, 256)
            cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init(bar + 2, 256)
            cute.arch.mbarrier_init(bar + 3, 1)
            cute.arch.mbarrier_init(bar + 4, 1)
            cute.arch.mbarrier_init(bar + 5, 1)
            cute.arch.mbarrier_init_fence()
            p_bytes: cutlass.Constexpr = self.heads * (128 + 32 + 4)
            m_bytes: cutlass.Constexpr = self.heads * 4
            cute.arch.mbarrier_arrive_and_expect_tx(bar + 3, p_bytes)
            cute.arch.mbarrier_arrive_and_expect_tx(bar + 4, m_bytes)
        if warp == 8:
            cute.arch.alloc_tmem(512, tslot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        if cutlass.const_expr(self.push is not None):
            peers, lpeers = args[8], args[9]
            if tid < self.push[1]:
                pp = ba + self.off_pp + 8 * tid
                for k, table in ((0, peers), (128, lpeers)):
                    word = table[tid]
                    halves = (Uint32(word & 0xFFFFFFFF), Uint32(word >> 32))
                    store_shared_words(pp + k, halves)
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = Int32(tslot[0])
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if warp < 8:
            self.load_rows(ba, args[2], b, 0, tid)
        n = valid_in[b]
        n = max(min(n, Int32(self.width)), Int32(0))
        passes = (n + self.npass - 1) // self.npass
        if warp == 8:
            self.issue(tmem, ba, bar, n, passes, g, tid % 32 == 0)
            cute.arch.sync_warp()
            cute.arch.cluster_wait()
        else:
            self.work(args, tmem, ba, bar, n, passes, g, b, tid)
        release_tmem(tmem, 512, warp, 8)
        if (warp == 8) | (passes == 0):
            cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        if cutlass.const_expr(self.push is not None):  # noqa: SIM102
            if (passes == 0) & (warp < 8):
                self.rearm(query, g, tid)

    @cute.jit
    def rearm(self, query, g, tid):
        """Q_SENT into this CTA's share of the row's q mailbox rows."""
        _, b, _ = cute.arch.block_idx()
        part: cutlass.Constexpr = self.heads * (ROW // 16) // self.groups
        for i in range(tid, part, 256):
            flat = g * part + i
            q_row, q_chunk = flat // (ROW // 16), flat % (ROW // 16)
            sentinel = (Uint32(Q_SENT),) * 4
            store_words(at(query, b, q_row, 16 * q_chunk), sentinel)

    @cute.jit
    def issue(self, tmem, ba, bar, n, passes, g, leader):
        """Warp 8: per pass QK of this CTA's keys into S, then PV of this CTA's keys
        and of the peer's keys into the accumulator (two N = 128 halves per k-step)."""
        base_qk: cutlass.Constexpr = tcs_instruction(16, False) & ~(63 << 17)
        pv: cutlass.Constexpr = tcs_instruction(128, True)
        for p in cutlass.range(passes, unroll=1):
            cnt = min(n - p * self.npass, Int32(self.npass))
            own = pair_owned(cnt, g)
            wait(bar, p % 2, scope="cta")
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem()
            if own > 0:
                n16 = (own + 15) // 16 * 16
                qk = Uint32(base_qk) | (Uint32(n16) >> 3) << 17
                for ks in cutlass.range_constexpr(18):
                    if cutlass.const_expr(ks < 16):
                        at_q = ba + PAIR_Q + ks // 4 * TCS_REGION + ks % 4 * 32
                        a = tcs_desc(at_q, 16, 1024, TCS_SW128)
                        at_k = ba + self.off_k + ks // 4 * self.kreg + ks % 4 * 32
                        bd = tcs_desc(at_k, 16, 1024, TCS_SW128)
                    else:
                        a = tcs_desc(
                            ba + PAIR_Q + 4 * TCS_REGION + (ks - 16) * 256,
                            128,
                            512,
                            TCS_INTERLEAVE,
                        )
                        bd = tcs_desc(
                            ba + self.off_k + 4 * self.kreg + (ks - 16) * 256,
                            128,
                            512,
                            TCS_INTERLEAVE,
                        )
                    if leader:
                        tcs_mma_ss(tmem + PAIR_S, a, bd, qk, Int32(ks > 0))
            if leader:
                tcs_commit(bar + 1)
            for d in cutlass.range_constexpr(PAIR_GROUP):
                j = (g + d) % PAIR_GROUP
                if cutlass.const_expr(d == 0):
                    wait(bar + 2, p % 2, scope="cta")
                    v_at = ba + self.off_k + 2 * g * self.kreg
                else:
                    wait(bar + 3, p % 2)
                    v_at = ba + self.off_v
                fence_tmem()
                steps = (pair_owned(cnt, j) + 31) // 32
                # The pass's first MMA overwrites the accumulator (pass 0) or adds.
                mine = pair_owned(cnt, g)
                for kk in cutlass.range(steps, unroll=1):
                    a = tcs_desc(
                        ba + self.off_pt + j * PAIR_TAIL, 128, 256, TCS_INTERLEAVE
                    )
                    if kk < 4:
                        a = tcs_desc(
                            ba + self.off_p + j * TCS_REGION + kk * 32,
                            16,
                            1024,
                            TCS_SW128,
                        )
                    add = Int32(p > 0) | Int32(kk > 0)
                    if cutlass.const_expr(d > 0):
                        add = add | Int32(mine > 0)
                    for half in cutlass.range_constexpr(2):
                        bd = tcs_desc(
                            v_at + half * self.kreg + kk * 4096,
                            self.kreg,
                            1024,
                            TCS_SW128,
                        )
                        if leader:
                            tcs_mma_ss(tmem + 128 * half, a, bd, Uint32(pv), add)
            if leader:
                tcs_commit(bar + 5)

    @cute.jit
    def work(self, args, tmem, ba, bar, n, passes, g, b, tid):
        """Warps 0-7: per pass the loads, the maxima exchange, P and the sums (and the
        accumulator's rescale), then the epilogue."""
        query, kv, rows_in = args[0], args[1], args[2]
        warp, lane = tid // 32, tid % 32
        kv_base = Int64(kv.iterator.toint())
        m_run, l_run = Float32(-math.inf), Float32(0)
        lane_tmem = tmem + ((32 * (warp % 4)) << 16)
        if passes == 0:
            cute.arch.cluster_wait()
        for p in cutlass.range(passes, unroll=1):
            base = p * self.npass
            cnt = min(n - base, Int32(self.npass))
            if p > 0:
                self.load_rows(ba, rows_in, b, base, tid)
            if p == 0:
                for j in cutlass.range_constexpr(self.heads // 8):
                    h = warp + 8 * j
                    src = at(query, b, h, 0).toint()
                    copy_g2s(tcs_key_address(ba + PAIR_Q, h, lane), src + lane * 16)
                    if lane < 4:
                        dst = tcs_key_address(ba + PAIR_Q, h, 32 + lane)
                        copy_g2s(dst, src + 512 + lane * 16)
            named_barrier(1, 256)
            self.gather(ba, kv_base, cnt, g, warp, lane)
            tcs_cp_commit_wait()
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem(before=True)
            cute.arch.mbarrier_arrive(bar)
            if p == 0:
                cute.arch.cluster_wait()
            m_run, l_run = self.softmax(
                args, ba, bar, lane_tmem, p, passes, cnt, g, m_run, l_run
            )
            wait(bar + 5, p % 2, scope="cta")  # PV of this pass done
            cute.arch.sync_warp()
            fence_tmem()
        self.epilogue(args, ba, lane_tmem, m_run, l_run, passes, g, b, tid)

    @cute.jit
    def load_rows(self, ba, rows_in, b, base, tid):
        last = Int32(self.width - 1)
        for i in cutlass.range_constexpr(-(-self.npass // 256)):
            k = tid + 256 * i
            if k < self.npass:
                row = rows_in[b, min(base + k, last)]
                store_shared_words(ba + self.off_rw + 4 * k, (Uint32(row),))

    @cute.jit
    def gather(self, ba, kv_base, cnt, g, warp, lane):
        """cp.async of this pass: the CTA's own key rows, then the peer's keys' two
        128-byte V blocks of this CTA's columns (2 keys per warp instruction); V rows
        past a count up to its 32-key step zeroed."""
        own = pair_owned(cnt, g)
        first = pair_first(cnt, g)
        span = (own + 31) // 32 * 32
        for j in cutlass.range(self.own // 8, unroll=4):
            i = warp + 8 * j
            if i < own:
                row = load_words(
                    ba + self.off_rw + 4 * (first + i), count=1, shared=True
                )[0]
                src = kv_base + Int64(row) * ROW
                copy_g2s(
                    pair_key_address(ba + self.off_k, i, lane, self.kreg),
                    src + lane * 16,
                )
                if lane < 4:
                    copy_g2s(
                        pair_key_address(ba + self.off_k, i, 32 + lane, self.kreg),
                        src + 512 + lane * 16,
                    )
            elif (i < span) & (lane // 16 == g):
                store_shared_words(
                    pair_key_address(ba + self.off_k, i, lane, self.kreg),
                    (Uint32(0),) * 4,
                )
        peer = 1 - g
        nj = pair_owned(cnt, peer)
        fj = pair_first(cnt, peer)
        chunk = lane % 16
        block, sub = chunk // 8, chunk % 8
        for it in cutlass.range(-(-self.own // 16), unroll=2):
            i = 2 * (warp + 8 * it) + lane // 16
            dst = ba + self.off_v + block * self.kreg + i // 8 * 1024 + i % 8 * 128
            dst = dst + (sub ^ (i % 8)) * 16
            if i < nj:
                row = load_words(ba + self.off_rw + 4 * (fj + i), count=1, shared=True)[
                    0
                ]
                src = kv_base + Int64(row) * ROW + g * PAIR_COLS + chunk * 16
                copy_g2s(dst, src)
            elif i < (nj + 31) // 32 * 32:
                store_shared_words(dst, (Uint32(0),) * 4)

    @cute.jit
    def softmax(self, args, ba, bar, lane_tmem, p, passes, cnt, g, m_run, l_run):
        """Warps 0-7, thread = head h = 32 (w % 4) + lane, warp w the S chunks w // 4
        and w // 4 + 2: this CTA's maximum to the peer, the anchor max(both maxima,
        running), the accumulator rescaled to it (passes after the first), P (E4M3 at
        448 of the anchor) and its FP32 sum, the P block and sum to the peer.
        -> (anchor, running sum)."""
        scale_log2 = args[6]
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        quarter, half = warp % 4, warp // 4
        h = 32 * quarter + lane
        live = h < self.heads
        own = pair_owned(cnt, g)
        span = (own + 31) // 32 * 32
        peer = 1 - g
        wait(bar + 1, p % 2, scope="cta")
        cute.arch.sync_warp()
        fence_tmem()
        top = Float32(-math.inf)
        for i in cutlass.range_constexpr(3):
            c = half + 2 * i
            if 32 * c < own:
                for hf in cutlass.range_constexpr(2):
                    s = load_tmem(lane_tmem + PAIR_S + 32 * c + 16 * hf, 16)
                    for e in cutlass.range_constexpr(16):
                        top = max(
                            top, tcs_scaled(s[e], scale_log2, 32 * c + 16 * hf + e, own)
                        )
        tmp = ba + self.off_t + 4 * h
        store_shared_words(tmp + 512 * half, (float_bits(top),))
        named_barrier(1, 256)
        top = max(
            bits_float(load_words(tmp, count=1, shared=True)[0]),
            bits_float(load_words(tmp + 512, count=1, shared=True)[0]),
        )
        if live & (half == 0):
            push_float(ba + self.off_x + 4 * (g * 128 + h), top, bar + 4, peer)
        wait(bar + 4, p % 2)
        other = bits_float(
            load_words(ba + self.off_x + 4 * (peer * 128 + h), count=1, shared=True)[0]
        )
        anchor = max(max(top, other), m_run)  # m_run: -inf before the first pass
        alpha = Float32(1)
        if p > 0:
            alpha = cute.math.exp2(m_run - anchor, fastmath=True)
            if 32 * quarter < self.heads:  # warp-uniform: this warp's TMEM lanes
                for c0 in cutlass.range_constexpr(0, PAIR_COLS // 2, 16):
                    o_at = lane_tmem + PAIR_COLS // 2 * half + c0
                    o = load_tmem(o_at, 16)
                    tcs_store_tmem(o_at, [o[e] * alpha for e in range(16)])
        bias = anchor - Float32(LOG2_P_SCALE)
        total = Float32(0)
        row_at = ba + self.off_p + g * TCS_REGION + h // 8 * 1024 + h % 8 * 128
        tail_at = ba + self.off_pt + g * PAIR_TAIL + h // 8 * 256 + h % 8 * 16
        for i in cutlass.range_constexpr(3):
            c = half + 2 * i
            if 32 * c < span:
                for hf in cutlass.range_constexpr(2):
                    s = load_tmem(lane_tmem + PAIR_S + 32 * c + 16 * hf, 16)
                    words = []
                    for q in cutlass.range_constexpr(4):
                        probs = []
                        for e in cutlass.range_constexpr(4):
                            k = 16 * hf + 4 * q + e
                            x = tcs_scaled(s[4 * q + e], scale_log2, 32 * c + k, own)
                            prob = cute.math.exp2(x - bias, fastmath=True)
                            total += prob
                            probs.append(prob)
                        words.append(pack_fp8(tuple(probs)))
                    chunk = 2 * c + hf
                    if live & (c < 4):
                        store_shared_words(
                            row_at + (chunk ^ (h % 8)) * 16, tuple(words)
                        )
                    if live & (c >= 4):
                        store_shared_words(tail_at + hf * 128, tuple(words))
        store_shared_words(tmp + 1024 + 512 * half, (float_bits(total),))
        cute.arch.fence_proxy("async.shared", space="cta")
        tcs_bar_sync(3 + quarter, 64)  # warps w and w + 4: this quarter's P rows
        total = bits_float(load_words(tmp + 1024, count=1, shared=True)[0])
        total += bits_float(load_words(tmp + 1536, count=1, shared=True)[0])
        if live & (half == 0):
            push_float(
                ba + self.off_x + PAIR_GROUP * 512 + 4 * (g * 128 + h),
                total,
                bar + 3,
                peer,
            )
            if lane == 0:  # these 32 heads' P rows (4 KB + 1 KB of tail) to the peer
                block = ba + self.off_p + g * TCS_REGION + 4096 * quarter
                copy_s2cluster(block, block, bar + 3, peer, size=4096)
                tail = ba + self.off_pt + g * PAIR_TAIL + 1024 * quarter
                copy_s2cluster(tail, tail, bar + 3, peer, size=1024)
        fence_tmem(before=True)  # the rescale's TMEM stores precede the PV
        cute.arch.mbarrier_arrive(bar + 2)
        wait(bar + 3, p % 2)
        theirs = bits_float(
            load_words(
                ba + self.off_x + PAIR_GROUP * 512 + 4 * (peer * 128 + h),
                count=1,
                shared=True,
            )[0]
        )
        l_new = l_run * alpha + total + theirs
        if cutlass.const_expr(self.push is not None):  # noqa: SIM102
            if p == 0:
                self.rearm(args[0], g, tid)
        if p + 1 == passes:
            cute.arch.cluster_arrive_relaxed()
        if (tid == 0) & (p + 1 < passes):
            p_bytes: cutlass.Constexpr = self.heads * (128 + 32 + 4)
            m_bytes: cutlass.Constexpr = self.heads * 4
            cute.arch.mbarrier_arrive_and_expect_tx(bar + 3, p_bytes)
            cute.arch.mbarrier_arrive_and_expect_tx(bar + 4, m_bytes)
        return anchor, l_new

    @cute.jit
    def epilogue(self, args, ba, lane_tmem, m_run, l_run, passes, g, b, tid):
        """Warps 0-7: the accumulator times out_scale / l as BF16 into shared memory,
        then the rows and LSEs out (pushed to their owners, or the latent / LSE)."""
        latent, lse_out, out_scale = args[4], args[5], args[7]
        warp, lane = tid // 32, tid % 32
        h = 32 * (warp % 4) + lane
        half = warp // 4
        pitch: cutlass.Constexpr = 2 * PAIR_COLS + 16
        stage = ba + self.off_k
        if (passes > 0) & (32 * (warp % 4) < self.heads):
            inverse = out_scale / l_run
            for c0 in cutlass.range_constexpr(0, PAIR_COLS // 2, 16):
                o = load_tmem(lane_tmem + PAIR_COLS // 2 * half + c0, 16)
                for q in cutlass.range_constexpr(2):
                    words = tuple(
                        bf16x2(
                            o[8 * q + 2 * e] * inverse, o[8 * q + 2 * e + 1] * inverse
                        )
                        for e in range(4)
                    )
                    if cutlass.const_expr(self.push is not None):
                        words = tuple(no_neg_zero(w) for w in words)
                    at_byte = h * pitch + 2 * (PAIR_COLS // 2 * half + c0 + 8 * q)
                    store_shared_words(stage + at_byte, words)
        named_barrier(1, 256)
        lse2 = m_run - Float32(LOG2_P_SCALE) + cute.math.log2(l_run, fastmath=False)
        chunks: cutlass.Constexpr = PAIR_COLS // 8
        if cutlass.const_expr(self.push is None):
            for i in range(tid, self.heads * chunks, 256):
                hh, c = i // chunks, i % chunks
                words = (Uint32(0),) * 4
                if passes > 0:
                    words = load_words(
                        stage + hh * pitch + 16 * c, count=4, shared=True
                    )
                store_words(at(latent, b, hh, g * PAIR_COLS + 8 * c), words)
            if (g == 0) & (tid < self.heads):
                value = Float32(-math.inf)
                if passes > 0:
                    value = lse2 * LN2
                lse_out[b, tid] = value
        else:
            parity, rank = args[10], args[11]
            H, D, cap = self.push
            if passes > 0:
                for i in cutlass.range_constexpr(self.heads * chunks // 256):
                    hh = (tid + 256 * i) // chunks
                    c = (tid + 256 * i) % chunks
                    words = load_words(
                        stage + hh * pitch + 16 * c, count=4, shared=True
                    )
                    slot = ((parity * D + rank) * cap + b) * H + hh % H
                    pw = load_words(
                        ba + self.off_pp + 8 * (hh // H), count=2, shared=True
                    )
                    owner = Int64(pw[0]) | (Int64(pw[1]) << 32)
                    address = owner + (Int64(slot) * LATENT + g * PAIR_COLS + 8 * c) * 2
                    store_relaxed(address, words)
            if (g == 0) & (tid < self.heads):
                value = Float32(EMPTY_LSE)
                if passes > 0:
                    value = lse2
                    bad = (value != value) | (value <= Float32(EMPTY_LSE))
                    if bad | (value >= Float32(-EMPTY_LSE)):
                        value = Float32(EMPTY_LSE)
                slot = ((parity * D + rank) * cap + b) * H + tid % H
                lw = load_words(
                    ba + self.off_pp + 128 + 8 * (tid // H), count=2, shared=True
                )
                laddr = Int64(lw[0]) | (Int64(lw[1]) << 32)
                store_relaxed(laddr + Int64(slot) * 4, (float_bits(value),))


def tcs_groups(batch, dcp):
    """CTAs per row of TcsDecode for the pushing decode of `batch` rows under DCP
    `dcp` (TCS_ROWS, VLLM_KIMI_K3_DCP8_TC_SPARSE=1), else 0."""
    if not TCS_ON or not dcp:
        return 0
    return batch_table(TCS_ROWS.get(dcp, "")).get(batch, 0)


def tcs_pair_own(batch, dcp):
    """Keys per CTA per pass of TcsDecodePair at `batch` rows under DCP `dcp`
    (TCS_PAIR_OWN, else PAIR_OWN)."""
    return (
        batch_table(TCS_PAIR_OWN.get(dcp, "")).get(batch, PAIR_OWN) if dcp else PAIR_OWN
    )


def tcs_carry(batch, dcp):
    """Whether TcsDecode of `batch` rows under DCP `dcp` anchors P at the running
    maximum from its second pass on (the VLLM_KIMI_K3_DCP8_TC_SPARSE_WIDE batches)."""
    return bool(
        TCS_WIDE_ON and dcp and batch_table(TCS_WIDE_ROWS.get(dcp, "")).get(batch, 0)
    )


def _dcp_world():
    """The decode context parallel size of this process (0 if unknown)."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group

        return get_dcp_group().world_size
    except Exception:  # noqa: BLE001 (no distributed state: not a DCP decode)
        return 0


def sparse_for(rows, gathered, padded):
    """Whether a DCP batch of `rows` runs SparseDcpDecode (`gathered` query
    heads of a `padded`-row q mailbox), or TcsDecode (tcs_groups)."""
    fits = gathered % HEADS == 0 and gathered <= padded
    if fits and TCS_ON and gathered <= 128 and tcs_groups(rows, _dcp_world()):
        return True
    return fits and batch_table(SPARSE_DCP).get(rows) == "sparse"


def sparse_px(batch, heads, push=None):
    """Whether the pushing decode of `batch` rows runs SparseDcpDecodePx (SPARSE_PX
    of its DCP size, two-tile CTAs)."""
    if push is None or heads % (2 * HEADS):
        return False
    return batch_table(SPARSE_PX.get(push[1], "")).get(batch, 0) == 1


def sparse_layout(batch, heads, push=None):
    """(head tiles per CTA, bulk exchange) of SparseDcpDecode over `batch` rows of
    `heads` gathered heads: SPARSE_TILES / SPARSE_BULK of the pushing decode's
    DCP size (`push`: (owner heads, DCP)), else (1, False)."""
    if push is None:
        return 1, False
    dcp = push[1]
    tiles = batch_table(SPARSE_TILES.get(dcp, "")).get(batch, 1)
    if heads % (HEADS * tiles):
        tiles = 1
    bulk = batch_table(SPARSE_BULK.get(dcp, "")).get(batch, 0) == 1
    return tiles, bulk


@cache
def compile_sparse(
    batch,
    heads,
    padded,
    width,
    cap,
    device,
    push=None,
    tiles=None,
    bulk=None,
    px=None,
    groups=None,
):
    """SparseDcpDecode of `batch` rows (the q mailbox of `cap` rows; `push`:
    (owner heads, DCP) to push into the DcpValue mailbox; `tiles` / `bulk`:
    sparse_layout's unless given; `px`: SparseDcpDecodePx, sparse_px's unless
    given; `groups`: TcsDecode's CTAs per row, tcs_groups' of the pushing decode
    unless given, 0: none)."""
    if groups is None:
        groups = tcs_groups(batch, push[1]) if push is not None else 0
    if groups:
        query = fake((cap, padded, ROW), None, torch.uint8)
        kv = fake((cute.sym_int(), ROW), (ROW, 1), torch.uint8)
        rows = fake((cute.sym_int(), width), (width, 1), torch.int32, 4)
        valid = fake((cute.sym_int(),), (1,), torch.int32, 4)
        latent = fake((batch, padded, LATENT))
        lse = fake((batch, padded), None, torch.float32, 4)
        owners = None if push is None else (*push, cap)
        carry = tcs_carry(batch, push[1]) if push is not None else False
        if groups == PAIR_GROUP:
            own = tcs_pair_own(batch, push[1]) if push is not None else PAIR_OWN
            tcs: TcsDecode | TcsDecodePair = TcsDecodePair(
                batch, heads, padded, width, owners, own
            )
        else:
            tcs = TcsDecode(batch, heads, padded, width, owners, groups, carry)
        tcs_args: tuple = (query, kv, rows, valid, latent, lse)
        tcs_args += (Float32(1.0), Float32(1.0))
        if push is not None:
            peers = fake((push[1],), None, torch.int64, 8)
            tcs_args += (peers, peers, Int32(0), Int32(0))
        with torch.accelerator.device_index(device):
            return cute.compile(tcs, *tcs_args, make_fake_stream())
    px = sparse_px(batch, heads, push) if px is None else px
    layout = sparse_layout(batch, heads, push)
    tiles = layout[0] if tiles is None else tiles
    bulk = layout[1] if bulk is None else bulk
    query = fake((cap, padded, ROW), None, torch.uint8)
    kv = fake((cute.sym_int(), ROW), (ROW, 1), torch.uint8)
    rows = fake((cute.sym_int(), width), (width, 1), torch.int32, 4)
    valid = fake((cute.sym_int(),), (1,), torch.int32, 4)
    latent = fake((batch, padded, LATENT))
    lse = fake((batch, padded), None, torch.float32, 4)
    owners = None if push is None else (*push, cap)
    kernel: SparseDcpDecode
    if px:
        kernel = SparseDcpDecodePx(batch, heads, padded, width, owners)
    else:
        kernel = SparseDcpDecode(batch, heads, padded, width, owners, tiles, bulk)
    args: tuple = (query, kv, rows, valid, latent, lse, Float32(1.0), Float32(1.0))
    if push is not None:
        peers = fake((push[1],), None, torch.int64, 8)
        args += (peers, peers, Int32(0), Int32(0))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


def run_sparse(
    kernel,
    query,
    kv,
    rows,
    valid,
    latent,
    lse,
    scale_log2,
    out_scale,
    push=None,
):
    """`query` [cap, padded, 576] (FP8 bytes), `kv` [N, 576] (the flat cache
    rows), `rows` [>= B, K] / `valid`: this rank's flat rows and counts ->
    `latent` [B, padded, 512] BF16 and `lse` [B, padded] FP32 (first `heads`
    heads), or with `push` (DcpMailbox, parity, rank) into the owners' mailbox."""
    args: tuple = (
        to_cute(query.view(torch.uint8)),
        to_cute(kv.view(torch.uint8)),
        to_cute(rows, 4),
        to_cute(valid, 4),
        to_cute(latent),
        to_cute(lse, 4),
        Float32(scale_log2),
        Float32(out_scale),
    )
    if push is not None:
        mail, parity, rank = push
        peers = to_cute(mail.peers, 8), to_cute(mail.lse_peers, 8)
        args += (*peers, Int32(parity), Int32(rank))
    kernel(*args, stream(latent.device))
    return latent, lse
