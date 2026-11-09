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
from cutlass import Float32, Int32, Int64, Uint8, Uint32
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
    float_bits,
    launch,
    ldmatrix,
    load_words,
    mma_into,
    no_neg_zero,
    push_float,
    store_relaxed,
    store_words,
    wait,
    zeros,
)
from ..mla.dcp import EMPTY_LSE, Q_SENT

ROW = 576  # FP8 bytes per cache / query row (512 latent + 64 rope)
LATENT = 512
SPLIT = 4  # CTAs per (16 heads, row): a cluster
CHUNK = 64  # keys per CTA per shared-memory chunk (DCP8: ~2048 / 8 / SPLIT)
PITCH = ROW + 16  # 37 x 16 B: the 8 rows of an ldmatrix hit distinct banks
P_PITCH = CHUNK + 16  # 5 x 16 B
HEADS = 16  # query heads per CTA (one m16 tile)
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
    """`batch` rows, `heads` gathered query heads (a multiple of 16; the q
    mailbox and outputs hold `padded` rows per token), the merge's row width
    `width` (its K). `push` (owner heads H, DCP, mailbox rows): push the
    merged rows and base-2 LSEs into their owners' DcpValue "bf16" mailbox and
    re-arm the q mailbox rows read (DcpValue `pushed` then only merges)."""

    def __init__(self, batch, heads, padded, width, push=None):
        assert heads % HEADS == 0 and heads <= padded
        self.batch, self.heads, self.padded, self.width = batch, heads, padded, width
        self.push = push
        self.grid = (heads // HEADS * SPLIT, batch, 1)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS, cluster=(SPLIT, 1, 1))

    @cute.kernel
    def kernel(self, args):
        query, kv, rows_in, valid_in = args[:4]
        scale_log2, out_scale = args[6:8]
        smem = cutlass.utils.SmemAllocator()
        kvs = alloc(smem, Uint8, (CHUNK, PITCH), 128)
        qs = alloc(smem, Uint8, (HEADS, PITCH), 128)
        ps = alloc(smem, Uint8, (HEADS, P_PITCH), 128)
        red = alloc(smem, Float32, (2, QK_WARPS, HEADS))
        # The cluster's partials of this CTA's columns: rows and (max, sum) per
        # source CTA, pushed by the sources (st.async on ``inbound``).
        inbox = alloc(smem, Float32, (SPLIT, HEADS, COLS), 16)
        stats = alloc(smem, Float32, (SPLIT, 2, HEADS))
        bar = smem.allocate(cutlass.Uint64, 8)
        inbound = smem.allocate(cutlass.Uint64, 8)
        tid, _, _ = cute.arch.thread_idx()
        x, b, _ = cute.arch.block_idx()
        tile, split = x // SPLIT, x % SPLIT
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        if tid == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(inbound, 1)
            cute.arch.mbarrier_init_fence()
            inbound_bytes = SPLIT * (HEADS * COLS + 2 * HEADS) * 4
            cute.arch.mbarrier_arrive_and_expect_tx(inbound, inbound_bytes)
        cute.arch.sync_threads()
        # Every CTA's ``inbound`` is armed before any peer pushes (waited below).
        cute.arch.cluster_arrive_relaxed()
        # Nothing above reads the inputs: the kernel before may launch this early.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        n = valid_in[b]
        n = max(min(n, Int32(self.width)), Int32(0))
        # This CTA's keys: [first, first + mine).
        per = (n + SPLIT - 1) // SPLIT
        first = min(n, split * per)
        mine = min(n - first, per)
        kv_base = kv.iterator.toint()
        neg_inf = Float32(-math.inf)
        m_run = zeros(2, Float32)  # rows group, group + 8 (log2 units)
        m_run.fill(neg_inf)
        l_run = zeros(2, Float32)
        acc = zeros((4, 4), Float32)  # latent n-tiles j: columns 4 g + j
        # ldmatrix lane addresses: A (16 heads x 32 B), B (8 keys x 2 x 32 B).
        a_row = lane % 8 + 8 * ((lane // 8) % 2)
        a_col = 16 * (lane // 16)
        b_row = 8 * warp + lane % 8
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
                chunks16 = ROW // 16
                for i in cutlass.range_constexpr(-(-HEADS * chunks16 // THREADS)):
                    flat = tid + THREADS * i
                    if flat < HEADS * chunks16:
                        q_row, q_chunk = flat // chunks16, flat % chunks16
                        src = at(query, b, tile * HEADS + q_row, 16 * q_chunk)
                        words = load_words(src, count=4, mutable=True)
                        dst = qs.iterator + q_row * PITCH + 16 * q_chunk
                        store_words(dst, words, shared=True)
            wait(bar, phase, scope="cta")
            phase = phase ^ 1
            cute.arch.sync_threads()  # the zeroed V rows
            # ---- QK: warp w < QK_WARPS, keys 8 w .. 8 w + 7 --------------------------
            s = zeros(4, Float32)
            if (warp < QK_WARPS) & (8 * warp < span):
                parts = zeros((QK_CHAINS, 4), Float32)
                for kp in cutlass.range_constexpr(ROW // 64):
                    k = ldmatrix(kvs.iterator + b_row * PITCH + 64 * kp + b_col)
                    for h in cutlass.range_constexpr(2):
                        ks = 2 * kp + h
                        a = ldmatrix(qs.iterator + a_row * PITCH + 32 * ks + a_col)
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
                key = 8 * warp + 2 * quarter + e % 2
                value = neg_inf
                if key < count:
                    value = s[e] * scale_log2
                s[e] = value
                top[e // 2] = max(top[e // 2], value)
            for r in cutlass.range_constexpr(2):
                for shift in cutlass.range_constexpr(2):
                    other = cute.arch.shuffle_sync_bfly(top[r], 1 << shift)
                    top[r] = max(top[r], other)
            if (quarter == 0) & (warp < QK_WARPS):
                red[0, warp, group] = top[0]
                red[0, warp, group + 8] = top[1]
            cute.arch.sync_threads()
            alpha = zeros(2, Float32)
            alpha.fill(1.0)
            m_new = zeros(2, Float32)
            for r in cutlass.range_constexpr(2):
                chunk_max = red[0, 0, group + 8 * r]
                for w in cutlass.range_constexpr(1, QK_WARPS):
                    chunk_max = max(chunk_max, red[0, w, group + 8 * r])
                m_new[r] = max(m_run[r], chunk_max)
                if m_run[r] != neg_inf:
                    alpha[r] = cute.math.exp2(m_run[r] - m_new[r], fastmath=True)
            # ---- P (E4M3 at 448 of the running max) and its FP32 sums -------------
            sums = zeros(2, Float32)
            if (warp < QK_WARPS) & (8 * warp < span):
                for r in cutlass.range_constexpr(2):
                    bias = m_new[r] - Float32(LOG2_P_SCALE)
                    p = []
                    for e in cutlass.range_constexpr(2):
                        value = s[2 * r + e]
                        prob = Float32(0)
                        if value != neg_inf:
                            prob = cute.math.exp2(value - bias, fastmath=True)
                        sums[r] += prob
                        p.append(prob)
                    key = 8 * warp + 2 * quarter
                    dst = ps.iterator + (group + 8 * r) * P_PITCH + key
                    store_fp8_pair_shared(dst, p[0], p[1])
            for r in cutlass.range_constexpr(2):
                for shift in cutlass.range_constexpr(2):
                    sums[r] += cute.arch.shuffle_sync_bfly(sums[r], 1 << shift)
            if (quarter == 0) & (warp < QK_WARPS):
                red[1, warp, group] = sums[0]
                red[1, warp, group + 8] = sums[1]
            cute.arch.sync_threads()
            for r in cutlass.range_constexpr(2):
                total = red[1, 0, group + 8 * r]
                for w in cutlass.range_constexpr(1, QK_WARPS):
                    total += red[1, w, group + 8 * r]
                l_run[r] = l_run[r] * alpha[r] + total
                m_run[r] = m_new[r]
                for j in cutlass.range_constexpr(4):
                    for e in cutlass.range_constexpr(2):
                        acc[j, 2 * r + e] *= alpha[r]
            # ---- PV: warp w, latent columns 32 w .. 32 w + 31 ---------------------
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
                    for j in cutlass.range_constexpr(4):
                        b_frag = (halves[0][j], halves[1][j])
                        mma_into(acc[j, None], a, b_frag, fp8=True)
        # ---- push this CTA's partial to the column owners -----------------------
        cute.arch.cluster_wait()
        owner = warp // (COLS // 32)  # columns 32 w .. 32 w + 31
        for r in cutlass.range_constexpr(2):
            row = group + 8 * r
            for e in cutlass.range_constexpr(2):
                # acc[j, 2 r + e]: column 32 w + 8 q + 4 e + j
                values = tuple(acc[j, 2 * r + e] for j in range(4))
                column = 32 * (warp % (COLS // 32)) + 8 * quarter + 4 * e
                dst = at(inbox, split, row, column)
                push_float(dst, values, inbound, owner)
            if (warp == 0) & (quarter == 0):
                for peer in cutlass.range_constexpr(SPLIT):
                    push_float(at(stats, split, 0, row), m_run[r], inbound, peer)
                    push_float(at(stats, split, 1, row), l_run[r], inbound, peer)
        wait(inbound, 0)  # every source's partial of these columns landed
        self.merge(args, inbox, stats, tile, split, b, tid)
        if cutlass.const_expr(self.push is not None):  # noqa: SIM102 (compile time)
            # Every CTA of the cluster has read the q rows (it pushed after PV).
            if split == 0:
                chunks16 = ROW // 16
                for i in range(tid, HEADS * chunks16, THREADS):
                    q_row, q_chunk = i // chunks16, i % chunks16
                    sentinel = (Uint32(Q_SENT),) * 4
                    q_dst = at(query, b, tile * HEADS + q_row, 16 * q_chunk)
                    store_words(q_dst, sentinel)

    @cute.jit
    def merge(self, args, inbox, stats, tile, split, b, tid):
        """Columns split * COLS .. + COLS of the 16 heads over the cluster's
        partials (FP32, in `inbox` / `stats`): thread t takes head t // 32, 4
        columns."""
        latent, lse_out = args[4:6]
        out_scale = args[7]
        neg_inf = Float32(-math.inf)
        head, lane = tid // 32, tid % 32
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
        gh = tile * HEADS + head
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


def sparse_for(rows, gathered, padded):
    """Whether a DCP batch of `rows` runs SparseDcpDecode (`gathered` query
    heads of a `padded`-row q mailbox)."""
    fits = gathered % HEADS == 0 and gathered <= padded
    return fits and batch_table(SPARSE_DCP).get(rows) == "sparse"


@cache
def compile_sparse(batch, heads, padded, width, cap, device, push=None):
    """SparseDcpDecode of `batch` rows (the q mailbox of `cap` rows; `push`:
    (owner heads, DCP) to push into the DcpValue mailbox)."""
    query = fake((cap, padded, ROW), None, torch.uint8)
    kv = fake((cute.sym_int(), ROW), (ROW, 1), torch.uint8)
    rows = fake((cute.sym_int(), width), (width, 1), torch.int32, 4)
    valid = fake((cute.sym_int(),), (1,), torch.int32, 4)
    latent = fake((batch, padded, LATENT))
    lse = fake((batch, padded), None, torch.float32, 4)
    owners = None if push is None else (*push, cap)
    kernel = SparseDcpDecode(batch, heads, padded, width, owners)
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
