# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query overlap of a K3.1 DSA layer. The indexer path (wq_b -> DsaIndex -> scoring
and top-k, under DCP the top-k merge and the local slots) needs only the normalized
q_c rows of the query, so the rest of the query stage leaves the critical path:

  QueryNorm   q_a RMSNorm -> the q_c rows, on the main stream ahead of wq_b
              (bit-identical to MlaQuery's: its scalar form below tiled_from rows,
              its tiled kernel's packed FP32 pairs from there)
  MlaQuery    Q_B, W_UK absorption, the query rows, the MLA KV insert and, under
              DCP, the q mailbox publish, on a side stream (`store_qc=False`,
              `poll=False`)
  QueryWait   DCP: the q mailbox poll (until every rank's heads landed) after the
              MlaQuery on the side stream, a few CTAs instead of MlaQuery's grid

The main stream joins the side stream just before the sparse core."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32

from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    fma,
    fma2,
    launch,
    load_bf16x8,
    load_volatile,
    load_words,
    mul2,
    published,
    query_norm_rsqrt,
    store_words,
    unpack_bf16,
    warp_sum,
)
from ..mla.query import Q_SENT, tiled_from

# serial: MlaQuery (q_c, publish and poll) -> wq_b -> ... -> core (one stream).
# fork: QueryNorm -> wq_b -> ... -> join -> core, MlaQuery (+ QueryWait) forked.
# wait: MlaQuery (q_c, publish) -> wq_b -> ... -> QueryWait -> core (one stream).
# front: as fork, with a fused index front for QueryNorm -> wq_b -> DsaIndex: the
#   cluster IndexFront (front.py) up to 40 rows, then IndexFrontMma (front_mma.py).
#   Under DCP the fork may follow the index front (QUERY_AFTER_FRONT).
# late (TP, fused indexer): QueryNorm -> wq_b -> ... -> IndexerSelect -> MlaQuery ->
#   core on one stream: MlaQuery reads none of the indexer's outputs, so it starts
#   with the select (PDL) and waits for it only at its end (MlaQuery(late=True)).
# late_front: as late, with the fused index front for QueryNorm -> wq_b -> DsaIndex.
# Under serial the q_c rows feed IndexFrontMma in place of wq_b -> DsaIndex
# (prepared.FUSED_INDEX).
# Per topology and batch of rows, measured on the integrated stack (GB300, fp8 KV,
# whole DSA layer, us; the fused indexer at TP, the #63 DCP top-k merge under DCP):
# - TP8 (fused indexer; with prims_ts' PDL hold, attn_fi.py): late_front wins up to 20
#   rows (merged stack, one allocation, 8K/32K/128K/AgentX: R1 q5 35.1/39.0/49.5/35.1
#   vs serial with IndexFrontMma from q_c 38.7/42.4/52.9/39.1 and fork
#   40.9/45.2/55.4/41.7; q1 M1-M16 by 1-4 us over serial); at 21-29 rows serial and
#   fork tie within run-to-run noise (+-2 us); from 30 rows fork wins (R8 q5 by 3-8
#   us; MlaQuery outlasts the select under late, IndexFrontMma's 128 CTAs crowd
#   the forked MlaQuery under front).
# - DCP8 (the #63 merge; IndexFrontMma from 41 rows): front beats fork by 2.5-6.5 us
#   at 5-64 rows at every context. At 80-320 rows it is context-dependent: front
#   wins at 128K / AgentX (R64 -10.2 / -0.4, R48 -7.5 / -5.0, R32 -3.4 / -2.6 us)
#   and loses at 8K / 32K (R26 +6.3 / +4.4, R48 +4.3 / +1.6 us); front everywhere
#   (the long-context AgentX workload). wait with IndexFrontMma lost 1-8 us.
# - TP1 (DEP, 96 heads: the split query, mla.query_wide): fork (QueryNorm, then the
#   Q_B projection and MlaAbsorb on the side stream beside the indexer path).
# - DCP8 with the split query (mla.query_wide, from SPLIT_FROM rows): its four
#   kernels (QueryNorm, Q_B, MlaAbsorb publishing, QueryWait) forked beside the
#   index front and the persistent scorer wait for SMs the indexer holds, so at the
#   widest batches they run ahead of the indexer on the main stream ("wait":
#   QueryWait before the core). GB300 DCP8 whole layer, us (32K / AgentX): R8 fork
#   117.2 / 120.8, wait 123.4 / 127.5; R16 fork 134.0, front 137.7, wait 142.1
#   (MlaQuery front 147.0); R32 fork 169.9 / 184.7, wait 183.6 / 190.6; R48 wait
#   207.5 / 219.4, fork 233.6 / 249.8; R64 wait 250.0, fork 295.5, front 320.1
#   (MlaQuery front 271.8).
QUERY_OVERLAP = {
    "tp": "1-20:late_front,21-320:fork",
    "dcp": "1-320:front",
    "dcp_split": "1-239:fork,240-320:wait",
    "tp1": "1-320:fork",
}
# TP without the fused indexer (the late modes follow its select): the TP table the
# stack ran before them.
QUERY_OVERLAP_UNFUSED = "1:front,2-9:serial,10:front,11-320:fork"
MODES = ("serial", "fork", "wait", "front", "late", "late_front")
LATE = ("late", "late_front")  # TP with the fused indexer only (else: fork, front)
NORM = ("fork", "late")  # QueryNorm writes the q_c rows
FRONT = ("front", "late_front")  # IndexFront


# The modes of the split query (TP1): QueryNorm ahead of it, or forked beside it.
WIDE_MODES = ("serial", "fork")

# DCP "front": MlaQuery forked after IndexFront instead of before it, launched
# without PDL, per DCP size ("rows:1"; other sizes and rows fork before IndexFront).
# Forked before, MlaQuery runs beside IndexFront and then beside the persistent
# scorer that follows it; forked after IndexFront without PDL, it starts once
# IndexFront completes, beside the scorer only (with PDL it would start early
# again). Only the launch order changes, not what any kernel computes. Chosen by an
# end-to-end A/B at c = 1 (TP16 + DCP16, DSpark k = 7), together with the local
# top-k cluster policy (indexer.LOCAL_CLUSTER); other DCP sizes were not measured.
QUERY_AFTER_FRONT = {16: "6-8:1"}


def query_after_front(batch, dcp):
    """Whether the "front" mode's MlaQuery of `batch` rows under DCP `dcp`
    forks after IndexFront (QUERY_AFTER_FRONT)."""
    return batch_table(QUERY_AFTER_FRONT.get(dcp, "")).get(batch, 0) == 1


def overlap_for(batch, policy=None, dcp=False, wide=False, split=False):
    """The query overlap mode of `batch` rows: `policy` is a mode or a
    "rows:mode,..." table (default QUERY_OVERLAP of the topology, `dcp`;
    `wide`: the split query of TP1, WIDE_MODES only; `split`: the split query
    of fewer heads, under DCP its own table)."""
    topology = "tp1" if wide else "dcp" if dcp else "tp"
    if split and dcp and not wide:
        topology = "dcp_split"
    policy = policy or QUERY_OVERLAP[topology]
    mode = policy if policy in MODES else batch_table(policy).get(batch, "fork")
    modes = WIDE_MODES if wide else MODES
    if mode not in modes:
        raise ValueError(f"Kimi K3.1 DSA query overlap {mode!r}: expected {modes}")
    return mode


@cache
def side_stream(device_index):
    """The DSA layers' query side stream of a device."""
    return torch.cuda.Stream(device=device_index)


class QueryNorm:
    """The q_a RMSNorm of `batch` rows of the packed projection -> q_c [B, 1536]
    BF16, exactly as MlaQuery (`heads`) computes it for the same batch."""

    def __init__(self, heads: int, batch: int):
        self.batch = batch
        self.wide = batch >= tiled_from(heads)
        self.grid = -(-batch // 8) if self.wide else batch

    @cute.jit
    def __call__(self, *args):
        launch(self.tiled if self.wide else self.kernel, args, self.grid, 256)

    @cute.kernel
    def kernel(self, args):
        """One row per CTA: 8 columns per thread, MlaQuery.kernel's FMA order."""
        qkv, w_qa, qc, eps = args
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        ss = alloc(smem, Float32, 8)
        weights = load_bf16x8(w_qa.iterator + (tid % 192) * 8)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        xv = cute.make_rmem_tensor(8, Float32)
        total = Float32(0)
        if tid < 192:
            values = load_bf16x8(at(qkv, row, tid * 8), mutable=True)
            for i in cutlass.range_constexpr(8):
                xv[i] = values[i]
                total = fma(values[i], values[i], total)
        total = warp_sum(total)
        if lane == 0:
            ss[warp] = total
        cute.arch.sync_threads()
        if tid < 192:
            total = Float32(0)
            for w in cutlass.range_constexpr(8):
                total += ss[w]
            rr = query_norm_rsqrt(total, eps)
            words = [
                bf16x2(xv[i] * rr * weights[i], xv[i + 1] * rr * weights[i + 1])
                for i in range(0, 8, 2)
            ]
            store_words(at(qc, row, tid * 8), tuple(words))

    @cute.kernel
    def tiled(self, args):
        """One row per warp: 48 columns per lane, MlaQuery.tiled's packed pairs."""
        qkv, w_qa, qc, eps = args
        tid, _, _ = cute.arch.thread_idx()
        tile, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        row = tile * 8 + warp
        gamma = []
        for i in cutlass.range_constexpr(6):
            gamma += list(load_bf16x8(w_qa.iterator + 256 * i + 8 * lane))
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if row < self.batch:
            values: list = []
            for i in cutlass.range_constexpr(6):
                source = at(qkv, row, 256 * i + 8 * lane)
                for word in load_words(source, count=4, mutable=True):
                    values += list(unpack_bf16(word))
            zero = Float32(0), Float32(0)
            chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
            for e in cutlass.range_constexpr(0, 48, 2):
                x = values[e], values[e + 1]
                chains[e % 4 // 2] = fma2(x, x, chains[e % 4 // 2])
            (c0, c1), (c2, c3) = chains
            total = warp_sum((c0 + c1) + (c2 + c3))
            rr = query_norm_rsqrt(total, eps)
            for i in cutlass.range_constexpr(6):
                words = []
                for e in cutlass.range_constexpr(8 * i, 8 * i + 8, 2):
                    x = values[e], values[e + 1]
                    y = mul2(mul2(x, (rr, rr)), (gamma[e], gamma[e + 1]))
                    words.append(bf16x2(*y))
                store_words(at(qc, row, 256 * i + 8 * lane), tuple(words))


class QueryWait:
    """DCP: poll the gathered q mailbox (`batch` tokens of `gathered` = DCP x H
    published heads in rows of `q_heads`, `q_bytes` per element) until no 16 B
    chunk is the sentinel. It first waits for the MlaQuery before it, so the KV
    insert is complete too when the main stream joins."""

    def __init__(self, batch, gathered, q_heads, q_bytes):
        self.row = 36 * q_bytes  # 16 B chunks per head row
        self.token, self.q_heads = gathered * self.row, q_heads
        self.chunks = batch * self.token
        self.ctas = min(32, -(-self.chunks // (256 * 8)))

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, 256)

    @cute.kernel
    def kernel(self, args):
        (mailbox,) = args
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        threads: cutlass.Constexpr = self.ctas * 256
        token, row, chunks = self.token, self.row, self.chunks
        first = cta * 256 + tid
        steps: cutlass.Constexpr = -(-chunks // threads)
        for i0 in cutlass.range_constexpr(0, steps, 8):
            n: cutlass.Constexpr = min(8, steps - i0)
            picks = [min(first + (i0 + i) * threads, chunks - 1) for i in range(n)]
            sources = [
                mailbox.iterator + (c // token * self.q_heads * row + c % token) * 16
                for c in picks
            ]
            loads = [load_volatile(source, count=4) for source in sources]
            for i in cutlass.range_constexpr(n):  # the loads above are in flight
                words = loads[i]
                while not published(words, sentinel=Q_SENT):
                    words = load_volatile(sources[i], count=4)
