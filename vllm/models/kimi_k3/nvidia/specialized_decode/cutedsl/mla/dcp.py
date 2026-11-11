# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA decode under decode context parallelism (DCP = TP, rank d owns head block d)
without collectives, three kernels per layer. Each rank's KV cache holds 1 / DCP of
every context.

  MlaQuery     FP8 query rows [B, H, 576] (+ the KV insert where this rank owns the
               slot); each CTA multicasts its bytes into every rank's Lamport q
               mailbox [B, DCP*H, 576], then polls its share until all landed
  tokenspeed   all DCP*H heads over the local KV shard -> BF16 rows + LSE (log2)
  DcpValue     re-arms the q mailbox; rows pushed to their head owner's Lamport
               mailbox (peer stores; BF16 latent rows, or "proj": BF16 value
               rows through W_UV); the owner LSE-merges its H heads over the DCP
               sources, then W_UV and the gate (padding from the global
               lengths). A source with nothing to merge (an empty local shard)
               sends only its LSE.
"""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from ....specialized_decode.symm import multicast_rendezvous, peer_rendezvous
from ..common.host import fake, to_cute
from ..common.policy import batch_table
from ..common.primitives import (
    alloc,
    at,
    bf16x2,
    bits_float,
    div_rn,
    float_bits,
    launch,
    load_volatile,
    load_words,
    mma_into,
    named_barrier,
    no_neg_zero,
    pack_pair,
    published,
    sigmoid,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    warp_sum,
    zeros,
)

LSE_SENT = 0xFFFFFFFF  # a NaN pattern the merge never sends
# The LSE of an empty or invalid source: finite, so the merge never forms inf - inf.
EMPTY_LSE = -3.0e38
Q_SENT = 0xFFFFFFFF  # four FP8 0xFF bytes: never written (satfinite, canonical NaN)
PAIR_SENT = 0x80000000  # BF16 pair (+0, -0); sent pairs have no negative zero
# "proj": every source applies W_UV of each gathered head to its own partial row before
# the exchange (the merge is linear in the rows: sum_s w_s O_s W = sum_s w_s (O_s W)),
# so the owners receive BF16 [H, 128] value rows instead of [H, 512] latents: 4x fewer
# NVLink bytes, for a replicated W_UV (all DCP * H heads) per layer.
PROJECTED = "proj"
# Latent rows a "proj" projecting CTA stages per pipelined step (at most; DCP8
# GB300 DcpValue at 320 rows: 16 -> 64.7 us, 32 -> 55.7, 64 -> 54.6, but 64 loses
# 1-2 us at 80-160 rows).
PROJ_ROWS = 32
# Both formats are exact up to BF16 rounding (no lossy 8-bit partials).
FORMATS = ("bf16", PROJECTED)
# The default exchange per layer kind, from where "proj" wins on the
# DCP8 GB300 attention layer (q5, slowest rank): K3.1 DSA from 160 rows (-6 to -7 us
# at 160, -17 to -19 at 320; it ties BF16 at 5-80 rows, where its serial per-head
# projection costs what the 4x fewer NVLink bytes save); MLA from 40 rows (-1 at
# 40, -2 to -3 at 60-80, -9 at 160, -17 to -23 at 320; +1 to +3.5 at 5-20 rows).
# MLA starts at 65 rows for now: the step serves at most 64 rows today, so K3's
# served numerics stay those of the BF16 exchange until a gsm8k run gates "proj"
# (then lower it to 40). The DcpBox allocates the "proj" mailbox and the layers
# gather W_UV only when a batch up to its cap uses "proj".
PARTIALS_DEFAULT = {"mla": "1-64:bf16,65-320:proj", "dsa": "1-159:bf16,160-320:proj"}
# Request rows per merging CTA of the "pushed" merge (the sparse decode pushed the
# rows), batch tables per DCP size; absent: one CTA per (head, 16 rows). One CTA per
# (head, R rows) reads each row's DCP partial chunks on 128 threads in one round
# (instead of 64 threads per row polling 8 sources at a time), with the same
# results bit for bit. R = 2 rather than 1 keeps SMs free for the clusters of the
# kernel after it (o_proj), which otherwise partly start only once DcpValue ends.
# Set from a one-GPU microbenchmark (DCP16: 6 heads, 16 sources; 8 rows, L2
# flushed) from the start of the sparse decode (two head tiles per CTA) to the end
# of a kernel of o_proj's launch shape after DcpValue: R = 2 ahead of R = 1, both
# ahead of the 16-row CTAs.
VALUE_ROWS = {16: "7-8:2"}


def resolve_partials(spec, kind):
    """The DcpBox policy of a `spec` ("default": PARTIALS_DEFAULT of the layer
    `kind`, "mla" or "dsa"; anything else as given)."""
    return PARTIALS_DEFAULT[kind] if spec in (None, "default") else spec


def partials_for(spec, batch):
    """The partials format of a `batch` of rows under the `spec` policy: a format
    ("bf16", "proj") or a batch table of formats."""
    if spec in FORMATS:
        return spec
    table = batch_table(spec)
    fmt = table.get(batch, "bf16")
    assert fmt in FORMATS, (spec, fmt)
    return fmt


class DcpMailbox:
    """One partials format's peer-store mailboxes: partial rows [2, DCP, CAP, H, 512]
    BF16 ("proj": BF16 value rows [..., 128]) and LSE [2, DCP, CAP, H] FP32,
    sentinel-filled, and their peers' addresses."""

    def __init__(self, group, device, size, cap, heads, fmt):
        assert fmt in FORMATS, fmt
        shape, empty = (2, size, cap, heads), symm_mem.empty
        self.format = fmt
        width = 128 if fmt == PROJECTED else 512
        self.partial = empty((*shape, width), dtype=torch.bfloat16, device=device)
        self.lse = empty(shape, dtype=torch.float32, device=device)
        # Peer stores only: no multicast object.
        handles = [peer_rendezvous(t, group) for t in (self.partial, self.lse)]
        i64 = dict(dtype=torch.int64, device=device)
        self.peers = torch.tensor(handles[0].buffer_ptrs, **i64)
        self.lse_peers = torch.tensor(handles[1].buffer_ptrs, **i64)
        self.partial.view(torch.int32).fill_(PAIR_SENT - (1 << 32))
        self.lse.view(torch.int32).fill_(-1)


class DcpBox:
    """Model-owned DCP state over `group`, Lamport mailboxes (sentinel-filled,
    re-armed by their reader): q [2, CAP, DCP*H, 576] FP8 bytes (multicast stores),
    one DcpMailbox of partial rows and LSE (peer stores) per partials format the
    `partials` policy uses (see partials_for: "bf16" latent rows; "proj": value
    rows), and the local tokenspeed rows / LSE.
    Parity alternates by layer. `interleave`: the KV cache's DCP interleave (global
    masking of spec verify rows)."""

    def __init__(
        self,
        group,
        device,
        size,
        rank,
        heads,
        cap=64,
        interleave=1,
        q_heads=None,
        q_bf16=False,
        partials="bf16",
    ):
        self.size, self.rank, self.heads, self.cap = size, rank, heads, cap
        self.interleave, self.partials, self.group = interleave, partials, group
        # DSA: q_heads query rows per token (the sparse decode's head padding), BF16
        # for FlashMLA sparse, FP8 bytes for FlashInfer.
        self.q_heads = q_heads or size * heads
        q_dtype = torch.bfloat16 if q_bf16 else torch.uint8
        self.q = symm_mem.empty(
            (2, cap, self.q_heads, 576), dtype=q_dtype, device=device
        )
        _, self.q_mc = multicast_rendezvous(self.q, group, "Kimi K3 DCP q mailbox")
        if not self.q_mc:
            raise RuntimeError("Kimi K3 specialized MLA DCP needs multicast memory")
        formats = sorted({partials_for(partials, b) for b in range(1, cap + 1)})
        self.mailboxes = {
            fmt: DcpMailbox(group, device, size, cap, heads, fmt) for fmt in formats
        }
        self.q.zero_()  # padded heads stay zero
        self.q[:, :, : size * heads].view(torch.uint8).fill_(0xFF)
        rows = (cap, 1, size * heads)
        self.rows = torch.empty((*rows, 512), dtype=torch.bfloat16, device=device)
        self.rows_lse = torch.empty(rows, dtype=torch.float32, device=device)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def mailbox(self, batch):
        """The DcpMailbox a `batch` of rows exchanges its partials through."""
        return self.mailboxes[partials_for(self.partials, batch)]

    @property
    def projected(self):
        """Whether some batch exchanges W_UV-projected rows ("proj")."""
        return PROJECTED in self.mailboxes

    def gather_heads(self, w_uv):
        """Every rank's W_UV [H, 512, 128] (latent-contiguous) -> [DCP * H, 512, 128]
        in the same layout, gathered head block d = rank d's heads ("proj" sources
        project the rows of every gathered head). Collective: call on every rank."""
        heads = w_uv.transpose(1, 2).contiguous()  # [H, 128, 512]
        gathered = heads.new_empty((self.size * heads.shape[0], *heads.shape[1:]))
        torch.distributed.all_gather_into_tensor(gathered, heads, group=self.group)
        return gathered.transpose(1, 2)

    def value_weights(self, w_uv):
        """The gathered W_UV (gather_heads) a layer's "proj" batches read, with its
        CuTe view, or None when the policy never exchanges "proj" rows. A copy of
        every rank's W_UV per layer: DCP x the local one (BF16 DCP * H * 512 * 128:
        12 MiB per Kimi K3 / K3.1 attention layer of 96 heads), taken once.
        Collective: the layer providers call it from their constructors, which
        build_providers runs on every rank over the same layers in the same order,
        after the weights are loaded and before the KV cache is profiled."""
        if not self.projected:
            return None
        gathered = self.gather_heads(w_uv)
        return gathered, to_cute(gathered)

    def mc(self, parity):
        """The q mailbox multicast address of `parity`."""
        return Int64(self.q_mc + parity * self.q[0].numel() * self.q.element_size())


class DcpValue:
    """The DCP tail of one MLA layer, one 512-thread CTA per (head, 16 requests).
    Every rank re-arms this parity's q mailbox (tokenspeed has read it) and pushes its
    local tokenspeed rows (+ LSE) to their head owner's Lamport mailbox; a row of an
    empty local shard (or an invalid LSE) sends only the LSE EMPTY_LSE, and the owner
    neither polls nor re-arms its partial. Warp r then LSE-weights request t0 + r of
    this head over the DCP sources (exp2(lse - max) / sum in FP32), the CTA merges the
    polled rows in FP32 (one BF16 rounding, re-arming what it read) into shared
    memory, and W_UV (BF16 MMA, FP32 accumulation) and the sigmoid gate follow as in
    MlaValue: warp w owns outputs 32 (w // 4) + 8 (w % 4) .. + 7. Rows are tokens,
    `q` per request. Above 64 rows a merging CTA owns every `grid[1]`-th 16-row
    tile: at most one CTA per SM, so every CTA's pushes land while the merging CTAs
    poll.

    `per_row`: the emptiness test reads `lens` per row (a sparse decode's valid
    counts of the local shard), else per request (the local lengths).

    `partials` "proj" (`w_uv`: the gathered [DCP * H, 512, 128] W_UV, see
    DcpBox.gather_heads): CTA (h, d < DCP) projects the local rows of gathered head
    d * H + h through its W_UV (steps of up to PROJ_ROWS rows staged in shared
    memory, the BF16 MMAs of the merge; the next step's latents and LSEs load while
    one projects) and pushes BF16 [rows, 128] value rows; CTA (h, DCP + m) merges
    request tiles m, m + M, .. of head h over the DCP sources (FP32, one BF16
    rounding: the merge is linear, sum_s w_s (O_s W) = (sum_s w_s O_s) W, so only
    the rounding points differ from the latent merge) and applies the gate.
    No CTA waits on a CTA that may not be dispatched: projecting CTAs wait on
    nothing but the sparse decode before them (griddepcontrol_wait); merging CTAs
    poll only values and LSEs the projecting CTAs of every rank push. Those have
    the lowest linear block indices, so the scheduler dispatches them before any
    merging CTA, and independently the grid is at most one CTA per SM, while
    everything else that can hold SMs during DcpValue (the PDL-overlapped sparse
    decode, the prefetch stream) finishes without waiting on it, and the next
    kernel only launches once every CTA here passed its PDL trigger, which the
    projecting CTAs reach after their projections. M fills the SMs left by the
    DCP * H projecting CTAs.

    `rows_per_cta` R (`pushed` only; VALUE_ROWS by default): one CTA per (head,
    R request rows). Row slot r of a CTA is merged by threads 128 r .. 128 r + 127
    (4 columns, 8 bytes each: the sparse decode's store width), which issue every
    source's load of the row right after the PDL wait while warp 4 r polls the
    row's LSEs; then the merge in source order (FP32, one BF16 rounding), W_UV (the
    16-row MMAs above; only rows 0 .. R - 1 are written and stored, an MMA row's
    outputs depend on its own inputs only) and the gate, so every value goes
    through DcpValue's operations in its order. As above, no CTA
    waits on another CTA of this grid: each waits on the sparse decode before it
    (griddepcontrol_wait) and polls only what the sparse decodes of every rank push,
    and those never wait on DcpValue."""

    def __init__(
        self,
        heads,
        size,
        batch,
        sms=148,
        q=1,
        q_chunks=36,
        q_heads=0,
        lse_scale=1.0,
        per_row=False,
        partials="bf16",
        preprojected=False,
        pushed=False,
        rows_per_cta=None,
    ):
        """`q_chunks`: 16 B chunks per q mailbox head row (36 FP8, 72 BF16);
        `q_heads`: q mailbox rows per token (default DCP * H); `lse_scale`: to
        base 2 (LOG2E for a natural-log LSE). `preprojected` ("proj" only): the
        W_UV argument holds the value rows [DCP * H, B, 128] BF16 a batched GEMM
        already projected (the stock W_UV bmm over the gathered heads), so the
        projecting CTAs only push them. `pushed` ("bf16" only): the sparse
        decode before pushed every rank's rows and LSEs and re-armed the q mailbox
        (dsa.attn_sparse), so DcpValue only merges; `rows_per_cta` (`pushed`
        only): request rows per merging CTA (None: VALUE_ROWS of the DCP size and
        batch; 0: one CTA per (head, 16 rows))."""
        self.heads, self.size, self.batch = heads, size, batch
        assert partials in FORMATS, partials
        self.q, self.per_row, self.partials = q, per_row, partials
        self.projected = partials == PROJECTED
        self.preprojected = preprojected and self.projected
        assert not (pushed and self.projected), partials
        self.pushed = pushed
        self.rows_per_cta = 0  # pushed only (below)
        # Rows with nothing to merge send only their LSE (neither polled nor
        # re-armed): "proj", and BF16 for a sparse decode's per-row counts. The MLA
        # BF16 layout (rarely empty: padded requests, short contexts) sends zeros
        # and polls every row, which keeps its merge loop branch-free.
        self.skip_empty = per_row or self.projected
        self.q_chunks, self.q_heads = q_chunks, q_heads or size * heads
        self.lse_scale = lse_scale
        # Push-only CTAs (z > 0) spread the NVLink stores over more SMs: up to one
        # wave, at least 4 16 B chunks per thread.
        self.tiles = tiles = (batch + 15) // 16
        if self.projected:
            # Rows each projecting step stages and pushes (16-row MMA blocks).
            self.prows = min(PROJ_ROWS, tiles * 16)
            merge = min(tiles, max(1, (sms - size * heads) // heads))
            self.grid = (heads, size + merge, 1)
            # One CTA per SM at most (the progress argument above).
            assert heads * (size + merge) <= sms, (self.grid, sms)
            return
        if batch > 64:
            tiles = min(tiles, max(1, sms // heads))
        merge = heads * tiles
        chunks = batch * size * heads * 64
        pushers = min(max(1, sms // merge), -(-chunks // (merge * 512 * 4)))
        if pushed:
            assert per_row  # the sparse decode's per-row counts
            pushers = 1
        self.grid = (heads, tiles, pushers)
        if pushed and rows_per_cta is None:
            rows_per_cta = batch_table(VALUE_ROWS.get(size, "")).get(batch, 0)
        if pushed and rows_per_cta:
            # 128 threads per row slot; at most 64 rows (16-row tiles above).
            assert 1 <= rows_per_cta <= 4 and batch <= 64, (rows_per_cta, batch)
            self.rows_per_cta = rows_per_cta
            self.grid = (heads, -(-batch // rows_per_cta), 1)

    @cute.jit
    def __call__(self, *args):
        kernel = self.kernel_projected if self.projected else self.kernel
        kernel = self.kernel_rows if self.rows_per_cta else kernel
        launch(kernel, args, self.grid, 512)

    @cute.jit
    def rearm(self, mailbox, me, threads):
        """Re-arm this parity's q mailbox (the sparse decode has read it): thread
        `me` of `threads` stores the sentinel into every threads-th 16 B chunk of
        the published rows."""
        H, D, B = self.heads, self.size, self.batch
        C, P = self.q_chunks, self.q_heads
        for c in range(me, B * D * H * C, threads):  # 16 B q mailbox chunks
            row = c // (D * H * C) * P * C + c % (D * H * C)
            store_words(mailbox.iterator + row * 16, (Uint32(Q_SENT),) * 4)

    @cute.kernel
    def kernel(self, args):
        rows, rows_lse, lens, box, lbox, peers, lpeers, mailbox = args[0:8]
        w_uv, gate, totals, output, parity, rank = args[8:14]
        D = self.size
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (16, 544), 128)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, pusher = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.pushed):
            self.push_rows(args, head, tile, pusher, tid)
        cute.arch.griddepcontrol_launch_dependents()
        if pusher == 0:
            if cutlass.const_expr(self.grid[1] == self.tiles):
                self.merge(args, latent, scales, wb, tid, tile)
            else:
                for t in range(tile, self.tiles, self.grid[1]):
                    self.merge(args, latent, scales, wb, tid, t)

    @cute.kernel
    def kernel_rows(self, args):
        """`pushed` with `rows_per_cta` R: CTA (head, m) merges request rows
        R m .. R m + R - 1 of this head (see the class docstring)."""
        w_uv, gate, totals, output = args[8], args[9], args[10], args[11]
        R, B = self.rows_per_cta, self.batch
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (16, 544), 128)
        scales = alloc(smem, Float32, (R, self.size))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        wb = [
            load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            for c in range(16)
        ]
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        t0 = tile * R
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gate_word = Uint32(0)
        if (group < R) & (t0 + group < B):
            gate_word = load_words(at(gate, t0 + group, col), count=1, mutable=True)[0]
        for r in cutlass.range_constexpr(R):
            if (tid // 128 == r) & (t0 + r < B):
                self.merge_row(args, latent, scales, tid, t0 + r, r)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(2)]
        if (group < R) & (t0 + group < B):  # rows 0 .. R - 1 of the MMA tile
            gates = unpack_bf16(gate_word)
            # Padded rows may hold a NaN gate; keep inactive rows zero.
            lo, hi = Float32(0), Float32(0)
            if totals[(t0 + group) // self.q] > 0:
                lo, hi = (
                    Float32(BFloat16(acc[e])) * sigmoid(gates[e], False)
                    for e in range(2)
                )
            store_words(at(output, t0 + group, col), (bf16x2(lo, hi),))

    @cute.jit
    def merge_row(self, args, latent, scales, tid, t, r: cutlass.Constexpr[int]):
        """`rows_per_cta`: request row `t` of this head (row slot `r`) over the
        DCP sources, with merge's weights and order, into MMA row `r` of
        `latent`; threads 128 r .. 128 r + 127, 4 columns each."""
        box, lbox = args[3], args[4]
        D = self.size
        head, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        c4 = 4 * (tid - 128 * r)
        # Every source's 8 bytes of the row in flight while warp 4 r polls the LSEs.
        sources = [at(box, s, t, head, c4) for s in range(D)]
        loads = [load_volatile(source, count=2) for source in sources]
        if warp == 4 * r:  # the row's source weights (as merge's warp r)
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t, head)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            # As dcp_direct_a2a_lse_reduce: weight * (1 / sum) (IEEE reciprocal).
            inverse = Float32(0)
            if total > 0:
                inverse = div_rn(Float32(1), total)
            scale = weight * inverse
            if lse <= Float32(EMPTY_LSE):
                scale = Float32(-1)  # no partial row: neither polled nor re-armed
            if lane < D:
                scales[r, lane] = scale
        named_barrier(1 + r, 128)  # the weights, for this slot's 4 warps
        merged = zeros(4, Float32)
        for s in cutlass.range_constexpr(D):
            w = scales[r, s]
            if w >= Float32(0):
                self.poll_pair(sources[s], loads[s], w, merged)
        pairs = tuple(
            pack_pair(merged[2 * e], merged[2 * e + 1], policy=None) for e in range(2)
        )
        store_shared_words(latent.iterator + r * 544 + c4, pairs)

    @cute.jit
    def push_rows(self, args, head, tile, pusher, tid):
        """Every rank's local rows (+ LSE) to their head owner's mailbox, then the
        q mailbox re-arm (see the class docstring)."""
        rows, rows_lse, lens, box, lbox, peers, lpeers, mailbox = args[0:8]
        parity, rank = args[12], args[13]
        H, D, B, cap = self.heads, self.size, self.batch, box.shape[1]
        threads = H * self.grid[1] * self.grid[2] * 512
        me = ((pusher * self.grid[1] + tile) * H + head) * 512 + tid
        for i in range(me, B * D * H * 64, threads):
            r, k = i // 64, i % 64  # r = b * D * H + global head
            b, gh = r // (D * H), r % (D * H)
            sent = self.sent_lse(rows_lse, lens, b, gh) > Float32(EMPTY_LSE)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            address = peers[gh // H] + (Int64(slot) * 512 + 8 * k) * 2
            if cutlass.const_expr(self.skip_empty):
                if sent:
                    words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
                    store_relaxed(address, tuple(no_neg_zero(w) for w in words))
            else:  # zeros for an empty row: every row is polled
                words = (Uint32(0),) * 4
                if sent:
                    words = load_words(at(rows, b, 0, gh, 8 * k), count=4, mutable=True)
                store_relaxed(address, tuple(no_neg_zero(w) for w in words))
        for i in range(me, B * D * H, threads):
            b, gh = i // (D * H), i % (D * H)
            value = self.sent_lse(rows_lse, lens, b, gh)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(lpeers[gh // H] + Int64(slot) * 4, (float_bits(value),))
        # The peer stores leave first; the local re-arm overlaps their flight.
        self.rearm(mailbox, me, threads)

    @cute.kernel
    def kernel_projected(self, args):
        """`partials` "proj": the projecting CTAs (y < DCP) and the merging CTAs
        (y >= DCP). The projecting CTAs come first in launch order and never wait
        on another CTA (only griddepcontrol_wait on the sparse decode before them),
        so every rank's projections finish however many merging CTAs are
        resident; see the class docstring."""
        mailbox, w_uv = args[7], args[8]
        H, D, R = self.heads, self.size, self.prows
        merges: cutlass.Constexpr = self.grid[1] - D
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (R, 544), 128)
        values = alloc(smem, BFloat16, (R, 136), 16)
        scales = alloc(smem, Float32, (16, D))
        tid, _, _ = cute.arch.thread_idx()
        head, y, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        threads: cutlass.Constexpr = H * self.grid[1] * 512
        me = (y * H + head) * 512 + tid
        if y < D:
            gh = y * H + head
            if cutlass.const_expr(self.preprojected):
                cute.arch.griddepcontrol_wait()
                self.push_values(args, gh, tid)
            else:
                v = 32 * (warp // 4) + 8 * (warp % 4) + group
                wb = [
                    load_words(at(w_uv, gh, 32 * c + 8 * quarter, v), count=4)
                    for c in range(16)
                ]
                cute.arch.griddepcontrol_wait()
                # Software pipeline: step t + 1's latents and LSEs load while
                # step t projects and pushes.
                ahead = self.load_tile(args, gh, 0, tid)
                for t0 in range(0, self.batch, R):
                    ahead = self.project(args, latent, values, wb, gh, t0, tid, ahead)
            self.rearm(mailbox, me, threads)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            cute.arch.griddepcontrol_wait()
            self.rearm(mailbox, me, threads)
            cute.arch.griddepcontrol_launch_dependents()
            for t in range(y - D, self.tiles, merges):
                self.merge_projected(args, scales, tid, t)

    @cute.jit
    def push_values(self, args, gh, tid):
        """`preprojected`: push gathered head `gh`'s value rows (the W_UV
        argument, [DCP * H, B, 128] BF16) and every row's LSE to the head's owner,
        a row with nothing to merge only its LSE (as `project`). Thread tid
        takes 16 B chunk tid % 16 of rows tid // 16, + 32, ..: four rows' loads in
        flight per thread."""
        rows_lse, lens, box, _, peers, lpeers = args[1:7]
        values, parity, rank = args[8], args[12], args[13]
        H, D, B = self.heads, self.size, self.batch
        cap = box.shape[1]
        chunk, owner = tid % 16, gh // H
        per: cutlass.Constexpr = 4  # rows per thread per round
        for r0 in range(tid // 16, B, 32 * per):
            loads = []
            for i in cutlass.range_constexpr(per):
                t = r0 + 32 * i
                lse = Float32(EMPTY_LSE)
                words = (Uint32(0),) * 4
                if t < B:
                    lse = self.sent_lse(rows_lse, lens, t, gh)
                    if lse > Float32(EMPTY_LSE):
                        words = load_words(
                            at(values, gh, t, 8 * chunk), count=4, mutable=True
                        )
                loads.append((t, lse, words))
            for t, lse, words in loads:
                if t < B:
                    slot = ((parity * D + rank) * cap + t) * H + gh % H
                    if lse > Float32(EMPTY_LSE):
                        target = peers[owner] + (Int64(slot) * 128 + 8 * chunk) * 2
                        store_relaxed(target, tuple(no_neg_zero(w) for w in words))
                    if chunk == 0:
                        address = lpeers[owner] + Int64(slot) * 4
                        store_relaxed(address, (float_bits(lse),))

    @cute.jit
    def load_tile(self, args, gh, t0, tid):
        """This thread's 16 B chunks of the BF16 latent rows t0.. t0 + prows - 1 of
        gathered head `gh` (chunk tid + 512 j: row // 64, 8 (chunk % 64)); zeros
        beyond the batch), led by the LSEs of the rows it pushes ((tid + 512 j)
        // 16; sent_lse, EMPTY_LSE beyond the batch)."""
        rows, rows_lse, lens = args[0:3]
        R = self.prows
        chunks = []
        for j in cutlass.range_constexpr((16 * R + 511) // 512):  # pushed rows
            lse = Float32(EMPTY_LSE)
            row = (tid + 512 * j) // 16
            if (row < R) & (t0 + row < self.batch):
                lse = self.sent_lse(rows_lse, lens, t0 + row, gh)
            chunks.append(lse)
        for j in cutlass.range_constexpr(R // 8):
            r, k = (tid + 512 * j) // 64, (tid + 512 * j) % 64
            words = (Uint32(0),) * 4
            if t0 + r < self.batch:
                words = load_words(
                    at(rows, t0 + r, 0, gh, 8 * k), count=4, mutable=True
                )
            chunks.append(words)
        return tuple(chunks)

    @cute.jit
    def project(self, args, latent, values, wb, gh, t0, tid, tile):
        """Rows t0.. t0 + prows - 1 of gathered head `gh` (`tile`: this
        thread's latent chunks and LSE of them, load_tile): stage the BF16 latents,
        issue the next step's loads, W_UV (BF16 MMA, FP32 accumulation, rounded to
        BF16; one 16-row block per accumulator set) and push the [prows, 128] value
        rows (and every row's LSE; a row with nothing to merge sends only its LSE).
        Returns the next step's load_tile."""
        box, _, peers, lpeers = args[3:7]
        parity, rank = args[12], args[13]
        H, D, B, R = self.heads, self.size, self.batch, self.prows
        cap = box.shape[1]
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        cute.arch.sync_threads()  # the previous step's readers are done
        pushes = (16 * R + 511) // 512  # the LSEs lead the tile (load_tile)
        for j in cutlass.range_constexpr(R // 8):
            r, k = (tid + 512 * j) // 64, (tid + 512 * j) % 64
            store_shared_words(latent.iterator + r * 544 + 8 * k, tile[pushes + j])
        cute.arch.sync_threads()
        ahead = self.load_tile(args, gh, t0 + R, tid)
        col = 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        for g in cutlass.range_constexpr(R // 16):
            parts = zeros((2, 4), Float32)  # two accumulation chains
            for c in cutlass.range_constexpr(16):
                x, y = (
                    load_words(
                        latent.iterator + (16 * g + r) * 544 + 32 * c + 8 * quarter,
                        count=4,
                        shared=True,
                    )
                    for r in (group, group + 8)
                )
                for s in cutlass.range_constexpr(2):
                    a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                    b = (wb[c][2 * s], wb[c][2 * s + 1])
                    mma_into(parts[s, None], a, b, bf16=True)
            for r in cutlass.range_constexpr(2):
                pair = bf16x2(
                    parts[0, 2 * r] + parts[1, 2 * r],
                    parts[0, 2 * r + 1] + parts[1, 2 * r + 1],
                )
                address = values.iterator + (16 * g + group + 8 * r) * 136 + col
                store_shared_words(address, (no_neg_zero(pair),))
        cute.arch.sync_threads()
        for j in cutlass.range_constexpr(pushes):
            row, chunk = (tid + 512 * j) // 16, tid % 16
            t = t0 + row
            if (row < R) & (t < B):
                lse = tile[j]
                slot = ((parity * D + rank) * cap + t) * H + gh % H
                owner = gh // H
                if lse > Float32(EMPTY_LSE):
                    sent = load_words(
                        values.iterator + row * 136 + 8 * chunk, count=4, shared=True
                    )
                    target = peers[owner] + (Int64(slot) * 128 + 8 * chunk) * 2
                    store_relaxed(target, sent)
                if chunk == 0:
                    store_relaxed(lpeers[owner] + Int64(slot) * 4, (float_bits(lse),))
        return ahead

    @cute.jit
    def merge_projected(self, args, scales, tid, tile):
        """The owner's merge of the value rows of request rows t0.. of this head over
        the DCP sources (FP32, polled 8 at a time, re-armed as read), rounded to
        BF16 as the latent merge's W_UV output, then the gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        t0 = tile * 16
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t0 + warp, head)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            # As dcp_direct_a2a_lse_reduce: weight * (1 / sum) (IEEE reciprocal).
            inverse = Float32(0)
            if total > 0:
                inverse = div_rn(Float32(1), total)
            scale = weight * inverse
            if lse <= Float32(EMPTY_LSE):
                scale = Float32(-1)  # no value row: neither polled nor re-armed
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        if tid < 256:
            r, chunk = tid // 16, tid % 16
            t = t0 + r
            if t < B:
                merged = zeros(8, Float32)
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, t, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        w = scales[r, s0 + i]
                        if w >= Float32(0):
                            self.poll_pair(sources[i], loads[i], w, merged)
                col = head * 128 + 8 * chunk
                gates = load_words(at(gate, t, col), count=4, mutable=True)
                words = []
                for e in cutlass.range_constexpr(4):
                    # Padded rows may hold a NaN gate; keep inactive rows zero.
                    lo, hi = Float32(0), Float32(0)
                    if totals[t // self.q] > 0:
                        g = unpack_bf16(gates[e])
                        lo, hi = (
                            Float32(BFloat16(merged[2 * e + k])) * sigmoid(g[k], False)
                            for k in range(2)
                        )
                    words.append(bf16x2(lo, hi))
                store_words(at(output, t, col), tuple(words))
        cute.arch.sync_threads()  # the next tile rewrites the weights

    @cute.jit
    def sent_lse(self, rows_lse, lens, b, gh):
        """The base-2 LSE row (b, gh) sends: EMPTY_LSE for an empty local shard or a
        NaN / infinite LSE (then its partial row is not sent)."""
        value = rows_lse[b, 0, gh] * Float32(self.lse_scale)
        n = lens[b] if cutlass.const_expr(self.per_row) else lens[b // self.q]
        bad = (value != value) | (value <= Float32(EMPTY_LSE))  # NaN, -inf
        if (n <= 0) | bad | (value >= Float32(-EMPTY_LSE)):  # or +inf
            value = Float32(EMPTY_LSE)
        return value

    @cute.jit
    def merge(self, args, latent, scales, wb, tid, tile):
        """The owner's merge of request rows t0.. of this head, then W_UV + gate."""
        box, lbox, gate, totals, output = args[3], args[4], args[9], args[10], args[11]
        D, B = self.size, self.batch
        head, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = tile * 16
        col = head * 128 + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gates = []
        for r in cutlass.range_constexpr(2):
            word = Uint32(0)
            if t0 + group + 8 * r < B:
                word = load_words(
                    at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                )[0]
            gates.append(unpack_bf16(word))
        if t0 + warp < B:  # warp r: the source weights of request t0 + r
            lse = Float32(EMPTY_LSE)
            if lane < D:
                source = at(lbox, lane, t0 + warp, head)
                word = load_volatile(source)[0]
                while word == Uint32(LSE_SENT):
                    word = load_volatile(source)[0]
                store_words(source, (Uint32(LSE_SENT),))
                lse = bits_float(word)
            top = lse
            for shift in cutlass.range_constexpr(4, -1, -1):
                top = max(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
            weight = Float32(0)
            if lse > Float32(EMPTY_LSE):
                weight = cute.math.exp2(lse - top, fastmath=True)
            total = warp_sum(weight)
            # As dcp_direct_a2a_lse_reduce: weight * (1 / sum) (IEEE reciprocal).
            inverse = Float32(0)
            if total > 0:
                inverse = div_rn(Float32(1), total)
            scale = weight * inverse
            if cutlass.const_expr(self.skip_empty):  # noqa: SIM102 (compile time)
                if lse <= Float32(EMPTY_LSE):
                    scale = Float32(-1)  # no partial row: neither polled nor re-armed
            if lane < D:
                scales[warp, lane] = scale
        cute.arch.sync_threads()
        chunk = tid % 64
        for half in cutlass.range_constexpr(2):  # rows r and r + 8, 16 B chunk
            r = tid // 64 + 8 * half
            merged = zeros(8, Float32)
            if t0 + r < B:
                for s0 in cutlass.range_constexpr(0, D, 8):  # 8 polls in flight
                    sources = [
                        at(box, s, t0 + r, head, 8 * chunk)
                        for s in range(s0, min(D, s0 + 8))
                    ]
                    loads = [load_volatile(source, count=4) for source in sources]
                    for i in cutlass.range_constexpr(len(sources)):
                        w = scales[r, s0 + i]
                        if cutlass.const_expr(self.skip_empty):
                            if w >= Float32(0):
                                self.poll_pair(sources[i], loads[i], w, merged)
                        else:
                            self.poll_pair(sources[i], loads[i], w, merged)
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            store_shared_words(latent.iterator + r * 544 + 8 * chunk, pairs)
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * 544 + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
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
            if t < B:
                # Padded rows may hold a NaN gate; keep inactive rows zero.
                lo, hi = Float32(0), Float32(0)
                if totals[t // self.q] > 0:
                    lo, hi = (
                        Float32(BFloat16(acc[2 * r + e])) * sigmoid(gates[r][e], False)
                        for e in range(2)
                    )
                store_words(at(output, t, col), (bf16x2(lo, hi),))

    @cute.jit
    def poll_pair(self, source, words, w, merged):
        """Polls a BF16 partial chunk (2 len(words) values) until published, re-arms
        it and adds it, weighted by `w`, to `merged`."""
        n: cutlass.Constexpr = len(words)
        while not published(words):
            words = load_volatile(source, count=n)
        store_words(source, (Uint32(PAIR_SENT),) * n)
        for e in cutlass.range_constexpr(n):
            lo, hi = unpack_bf16(words[e])
            merged[2 * e] += w * lo
            merged[2 * e + 1] += w * hi


@cache
def compile_value(
    heads,
    size,
    batch,
    cap,
    uv,
    pitch,
    device,
    sms=148,
    q=1,
    dsa_heads=0,
    dsa_bf16=True,
    lse_base_e=True,
    partials="bf16",
    preprojected=False,
    pushed=False,
    rows_per_cta=None,
):
    """DcpValue of one batch (`q` rows per request); `uv`: the (shape, stride,
    dtype) of W_UV, `pitch`: the projected rows' (the gate follows the 2112 query /
    KV columns). `dsa_heads`: DSA layout, its padded head count: q mailbox rows
    (BF16 for FlashMLA sparse, else FP8), rows / LSE as the sparse decode returns
    them (`lse_base_e`: natural-log, else base 2) and `lens` the sparse decode's
    per-row valid counts. `partials`: the format of its DcpMailbox;
    `rows_per_cta`: DcpValue's (`pushed`)."""
    f32, i32 = torch.float32, torch.int32
    w_uv, gate = fake(*uv), fake((batch, heads * 128), (pitch, 1))
    rows = fake((batch, 1, size * heads, 512))
    rows_lse = fake((batch, 1, size * heads), None, f32, 4)
    mailbox = fake((cap, size * heads, 576), None, torch.uint8)
    lens = fake((batch // q,), None, i32, 4)
    value = DcpValue(heads, size, batch, sms, q, partials=partials)
    if dsa_heads:
        rows = fake((batch, 1, size * heads, 512), (dsa_heads * 512, 0, 512, 1))
        rows_lse = fake((batch, 1, size * heads), (dsa_heads, 0, 1), f32, 4)
        row_bytes = 1152 if dsa_bf16 else 576
        mailbox = fake((cap, dsa_heads, row_bytes), None, torch.uint8)
        lens = fake((batch,), None, i32, 4)
        scale = math.log2(math.e) if lse_base_e else 1.0
        chunks = row_bytes // 16
        value = DcpValue(
            heads,
            size,
            batch,
            sms,
            q,
            chunks,
            dsa_heads,
            scale,
            True,
            partials,
            preprojected,
            pushed,
            rows_per_cta,
        )
    totals = fake((batch // q,), None, i32, 4)
    width = 512
    if partials == PROJECTED:  # the gathered W_UV (DcpBox.gather_heads) and values
        gathered = size * heads
        w_uv, width = fake((gathered, 512, 128), (128 * 512, 1, 512)), 128
        if preprojected:  # the bmm's value rows instead
            w_uv = fake((gathered, batch, 128), (batch * 128, 128, 1))
    box = fake((size, cap, heads, width), None, torch.bfloat16)
    lbox = fake((size, cap, heads), None, f32, 4)
    peers = fake((size,), None, torch.int64, 8)
    output = fake((batch, heads * 128), (heads * 128, 1))
    args: tuple = (rows, rows_lse, lens, box, lbox, peers, peers, mailbox, w_uv)
    args += (gate, totals, output, Int32(0), Int32(0), make_fake_stream())
    with torch.accelerator.device_index(device):
        return cute.compile(value, *args)


_WORKSPACE: dict[tuple[torch.device, int], torch.Tensor] = {}


def split_plan(batch, q_len, heads, device, max_seq, min_split, pack):
    """tokenspeed_mla_decode's FP8 schedule of `batch` requests of `q_len` rows:
    (tiler, pack, fold, SMs, split count, workspace bytes)."""
    from tokenspeed_mla import mla_decode as md

    arch = torch.cuda.get_device_capability(device)
    tiler, _ = md.select_mla_decode_tilers(
        heads, q_len, is_fp8=True, compute_capability=arch
    )
    pack = pack and tiler[0] == 128
    fold = 1 if pack else md.get_mla_decode_fold_sq_factor(heads, q_len, tiler[0])
    rows = (q_len // fold, heads * fold)
    if pack:
        rows = (md.compute_q_tile_layout(heads, q_len, 128)[1], 128)
    sms = md.get_num_sm(device)
    split, size = md._get_split_kv_and_workspace_size(
        batch, *rows, 512, sms, max_seq, torch.float8_e4m3fn, tiler, min_split
    )
    return tiler, pack, fold, sms, split, size


def reserve_workspace(batch, q_len, heads, device, max_seq, min_split, pack=False):
    """The split-KV workspace of one stock_decode shape (one per (device, heads),
    shared by every MLA layer: the calls are stream ordered), grown to fit it.
    PreparedMla reserves every prepared shape's at construction, before vLLM profiles
    its memory for the KV cache, so prepare_cache and the steps allocate nothing;
    growing it during a capture would leave the graphs captured before with the
    freed buffer."""
    device = torch.device(device)
    plan = split_plan(batch, q_len, heads, device, max_seq, min_split, pack)
    return _workspace(device, heads, plan[3], plan[5])


def _workspace(device, heads, sms, size):
    key = device, heads
    if key not in _WORKSPACE or _WORKSPACE[key].numel() < size:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the MLA split-KV workspace must be sized before capture"
            )
        bound = max(size, sms * heads * 513 * 4)
        _WORKSPACE[key] = torch.empty(bound, dtype=torch.int8, device=device)
    return _WORKSPACE[key]


def stock_decode(
    query,
    cache,
    block_table,
    lengths,
    max_seq,
    scales,
    out,
    lse=None,
    min_split=1,
    pdl=True,
    cp=(1, 0, 1),
    totals=None,
    pack=False,
):
    """tokenspeed_mla_decode's FP8 path with caller-owned `out` and `lse` (log2)
    buffers and no per-call allocation: the split count follows from (B, max_seq,
    min_split); `lengths` bound the work, empty splits are skipped by its reducer.
    The q rows of a request mask causally; under DCP in global positions (`cp`:
    world, rank, interleave; `totals`: the global lengths). `pack`: M128 tiles hold
    consecutive (query row, head) rows across the q rows."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    batch, q_len, heads, _ = query.shape
    fp8, device = torch.float8_e4m3fn, query.device
    arch = torch.cuda.get_device_capability(device)
    plan = split_plan(batch, q_len, heads, device, max_seq, min_split, pack)
    tiler, pack, fold, sms, split, size = plan
    # Reserved by PreparedMla at construction (reserve_workspace): no allocation.
    workspace = _workspace(device, heads, sms, size)
    kernel = md._get_compiled_mla_kernel(
        torch_dtype=fp8,
        page_size=cache.shape[1],
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        is_persistent=False,
        is_var_seq=True,
        is_var_split_kv=False,
        compute_capability=arch,
        has_local_visible_lens=False,
        skip_correction_threshold=8.0,
        is_workspace_size_zero=size == 0,
        fold_sq_factor=fold,
        causal_mask=True,
        num_heads=heads,
        seq_len_q=q_len,
        window_left=-1,
        cp_world=cp[0],
        cp_rank=cp[1],
        cp_interleave_size=cp[2],
        use_pdl=pdl,
        return_lse=lse is not None,
        reducer_d_tiles=md._get_reducer_d_tiles(batch, q_len, heads, sms, split),
        reducer_max_splits=md._get_reducer_max_splits(split),
        pack_q=pack,
        partial_fp16=False,
    )
    cache = cache.view(fp8)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, workspace[:size] if size else None]
    args += [Int32(split), lengths, lengths if totals is None else totals, None]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)


def fill_splits(batch, device):
    """Splits filling the SMs (one CTA pair each); tokenspeed caps itself at 32."""
    from tokenspeed_mla import get_num_sm

    return min(256, max(1, get_num_sm(device) // (2 * batch)))


def local_max(max_context, size):
    """Longest local shard of a `max_context` sequence (block-interleaved)."""
    return math.ceil(max_context / (size * 64)) * 64
