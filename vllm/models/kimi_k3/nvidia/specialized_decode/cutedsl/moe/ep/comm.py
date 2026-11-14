# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-parallel dispatch and combine of the latent MoE over NVLink symmetric memory
(DEP: data-parallel attention at TP1, the 896 routed experts split over N ranks).

Every rank owns one symmetric buffer (`EpLayout`):

- receive rows, source-major (row = source x stride(T) + token): MXFP8 latent (3584
  B), its E8M0 block scales (112 B, the linear layout of `flashinfer.mxfp8_quantize`),
  the token's 16 expert ids and FP32 routing weights, and a source word (one 128-byte
  line per row, -1 when empty);
- combine slots [T, N, 3584] BF16: the per-destination-rank partial of every own token;
- control words: arrival counters and the dispatch / combine flags (one line each).

`EpDispatch` (one CTA per token) quantizes the BF16 latent to MXFP8 bit-identically
to `mxfp8_quantize` (two threads per 32-block, 16-byte stores) and writes it, with
ids, weights and source, into its static row on every rank holding one of its
experts. Empty rows keep non-local ids (weight 0), so the expert GEMM over the N T
rows skips them. The last CTA of the rank raises its dispatch flag on every rank and
waits for all N; on completion the rank's receive rows are complete.

`EpCombine` (G CTAs) sends every received row's BF16 output (the routing-weighted
sum over this rank's experts) to its source's slot [token, rank] and re-arms the row;
the last CTA raises the combine flag on every rank. The CTAs with own tokens then wait
for all N flags and reduce: the FP32 sum of the destination partials in rank order,
rounded once to BF16 (the routed latent), then vLLM's RMSNorm of it (FP32 sum of
squares, rsqrt(mean + eps), x * rstd * w rounded once).

Measured on GB300 NVL72 (16 ranks over 4 nodes), these rules decide the latency:
no two ranks write one 128-byte line (row blocks padded to 8 rows, one line per flag
and per source word: 200 -> 40 us per dispatch + combine at T=5); a rank reaches its
own buffer through its local mapping, never its peer mapping; flags are polled by
one lane each (a local poll costs a round trip). Every thread that stored remotely
fences at system scope before the CTA's arrival is counted (a single fence after the
CTA barrier lost partials at 32 ranks), and every reader fences after the flags.

Flags are single-reader: a writer sets 1, the reader re-arms 0. No rank can raise the
next layer's flag before the reader re-armed it: the next dispatch from a source needs
the source's combine to finish, which needs this rank's combine flag, raised after
this rank's dispatch re-armed; the next combine flag needs this rank's next dispatch,
issued after its combine re-armed.
"""

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32

from .....specialized_decode.symm import peer_rendezvous
from ...common.primitives import (
    asm,
    launch,
    load_words,
    mx_exponent,
    mx_rescale,
    pack_fp8,
    pack_pair,
    store_words,
    unpack_bf16,
    warp_sum,
)

LATENT, TOPK, EXPERTS = 3584, 16, 896
THREADS = 224  # 16 latent elements per thread
SF_BYTES = LATENT // 32
FLAG = 128  # bytes between two flags


def _align(x, a=256):
    return (x + a - 1) // a * a


def stride(tokens):
    """Receive rows per source rank: a multiple of 8, so no 128-byte line of the
    receive arrays (3584 / 112 / 64-byte rows) holds rows of two sources (two
    remote writers of one line serialize on the fabric)."""
    return (tokens + 7) // 8 * 8


class EpLayout:
    """Byte offsets of one rank's symmetric EP buffer for `ranks` ranks and at most
    `tokens` tokens per rank."""

    def __init__(self, ranks, tokens):
        # Rank masks are 32-bit, one flag line per rank (32 slots), the source word
        # packs the token into 16 bits, and the experts split evenly.
        assert 1 <= ranks <= 32 and EXPERTS % ranks == 0, ranks
        assert 1 <= tokens < 1 << 16, tokens
        self.ranks, self.tokens = ranks, tokens
        self.rows = rows = ranks * stride(tokens)
        offset = 0

        def take(size):
            nonlocal offset
            start, offset = offset, _align(offset + size)
            return start

        self.x = take(rows * LATENT)
        self.sf = take(rows * SF_BYTES)
        self.ids = take(rows * TOPK * 4)
        self.weights = take(rows * TOPK * 4)
        self.src = take(rows * FLAG)  # one line per row
        self.comb = take(tokens * ranks * LATENT * 2)
        # (unused), tokens done, combine CTAs done, combine waiters that saw every
        # flag, then the dispatch and combine flags [32] each
        # (one 128-byte line per flag: the ranks' writes never share a line)
        self.ctrl = take(128 + 2 * 32 * FLAG)
        self.count, self.tdone, self.cdone, self.seen = (
            self.ctrl + 4 * i for i in range(4)
        )
        self.dflag, self.cflag = self.ctrl + 128, self.ctrl + 128 + 32 * FLAG
        self.bytes = offset


@asm(Int32)
def atomic_add_gpu(address, value):
    return (
        "atom.acq_rel.gpu.global.add.u32 $0,[$1],$2;",
        "=r,l,r,~{memory}",
        (
            Int64(address),
            Int32(value),
        ),
    )


@asm()
def fence_sys():
    return "fence.acq_rel.sys;", "~{memory}", ()


@asm()
def store_word(address, value):
    return (
        "st.relaxed.gpu.global.u32 [$0],$1;",
        "l,r,~{memory}",
        (
            Int64(address),
            Int32(value),
        ),
    )


@asm(Int32)
def load_relaxed(address):
    return "ld.relaxed.sys.global.u32 $0,[$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Int32)
def load_plain(address):
    return "ld.global.cg.u32 $0,[$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Uint32)
def warp_or(value):
    return "redux.sync.or.b32 $0,$1,0xffffffff;", "=r,r", (Uint32(value),)


@asm()
def store_sys(address, value):
    return (
        "st.relaxed.sys.global.u32 [$0],$1;",
        "l,r,~{memory}",
        (
            Int64(address),
            Int32(value),
        ),
    )


@cute.jit
def last_arrival(counter, total, lane):
    """Warp-uniform: whether this CTA is the `total`-th arrival on `counter`
    (lane 0's atomic after a system fence of the CTA's stores; the last arrival
    re-arms the counter). Every lane fences again so its later stores follow."""
    last = Int32(0)
    if lane == 0:
        fence_sys()
        last = Int32(atomic_add_gpu(counter, 1) == total - 1)
        if last != 0:
            store_word(counter, 0)
    last = cute.arch.shuffle_sync(last, 0)
    fence_sys()
    return last


@cute.jit
def raise_flags(peers, offset, rank, lane, ranks: cutlass.Constexpr[int]):
    """Lane d sets this rank's flag on rank d (all ranks in parallel)."""
    if lane < ranks:
        target = lane
        if lane == rank:
            target = ranks  # own flag through the local mapping
        store_sys(Int64(peers[target]) + offset + FLAG * rank, 1)


@cute.jit
def wait_flags(base, lane, ranks: cutlass.Constexpr[int]):
    """Warp: lane s polls flag s (relaxed, all in flight together) until every flag
    is set, then an acquire fence. Local polls cost a round trip each, so the ranks'
    flags are never polled one after another."""
    pending = Int32(lane < ranks)
    while warp_or(Uint32(pending)) != Uint32(0):
        if pending != 0:  # noqa: SIM102 (DSL: the load stays under the guard)
            if load_relaxed(base + FLAG * lane) != 0:
                pending = Int32(0)
    fence_sys()


@cute.jit
def rearm_flags(base, lane, ranks: cutlass.Constexpr[int]):
    if lane < ranks:
        store_word(base + FLAG * lane, 0)


class EpDispatch:
    """(x [T, 3584] BF16, ids [T, 16] int32, weights [T, 16] FP32, peers [N] int64) ->
    compact MXFP8 receive rows on the experts' ranks."""

    def __init__(self, layout: EpLayout, tokens: int, rank: int):
        self.layout, self.tokens, self.rank = layout, tokens, rank
        self.ranks = layout.ranks
        self.local = EXPERTS // layout.ranks

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.tokens, THREADS)

    @cute.kernel
    def kernel(self, args):
        x, ids, weights, peers = args
        lay, ranks, rank = self.layout, self.ranks, self.rank
        tid, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        rows = smem.allocate_tensor(Int32, cute.make_layout(33), 16)
        # Static receive slot of this token on every rank: source-major rows.
        slot = Int64(rank * stride(self.tokens) + token)
        cute.arch.griddepcontrol_wait()
        if tid < 32:
            bit = Uint32(0)
            if lane < TOPK:
                bit = Uint32(1) << Uint32(ids[token, lane] // self.local)
            mask = warp_or(bit)
            if lane == 0:
                rows[32] = Int32(mask)
        cute.arch.sync_threads()
        mask = Uint32(rows[32])
        # MXFP8 of elements [16 tid, 16 tid + 16): two threads per 32-block, one
        # 16-byte FP8 store per destination; lanes 8j hold 4-block scale words.
        values = cute.make_rmem_tensor(16, Float32)
        for i in cutlass.range_constexpr(2):
            words = load_words(
                x.iterator + token * LATENT + tid * 16 + 8 * i, count=4, mutable=True
            )
            for j in cutlass.range_constexpr(4):
                pair = unpack_bf16(words[j])
                values[8 * i + 2 * j], values[8 * i + 2 * j + 1] = pair[0], pair[1]
        top = abs(values[0])
        for j in cutlass.range_constexpr(1, 16):
            top = cute.arch.fmax(top, abs(values[j]))
        top = cute.arch.fmax(top, cute.arch.shuffle_sync_bfly(top, 1))
        scale = mx_exponent(top)
        reciprocal = mx_rescale(scale)
        packed = [
            pack_fp8(tuple(values[4 * j + k] * reciprocal for k in range(4)))
            for j in range(4)
        ]
        word = Uint32(scale)
        for s in cutlass.range_constexpr(1, 4):
            word |= cute.arch.shuffle_sync(Uint32(scale), lane + 2 * s) << Uint32(8 * s)
        for d in cutlass.range_constexpr(ranks):
            if (mask >> Uint32(d)) & Uint32(1) != Uint32(0):
                base = Int64(peers[ranks if d == rank else d])
                store_words(base + lay.x + slot * LATENT + tid * 16, packed)
                if tid % 8 == 0:
                    store_words(base + lay.sf + slot * SF_BYTES + tid // 2, (word,))
        if tid < TOPK + 1:
            for d in cutlass.range_constexpr(ranks):
                if (mask >> Uint32(d)) & Uint32(1) != Uint32(0):
                    row = slot
                    base = Int64(peers[ranks if d == rank else d])
                    if tid < TOPK:
                        store_words(
                            base + lay.ids + row * 64 + tid * 4,
                            (Uint32(ids[token, tid]),),
                        )
                        w = load_words(
                            weights.iterator + token * TOPK + tid, count=1, mutable=True
                        )
                        store_words(base + lay.weights + row * 64 + tid * 4, w)
                    else:
                        store_words(
                            base + lay.src + row * FLAG, (Uint32((rank << 16) | token),)
                        )
        fence_sys()  # every storing thread: its remote stores precede the arrival
        cute.arch.sync_threads()
        if tid < 32:
            local = Int64(peers[ranks])
            if last_arrival(local + lay.tdone, self.tokens, lane) != 0:
                raise_flags(peers, lay.dflag, rank, lane, ranks)
                wait_flags(local + lay.dflag, lane, ranks)
                rearm_flags(local + lay.dflag, lane, ranks)
        cute.arch.griddepcontrol_launch_dependents()


class EpCombine:
    """(out [N T, 3584] BF16 per received row, ids [T, 16] own routing, norm weight
    [3584] BF16, normed [T, 3584] BF16, peers [N] int64) -> the RMS-normalized routed
    latent of the rank's own tokens."""

    def __init__(self, layout: EpLayout, tokens: int, rank: int, grid: int, eps=1e-5):
        self.layout, self.tokens, self.rank, self.grid = layout, tokens, rank, grid
        self.ranks, self.eps = layout.ranks, eps
        self.local = EXPERTS // layout.ranks
        # Re-armed rows point at the next rank's first expert (never local).
        self.invalid = ((rank + 1) % layout.ranks) * self.local

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        out, ids, norm, normed, peers = args
        lay, ranks, rank, grid = self.layout, self.ranks, self.rank, self.grid
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        smem = cutlass.utils.SmemAllocator()
        part = smem.allocate_tensor(Float32, cute.make_layout(8), 16)
        masks = smem.allocate_tensor(Int32, cute.make_layout(1), 16)
        local = Int64(peers[ranks])  # this rank's own mapping of its buffer
        cute.arch.griddepcontrol_wait()
        r = cta
        while r < ranks * stride(self.tokens):
            src = load_plain(local + lay.src + Int64(r) * FLAG)
            # every warp has read the source word before warp 0 re-arms it
            cute.arch.sync_threads()
            if src != -1:
                source = src >> 16
                if source == rank:
                    source = Int32(ranks)
                dest = Int64(peers[source]) + lay.comb
                slot = Int64((src & 0xFFFF) * ranks + rank) * LATENT
                for i in cutlass.range_constexpr(2):
                    column = tid * 16 + 8 * i
                    words = load_words(
                        out.iterator + Int64(r) * LATENT + column, count=4, mutable=True
                    )
                    store_words(dest + (slot + column) * 2, words)
                if tid < TOPK:
                    store_word(local + lay.ids + Int64(r) * 64 + tid * 4, self.invalid)
                    store_word(local + lay.weights + Int64(r) * 64 + tid * 4, 0)
                if tid == TOPK:
                    store_word(local + lay.src + Int64(r) * FLAG, -1)
            r += grid
        fence_sys()  # every storing thread: its remote stores precede the arrival
        cute.arch.sync_threads()
        # Only the CTAs that reduce own tokens wait for the destinations.
        waiters = min(grid, self.tokens)
        if tid < 32:
            if last_arrival(local + lay.cdone, grid, lane) != 0:
                raise_flags(peers, lay.cflag, rank, lane, ranks)
            if cta < waiters:
                wait_flags(local + lay.cflag, lane, ranks)
                if last_arrival(local + lay.seen, waiters, lane) != 0:
                    rearm_flags(local + lay.cflag, lane, ranks)
        cute.arch.sync_threads()
        fence_sys()  # every reading thread acquires the destinations' partials
        t = cta
        while t < self.tokens:
            if tid < 32:
                bit = Uint32(0)
                if lane < TOPK:
                    bit = Uint32(1) << Uint32(ids[t, lane] // self.local)
                mask = warp_or(bit)
                if lane == 0:
                    masks[0] = Int32(mask)
            cute.arch.sync_threads()
            mask = Uint32(masks[0])
            values = cute.make_rmem_tensor(16, Float32)
            for j in cutlass.range_constexpr(16):
                values[j] = Float32(0)
            for d in cutlass.range_constexpr(ranks):
                if (mask >> Uint32(d)) & Uint32(1) != Uint32(0):
                    slot = Int64(t * ranks + d) * LATENT
                    for i in cutlass.range_constexpr(2):
                        column = tid * 16 + 8 * i
                        words = load_words(
                            local + lay.comb + (slot + column) * 2,
                            count=4,
                            volatile=True,
                        )
                        for j in cutlass.range_constexpr(4):
                            pair = unpack_bf16(words[j])
                            values[8 * i + 2 * j] += pair[0]
                            values[8 * i + 2 * j + 1] += pair[1]
            total = Float32(0)
            for j in cutlass.range_constexpr(8):
                pair = unpack_bf16(
                    pack_pair(values[2 * j], values[2 * j + 1], policy=None)
                )
                values[2 * j], values[2 * j + 1] = pair[0], pair[1]
                total += pair[0] * pair[0] + pair[1] * pair[1]
            total = warp_sum(total)
            if lane == 0:
                part[warp] = total
            cute.arch.sync_threads()
            total = Float32(0)
            for w in cutlass.range_constexpr(THREADS // 32):
                total += part[w]
            rstd = rms_rstd(total, Float32(self.eps))
            for i in cutlass.range_constexpr(2):
                column = tid * 16 + 8 * i
                gamma = load_words(norm.iterator + column, count=4)
                words = []
                for j in cutlass.range_constexpr(4):
                    g = unpack_bf16(gamma[j])
                    a = values[8 * i + 2 * j] * rstd * g[0]
                    b = values[8 * i + 2 * j + 1] * rstd * g[1]
                    words.append(pack_pair(a, b, policy=None))
                store_words(normed.iterator + Int64(t) * LATENT + column, words)
            cute.arch.sync_threads()
            t += grid
        cute.arch.griddepcontrol_launch_dependents()


@asm(Float32, pure=True)
def rms_rstd(total, eps):
    """rsqrtf(total / 3584 + eps) of vLLM's rms_norm (IEEE divide)."""
    code = "{ .reg .f32 m; div.rn.f32 m,$1,0f45600000; add.rn.f32 m,m,$2; "
    code += "rsqrt.approx.f32 $0,m; }"
    return code, "=f,f,f", (total, eps)


class EpBuffers:
    """One rank's symmetric EP buffer, its peers' base addresses and tensor views of
    the receive rows (the expert kernel's inputs)."""

    def __init__(self, group, ranks, rank, tokens, device):
        self.layout = lay = EpLayout(ranks, tokens)
        self.ranks, self.rank, self.tokens = ranks, rank, tokens
        self.buffer = symm_mem.empty(lay.bytes, dtype=torch.uint8, device=device)
        # Peer stores only: no multicast object.
        handle = peer_rendezvous(self.buffer, group)
        # The peers' base addresses, then this rank's own mapping (local accesses
        # never go through its peer mapping).
        bases = [int(p) for p in handle.buffer_ptrs] + [self.buffer.data_ptr()]
        self.peers = torch.tensor(bases, dtype=torch.int64, device=device)
        self.buffer.zero_()
        self.src(lay.rows).fill_(-1)
        local = EXPERTS // ranks
        self.ids(lay.rows).fill_(((rank + 1) % ranks) * local)
        torch.accelerator.synchronize(device)
        torch.distributed.barrier(group=group)

    def _view(self, offset, rows, width, dtype):
        size = rows * width * torch.tensor([], dtype=dtype).element_size()
        return self.buffer[offset : offset + size].view(dtype).view(rows, width)

    def x(self, rows):
        return self._view(self.layout.x, rows, LATENT, torch.float8_e4m3fn)

    def sf(self, rows):
        return self._view(self.layout.sf, rows, SF_BYTES, torch.uint8)

    def ids(self, rows):
        return self._view(self.layout.ids, rows, TOPK, torch.int32)

    def weights(self, rows):
        return self._view(self.layout.weights, rows, TOPK, torch.float32)

    def src(self, rows):
        return self._view(self.layout.src, rows, FLAG // 4, torch.int32)[:, 0]
