# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clustered MoE tail (TP8 B1..4, TP16 B1..8): routed finalization/publication,
all-reduce, RMS tree, K-split up projection, shared reduce-scatter and output multicast.
TMA weight loads overlap the mailbox wait. The caller owns the three rotating mailboxes
and must use different parities on successive calls. TP8 B5+ and TP16 B9+ run the
one-wave WaveMoeTail over latent rows published by the routed experts and shards
pushed by WaveMoeTail.push."""

from functools import reduce
from operator import or_

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, Uint64

from ..common.primitives import (
    acquire_tensormap,
    alloc,
    bf16,
    copy_g2s,
    fence_tmem,
    fma,
    fma_bf16,
    front_dirty,
    init_barriers,
    inv_rms,
    launch,
    load_f32x4,
    load_tma2,
    load_tma3,
    load_tmem,
    load_volatile,
    load_words,
    mailbox_store,
    mma,
    mma_bf16_block,
    mul2,
    multicast_words,
    no_neg_zero,
    pack_pair,
    push_float,
    release_tmem,
    round_pair_square,
    scatter_reduce,
    sleep_ns,
    smem_desc,
    smem_words,
    store_relaxed,
    store_shared_words,
    store_words,
    thread,
    view,
    wait,
    wait_tmem,
    zeros,
)

shuffle = cute.arch.shuffle_sync


def scatter_sums(values, lane, width):
    """``warp_sum(values[g], width)`` of every g (4 or 8 values) as a reduce-scatter:
    each butterfly level keeps half of the values and receives the partner's other
    half, so every partial sum (and the total) is the butterfly's. Returns the lane's
    total; value g's sits in lanes g << (log2(width) - log2(len(values)))."""
    values, offset = list(values), width // 2
    while len(values) > 1:
        half, upper = len(values) // 2, (lane & offset) != 0
        kept = []
        for j in range(half):
            low, high = values[j], values[half + j]
            keep = Float32(cutlass.select_(upper, high, low))
            send = Float32(cutlass.select_(upper, low, high))
            kept.append(keep + cute.arch.shuffle_sync_bfly(send, offset))
        values, offset = kept, offset // 2
    total = values[0]
    while offset >= 1:
        total += cute.arch.shuffle_sync_bfly(total, offset)
        offset //= 2
    return total


def gather_sums(total, count, width, padded):
    """Every lane's copy of ``scatter_sums``' first ``count`` totals."""
    shift = width.bit_length() - padded.bit_length()
    return [shuffle(total, g << shift) for g in range(count)]


def spread(pointer, lane, swap):
    first = smem_words(pointer + 8 * (2 * lane + swap))
    return first, smem_words(pointer + 8 * (2 * lane + 1 - swap))


class MoeTail:
    """One MoE tail launch over caller-owned persistent buffers. ``front_poll`` inputs
    have three parity slots armed with 0xffff BF16; latent/shared mailboxes [3, TP,
    capacity, 3584 or 7168/TP] are armed with 0x80000000. Only complete chains may use
    the early B1 trigger, which waits for the predecessor before releasing
    successors."""

    threads = 128
    options: dict = {}

    def __init__(self, tp: int, batch: int, *, mode="front", trigger=2):
        self.tp, self.batch, self.mode, self.trigger = tp, batch, mode, trigger
        self.cluster, self.tokens = (7 if tp == 8 else 8), min(batch, 2)
        self.poll_ns, self.front_poll_ns = (0, 0) if tp == 8 else (64, 128)
        self.cols, self.shard = 3584 // self.cluster, 7168 // tp
        self.frags, self.nc = self.cols // 8, self.shard // 32
        self.grid = self.cluster * self.nc * ((batch + self.tokens - 1) // self.tokens)

    @cute.jit
    def __call__(self, *args):
        common = dict(cluster=self.cluster, preferred_smem_carveout=100)
        launch(self.kernel, args, self.grid, self.threads, **common, **self.options)

    @cute.jit
    def poll(self, base, stride, second, second_stride, dual):
        values = cute.make_rmem_tensor((16, 4), Uint32)
        sources = (base, stride, second, second_stride)
        pending = Uint32(0)
        for source in cutlass.range_constexpr(16):
            if self.tp == 16 or source < 8 or dual:
                pending |= self.poll_source(values, source, sources)
        while pending != 0:
            if cutlass.const_expr(self.poll_ns):
                sleep_ns(Uint32(self.poll_ns))
            next_pending = Uint32(0)
            for source in cutlass.range_constexpr(16):
                if (pending & (Uint32(1) << source)) != 0:
                    next_pending |= self.poll_source(values, source, sources)
            pending = next_pending
        return values

    @cute.jit
    def poll_source(self, values, source: cutlass.Constexpr[int], sources):
        base, stride, second, second_stride = sources
        if cutlass.const_expr(self.tp == 16 or source < 8):
            word = load_volatile(base + source * stride, count=4)
        else:
            word = load_volatile(second + (source - 8) * second_stride, count=4)
        for k in cutlass.range_constexpr(4):
            values[source, k] = word[k]
        return Uint32(reduce(or_, (v == 0x80000000 for v in word))) << source

    @cute.jit
    def sum_sources(self, values, start: cutlass.Constexpr[int]):
        acc = zeros(8, Float32)
        for source in cutlass.range_constexpr(self.tp):
            for k in cutlass.range_constexpr(8):
                acc[k] = acc[k] + bf16(values[source + start, k // 2], k % 2)
        return acc

    def poll_strides(self, io):
        if self.mode != "front_poll":
            return 0, 0, 0
        return tuple(t.shape[0] // 3 for t in io[2:5])

    @cute.jit
    def front(self, io, latent_mc, par, rank, capacity, block, warp, lane):
        _, _, g, weights, s, peers = io
        if cutlass.const_expr(self.mode == "front"):
            cute.arch.griddepcontrol_wait()
        gs, ws, ss = self.poll_strides(io)
        start = block + self.grid * warp
        for task in cutlass.range(start, self.batch * 168, self.grid * 4, unroll=1):
            if task < self.batch * 56:
                token, line = task // 56, task % 56
                src = g.iterator + par * gs + token * 16 * 3584 + 64 * line + 2 * lane
                wp = weights.iterator + par * ws + token * 16
                w = cute.make_rmem_tensor(8, Uint32)
                rows = cute.make_rmem_tensor(16, Uint32)
                dirty = Int32(1)
                while dirty != 0:
                    for h in cutlass.range_constexpr(2):
                        for k, v in enumerate(load_volatile(wp + 8 * h, count=4)):
                            w[4 * h + k] = v
                    for j in cutlass.range_constexpr(16):
                        rows[j] = load_volatile(src + j * 3584)[0]
                    if cutlass.const_expr(self.mode == "front_poll"):
                        words = [w[k] for k in range(8)] + [rows[j] for j in range(16)]
                        bad = reduce(or_, map(front_dirty, words))
                        dirty = Int32(cute.arch.vote_any_sync(bad))
                        if self.front_poll_ns > 0 and dirty != 0:
                            sleep_ns(Uint32(self.front_poll_ns))
                    else:
                        dirty = 0
                a0, a1 = Float32(0), Float32(0)
                for j in cutlass.range_constexpr(16):
                    weight = bf16(w[j // 2], j % 2)
                    a0 = fma(bf16(rows[j]), weight, a0)
                    a1 = fma(bf16(rows[j], True), weight, a1)
                word = pack_pair(a0, a1, policy="latent")
                packed = tuple(shuffle(word, (4 * lane + k) & 31) for k in range(4))
                if lane < 8:
                    row = (par * self.tp + rank) * capacity + token
                    offset = row * 3584 + 64 * line + 8 * lane
                    multicast_words(latent_mc + Int64(offset) * 2, packed)
            elif lane < 8:
                u = task - self.batch * 56
                token, line = u // 112, u % 112
                destination = line // (self.shard // 64)
                own_line = line % (self.shard // 64)
                src = s.iterator + par * ss + token * 7168 + 64 * line + 8 * lane
                word = load_volatile(src, count=4)
                if cutlass.const_expr(self.mode == "front_poll"):
                    while reduce(or_, map(front_dirty, word)):
                        if cutlass.const_expr(self.front_poll_ns):
                            sleep_ns(Uint32(self.front_poll_ns))
                        word = load_volatile(src, count=4)
                packed = tuple(no_neg_zero(v) for v in word)
                row = (par * self.tp + rank) * capacity + token
                offset = row * self.shard + 64 * own_line + 8 * lane
                store_relaxed(peers[destination] + Int64(offset) * 2, packed)

    @cute.jit
    def gemv(self, sw, sx, partial, bars, q, warp, lane, mt: cutlass.Constexpr[int]):
        weight, xs, out = sw.iterator, sx.iterator, partial.iterator
        barrier = bars.iterator + 2
        acc = zeros(8 * mt, Float32)
        if cutlass.const_expr(self.tp == 8):
            # Two bank-spread loads; columns 16*lane .. 16*lane+15 accumulate in order.
            swap = (lane >> 2) & 1
            xl = cute.make_rmem_tensor((mt, 8), Uint32)
            for token in cutlass.range_constexpr(mt):
                aa, bb = spread(xs + token * self.cols, lane, swap)
                for k in cutlass.range_constexpr(4):
                    xl[token, k] = bb[k] if swap != 0 else aa[k]
                    xl[token, k + 4] = aa[k] if swap != 0 else bb[k]
            for row in cutlass.range_constexpr(8):
                wa, wb = spread(weight + (warp * 8 + row) * self.cols, lane, swap)
                for token in cutlass.range_constexpr(mt):
                    value = acc[row * mt + token]
                    for k in cutlass.range_constexpr(8):
                        ww = (wb + wa)[k] if swap != 0 else (wa + wb)[k]
                        value = fma_bf16(ww, xl[token, k], value)
                        value = fma_bf16(ww, xl[token, k], value, high=True)
                    acc[row * mt + token] = value
        elif lane < 28:
            for half in cutlass.range_constexpr(2):
                chunk = lane + 28 * half
                x = [smem_words(xs + t * self.cols + 8 * chunk) for t in range(mt)]
                for row in cutlass.range_constexpr(8):
                    ww = smem_words(weight + (warp * 8 + row) * self.cols + 8 * chunk)
                    for token in cutlass.range_constexpr(mt):
                        value = acc[row * mt + token]
                        for k in cutlass.range_constexpr(4):
                            value = fma_bf16(ww[k], x[token][k], value)
                            value = fma_bf16(ww[k], x[token][k], value, high=True)
                        acc[row * mt + token] = value
        # One final row/token per lane group, immediately pushed to the cluster leader.
        value, selected = scatter_reduce(acc, lane, 8 * mt)
        if (lane & ((4 if mt == 1 else 2) - 1)) == 0:
            row, token = selected // mt, selected % mt
            push_float(out + (q * 32 + warp * 8 + row) * mt + token, value, barrier, 0)

    @cute.kernel
    def kernel(self, args):
        latent, shared, weight_map, gamma, gemm2, weights, shared_output = args[0:7]
        peers, output, latent_mc, output_mc, par, rank, capacity, eps = args[7:15]
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        q = cute.arch.block_idx_in_cluster()
        warp, lane = tid // 32, tid % 32
        cluster = block // self.cluster
        group, row_cluster = cluster // self.nc, cluster % self.nc
        token0, row0 = group * self.tokens, row_cluster * 32
        local_tokens = min(self.tokens, self.batch - token0)
        allocator = cutlass.utils.SmemAllocator()
        sw = alloc(allocator, BFloat16, 32 * self.cols, 128)
        sg = alloc(allocator, BFloat16, self.cols)
        sx = alloc(allocator, BFloat16, self.tokens * self.cols)
        partial = alloc(allocator, Float32, self.cluster * 32 * self.tokens)
        squares = alloc(allocator, Float32, self.cluster * 2 * self.tokens)
        shared_sum = alloc(allocator, Float32, self.tokens * 32)
        bars = alloc(allocator, Uint64, 4, 8)
        bar, expect = bars.iterator, cute.arch.mbarrier_arrive_and_expect_tx
        if tid == 127:
            init_barriers(bar, 4)
            expect(bar + 3, self.cols * 2)
            gamma_q = gamma.iterator + q * self.cols
            copy_g2s(sg.iterator, gamma_q, size=self.cols * 2, barrier=bar + 3)
            expect(bar + 1, self.cluster * 2 * local_tokens * 4)
            expect(bar + 2, self.cluster * 32 * local_tokens * 4 if q == 0 else 0)
            expect(bar, 32 * self.cols * 2)
            acquire_tensormap(weight_map.iterator)
            load_tma3(sw.iterator, weight_map.iterator, bar, 2 * q, row0)
        cute.arch.cluster_arrive_relaxed()
        io = (latent, shared, gemm2, weights, shared_output, peers)
        self.front(io, latent_mc, par, rank, capacity, block, warp, lane)
        pt, pk = tid // 64, tid % 64
        lat_thread = (pt < local_tokens) & (pk < self.frags)
        rs_j = tid - 64 if self.tokens == 1 else pk if self.tp == 8 else pk - self.frags
        rs_thread = (q == 0) & (rs_j >= 0) & (rs_j < 4)
        if cutlass.const_expr(self.tokens != 1):
            rs_thread &= pt < local_tokens
        token = Int32(0) if cutlass.const_expr(self.tokens == 1) else pt
        s2 = Float32(0)
        if lat_thread | rs_thread:
            slot = par * self.tp * capacity + token0 + token
            lat_ptr = latent.iterator + (slot * 3584 + q * self.cols + 8 * pk)
            rs_ptr = shared.iterator + (slot * self.shard + row0 + 8 * max(rs_j, 0))
            base = lat_ptr if lat_thread else rs_ptr
            stride = capacity * (3584 if lat_thread else self.shard)
            dual = rs_thread & lat_thread
            values = self.poll(base, stride, rs_ptr, capacity * self.shard, dual)
            sums = self.sum_sources(values, 0)
            if lat_thread:
                packed = []
                for k in cutlass.range_constexpr(4):
                    packed.append(pack_pair(sums[2 * k], sums[2 * k + 1]))
                    square = round_pair_square(packed[k])
                    s2 += bf16(square)
                    s2 += bf16(square, True)
                store_shared_words(sx.iterator + token * self.cols + pk * 8, packed)
            else:
                for k in cutlass.range_constexpr(8):
                    shared_sum[token * 32 + 8 * rs_j + k] = Float32(BFloat16(sums[k]))
            if cutlass.const_expr(self.tp == 8 and self.tokens == 2):  # noqa: SIM102
                if dual:
                    rs = self.sum_sources(values, 8)
                    for k in cutlass.range_constexpr(8):
                        shared_sum[token * 32 + 8 * rs_j + k] = Float32(BFloat16(rs[k]))
        cute.arch.sync_warp()
        for stage in cutlass.range_constexpr(5):
            s2 += cute.arch.shuffle_sync_bfly(s2, 16 >> stage)
        cute.arch.cluster_wait()
        if pt < local_tokens and lane < self.cluster:
            sq = squares.iterator + (2 * q + (warp & 1)) * local_tokens + pt
            push_float(sq, s2, bar + 1, lane)
        cute.arch.sync_warp()
        wait(bar + 1)
        cute.arch.sync_warp()
        if cutlass.const_expr(self.trigger == 3):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        wait(bar + 3)
        if lat_thread:
            total = Float32(0)
            for peer in cutlass.range_constexpr(2 * self.cluster):
                total += squares[peer * local_tokens + pt]
            inverse = inv_rms(total, eps)
            gx = smem_words(sg.iterator + pk * 8)
            xx = smem_words(sx.iterator + pt * self.cols + pk * 8)
            scaled = lambda k, h: bf16(xx[k], h) * inverse * bf16(gx[k], h)
            packed = [pack_pair(scaled(k, 0), scaled(k, 1)) for k in range(4)]
            store_shared_words(sx.iterator + pt * self.cols + pk * 8, packed)
        wait(bar)
        cute.arch.sync_warp()
        cute.arch.sync_threads()
        if self.tokens == 1 or self.batch % 2 == 0 or local_tokens == 2:
            self.gemv(sw, sx, partial, bars, q, warp, lane, self.tokens)
        else:
            self.gemv(sw, sx, partial, bars, q, warp, lane, 1)
        if q == 0:  # noqa: SIM102 - Keep the owner CTA branch separate from its warp selection.
            if warp < local_tokens:
                wait(bar + 2)
                cute.arch.sync_warp()
                value = Float32(0)
                for peer in cutlass.range_constexpr(self.cluster):
                    value += partial[(peer * 32 + lane) * local_tokens + warp]
                rounded = Float32(BFloat16(value))
                out = Float32(BFloat16(rounded + shared_sum[warp * 32 + lane]))
                # Pack paired rows, then four pairs into a 16-byte writer.
                pair = pack_pair(out, shuffle(out, (lane + 1) & 31), policy="latent")
                words = tuple(shuffle(pair, (8 * lane + 2 * k) & 31) for k in range(4))
                if lane < 4:
                    offset = (token0 + warp) * 7168 + rank * self.shard + row0
                    multicast_words(output_mc + Int64(offset + 8 * lane) * 2, words)
        if cutlass.const_expr(self.trigger == 2):
            # front_poll: the next AttnRes reads the prefix before any wait, and no
            # kernel since the o-proj consumer that wrote it (the MoE front and
            # experts) waits before releasing: this grid must, at every batch.
            if cutlass.const_expr(self.mode == "front_poll"):
                cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        self.rearm(io, capacity, *args[15:17], block, tid)
        # Orders deferred rearm, workspace reuse and producer lifetimes (all modes).
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def rearm(self, io, capacity, parity, batch, block, tid, reduced=None):
        """Re-arms ``parity``'s first ``batch`` tokens (and WaveMoeTail's reduced
        rows)."""
        latent, shared, g, w, s, _ = io
        if parity >= 0:
            nl, nr = self.tp * batch * 448, self.tp * batch * (self.shard // 8)
            ng, nw, ns = Int32(0), Int32(0), Int32(0)
            if cutlass.const_expr(self.mode == "front_poll"):
                gs, ws, ss = self.poll_strides(io)
                ng = min(batch, gs // (16 * 3584)) * 16 * 448
                nw = min(batch, ws // 16) * 2
                ns = min(batch, ss // 7168) * 896
            elif cutlass.const_expr(reduced is not None):
                ng = batch * 448
            ring = (parity, capacity, batch)
            start, step = block * self.threads + tid, self.grid * self.threads
            for i in cutlass.range(start, nl + nr + ng + nw + ns, step, unroll=1):
                if i < nl:
                    self.rearm_row(latent, 3584, i, ring)
                elif i < nl + nr:
                    self.rearm_row(shared, self.shard, i - nl, ring)
                elif cutlass.const_expr(self.mode == "front_poll"):
                    j = i - nl - nr
                    fill = (Uint32(0xFFFFFFFF),) * 4
                    if j < ng:
                        store_words(g.iterator + parity * gs + 8 * j, fill)
                    elif j < ng + nw:
                        store_words(w.iterator + parity * ws + 8 * (j - ng), fill)
                    else:
                        store_words(s.iterator + parity * ss + 8 * (j - ng - nw), fill)
                elif cutlass.const_expr(reduced is not None):
                    rows = reduced.iterator + parity * capacity * 3584
                    store_words(rows + 8 * (i - nl - nr), (Uint32(0x80000000),) * 4)

    @cute.jit
    def rearm_row(self, box, width: cutlass.Constexpr[int], i, ring):
        parity, capacity, batch = ring
        row, frag = i // (width // 8), i % (width // 8)
        source, token = row // batch, row % batch
        dst = box.iterator + ((parity * self.tp + source) * capacity + token) * width
        store_words(dst + 8 * frag, (Uint32(0x80000000),) * 4)


def pow2_spans(n):
    """(start, count) spans of n columns, power-of-two counts, largest first."""
    bits = [1 << i for i in reversed(range(n.bit_length())) if n >> i & 1]
    return tuple((sum(bits[:i]), b) for i, b in enumerate(bits))


class WaveMoeTail(MoeTail):
    """One-wave tail over published inputs (TP8 B5+, TP16 B9+): seven 16-CTA clusters
    split K into 224-column slices of a 128-row (TP16: 64-row) tile. Weight fragments
    stay in registers through the latent poll, RMS all-gather and normalization; row
    warps (TP16: two sets taking alternate groups) run two seven-step BF16 MMA chains
    per 8-token group, and each 64-row output line has one owner CTA and one multicast
    writer. Past one group (and at TP16) each cluster instead all-reduces every seventh
    token of its slices once and shares them as BF16 rows through ``reduced``, so all
    112 CTAs must be co-resident; RS lines spread over warp pairs, every group's RMS
    sums travel as one vector per peer, all groups normalize before any MMA, the
    groups' MMA chains interleave per K step, and the previous layer's rows are
    re-armed after the output multicast. ``twoshot`` (B9+): the latent
    mailbox is this rank's ``common.twoshot`` receive slots [3, TP, capacity, 3584/TP]
    (pushed by the experts), the CTAs of its owned slices reduce them like TP16 but
    multicast the rows into every rank's ``reduced`` (the gather; its multicast
    address replaces the latent one), so each GPU receives 2 x B x 7 KB of latent
    instead of TP x B x 7 KB."""

    threads, options = 256, dict(min_blocks_per_mp=1)

    def __init__(self, batch, tp=8, twoshot=False, early=False, umma=False):
        super().__init__(tp, batch, mode="publish")
        assert not twoshot or batch > 8
        assert not umma or batch > 8
        # ``early``: release the next AttnRes at entry, so its block statistics and
        # exchange overlap this kernel (it reads only earlier layers' data before its
        # Lamport poll of the output rows), instead of after the output multicast.
        self.early = early
        self.lines, self.groups = 2 if tp == 8 else 1, (batch + 7) // 8
        self.slots = (batch * self.lines + 15) // 16  # owned output lines per CTA
        self.twoshot, self.width = twoshot, 3584 // tp if twoshot else 3584
        # Past one group (and at TP16) clusters share reduced latent rows; TP8 B<=8
        # polls its sources directly.
        self.reduce = self.groups > 1 or tp == 16
        self.grid, self.poll_ns, self.cluster = 112, 64, 16
        # Shared memory: per group x [8, 224] BF16 and MMA partials [8 warps, 8, 16]
        # (TP16 pads to whole warp sets); gathered squares (one group keeps each
        # slice's 28 lane sums, more one sum); per owned line its partials [16 slices,
        # 64] and RS rows [64]; two barriers.
        n, s = self.groups + self.groups % (2 // self.lines), self.slots
        # Past one group each (warp, slice) row holds the groups' sums, padded to v4.
        self.padded = -(-self.groups // 4) * 4
        self.squares = 3584 if n == 1 else 128 * max(n, self.padded)
        partials = 7680 * n + 4 * self.squares
        self.offsets: tuple = (3584 * n, 7680 * n, partials, partials + 4096 * s)
        self.offsets += (partials + 4352 * s + 32,)
        self.smem = 116 * 1024  # one CTA per SM
        # ``umma`` (past one group): the CTA's [M, 224] x [N, 224] up projection on
        # tcgen05 (M = 64 x lines rows, N = the padded tokens, K padded to 256) from
        # 128B-swizzled W and x tiles (W: four TMA boxes [M, 64] at entry, the last
        # running into the next slice, whose columns meet the zeroed x padding; the
        # weight argument is then the shard's tensor map), D in TMEM, staged as
        # [token, row] in the drained W tiles for the owners' pushes. FP32
        # accumulation in the tensor core's order.
        self.umma = umma
        if umma:
            self.m = 64 * self.lines
            self.n = -(-8 * self.groups // (16 if self.lines == 2 else 8))
            self.n *= 16 if self.lines == 2 else 8
            self.columns = max(32, 1 << (self.n - 1).bit_length())
            w_bytes, x_bytes = self.m * 512, self.n * 512
            squares = w_bytes + x_bytes
            partials = squares + 4 * self.squares
            self.offsets = (0, squares, partials, partials + 4096 * s)
            self.offsets += (partials + 4352 * s + 32,)
            self.smem = max(self.smem, self.offsets[4] + 40)

    def group_smem(self, sm, g):
        """Group g's x (BF16 pointer) and MMA partials."""
        x = view(sm, 3584 * g, BFloat16, 1792).iterator
        return x, view(sm, self.offsets[0] + 4096 * g, Float32, 1024)

    def line_smem(self, sm, s):
        """Owned line s's partials [16 slices, 64] and RS rows."""
        partial, rs = self.offsets[2:4]
        cells = view(sm, partial + 4096 * s, Float32, 1024)
        return cells, view(sm, rs + 256 * s, Float32, 64)

    @cute.kernel
    def kernel(self, args):
        latent, shared, weight, gamma = args[0:4]
        output_mc, par, rank, capacity, eps = args[10:15]
        tid, warp, lane, block = thread()
        if cutlass.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        q = cute.arch.block_idx_in_cluster()
        n, batch, lines, rows = self.groups, self.batch, self.lines, 64 * self.lines
        row0, gq, tq, owner = block // 16 * rows, lane // 4, lane % 4, q < lines * batch
        own_token, own_line = q // lines, q % lines
        rs_j, lat = 4 * warp + lane - 28, lane < 28
        owns = lambda s: owner if s == 0 else q < lines * batch - 16 * s
        rs_lane = lambda g: (lane >= 28) & owns(g) & (rs_j < 8)
        align = 1024 if self.umma else 128
        sm = alloc(cutlass.utils.SmemAllocator(), Uint8, self.smem, align).iterator
        groups = [self.group_smem(sm, g) for g in range(n)]
        owned = [self.line_smem(sm, s) for s in range(self.slots)]
        squares = view(sm, self.offsets[1], Float32, self.squares)
        bar = view(sm, self.offsets[4], Uint64, 4).iterator
        tmem_slot = view(sm, self.offsets[4] + 32, Uint32, 1)
        if cutlass.const_expr(self.umma):  # noqa: SIM102
            if warp == 0:
                cute.arch.alloc_tmem(self.columns, tmem_slot.iterator)
                cute.arch.relinquish_tmem_alloc_permit()
        if tid == 0:
            init_barriers(bar, 4 if self.umma else 2)
            if cutlass.const_expr(self.umma):  # W [M, 64] x 4 by TMA on barrier 3
                acquire_tensormap(weight.iterator)
                cute.arch.mbarrier_arrive_and_expect_tx(bar + 3, self.m * 512)
                for kb in cutlass.range_constexpr(4):
                    tile = sm + kb * (self.m * 128)
                    load_tma2(tile, weight.iterator, bar + 3, 224 * q + 64 * kb, row0)
            # One group gathers 28 lane sums per (token, slice), more one sum for
            # every warp's token.
            gathered = 16 * batch * 28 * 4 if n == 1 else 16 * 8 * self.padded * 4
            cute.arch.mbarrier_arrive_and_expect_tx(bar, gathered)
            size = 4096 if owns(self.slots - 1) else 0  # earlier lines are all owned
            size = size + 4096 * (self.slots - 1) if self.slots > 1 else size
            cute.arch.mbarrier_arrive_and_expect_tx(bar + 1, size)
        cute.arch.cluster_arrive_relaxed()
        xs = groups[0][0]
        if cutlass.const_expr(not self.umma):
            for i in cutlass.range(tid, (8 * n - batch) * 28, 256, unroll=1):
                store_shared_words(xs + batch * 224 + 8 * i, (Uint32(0),) * 4)
        gx, gamma_q = cute.make_rmem_tensor(4, Uint32), gamma.iterator + q * 224
        if lat & (warp < batch):
            for k, v in enumerate(load_words(gamma_q + 8 * lane, count=4)):
                gx[k] = v
        # Rows 16 warp + gq (+ 8) of the cluster's tile (TP16: warp mod 4), columns
        # 224 q + 32 k + 8 tq.
        rw = warp if lines == 2 else warp % 4
        base = weight.iterator + (row0 + 16 * rw + gq) * 3584 + 224 * q + 8 * tq
        fetch = lambda p: load_words(p, count=4, evict_first=True, mutable=True)
        a = None
        if cutlass.const_expr(not self.umma):
            a = [
                [fetch(base + h * 8 * 3584 + 32 * k) for h in range(2)]
                for k in range(7)
            ]
        xwords = [cute.make_rmem_tensor(4, Uint32) for _ in range(n)]
        reduced = args[17] if self.reduce else None
        ring = (*args[15:17], block, tid, reduced)
        if cutlass.const_expr(self.reduce):
            s2 = zeros(n, Float32)
            ids: tuple = (q, warp, lane, block // 16)
            places = (row0, owned[0][1].iterator)
            gather = (args[9], rank) if self.twoshot else None
            boxes = (latent, reduced, shared)
            self.poll_inputs(boxes, (par, capacity), ids, places, gather)
            if lat:
                self.read_reduced(reduced, par * capacity, xwords, s2, ids)
        else:
            # Lanes 0..27 of warp t poll token t's latent; owners' lanes 28..31 the RS.
            sq = Float32(0)
            if lat & (warp < batch) | rs_lane(0):
                slot = par * self.tp * capacity
                rs = (slot + own_token) * self.shard + row0 + 64 * own_line
                rs += 8 * max(rs_j, 0)
                ptr = latent.iterator + (slot + warp) * 3584 + 224 * q + 8 * lane
                ptr = ptr if lat else shared.iterator + rs
                stride = capacity * (3584 if lat else self.shard)
                sums = self.sum_sources(self.poll(ptr, stride, ptr, 0, False), 0)
                if lat:
                    sq = self.latent_words(sums, xwords[0])
                else:
                    self.rs_words(sums, owned[0][1], rs_j)
            s2 = [sq]
        cute.arch.sync_warp()
        cute.arch.cluster_wait()
        if cutlass.const_expr(n == 1):
            self.push_squares(squares, s2[0], bar, (q, warp, lane))
            wait(bar)
            cute.arch.sync_warp()
            row = (xwords[0], gx, eps)
            self.normalize(squares, row, groups[0][0], (warp, lane, lat))
        else:
            self.exchange_squares(squares, s2, bar, (q, warp, lane))
            wait(bar)
            cute.arch.sync_warp()
            outs = [groups[g][0] for g in range(n)]
            if cutlass.const_expr(self.umma):  # x tiles [N, 64] x 4 after W
                outs = [sm + self.m * 512 + 1024 * g for g in range(n)]
            self.normalize_groups(squares, (xwords, gx, eps), outs, (warp, lane, lat))
        if cutlass.const_expr(self.umma):  # x stores -> tcgen05
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem(before=True)
        cute.arch.sync_warp()
        cute.arch.sync_threads()
        ids = (q, warp, lane, rw, gq, tq)
        if cutlass.const_expr(n == 1):
            self.mma_group(a, groups[0], owned, bar, 0, ids)
        elif cutlass.const_expr(self.umma):
            tmem = tmem_slot[0]
            self.umma_partials(sm, tmem, bar, (q, warp, lane, tid))
        else:
            size = (self.offsets[1] - self.offsets[0]) // 4
            products = view(sm, self.offsets[0], Float32, size)
            self.mma_groups(a, xs, products, owned, bar, ids)
        line = (lane, rank, row0, own_token, own_line)
        for s in cutlass.range_constexpr(self.slots):
            if owns(s) & (warp == s):
                self.write_line(owned[s], bar, output_mc, line, s)
        if cutlass.const_expr(not self.early):
            cute.arch.griddepcontrol_launch_dependents()
        # The previous layer's rows, off the output path.
        if cutlass.const_expr(n == 1):
            self.rearm(args[0:2] + args[4:8], capacity, *ring)
        else:
            self.rearm_rows(args[0:2], capacity, *ring)
        if cutlass.const_expr(self.umma):
            release_tmem(tmem, self.columns, warp, 0)
        cute.arch.griddepcontrol_wait()

    @cute.jit
    def load_sources(self, values, ptr, stride):
        """Issue the TP source loads of a lane's 16-byte column without waiting."""
        for source in cutlass.range_constexpr(self.tp):
            word = load_volatile(ptr + source * stride, count=4)
            for k in cutlass.range_constexpr(4):
                values[source, k] = word[k]

    @cute.jit
    def settle(self, values, ptr, stride):
        """Re-poll loaded sources until no word is the 0x80000000 sentinel."""
        pending = Uint32(0)
        for source in cutlass.range_constexpr(self.tp):
            words = (values[source, k] for k in range(4))
            pending |= Uint32(reduce(or_, (v == 0x80000000 for v in words))) << source
        while pending != 0:
            sleep_ns(Uint32(self.poll_ns))
            again = Uint32(0)
            for source in cutlass.range_constexpr(self.tp):
                if (pending & (Uint32(1) << source)) != 0:
                    again |= self.poll_row(values, source, ptr + source * stride)
            pending = again

    @cute.jit
    def column_sums(self, values):
        """``sum_sources`` (same order and rounding) as one BF16 FMA by 1.0 per term."""
        acc, one = zeros(8, Float32), Uint32(0x3F803F80)
        for source in cutlass.range_constexpr(self.tp):
            for k in cutlass.range_constexpr(8):
                acc[k] = fma_bf16(values[source, k // 2], one, acc[k], high=k % 2 == 1)
        return acc

    @cute.jit
    def rearm_rows(self, boxes, capacity, parity, batch, block, tid, reduced):
        """Division-free ``rearm``: each source's latent and RS rows, then the reduced
        rows, of the previous layer (complete before this launch)."""
        if parity >= 0:
            start, step = block * self.threads + tid, self.grid * self.threads
            fill, widths = (Uint32(0x80000000),) * 4, (self.width, self.shard)
            for s in cutlass.range_constexpr(self.tp):
                first = (parity * self.tp + s) * capacity
                for b in cutlass.range_constexpr(2):
                    row, w = boxes[b].iterator + first * widths[b], widths[b]
                    for i in cutlass.range(start, batch * (w // 8), step, unroll=1):
                        store_words(row + 8 * i, fill)
            if cutlass.const_expr(reduced is not None):
                row = reduced.iterator + parity * capacity * 3584
                for i in cutlass.range(start, batch * 448, step, unroll=1):
                    store_words(row + 8 * i, fill)

    @cute.jit
    def poll_inputs(self, boxes, ring, ids, places, gather=None):
        """Phase 1 and the RS lines in one convergent pass (every round's loads issued
        first). Lanes 0..27 all-reduce the cluster's tokens tile + 7 j of the CTA's
        slice once into BF16 ``reduced`` rows (``gather`` (multicast address, rank):
        only the CTAs of owned slices, from the receive slots into every rank's);
        lanes 28..31 of warps 2 p and 2 p + 1 sum owned lines p, p + 4's RS shard
        (line 16 i + q: token (16 i + q) / lines)."""
        (latent, reduced, shared), (par, capacity) = boxes, ring
        (q, warp, lane, tile), (row0, rs_rows) = ids, places
        lat, width = lane < 28, self.width
        col = 224 * q + 8 * lane
        local = col if gather is None else col - gather[1] * width
        src = latent.iterator + (par * self.tp * capacity + tile) * width + local
        dst = reduced.iterator + (par * capacity + tile) * 3584 + col
        reduces = q >= 0 if gather is None else q * self.tp // 16 == gather[1]
        rs_j = 4 * (warp % 2) + lane - 28
        rs = shared.iterator + par * self.tp * capacity * self.shard + row0 + 8 * rs_j
        stride = capacity * width if lat else capacity * self.shard
        latent_rounds = ((self.batch - 1) // 7 + 8) // 8
        rounds = max(latent_rounds, (self.slots + 3) // 4)
        tokens = [7 * (warp + 8 * r) for r in range(rounds)]
        lines = [4 * r + warp // 2 for r in range(rounds)]
        live, ptrs = [], []
        for r in cutlass.range_constexpr(rounds):
            token, line = tokens[r], lines[r]
            on = lane < 0  # no latent round r
            if cutlass.const_expr(r < latent_rounds):
                on = reduces & (tile + token < self.batch)
            rs_on = (line < self.slots) & (16 * line + q < self.lines * self.batch)
            live.append(on if lat else rs_on)
            rs_row = rs + (16 * line + q) // self.lines * self.shard
            rs_row += 64 * ((16 * line + q) % self.lines)
            ptrs.append(src + token * width if lat else rs_row)
        values = [cute.make_rmem_tensor((self.tp, 4), Uint32) for _ in range(rounds)]
        for r in cutlass.range_constexpr(rounds):
            if live[r]:
                self.load_sources(values[r], ptrs[r], stride)
        for r in cutlass.range_constexpr(rounds):
            if live[r]:
                self.settle(values[r], ptrs[r], stride)
                sums = self.column_sums(values[r])
                if lat:
                    if cutlass.const_expr(gather is None):
                        self.store_reduced(sums, dst + tokens[r] * 3584)
                    else:
                        offset = (par * capacity + tile + tokens[r]) * 3584 + col
                        self.store_reduced(sums, gather[0] + Int64(offset) * 2, True)
                else:
                    rows = view(rs_rows, 64 * lines[r], Float32, 64)
                    self.rs_words(sums, rows, rs_j)

    @cute.jit
    def store_reduced(self, sums, dst, multicast: cutlass.Constexpr[bool] = False):
        pack = lambda j: pack_pair(sums[2 * j], sums[2 * j + 1], policy="latent")
        if cutlass.const_expr(multicast):
            multicast_words(dst, [pack(j) for j in range(4)])
        else:
            mailbox_store(dst, [pack(j) for j in range(4)])

    @cute.jit
    def read_reduced(self, reduced, slot, xwords, s2, ids):
        """Phase 2: the lane's words of tokens 8 g + warp and their squares."""
        q, warp, lane, _ = ids
        base = reduced.iterator + (slot + warp) * 3584 + 224 * q + 8 * lane
        values = self.poll_rows(base, 8 * 3584, (self.batch - warp + 7) // 8)
        for g in cutlass.range_constexpr(self.groups):
            for k in cutlass.range_constexpr(4):
                xwords[g][k] = values[g, k]
            s2[g] = self.square_sum(xwords[g])

    @cute.jit
    def poll_rows(self, base, stride, count):
        """Lamport-poll the first ``count`` 16-byte rows base + r stride."""
        values = cute.make_rmem_tensor((self.groups, 4), Uint32)
        pending = Uint32(0)
        for r in cutlass.range_constexpr(self.groups):
            if r < count:
                pending |= self.poll_row(values, r, base + r * stride)
        while pending != 0:
            sleep_ns(Uint32(self.poll_ns))
            again = Uint32(0)
            for r in cutlass.range_constexpr(self.groups):
                if (pending & (Uint32(1) << r)) != 0:
                    again |= self.poll_row(values, r, base + r * stride)
            pending = again
        return values

    @cute.jit
    def poll_row(self, values, r: cutlass.Constexpr[int], ptr):
        word = load_volatile(ptr, count=4)
        for k in cutlass.range_constexpr(4):
            values[r, k] = word[k]
        return Uint32(reduce(or_, (v == 0x80000000 for v in word))) << r

    @cute.jit
    def mma_group(self, a, group, owned, bar, g: cutlass.Constexpr[int], ids):
        """Group g's MMA chains; each 4-row partial goes to its 64-row line's owner."""
        (q, warp, lane, rw, gq, tq), (xg, pg) = ids, group
        b = [smem_words(xg + gq * 224 + 8 * tq + 32 * k) for k in range(7)]
        d = e = (Float32(0),) * 4
        for k in cutlass.range_constexpr(7):
            (w0, w1), x = a[k], b[k]
            d = mma((w0[0], w1[0], w0[1], w1[1]), x[0:2], d, bf16=True)
            e = mma((w0[2], w1[2], w0[3], w1[3]), x[2:4], e, bf16=True)
        for i in cutlass.range_constexpr(4):  # [warp, token, row] partial products
            pg[warp * 128 + (2 * tq + i % 2) * 16 + gq + 8 * (i // 2)] = d[i] + e[i]
        cute.arch.sync_warp()
        token, row = lane // 4, 16 * rw + 4 * (lane % 4)
        if token < self.batch - 8 * g:
            values = load_f32x4(pg.iterator + warp * 128 + 4 * lane, shared=True)
            # Line lines (8 g + token) + row // 64 is owned by CTA (line % 16).
            peer = self.lines * token + row // 64 + 8 * g * self.lines % 16
            cells = owned[8 * g * self.lines // 16][0]
            push_float(cells.iterator + q * 64 + row % 64, values, bar + 1, peer)

    @cute.jit
    def mma_groups(self, a, xs, products, owned, bar, ids):
        """``mma_group`` over this warp's groups g = sets j + set (TP16: two warp
        sets): every MMA chain (the groups' chains interleaved per K step), one warp
        sync, then the owners' pushes (slot j)."""
        q, warp, lane, rw, gq, tq = ids
        sets = 2 // self.lines
        h = 0 if sets == 1 else warp // 4
        mine = (self.groups + sets - 1) // sets
        x = [xs + (sets * j + h) * 1792 + gq * 224 + 8 * tq for j in range(mine)]
        d = [(Float32(0),) * 4 for _ in range(mine)]
        e = [(Float32(0),) * 4 for _ in range(mine)]
        for k in cutlass.range_constexpr(7):
            w0, w1 = a[k]
            for j in cutlass.range_constexpr(mine):
                xk = smem_words(x[j] + 32 * k)
                d[j] = mma((w0[0], w1[0], w0[1], w1[1]), xk[0:2], d[j], bf16=True)
                e[j] = mma((w0[2], w1[2], w0[3], w1[3]), xk[2:4], e[j], bf16=True)
        acc = [[d[j][i] + e[j][i] for i in range(4)] for j in range(mine)]
        for j in cutlass.range_constexpr(mine):
            cell = (sets * j + h) * 1024 + warp * 128 + 2 * tq * 16 + gq
            for i in cutlass.range_constexpr(4):
                products[cell + i % 2 * 16 + 8 * (i // 2)] = acc[j][i]
        cute.arch.sync_warp()
        token, row = lane // 4, 16 * rw + 4 * (lane % 4)
        peer = self.lines * token + row // 64 + (8 * h if sets > 1 else 0)
        for j in cutlass.range_constexpr(mine):
            if token < self.batch - 8 * (sets * j + h):
                src = products.iterator + (sets * j + h) * 1024 + warp * 128 + 4 * lane
                values = load_f32x4(src, shared=True)
                cells = owned[j][0]
                push_float(cells.iterator + q * 64 + row % 64, values, bar + 1, peer)

    @cute.jit
    def write_line(self, line, bar, output_mc, ids, s: cutlass.Constexpr[int]):
        """Owned line s: sum the 16 slices' partials, add the RS rows and multicast the
        64 output rows."""
        (cells, rs_sum), (lane, rank, row0, own_token, own_line) = line, ids
        wait(bar + 1)
        cute.arch.sync_warp()
        out = []
        for h in cutlass.range_constexpr(2):
            parts = (cells[64 * p + 2 * lane + h] for p in range(16))
            value = Float32(BFloat16(sum(parts, Float32(0))))
            out.append(value + rs_sum[2 * lane + h])
        pair = pack_pair(*out, policy="latent")
        words = tuple(shuffle(pair, (4 * lane + k) & 31) for k in range(4))
        if lane < 8:
            t = 16 // self.lines * s + own_token
            offset = t * 7168 + rank * self.shard + row0 + 64 * own_line
            multicast_words(output_mc + Int64(offset + 8 * lane) * 2, words)

    @cute.jit
    def latent_words(self, sums, words):
        """BF16 latent words of a lane's eight columns; returns their square sum."""
        s2 = Float32(0)
        for j in cutlass.range_constexpr(4):
            words[j] = pack_pair(sums[2 * j], sums[2 * j + 1])
            square = round_pair_square(words[j])
            s2 += bf16(square)
            s2 += bf16(square, True)
        return s2

    @cute.jit
    def square_sum(self, words):
        s2 = Float32(0)
        for j in cutlass.range_constexpr(4):
            square = round_pair_square(words[j])
            s2 += bf16(square)
            s2 += bf16(square, True)
        return s2

    @cute.jit
    def rs_words(self, sums, rs_sum, rs_j):
        for j in cutlass.range_constexpr(8):
            rs_sum[8 * rs_j + j] = Float32(BFloat16(sums[j]))

    @cute.jit
    def push_squares(self, squares, s2, bar, ids):
        """One group: the warp's token's 28 lane sums of every slice to every peer."""
        q, warp, lane = ids
        if warp < self.batch:
            chunk, first = lane % 7, lane // 7
            vector = tuple(shuffle(s2, 4 * chunk + i) for i in range(4))
            if first < 4:
                dst = squares.iterator + warp * 448 + 28 * q + 4 * chunk
                for peer in cutlass.range(first, 16, 4, unroll=1):
                    push_float(dst, vector, bar, peer)

    @cute.jit
    def exchange_squares(self, squares, s2, bar, ids):
        """Past one group: every group's warp sum for the warp's token; lane p < 16
        pushes them to peer p's (warp, slice) row as v4 words."""
        q, warp, lane = ids
        values = [s2[g] for g in range(self.groups)]
        values += [Float32(0)] * (self.padded - self.groups)
        total = scatter_sums(values, lane, 32)
        totals = gather_sums(total, self.padded, 32, self.padded)
        if lane < 16:
            dst = squares.iterator + (warp * 16 + q) * self.padded
            for c in cutlass.range_constexpr(self.padded // 4):
                push_float(dst + 4 * c, tuple(totals[4 * c : 4 * c + 4]), bar, lane)

    @cute.jit
    def normalize_groups(self, squares, row, xs, ids):
        """Past one group: every group's RMS (the 16 slices' sums, ``square_total``'s
        order) first, then the scaled latent words (padding tokens too)."""
        (words, gx, eps), (warp, lane, lat) = row, ids
        base = squares.iterator + (warp * 16 + lane % 16) * self.padded
        sums = []
        for c in cutlass.range_constexpr(self.padded // 4):
            sums += list(load_f32x4(base + 4 * c, shared=True))
        inverse = inv_rms(scatter_sums(sums, lane, 16), eps, m8=True)
        inverse = gather_sums(inverse, self.groups, 16, self.padded)
        if lat:
            for g in cutlass.range_constexpr(self.groups):
                self.scale_words(words[g], gx, inverse[g], xs[g], warp, lane)
        else:
            if cutlass.const_expr(self.umma):  # K columns 224..255 of the x tiles
                for g in cutlass.range_constexpr(self.groups):
                    dst = self.x_chunk(xs[g], warp, lane)
                    store_shared_words(dst, (Uint32(0),) * 4)

    @cute.jit
    def normalize(self, squares, row, xs, ids):
        """One group: RMS-normalize the warp's token into ``xs``."""
        (words, gx, eps), (warp, lane, _) = row, ids
        if warp < self.batch:
            total = self.square_total(squares, warp, lane)
            self.scale(words, gx, inv_rms(total, eps, m8=True), xs, ids)

    @cute.jit
    def square_total(self, squares, warp, lane):
        groups = cute.make_rmem_tensor(16, Float32)
        for c in cutlass.range_constexpr(16):
            groups[c] = Float32(0)
            if lane < (16 if c % 4 == 3 else 32):
                groups[c] = squares[warp * 448 + c // 4 * 112 + c % 4 * 32 + lane]
        group_sum, _ = scatter_reduce(groups, lane, 16)
        return sum((shuffle(group_sum, 2 * c) for c in range(16)), Float32(0))

    @cute.jit
    def scale(self, words, gx, inverse, xs, ids):
        warp, lane, lat = ids
        if lat:
            self.scale_words(words, gx, inverse, xs, warp, lane)

    def scale_words(self, words, gx, inverse, xs, warp, lane):
        """(x * inverse) * gamma per column, as packed FP32x2 multiplies."""
        pair = lambda w: (bf16(w), bf16(w, True))
        scaled = lambda k: mul2(mul2(pair(words[k]), (inverse, inverse)), pair(gx[k]))
        packed = [pack_pair(*scaled(k)) for k in range(4)]
        dst = xs + warp * 224 + 8 * lane
        if cutlass.const_expr(self.umma):
            dst = self.x_chunk(xs, warp, lane)
        store_shared_words(dst, packed)

    def x_chunk(self, tile, row, chunk):
        """16-byte chunk ``chunk`` (K columns 8 chunk ..) of row ``row`` in 128B-
        swizzled [rows, 64] K tiles of ``self.n`` rows (``tile``: the 8-row block)."""
        atom, part = chunk // 8, chunk % 8
        return tile + atom * (self.n * 128) + row * 128 + (part ^ row) * 16

    @cute.jit
    def umma_partials(self, sm, tmem, bar, ids):
        """D [M, N] = W x^T on tcgen05 (warp 0 issues 4 K tiles), staged [token, row]
        in the W tiles, then the 4-row partials pushed to their lines' owners."""
        q, warp, lane, tid = ids
        m, n = self.m, self.n
        done = bar + 2
        instruction: cutlass.Constexpr = 1 << 4 | 1 << 7 | 1 << 10
        instruction |= n >> 3 << 17 | m >> 4 << 24
        if warp == 0:
            wait(bar + 3)  # W tiles
            fence_tmem()
            for kb in cutlass.range_constexpr(4):
                a = smem_desc(sm + kb * (m * 128), True)
                b = smem_desc(sm + m * 512 + kb * (n * 128), True)
                mma_bf16_block(tmem, a, b, instruction, int(kb > 0), done, int(kb == 3))
        wait_tmem(done)
        # The MMAs (async proxy) read the W tiles the staging now overwrites.
        cute.arch.fence_proxy("async.shared", space="cta")
        # Warp w reads TMEM lanes 32 (w mod 4) .. (M = 64: the first 16 of each),
        # half w / 4 of the columns.
        quad, half, cols = warp % 4, warp // 4, n // 2
        values = ()
        for start, count in pow2_spans(cols):  # TMEM loads: power-of-two columns
            got = load_tmem(tmem + (32 * quad << 16) + half * cols + start, count)
            values += got if count > 1 else (got,)
        row = 32 * quad + lane if m == 128 else 16 * quad + lane
        stage = view(sm, 0, Float32, n * m)
        if lane < (32 if m == 128 else 16):
            for t in cutlass.range_constexpr(cols):
                stage[(half * cols + t) * m + row] = values[t]
        fence_tmem(before=True)
        cute.arch.sync_threads()
        partials = view(sm, self.offsets[2], Float32, 1024 * self.slots).iterator
        for p in cutlass.range(tid, self.batch * (m // 4), 256, unroll=1):
            token, r = p // (m // 4), p % (m // 4) * 4
            line = self.lines * token + r // 64
            values4 = load_f32x4(stage.iterator + token * m + r, shared=True)
            dst = partials + line // 16 * 1024 + q * 64 + r % 64
            push_float(dst, values4, bar + 1, line % 16)

    @cute.jit
    def push(self, *args):
        """WaveMoeTail's shared inputs: peer-store this rank's shared-expert rows [B,
        7168] into the RS mailboxes (-0 made +0); an ordinary launch after the down
        GEMM."""
        launch(self.push_kernel, args, self.batch, 128, pdl=False)

    @cute.kernel
    def push_kernel(self, args):
        shared, peers, parity, rank, capacity = args
        tid, _, _, token = thread()
        for i in cutlass.range_constexpr(7):
            column = 8 * (tid + 128 * i)
            words = load_words(shared.iterator + token * 7168 + column, count=4)
            offset = ((parity * self.tp + rank) * capacity + token) * self.shard
            offset += column % self.shard
            packed = tuple(no_neg_zero(w) for w in words)
            store_relaxed(peers[column // self.shard] + Int64(offset) * 2, packed)
