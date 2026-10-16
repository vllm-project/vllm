# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""O-projection producer for the split and fused AttnRes paths: 112 CTAs of 64 output
rows. Each warp keeps ``rows`` 16-row weight tiles of one of ``4 * rows`` K parts in
registers (TP8 stages some K blocks in shared memory), runs BF16 MMA chains and the K
parts are reduced in order. Weights load before the PDL wait; the split producer
triggers its consumer right after it. With ``group_tiles`` the batch runs in groups of
that many 8-token tiles: each group's x arrives through a double-buffered cp.async ring
during the previous group's MMAs, and is reduced and published during the next one's.
With ``twoshot`` the partial chunks go to their owners' receive slots instead (the
first shot of ``common.twoshot``; args 2-4: receive view, owners' bases, rank).
``OprojUmma`` is the tcgen05 engine of
the same producer for B>8."""

import cutlass
import cutlass.cute as cute
from cutlass import Constexpr, Float32, Int64, Uint8, Uint32, Uint64

from ..common.primitives import (
    add_rn,
    alloc,
    at,
    bits_float,
    copy_g2s,
    fence_tmem,
    launch,
    load_tmem,
    load_vector,
    mma_bf16_block,
    mma_into,
    multicast_words,
    named_barrier,
    pack_pair,
    prefetch_bytes,
    release_tmem,
    smem_desc,
    smem_words,
    stage_weight,
    store_shared_words,
    thread,
    view,
    wait,
)
from ..common.twoshot import push


def pack8(values, policy="plain"):
    """Four BF16x2 words of ``values[0:8]``."""
    return tuple(
        pack_pair(values[2 * j], values[2 * j + 1], policy=policy) for j in range(4)
    )


def tree_reduce(values, op):
    tree = list(values)
    for stride in (1, 2, 4):
        for block in range(0, len(tree) - stride, 2 * stride):
            tree[block] = op(tree[block], tree[block + stride])
    return tree[0]


class OprojProducer:
    def __init__(self, k, *, staged_blocks, rows=1, group_tiles=0, twoshot=False):
        self.k, self.staged, self.rows = k, staged_blocks, rows
        self.twoshot = twoshot
        self.parts = 4 * rows
        self.kblocks = k // (32 * self.parts)
        self.register_blocks = self.kblocks - self.staged
        self.group_tiles = group_tiles
        # Single pass: TP8 copies the batch to shared memory, TP16 loads registers.
        self.shared_x = self.staged > 0 and not group_tiles
        self.register_x = k == 768 and not group_tiles
        self.x_stride = k // 8 + 4
        self.chains = max(1, 4 // (rows * max(1, group_tiles)))

    def schedule(self, batches):
        """(8-token tiles, tiles per group, groups) of ``batches`` tokens."""
        tiles = (batches + 7) // 8
        per_group = self.group_tiles or tiles
        return tiles, per_group, (tiles + per_group - 1) // per_group

    @cute.jit
    def allocate(self, batches: cutlass.Constexpr[int]):
        nt, gt, ng = self.schedule(batches)
        smem = cutlass.utils.SmemAllocator()
        red = alloc(smem, Float32, (gt, 4, self.parts, 8, 20))
        rows = nt * 8 if self.shared_x else 0
        if cutlass.const_expr(self.group_tiles):
            rows = min(2, ng) * gt * 8
        xs = alloc(smem, Uint32, rows * self.x_stride * 4 if rows else 1)
        staged = alloc(smem, Uint32, max(1, 16 * self.staged * self.rows * 2 * 32 * 4))
        return red, xs, staged

    @cute.jit
    def load_weight(self, weight, staged, cta):
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group, part = lane // 4, lane % 4
        tile = cta * 4 + warp // self.parts * self.rows
        kq = warp % self.parts
        base = (tile * 16 + group) * self.k + kq * self.kblocks * 32 + part * 8
        if cutlass.const_expr(self.staged > 0):
            for j in cutlass.range_constexpr(self.staged):
                for t in cutlass.range_constexpr(self.rows):
                    unit = (warp * self.staged + j) * self.rows + t
                    dst = staged.iterator + (unit * 2 * 32 + lane) * 4
                    src = weight.iterator + base + t * 16 * self.k
                    src += (self.register_blocks + j) * 32
                    stage_weight(dst, src)
                    stage_weight(dst + 128, src + 8 * self.k)
            cute.arch.cp_async_commit_group()
        wr = cute.make_rmem_tensor((self.register_blocks, self.rows, 2, 4), Uint32)
        for j in cutlass.range_constexpr(self.register_blocks):
            for t in cutlass.range_constexpr(self.rows):
                for r in cutlass.range_constexpr(2):
                    src = weight.iterator + base + (t * 2 + r) * 8 * self.k + j * 32
                    words = load_vector(src, cache="weight")
                    for v in cutlass.range_constexpr(4):
                        wr[j, t, r, v] = words[v]
        return wr

    @cute.jit
    def stage_x(self, x, xs, first: Constexpr[int], buffer: Constexpr[int]):
        """One cp.async group: x tokens [first, first + group) into ring ``buffer``."""
        tid, _, _ = cute.arch.thread_idx()
        chunks: Constexpr = self.k // 8
        count: Constexpr = min(x.shape[0] - first, self.group_tiles * 8)
        rows: Constexpr = buffer * self.group_tiles * 8
        for i in cutlass.range_constexpr((count * chunks + 511) // 512):
            chunk = tid + i * 512
            token, col = chunk // chunks, chunk % chunks
            dst = xs.iterator + ((rows + token) * self.x_stride + col) * 4
            if cutlass.const_expr((i + 1) * 512 <= count * chunks):
                copy_g2s(dst, at(x, first + token, col * 8))
            else:
                if chunk < count * chunks:  # noqa: SIM102
                    copy_g2s(dst, at(x, first + token, col * 8))
        cute.arch.cp_async_commit_group()

    def x_words(self, xs, row, kofs, j):
        return smem_words(xs.iterator + (row * self.x_stride + kofs // 8 + j * 4) * 4)

    @cute.jit
    def multiply(self, smem, wr, xr, first: Constexpr, buffer: Constexpr, M: Constexpr):
        """FP32 chains of tokens [first, first + group); x from ``xr`` (registers) or
        shared-memory rows (ring ``buffer``)."""
        red, xs, staged = smem
        _, gt, _ = self.schedule(M)
        tiles: Constexpr = min(gt, (M - first + 7) // 8)
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group, part = lane // 4, lane % 4
        kofs = warp % self.parts * self.kblocks * 32 + part * 8
        acc = cute.make_rmem_tensor((gt, self.rows, self.chains, 4), Float32)
        acc.fill(0)
        for j in cutlass.range_constexpr(self.kblocks):
            wa, wb = [], []
            for t in cutlass.range_constexpr(self.rows):
                if cutlass.const_expr(j < self.register_blocks):
                    wa.append(tuple(wr[j, t, 0, v] for v in range(4)))
                    wb.append(tuple(wr[j, t, 1, v] for v in range(4)))
                else:
                    sj: Constexpr = j - self.register_blocks
                    unit = (warp * self.staged + sj) * self.rows + t
                    ptr = staged.iterator + (unit * 2 * 32 + lane) * 4
                    wa.append(smem_words(ptr))
                    wb.append(smem_words(ptr + 128))
            for nt in cutlass.range_constexpr(tiles):
                if cutlass.const_expr(self.register_x):
                    xv = tuple(xr[nt, j, v] for v in range(4))
                else:
                    row = buffer * gt * 8 + nt * 8
                    if cutlass.const_expr(self.group_tiles):
                        xv = self.x_words(xs, row + group, kofs, j)
                    else:
                        xv = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        if nt * 8 + group < M:
                            xv = self.x_words(xs, row + group, kofs, j)
                for t in cutlass.range_constexpr(self.rows):
                    for half in cutlass.range_constexpr(2):
                        chain: Constexpr = (j * 2 + half) % self.chains
                        a: tuple = (wa[t][2 * half], wb[t][2 * half])
                        a += (wa[t][2 * half + 1], wb[t][2 * half + 1])
                        b = (xv[2 * half], xv[2 * half + 1])
                        mma_into(acc[nt, t, chain, None], a, b, bf16=True)
        return acc

    @cute.jit
    def store_partials(self, red, acc, tiles: Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group, part = lane // 4, lane % 4
        tile, kq = warp // self.parts * self.rows, warp % self.parts
        for nt in cutlass.range_constexpr(tiles):
            for t in cutlass.range_constexpr(self.rows):
                for v in cutlass.range_constexpr(4):
                    chains = [acc[nt, t, c, v] for c in range(self.chains)]
                    token, row = 2 * part + v % 2, group + (v // 2) * 8
                    red[nt, tile + t, kq, token, row] = tree_reduce(chains, add_rn)

    @cute.jit
    def publish(self, args, red, cta, first: Constexpr[int]):
        """Sum the K parts of tokens [first, first + group) in order; multicast them."""
        x, _, mailbox, mc, slot = args[:5]
        _, gt, _ = self.schedule(x.shape[0])
        count: Constexpr = min(x.shape[0] - first, gt * 8)
        tid, _, _ = cute.arch.thread_idx()
        if tid < 8 * count:
            reduced = cute.make_rmem_tensor(8, Float32)
            if cutlass.const_expr(self.group_tiles):
                # Token-major lanes: conflict-free 16-byte partial loads.
                token, out_half = tid % count, tid // count % 2
                row_tile = tid // count // 2
                tile, row = token // 8, token % 8
                for i in cutlass.range_constexpr(2):
                    column = out_half * 8 + i * 4
                    words = smem_words(at(red, tile, row_tile, 0, row, column))
                    sums = [bits_float(w) for w in words]
                    for q in cutlass.range_constexpr(1, self.parts):
                        words = smem_words(at(red, tile, row_tile, q, row, column))
                        sums = [add_rn(s, bits_float(w)) for s, w in zip(sums, words)]
                    for v in cutlass.range_constexpr(4):
                        reduced[i * 4 + v] = sums[v]
            else:
                out_half, row_tile, token = tid % 2, (tid // 2) % 4, tid // 8
                tile, row = token // 8, token % 8
                for v in cutlass.range_constexpr(8):
                    value = red[tile, row_tile, 0, row, out_half * 8 + v]
                    for q in cutlass.range_constexpr(1, self.parts):
                        part = red[tile, row_tile, q, row, out_half * 8 + v]
                        value = add_rn(value, part)
                    reduced[v] = value
            words = pack8(reduced, "latent")
            out_col = (cta * 4 + row_tile) * 16 + out_half * 8
            if cutlass.const_expr(self.twoshot):
                push(mailbox, mc, slot, first + token, out_col, words)
            else:
                offset = cute.crd2idx((slot, first + token, out_col), mailbox.layout)
                multicast_words(mc + Int64(offset) * 2, words)

    @cute.jit
    def compute(self, args, smem, wr, cta, trigger_after_wait: cutlass.Constexpr[bool]):
        x = args[0]
        red, xs, staged = smem
        M: Constexpr = x.shape[0]
        NT, GT, NG = self.schedule(M)
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        group, part = lane // 4, lane % 4
        kofs = warp % self.parts * self.kblocks * 32 + part * 8
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(trigger_after_wait):
            cute.arch.griddepcontrol_launch_dependents()
        xr = cute.make_rmem_tensor((NT, self.kblocks, 4), Uint32)
        if cutlass.const_expr(self.group_tiles):
            self.stage_x(x, xs, 0, 0)
        if cutlass.const_expr(self.shared_x):
            for chunk in range(tid, M * (self.k // 8), 512):
                token, col = chunk // (self.k // 8), chunk % (self.k // 8)
                words = load_vector(at(x, token, col * 8))
                store_shared_words(
                    xs.iterator + (token * self.x_stride + col) * 4, words
                )
            cute.arch.sync_threads()
        if cutlass.const_expr(self.register_x):
            xr.fill(0)
            for nt in cutlass.range_constexpr(NT):
                token = nt * 8 + group
                if token < M:
                    for j in cutlass.range_constexpr(self.kblocks):
                        words = load_vector(at(x, token, kofs + j * 32))
                        for v in cutlass.range_constexpr(4):
                            xr[nt, j, v] = words[v]
        if cutlass.const_expr(self.staged > 0 and not self.group_tiles):
            cute.arch.cp_async_wait_group(0)
        for g in cutlass.range_constexpr(NG):
            if cutlass.const_expr(self.group_tiles):
                # x(g) and group g - 1's partials are visible: stage x(g + 1), publish.
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if cutlass.const_expr(g + 1 < NG):
                    self.stage_x(x, xs, (g + 1) * GT * 8, (g + 1) % 2)
                if cutlass.const_expr(g > 0):
                    self.publish(args, red, cta, (g - 1) * GT * 8)
            acc = self.multiply(smem, wr, xr, g * GT * 8, g % 2, M)
            if cutlass.const_expr(self.group_tiles):
                cute.arch.sync_threads()
            self.store_partials(red, acc, min(GT, NT - g * GT))
        cute.arch.sync_threads()
        self.publish(args, red, cta, (NG - 1) * GT * 8)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, 112, 512, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        cta, _, _ = cute.arch.block_idx()
        smem = self.allocate(args[0].shape[0])
        wr = self.load_weight(args[1], smem[2], cta)
        self.compute(args, smem, wr, cta, True)


class OprojUmma(OprojProducer):
    """B>8 producer on tcgen05: 112 CTAs x 64 rows, tokens on N. Four loader warps
    stream 128B-swizzled W [64, 64] and x [N, 64] K blocks through a cp.async ring
    (W of the first stages before the PDL wait, the rest prefetched to L2), the MMA
    warp accumulates D [64, N] FP32 in TMEM with M=64 kind::f16 MMAs, and the loader
    warps round D to BF16 and publish it like ``OprojProducer``."""

    def __init__(self, k, *, twoshot=False, budget=184 * 1024):
        super().__init__(k, staged_blocks=0, twoshot=twoshot)
        self.blocks, self.budget = k // 64, budget

    def geometry(self, batches):
        """(N, ring stage bytes, stages, TMEM columns)."""
        n = (batches + 7) // 8 * 8
        stage = 8192 + n * 128
        stages = min(self.blocks, self.budget // stage)
        return n, stage, stages, max(32, 1 << (n - 1).bit_length())

    @cute.jit
    def load(self, source, ring, rows: Constexpr, valid: Constexpr, row0, column):
        """cp.async ``rows`` x 64 BF16 of ``source`` from (row0, column) into a K-major
        128B-swizzled tile; rows >= ``valid`` are left alone. ``valid`` < 0: weights
        (evict-first, every row)."""
        tid, _, _ = cute.arch.thread_idx()
        for i in cutlass.range_constexpr((rows * 8 + 127) // 128):
            c = tid + 128 * i
            row, chunk = c // 8, c % 8
            dst = ring + row // 8 * 1024 + row % 8 * 128 + (chunk ^ row % 8) * 16
            src = at(source, row0 + row, column + chunk * 8)
            if cutlass.const_expr(valid < 0):
                stage_weight(dst, src)
            elif cutlass.const_expr((i + 1) * 128 <= valid * 8):
                copy_g2s(dst, src)
            else:
                if c < valid * 8:  # noqa: SIM102
                    copy_g2s(dst, src)

    @cute.jit
    def multiply(self, ring, bars, tmem, n: Constexpr, stage: Constexpr):
        """The MMA warp: wait for each K block, issue its four K16 MMAs from the
        elected lane and release its stage (the last block: the accumulator)."""
        stages: Constexpr = self.geometry(n)[2]
        # kind::f16 descriptor: FP32 D, BF16 A and B, both K-major, N, M = 64
        instruction: Constexpr = 1 << 4 | 1 << 7 | 1 << 10 | n >> 3 << 17 | 4 << 24
        for kb in cutlass.range_constexpr(self.blocks):
            s: Constexpr = kb % stages
            wait(bars.iterator + s, kb // stages % 2)
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem()
            a = smem_desc(ring.iterator + s * stage, True)
            b = smem_desc(ring.iterator + s * stage + 8192, True)
            last: Constexpr = kb + 1 == self.blocks
            release: Constexpr = last or kb + stages < self.blocks
            barrier = bars.iterator + (2 * stages if last else stages + s)
            mma_bf16_block(tmem, a, b, instruction, int(kb > 0), barrier, int(release))

    @cute.jit
    def epilogue(self, args, ring, tmem, cta, n: Constexpr):
        """D rows 16 * warp + lane (lane < 16: M=64 TMEM layout) -> FP32 [token, row]
        tile in the drained ring -> 8-row BF16 chunks to the mailbox / partials."""
        x, _, mailbox, mc, slot = args[:5]
        M: Constexpr = x.shape[0]
        tid, warp, lane, _ = thread()
        values = load_tmem(tmem + (32 * warp << 16), self.geometry(M)[3])
        tile = view(ring.iterator, 0, Float32, (n, 68), (68, 1))  # [token, row]
        if lane < 16:
            for t in cutlass.range_constexpr(M):
                tile[t, 16 * warp + lane] = values[t]
        named_barrier(1, 128)
        for i in cutlass.range_constexpr((M * 8 + 127) // 128):
            c = tid + 128 * i
            if c < M * 8:
                token, part = c // 8, c % 8
                reduced = cute.make_rmem_tensor(8, Float32)
                for h in cutlass.range_constexpr(2):
                    words = smem_words(at(tile, token, part * 8 + h * 4))
                    for v in cutlass.range_constexpr(4):
                        reduced[h * 4 + v] = bits_float(words[v])
                words = pack8(reduced, "latent")
                column = cta * 64 + part * 8
                if cutlass.const_expr(self.twoshot):
                    push(mailbox, mc, slot, token, column, words)
                else:
                    offset = cute.crd2idx((slot, token, column), mailbox.layout)
                    multicast_words(mc + Int64(offset) * 2, words)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, 112, 160, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, weight = args[0], args[1]
        n, stage, stages, columns = self.geometry(x.shape[0])
        tid, warp, _, cta = thread()
        smem = cutlass.utils.SmemAllocator()
        ring = alloc(smem, Uint8, stages * stage, 1024)
        bars = alloc(smem, Uint64, 2 * stages + 1, 8)
        tmem_slot = alloc(smem, Uint32, 1)
        if tid == 0:
            for s in cutlass.range_constexpr(stages):
                cute.arch.mbarrier_init(bars.iterator + s, 256)
                cute.arch.mbarrier_init(bars.iterator + stages + s, 1)
            cute.arch.mbarrier_init(bars.iterator + 2 * stages, 1)
            cute.arch.mbarrier_init_fence()
        if warp == 4:
            cute.arch.alloc_tmem(columns, tmem_slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_slot[0]
        row0 = cta * 64
        if warp < 4:  # weights do not depend on the previous kernel
            if cutlass.const_expr(stages < self.blocks) and tid < 64:
                rest = at(weight, row0 + tid, stages * 64)
                prefetch_bytes(rest, (self.blocks - stages) * 128)
            for s in cutlass.range_constexpr(stages):
                self.load(weight, ring.iterator + s * stage, 64, -1, row0, s * 64)
                cute.arch.cp_async_mbarrier_arrive_noinc(bars.iterator + s)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if warp < 4:
            for kb in cutlass.range_constexpr(self.blocks):
                slot: Constexpr = kb % stages
                tile = ring.iterator + slot * stage
                if cutlass.const_expr(kb >= stages):
                    wait(bars.iterator + stages + slot, (kb // stages - 1) % 2)
                    self.load(weight, tile, 64, -1, row0, kb * 64)
                    cute.arch.cp_async_mbarrier_arrive_noinc(bars.iterator + slot)
                self.load(x, tile + 8192, n, x.shape[0], 0, kb * 64)
                cute.arch.cp_async_mbarrier_arrive_noinc(bars.iterator + slot)
            wait(bars.iterator + 2 * stages, 0)
            fence_tmem()
            # The MMAs (async proxy) read the ring the epilogue now overwrites.
            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilogue(args, ring, tmem, cta, n)
        else:
            self.multiply(ring, bars, tmem, n, stage)
        release_tmem(tmem, columns, warp, 4)
