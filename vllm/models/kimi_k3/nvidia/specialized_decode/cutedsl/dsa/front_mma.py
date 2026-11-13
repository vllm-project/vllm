# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DSA indexer front on tcgen05: the wq_b projection (N 4096, K 1536) and
DsaIndex (the FP8 indexer query, its folded head weights and the indexer key insert)
in one kernel, from MlaQuery's normalized q_c rows (serial / wait query overlap) or
(`normalized=False`) from the packed rows with the q_a RMSNorm in the kernel.

A CTA owns one indexer head (its 128 wq_b rows: tcgen05 M = 128, swap-AB) for one
tile of `n` rows (N) and one of `split` K slices; the K slices of a head and row
tile, times `group` heads, form one cluster.

- Warp 5 (TMA): the first stages' 128 x 64 weight blocks before the PDL wait (L2
  evict_first, a device tensor map), then the rows' 64-column blocks (zero past the
  batch) and the later weight blocks into the ring; then indexer keys (DsaIndex's
  code), the grid's CTAs taking the tokens round robin.
- Warp 4: per block, the K16 BF16 MMAs into TMEM (FP32, column j = row j).
- Warps 0-3, from packed rows only: the RMSNorm factors (QueryNorm.tiled's packed FMA
  order; a cluster shares its rows' factors over DSMEM), then each block normalized
  in place (BF16 (x * rr) * gamma, as QueryNorm).
- Warps 0-3 (thread = channel): with `split` > 1 the partial columns go to their K
  slice's owner (DSMEM), which adds the `split` partials in a fixed order; each
  column is rounded to BF16 (the projection's output) and quantized as DsaIndex
  does: the head amax over the 128 channels, a power-of-two scale (exact inverse),
  FP8 bytes and the folded BF16 head weight."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Constexpr, Float32, Int32, Uint8, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync, tcgen05

from ..common.primitives import (
    acquire_tensormap_asm,
    alloc,
    at,
    bf16x2,
    bits_float,
    fence_tmem,
    float_bits,
    fma2,
    fp8_bits,
    load_bf16x8,
    load_tma2,
    load_tmem,
    load_words,
    mma_bf16_block,
    mul2,
    named_barrier,
    push_float,
    query_norm_rsqrt,
    redux,
    release_tmem,
    smem_desc,
    smem_words,
    store_shared_words,
    unpack_bf16,
    wait,
    warp_sum,
)
from ..mla.query import pow2_ceil
from .index import (
    DIM,
    FP4_SCALES,
    FP4_WEIGHT,
    HEADS,
    e2m1x2,
    insert_key,
    is_fp4_cache,
    mxfp4_scale,
    store_b8,
)

K = 1536
WEIGHT_BLOCK = 128 * 64 * 2  # one head's 128 rows x 64 K, BF16
SMEM_BUDGET = 200 * 1024  # shared memory for the stages and the mailbox (bytes)


def front_tiling(batch: int):
    """(n, tiles, split, group) of `batch` rows: the rows per tile (a multiple of
    16), the row tiles, the K slices and the CTAs of a cluster along the heads. Up
    to 128 CTAs: K slices while the weights dominate, row tiles from there (clusters
    of at most 4: 8-CTA clusters of these ~200 KB CTAs do not all fit at once).
    No tile lies past the batch (129-144 rows: 3 tiles of 48, not 4)."""
    pad = lambda rows: max(16, -(-rows // 16) * 16)  # noqa: E731
    if batch <= 48:
        return pad(batch), 1, 4, 1
    if batch <= 128:
        n = pad(-(-batch // 2))
        return n, -(-batch // n), 2, 1
    n = pad(-(-batch // 4))
    return n, -(-batch // n), 1, 4


def load_columns(address, count):
    """`count` FP32 TMEM columns of this lane from `address` (power-of-two loads
    of at most 64)."""
    values: list = []
    while count:
        piece = min(64, 1 << (count.bit_length() - 1))
        loaded = load_tmem(address + len(values), piece)
        values += list(loaded) if piece > 1 else [loaded]
        count -= piece
    return values


class IndexFrontMma:
    """`batch` rows; `key`/`weight`: the wk and wv column offsets in the packed
    projection row; `mx`: key scales as packed UE8M0 (MX) bytes; the tiling
    (n, tiles, split, group) is `front_tiling`'s; `normalized`: the
    rows are the normalized q_c; `row_pitch`: their pitch (elements)."""

    def __init__(
        self,
        batch: int,
        key: int,
        weight: int,
        mx: bool = False,
        normalized: bool = False,
        row_pitch: int = 0,
    ):
        self.batch, self.key, self.weight, self.mx = batch, key, weight, mx
        # Elements between two input rows (q_c rows: 1536; packed rows: their pitch).
        self.row_pitch = row_pitch or K
        self.normalized = normalized
        n, tiles, split, group = front_tiling(batch)
        # Every tile holds rows (an empty one would arm its row barrier with a
        # negative byte count).
        assert n % 16 == 0 and n <= 256 and n * (tiles - 1) < batch <= n * tiles
        assert K % (64 * split) == 0 and HEADS % group == 0
        self.n, self.tiles, self.split, self.group = n, tiles, split, group
        self.cluster = split * group
        assert self.cluster <= 8
        self.blocks = K // 64 // split  # K blocks per CTA
        self.owned = n // split  # columns a CTA reduces and writes
        assert self.owned % 4 == 0
        self.pitch = self.owned + 4  # mailbox row (FP32), 16 B aligned
        mailbox = split * 128 * self.pitch * 4 if split > 1 else 0
        self.xstage = n * 128
        stage = WEIGHT_BLOCK + self.xstage
        self.stages = min(self.blocks, (SMEM_BUDGET - mailbox - n * 4) // stage)
        assert self.stages >= 2
        self.columns = max(32, 1 << (n - 1).bit_length())
        self.ctas = split * HEADS * tiles
        # Small tiles: each CTA computes every row's RMSNorm factor itself.
        self.local_rr = n <= 32 or self.cluster == 1

    @cute.jit
    def __call__(self, *args):
        rows_in = args[1]
        # The row pitch from the host: a one-row tensor's row stride may be
        # canonicalized away, and the TMA descriptor needs it.
        x = cute.make_tensor(
            rows_in.iterator,
            cute.make_layout(
                (cutlass.Int32(self.batch), K),
                stride=(cutlass.Int32(self.row_pitch), 1),
            ),
        )
        atom = sm100_utils.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, BFloat16
        )
        x_layout = cute.tile_to_shape(atom, (self.n, 64, self.stages), (0, 1, 2))
        tma_x, tma_x_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            cute.slice_(x_layout, (None, None, 0)),
            (self.n, 64),
        )
        grid = (self.split, HEADS, self.tiles)
        self.kernel(args[:-1], tma_x, tma_x_tensor, x_layout).launch(
            grid=grid,
            block=(192, 1, 1),
            cluster=(self.split, self.group, 1),
            stream=args[-1],
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def normalize(self, tile, valid, k0, rr, gamma_s):
        """BF16 (x * rr) * gamma in place on this thread's chunks of a K block
        (`k0`: the block's first column in this CTA's K slice; every chunk of a
        thread is the same 8 columns)."""
        tid, _, _ = cute.arch.thread_idx()
        chunk = tid % 8
        gamma = load_bf16x8(gamma_s.iterator + k0 + chunk * 8, shared=True)
        for i in cutlass.range_constexpr(-(-self.n * 8 // 128)):
            row = (tid + 128 * i) // 8
            if row < valid:
                address = tile + row // 8 * 1024 + row % 8 * 128
                address = address + (chunk ^ row % 8) * 16
                r = rr[row]
                words = []
                for j, word in enumerate(smem_words(address)):
                    x = unpack_bf16(word)
                    y = mul2(mul2(x, (r, r)), (gamma[2 * j], gamma[2 * j + 1]))
                    words.append(bf16x2(*y))
                store_shared_words(address, tuple(words))

    @cute.jit
    def row_factors(self, x, rows, eps, lane):
        """QueryNorm.tiled's RMSNorm factors of a few (valid) rows, one warp:
        every load first, then each row's packed FMA chains."""
        loads = []
        for row in rows:
            words = []
            for i in cutlass.range_constexpr(6):
                source = at(x, row, 256 * i + 8 * lane)
                words.append(load_words(source, count=4, mutable=True))
            loads.append(words)
        factors = []
        for words in loads:
            values: list = []
            for four in words:
                for word in four:
                    values += list(unpack_bf16(word))
            zero = Float32(0), Float32(0)
            chains = [zero, zero]  # (0, 1), (2, 3): element e feeds chain e % 4
            for e in cutlass.range_constexpr(0, 48, 2):
                pair = values[e], values[e + 1]
                chains[e % 4 // 2] = fma2(pair, pair, chains[e % 4 // 2])
            (c0, c1), (c2, c3) = chains
            total = warp_sum((c0 + c1) + (c2 + c3))
            factors.append(query_norm_rsqrt(total, eps))
        return factors

    @cute.jit
    def multiply(self, wring, xring, bars, tmem):
        """Warp 4: per K block, the K16 MMAs; release the stage (the last block:
        the accumulator barrier)."""
        stages: Constexpr = self.stages
        n: Constexpr = self.n
        instruction: Constexpr = 1 << 4 | 1 << 7 | 1 << 10 | n >> 3 << 17 | 8 << 24
        x_ready = bars + (3 * stages if self.normalized else stages)
        for kb in cutlass.range(self.blocks, unroll=1):
            slot, phase = kb % stages, kb // stages % 2
            wait(bars + slot, phase)  # the weights (TMA)
            wait(x_ready + slot, phase)  # the (normalized) rows
            cute.arch.fence_proxy("async.shared", space="cta")
            fence_tmem()
            a = smem_desc(wring + slot * WEIGHT_BLOCK, True)
            b = smem_desc(xring + slot * self.xstage, True)
            last = kb + 1 == self.blocks
            release = Int32(last | (kb + stages < self.blocks))
            barrier = bars + 2 * stages + slot
            if last:
                barrier = bars + 4 * stages
            mma_bf16_block(tmem, a, b, instruction, Int32(kb > 0), barrier, release)

    @cute.kernel
    def kernel(self, args, tma_x, mX, x_layout):
        projected, rows_in, w_qa, w_map, norm_w, norm_b, cache, slots = args[0:8]
        eps, key_eps = args[10:12]
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        s, head, tile = cute.arch.block_idx()
        rank = cute.arch.block_idx_in_cluster()
        n, stages, split, owned = self.n, self.stages, self.split, self.owned
        B, CL = self.batch, self.cluster
        smem = cutlass.utils.SmemAllocator()
        wring = alloc(smem, Uint8, stages * WEIGHT_BLOCK, 1024).iterator
        sX = smem.allocate_tensor(
            BFloat16, x_layout.outer, 1024, swizzle=x_layout.inner
        )
        xring = cute.make_ptr(
            Uint8, sX.iterator.toint(), cute.AddressSpace.smem, assumed_align=1024
        )
        mailbox = None
        if cutlass.const_expr(split > 1):
            mailbox = alloc(smem, Float32, (split, 128, self.pitch), 16)
        rr = alloc(smem, Float32, n)
        gamma_s = alloc(smem, BFloat16, K // split)
        wv_s = alloc(smem, Float32, owned)
        amax_s = alloc(smem, Uint32, (4, owned))
        # Per stage: weights full (TMA), rows normalized, stage empty, rows full
        # (TMA); then the accumulator, the RMSNorm factors, the split-K partials.
        bars = alloc(smem, Uint64, 4 * stages + 3, 8).iterator
        tmem_slot = alloc(smem, Uint32, 1)
        bar_acc, bar_rr, bar_red = tuple(bars + 4 * stages + i for i in range(3))
        row0 = tile * n
        valid = cutlass.min(n, B - row0)
        k_first = s * (K // split)
        if tid == 0:
            for i in cutlass.range_constexpr(stages):
                for role in cutlass.range_constexpr(4):
                    count: Constexpr = 128 if role == 1 else 1
                    cute.arch.mbarrier_init(bars + role * stages + i, count)
            for barrier in (bar_acc, bar_rr, bar_red):
                cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_init_fence()
            if cutlass.const_expr(not (self.normalized or self.local_rr)):
                cute.arch.mbarrier_arrive_and_expect_tx(bar_rr, valid * 4)
            if cutlass.const_expr(split > 1):
                expected: Constexpr = (split - 1) * 128 * owned * 4
                cute.arch.mbarrier_arrive_and_expect_tx(bar_red, expected)
        if warp == 4:
            cute.arch.alloc_tmem(self.columns, tmem_slot.iterator)
            cute.arch.relinquish_tmem_alloc_permit()
        fence_tmem(before=True)
        cute.arch.sync_threads()
        fence_tmem()
        tmem = tmem_slot[0]
        if cutlass.const_expr(CL > 1):
            cute.arch.cluster_arrive_relaxed()  # the barriers before DSMEM stores
        if warp == 5:  # the first stages' weights do not depend on the last kernel
            acquire_tensormap_asm(w_map)
            cpasync.prefetch_descriptor(tma_x)
            if lane == 0:
                for kb in cutlass.range_constexpr(stages):
                    self.load_weights(wring, bars, w_map, kb, k_first, head)
        if cutlass.const_expr(not self.normalized):  # noqa: SIM102
            if warp < 4:  # this K slice's q_a norm weights
                for c in range(tid, K // split // 8, 128):
                    words = load_words(at(w_qa, k_first + c * 8), count=4)
                    store_shared_words(gamma_s.iterator + c * 8, words)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(CL > 1):
            cute.arch.cluster_wait()
        if warp == 5:
            gX = cute.local_tile(mX, (n, 64), (None, None))
            tXsX, tXgX = cpasync.tma_partition(
                tma_x,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX, 0, 2),
            )
            for kb in cutlass.range(self.blocks, unroll=1):
                slot = kb % stages
                if kb >= stages:
                    wait(bars + 2 * stages + slot, (kb // stages - 1) % 2)
                    if lane == 0:
                        self.load_weights(wring, bars, w_map, kb, k_first, head)
                x_full = bars + 3 * stages + slot
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(x_full, self.xstage)
                cute.copy(
                    tma_x,
                    tXgX[(None, tile, s * self.blocks + kb)],
                    tXsX[(None, slot)],
                    tma_bar_ptr=x_full,
                )
            # The indexer keys, the grid's CTAs taking the tokens round robin.
            first = (tile * HEADS + head) * split + s
            for token in range(first, B, self.ctas):
                key = projected, self.key, norm_w, norm_b, cache, slots, token, key_eps
                insert_key(*key, self.mx, lane)
        elif warp == 4:
            self.multiply(wring, xring, bars, tmem)
        else:
            column = s * owned + tid  # this CTA's columns' head weight inputs
            wv = Float32(0)
            if (tid < owned) & (column < valid):
                wv = Float32(projected[row0 + column, self.weight + head])
            if cutlass.const_expr(not self.normalized):
                self.factors(rows_in, rr, bar_rr, rank, row0, valid, eps)
                for kb in cutlass.range(self.blocks, unroll=1):
                    slot = kb % stages
                    wait(bars + 3 * stages + slot, kb // stages % 2)
                    tile_x = xring + slot * self.xstage
                    self.normalize(tile_x, valid, kb * 64, rr, gamma_s)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.mbarrier_arrive(bars + stages + slot)
            if tid < owned:
                wv_s[tid] = wv
            wait(bar_acc, 0)
            # tcgen05.ld is .sync.aligned: reconverge after the per-thread wait.
            cute.arch.sync_warp()
            fence_tmem()
            self.epilogue(args, tmem, mailbox, amax_s, wv_s, bar_red, row0, valid)
        release_tmem(tmem, self.columns, warp, 4)
        if cutlass.const_expr(CL > 1):
            cute.arch.cluster_arrive_relaxed()  # no CTA exits while a peer writes it
            cute.arch.cluster_wait()

    @cute.jit
    def factors(self, rows_in, rr, bar_rr, rank, row0, valid, eps):
        """Warps 0-3: the tile's RMSNorm factors into `rr`: every row here (small
        tiles, one-CTA clusters), else this CTA's share of the rows, sent to every
        CTA of the cluster."""
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        CL = self.cluster
        share: Constexpr = 1 if self.local_rr else CL
        per_warp: Constexpr = -(-self.n // (share * 4))
        for m0 in cutlass.range_constexpr(0, per_warp, 4):
            rows = []
            for m in cutlass.range_constexpr(m0, min(per_warp, m0 + 4)):
                row = (warp + 4 * m) * share
                if cutlass.const_expr(share > 1):
                    row = row + rank
                rows.append(row)
            clamped = [row0 + cutlass.min(row, valid - 1) for row in rows]
            factors = self.row_factors(rows_in, clamped, eps, lane)
            for row, factor in zip(rows, factors):
                if row < valid:
                    if cutlass.const_expr(share == 1):
                        if lane == 0:
                            rr[row] = factor
                    elif lane < CL:
                        push_float(rr.iterator + row, factor, bar_rr, lane)
        if cutlass.const_expr(share > 1):
            wait(bar_rr, 0)
        named_barrier(1, 128)

    @cute.jit
    def load_weights(self, wring, bars, w_map, kb, k_first, head):
        """One lane: block kb's 128 x 64 weight box on its full barrier."""
        slot = kb % self.stages
        full = bars + slot
        cute.arch.mbarrier_arrive_and_expect_tx(full, WEIGHT_BLOCK)
        tile_w = wring + slot * WEIGHT_BLOCK
        load_tma2(tile_w, w_map, full, k_first + kb * 64, head * 128)

    @cute.jit
    def epilogue(self, args, tmem, mailbox, amax_s, wv_s, bar_red, row0, valid):
        """Warps 0-3 (thread = channel): the split-K reduction of the owned columns,
        then per column BF16, the head amax, FP8 and the folded weight."""
        q_out, w_out = args[8], args[9]
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        s, head, _ = cute.arch.block_idx()
        split, owned = self.split, self.owned
        lane_base = tmem + (32 * warp << 16)
        if cutlass.const_expr(split > 1):
            group = cute.arch.block_idx_in_cluster() - s  # rank of K slice 0
            loaded = load_columns(lane_base, self.n)
            for o in cutlass.range_constexpr(split):
                for c in cutlass.range_constexpr(0, owned, 4):
                    four = tuple(loaded[o * owned + c + j] for j in range(4))
                    target = at(mailbox, s, tid, c)
                    if o == s:
                        store_shared_words(target, tuple(float_bits(v) for v in four))
                    else:
                        push_float(target, four, bar_red, group + o)
            wait(bar_red, 0)
            named_barrier(1, 128)
        # Pass 1: BF16 (the projection's output) and each warp's column amax.
        for g in cutlass.range(owned // 4, unroll=1):
            values = self.reduced(lane_base, mailbox, s, g * 4)
            for j in cutlass.range_constexpr(4):
                top = redux(float_bits(abs(values[j])), "max")
                if lane == 0:
                    amax_s[warp, g * 4 + j] = top
        named_barrier(1, 128)
        # Pass 2: the head scale (a power of two: its inverse is exact), FP8 bytes
        # and the folded head weight.
        for g in cutlass.range(owned // 4, unroll=1):
            values = self.reduced(lane_base, mailbox, s, g * 4)
            for j in cutlass.range_constexpr(4):
                c = g * 4 + j
                column = s * owned + c
                if cutlass.const_expr(is_fp4_cache(args[6])):
                    if column < valid:  # MXFP4 (index.py): one warp, one block
                        amax = bits_float(amax_s[warp, c])
                        inverse, byte = mxfp4_scale(amax)
                        value = values[j] * inverse
                        pair = cute.arch.shuffle_sync_bfly(value, 1)
                        row = at(q_out, row0 + column, 0, 0)
                        if tid % 2 == 0:
                            store_b8(
                                row + head * (DIM // 2) + tid // 2, e2m1x2(value, pair)
                            )
                        if lane == 0:
                            store_b8(row + FP4_SCALES + head * 4 + warp, byte)
                        if tid == 0:
                            weight = wv_s[c] * Float32(FP4_WEIGHT)
                            w_out[row0 + column, head] = BFloat16(weight)
                elif column < valid:
                    amax = Float32(1e-10)
                    for w in cutlass.range_constexpr(4):
                        amax = cute.arch.fmax(amax, bits_float(amax_s[w, c]))
                    scale = pow2_ceil(amax / Float32(448))  # amax >= 1e-10
                    inverse = bits_float((Uint32(254) << 23) - float_bits(scale))
                    b = row0 + column
                    q_out[b, head, tid] = fp8_bits(values[j] * inverse)
                    if tid == 0:
                        fold = (wv_s[c] * scale) * Float32((HEADS * DIM) ** -0.5)
                        w_out[b, head] = BFloat16(fold)

    @cute.jit
    def reduced(self, lane_base, mailbox, s, c):
        """Columns c..c+3 of this thread's channel, BF16-rounded: TMEM (one K
        slice) or the `split` partials in order."""
        tid, _, _ = cute.arch.thread_idx()
        values = cute.make_rmem_tensor(4, Float32)
        if cutlass.const_expr(self.split == 1):
            four = load_tmem(lane_base + c, 4)
            for j in cutlass.range_constexpr(4):
                values[j] = four[j]
        else:
            for j in cutlass.range_constexpr(4):
                total = mailbox[0, tid, c + j]
                for o in cutlass.range_constexpr(1, self.split):
                    total += mailbox[o, tid, c + j]
                values[j] = total
        for j in cutlass.range_constexpr(4):
            values[j] = Float32(BFloat16(values[j]))
        return values
