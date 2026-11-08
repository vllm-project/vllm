# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused decode indexer of a K3.1 DSA layer. TP (every rank scores the whole
context): the FP8 indexer query and head weights of the step's rows over the
paged FP8 indexer cache -> the exact top-k of every row as the physical KV rows
the sparse attention reads, compacted to a prefix, and their counts. Two
PDL-chained kernels replace the stock chain (DeepGEMM logits, FP32 upcast, top-k,
then the logical -> physical conversion on a side stream):

  IndexerScore   key-balanced persistent tcgen05 scorer: the 128-key tiles of
                 all requests are cut into equal ranges, one per SM, and one
                 [128 keys x q * 32] FP8 MMA per tile scores every verify row of
                 the request, so each key page is read once per step. The
                 epilogue rounds each head's relu(q . k) to BF16, sums the heads
                 with the BF16 head weights by BF16 FMAs (stock DeepGEMM
                 precision, see below), scales by the key's UE8M0 scale, and
                 stores the BF16 logits into a request-compact
                 scratch, and each 8 keys' largest (order-preserving 16-bit) key
                 into a scratch 1/8 its size. Requests whose rows all fit the
                 top-k are not scored.
  IndexerSelect  one CTA per row: the K-th largest of the row's 8-key maxima
                 bounds its K-th largest key from below, so only the keys of the
                 about K vectors at or above it are candidates; an exact radix
                 select over them (11-bit bins, then the low 5 bits) and one
                 ordered pass write the selected keys' physical rows in key
                 order. Ties at the threshold go to the lowest key indices, so
                 the result is deterministic. A row of at most `topk` keys
                 takes all of them without reading logits (the short-context
                 bypass).

Under DCP the scorer writes this rank's shard in DeepGEMM's logits layout and
(FusedIndexer.packs) the select writes the #63 merge's B' rows (run_dcp).

logits[t, k] = scale[k] * sum_h bf16(relu(q[t, h] . key[k])) * w[t, h] at stock's
precision (DeepGEMM SM100 paged MQA logits): each head's value rounded to BF16 and
accumulated by BF16 FMAs (two BF16x2 sums, then a BF16 add), on the FP8 query
with unit MX scales and the BF16 head weights that fold its scale (DsaIndex).
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_stream

from ..common.host import dynamic_fake, dynamic_view, fake, stream, to_cute
from ..common.primitives import (
    asm,
    bits_float,
    copy_scale,
    fence_tmem,
    float_bits,
    load_words,
    shared_address,
    smem_words,
    store_shared_words,
)
from .index import FP4_ENTRY, fp4_query_views, is_fp4_cache
from .topk_exchange import ll_push

PAGE = 64  # keys per indexer cache block
DIM = 128
HEADS = 32
TILE = 128  # keys per MMA tile (two pages)
ITEM_TILES = 16  # tiles per work item (the producer caches 32 page ids)
MAX_REQUESTS = 512  # tile prefix sum in shared memory
STAGES = 6
FP4_STAGES = 12
EPI_GROUPS = 2  # TMEM accumulators, each tile's epilogue on one of them
# Epilogue warpgroups per accumulator, each taking a share of the tile's verify
# rows: the BF16 head sum (stock's FMA order) is latency bound, more warps hide it.
EPI_SPLIT = 1
EPI_WARPS = 4 * EPI_GROUPS * EPI_SPLIT
PRODUCER = EPI_WARPS
MMA_WARP = PRODUCER + 1
SCORE_THREADS = 32 * (MMA_WARP + 1)

# MXFP4 keys and query (FP4_ENTRY-byte indexer cache): 64-byte E2M1 rows (128B-
# aligned 64B-swizzled K-major operands), UE8M0 x4 words in TMEM (tcgen05.cp).
FP4_ROW = DIM // 2
FP4_TILE_BYTES = TILE * FP4_ROW
SF_CHUNK = 128  # scale words per tcgen05.cp (4 TMEM columns)
TMEM_COLUMNS = 512

SELECT_THREADS = 1024
SELECT_WARPS = SELECT_THREADS // 32
BINS = 2048  # first radix digit: the top 11 bits of the 16-bit key
LOW = 32  # second digit: the low 5 bits
UNROLL = 4  # 16-byte loads in flight per thread
# Vector maxima of a row kept in shared memory (32 KB: rows of up to 128K keys;
# longer rows read the rest from the scorer's scratch). The smaller allocation
# leaves L1 to the select's global reads (hot vectors, block table): GB300 q5
# MXFP4, select us at 65536 -> 16384: 32K R1 / R16 / R64 12.1 / 12.6 / 53.4 ->
# 11.7 / 12.0 / 51.4, 128K 16.9 / 19.4 / 62.3 -> 15.0 / 15.8 / 52.4 (512 or 256
# threads per CTA lose: 13-20 / 13-31 / 63-101).
TOPS = 16384
CANDIDATES = 6144  # keys at or above the bound (more: the exact fallback)
HOT = 4096  # vectors holding a candidate (more: the exact fallback)
FINE = 256  # candidate bins above the bound's bin (the last one open-ended)


# --------------------------------------------------------------------------
# PTX helpers
# --------------------------------------------------------------------------
@asm()
def bulk_g2s(dst, src, barrier, size):
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
    code += "[$0], [$1], $3, [$2];"
    return code, "r,l,r,r,~{memory}", (dst, src, barrier, Int32(size))


@asm()
def store_b16(address, value):
    code = "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b16 [$0], t; }"
    return code, "l,r,~{memory}", (Int64(address), Uint32(value))


@asm()
def shared_add(pointer, value):
    return "red.shared.add.u32 [$0], $1;", "r,r,~{memory}", (pointer, Uint32(value))


@asm()
def load_16b(address):
    """16 bytes (four words) of the scratch the scorer wrote."""
    code = "ld.global.v4.u32 {$0,$1,$2,$3}, [$4];"
    return code, "=r,=r,=r,=r,l,~{memory}", (Int64(address),), (Uint32,) * 4


@asm(Uint32)
def load_u16(address):
    return "ld.global.u16 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@asm(Int32)
def load_i32(address):
    return "ld.global.b32 $0, [$1];", "=r,l,~{memory}", (Int64(address),)


@cute.jit
def stride_pages(page, tensor):
    """The element offset of page `page` of `tensor` (mode 0), in 64 bits."""
    return Int64(page) * Int64(tensor.stride[0])


@cute.jit
def table_entry(table, r: Int32, column: Int32) -> Int32:
    """`table`: (address, row stride, width) of an int32 block table."""
    address, pitch, _ = table
    return load_i32(address + (Int64(r) * Int64(pitch) + Int64(column)) * 4)


@asm(Uint32, pure=True)
def relu_bf16x2(lo: Float32, hi: Float32):
    """BF16x2 (`lo` in the low half) of relu(x), round to nearest even (DeepGEMM's
    cvt_relu_bf16x2_f32)."""
    return "cvt.rn.relu.bf16x2.f32 $0, $2, $1;", "=r,f,f", (lo, hi)


@asm(Uint32, pure=True)
def fma_bf16x2(a: Uint32, b: Uint32, c: Uint32):
    """Return a * b + c on BF16x2, one rounding per lane (__hfma2)."""
    return "fma.rn.bf16x2 $0, $1, $2, $3;", "=r,r,r,r", (a, b, c)


@asm(Uint32, pure=True)
def add_bf16x2(a: Uint32, b: Uint32):
    """Return a + b on BF16x2 (__hadd2_rn)."""
    return "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", (a, b)


@cute.jit
def bf16_bits(x: Float32) -> Uint32:
    """FP32 -> BF16 bits, round to nearest even; -0 stored as +0."""
    u = Uint32(float_bits(x))
    rounded = (u + Uint32(0x7FFF) + ((u >> Uint32(16)) & Uint32(1))) >> Uint32(16)
    bits = rounded & Uint32(0xFFFF)
    if bits == Uint32(0x8000):
        bits = Uint32(0)
    return bits


def half(words, e):
    """BF16 element `e` of eight packed in four words."""
    return (words[e // 2] >> Uint32(16 * (e % 2))) & Uint32(0xFFFF)


@cute.jit
def order16(bits: Uint32) -> Uint32:
    """Order-preserving 16-bit key of BF16 `bits` (a larger value, a larger
    key)."""
    flip = Uint32(0x8000)
    if (bits & Uint32(0x8000)) != Uint32(0):
        flip = Uint32(0xFFFF)
    return bits ^ flip


@cute.jit
def warp_inclusive(value: Uint32, lane: Int32) -> Uint32:
    for i in cutlass.range_constexpr(5):
        up = cute.arch.shuffle_sync_up(value, offset=1 << i, mask_and_clamp=0)
        if lane >= Int32(1 << i):
            value += up
    return value


@asm(Uint32, pure=True)
def max_u16x2(a, b):
    return "max.u16x2 $0, $1, $2;", "=r,r,r", (Uint32(a), Uint32(b))


@asm()
def mma_mxf4(d, a, b, instruction, sfa, sfb, accumulate):
    """One block-scaled MXF4 tcgen05 MMA (K 64, UE8M0 per 32, DeepGEMM's FP4 MQA
    logits instruction); `accumulate` is a runtime flag."""
    words = (Uint32(v) for v in (instruction, sfa, sfb, accumulate))
    code = "{ .reg .pred p; setp.ne.b32 p,$6,0; tcgen05.mma.cta_group::1"
    code += ".kind::mxf4.block_scale.scale_vec::2X [$0],$1,$2,$3,[$4],[$5],p; }"
    return code, "r,l,l,r,r,r,r", (Uint32(d), Uint64(a), Uint64(b), *words)


def fp4_instruction(n: int, k: int) -> int:
    """MXF4 instruction descriptor: E2M1 A and B (K-major), FP32 D, UE8M0 scales,
    M 128, N `n`; K step `k` of two reads scale bytes 2k, 2k + 1."""
    sf = 2 * k
    return 1 << 7 | 1 << 10 | (n >> 3) << 17 | 1 << 23 | 8 << 24 | sf << 29 | sf << 4


@cute.jit
def sw64_desc(address: Uint32):
    """K-major 64B-swizzled operand descriptor (8-row groups 512 B apart)."""
    value = Uint64((address & 262143) >> 4) | Uint64(1) << 46
    return value | Uint64(32) << 32 | Uint64(4) << 61


@cute.jit
def scale_desc(address: Uint32):
    """tcgen05.cp source: 32 rows of 16 B, unswizzled (8-row groups 128 B apart)."""
    return Uint64((address & 262143) >> 4) | Uint64(1) << 46 | Uint64(8) << 32


@cute.jit
def transpose_scales(words, chunk, lane):
    """In place, the 128 UE8M0 x4 words of rows 0..127 at `chunk` (Int32 smem
    tensor offset) into tcgen05.cp's 32x128b layout: lane l's 16 bytes are rows l,
    32 + l, 64 + l, 96 + l (TMEM columns 0..3 of lane l: rows c * 32 + l)."""
    values = [words[chunk + i * 32 + lane] for i in range(4)]
    cute.arch.sync_warp()
    address = words.iterator + chunk + lane * 4
    store_shared_words(address, tuple(Uint32(v) for v in values))


@cute.jit
def vector_keys(words, v: Int32, length: Int32):
    """The order-preserving keys of a vector's eight BF16 logits (two per word);
    keys at or past `length` are 0."""
    keys = []
    for i in cutlass.range_constexpr(4):
        w = words[i]
        flip = Uint32(0x80008000) | (
            ((w >> Uint32(15)) & Uint32(0x00010001)) * Uint32(0x7FFF)
        )
        keys.append(w ^ flip)
    if (v + 1) * 8 > length:
        for i in cutlass.range_constexpr(4):
            if v * 8 + 2 * i >= length:
                keys[i] = keys[i] & Uint32(0xFFFF0000)
            if v * 8 + 2 * i + 1 >= length:
                keys[i] = keys[i] & Uint32(0x0000FFFF)
    return keys


@cute.jit
def vector_max(keys) -> Uint32:
    m = max_u16x2(max_u16x2(keys[0], keys[1]), max_u16x2(keys[2], keys[3]))
    return cutlass.max(m & Uint32(0xFFFF), m >> Uint32(16))


@cute.jit
def fine_bin_of(key: Uint32, b0: Uint32) -> Uint32:
    """A candidate's 11-bit bin above the bound's `b0`; the keys FINE - 1 or
    more bins above share the last."""
    return cutlass.min((key >> Uint32(5)) - b0, Uint32(FINE - 1))


@cute.jit
def topdown(
    counts,
    bins: cutlass.Constexpr[int],
    need,
    info,
    slot: cutlass.Constexpr[int],
    scan,
    tid,
    warp,
    lane,
):
    """info[slot] = the bin where the counts summed from the top bin down reach
    `need`, info[slot + 1] = the count above it (unchanged when they never
    do)."""
    per = max(1, bins // SELECT_THREADS)
    top = bins - 1 - per * tid
    mine = Uint32(0)
    if top >= 0:
        for i in cutlass.range_constexpr(per):
            mine += counts[top - i]
    inclusive = warp_inclusive(mine, lane)
    if lane == 31:
        scan[warp] = inclusive
    cute.arch.sync_threads()
    prior, _ = warp_totals(scan, warp, lane)
    above = prior + inclusive - mine
    if above < need and above + mine >= need:
        running = above
        found = Int32(0)
        for i in cutlass.range_constexpr(per):
            count = counts[top - i]
            if found == 0 and running + count >= need:
                info[slot] = top - i
                info[slot + 1] = Int32(running)
                found = Int32(1)
            running += count
    cute.arch.sync_threads()


@cute.jit
def low_digit(count: Uint32, need: Uint32, lane: Int32):
    """Warp-wide over 32 bins (lane b holds bin b's `count`): the bin where the
    counts summed from bin 31 down reach `need`, and the count above it."""
    rev = cute.arch.shuffle_sync(count, 31 - lane)
    above = warp_inclusive(rev, lane) - rev  # the bins above bin 31 - lane
    hit = Int32(0)
    if above < need and above + rev >= need:
        hit = lane
    for i in cutlass.range_constexpr(5):  # one lane hits
        hit += cute.arch.shuffle_sync_bfly(hit, 1 << i)
    return Uint32(31) - Uint32(hit), cute.arch.shuffle_sync(above, hit)


@cute.jit
def warp_totals(scan, warp, lane):
    """(sum of `scan` over the warps before `warp`, sum over all): every
    warp scans the per-warp words itself."""
    x = Uint32(0)
    if lane < SELECT_WARPS:
        x = scan[lane]
    inclusive = warp_inclusive(x, lane)
    return (
        cute.arch.shuffle_sync(inclusive - x, warp),
        cute.arch.shuffle_sync(inclusive, 31),
    )


@cute.jit
def block_exclusive(value: Uint32, scan, warp, lane):
    """(exclusive prefix over the CTA's threads in order, total); `scan`:
    one word per warp."""
    inclusive = warp_inclusive(value, lane)
    if lane == 31:
        scan[warp] = inclusive
    cute.arch.sync_threads()
    prior, total = warp_totals(scan, warp, lane)
    cute.arch.sync_threads()
    return prior + inclusive - value, total


@cute.jit
def request_tiles(row_ends: cute.Tensor, r: Int32, q: cutlass.Constexpr[int], topk):
    """128-key tiles of request `r`; 0 when all its rows fit the top-k (its
    last row sees the most keys)."""
    end = row_ends[r * q + q - 1]
    tiles = Int32(0)
    if end > topk:
        tiles = (end + Int32(TILE - 1)) // Int32(TILE)
    return tiles


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------
@cute.jit
def _first_request(start: cute.Tensor, requests: Int32, g: Int32) -> Int32:
    """The request whose tiles contain global tile `g` (binary search)."""
    lo = Int32(0)
    hi = requests
    while lo < hi:
        mid = (lo + hi) // 2
        if start[mid + 1] > g:
            hi = mid
        else:
            lo = mid + 1
    return lo


@cute.jit
def _next_item(start: cute.Tensor, requests: Int32, g: Int32, r: Int32, hi: Int32):
    """The work item at global tile `g` of request `r` (the CTA range ends at
    `hi`): (request, first key, tiles, next g, next r)."""
    r_end = start[r + 1]
    tiles = cutlass.min(cutlass.min(hi, r_end) - g, Int32(ITEM_TILES))
    k0 = (g - start[r]) * Int32(TILE)
    req = r
    g += tiles
    while r < requests and start[r + 1] <= g:
        r += 1
    return req, k0, tiles, g, r


def _mn_layout(t):
    """A tcgen05 accumulator as ((M), (N), stages)."""
    sh, sd = t.layout.shape, t.layout.stride
    new = cute.make_layout(
        ((sh[0][0], sh[1]), (sh[0][1], sh[2]), *sh[3:]),
        stride=((sd[0][0], sd[1]), (sd[0][1], sd[2]), *sd[3:]),
    )
    return cute.make_tensor(t.iterator, new)


class IndexerScore:
    """BF16 logits of `rows` rows (`q` per request) into the request-compact
    scratch: request r's rows are consecutive `tiles(r) * 128`-key rows starting
    at element `q * 128 * (tiles of the requests before r)`.

    `pitch` > 0 (DCP): DeepGEMM's paged MQA logits layout instead, row `t` at
    element `t * pitch` (key index = column; `pitch` a multiple of 512, so
    1 KiB rows), no vector maxima, and every request is scored (`topk` 0: a
    rank's share of the global top-k is unknown before the merge)."""

    def __init__(
        self, rows: int, q: int, ctas: int, topk: int, pitch: int = 0, fp4=False
    ):
        assert rows % q == 0 and 1 <= q <= 8 and rows // q <= MAX_REQUESTS
        assert pitch % 512 == 0 and not (pitch and topk)
        self.rows, self.q, self.ctas, self.topk = rows, q, ctas, topk
        self.pitch, self.fp4 = pitch, fp4
        # Key tiles in flight per SM (memory-level parallelism: ~100 KB keeps the
        # HBM busy): 12 MXFP4 tiles of 8.5 KB, 6 FP8 tiles of 16.5 KB.
        self.stages = FP4_STAGES if fp4 else STAGES
        self.n = q * HEADS
        self.tiler = (TILE, self.n, DIM)
        # TMEM accumulators (EPI_GROUPS, one per epilogue warpgroup; FP4: one when
        # two do not fit next to the scale columns, q = 8).
        self.accs = EPI_GROUPS
        used = EPI_GROUPS * self.n
        if fp4:
            # Scale columns: the keys' per K stage, the query's per Q stage.
            self.sfb_columns = 4 * -(-self.n // SF_CHUNK)
            scales = 4 * self.stages + 2 * self.sfb_columns
            if EPI_GROUPS * self.n + scales > TMEM_COLUMNS:
                self.accs = 1
            self.sfa_column = self.accs * self.n
            self.sfb_column = self.sfa_column + 4 * self.stages
            used = self.sfb_column + 2 * self.sfb_columns
        self.columns = max(32, 1 << (used - 1).bit_length())

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,  # FP8 [rows * 32, 128, 1]
        weights: cute.Tensor,  # BF16 [rows * 32]
        keys: cute.Tensor,  # FP8 [64, 128, blocks]
        scales: cute.Tensor,  # Int32 (UE8M0 x4) [blocks, 64]
        table_address: Int64,  # Int32 [requests, pages]: index block ids
        table_pitch: Int32,
        table_width: Int32,
        row_ends: cute.Tensor,  # Int32 [rows]
        logits: cute.Tensor,  # BF16 bits as Int16 [capacity]
        tops: cute.Tensor,  # Int16 [capacity / 8]: each 8 logits' largest key
        q_scales: cute.Tensor,  # FP4: Int32 (UE8M0 x4) [rows, 32]; FP8: unused
        s,
    ):
        """FP4: `query` is E2M1 bytes [64, 32, rows] (row stride 4 KiB), `keys`
        E2M1 bytes [64, 64, blocks] and `scales` the keys' UE8M0 x4 words."""
        ab = cutlass.Float8E4M3FN
        mma = sm100_utils.make_trivial_tiled_mma(
            ab,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.tiler[:2],
        )
        vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        a_layout = sm100_utils.make_smem_layout_a(mma, self.tiler, ab, self.stages)
        b_layout = sm100_utils.make_smem_layout_b(mma, self.tiler, ab, 2)
        atom = sm100_utils.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, ab)
        page_layout = cute.tile_to_shape(atom, (PAGE, DIM, 2 * self.stages), (0, 1, 2))
        tma_k, tma_k_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            keys,
            cute.slice_(page_layout, (None, None, 0)),
            (PAGE, DIM),
        )
        b_one = cute.slice_(b_layout, (None, None, None, 0))
        tma_q, tma_q_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            query,
            b_one,
            self.tiler,
            mma,
            vmnk.shape,
        )
        self.q_bytes = cute.size_in_bytes(ab, b_one)
        if cutlass.const_expr(self.fp4):
            # Byte operands, 64B swizzle (Swizzle<2, 4, 3>): TMA only; the MMAs
            # take raw descriptors (mma_mxf4).
            sw64 = cute.make_swizzle(2, 4, 3)
            shape = (FP4_ROW, PAGE, 2 * self.stages)
            pages = cute.make_layout(shape, stride=(1, FP4_ROW, PAGE * FP4_ROW))
            page_layout = cute.make_composed_layout(sw64, 0, pages)
            tma_k, tma_k_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                keys,
                cute.slice_(page_layout, (None, None, 0)),
                (FP4_ROW, PAGE),
            )
            q_shape = (FP4_ROW, HEADS, self.q, 2)
            q_stride = (1, FP4_ROW, HEADS * FP4_ROW, self.n * FP4_ROW)
            q_layout = cute.make_layout(q_shape, stride=q_stride)
            b_layout = cute.make_composed_layout(sw64, 0, q_layout)
            tma_q, tma_q_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                query,
                cute.slice_(b_layout, (None, None, None, 0)),
                (FP4_ROW, HEADS, self.q),
            )
            a_layout = page_layout
            self.q_bytes = self.n * FP4_ROW
        self.kernel(
            mma,
            tma_k,
            tma_k_tensor,
            tma_q,
            tma_q_tensor,
            a_layout,
            b_layout,
            page_layout,
            scales,
            weights,
            (table_address, table_pitch, table_width),
            row_ends,
            logits,
            tops,
            q_scales,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[SCORE_THREADS, 1, 1],
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def row_weights(
        self, sW, qs, lo: cutlass.Constexpr[int], hi: cutlass.Constexpr[int]
    ):
        """Rows [lo, hi)'s BF16 head weights of Q stage `qs` (16 BF16x2 words a
        row; nothing for an empty range)."""
        rows = []
        for tok in cutlass.range_constexpr(lo, hi):
            words = []
            for h in cutlass.range_constexpr(0, HEADS, 8):
                address = sW.iterator + qs * self.n + tok * HEADS + h
                words += list(load_words(address, count=4, shared=True))
            rows.append(words)
        return rows

    @cute.jit
    def rows_out(
        self,
        lo: cutlass.Constexpr[int],
        hi: cutlass.Constexpr[int],
        t2r,
        tAll,
        cD,
        sW,
        sEnd,
        rows,
        bases,
    ):
        """Verify rows [lo, hi) of one tile: relu, the head weights and the key
        scale -> BF16 logits, and each 8 keys' largest (the select's bound). The
        rows' accumulators are read from TMEM and packed to relu'd BF16 pairs
        first (16 registers a row), then the accumulator is released
        (`release`: its empty barrier) so the next MMA overlaps the head sums."""
        row0, pitch, key, kscale, qs, acc, lt = rows
        out_base, tops_base, release, weights = bases
        # tcgen05.ld's registers are valid only after tcgen05.wait::ld: each row
        # is packed after the wait that covers its load, while the next row's
        # load is in flight (reading a row right after its own load gave wrong
        # logits for one warp's 32 keys, ~1 in 100 runs of the wide240 test).
        packed = []
        frags = [cute.make_fragment_like(cD, Float32) for _ in range(lo, hi)]
        if cutlass.const_expr(lo < hi):
            cute.copy(t2r, tAll[(None, None, None, 0, lo, acc)], frags[0])
            cute.arch.fence_view_async_tmem_load()
        for tok in cutlass.range_constexpr(lo, hi):
            if cutlass.const_expr(tok + 1 < hi):
                nxt = tAll[(None, None, None, 0, tok + 1, acc)]
                cute.copy(t2r, nxt, frags[tok + 1 - lo])
            frag = frags[tok - lo]
            packed.append(
                [relu_bf16x2(frag[h], frag[h + 1]) for h in range(0, HEADS, 2)]
            )
            if cutlass.const_expr(tok + 1 < hi):
                cute.arch.fence_view_async_tmem_load()
        # The loads are complete (the waits above) and ordered before the
        # release that lets the next MMA overwrite the accumulator.
        fence_tmem(before=True)
        cute.arch.mbarrier_arrive(release)
        for tok in cutlass.range_constexpr(lo, hi):
            pairs = packed[tok - lo]
            # Stock's precision (DeepGEMM SM100 paged MQA logits): each head's
            # relu(q . k) rounded to BF16, the BF16 head weights accumulated by
            # BF16 FMAs in two BF16x2 sums (heads 4i, 4i + 1 and 4i + 2, 4i + 3),
            # then summed in BF16. FP8: the key's UE8M0 scale (a power of two,
            # inside DeepGEMM's MMA) scales the BF16 result exactly; MXFP4: the
            # scales are in the MMA (kscale 1).
            s0 = Uint32(0)
            s1 = Uint32(0)
            for h in cutlass.range_constexpr(0, HEADS, 8):
                w = weights[tok - lo][h // 2 : h // 2 + 4]
                s0 = fma_bf16x2(pairs[h // 2], w[0], s0)
                s1 = fma_bf16x2(pairs[h // 2 + 1], w[1], s1)
                s0 = fma_bf16x2(pairs[h // 2 + 2], w[2], s0)
                s1 = fma_bf16x2(pairs[h // 2 + 3], w[3], s1)
            pair = add_bf16x2(s0, s1)
            total = add_bf16x2(pair, pair >> Uint32(16)) & Uint32(0xFFFF)
            if cutlass.const_expr(self.fp4):  # the scales are in the MMA
                bits = total
                if bits == Uint32(0x8000):  # -0 as +0
                    bits = Uint32(0)
            else:
                bits = bf16_bits(bits_float(total << Uint32(16)) * kscale)
            element = row0 + Int64(tok) * pitch + Int64(key)
            if cutlass.const_expr(self.pitch):
                if key < sEnd[tok, qs]:
                    store_b16(out_base + element * 2, bits)
            else:
                top = Uint32(0)
                if key < sEnd[tok, qs]:
                    store_b16(out_base + element * 2, bits)
                    top = order16(bits)
                for b in cutlass.range_constexpr(3):
                    top = cutlass.max(top, cute.arch.shuffle_sync_bfly(top, 1 << b))
                if lt % 8 == 0:
                    store_b16(tops_base + (element >> 3) * 2, top)

    @cute.jit
    def mma_fp4(self, sA, sB, sSc, sQsc, tmem_ptr, bars, cursor, sStart, requests):
        """The MMA warp (MXFP4): per item, the query's scale words into TMEM; per
        tile, the keys' scale words into TMEM (tcgen05.cp after an in-place smem
        transpose), then two K64 MXF4 MMAs into the tile's accumulator. The copies
        and MMAs are issued by one lane, in order."""
        kv_full, kv_empty, q_full, q_empty, acc_full, acc_empty = bars
        g, r, hi = cursor
        lane = cute.arch.lane_idx()
        n, ACCS = self.n, self.accs
        tmem = Uint32(tmem_ptr.toint())
        sfa, sfb = tmem + self.sfa_column, tmem + self.sfb_column
        a_base = shared_address(sA.iterator)
        b_base = shared_address(sB.iterator)
        sc_base = shared_address(sSc.iterator)
        qsc_base = shared_address(sQsc.iterator)
        chunks = self.sfb_columns // 4
        it = Int32(0)
        j = Int32(0)
        while g < hi:
            _req, _k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
            qs = j % 2
            cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
            for c in cutlass.range_constexpr(chunks):
                transpose_scales(sQsc, qs * 2 * SF_CHUNK + c * SF_CHUNK, lane)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            q_sf = sfb + qs * self.sfb_columns
            b_desc = sw64_desc(b_base + qs * (n * FP4_ROW))
            with cute.arch.elect_one():
                for c in cutlass.range_constexpr(chunks):
                    address = qsc_base + (qs * 2 * SF_CHUNK + c * SF_CHUNK) * 4
                    copy_scale(q_sf + 4 * c, scale_desc(address))
            for _t in cutlass.range(ntile, unroll=1):
                stage = it % self.stages
                a = it % ACCS
                cute.arch.mbarrier_wait(acc_empty + a, ((it // ACCS) & 1) ^ 1)
                cute.arch.mbarrier_wait(kv_full + stage, (it // self.stages) & 1)
                fence_tmem()  # after the epilogue's release of the accumulator
                transpose_scales(sSc, stage * TILE, lane)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                k_sf = sfa + stage * 4
                a_desc = sw64_desc(a_base + stage * FP4_TILE_BYTES)
                with cute.arch.elect_one():
                    copy_scale(k_sf, scale_desc(sc_base + stage * TILE * 4))
                    for kb in cutlass.range_constexpr(2):
                        mma_mxf4(
                            tmem + a * n,
                            a_desc + Uint64(2 * kb),  # +32 bytes of K
                            b_desc + Uint64(2 * kb),
                            fp4_instruction(n, kb),
                            k_sf,
                            q_sf,
                            Int32(kb),
                        )
                    tcgen05.commit(kv_empty + stage)
                    tcgen05.commit(acc_full + a)
                it += 1
            with cute.arch.elect_one():
                tcgen05.commit(q_empty + qs)
            j += 1

    @cute.kernel
    def kernel(
        self,
        mma: cute.TiledMma,
        tma_k: cute.CopyAtom,
        mK: cute.Tensor,
        tma_q: cute.CopyAtom,
        mQ: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        page_layout: cute.ComposedLayout,
        scales: cute.Tensor,
        weights: cute.Tensor,
        table,
        row_ends: cute.Tensor,
        logits: cute.Tensor,
        tops: cute.Tensor,
        q_scales: cute.Tensor,
    ):
        Q = self.q
        ACCS = self.accs
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()

        @cute.struct
        class Storage:
            kv_full: cute.struct.MemRange[cutlass.Int64, self.stages]
            kv_empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            sc_empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            q_full: cute.struct.MemRange[cutlass.Int64, 2]
            q_empty: cute.struct.MemRange[cutlass.Int64, 2]
            acc_full: cute.struct.MemRange[cutlass.Int64, EPI_GROUPS]
            acc_empty: cute.struct.MemRange[cutlass.Int64, EPI_GROUPS]
            tmem_hold: Int32
            row_end: cute.struct.MemRange[Int32, 8 * 2]
            start: cute.struct.MemRange[Int32, MAX_REQUESTS + 1]
            w: cute.struct.Align[cute.struct.MemRange[BFloat16, 2 * self.n], 16]
            scale: cute.struct.Align[
                cute.struct.MemRange[Int32, self.stages * TILE], 16
            ]
            # FP4: the query's scale words, per Q stage (SF_CHUNK-word chunks).
            q_scale: cute.struct.Align[
                cute.struct.MemRange[Int32, 2 * 2 * SF_CHUNK], 16
            ]

        smem = utils.SmemAllocator()
        st = smem.allocate(Storage)
        if cutlass.const_expr(self.fp4):
            sA = smem.allocate_tensor(
                cutlass.Uint8, a_layout.outer, 1024, swizzle=a_layout.inner
            )
            sB = smem.allocate_tensor(
                cutlass.Uint8, b_layout.outer, 1024, swizzle=b_layout.inner
            )
            sPage = sA
        else:
            sA = smem.allocate_tensor(
                cutlass.Float8E4M3FN, a_layout.outer, 1024, swizzle=a_layout.inner
            )
            sB = smem.allocate_tensor(
                cutlass.Float8E4M3FN, b_layout.outer, 1024, swizzle=b_layout.inner
            )
            sPage = cute.make_tensor(sA.iterator, page_layout.outer)
        sW = st.w.get_tensor(cute.make_layout((self.n, 2)))
        sEnd = st.row_end.get_tensor(cute.make_layout((8, 2)))
        sStart = st.start.get_tensor(cute.make_layout((MAX_REQUESTS + 1,)))
        sSc = st.scale.get_tensor(cute.make_layout((TILE, self.stages)))
        sQsc = st.q_scale.get_tensor(cute.make_layout((2 * SF_CHUNK, 2)))

        kv_full = st.kv_full.data_ptr()
        kv_empty = st.kv_empty.data_ptr()
        sc_empty = st.sc_empty.data_ptr()
        q_full = st.q_full.data_ptr()
        q_empty = st.q_empty.data_ptr()
        acc_full = st.acc_full.data_ptr()
        acc_empty = st.acc_empty.data_ptr()
        if tidx == 0:
            for i in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(kv_full + i, 1)
                cute.arch.mbarrier_init(kv_empty + i, 1)
                cute.arch.mbarrier_init(sc_empty + i, 128 * EPI_SPLIT)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(q_full + i, 1)
                cute.arch.mbarrier_init(q_empty + i, 1 + EPI_WARPS)
            for i in cutlass.range_constexpr(EPI_GROUPS):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, 128 * EPI_SPLIT)
            cute.arch.mbarrier_init_fence()
        if warp == PRODUCER:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_q)

        tmem = utils.TmemAllocator(
            st.tmem_hold.ptr,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=1, num_threads=SCORE_THREADS
            ),
            allocator_warp_id=MMA_WARP,
        )
        tmem.allocate(self.columns)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        acc_shape = mma.partition_shape_C(self.tiler[:2])
        acc_fake = mma.make_fragment_C(cute.append(acc_shape, EPI_GROUPS))
        tAcc = cute.make_tensor(tmem_ptr, acc_fake.layout)

        # The schedule: sStart[r] = first global tile of request r (row ends are
        # step metadata, written before the graph runs).
        requests = row_ends.shape[0] // Q
        for r in cutlass.range(tidx, requests, SCORE_THREADS, unroll=1):
            sStart[r + 1] = request_tiles(row_ends, r, Q, self.topk)
        cute.arch.sync_threads()
        if warp == 0:
            carry = Int32(0)
            for base in cutlass.range(0, requests, 32, unroll=1):
                r = base + lane
                tiles = Int32(0)
                if r < requests:
                    tiles = sStart[r + 1]
                tiles = Int32(warp_inclusive(Uint32(tiles), lane))
                if r < requests:
                    sStart[r + 1] = carry + tiles
                carry += cute.arch.shuffle_sync(tiles, 31)
            if lane == 0:
                sStart[0] = Int32(0)
        cute.arch.sync_threads()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        total = sStart[requests]
        lo = Int32((Int64(total) * Int64(bid)) // Int64(self.ctas))
        hi = Int32((Int64(total) * Int64(bid + 1)) // Int64(self.ctas))
        r_lo = _first_request(sStart, requests, lo)

        if cutlass.const_expr(self.fp4):
            gQ = cute.local_tile(mQ, (FP4_ROW, HEADS, Q), (None, None, None))
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(gQ, 0, 3),
            )
            gK = cute.local_tile(mK, (FP4_ROW, PAGE), (None, None, None))
        else:
            gQ = cute.local_tile(
                mQ, cute.slice_(self.tiler, (0, None, None)), (None, None, None)
            )
            thr_mma = mma.get_slice(0)
            tCgQ = thr_mma.partition_B(gQ)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgQ, 0, 3),
            )
            gK = cute.local_tile(mK, (PAGE, DIM), (None, None, None))
        tKsK, tKgK = cpasync.tma_partition(
            tma_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sPage, 0, 2),
            cute.group_modes(gK, 0, 2),
        )
        it = Int32(0)  # tiles issued / consumed by this role
        j = Int32(0)  # items issued / consumed by this role
        g = lo
        r = r_lo
        max_page = table[2] - 1
        blocks = scales.shape[0]
        if warp == PRODUCER:
            # One item of metadata ahead: the next item's dependent loads overlap
            # the current item's tiles.
            n_req = Int32(0)
            n_k0 = Int32(0)
            n_nt = Int32(0)
            n_page = Int32(0)
            n_end = Int32(0)
            if g < hi:
                n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                n_page = table_entry(
                    table, n_req, cutlass.min(n_k0 // PAGE + lane, max_page)
                )
                n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
            has_item = n_nt > 0
            while has_item:
                req = n_req
                ntile = n_nt
                my_page = n_page
                my_end = n_end
                n_nt = Int32(0)
                if g < hi:
                    n_req, n_k0, n_nt, g, r = _next_item(sStart, requests, g, r, hi)
                    n_page = table_entry(
                        table, n_req, cutlass.min(n_k0 // PAGE + lane, max_page)
                    )
                    n_end = row_ends[n_req * Q + cutlass.min(lane, Q - 1)]
                qs = j % 2
                cute.arch.mbarrier_wait(q_empty + qs, ((j // 2) & 1) ^ 1)
                if lane < Q:
                    sEnd[lane, qs] = my_end
                cute.arch.sync_warp()
                q_tx = self.q_bytes + self.n * 2
                if cutlass.const_expr(self.fp4):
                    q_tx += self.n * 4
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_full + qs, q_tx)
                    bulk_g2s(
                        sW.iterator + qs * self.n,
                        weights.iterator + req * Q * HEADS,
                        q_full + qs,
                        self.n * 2,
                    )
                    if cutlass.const_expr(self.fp4):
                        # The rows' UE8M0 x4 words: row t's 32 heads.
                        for t in cutlass.range_constexpr(Q):
                            bulk_g2s(
                                sQsc.iterator + qs * 2 * SF_CHUNK + t * HEADS,
                                q_scales.iterator
                                + Int64(req * Q + t) * Int64(q_scales.stride[0]),
                                q_full + qs,
                                HEADS * 4,
                            )
                if cutlass.const_expr(self.fp4):
                    cute.copy(
                        tma_q,
                        tQgQ[(None, 0, 0, req)],
                        tQsQ[(None, qs)],
                        tma_bar_ptr=q_full + qs,
                    )
                else:
                    cute.copy(
                        tma_q,
                        tQgQ[(None, req, 0, 0)],
                        tQsQ[(None, qs)],
                        tma_bar_ptr=q_full + qs,
                    )
                j += 1
                for t in cutlass.range(ntile, unroll=1):
                    stage = it % self.stages
                    p0 = cute.arch.shuffle_sync(my_page, 2 * t)
                    p1 = cute.arch.shuffle_sync(my_page, 2 * t + 1)
                    # A partial last tile's second page may lie past the request's
                    # blocks: keep its (unused) scale read in bounds.
                    p1 = cutlass.min(cutlass.max(p1, Int32(0)), blocks - 1)
                    ph = ((it // self.stages) & 1) ^ 1
                    cute.arch.mbarrier_wait(kv_empty + stage, ph)
                    if cutlass.const_expr(not self.fp4):  # FP4: no epilogue read
                        cute.arch.mbarrier_wait(sc_empty + stage, ph)
                    key_bytes = FP4_TILE_BYTES if self.fp4 else TILE * DIM
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            kv_full + stage, key_bytes + TILE * 4
                        )
                        # 64-bit page offsets: page ids index the whole KV pool
                        # view, so page x stride (2112 words) passes Int32 from
                        # page 1,016,801 (the static stride makes it a 32-bit
                        # multiply, and the bulk copy is not bounds-checked).
                        bulk_g2s(
                            sSc.iterator + stage * TILE,
                            scales.iterator + stride_pages(p0, scales),
                            kv_full + stage,
                            PAGE * 4,
                        )
                        bulk_g2s(
                            sSc.iterator + stage * TILE + PAGE,
                            scales.iterator + stride_pages(p1, scales),
                            kv_full + stage,
                            PAGE * 4,
                        )
                    cute.copy(
                        tma_k,
                        tKgK[(None, 0, 0, p0)],
                        tKsK[(None, 2 * stage)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    cute.copy(
                        tma_k,
                        tKgK[(None, 0, 0, p1)],
                        tKsK[(None, 2 * stage + 1)],
                        tma_bar_ptr=kv_full + stage,
                    )
                    it += 1
                has_item = n_nt > 0
        elif warp == MMA_WARP:
            if cutlass.const_expr(self.fp4):
                bars = kv_full, kv_empty, q_full, q_empty, acc_full, acc_empty
                cursor = g, r, hi
                self.mma_fp4(
                    sA, sB, sSc, sQsc, tmem_ptr, bars, cursor, sStart, requests
                )
            else:
                tCrA = mma.make_fragment_A(sA)
                tCrB = mma.make_fragment_B(sB)
                while g < hi:
                    _req, _k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                    qs = j % 2
                    cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                    for _t in cutlass.range(ntile, unroll=1):
                        stage = it % self.stages
                        a = it % EPI_GROUPS
                        cute.arch.mbarrier_wait(
                            acc_empty + a, ((it // EPI_GROUPS) & 1) ^ 1
                        )
                        cute.arch.mbarrier_wait(
                            kv_full + stage, (it // self.stages) & 1
                        )
                        fence_tmem()  # after the epilogue's release
                        # Reconverge after the per-thread waits: cute.gemm and the
                        # commits below each elect one lane, and tcgen05.commit
                        # tracks only the MMAs its own lane issued. Defensive only:
                        # it did NOT change the rate of the open FP8-cache flake
                        # (row 0 of an item's first tile, one or two warps' keys
                        # partially accumulated: 0.3-0.6 % of runs of
                        # test_fused_indexer_selects_exact_topk[wide320-36-2048],
                        # before and after; root cause still open).
                        cute.arch.sync_warp()
                        mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            cute.gemm(
                                mma,
                                tAcc[(None, None, None, a)],
                                tCrA[(None, None, kb, stage)],
                                tCrB[(None, None, kb, qs)],
                                tAcc[(None, None, None, a)],
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        with cute.arch.elect_one():
                            tcgen05.commit(kv_empty + stage)
                            tcgen05.commit(acc_full + a)
                        it += 1
                    with cute.arch.elect_one():
                        tcgen05.commit(q_empty + qs)
                    j += 1
        else:
            # Epilogue: warpgroups 2 acc and 2 acc + 1 own TMEM accumulator acc
            # (every other tile), the first the first ceil(q / 2) rows, the
            # second the rest.
            acc = warp // 4 // EPI_SPLIT
            part = warp // 4 % EPI_SPLIT
            lt = tidx % 128
            ld_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(HEADS)), Float32
            )
            acc_cols = cute.flat_divide(_mn_layout(tAcc), (TILE, HEADS))
            t2r = tcgen05.make_tmem_copy(ld_atom, acc_cols[(None, None, 0, 0, 0)])
            thr = t2r.get_slice(lt)
            tAll = thr.partition_S(acc_cols)
            cD = thr.partition_D(cute.make_identity_tensor((TILE, HEADS)))
            bases = logits.iterator.toint(), tops.iterator.toint()
            while g < hi:
                req, k0, ntile, g, r = _next_item(sStart, requests, g, r, hi)
                if cutlass.const_expr(self.pitch):
                    pitch = Int64(self.pitch)
                    row0 = Int64(req * Q) * pitch
                else:
                    tiles = sStart[req + 1] - sStart[req]
                    row0 = Int64(sStart[req]) * Int64(TILE * Q)
                    pitch = Int64(tiles) * Int64(TILE)
                qs = j % 2
                cute.arch.mbarrier_wait(q_full + qs, (j // 2) & 1)
                # This part's rows' BF16 head weights (BF16x2 words), for every
                # tile of the item.
                mid = -(-Q // EPI_SPLIT)
                w0 = self.row_weights(sW, qs, 0, mid)
                w1 = self.row_weights(sW, qs, mid, Q)
                for t in cutlass.range(ntile, unroll=1):
                    if it % ACCS == acc:
                        stage = it % self.stages
                        key = k0 + t * TILE + lt
                        kscale = Float32(1)  # FP4: the scales are in the MMA
                        if cutlass.const_expr(not self.fp4):
                            phase = (it // self.stages) & 1
                            cute.arch.mbarrier_wait(kv_full + stage, phase)
                            scale_bits = sSc[lt, stage]
                            cute.arch.mbarrier_arrive(sc_empty + stage)
                            # UE8M0: the low byte is the biased exponent.
                            kscale = cute.math.exp2(Float32((scale_bits & 0xFF) - 127))
                        cute.arch.mbarrier_wait(acc_full + acc, (it // ACCS) & 1)
                        # tcgen05.ld is .sync.aligned: reconverge the warp after
                        # the per-thread barrier wait.
                        cute.arch.sync_warp()
                        # Order the TMEM loads after the MMAs' commit.
                        fence_tmem()
                        rows = row0, pitch, key, kscale, qs, acc, lt
                        out = t2r, tAll, cD, sW, sEnd, rows
                        release = acc_empty + acc
                        if part == 0:
                            self.rows_out(0, mid, *out, (*bases, release, w0))
                        else:
                            if cutlass.const_expr(mid < Q):
                                self.rows_out(mid, Q, *out, (*bases, release, w1))
                            else:  # no rows: release at once
                                cute.arch.mbarrier_arrive(acc_empty + acc)
                    it += 1
                cute.arch.sync_warp()
                if lane == 0:
                    cute.arch.mbarrier_arrive(q_empty + qs)
                j += 1

        cute.arch.sync_threads()
        if warp == MMA_WARP:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# --------------------------------------------------------------------------
# Select
# --------------------------------------------------------------------------
class IndexerSelect:
    """The exact top-`topk` of every row (one CTA per row) -> `out` [rows,
    topk]: the physical KV rows (`block_rows` rows per `page`-token block of
    the MLA cache, a power of two: 64, or e.g. 2048 in hybrid models whose
    attention pages match the mamba state's) of the selected keys in key order,
    -1 past the row's count, and `valid` [rows]: the counts. `ids`
    (`logical`): the key indices too.

    The K-th largest of the scorer's per-8 vector maxima bounds the K-th largest
    key from below (each of those vectors holds a key at least as large), so only
    the keys at or above it, about K of them, are candidates: pass 1
    histograms the maxima (a row of fewer than K vectors: its keys), pass 2
    gathers the candidates in key order, and a radix select over them (11-bit bin,
    then the low 5 bits) finds the threshold. Rows whose candidates overflow the
    shared list take the exact select over every key."""

    def __init__(
        self,
        rows: int,
        q: int,
        topk: int,
        logical: bool = False,
        release=True,
        page: int = PAGE,
    ):
        assert topk % SELECT_THREADS == 0 and rows % q == 0
        assert page > 0 and page & (page - 1) == 0, page
        self.rows, self.q, self.topk, self.logical = rows, q, topk, logical
        # release: launch the next kernel early (PDL). Off when it reads the rows
        # without waiting for this grid first (FlashInfer prims_ts' split-KV decode
        # launches with PDL but does not wait before it reads its indices).
        self.release = release
        self.page_shift = page.bit_length() - 1

    @cute.jit
    def pdl(self):
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.release):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        logits: cute.Tensor,
        tops: cute.Tensor,
        row_ends: cute.Tensor,
        table_address: Int64,
        table_pitch: Int32,
        block_rows: Int32,
        out: cute.Tensor,
        valid: cute.Tensor,
        ids: cute.Tensor,
        s,
    ):
        table = (table_address, table_pitch, Int32(0))
        args = (logits, tops, row_ends, table, block_rows, out, valid, ids)
        self.kernel(args).launch(
            grid=(self.rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def emit(self, out, ids, table, request, row, pos, key, block_rows):
        shift: cutlass.Constexpr = self.page_shift
        page = table_entry(table, request, key >> shift)
        out[row, pos] = page * block_rows + (key & ((1 << shift) - 1))
        if cutlass.const_expr(self.logical):
            ids[row, pos] = key

    @cute.jit
    def vector(self, address, v):
        return load_words(address + Int64(v) * 16, count=4)

    @cute.jit
    def vector_top(self, tops_s, tops_address, v):
        """Vector `v`'s largest key: pass 1 kept the first TOPS in shared
        memory."""
        top = Uint32(0)
        if v < TOPS:  # noqa: SIM108 (a branch: the global load stays conditional)
            top = Uint32(tops_s[v])
        else:
            top = load_u16(tops_address + Int64(v) * 2)
        return top

    @cute.jit
    def count_at_least(self, keys, v, length, low) -> Uint32:
        n = Uint32(0)
        for e in cutlass.range_constexpr(8):
            if half(keys, e) >= low and v * 8 + e < length:
                n += Uint32(1)
        return n

    @cute.jit
    def publish(self, whist, wabove, pub, tid):
        """Into `pub`: the low-digit histogram and the count above its bin."""
        if tid < LOW:
            total = Uint32(0)
            for w in cutlass.range_constexpr(SELECT_WARPS):
                total += whist[tid, w]
            pub[tid] = total
        if tid == LOW:
            total = Uint32(0)
            for w in cutlass.range_constexpr(SELECT_WARPS):
                total += wabove[w]
            pub[LOW] = total
        cute.arch.sync_threads()

    @cute.jit
    def resolve(self, pub, info, whist, wabove, wpre, above_high, warp, lane):
        """From `pub` and `above_high` (keys above the threshold's bin):
        info[4] = the threshold's low digit, info[5] = the ties at the threshold
        to take, and wpre = each warp's count of keys above the threshold and of
        ties before it (in key order)."""
        if warp == 0:
            need = Uint32(self.topk) - above_high
            low, above_low = low_digit(pub[lane], need, lane)
            take = need - above_low
            a = Uint32(0)
            ties = Uint32(0)
            if lane < SELECT_WARPS:
                a = wabove[lane]
                for b in cutlass.range_constexpr(LOW):
                    if Uint32(b) > low:
                        a += whist[b, lane]
                ties = whist[Int32(low), lane]
            a_incl = warp_inclusive(a, lane)
            t_incl = warp_inclusive(ties, lane)
            if lane < SELECT_WARPS:
                wpre[0, lane] = a_incl - a
                wpre[1, lane] = t_incl - ties
            if lane == 0:
                info[4] = Int32(low)
                info[5] = Int32(take)
        cute.arch.sync_threads()

    @cute.jit
    def write_selected(
        self, keys, indices, n, threshold, take, carry_a, carry_t, lane, emit_args
    ):
        """One warp round of the ordered output: `n` (0/1) per lane of key
        `keys` at `indices`; returns the carries."""
        out, ids, table, request, row, block_rows = emit_args
        n_a = Uint32(0)
        n_t = Uint32(0)
        if n != Uint32(0):
            if keys > threshold:
                n_a = Uint32(1)
            elif keys == threshold:
                n_t = Uint32(1)
        packed = (n_a << Uint32(16)) | n_t
        inclusive = warp_inclusive(packed, lane)
        exclusive = inclusive - packed
        total = cute.arch.shuffle_sync(inclusive, 31)
        pos_a = carry_a + (exclusive >> Uint32(16))
        pos_t = carry_t + (exclusive & Uint32(0xFFFF))
        if n_a != Uint32(0):
            pos = Int32(pos_a + cutlass.min(pos_t, take))
            self.emit(out, ids, table, request, row, pos, indices, block_rows)
        if n_t != Uint32(0) and pos_t < take:
            pos = Int32(pos_a + pos_t)
            self.emit(out, ids, table, request, row, pos, indices, block_rows)
        return carry_a + (total >> Uint32(16)), carry_t + (total & Uint32(0xFFFF))

    @cute.jit
    def exact(self, sm, address, length, emit_args, tid, warp, lane):
        """The exact select over every key of the row (11-bit, then 5-bit
        histograms): the fallback of rows whose candidates overflow the list,
        e.g. rows of many equal logits."""
        hist, whist, wabove, wpre, pub, scan, info = sm
        K = self.topk
        vectors = (length + 7) // 8
        per_warp = (vectors + SELECT_WARPS - 1) // SELECT_WARPS
        w0 = cutlass.min(warp * per_warp, vectors)
        w1 = cutlass.min(w0 + per_warp, vectors)
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            hist[b] = Uint32(0)
        whist[lane, warp] = Uint32(0)
        cute.arch.sync_threads()
        for v in cutlass.range(w0 + lane, w1, 32, unroll=1):
            keys = vector_keys(self.vector(address, v), v, length)
            for e in cutlass.range_constexpr(8):
                if v * 8 + e < length:
                    shared_add(hist.iterator + (half(keys, e) >> Uint32(5)), 1)
        cute.arch.sync_threads()
        topdown(hist, BINS, Uint32(K), info, 2, scan, tid, warp, lane)
        high = Uint32(info[2])
        above = Uint32(0)
        for v in cutlass.range(w0 + lane, w1, 32, unroll=1):
            keys = vector_keys(self.vector(address, v), v, length)
            for e in cutlass.range_constexpr(8):
                if v * 8 + e < length:
                    key = half(keys, e)
                    if (key >> Uint32(5)) > high:
                        above += Uint32(1)
                    elif (key >> Uint32(5)) == high:
                        target = whist.iterator + warp * LOW + (key & Uint32(31))
                        shared_add(target, 1)
        for i in cutlass.range_constexpr(5):
            above += cute.arch.shuffle_sync_bfly(above, 1 << i)
        if lane == 0:
            wabove[warp] = above
        cute.arch.sync_threads()
        self.publish(whist, wabove, pub, tid)
        self.resolve(pub, info, whist, wabove, wpre, Uint32(info[3]), warp, lane)
        threshold = (high << Uint32(5)) | Uint32(info[4])
        take = Uint32(info[5])
        carry_a = wpre[0, warp]
        carry_t = wpre[1, warp]
        out, ids, table, request, row, block_rows = emit_args
        for base in cutlass.range(w0, w1, 32, unroll=1):
            v = base + lane
            keys = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
            if v < w1:
                keys = vector_keys(self.vector(address, v), v, length)
            n_a = Uint32(0)
            n_t = Uint32(0)
            for e in cutlass.range_constexpr(8):
                if v < w1 and v * 8 + e < length:
                    if half(keys, e) > threshold:
                        n_a += Uint32(1)
                    elif half(keys, e) == threshold:
                        n_t += Uint32(1)
            packed = (n_a << Uint32(16)) | n_t
            inclusive = warp_inclusive(packed, lane)
            exclusive = inclusive - packed
            total = cute.arch.shuffle_sync(inclusive, 31)
            pos_a = carry_a + (exclusive >> Uint32(16))
            pos_t = carry_t + (exclusive & Uint32(0xFFFF))
            if packed != Uint32(0):
                for e in cutlass.range_constexpr(8):
                    key = half(keys, e)
                    if v * 8 + e < length:
                        if key > threshold:
                            pos = Int32(pos_a + cutlass.min(pos_t, take))
                            self.emit(
                                out,
                                ids,
                                table,
                                request,
                                row,
                                pos,
                                v * 8 + e,
                                block_rows,
                            )
                            pos_a += Uint32(1)
                        elif key == threshold:
                            if pos_t < take:
                                pos = Int32(pos_a + pos_t)
                                self.emit(
                                    out,
                                    ids,
                                    table,
                                    request,
                                    row,
                                    pos,
                                    v * 8 + e,
                                    block_rows,
                                )
                            pos_t += Uint32(1)
            carry_a += total >> Uint32(16)
            carry_t += total & Uint32(0xFFFF)

    @cute.kernel
    def kernel(self, args):
        logits, tops, row_ends, table, block_rows, out, valid, ids = args
        Q, K = self.q, self.topk
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        request = row // Q

        smem = utils.SmemAllocator()
        hist = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        whist = smem.allocate_tensor(Uint32, cute.make_layout((LOW, SELECT_WARPS)), 16)
        wabove = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        wpre = smem.allocate_tensor(Uint32, cute.make_layout((2, SELECT_WARPS)), 16)
        pub = smem.allocate_tensor(Uint32, cute.make_layout(LOW + 1), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        fine_hist = smem.allocate_tensor(Uint32, cute.make_layout(FINE), 16)
        cand_index = smem.allocate_tensor(Int32, cute.make_layout(CANDIDATES), 16)
        hot_list = smem.allocate_tensor(Int32, cute.make_layout(HOT), 16)
        cand_key = smem.allocate_tensor(Uint32, cute.make_layout(CANDIDATES), 16)
        tops_s = smem.allocate_tensor(cutlass.Uint16, cute.make_layout(TOPS), 16)

        # This row's logits: request ``request``'s row ``row % Q`` of the scratch.
        if warp == 0:
            before = Int32(0)
            for base in cutlass.range(0, request + 1, 32, unroll=1):
                r = base + lane
                tiles = Int32(0)
                if r < request:
                    tiles = request_tiles(row_ends, r, Q, K)
                for i in cutlass.range_constexpr(5):
                    tiles += cute.arch.shuffle_sync_bfly(tiles, 1 << i)
                before += tiles
            if lane == 0:
                info[0] = before
                info[1] = request_tiles(row_ends, request, Q, K)
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            hist[b] = Uint32(0)
        whist[lane, warp] = Uint32(0)
        if tid < FINE:
            fine_hist[tid] = Uint32(0)
        cute.arch.sync_threads()
        length = row_ends[row]
        emit_args = out, ids, table, request, row, block_rows
        if length <= K:
            # Short-context bypass: every key, no logits.
            self.pdl()
            n = cutlass.max(length, Int32(0))
            for e in cutlass.range(tid, K, SELECT_THREADS, unroll=1):
                if e < n:
                    self.emit(out, ids, table, request, row, e, e, block_rows)
                else:
                    out[row, e] = Int32(-1)
                    if cutlass.const_expr(self.logical):
                        ids[row, e] = Int32(-1)
            if tid == 0:
                valid[row] = n
        else:
            element = Int64(info[0]) * Int64(TILE * Q)
            element += Int64(row % Q) * Int64(info[1]) * Int64(TILE)
            address = logits.iterator.toint() + element * 2
            tops_address = tops.iterator.toint() + (element >> 3) * 2
            vectors = (length + 7) // 8
            # Under 2K vectors the K-th largest maximum bounds loosely: histogram
            # the keys (the bound is then the threshold's own bin).
            dense = vectors < 2 * K
            # Lane runs (in key order); an odd run spreads the lanes' shared reads
            # over the banks.
            run = (vectors + SELECT_THREADS - 1) // SELECT_THREADS
            run = run + 1 - run % 2
            l0 = cutlass.min(tid * run, vectors)
            l1 = cutlass.min(l0 + run, vectors)
            self.pdl()

            # Pass 1: histogram of the vector maxima (dense: of the keys).
            if dense:
                for v in cutlass.range(tid, vectors, SELECT_THREADS, unroll=1):
                    keys = vector_keys(self.vector(address, v), v, length)
                    tops_s[v] = cutlass.Uint16(vector_max(keys))
                    for e in cutlass.range_constexpr(8):
                        if v * 8 + e < length:
                            bin_ = half(keys, e) >> Uint32(5)
                            shared_add(hist.iterator + bin_, 1)
            else:
                chunks = (vectors + 7) // 8
                for base in cutlass.range(0, chunks, SELECT_THREADS * UNROLL, unroll=1):
                    batch = []
                    for u in cutlass.range_constexpr(UNROLL):
                        j = base + u * SELECT_THREADS + tid
                        words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                        if j < chunks:
                            words = load_16b(tops_address + Int64(j) * 16)
                        batch.append(words)
                    for u in cutlass.range_constexpr(UNROLL):
                        j = base + u * SELECT_THREADS + tid
                        if j < chunks:
                            if j * 8 < TOPS:
                                store_shared_words(tops_s.iterator + j * 8, batch[u])
                            for e in cutlass.range_constexpr(8):
                                if j * 8 + e < vectors:
                                    bin_ = half(batch[u], e) >> Uint32(5)
                                    shared_add(hist.iterator + bin_, 1)
            cute.arch.sync_threads()
            topdown(hist, BINS, Uint32(K), info, 2, scan, tid, warp, lane)
            b0 = Uint32(info[2])
            low = b0 << Uint32(5)
            if not dense:
                # The low 5 bits of the maxima in bin b0: the bound becomes the
                # K-th largest vector maximum.
                for v in cutlass.range(l0, l1, unroll=1):
                    top = self.vector_top(tops_s, tops_address, v)
                    if (top >> Uint32(5)) == b0:
                        shared_add(fine_hist.iterator + (top & Uint32(31)), 1)
                cute.arch.sync_threads()
                if warp == 0:
                    need = Uint32(K) - Uint32(info[3])
                    digit, _ = low_digit(fine_hist[lane], need, lane)
                    if lane == 0:
                        info[4] = Int32(low | digit)
                cute.arch.sync_threads()
                if tid < LOW:
                    fine_hist[tid] = Uint32(0)
                cute.arch.sync_threads()
                low = Uint32(info[4])

            # Pass 2: the candidates (keys >= low) in key order. The hot vectors
            # (largest key >= low; about K of them) are listed from the maxima
            # in shared memory, then every thread loads a run of the list at
            # once, counts its keys >= low and writes them at its offset, with
            # their bins above b0 (the last bin also counts the keys past it).
            count = Uint32(0)
            for v in cutlass.range(l0, l1, unroll=1):
                if self.vector_top(tops_s, tops_address, v) >= low:
                    count += Uint32(1)
            slot, hot = block_exclusive(count, scan, warp, lane)
            if hot <= Uint32(HOT):
                for v in cutlass.range(l0, l1, unroll=1):
                    if self.vector_top(tops_s, tops_address, v) >= low:
                        hot_list[slot] = v
                        slot += Uint32(1)
            cute.arch.sync_threads()
            listed = Uint32(CANDIDATES + 1)
            if hot <= Uint32(HOT):
                per = (hot + SELECT_THREADS - 1) // SELECT_THREADS
                h0 = cutlass.min(Uint32(tid) * per, hot)
                h1 = cutlass.min(h0 + per, hot)
                batch = []
                for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                    words = (Uint32(0), Uint32(0), Uint32(0), Uint32(0))
                    v = Int32(0)
                    if h0 + u < h1:
                        v = hot_list[h0 + u]
                        words = self.vector(address, v)
                    batch.append((v, words))  # type: ignore[arg-type]
                count = Uint32(0)
                for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                    v, words = batch[u]
                    if h0 + u < h1:
                        keys = vector_keys(words, v, length)
                        count += self.count_at_least(keys, v, length, low)
                slot, listed = block_exclusive(count, scan, warp, lane)
                if listed <= Uint32(CANDIDATES):
                    for u in cutlass.range_constexpr(HOT // SELECT_THREADS):
                        v, words = batch[u]
                        if h0 + u < h1:
                            keys = vector_keys(words, v, length)
                            for e in cutlass.range_constexpr(8):
                                key = half(keys, e)
                                if key >= low and v * 8 + e < length:
                                    cand_index[slot] = v * 8 + e
                                    cand_key[slot] = key
                                    fine = fine_bin_of(key, b0)
                                    shared_add(fine_hist.iterator + fine, 1)
                                    slot += Uint32(1)
            if tid == 0:
                info[2] = Int32(FINE - 1)
                info[3] = Int32(0)
            cute.arch.sync_threads()
            topdown(fine_hist, FINE, Uint32(K), info, 2, scan, tid, warp, lane)
            fine_bin = Uint32(info[2])
            above_fine = Uint32(info[3])
            # The last fine bin is open-ended: a threshold there falls back.
            sm = hist, whist, wabove, wpre, pub, scan, info
            if listed > Uint32(CANDIDATES) or fine_bin == Uint32(FINE - 1):
                self.exact(sm, address, length, emit_args, tid, warp, lane)
            else:
                # The low 5 bits of the candidates in the threshold's bin; each
                # warp owns one contiguous share of the list.
                share = (listed + SELECT_WARPS - 1) // SELECT_WARPS
                first = cutlass.min(warp * share, listed)
                last = cutlass.min(first + share, listed)
                above = Uint32(0)
                for i in cutlass.range(first + lane, last, 32, unroll=1):
                    key = cand_key[i]
                    fine = fine_bin_of(key, b0)
                    if fine > fine_bin:
                        above += Uint32(1)
                    elif fine == fine_bin:
                        shared_add(whist.iterator + warp * LOW + (key & Uint32(31)), 1)
                for i in cutlass.range_constexpr(5):
                    above += cute.arch.shuffle_sync_bfly(above, 1 << i)
                if lane == 0:
                    wabove[warp] = above
                cute.arch.sync_threads()
                self.publish(whist, wabove, pub, tid)
                self.resolve(pub, info, whist, wabove, wpre, above_fine, warp, lane)
                threshold = ((b0 + fine_bin) << Uint32(5)) | Uint32(info[4])
                take = Uint32(info[5])
                # Pass 3: the selected candidates' positions in key order.
                carry_a = wpre[0, warp]
                carry_t = wpre[1, warp]
                for start in cutlass.range(first, last, 32, unroll=1):
                    i = start + lane
                    key = Uint32(0)
                    index = Int32(0)
                    n = Uint32(0)
                    if i < last:
                        key = cand_key[i]
                        index = cand_index[i]
                        n = Uint32(1)
                    carry_a, carry_t = self.write_selected(
                        key,
                        index,
                        n,
                        threshold,
                        take,
                        carry_a,
                        carry_t,
                        lane,
                        emit_args,
                    )
            if tid == 0:
                valid[row] = Int32(K)


# --------------------------------------------------------------------------
# DCP local top-N
# --------------------------------------------------------------------------
LOCAL_CACHED = 2  # vectors a thread keeps in registers across the passes


@asm()
def shared_or(pointer, value):
    return "red.shared.or.b32 [$0], $1;", "r,r,~{memory}", (pointer, Uint32(value))


@cute.jit
def warp_topdown(hist, need: Uint32, lane: Int32):
    """One warp over the BINS-bin `hist`: (the bin where the counts summed from
    the top bin down reach `need`, the count above it); (0, 0) when they never
    do. Lane l first sums the 64 bins [BINS - 64 (l + 1), BINS - 64 l), then the
    warp splits the hit chunk two bins per lane."""
    per = BINS // 32
    first = BINS - per * (lane + 1)
    mine = Uint32(0)
    for i in cutlass.range_constexpr(per // 4):
        w = smem_words(hist.iterator + first + 4 * i)
        mine += (w[0] + w[1]) + (w[2] + w[3])
    above = warp_inclusive(mine, lane) - mine
    hit = Int32(-1)
    if above < need and above + mine >= need:
        hit = lane
    for i in cutlass.range_constexpr(5):
        hit = cutlass.max(hit, cute.arch.shuffle_sync_bfly(hit, 1 << i))
    bin_ = Uint32(0)
    above_bin = Uint32(0)
    if hit >= 0:
        start = cute.arch.shuffle_sync(above, hit)
        # Bins top - 2 lane and top - 2 lane - 1 of the hit chunk.
        top = BINS - per * hit - 1 - 2 * lane
        hi = hist[top]
        lo = hist[top - 1]
        pair = hi + lo
        prior = start + warp_inclusive(pair, lane) - pair
        found = Int32(-1)
        if prior < need and prior + pair >= need:
            found = lane
        for i in cutlass.range_constexpr(5):
            found = cutlass.max(found, cute.arch.shuffle_sync_bfly(found, 1 << i))
        bin_ = Uint32(top)
        above_bin = prior
        if prior + hi < need:
            bin_ = Uint32(top - 1)
            above_bin = prior + hi
        bin_ = cute.arch.shuffle_sync(bin_, found)
        above_bin = cute.arch.shuffle_sync(above_bin, found)
    return bin_, above_bin


LOCAL_CANDIDATES = 4096  # keys at or above the bound (more: every key re-read)
LOCAL_ABOVE_BITS = 10  # a scanned count: keys above the threshold (< n), then ties


class IndexerLocalTopk:
    """DCP: the local top-`n` keys of every row (one CTA per row) of the
    DeepGEMM-layout logits (`IndexerScore` with `pitch`) as the DCP top-k
    merge's B' rows (`dcp_topk_merge`): `out` [rows, n + n / 32] int32,
    word j < n `(ord16 << 16) | (0xFFFF - (l & 0xFFFF))` of a selected key l
    (0: empty), bit j % 32 of word n + j // 32 bit 16 of l; `valid`: the
    counts. The selected keys are in key order: those above the threshold, then
    the ties at it (the lowest key indices). A row of at most `n` keys takes
    all of them.

    A row holds 1 / DCP of a context (a few thousand keys), so the select is
    bound by its chain of dependent steps: each thread owns a contiguous run of
    the row's 8-key vectors (the first `LOCAL_CACHED` kept in registers) and
    every histogram is scanned by one warp.
      0  each thread's largest key; the n-th largest of those bounds the n-th
         largest key from below (n threads hold a key at least as large), so
         only the keys at or above the bound's 11-bit bin are candidates (all
         keys when under n threads hold keys);
      1  the candidates, block-scanned into a shared list in key order (a few
         n at most; past `LOCAL_CANDIDATES` the passes below read every key
         instead, e.g. rows of many equal logits);
      2  their 11-bit histogram -> the threshold's bin;
      3  the low 5 bits of the candidates in it -> the threshold, the ties;
      4  each thread's count of candidates above it and ties at it, block-
         scanned -> their slots."""

    def __init__(self, rows: int, n: int, pitch: int, exchange: int = 0):
        assert n % 32 == 0 and n < 1 << LOCAL_ABOVE_BITS and pitch % 512 == 0
        self.rows, self.n, self.pitch = rows, n, pitch
        # exchange (W > 0): also push each row's B' words into every rank's
        # TopkExchange slot (topk_exchange.py) for ExchangeMerge.
        self.exchange = exchange

    @cute.jit
    def __call__(self, logits, row_ends, out, valid, peers, epoch, xrank, xrows, s):
        """`peers` [W] int64 / `epoch` [>= rows] / `xrank` / `xrows`: the
        TopkExchange's slot addresses, epochs, this rank and its rows (unused
        without `exchange`)."""
        self.kernel(logits, row_ends, out, valid, peers, epoch, xrank, xrows).launch(
            grid=(self.rows, 1, 1),
            block=(SELECT_THREADS, 1, 1),
            stream=s,
            use_pdl=True,
        )

    @cute.jit
    def push(self, out, row, tid, peers, epoch, xrank, xrows):
        """The row's B' words (written by this CTA) -> every rank's exchange
        slot [epoch parity, rank, row] as LL pairs (topk_exchange.ll_push)."""
        words = self.n + self.n // 32
        pairs = (words + 1) // 2
        cute.arch.sync_threads()
        ep = epoch[row] + Int32(1)
        tag = Uint32(ep)
        part = Int64(ep & Int32(1))
        src = out.iterator.toint() + Int64(row) * Int64(words * 4)
        for i in cutlass.range(tid, self.exchange * pairs, SELECT_THREADS, unroll=1):
            p = i // pairs
            c = i - p * pairs
            w0 = load_words(src + Int64(c) * 8, count=1, mutable=True)[0]
            w1 = Uint32(0)
            if 2 * c + 1 < words:
                w1 = load_words(src + Int64(c) * 8 + 4, count=1, mutable=True)[0]
            slot = (part * Int64(self.exchange) + Int64(xrank)) * Int64(xrows)
            dst = peers[p] + ((slot + Int64(row)) * Int64(pairs) + Int64(c)) * 16
            ll_push(dst, w0, w1, tag)

    @cute.jit
    def word(self, out, row, plane, slot, index, key):
        """Key `index` (16-bit `key`) into B' slot `slot`."""
        out[row, slot] = Int32(
            (key << Uint32(16)) | (Uint32(0xFFFF) - (Uint32(index) & Uint32(0xFFFF)))
        )
        if (index >> 16) != 0:
            shared_or(plane.iterator + slot // 32, Uint32(1) << Uint32(slot % 32))

    @cute.jit
    def candidate(self, step: cutlass.Constexpr[int], key, index, state, smem, args):
        """One candidate (key >= the bound) in pass `step` (2: histogram, 3:
        low histogram, 4: count, 5: write); `state`: (the threshold's bin, the
        threshold, the ties to take, the next slot above, the next tie rank)."""
        hist, low_hist, plane = smem
        out, row = args
        bin_, threshold, take, slot, rank = state
        count = Uint32(0)
        if cutlass.const_expr(step == 2):
            shared_add(hist.iterator + (key >> Uint32(5)), 1)
        if cutlass.const_expr(step == 3):  # noqa: SIM102
            if (key >> Uint32(5)) == bin_:
                shared_add(low_hist.iterator + (key & Uint32(31)), 1)
        if cutlass.const_expr(step == 4):
            if key > threshold:
                count = Uint32(1)
            elif key == threshold:
                count = Uint32(1 << LOCAL_ABOVE_BITS)
        if cutlass.const_expr(step == 5):
            if key > threshold:
                self.word(out, row, plane, Int32(slot), index, key)
                slot += Uint32(1)
            elif key == threshold:
                if rank < take:
                    tie = Int32(Uint32(self.n) - take + rank)
                    self.word(out, row, plane, tie, index, key)
                rank += Uint32(1)
        return count, (bin_, threshold, take, slot, rank)

    @cute.jit
    def keys_pass(
        self, step: cutlass.Constexpr[int], vec, cached, low, state, smem, args
    ):
        """Pass `step` over this thread's keys >= `low` (1: count them; else
        `candidate`'s) -> (the count, the state)."""
        address, v0, v1, length, per = vec
        count = Uint32(0)
        for i in cutlass.range(per, unroll=1):
            keys = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
            vmax = Uint32(0)
            for c in cutlass.range_constexpr(LOCAL_CACHED):
                if i == c:
                    keys, vmax = cached[c]
            if i >= LOCAL_CACHED and v0 + i < v1:
                words = load_words(address + Int64(v0 + i) * 16, count=4)
                keys = vector_keys(words, v0 + i, length)
                vmax = vector_max(keys)
            if vmax >= low:
                for e in cutlass.range_constexpr(8):
                    key = half(keys, e)
                    if key >= low and key != Uint32(0):
                        if cutlass.const_expr(step == 1):
                            count += Uint32(1)
                        else:
                            index = (v0 + i) * 8 + e
                            c, state = self.candidate(
                                step, key, index, state, smem, args
                            )
                            count += c
        return count, state

    @cute.kernel
    def kernel(self, logits, row_ends, out, valid, peers, epoch, xrank, xrows):
        N = self.n
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        smem = utils.SmemAllocator()
        tops = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        hist = smem.allocate_tensor(Uint32, cute.make_layout(BINS), 16)
        low_hist = smem.allocate_tensor(Uint32, cute.make_layout(LOW), 16)
        plane = smem.allocate_tensor(Uint32, cute.make_layout(N // 32), 16)
        scan = smem.allocate_tensor(Uint32, cute.make_layout(SELECT_WARPS), 16)
        info = smem.allocate_tensor(Int32, cute.make_layout(8), 16)
        cand_key = smem.allocate_tensor(Uint32, cute.make_layout(LOCAL_CANDIDATES), 16)
        cand_index = smem.allocate_tensor(Int32, cute.make_layout(LOCAL_CANDIDATES), 16)
        for b in cutlass.range(tid, BINS, SELECT_THREADS, unroll=1):
            tops[b] = Uint32(0)
            hist[b] = Uint32(0)
        if tid < LOW:
            low_hist[tid] = Uint32(0)
        if tid < N // 32:
            plane[tid] = Uint32(0)
        cute.arch.sync_threads()
        length = row_ends[row]
        address = logits.iterator.toint() + Int64(row) * Int64(self.pitch * 2)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if length <= N:
            # Every key, in key order (zero words past the count).
            for j in cutlass.range(tid, N, SELECT_THREADS, unroll=1):
                word = Uint32(0)
                high = False
                if j < length:
                    key = order16(load_u16(address + Int64(j) * 2))
                    word = (key << Uint32(16)) | (Uint32(0xFFFF) - Uint32(j & 0xFFFF))
                    high = (j >> 16) != 0
                out[row, j] = Int32(word)
                bits = cute.arch.vote_ballot_sync(high)
                if lane == 0:
                    out[row, N + j // 32] = Int32(bits)
            if tid == 0:
                valid[row] = cutlass.max(length, Int32(0))
        else:
            vectors = (length + 7) // 8
            per = (vectors + SELECT_THREADS - 1) // SELECT_THREADS
            v0 = cutlass.min(tid * per, vectors)
            v1 = cutlass.min(v0 + per, vectors)
            cached = []
            top = Uint32(0)
            for i in cutlass.range_constexpr(LOCAL_CACHED):
                keys = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
                if v0 + i < v1:
                    words = load_words(address + Int64(v0 + i) * 16, count=4)
                    keys = vector_keys(words, v0 + i, length)
                vmax = vector_max(keys)
                top = cutlass.max(top, vmax)
                cached.append((keys, vmax))
            vec = address, v0, v1, length, per
            sm = hist, low_hist, plane
            args = out, row
            zero = Uint32(0)
            # 0: the bound (when at least n threads hold keys).
            low = Uint32(1)
            if (vectors + per - 1) // per >= N:
                for v in cutlass.range(v0 + LOCAL_CACHED, v1, unroll=1):
                    words = load_words(address + Int64(v) * 16, count=4)
                    top = cutlass.max(top, vector_max(vector_keys(words, v, length)))
                if top != zero:
                    shared_add(tops.iterator + (top >> Uint32(5)), 1)
                cute.arch.sync_threads()
                if warp == 0:
                    b0, _ = warp_topdown(tops, Uint32(N), lane)
                    if lane == 0:
                        info[0] = Int32(b0)
                cute.arch.sync_threads()
                low = cutlass.max(Uint32(info[0]) << Uint32(5), Uint32(1))
            # 1: the candidates into the shared list.
            state = zero, zero, zero, zero, zero
            mine, _ = self.keys_pass(1, vec, cached, low, state, sm, args)
            first, listed = block_exclusive(mine, scan, warp, lane)
            listed_ok = listed <= Uint32(LOCAL_CANDIDATES)
            if listed_ok and mine != zero:
                slot = first
                for i in cutlass.range(per, unroll=1):
                    keys = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
                    vmax = Uint32(0)
                    for c in cutlass.range_constexpr(LOCAL_CACHED):
                        if i == c:
                            keys, vmax = cached[c]
                    if i >= LOCAL_CACHED and v0 + i < v1:
                        words = load_words(address + Int64(v0 + i) * 16, count=4)
                        keys = vector_keys(words, v0 + i, length)
                        vmax = vector_max(keys)
                    if vmax >= low:
                        for e in cutlass.range_constexpr(8):
                            key = half(keys, e)
                            if key >= low and key != zero:
                                cand_key[Int32(slot)] = key
                                cand_index[Int32(slot)] = (v0 + i) * 8 + e
                                slot += Uint32(1)
            cute.arch.sync_threads()
            # This thread's share of the list (in key order).
            share = (listed + SELECT_THREADS - 1) // SELECT_THREADS
            c0 = cutlass.min(Uint32(tid) * share, listed)
            c1 = cutlass.min(c0 + share, listed)
            ranges = c0, c1, listed_ok
            # 2: the threshold's 11-bit bin.
            self.list_pass(
                2, ranges, vec, cached, low, state, sm, args, cand_key, cand_index
            )
            cute.arch.sync_threads()
            if warp == 0:
                b1, above = warp_topdown(hist, Uint32(N), lane)
                if lane == 0:
                    info[1] = Int32(b1)
                    info[2] = Int32(above)
            cute.arch.sync_threads()
            bin_ = Uint32(info[1])
            need = Uint32(N) - Uint32(info[2])
            # 3: its low 5 bits -> the threshold, the ties to take.
            state = bin_, zero, zero, zero, zero
            self.list_pass(
                3, ranges, vec, cached, low, state, sm, args, cand_key, cand_index
            )
            cute.arch.sync_threads()
            if warp == 0:
                digit, above = low_digit(low_hist[lane], need, lane)
                if lane == 0:
                    info[3] = Int32((bin_ << Uint32(5)) | digit)
                    info[4] = Int32(need - above)
            cute.arch.sync_threads()
            threshold = Uint32(info[3])
            take = Uint32(info[4])
            # 4: the slots: the keys above in key order, then the ties taken.
            state = bin_, threshold, take, zero, zero
            count, _ = self.list_pass(
                4, ranges, vec, cached, threshold, state, sm, args, cand_key, cand_index
            )
            prior, _ = block_exclusive(count, scan, warp, lane)
            mask = Uint32((1 << LOCAL_ABOVE_BITS) - 1)
            above = count & mask
            rank = prior >> Uint32(LOCAL_ABOVE_BITS)
            if above != zero or (count != above and rank < take):
                state = bin_, threshold, take, prior & mask, rank
                self.list_pass(
                    5,
                    ranges,
                    vec,
                    cached,
                    threshold,
                    state,
                    sm,
                    args,
                    cand_key,
                    cand_index,
                )
            cute.arch.sync_threads()
            if tid < N // 32:
                out[row, N + tid] = Int32(plane[tid])
            if tid == 0:
                valid[row] = Int32(N)
        if cutlass.const_expr(self.exchange > 0):
            self.push(out, row, tid, peers, epoch, xrank, xrows)

    @cute.jit
    def list_pass(
        self,
        step: cutlass.Constexpr[int],
        ranges,
        vec,
        cached,
        low,
        state,
        smem,
        args,
        cand_key,
        cand_index,
    ):
        """Pass `step` over this thread's share of the candidate list, or over
        its keys >= `low` when the list overflowed -> (pass 4's count, the
        state)."""
        c0, c1, listed_ok = ranges
        count = Uint32(0)
        if listed_ok:
            for i in cutlass.range(c0, c1, unroll=1):
                c, state = self.candidate(
                    step, cand_key[i], cand_index[i], state, smem, args
                )
                count += c
        else:
            count, state = self.keys_pass(step, vec, cached, low, state, smem, args)
        return count, state


# --------------------------------------------------------------------------
# Host
# --------------------------------------------------------------------------
_SCRATCH: dict = {}


def logits_scratch(device, elements: int, name: str = "logits") -> torch.Tensor:
    """One scratch per device shared by every layer (every layer prepares before
    any graph capture, so the last, largest allocation serves them all)."""
    t = _SCRATCH.get((device, name))
    if t is None or t.numel() < elements:
        t = torch.empty(elements, dtype=torch.int16, device=device)
        _SCRATCH[device, name] = t
    return t


@functools.cache
def dummy_peers(device):
    """The local select's placeholder exchange addresses (no exchange)."""
    return torch.zeros(1, dtype=torch.int64, device=device)


@functools.cache
def select_outputs(device, rows, topk, fill=-1):
    """The physical rows (DCP: B' rows) and counts, shared by every layer (each
    layer's consumer reads them before the next layer's indexer runs), and the
    per-row ends of steps with fewer requests than rows."""
    out = torch.full((rows, topk), fill, dtype=torch.int32, device=device)
    valid = torch.zeros(rows, dtype=torch.int32, device=device)
    return out, valid, torch.zeros(rows, dtype=torch.int32, device=device)


def reserve(indexer, device, max_rows, local_topk=0, blocks=None):
    """The fused indexer's buffers of `max_rows` rows, shared by every layer:
    (logits, tops, rows, counts, ends, pitch). `local_topk` > 0: DCP (the
    merge's candidates per rank). `blocks`: the indexer cache's (None: no KV
    cache yet, so the longest context bounds the TP logits). PreparedDsa reserves
    them when the specialized decode is built, before vLLM profiles its memory for
    the KV cache; FusedIndexer (prepare_cache, after the KV cache is allocated)
    then takes the same buffers and allocates nothing."""
    if local_topk:
        # A rank's keys of a context (interleave 1), in 1 KiB rows.
        max_local_len = -(-indexer.max_model_len // indexer.dcp_world_size)
        pitch = -(-max_local_len // 512) * 512
        capacity = max_rows * pitch
        width, fill = local_topk + local_topk // 32, 0
    else:
        # Request-compact rows: at most every row a whole context (requests
        # may share their prefix blocks, so the cache size does not bound
        # the sum).
        pitch, context = 0, indexer.max_model_len
        if blocks is not None:
            context = min(context, blocks * PAGE)
        capacity = max_rows * (-(-context // TILE) * TILE + TILE)
        width, fill = indexer.topk_tokens, -1
    logits = logits_scratch(device, capacity)
    tops = logits_scratch(device, capacity // 8, "tops")
    out, valid, ends = select_outputs(device, max_rows, width, fill=fill)
    return logits, tops, out, valid, ends, pitch


def static_unsupported(indexer, rows, q) -> str | None:
    """Why the fused indexer cannot run this layer's (rows, q), from what is known
    before the KV cache exists (None: it can, if its cache layout does)."""
    from vllm.utils.deep_gemm import mqa_logits_use_mx_scales

    if not mqa_logits_use_mx_scales():
        return "needs the SM100 MX-scaled indexer cache"
    if indexer.topk_tokens % SELECT_THREADS or indexer.topk_tokens > 4096:
        return f"top-k {indexer.topk_tokens}"
    if not 1 <= q <= 8 or rows % q or rows // q > MAX_REQUESTS:
        return f"rows {rows} q {q}"
    return None


def supports(indexer, rows, q) -> str | None:
    """Why the fused indexer cannot run this layer (None: it can)."""
    reason = static_unsupported(indexer, rows, q)
    cache = indexer.k_cache.kv_cache
    if reason is None and (
        cache.dim() != 3
        or cache.shape[1] != PAGE
        or cache.shape[2] not in (DIM + 4, FP4_ENTRY)
    ):
        reason = (
            f"indexer cache {tuple(cache.shape)} (needs FP8 [blocks, 64, 132] or "
            f"MXFP4 [blocks, 64, {FP4_ENTRY}])"
        )
    return reason


@functools.cache
def compile_indexer(
    rows,
    q,
    topk,
    ctas,
    keys: tuple[tuple[int, ...], tuple[int, ...]],
    scales: tuple[tuple[int, ...], tuple[int, ...]],
    logits,
    tops,
    device,
    ids,
    pitch=0,
    packed=False,
    release=True,
    page=PAGE,
    fp4=False,
    exchange=0,
):
    """The score and select kernels of one (rows, q) for an indexer cache layout
    (every DSA layer's is the same). DCP (`pitch`): DeepGEMM-layout logits and,
    `packed`, the local top-`topk` select into B' rows (else none), which
    with `exchange` (W > 0) also pushes them to the W ranks (topk_exchange).
    `page`: the MLA cache's tokens per block (TP select). `fp4`: MXFP4 keys
    and query."""
    score = IndexerScore(rows, q, ctas, 0 if pitch else topk, pitch, fp4=fp4)
    fp8 = cutlass.Float8E4M3FN
    with torch.accelerator.device_index(device):
        if fp4:
            stride = (1, FP4_ROW, HEADS * DIM)
            query = fake((FP4_ROW, HEADS, rows), stride, torch.uint8)
            key_view = fake(*keys, torch.uint8)
            q_scales = fake((rows, HEADS), (HEADS * DIM // 4, 1), torch.int32, 4)
        else:
            query = fake((rows * HEADS, DIM, 1), (DIM, 1, rows * HEADS * DIM), fp8)
            key_view = fake(*keys, fp8)
            q_scales = fake((rows,), (1,), torch.int32, 4)  # unused: the row ends
        weights = fake((rows * HEADS,), (1,), torch.bfloat16)
        scale_view = fake(*scales, torch.int32)
        table = Int64(0), Int32(0), Int32(0)
        ends = fake((rows,), (1,), torch.int32, 4)
        scratch = fake(logits, (1,), cutlass.Int16)
        tops = fake(tops, (1,), cutlass.Int16)
        s = make_fake_stream()
        args: tuple = query, weights, key_view, scale_view, *table, ends, scratch, tops
        args += (q_scales, s)
        score_kernel = cute.compile(score, *args)
        if pitch:
            if not packed:
                return score_kernel, None
            width = topk + topk // 32
            out = fake((rows, width), (width, 1), torch.int32, 4)
            valid = fake((rows,), (1,), torch.int32, 4)
            local = IndexerLocalTopk(rows, topk, pitch, exchange)
            peers = dynamic_fake(1, torch.int64)
            epoch = dynamic_fake(1, torch.int32)
            xargs = peers, epoch, Int32(0), Int32(1)
            local = cute.compile(local, scratch, ends, out, valid, *xargs, s)
            return score_kernel, local
        select = IndexerSelect(rows, q, topk, logical=ids, release=release, page=page)
        out = fake((rows, topk), (topk, 1), torch.int32, 4)
        valid = fake((rows,), (1,), torch.int32, 4)
        args = scratch, tops, ends, *table[:2], Int32(0), out, valid, out, s
        select_kernel = cute.compile(select, *args)
    return score_kernel, select_kernel


class FusedIndexer:
    """The fused decode indexer of one DSA layer, compiled per (rows, q) for the
    layer's indexer cache. TP: `run` -> (physical rows [B, topk] of the MLA
    cache, valid counts [B]). DCP (`local_topk` N > 0, the top-k merge's
    candidates per rank): `run_dcp` scores this rank's keys into DeepGEMM's
    logits layout and, `packed`, selects each row's local top-N as the merge's
    B' rows. With an `exchange` (topk_exchange.TopkExchange) and the #63
    merge's rescue workspace, `run_exchange` instead pushes the B' rows to
    every rank over NVLink and merges them (ExchangeMerge) at the rows
    `exchanges` (`page`: the MLA cache's tokens per block)."""

    def __init__(
        self,
        indexer,
        shapes,
        device,
        max_rows,
        local_topk=0,
        packed=False,
        hold=(),
        page=PAGE,
        exchange=None,
    ):
        cache = indexer.k_cache.kv_cache
        # The (rows, q) whose select must not launch the next kernel early (TP: it
        # is a decode that reads the rows before its PDL wait; IndexerSelect).
        self.hold = frozenset(hold)
        self.indexer, self.device, self.page = indexer, device, page
        self.topk = local_topk or indexer.topk_tokens
        self.packed = packed and local_topk > 0
        self.fp4 = is_fp4_cache(cache)
        blocks, unit = cache.shape[0], cache.stride(0)
        row = FP4_ROW if self.fp4 else DIM  # key bytes per token
        if self.fp4:  # E2M1 bytes [64 B, 64 keys, blocks]
            self.keys = torch.as_strided(cache, (row, PAGE, blocks), (1, row, unit))
        else:
            self.keys = torch.as_strided(
                cache, (PAGE, DIM, blocks), (DIM, 1, unit)
            ).view(torch.float8_e4m3fn)
        words = cache.view(torch.int32)
        self.scales = torch.as_strided(
            words,
            (blocks, PAGE),
            (unit // 4, 1),
            storage_offset=words.storage_offset() + PAGE * row // 4,
        )
        if local_topk:  # a rank's keys of a context (interleave 1)
            self.max_local_len = -(-indexer.max_model_len // indexer.dcp_world_size)
        buffers = reserve(indexer, device, max_rows, local_topk, blocks)
        self.logits, self.tops, self.out, self.valid, self.ends, self.pitch = buffers
        self.ctas = torch.cuda.get_device_properties(device).multi_processor_count
        self.ids: torch.Tensor | None = None
        self.kernels = {shape: self.compile(*shape) for shape in shapes}
        # DCP exchange (run_exchange): its push-variant local select and merge
        # per shape whose rows the exchange and the rescue workspace hold.
        self.exchange, self.workspace = exchange, None
        self.exchange_kernels: dict = {}
        if exchange is not None and self.packed:
            from vllm.v1.attention.ops.dcp_topk_merge.kernels import (
                get_dcp_topk_rescue_workspace,
            )

            topk = indexer.topk_tokens
            self.workspace = get_dcp_topk_rescue_workspace(topk, local_topk)
            for rows, q in shapes:
                if self.exchanges(rows):
                    self.exchange_kernels[rows, q] = self.compile_exchange(rows, q)

    def compile(self, rows, q):
        logits = tuple(self.logits.shape)
        keys = tuple(self.keys.shape), tuple(self.keys.stride())
        scales = tuple(self.scales.shape), tuple(self.scales.stride())
        tops = tuple(self.tops.shape)
        args: tuple = rows, q, self.topk, self.ctas, keys, scales, logits, tops
        packed = self.packs(rows)
        args += (self.device.index, self.ids is not None, self.pitch, packed)
        release = (rows, q) not in self.hold
        return compile_indexer(*args, release=release, page=self.page, fp4=self.fp4)

    def exchanges(self, rows):
        """DCP: whether `run_exchange` serves `rows` rows (the exchange and
        rescue workspace hold them; the same on every rank). Also past the SM
        count: there the local select's rows share SMs (~1.7x slower than one
        row per SM), but the exchange still drops the NCCL all-gather, #63's
        DeepSelect + pack, select and rescue kernels (GB300 HIGH hybrid decode,
        310 rows at DCP8: the all-gather alone is 30-34 us per DSA layer)."""
        ws, ex = self.workspace, self.exchange
        return (
            ws is not None
            and self.packed
            and rows <= min(ws.max_rows, ex.rows)
            and (ws.world_size, ws.rank) == (ex.world, ex.rank)
        )

    def compile_exchange(self, rows, q, emit_ids=False):
        """(the push-variant local select, ExchangeMerge) of (rows, q)."""
        from .topk_exchange import ExchangeMerge

        logits = tuple(self.logits.shape)
        keys = tuple(self.keys.shape), tuple(self.keys.stride())
        scales = tuple(self.scales.shape), tuple(self.scales.stride())
        tops = tuple(self.tops.shape)
        args: tuple = rows, q, self.topk, self.ctas, keys, scales, logits, tops
        args += (self.device.index, False, self.pitch, True)
        world = self.exchange.world
        _, local = compile_indexer(*args, page=self.page, fp4=self.fp4, exchange=world)
        topk = self.indexer.topk_tokens
        merge = ExchangeMerge(
            self.exchange, self.workspace, rows, topk, self.page, emit_ids
        )
        return local, merge

    def packs(self, rows):
        """DCP: whether `run_dcp` selects the B' rows itself: one CTA per row
        on its own SM (past the SM count, rows sharing an SM select about 1.7x
        slower, behind DeepSelect + #63's pack; measured on GB300 at 160-320
        rows)."""
        return self.packed and rows <= self.ctas

    def write_ids(self, ids: torch.Tensor):
        """Also write the selected key indices (the logical top-k, -1 past the
        count) into `ids` [>= rows, topk] (tests; TP only)."""
        assert not self.pitch
        self.ids = ids
        self.kernels = {shape: self.compile(*shape) for shape in self.kernels}

    def _ids(self, out, rows):
        return out if self.ids is None else self.ids[:rows]

    @staticmethod
    def _table(t):
        """(address, row pitch, width) of an int32 block table view."""
        assert t.dtype == torch.int32 and t.stride(1) == 1
        return Int64(t.data_ptr()), Int32(t.stride(0)), Int32(t.shape[1])

    def score(self, q_fp8, head_w, index_decode, rows, q):
        """The scorer over the step's rows -> the per-row ends (cute)."""
        score, _ = self.kernels[rows, q]
        ends = index_decode.seq_lens.reshape(-1)
        index_table = index_decode.block_table
        if q > 1 and index_table.shape[0] == ends.shape[0]:
            index_table = index_table[::q]  # expanded per row
        if ends.shape[0] < rows:  # rows past the step's decodes: empty
            self.ends[ends.shape[0] : rows].zero_()
            self.ends[: ends.shape[0]].copy_(ends)
            ends = self.ends
        ends = ends[:rows]
        if self.fp4:  # the MXFP4 query in the query rows (index.py)
            packed, q_scales = fp4_query_views(q_fp8, rows)
            query = packed.permute(2, 1, 0)  # [64 B, 32, rows]
        else:
            query = torch.as_strided(
                q_fp8[:rows].view(torch.float8_e4m3fn),
                (rows * HEADS, DIM, 1),
                (DIM, 1, rows * HEADS * DIM),
            )
            q_scales = ends  # unused (any Int32 tensor)
        score(
            to_cute(query),
            to_cute(head_w[:rows].reshape(-1)),
            to_cute(self.keys),
            to_cute(self.scales),
            *self._table(index_table),
            to_cute(ends, 4),
            to_cute(self.logits),
            to_cute(self.tops),
            to_cute(q_scales, 4),
            stream(self.device),
        )
        return ends

    def run(self, q_fp8, head_w, index_decode, block_table, block_rows, rows, q):
        """TP. `index_decode`: the indexer's decode metadata (per-row ends and
        index block table); `block_table` / `block_rows`: the MLA cache's."""
        assert not self.pitch
        _, select = self.kernels[rows, q]
        ends = self.score(q_fp8, head_w, index_decode, rows, q)
        out, valid = self.out[:rows], self.valid[:rows]
        select(
            to_cute(self.logits),
            to_cute(self.tops),
            to_cute(ends, 4),
            *self._table(block_table[: rows // q])[:2],
            Int32(block_rows),
            to_cute(out, 4),
            to_cute(valid, 4),
            to_cute(self._ids(out, rows), 4),
            stream(self.device),
        )
        return out, valid

    def run_dcp(self, q_fp8, head_w, index_decode, rows, q):
        """DCP -> (this rank's local top-N of every row as B' rows [rows, N + N /
        32] when `packs(rows)`, else None; the BF16 logits [rows, pitch] in
        DeepGEMM's layout; the row ends [rows]). The B' rows' counts land in
        `valid`. The exact merge takes the logits with the B' rows (a saturated
        rank re-reads its row), so nothing here writes them until the next call."""
        assert self.pitch
        ends = self.score(q_fp8, head_w, index_decode, rows, q)
        logits = self.logits[: rows * self.pitch].view(torch.bfloat16)
        logits = logits.view(rows, self.pitch)
        if not self.packs(rows):
            return None, logits, ends
        _, select = self.kernels[rows, q]
        out, valid = self.out[:rows], self.valid[:rows]
        s = stream(self.device)
        select(
            to_cute(self.logits),
            to_cute(ends, 4),
            to_cute(out, 4),
            to_cute(valid, 4),
            *self.exchange_args(),
            s,
        )
        return out, logits, ends

    def exchange_args(self):
        """The local select's exchange arguments (placeholders without one)."""
        ex = self.exchange
        if ex is None:
            dummy = self.ends[:1]
            peers = dummy_peers(self.device)
            return dynamic_view(peers), dynamic_view(dummy), Int32(0), Int32(1)
        peers, epoch = dynamic_view(ex.peers), dynamic_view(ex.epoch)
        return peers, epoch, Int32(ex.rank), Int32(ex.rows)

    def run_exchange(
        self, q_fp8, head_w, index_decode, rows, q, merge, ids=None
    ):  # fmt: skip
        """DCP over the exchange (`exchanges(rows)`): score this rank's keys,
        push the local top-N B' rows to every rank, and merge every rank's
        into this rank's KV rows. `merge`: ExchangeMerge's (req_id,
        block_table, stride_rows, rows_out, valid_out, max_local_len) of the
        `rows` rows; `ids`: also the global ids [rows, K] (tests).
        Collective over the DCP group (every rank the same rows)."""
        assert self.pitch and (rows, q) in self.exchange_kernels
        req_id, block_table, stride_rows, rows_out, valid_out, max_local_len = merge
        ends = self.score(q_fp8, head_w, index_decode, rows, q)
        logits = self.logits[: rows * self.pitch].view(torch.bfloat16)
        logits = logits.view(rows, self.pitch)
        key = (rows, q) if ids is None else (rows, q, "ids")
        if key not in self.exchange_kernels:
            self.exchange_kernels[key] = self.compile_exchange(rows, q, emit_ids=True)
        local, merger = self.exchange_kernels[key]
        out, valid = self.out[:rows], self.valid[:rows]
        s = stream(self.device)
        local(
            to_cute(self.logits),
            to_cute(ends, 4),
            to_cute(out, 4),
            to_cute(valid, 4),
            *self.exchange_args(),
            s,
        )
        if max_local_len < logits.shape[1]:  # as the #63 merge's _bound_logits
            logits = logits[:, :max_local_len]
        merger(logits, ends, req_id, block_table, stride_rows, rows_out, valid_out, ids)
        return logits, ends
