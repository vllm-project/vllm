# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear layers at decode batch sizes on SM80, in CuTe DSL.

DeepSeek-V4.1's dense projections are MXFP8: FP8 E4M3 weights `W [N, K]`
with one UE8M0 scale (`2^(s - 127)`) per 32 x 32 weight block. Before SM100
they run as BF16 cuBLAS GEMMs on weights dequantized once at load. A decode
step multiplies them by a handful of rows (`M` = 1..48: up to eight
requests of six tokens), where a GEMM reads its weights once and does almost
no arithmetic per byte, so its time is the weight read plus fixed costs.
This kernel computes `y [M, N] = x [M, K] . dequant(W)^T` (`x` and `y`
BF16) reading the FP8 weights, half the bytes of the BF16 copy.

Products. The kernel decodes each FP8 weight to BF16 in registers and
multiplies on BF16 tensor cores (`mma.sync` m16n8k16, FP32 accumulation):
weight rows are the MMA's M (16 per tile) and the rows of `x` its N (8 per
tile, up to six tiles), as the sparse MLA kernel does with keys and heads.
An E4M3 code `s eeee mmm` placed as the BF16 `s | eeeemmm << 4` is its
value times `2^-120` (subnormal codes included, see codecs.py); one
`fma.rn.bf16x2` by `2^(s - 7)` (the block scale times `2^120`) gives the
dequantized weight. That product is exact whenever the result is a normal
BF16, so every weight equals the one the load-time dequantization produces,
and the result differs from the cuBLAS path only in the FP32 summation
order. The combined factor must itself be a BF16, which holds for scale
bytes up to 134 (scales up to `2^7`); `pack_weight` checks it.

Weight layout. `pack_weight` reorders the weights once, at load, into
512-byte blocks of 16 rows x 32 columns (one scale block's column range), in
which lane `l` of a warp finds, as 16 consecutive bytes, the 16 values its
two MMA A fragments need. A warp's load of one block is a single 16-byte
load per lane of 512 consecutive bytes. The K order inside a 32-column group
is permuted (both operands see the same permutation, which leaves the dot
product unchanged): lane `(g, q) = (l / 4, l % 4)` owns columns
`8q .. 8q + 7` of rows `g` and `g + 8`, so its B fragments are the 16
consecutive bytes `x[row, 8q .. 8q + 7]`, one 16-byte shared-memory read.
Within a word, bytes 0 and 2 are row `g` (two adjacent columns) and bytes
1 and 3 are row `g + 8`, so each word yields one BF16 pair of each row
with shifts and masks only. The scales become one word per 32 x 32 block
(`[N / 32, K / 32]`): the factor `2^(s - 7)` in both BF16 halves.

Work split. A CTA owns `16 warps_n` weight rows and `32 warps_k groups`
columns: warp `(wn, wk)` the 16 rows of tile `wn` and the `groups`
32-column groups at `wk`. All copies are `cp.async` to shared memory,
issued before any is used: first (one copy group) the CTA's columns of
`x` (rows past `M` zero-filled) and its scale factors, then each lane's
16 bytes of each of its warp's blocks, one copy group per block. A warp
then waits for its blocks one at a time (`cp.async.wait_group`), so it
decodes block `i` while the later ones are still in flight. (Loads into
registers would do the same with fewer instructions, but the assembler
sinks them next to their uses and keeps only a couple in flight.) After the
MMAs, the `warps_k` partial sums of each output meet in shared memory and
are added in warp order. When the columns are split over `splits` CTAs,
each CTA writes its FP32 partial sums to a workspace and takes an atomic
ticket; the last CTA of a row block adds the `splits` partial sums in
split order and writes BF16. The order of every FP32 addition is therefore
fixed: results are deterministic and do not depend on which CTA finishes
last. The ticket counters reset themselves (CUDA-graph safe). By default a
device's launches share one set, so they must be ordered on one stream at a
time; a caller whose launches can overlap on several streams gives each
weight its own counters.

`default_config` picks the split for a shape and device (its docstring
gives the model); `mxfp8_gemv` takes an explicit one too.
"""

from dataclasses import dataclass
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from .codecs import NEGATIVE_ZERO_BF16X2, fma_bf16x2, store_u32
from .mhc import fence_gpu, pack_bf16x2, ticket

TILE_ROWS = 16  # weight rows of one MMA tile (its M)
GROUP = 32  # columns of a scale block and of a packed weight block
BLOCK_BYTES = TILE_ROWS * GROUP
LANE_BYTES = BLOCK_BYTES // 32
X_TILE = 8  # rows of x per MMA tile (its N)
MAX_ROWS = 48
# The largest UE8M0 byte whose combined factor 2^(s - 7) is a BF16.
MAX_SCALE_CODE = 134
# Ticket counters per device: one per row block of a launch.
COUNTERS = 4096
# The weight copy: L2-only (weights are read once).
WEIGHT_COPY = "cp.async.cg.shared.global"


def _asm(result, operands, text, constraints, side_effects, loc, ip):
    """One inline PTX statement."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _words(loaded, count: int, kind, wrap, loc, ip) -> list:
    """The `count` members of an inline-assembly result struct."""
    return [
        wrap(llvm.extractvalue(kind, loaded, [i], loc=loc, ip=ip)) for i in range(count)
    ]


@dsl_user_op
def copy_async_weights(shared: Int32, address: Int64, *, loc=None, ip=None):
    """`cp.async` of 16 weight bytes to shared memory (L1 bypassed)."""
    _asm(
        None,
        [
            Int32(shared).ir_value(loc=loc, ip=ip),
            Int64(address).ir_value(loc=loc, ip=ip),
        ],
        f"{WEIGHT_COPY} [$0], [$1], 16;",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def copy_async_word(shared: Int32, address: Int64, *, loc=None, ip=None):
    """`cp.async` of one 4-byte word to shared memory."""
    _asm(
        None,
        [
            Int32(shared).ir_value(loc=loc, ip=ip),
            Int64(address).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def load_shared_word(shared: Int32, *, loc=None, ip=None) -> Uint32:
    """One word of shared memory."""
    return Uint32(
        _asm(
            T.i32(),
            [Int32(shared).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def copy_async(shared: Int32, address: Int64, size: Int32, *, loc=None, ip=None):
    """`cp.async` of 16 bytes to shared memory, of which the first `size`
    (16 or 0) come from `address` and the rest are zeros."""
    _asm(
        None,
        [
            Int32(shared).ir_value(loc=loc, ip=ip),
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def load_shared(shared: Int32, *, loc=None, ip=None) -> list:
    """16 bytes of shared memory (16-byte aligned) as four words."""
    loaded = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(shared).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return _words(loaded, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def mma_bf16(c: list, a: list, b: list, *, loc=None, ip=None) -> list:
    """`mma.sync` m16n8k16 with BF16 operands and FP32 accumulators: the
    four accumulators of `c` plus the product of A (four words) and B (two
    words)."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (*a, *b)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return _words(loaded, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def load_partials(address: Int64, *, loc=None, ip=None) -> list:
    """Four FP32 partial sums another CTA wrote, read from L2; the load stays
    after the ticket that orders it."""
    loaded = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        True,
        loc,
        ip,
    )
    return _words(loaded, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def store_partials(address: Int64, values: list, *, loc=None, ip=None) -> None:
    """Store four FP32 values (16-byte aligned)."""
    _asm(
        None,
        [Int64(address).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def store_pair(address: Int64, low: Uint32, high: Uint32, *, loc=None, ip=None):
    """Store two words (8-byte aligned)."""
    _asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(low).ir_value(loc=loc, ip=ip),
            Uint32(high).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        True,
        loc,
        ip,
    )


@cute.jit
def decode_pairs(word: Uint32, scale: Uint32) -> tuple:
    """The BF16 pairs of a packed weight word: bytes 0 and 2 (row `g`) and
    bytes 1 and 3 (row `g + 8`), each times the BF16x2 factor `scale`
    (exact; module docstring)."""
    even = ((word << 8) & 0x80008000) | ((word << 4) & 0x07F007F0)
    odd = (word & 0x80008000) | ((word >> 4) & 0x07F007F0)
    negative_zero = Uint32(NEGATIVE_ZERO_BF16X2)
    return (
        fma_bf16x2(even, scale, negative_zero),
        fma_bf16x2(odd, scale, negative_zero),
    )


@dataclass(frozen=True)
class GemvConfig:
    """How a launch splits the work: `warps_n` x `warps_k` warps per CTA,
    each with `groups` 32-column groups of one 16-row tile."""

    warps_n: int
    warps_k: int
    groups: int

    def splits(self, k: int) -> int:
        """CTAs that share a row block's columns."""
        return k // GROUP // (self.warps_k * self.groups)


class Mxfp8Gemv:
    """`y = x . dequant(W)^T` for one weight shape and `x_tiles` x 8 rows
    of `x`; see the module docstring."""

    def __init__(self, n: int, k: int, x_tiles: int, config: GemvConfig):
        assert n % TILE_ROWS == 0 and k % GROUP == 0
        assert 1 <= x_tiles <= MAX_ROWS // X_TILE
        self.n = n
        self.k = k
        self.x_tiles = x_tiles
        self.warps_n = config.warps_n
        self.warps_k = config.warps_k
        self.groups = config.groups
        assert (n // TILE_ROWS) % self.warps_n == 0
        self.cta_groups = self.warps_k * self.groups
        assert (k // GROUP) % self.cta_groups == 0
        self.splits = config.splits(k)
        self.row_blocks = n // TILE_ROWS // self.warps_n
        self.threads = 32 * self.warps_n * self.warps_k
        self.cta_rows = TILE_ROWS * self.warps_n
        self.cta_cols = GROUP * self.cta_groups
        self.x_rows = X_TILE * x_tiles
        # Shared x tile row stride in words: consecutive rows 64 bytes apart
        # modulo 128, so a quarter warp's eight 16-byte reads (two rows) hit
        # distinct banks.
        words = self.cta_cols // 2
        self.row_words = words if words % 32 == 16 else words + 16
        self.x_words = self.x_rows * self.row_words
        # Each warp's weight blocks follow the x tile.
        self.w_words = self.threads // 32 * self.groups * (BLOCK_BYTES // 4)
        # Scale rows the CTA's tiles touch (an odd first tile starts mid-row).
        self.scale_rows = self.warps_n // 2 + 1
        # Partial sums [warps_k][x row][weight row], padded against conflicts;
        # they reuse the x tile and weight space after the products.
        self.red_stride = self.cta_rows + 4
        self.buf_words = max(
            self.x_words + self.w_words,
            self.warps_k * self.x_rows * self.red_stride,
        )

    def smem_bytes(self) -> int:
        """Shared memory of one CTA (alignment padding included)."""
        return (self.buf_words + self.scale_rows * self.cta_groups) * 4 + 64

    @cute.jit
    def __call__(
        self,
        weight: Int64,
        scales: Int64,
        x: Int64,
        y: Int64,
        partials: Int64,
        counters: Int64,
        rows: Int32,
        x_stride: Int32,
        y_stride: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Storage:
            buf: cute.struct.Align[cute.struct.MemRange[Uint32, self.buf_words], 16]
            scale: cute.struct.Align[
                cute.struct.MemRange[Uint32, self.scale_rows * self.cta_groups], 16
            ]
            last: cute.struct.MemRange[Int32, 1]

        self.kernel(
            weight, scales, x, y, partials, counters, rows, x_stride, y_stride, Storage
        ).launch(
            grid=(self.row_blocks, self.splits, 1),
            block=[self.threads, 1, 1],
            # cute.struct adds size_in_bytes to the class at runtime.
            smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        weight: Int64,
        scales: Int64,
        x: Int64,
        y: Int64,
        partials: Int64,
        counters: Int64,
        rows: Int32,
        x_stride: Int32,
        y_stride: Int32,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, split, _ = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        wn = warp % self.warps_n
        wk = warp // self.warps_n
        g = lane // 4
        q = lane % 4
        groups = self.groups
        k_groups = self.k // GROUP
        tile = block * self.warps_n + wn
        first = split * self.cta_groups + wk * groups

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        x_base = Int32(storage.buf.data_ptr().toint())
        scale_base = Int32(storage.scale.data_ptr().toint())

        # ---- copy group 0: the CTA's columns of x and its scale factors ----
        vectors = self.cta_cols // 8  # 16-byte vectors per row
        total = self.x_rows * vectors
        for it in cutlass.range_constexpr((total + self.threads - 1) // self.threads):
            c = tid + it * self.threads
            if c < total:
                r = c // vectors
                v = c % vectors
                source = x
                size = Int32(0)
                if r < rows:
                    source = (
                        x
                        + (
                            Int64(r) * Int64(x_stride)
                            + Int64(split * self.cta_cols + v * 8)
                        )
                        * 2
                    )
                    size = Int32(16)
                copy_async(x_base + (r * self.row_words + v * 4) * 4, source, size)
        # Factors [scale row - first scale row][CTA group].
        scale_row0 = block * self.warps_n // 2
        total = self.scale_rows * self.cta_groups
        for it in cutlass.range_constexpr((total + self.threads - 1) // self.threads):
            c = tid + it * self.threads
            if c < total:
                r = cutlass.min(scale_row0 + c // self.cta_groups, self.n // GROUP - 1)
                column = split * self.cta_groups + c % self.cta_groups
                copy_async_word(
                    scale_base + c * 4,
                    scales + (Int64(r) * k_groups + Int64(column)) * 4,
                )
        cute.arch.cp_async_commit_group()

        # ---- copy groups 1..groups: this lane's 16 bytes of each block ----
        # Every block of the warp is in flight before any is used.
        w_lane = (
            x_base
            + (self.x_words + warp * groups * (BLOCK_BYTES // 4)) * 4
            + lane * LANE_BYTES
        )
        address = (
            weight
            + (Int64(tile) * k_groups + Int64(first)) * BLOCK_BYTES
            + Int64(lane * LANE_BYTES)
        )
        for i in cutlass.range_constexpr(groups):
            copy_async_weights(w_lane + i * BLOCK_BYTES, address + i * BLOCK_BYTES)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(groups)
        cute.arch.barrier()

        # ---- decode and multiply, one 32-column group at a time ----
        acc = cute.make_rmem_tensor(cute.make_layout((self.x_tiles, 4)), Float32)
        acc.fill(0.0)
        scale_row = (tile // 2 - scale_row0) * self.cta_groups + wk * groups
        for i in cutlass.range_constexpr(groups):
            # Only this lane's copies of blocks 0..i must have landed.
            cute.arch.cp_async_wait_group(groups - 1 - i)
            raw = load_shared(w_lane + i * BLOCK_BYTES)
            scale = load_shared_word(scale_base + (scale_row + i) * 4)
            a = []
            for j in cutlass.range_constexpr(4):
                even, odd = decode_pairs(raw[j], scale)
                a.append(even)
                a.append(odd)
            column = (wk * groups + i) * 16 + q * 4
            for t in cutlass.range_constexpr(self.x_tiles):
                b = load_shared(
                    x_base + ((t * X_TILE + g) * self.row_words + column) * 4
                )
                c = [acc[t, j] for j in range(4)]
                c = mma_bf16(c, a[0:4], b[0:2])
                c = mma_bf16(c, a[4:8], b[2:4])
                for j in cutlass.range_constexpr(4):
                    acc[t, j] = c[j]
        cute.arch.barrier()

        # ---- the warps' partial sums, [warps_k][x row][weight row] ----
        red = storage.buf.get_tensor(
            cute.make_layout(
                (self.warps_k, self.x_rows, self.cta_rows),
                stride=(self.x_rows * self.red_stride, self.red_stride, 1),
            ),
            dtype=Float32,
        )
        for t in cutlass.range_constexpr(self.x_tiles):
            for j in cutlass.range_constexpr(4):
                x_row = t * X_TILE + 2 * q + j % 2
                red[wk, x_row, wn * TILE_ROWS + g + 8 * (j // 2)] = acc[t, j]
        cute.arch.barrier()

        # ---- add them in warp order; BF16 out, or this split's FP32 sums ----
        quads = self.cta_rows // 4
        total = self.x_rows * quads
        trips = (total + self.threads - 1) // self.threads
        n0 = block * self.cta_rows
        for it in cutlass.range_constexpr(trips):
            c = tid + it * self.threads
            r = c // quads
            v = (c % quads) * 4
            if c < total and r < rows:
                sums = [red[0, r, v + e] for e in range(4)]
                for w in cutlass.range_constexpr(1, self.warps_k):
                    for e in cutlass.range_constexpr(4):
                        sums[e] = sums[e] + red[w, r, v + e]
                if cutlass.const_expr(self.splits == 1):
                    store_pair(
                        y + (Int64(r) * Int64(y_stride) + Int64(n0 + v)) * 2,
                        pack_bf16x2(sums[0], sums[1]),
                        pack_bf16x2(sums[2], sums[3]),
                    )
                else:
                    store_partials(
                        partials
                        + ((Int64(split) * self.x_rows + r) * self.n + Int64(n0 + v))
                        * 4,
                        sums,
                    )

        if cutlass.const_expr(self.splits > 1):
            # The last CTA of the row block adds the splits' sums in split
            # order. Every thread fences its stores before the barrier and
            # thread 0's ticket; in the last CTA, the ticket's acquire, the
            # barrier and each thread's fence order the other CTAs' sums
            # before its reads.
            last = storage.last.get_tensor(cute.make_layout(1))
            fence_gpu()
            cute.arch.barrier()
            counter = counters + Int64(block) * 4
            if tid == 0:
                last[0] = Int32(0)
                if ticket(counter) == Uint32(self.splits - 1):
                    last[0] = Int32(1)
            cute.arch.barrier()
            if last[0] == 1:
                fence_gpu()
                for it in cutlass.range_constexpr(trips):
                    c = tid + it * self.threads
                    r = c // quads
                    v = (c % quads) * 4
                    if c < total and r < rows:
                        base = partials + (Int64(r) * self.n + Int64(n0 + v)) * 4
                        loaded = [
                            load_partials(base + Int64(s * self.x_rows * self.n * 4))
                            for s in range(self.splits)
                        ]
                        sums = loaded[0]
                        for s in cutlass.range_constexpr(1, self.splits):
                            for e in cutlass.range_constexpr(4):
                                sums[e] = sums[e] + loaded[s][e]
                        store_pair(
                            y + (Int64(r) * Int64(y_stride) + Int64(n0 + v)) * 2,
                            pack_bf16x2(sums[0], sums[1]),
                            pack_bf16x2(sums[2], sums[3]),
                        )
                if tid == 0:
                    store_u32(counter, Uint32(0))


def pack_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's layout of an MXFP8 weight (done once, at load).

    `weight` is `[N, K]` E4M3 (`float8_e4m3fn` or its bytes) and
    `scales` the UE8M0 bytes `[N / 32, K / 32]`, or `[N, K / 32]` with
    each 32-row block's rows equal (the loader's row-expanded copy). Returns
    the packed weights `[N / 16, K / 32, 512]` (uint8; module docstring)
    and the scale factors `[N / 32, K / 32]` (int32): each block's
    `2^(s - 7)` as BF16 in both halves of a word.
    """
    n, k = weight.shape
    assert n % (2 * TILE_ROWS) == 0 and k % GROUP == 0
    codes = weight.contiguous().view(torch.uint8)
    blocks = scales.view(torch.uint8)
    if blocks.shape[0] == n:
        expanded = blocks.view(n // GROUP, GROUP, k // GROUP)
        blocks = expanded[:, 0]
        assert torch.equal(expanded, blocks.unsqueeze(1).expand_as(expanded)), (
            "the scales differ within a 32-row block"
        )
    assert blocks.shape == (n // GROUP, k // GROUP)
    assert int(blocks.max()) <= MAX_SCALE_CODE, "a weight scale exceeds 2^7"
    # [tile, half, g, group, q, word, byte pair] -> [tile, group, g, q, word,
    # column of the pair, half]: lane 4g + q's 16 bytes, rows g and g + 8
    # interleaved byte by byte.
    packed = (
        codes.view(n // TILE_ROWS, 2, 8, k // GROUP, 4, 4, 2)
        .permute(0, 3, 2, 4, 5, 6, 1)
        .contiguous()
        .view(n // TILE_ROWS, k // GROUP, BLOCK_BYTES)
    )
    # The BF16 exponent field of 2^(s - 7) is s + 120.
    factors = (blocks.to(torch.int32) + 120) * 0x00800080
    return packed, factors.contiguous()


@cache
def _device_limits(device_index: int) -> tuple[int, int]:
    """(SM count, opt-in shared memory per CTA in bytes) of a device."""
    props = torch.cuda.get_device_properties(device_index)
    # The property exists at runtime; torch's type stub does not list it.
    smem = props.shared_memory_per_block_optin  # type: ignore[attr-defined]
    return props.multi_processor_count, smem


# default_config's cost model, in bytes read by the busiest SM: an x byte
# counts as X_COST weight bytes, a column split costs SPLIT_COST plus the
# partial sums its last CTA reads back, and each block a warp handles in
# sequence BLOCK_COST (its decode and MMAs do not overlap one another).
X_COST = 1.0
SPLIT_COST = 32 << 10
BLOCK_COST = 2 << 10
# Column splits whose last CTA reads more partial-sum bytes are not tried.
MAX_SPLIT_READ = 96 << 10


@cache
def default_config(n: int, k: int, rows: int, device: torch.device) -> GemvConfig:
    """The work split of a shape (module docstring).

    Candidates: CTAs of 1 to 16 warps (a power of two) that fit the
    device's shared memory, at most 16 blocks per warp, between half and
    four times as many CTAs as SMs (when any split allows it), and column
    splits whose last CTA reads at most `MAX_SPLIT_READ`. It takes the
    one whose busiest SM has the least modeled cost (the constants above,
    fitted to sweeps of the DeepSeek-V4.1 shapes).
    """
    sms, smem_limit = _device_limits(device.index)
    x_tiles = (rows + X_TILE - 1) // X_TILE
    tiles = n // TILE_ROWS
    k_groups = k // GROUP
    candidates: list[tuple[bool, float, GemvConfig]] = []
    for warps_n in (1, 2, 4, 8, 16):
        for warps_k in (1, 2, 4, 8, 16):
            if tiles % warps_n or warps_n * warps_k > 16:
                continue
            for groups in range(1, 17):
                if k_groups % (warps_k * groups):
                    continue
                config = GemvConfig(warps_n, warps_k, groups)
                kernel = Mxfp8Gemv(n, k, x_tiles, config)
                split_read = kernel.splits * kernel.cta_rows * kernel.x_rows * 4
                if kernel.smem_bytes() > smem_limit or (
                    kernel.splits > 1 and split_read > MAX_SPLIT_READ
                ):
                    continue
                ctas = kernel.row_blocks * kernel.splits
                busiest = -(-ctas // sms)
                cost = busiest * (
                    kernel.cta_rows * kernel.cta_cols
                    + X_COST * 2 * rows * kernel.cta_cols
                    + BLOCK_COST * groups
                )
                if kernel.splits > 1:
                    cost += SPLIT_COST + kernel.splits * kernel.cta_rows * rows * 4
                fills = sms // 2 <= ctas <= 4 * sms
                candidates.append((not fills, cost, config))
    assert candidates, f"no split fits a {n} x {k} weight"
    return min(candidates, key=lambda c: (c[0], c[1]))[2]


@cache
def _kernel(device_index: int, n: int, k: int, x_tiles: int, config: GemvConfig):
    """`Mxfp8Gemv` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            Mxfp8Gemv(n, k, x_tiles, config),
            *(Int64(0),) * 6,
            *(Int32(1),) * 3,
            make_fake_stream(),
        )


@cache
def _counters(device_index: int) -> torch.Tensor:
    """The ticket counters of a device (zero between launches)."""
    return torch.zeros(COUNTERS, dtype=torch.int32, device=f"cuda:{device_index}")


def mxfp8_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor | None = None,
    config: GemvConfig | None = None,
    counters: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x [M, K] . dequant(W)^T` as BF16 `[M, N]`, for `M <= 48`.

    `weight` and `scales` come from `pack_weight`; `x` is BF16 with
    16-byte aligned rows. `config` overrides `default_config`.
    `counters` (int32, zero, at least `N / 16`) replaces the device's
    shared ticket counters.
    """
    rows, k = x.shape
    n = weight.shape[0] * TILE_ROWS
    assert weight.shape == (n // TILE_ROWS, k // GROUP, BLOCK_BYTES)
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    assert scales.shape == (n // GROUP, k // GROUP) and scales.is_contiguous()
    assert scales.dtype == torch.int32
    assert 1 <= rows <= MAX_ROWS and x.dtype == torch.bfloat16
    assert x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    if out is None:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=x.device)
    assert out.shape == (rows, n) and out.stride(1) == 1
    assert out.stride(0) % 4 == 0 and out.data_ptr() % 8 == 0
    if config is None:
        config = default_config(n, k, rows, x.device)
    x_tiles = (rows + X_TILE - 1) // X_TILE
    splits = config.splits(k)
    device = x.device.index
    if counters is None:
        counters = _counters(device)
    assert counters.dtype == torch.int32 and counters.is_contiguous()
    assert n // TILE_ROWS // config.warps_n <= counters.numel()
    work = splits * X_TILE * x_tiles * n if splits > 1 else 0
    partials = torch.empty(work, dtype=torch.float32, device=x.device)
    _kernel(device, n, k, x_tiles, config)(
        Int64(weight.data_ptr()),
        Int64(scales.data_ptr()),
        Int64(x.data_ptr()),
        Int64(out.data_ptr()),
        Int64(partials.data_ptr()),
        Int64(counters.data_ptr()),
        Int32(rows),
        Int32(x.stride(0)),
        Int32(out.stride(0)),
        current_cu_stream(),
    )
    return out
