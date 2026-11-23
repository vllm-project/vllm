# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 indexer logits on SM80, in CuTe DSL.

The indexer scores each query token `m` against keys `n` of its request:

    logits[m, n] = sum_h weights[m, h] * relu(q[m, h] . k[n])

with MXFP4 queries (`H` heads of 128 dims) and MXFP4 keys (128 dims, one
UE8M0 scale per 32). `MqaLogits` computes one query token per CTA row of
the grid and a run of 64-key tiles per CTA: it decodes the token's queries
once into a BF16 tile `[H, 128]` in shared memory, then per key tile decodes
the keys into `[64, 128]`, multiplies them on `mma.sync` BF16 tensor cores
(`ldmatrix` operands, FP32 accumulation; warps split the keys, 8 each),
applies the ReLU and the head weights to the accumulators and sums over the
heads with warp shuffles. Decoding is exact (codecs.py), so the result
differs from `reference.mqa_logits` only in the FP32 summation order, and
a key's score does not depend on which CTA computes it.

The grid is `(splits, rows)`: CTA `split` of a row takes an equal share,
`ceil(tiles / splits)` consecutive tiles, of the row's own key tiles, so a
row's length (read on the device) bounds the work and a CUDA graph captured
for the longest rows wastes nothing on shorter ones. Prefill passes enough
splits for at most 8 tiles per CTA; decode passes splits for a few CTAs per
SM, whatever the logits width.

A speculative decode step scores several tokens of each request against the
same keys. A decode launch may group `group` consecutive rows (equal block
tables) into one CTA: its query tile is `[group * H, 128]`, each key tile
is decoded once for the group, and the tile range covers the group's longest
row. Every row's scores are summed in the same order as alone, so grouping
changes no bit.

The keys come from one of two sources, chosen at compile time:

* ragged (prefill): the gathered keys of the chunk, values `[N, 64]` and
  scales `[N, 4]`; row `m` scores keys `[ks[m], ke[m])` into columns
  `0 ..`;
* paged (decode): the indexer key cache, whose pages hold `block_size`
  64-byte values then their 4-byte scales; row `m` scores the first
  `lens[m]` keys of request `m // next_n`'s block table.

Columns past a row's length are not written.

A decode step's candidate consumers (the two-level selection) score only
their row's candidate blocks of `block` keys (`candidate_mqa_logits`),
which the stock layer instead masks to `-inf` in dense logits: row `m`'s
compact column `c` is key `block * candidates[m, c // block] + c % block`,
so a row costs its candidates (`2048 * 8` keys) whatever its length. The
kernel scores the row's first `min(count, ceil(len / block))` candidate
slots, where its valid blocks must lie, and writes that many blocks' keys as
the row's compact length; a `-1` slot or a key at or past the row's length
scores `-inf`. A key scores the same bits as in a dense launch. Compact rows also
take SM90's FP8 index cache (pages of 128-byte E4M3 keys, then their FP32
scales) and FP8 queries whose scales the head weights carry: a key then
scores `scale * sum_h weights[m, h] * relu(q[m, h] . k)`.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from .codecs import (
    e2m1,
    e4m3,
    f32_from_bits,
    load_s32,
    load_u32,
    store_u32,
    ue8m0,
)

DIM = 128
KEYS = 64
THREADS = 256
WARPS = THREADS // 32
MAX_HEADS = 64
# 64-key tiles per CTA of a prefill row: amortizes the query decoding.
TILES_PER_CTA = 8
# CTAs per SM a decode launch aims at, for single rows and for row groups
# (whose CTAs hold the queries of a whole group).
DECODE_CTAS_PER_SM = 8
GROUP_CTAS_PER_SM = 4


@cute.jit
def _decode_e4m3_chunk(values: Int64, chunk: Int32, out: cute.Tensor) -> None:
    """Dims `8 chunk .. 8 chunk + 7` of an E4M3 row (128 bytes at
    `values`) into `out[0:8]` as BF16."""
    for half in cutlass.range_constexpr(2):
        codes = load_u32(values + chunk * 8 + half * 4)
        for i in cutlass.range_constexpr(4):
            out[4 * half + i] = e4m3(codes >> (8 * i)).to(BFloat16)


@cute.jit
def _decode_mxfp4_chunk(
    values: Int64, scales: Int64, chunk: Int32, out: cute.Tensor
) -> None:
    """Dims `8 chunk .. 8 chunk + 7` of an MXFP4 row (64 value bytes at
    `values`, 4 scale bytes at `scales`) into `out[0:8]` as BF16."""
    codes = load_u32(values + chunk * 4)
    scale_word = load_u32(scales)
    scale = ue8m0((scale_word >> (chunk // 4 * 8)) & 0xFF)
    for i in cutlass.range_constexpr(8):
        out[i] = (e2m1(codes >> (4 * i)) * scale).to(BFloat16)


class MqaLogits:
    """MXFP4 (or, compact, FP8) indexer logits; see the module docstring."""

    def __init__(
        self,
        paged: bool,
        heads: int,
        candidate_block: int = 0,
        group: int = 1,
        fp8: bool = False,
    ):
        assert heads % 16 == 0 and heads <= MAX_HEADS
        assert group == 1 or (paged and group * heads <= THREADS)
        assert candidate_block == 0 or (
            paged and group == 1 and KEYS % candidate_block == 0
        )
        assert not fp8 or candidate_block > 0
        self.paged = paged
        self.heads = heads
        # Rows of one request a CTA scores together (they share its keys).
        self.group = group
        # Keys per candidate block of compact rows, or 0 for dense rows.
        self.candidate_block = candidate_block
        # FP8 keys and queries in place of MXFP4.
        self.fp8 = fp8

    @cute.jit
    def __call__(
        self,
        q_values: Int64,
        q_scales: Int64,
        weights: cute.Pointer,
        out: cute.Pointer,
        k_values: Int64,
        k_scales: Int64,
        row_starts: Int64,
        row_ends: Int64,
        block_table: Int64,
        candidates: Int64,
        compact_lens: Int64,
        rows: Int32,
        splits: Int32,
        out_stride: Int32,
        block_table_stride: Int32,
        block_size: Int32,
        block_bytes: Int64,
        next_n: Int32,
        candidate_stride: Int32,
        candidate_count: Int32,
        stream: cuda.CUstream,
    ):
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        q_layout = cute.tile_to_shape(atom, (self.group * self.heads, DIM), (0, 1))
        k_layout = cute.tile_to_shape(atom, (KEYS, DIM), (0, 1))

        @cute.struct
        class Storage:
            q: cute.struct.Align[
                cute.struct.MemRange[BFloat16, self.group * self.heads * DIM], 1024
            ]
            k: cute.struct.Align[cute.struct.MemRange[BFloat16, KEYS * DIM], 1024]
            w: cute.struct.MemRange[Float32, self.group * self.heads]
            # A compact tile's key scales (FP8) or ones (MXFP4); 0: no key.
            scale: cute.struct.MemRange[Float32, KEYS]

        # 256 threads as 16 rows x 16 vectors of eight BF16 (one 128-dim row
        # per thread row).
        store = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            ),
            cute.make_layout((16, 16), stride=(16, 1)),
            cute.make_layout((1, 8)),
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (1, WARPS, 1)
        )
        self.kernel(
            q_values,
            q_scales,
            weights,
            out,
            k_values,
            k_scales,
            row_starts,
            row_ends,
            block_table,
            candidates,
            compact_lens,
            out_stride,
            block_table_stride,
            block_size,
            block_bytes,
            next_n,
            candidate_stride,
            candidate_count,
            q_layout,
            k_layout,
            store,
            tiled_mma,
            Storage,
        ).launch(grid=(splits, rows, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        q_values: Int64,
        q_scales: Int64,
        weights: cute.Pointer,
        out: cute.Pointer,
        k_values: Int64,
        k_scales: Int64,
        row_starts: Int64,
        row_ends: Int64,
        block_table: Int64,
        candidates: Int64,
        compact_lens: Int64,
        out_stride: Int32,
        block_table_stride: Int32,
        block_size: Int32,
        block_bytes: Int64,
        next_n: Int32,
        candidate_stride: Int32,
        candidate_count: Int32,
        q_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        store: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, row_group, _ = cute.arch.block_idx()
        splits, _, _ = cute.arch.grid_dim()
        heads = self.heads
        group = self.group
        row = row_group * group

        # Each row's key range [start, end) and its output columns from 0;
        # the rows of a group share their start and keys.
        start = Int32(0)
        if cutlass.const_expr(not self.paged):
            start = load_s32(row_starts + Int64(row) * 4)
        lengths = [
            load_s32(row_ends + Int64(row + r) * 4) - start for r in range(group)
        ]
        length = lengths[0]
        row_lengths = cute.make_rmem_tensor(cute.make_layout(group), Int32)
        for r in cutlass.range_constexpr(group):
            length = cutlass.max(length, lengths[r])
            row_lengths[r] = lengths[r]
        block = self.candidate_block
        positions = length
        if cutlass.const_expr(block > 0):
            # The listed blocks' keys.
            length = cutlass.min(candidate_count, (length + block - 1) // block) * block
            row_lengths[0] = length
            if split == 0 and tid == 0:
                store_u32(compact_lens + Int64(row) * 4, Uint32(length))
        tiles = (length + KEYS - 1) // KEYS
        per_cta = (tiles + splits - 1) // splits
        tile_begin = split * per_cta
        tile_end = cutlass.min(tile_begin + per_cta, tiles)
        if tile_begin < tile_end:
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(Storage)
            sQ = storage.q.get_tensor(q_layout)
            sK = storage.k.get_tensor(k_layout)
            sW = storage.w.get_tensor(cute.make_layout(group * heads))
            sScale = storage.scale.get_tensor(cute.make_layout(KEYS))

            thr_store = store.get_slice(tid)
            chunk = tid % 16
            # ---- the rows' queries, row r's heads at r * heads ----
            tQsQ = thr_store.partition_D(sQ)
            rQ = cute.make_rmem_tensor_like(tQsQ, BFloat16)
            for m in cutlass.range_constexpr(group * heads // 16):
                item = Int64(row) * heads + tid // 16 + 16 * m
                if cutlass.const_expr(self.fp8):
                    _decode_e4m3_chunk(q_values + item * DIM, chunk, rQ[None, m, None])
                else:
                    _decode_mxfp4_chunk(
                        q_values + item * 64,
                        q_scales + item * 4,
                        chunk,
                        rQ[None, m, None],
                    )
            cute.copy(store, rQ, tQsQ)
            if tid < group * heads:
                gW = cute.make_tensor(
                    weights + Int64(row) * heads, cute.make_layout(group * heads)
                )
                sW[tid] = Float32(gW[tid])

            thr_mma = tiled_mma.get_slice(tid)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tScS = thr_mma.partition_C(cute.make_identity_tensor((group * heads, KEYS)))
            copy_a = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
                ),
                tiled_mma,
            )
            copy_b = cute.make_tiled_copy_B(
                cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2), BFloat16
                ),
                tiled_mma,
            )
            thr_copy_a = copy_a.get_slice(tid)
            thr_copy_b = copy_b.get_slice(tid)
            tSsQ = thr_copy_a.partition_S(sQ)
            tSrQ_view = thr_copy_a.retile(tSrQ)
            tSsK = thr_copy_b.partition_S(sK)
            tSrK_view = thr_copy_b.retile(tSrK)

            tKsK = thr_store.partition_D(sK)
            rK = cute.make_rmem_tensor_like(tKsK, BFloat16)
            gOut = cute.make_tensor(
                out + Int64(row) * Int64(out_stride),
                cute.make_layout((group, length), stride=(out_stride, 1)),
            )
            request = row // next_n
            lane = tid % 32
            for tile in cutlass.range(tile_begin, tile_end, unroll=1):
                # ---- decode 64 keys ----
                for m in cutlass.range_constexpr(KEYS // 16):
                    key = tile * KEYS + tid // 16 + 16 * m
                    keys = rK[None, m, None]
                    position = key
                    present = Int32(0)
                    if key < length:
                        present = Int32(1)
                    if cutlass.const_expr(block > 0):
                        listed = Int32(-1)
                        if present == 1:
                            listed = load_s32(
                                candidates
                                + (
                                    Int64(row) * Int64(candidate_stride)
                                    + Int64(key // block)
                                )
                                * 4
                            )
                        position = listed * block + key % block
                        if listed < 0 or position >= positions:
                            present = Int32(0)
                    if present == 1:
                        values = Int64(0)
                        scales = Int64(0)
                        if cutlass.const_expr(self.paged):
                            page = load_s32(
                                block_table
                                + (
                                    Int64(request) * Int64(block_table_stride)
                                    + Int64(position // block_size)
                                )
                                * 4
                            )
                            base = k_values + Int64(page) * block_bytes
                            offset = Int64(position % block_size)
                            if cutlass.const_expr(self.fp8):
                                values = base + offset * DIM
                                scales = base + Int64(block_size) * DIM + offset * 4
                            else:
                                values = base + offset * 64
                                scales = base + Int64(block_size) * 64 + offset * 4
                        else:
                            index = Int64(start + key)
                            values = k_values + index * 64
                            scales = k_scales + index * 4
                        if cutlass.const_expr(self.fp8):
                            _decode_e4m3_chunk(values, chunk, keys)
                        else:
                            _decode_mxfp4_chunk(values, scales, chunk, keys)
                        if cutlass.const_expr(block > 0):  # noqa: SIM102
                            if chunk == 0:
                                scale = Float32(1.0)
                                if cutlass.const_expr(self.fp8):
                                    scale = f32_from_bits(load_u32(scales))
                                sScale[tid // 16 + 16 * m] = scale
                    else:
                        keys.fill(0)
                        if cutlass.const_expr(block > 0):  # noqa: SIM102
                            if chunk == 0:
                                sScale[tid // 16 + 16 * m] = Float32(0.0)
                cute.copy(store, rK, tKsK)
                cute.arch.barrier()

                # ---- S = Q K^T, then sum_h w_h relu(S) ----
                acc = cute.make_rmem_tensor(
                    thr_mma.partition_shape_C((group * heads, KEYS)), Float32
                )
                acc.fill(0.0)
                for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    cute.copy(copy_a, tSsQ[None, None, k], tSrQ_view[None, None, k])
                    cute.copy(copy_b, tSsK[None, None, k], tSrK_view[None, None, k])
                    cute.gemm(
                        tiled_mma,
                        acc,
                        tSrQ[None, None, k],
                        tSrK[None, None, k],
                        acc,
                    )
                # A thread holds rows g, g + 8 of each 16-head block and columns
                # 2t, 2t + 1 of its warp's 8 keys (g = lane / 4, t = lane % 4).
                # Value i is in the 16-head block i // 4 of row
                # (i // 4) * 16 // heads.
                sums = cute.make_rmem_tensor(cute.make_layout(2 * group), Float32)
                sums.fill(0.0)
                for i in cutlass.range_constexpr(cute.size(acc)):
                    head, column = tScS[i]
                    r = (i // 4) * 16 // heads
                    score = cute.arch.fmax(acc[i], Float32(0.0)) * sW[head]
                    sums[2 * r + column % 2] = sums[2 * r + column % 2] + score
                for r in cutlass.range_constexpr(group):
                    for j in cutlass.range_constexpr(2):
                        total = sums[2 * r + j]
                        for offset in cutlass.range_constexpr(3):
                            total = total + cute.arch.shuffle_sync_bfly(
                                total, offset=4 << offset
                            )
                        key = tile * KEYS + (tid // 32) * 8 + (lane % 4) * 2 + j
                        if lane < 4 and key < row_lengths[r]:
                            if cutlass.const_expr(block > 0):
                                scale = sScale[key - tile * KEYS]
                                total = total * scale
                                if scale == Float32(0.0):
                                    total = Float32(float("-inf"))
                            gOut[r, key] = total
                cute.arch.barrier()


def _fake(dtype: type[cutlass.Numeric], align: int):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to `tensor`'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


@cache
def _kernel(
    device_index: int,
    paged: bool,
    heads: int,
    candidate_block: int = 0,
    group: int = 1,
    fp8: bool = False,
):
    """`MqaLogits` compiled for a device, key source, head count, candidate
    block size (0: dense rows), row group and key format."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            MqaLogits(paged, heads, candidate_block, group, fp8),
            Int64(0),
            Int64(0),
            _fake(Float32, 4),
            _fake(Float32, 4),
            *(Int64(0),) * 7,
            *(Int32(1),) * 5,
            Int64(1),
            *(Int32(1),) * 3,
            make_fake_stream(),
        )


@cache
def _decode_splits_target(device_index: int, group: int) -> int:
    """CTAs a decode launch spreads its rows' tiles over."""
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return (DECODE_CTAS_PER_SM if group == 1 else GROUP_CTAS_PER_SM) * sms


def _check_query(
    q_values: torch.Tensor, q_scales: torch.Tensor, weights: torch.Tensor
) -> tuple[int, int]:
    """(rows, heads) of contiguous MXFP4 queries `[rows, heads, 64]` with
    scales `[rows, heads, 4]` and FP32 weights `[rows, heads]`."""
    rows, heads = q_values.shape[:2]
    assert q_values.shape[2:] == (64,) and q_values.dtype == torch.uint8
    assert q_scales.shape == (rows, heads, 4) and q_scales.dtype == torch.uint8
    assert q_values.is_contiguous() and q_scales.is_contiguous()
    assert weights.shape == (rows, heads) and weights.dtype == torch.float32
    assert weights.is_contiguous()
    return rows, heads


def mqa_logits(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Ragged logits `[rows, max_len]` (FP32): row `m` scores keys
    `[row_starts[m], row_ends[m])` of `k_values [N, 64]` /
    `k_scales [N, 4]` (uint8, contiguous)."""
    rows, heads = _check_query(q_values, q_scales, weights)
    assert k_values.is_contiguous() and k_scales.is_contiguous()
    assert row_starts.dtype == torch.int32 and row_ends.dtype == torch.int32
    out = torch.empty(
        (rows, max(max_len, 1)), dtype=torch.float32, device=q_values.device
    )
    splits = (max_len + KEYS * TILES_PER_CTA - 1) // (KEYS * TILES_PER_CTA)
    if rows == 0 or splits == 0:
        return out
    _kernel(q_values.device.index, False, heads)(
        Int64(q_values.data_ptr()),
        Int64(q_scales.data_ptr()),
        _pointer(Float32, weights, 4),
        _pointer(Float32, out, 4),
        Int64(k_values.data_ptr()),
        Int64(k_scales.data_ptr()),
        Int64(row_starts.contiguous().data_ptr()),
        Int64(row_ends.contiguous().data_ptr()),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(rows),
        Int32(splits),
        Int32(out.stride(0)),
        Int32(0),
        Int32(1),
        Int64(0),
        Int32(1),
        Int32(0),
        Int32(0),
        current_cu_stream(),
    )
    return out


def paged_mqa_logits(
    q_values: torch.Tensor,
    q_scales: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    lens: torch.Tensor,
    block_table: torch.Tensor,
    next_n: int,
    max_len: int,
    splits: int | None = None,
    group: int = 1,
) -> torch.Tensor:
    """Paged logits `[rows, max_len]` (FP32, rows 16-byte aligned): row
    `m` scores the first `lens[m]` keys of request `m // next_n` in the
    indexer key cache `k_cache` (`[blocks, block_size, ...]` uint8 pages of
    68-byte keys). Each `group` consecutive rows (tokens of one request:
    equal block tables) share a CTA and the keys it decodes. `splits` (CTAs
    per group) defaults to about eight CTAs per SM over all groups."""
    rows, heads = _check_query(q_values, q_scales, weights)
    assert lens.dtype == torch.int32 and lens.numel() == rows
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert k_cache.dtype == torch.uint8 and k_cache.shape[1] * 68 <= k_cache.stride(0)
    out = torch.empty(
        (rows, (max_len + 3) // 4 * 4), dtype=torch.float32, device=q_values.device
    )[:, :max_len]
    if rows == 0 or max_len == 0:
        return out
    assert rows % group == 0 and group * heads <= THREADS
    groups = rows // group
    if splits is None:
        tiles = (max_len + KEYS - 1) // KEYS
        target = _decode_splits_target(q_values.device.index, group)
        splits = min(tiles, max(1, (target + groups - 1) // groups))
    _kernel(q_values.device.index, True, heads, 0, group)(
        Int64(q_values.data_ptr()),
        Int64(q_scales.data_ptr()),
        _pointer(Float32, weights, 4),
        _pointer(Float32, out, 4),
        Int64(k_cache.data_ptr()),
        Int64(0),
        Int64(0),
        Int64(lens.contiguous().data_ptr()),
        Int64(block_table.data_ptr()),
        Int64(0),
        Int64(0),
        Int32(groups),
        Int32(splits),
        Int32(out.stride(0)),
        Int32(block_table.stride(0)),
        Int32(k_cache.shape[1]),
        Int64(k_cache.stride(0)),
        Int32(next_n),
        Int32(0),
        Int32(0),
        current_cu_stream(),
    )
    return out


def candidate_mqa_logits(
    q_values: torch.Tensor,
    q_scales: torch.Tensor | None,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    lens: torch.Tensor,
    block_table: torch.Tensor,
    next_n: int,
    candidates: torch.Tensor,
    candidate_block_size: int,
    splits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact paged logits of the candidate keys only, and the rows' compact
    lengths (int32 `[rows]`). `candidates` (int32 `[rows, count]`) lists
    row `m`'s blocks of `candidate_block_size` keys, in any order and
    `-1` padded, with every valid block within the row's first
    `min(count, ceil(lens[m] / candidate_block_size))` slots (the slots
    scored). Column `c` of row `m` (FP32
    `[rows, count * candidate_block_size]`, rows 16-byte aligned) scores key
    `candidate_block_size * candidates[m, c // candidate_block_size] +
    c % candidate_block_size` of request `m // next_n` as
    `paged_mqa_logits` does, or `-inf` for a `-1` slot or at or past
    `lens[m]`; the compact length counts the scored slots' keys, and later
    columns are not written.

    The queries are MXFP4 as in `paged_mqa_logits`, or with `q_scales`
    `None` FP8 (`[rows, heads, 128]`, their scales in `weights`) against
    the FP8 cache (`[blocks, block_size, ...]` uint8 pages of 128-byte keys,
    then their FP32 scales). `splits` (CTAs per row) defaults to about eight
    CTAs per SM over all rows."""
    fp8 = q_scales is None
    if q_scales is None:
        rows, heads = q_values.shape[:2]
        assert q_values.shape[2:] == (DIM,) and q_values.element_size() == 1
        assert q_values.is_contiguous() and heads % 16 == 0 and heads <= MAX_HEADS
        assert weights.shape == (rows, heads) and weights.dtype == torch.float32
        assert weights.is_contiguous()
    else:
        rows, heads = _check_query(q_values, q_scales, weights)
    record = DIM + 4 if fp8 else 68
    assert lens.dtype == torch.int32 and lens.numel() == rows
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert k_cache.dtype == torch.uint8
    assert k_cache.shape[1] * record <= k_cache.stride(0)
    assert candidates.dtype == torch.int32 and candidates.stride(1) == 1
    assert candidates.shape[0] >= rows and KEYS % candidate_block_size == 0
    width = candidates.shape[1] * candidate_block_size
    out = torch.empty(
        (rows, (width + 3) // 4 * 4), dtype=torch.float32, device=q_values.device
    )[:, :width]
    compact_lens = torch.empty(rows, dtype=torch.int32, device=q_values.device)
    if rows == 0:
        return out, compact_lens
    if splits is None:
        tiles = (width + KEYS - 1) // KEYS
        target = _decode_splits_target(q_values.device.index, 1)
        splits = min(tiles, max(1, (target + rows - 1) // rows))
    _kernel(q_values.device.index, True, heads, candidate_block_size, 1, fp8)(
        Int64(q_values.data_ptr()),
        Int64(q_scales.data_ptr() if q_scales is not None else 0),
        _pointer(Float32, weights, 4),
        _pointer(Float32, out, 4),
        Int64(k_cache.data_ptr()),
        Int64(0),
        Int64(0),
        Int64(lens.contiguous().data_ptr()),
        Int64(block_table.data_ptr()),
        Int64(candidates.data_ptr()),
        Int64(compact_lens.data_ptr()),
        Int32(rows),
        Int32(splits),
        Int32(out.stride(0)),
        Int32(block_table.stride(0)),
        Int32(k_cache.shape[1]),
        Int64(k_cache.stride(0)),
        Int32(next_n),
        Int32(candidates.stride(0)),
        Int32(candidates.shape[1]),
        current_cu_stream(),
    )
    return out, compact_lens
