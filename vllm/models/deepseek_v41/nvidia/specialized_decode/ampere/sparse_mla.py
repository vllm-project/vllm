# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 sparse MLA on SM80 in CuTe DSL.

Each query token attends with all its heads to its own list of keys: the
window keys (528-byte MXFP8 records named by SWA slots) followed by the
compressed keys its indexer selected (288-byte NVFP4 records named by
compressed-cache slots). Keys double as values. Per head the result is

    softmax([q . k * scale for each key] + [sink]) applied to the keys,

the sink adding `exp(sink)` to the denominator only, then the inverse RoPE
on dims 448-511 at the token's position, rounded to BF16 (`wo_a`'s input).

Two kernels compute it:

* `SparseMlaPartial` (grid `(splits, tokens, head groups)`, 256 threads)
  walks the token's key list in 64-key chunks; CTA `split` takes chunks
  `[split * chunks_per_split, ...)`. Per chunk it decodes the 64 records
  into a BF16 tile in shared memory (codecs.py: two codes at a time, exact),
  multiplies `S^T = K Q^T` for its 8 or 16 heads on `mma.sync` BF16
  tensor cores (`ldmatrix` operands, FP32 accumulation), takes the online
  softmax of each head, and accumulates `O^T += V^T P^T` with `P` rounded
  to BF16. It stores its FP32 output and the heads' running (max, sum) to a
  workspace.
* `SparseMlaMerge` (grid `(tokens, heads)`, 128 threads) combines the
  splits with the sink, divides, rounds to BF16, inverse-RoPEs and rounds.
  With one split per token the partial kernel finishes its rows itself with
  the same arithmetic (`_finish_pair`), and neither the FP32 workspace nor
  the merge is needed.

A token without keys outputs zeros. Scores are scaled in base 2
(`scale * log2(e)`) and exponentiated with `exp2`.

A variant (`v4_records`) reads DeepSeek's V4 records from both caches
instead, as SM90 stores them: 448 E4M3 dims with one UE8M0 scale per 64 and
64 BF16 RoPE dims. A page holds its tokens' 576 value bytes, then their 8
scale bytes (7 used); the cache tensor names each token's 584 bytes. The
RoPE dims are already BF16 and pass through; the others decode as above.
Its callers apply the inverse RoPE in their output projection, so the
variant's output keeps dims 448-511 rotated (`inverse_rope=False`).

Gathering a chunk is a chain of dependent global loads (a key's slot, then
its record), so the partial kernel keeps every load of a chunk in flight at
once: the 64 slots of chunk `c + 1` are loaded (one per thread of the first
64) while chunk `c` is decoded, turned into record addresses and staged in
shared memory; each thread then issues all the record words of its 16 keys
before decoding any. A chunk thus waits for about one memory round trip.

Tensor parallelism leaves 8 heads per GPU at TP8. The MMAs therefore take
keys and dims as their M (16 per tile) and heads as their N (8 per tile),
so no tile row is padding: `S^T` runs on two groups of 4 warps, one per
half of the 512 dims (16 keys per warp, the halves added in the softmax), and
`O^T` on 8 warps of 64 dims. A CTA holds 8 heads at TP8 (two CTAs fit an
SM and overlap each other's waits), 16 otherwise. `default_splits` spreads
each token's chunks over as many CTAs as fill one wave of resident CTAs: one
per chunk for a few tokens, one per token for a wide batch or prefill.
"""

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_stream, make_ptr

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from .codecs import (
    e2m1x2_bf16,
    e4m3_bf16x2,
    e4m3x2_bf16,
    issue_s32,
    issue_u32,
    load_s32,
    ue8m0_bf16x2,
)

DIM = 512
ROPE_DIM = 64
HEADS = 16  # query heads of one CTA: the MMA's N
KEYS = 64  # keys of one chunk
THREADS = 256
WARPS = THREADS // 32
SWA_RECORD = 528
NVFP4_RECORD = 288
# The V4 record (module docstring): 584 bytes per token, laid out per page as
# the tokens' 576 value bytes (448 E4M3 dims, then 64 BF16 dims), then their
# 8 scale bytes.
V4_RECORD = 584
V4_VALUES = 576
V4_SCALES = 8
V4_FP8_DIMS = 448
LOG2E = math.log2(math.e)


@cute.jit
def _finish_pair(
    acc_even: Float32,
    acc_odd: Float32,
    top: Float32,
    denominator: Float32,
    dim: Int32,
    cos_sin_row: cute.Tensor,
    out_row: cute.Tensor,
    inverse_rope: cutlass.Constexpr[bool],
):
    """Store output dims `dim, dim + 1` (`dim` even) of one (token, head):
    the accumulators over the softmax denominator rounded to BF16 (zeros when
    the row's maximum `top` is -inf: no keys), with `inverse_rope`
    inverse-RoPE'd on dims 448-511 with the token's cos/sin row and rounded
    to BF16 again."""
    even = Float32(0.0)
    odd = Float32(0.0)
    if top != -Float32.inf:
        even = (acc_even / denominator).to(BFloat16).to(Float32)
        odd = (acc_odd / denominator).to(BFloat16).to(Float32)
    if cutlass.const_expr(inverse_rope):  # noqa: SIM102
        if dim >= DIM - ROPE_DIM:
            pair = (dim - (DIM - ROPE_DIM)) // 2
            c = cos_sin_row[pair]
            s = cos_sin_row[ROPE_DIM // 2 + pair]
            rotated = even * c + odd * s
            odd = odd * c - even * s
            even = rotated
    out_row[dim] = even.to(BFloat16)
    out_row[dim + 1] = odd.to(BFloat16)


@cute.jit
def _cos_sin_row(
    positions: cute.Pointer,
    cos_sin: cute.Pointer,
    row: Int32,
    cos_sin_stride: Int32,
    inverse_rope: cutlass.Constexpr[bool],
) -> cute.Tensor:
    """The RoPE (cos, sin) row of token `row`'s position; without
    `inverse_rope` the table's first row, which nothing reads."""
    offset = Int64(0)
    if cutlass.const_expr(inverse_rope):
        gPosition = cute.make_tensor(positions, cute.make_layout(row + 1))
        offset = gPosition[row] * Int64(cos_sin_stride)
    return cute.make_tensor(cos_sin + offset, cute.make_layout(ROPE_DIM))


class SparseMlaPartial:
    """Partial sparse MLA of (token, split, head group) items; see the module
    docstring. `head_rows` is the number of heads a CTA holds (8 or 16);
    `v4_records` reads V4 records from both caches and `inverse_rope`
    rotates the output's RoPE dims back."""

    def __init__(
        self,
        has_compressed: bool,
        head_rows: int,
        v4_records: bool = False,
        inverse_rope: bool = True,
    ):
        assert head_rows in (8, HEADS)
        self.has_compressed = has_compressed
        self.head_rows = head_rows
        self.v4_records = v4_records
        self.inverse_rope = inverse_rope
        # Per staged key, a V4 record's scale bytes relative to its values.
        self.scale_offsets = 2 * KEYS if v4_records else 1
        # Record words a thread loads per key: three, four for V4's BF16 dims.
        self.words = 4 if v4_records else 3

    @cute.jit
    def __call__(
        self,
        q: cute.Pointer,
        part: cute.Pointer,
        stats: cute.Pointer,
        sink: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        out: cute.Pointer,
        swa_cache: Int64,
        swa_indices: Int64,
        swa_lens: Int64,
        cmp_cache: Int64,
        cmp_indices: Int64,
        cmp_lens: Int64,
        rows: Int32,
        heads: Int32,
        q_row_stride: Int32,
        swa_index_stride: Int32,
        cmp_index_stride: Int32,
        swa_block_size: Int32,
        swa_block_bytes: Int64,
        cmp_block_size: Int32,
        cmp_block_bytes: Int64,
        splits: Int32,
        chunks_per_split: Int32,
        scale_log2: Float32,
        out_row_stride: Int32,
        cos_sin_stride: Int32,
        stream: cuda.CUstream,
    ):
        head_rows = self.head_rows
        scale_offsets = self.scale_offsets
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        q_layout = cute.tile_to_shape(atom, (head_rows, DIM), (0, 1))
        k_layout = cute.tile_to_shape(atom, (KEYS, DIM), (0, 1))
        p_layout = cute.tile_to_shape(atom, (head_rows, KEYS), (0, 1))
        # Two partial score tiles [head, key], one per half of the dims.
        s_layout = cute.make_layout(
            (2, head_rows, KEYS),
            stride=(head_rows * (KEYS + 4), KEYS + 4, 1),
        )

        @cute.struct
        class Storage:
            q: cute.struct.Align[cute.struct.MemRange[BFloat16, head_rows * DIM], 1024]
            k: cute.struct.Align[cute.struct.MemRange[BFloat16, KEYS * DIM], 1024]
            p: cute.struct.Align[cute.struct.MemRange[BFloat16, head_rows * KEYS], 1024]
            s: cute.struct.Align[
                cute.struct.MemRange[Float32, 2 * head_rows * (KEYS + 4)], 16
            ]
            record: cute.struct.Align[cute.struct.MemRange[Int64, 2 * KEYS], 16]
            kind: cute.struct.MemRange[Int32, 2 * KEYS]
            scale_offset: cute.struct.MemRange[Int32, scale_offsets]
            row_max: cute.struct.MemRange[Float32, HEADS]
            row_sum: cute.struct.MemRange[Float32, HEADS]
            factor: cute.struct.MemRange[Float32, HEADS]

        # 256 threads as 4 rows x 64 vectors of eight BF16: one 512-dim row
        # per thread row.
        thread_layout = cute.make_layout((4, 64), stride=(64, 1))
        value_layout = cute.make_layout((1, 8))
        load_q = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                BFloat16,
                num_bits_per_copy=128,
            ),
            thread_layout,
            value_layout,
        )
        store_k = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            ),
            thread_layout,
            value_layout,
        )
        # Keys and dims are the MMAs' M, heads their N: S^T = K Q^T on 4 warps
        # (16 keys each; the two halves of the CTA take the two halves of the
        # dims), O^T = V^T P^T on 8 warps (64 dims each).
        atom_mma = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))
        mma_s = cute.make_tiled_mma(atom_mma, (WARPS // 2, 1, 1))
        mma_o = cute.make_tiled_mma(atom_mma, (WARPS, 1, 1))
        head_groups = (heads + head_rows - 1) // head_rows
        self.kernel(
            q,
            part,
            stats,
            sink,
            positions,
            cos_sin,
            out,
            swa_cache,
            swa_indices,
            swa_lens,
            cmp_cache,
            cmp_indices,
            cmp_lens,
            heads,
            q_row_stride,
            swa_index_stride,
            cmp_index_stride,
            swa_block_size,
            swa_block_bytes,
            cmp_block_size,
            cmp_block_bytes,
            splits,
            chunks_per_split,
            scale_log2,
            out_row_stride,
            cos_sin_stride,
            q_layout,
            k_layout,
            p_layout,
            s_layout,
            load_q,
            store_k,
            mma_s,
            mma_o,
            Storage,
        ).launch(
            grid=(splits, rows, head_groups),
            block=[THREADS, 1, 1],
            # cute.struct adds size_in_bytes to the class at runtime.
            smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=HEADS // head_rows,
        )

    @cute.jit
    def _load_slot(
        self,
        index: Int32,
        row: Int32,
        swa_len: Int32,
        cmp_len: Int32,
        swa_indices: Int64,
        swa_index_stride: Int32,
        cmp_indices: Int64,
        cmp_index_stride: Int32,
    ) -> Int32:
        """Issue the load of key `index` of `row`'s list (window keys,
        then compressed keys): its cache slot, -1 past the list."""
        slot = Int32(-1)
        if index < swa_len:
            slot = issue_s32(
                swa_indices + (Int64(row) * Int64(swa_index_stride) + Int64(index)) * 4
            )
        if cutlass.const_expr(self.has_compressed):  # noqa: SIM102
            if index >= swa_len and index < swa_len + cmp_len:
                slot = issue_s32(
                    cmp_indices
                    + (Int64(row) * Int64(cmp_index_stride) + Int64(index - swa_len))
                    * 4
                )
        return slot

    @cute.jit
    def _stage_key(
        self,
        key: Int32,
        index: Int32,
        slot: Int32,
        swa_len: Int32,
        swa_cache: Int64,
        swa_block_size: Int32,
        swa_block_bytes: Int64,
        cmp_cache: Int64,
        cmp_block_size: Int32,
        cmp_block_bytes: Int64,
        record: cute.Tensor,
        kind: cute.Tensor,
        scale_offset: cute.Tensor,
    ):
        """Store a key's kind (0 none, 1 window, 2 compressed) and record
        address; a key without a record gets the first window record, a valid
        address whose bytes are loaded and then ignored. A V4 record's
        address is that of its values, and `scale_offset` holds its scale
        bytes' offset from there."""
        address = swa_cache
        code = Int32(0)
        offset = Int32(0)
        if slot >= 0:
            if cutlass.const_expr(self.v4_records):
                block_size = swa_block_size
                row = slot % swa_block_size
                address = swa_cache + Int64(slot // swa_block_size) * swa_block_bytes
                code = Int32(1)
                if index >= swa_len:
                    block_size = cmp_block_size
                    row = slot % cmp_block_size
                    address = (
                        cmp_cache + Int64(slot // cmp_block_size) * cmp_block_bytes
                    )
                    code = Int32(2)
                address = address + Int64(row * V4_VALUES)
                offset = block_size * V4_VALUES + row * (V4_SCALES - V4_VALUES)
            else:
                if index < swa_len:
                    address = (
                        swa_cache
                        + Int64(slot // swa_block_size) * swa_block_bytes
                        + Int64(slot % swa_block_size) * SWA_RECORD
                    )
                    code = Int32(1)
                else:
                    address = (
                        cmp_cache
                        + Int64(slot // cmp_block_size) * cmp_block_bytes
                        + Int64(slot % cmp_block_size) * NVFP4_RECORD
                    )
                    code = Int32(2)
        record[key] = address
        kind[key] = code
        if cutlass.const_expr(self.v4_records):
            scale_offset[key] = offset

    @cute.kernel
    def kernel(
        self,
        q_base: cute.Pointer,
        part_base: cute.Pointer,
        stats_base: cute.Pointer,
        sink: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        out: cute.Pointer,
        swa_cache: Int64,
        swa_indices: Int64,
        swa_lens: Int64,
        cmp_cache: Int64,
        cmp_indices: Int64,
        cmp_lens: Int64,
        heads: Int32,
        q_row_stride: Int32,
        swa_index_stride: Int32,
        cmp_index_stride: Int32,
        swa_block_size: Int32,
        swa_block_bytes: Int64,
        cmp_block_size: Int32,
        cmp_block_bytes: Int64,
        splits: Int32,
        chunks_per_split: Int32,
        scale_log2: Float32,
        out_row_stride: Int32,
        cos_sin_stride: Int32,
        q_layout: cute.ComposedLayout,
        k_layout: cute.ComposedLayout,
        p_layout: cute.ComposedLayout,
        s_layout: cute.Layout,
        load_q: cute.TiledCopy,
        store_k: cute.TiledCopy,
        mma_s: cute.TiledMma,
        mma_o: cute.TiledMma,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, row, group = cute.arch.block_idx()
        head_rows = self.head_rows
        head0 = group * head_rows

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sQ = storage.q.get_tensor(q_layout)
        sK = storage.k.get_tensor(k_layout)
        sP = storage.p.get_tensor(p_layout)
        sS = storage.s.get_tensor(s_layout)
        stages = cute.make_layout((2, KEYS), stride=(KEYS, 1))
        sRecord = storage.record.get_tensor(stages)
        sKind = storage.kind.get_tensor(stages)
        sScaleOffset = sKind
        if cutlass.const_expr(self.v4_records):
            sScaleOffset = storage.scale_offset.get_tensor(stages)
        row_max = storage.row_max.get_tensor(cute.make_layout(HEADS))
        row_sum = storage.row_sum.get_tensor(cute.make_layout(HEADS))
        factor = storage.factor.get_tensor(cute.make_layout(HEADS))
        # V^T [DIM, KEYS] over the key tile.
        sVt = cute.composition(sK, cute.make_layout((DIM, KEYS), stride=(KEYS, 1)))

        # ---- the token's queries (head rows past `heads` are zero) ----
        # Rows start on 16-byte boundaries (the wrapper checks the stride).
        q_offset = cute.assume(
            Int64(row) * Int64(q_row_stride) + Int64(head0) * DIM, divby=8
        )
        gQ = cute.make_tensor(
            q_base + q_offset, cute.make_layout((head_rows, DIM), stride=(DIM, 1))
        )
        thr_load_q = load_q.get_slice(tid)
        tQgQ = thr_load_q.partition_S(gQ)
        tQsQ = thr_load_q.partition_D(sQ)
        tQcQ = thr_load_q.partition_S(cute.make_identity_tensor((head_rows, DIM)))
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if head0 + tQcQ[0, m, 0][0] < heads:
                cute.copy(load_q, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        cute.arch.cp_async_commit_group()
        if tid < HEADS:
            row_max[tid] = -Float32.inf
            row_sum[tid] = Float32(0.0)
            factor[tid] = Float32(0.0)

        # ---- MMA partitions ----
        # S^T = K Q^T: CTA half h (128 threads) sums dims [256 h, 256 h + 256).
        half = tid // (THREADS // 2)
        thr_s = mma_s.get_slice(tid % (THREADS // 2))
        tSrK = thr_s.make_fragment_A(thr_s.partition_A(sK))
        tSrQ = thr_s.make_fragment_B(thr_s.partition_B(sQ))
        tScS = thr_s.partition_C(cute.make_identity_tensor((KEYS, head_rows)))
        # O^T = V^T P^T.
        thr_o = mma_o.get_slice(tid)
        tOrVt = thr_o.make_fragment_A(thr_o.partition_A(sVt))
        tOrP = thr_o.make_fragment_B(thr_o.partition_B(sP))
        acc_O = cute.make_rmem_tensor(
            thr_o.partition_shape_C((DIM, head_rows)), Float32
        )
        acc_O.fill(0.0)
        tOcO = thr_o.partition_C(cute.make_identity_tensor((DIM, head_rows)))

        copy_k = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
            ),
            mma_s,
        )
        copy_q = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2), BFloat16
            ),
            mma_s,
        )
        copy_vt = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), BFloat16
            ),
            mma_o,
        )
        copy_p = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2), BFloat16
            ),
            mma_o,
        )
        thr_copy_k = copy_k.get_slice(tid % (THREADS // 2))
        thr_copy_q = copy_q.get_slice(tid % (THREADS // 2))
        thr_copy_vt = copy_vt.get_slice(tid)
        thr_copy_p = copy_p.get_slice(tid)
        tSsK = thr_copy_k.partition_S(sK)
        tSrK_view = thr_copy_k.retile(tSrK)
        tSsQ = thr_copy_q.partition_S(sQ)
        tSrQ_view = thr_copy_q.retile(tSrQ)
        tOsVt = thr_copy_vt.partition_S(sVt)
        tOrVt_view = thr_copy_vt.retile(tOrVt)
        tOsP = thr_copy_p.partition_S(sP)
        tOrP_view = thr_copy_p.retile(tOrP)

        # ---- key decoding: thread (r, c) decodes dims 8c..8c+7 of keys r + 4m
        thr_store_k = store_k.get_slice(tid)
        tKsK = thr_store_k.partition_D(sK)
        rK = cute.make_rmem_tensor_like(tKsK, BFloat16)
        raw = cute.make_rmem_tensor(cute.make_layout((KEYS // 4, self.words)), Uint32)
        dim_chunk = tid % 64
        # A V4 record's dims 8c..8c+7: E4M3 bytes 8c.. (scale byte c // 8)
        # below dim 448, BF16 bytes 448 + 16 (c - 56).. above.
        v4_bf16 = dim_chunk * 8 >= V4_FP8_DIMS
        v4_value_offset = dim_chunk * 8
        if v4_bf16:
            v4_value_offset = V4_FP8_DIMS + (dim_chunk * 8 - V4_FP8_DIMS) * 2
        swa_len = load_s32(swa_lens + Int64(row) * 4)
        cmp_len = Int32(0)
        if cutlass.const_expr(self.has_compressed):
            cmp_len = load_s32(cmp_lens + Int64(row) * 4)
        chunks = (swa_len + cmp_len + KEYS - 1) // KEYS
        chunk_begin = split * chunks_per_split
        chunk_end = cutlass.min(chunk_begin + chunks_per_split, chunks)

        # The first chunk's key records: thread k < 64 stages key k.
        if tid < KEYS and chunk_begin < chunk_end:
            index = chunk_begin * KEYS + tid
            slot = self._load_slot(
                index,
                row,
                swa_len,
                cmp_len,
                swa_indices,
                swa_index_stride,
                cmp_indices,
                cmp_index_stride,
            )
            self._stage_key(
                tid,
                index,
                slot,
                swa_len,
                swa_cache,
                swa_block_size,
                swa_block_bytes,
                cmp_cache,
                cmp_block_size,
                cmp_block_bytes,
                sRecord[0, None],
                sKind[0, None],
                sScaleOffset[0, None],
            )
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        warp_idx = tid // 32
        lane = tid % 32
        for chunk in cutlass.range(chunk_begin, chunk_end, unroll=1):
            stage = (chunk - chunk_begin) % 2
            # The next chunk's slot loads fly while this chunk's records do.
            next_index = (chunk + 1) * KEYS + tid
            next_slot = Int32(-1)
            if tid < KEYS and chunk + 1 < chunk_end:
                next_slot = self._load_slot(
                    next_index,
                    row,
                    swa_len,
                    cmp_len,
                    swa_indices,
                    swa_index_stride,
                    cmp_indices,
                    cmp_index_stride,
                )

            # ---- all record words of the thread's 16 keys, then decoding ----
            for m in cutlass.range_constexpr(KEYS // 4):
                key = tid // 64 + 4 * m
                record = sRecord[stage, key]
                if cutlass.const_expr(self.v4_records):
                    first = record + Int64(v4_value_offset)
                    third = first + 8
                    if not v4_bf16:
                        third = (
                            record
                            + Int64(sScaleOffset[stage, key])
                            + Int64(dim_chunk // 32 * 4)
                        )
                    raw[m, 0] = issue_u32(first)
                    raw[m, 1] = issue_u32(first + 4)
                    raw[m, 2] = issue_u32(third)
                    if v4_bf16:
                        raw[m, 3] = issue_u32(first + 12)
                else:
                    first = record + Int64(dim_chunk * 8)
                    second = first + 4
                    scales = record + Int64(DIM + dim_chunk // 16 * 4)
                    if cutlass.const_expr(self.has_compressed):  # noqa: SIM102
                        if sKind[stage, key] == 2:
                            first = record + Int64(dim_chunk * 4)
                            second = first
                            scales = record + Int64(DIM // 2 + dim_chunk // 8 * 4)
                    raw[m, 0] = issue_u32(first)
                    raw[m, 1] = issue_u32(second)
                    raw[m, 2] = issue_u32(scales)
            for m in cutlass.range_constexpr(KEYS // 4):
                key = tid // 64 + 4 * m
                kind = sKind[stage, key]
                # The key's 8 dims as BF16 pairs (codecs.py: exact).
                values = cute.recast_tensor(rK[None, m, None], Uint32)
                if cutlass.const_expr(self.v4_records):
                    if kind == 0:
                        values.fill(0)
                    elif v4_bf16:
                        for i in cutlass.range_constexpr(4):
                            values[i] = raw[m, i]
                    else:
                        scale_byte = dim_chunk // 8
                        scale = ue8m0_bf16x2((raw[m, 2] >> (scale_byte % 4 * 8)) & 0xFF)
                        for i in cutlass.range_constexpr(2):
                            values[i] = e4m3x2_bf16(raw[m, 0], i, scale)
                            values[2 + i] = e4m3x2_bf16(raw[m, 1], i, scale)
                elif kind == 1:
                    scale_byte = dim_chunk // 4
                    scale = ue8m0_bf16x2((raw[m, 2] >> (scale_byte % 4 * 8)) & 0xFF)
                    for i in cutlass.range_constexpr(2):
                        values[i] = e4m3x2_bf16(raw[m, 0], i, scale)
                        values[2 + i] = e4m3x2_bf16(raw[m, 1], i, scale)
                elif kind == 2:
                    scale_byte = dim_chunk // 2
                    scale = e4m3_bf16x2((raw[m, 2] >> (scale_byte % 4 * 8)) & 0xFF)
                    for i in cutlass.range_constexpr(4):
                        values[i] = e2m1x2_bf16(raw[m, 0], i, scale)
                else:
                    values.fill(0)
                cute.copy(store_k, rK[None, m, None], tKsK[None, m, None])
            if tid < KEYS and chunk + 1 < chunk_end:
                self._stage_key(
                    tid,
                    next_index,
                    next_slot,
                    swa_len,
                    swa_cache,
                    swa_block_size,
                    swa_block_bytes,
                    cmp_cache,
                    cmp_block_size,
                    cmp_block_bytes,
                    sRecord[1 - stage, None],
                    sKind[1 - stage, None],
                    sScaleOffset[1 - stage, None],
                )
            cute.arch.barrier()

            # ---- S^T = K Q^T, a half of the dims per CTA half ----
            acc_S = cute.make_rmem_tensor(
                thr_s.partition_shape_C((KEYS, head_rows)), Float32
            )
            acc_S.fill(0.0)
            k_steps = cute.size(tSsK.shape[2]) // 2
            for kk in cutlass.range_constexpr(k_steps):
                k = half * k_steps + kk
                cute.copy(copy_k, tSsK[None, None, k], tSrK_view[None, None, kk])
                cute.copy(copy_q, tSsQ[None, None, k], tSrQ_view[None, None, kk])
                cute.gemm(
                    mma_s, acc_S, tSrK[None, None, kk], tSrQ[None, None, kk], acc_S
                )
            for i in cutlass.range_constexpr(cute.size(acc_S)):
                key, head = tScS[i]
                sS[half, head, key] = acc_S[i]
            cute.arch.barrier()

            # ---- online softmax: warp w owns the head rows w * rows_per_warp + r
            rows_per_warp = head_rows // WARPS
            for r in cutlass.range_constexpr(rows_per_warp):
                head = warp_idx * rows_per_warp + r
                s0 = -Float32.inf
                s1 = -Float32.inf
                if sKind[stage, lane] != 0:
                    s0 = (sS[0, head, lane] + sS[1, head, lane]) * scale_log2
                if sKind[stage, lane + 32] != 0:
                    s1 = (sS[0, head, lane + 32] + sS[1, head, lane + 32]) * scale_log2
                top = cute.arch.fmax(s0, s1)
                for offset in cutlass.range_constexpr(5):
                    top = cute.arch.fmax(
                        top, cute.arch.shuffle_sync_bfly(top, offset=1 << offset)
                    )
                previous = row_max[head]
                current = cute.arch.fmax(previous, top)
                base = current
                if current == -Float32.inf:
                    base = Float32(0.0)
                p0 = cute.math.exp2(s0 - base, fastmath=True)
                p1 = cute.math.exp2(s1 - base, fastmath=True)
                sP[head, lane] = p0.to(BFloat16)
                sP[head, lane + 32] = p1.to(BFloat16)
                total = p0 + p1
                for offset in cutlass.range_constexpr(5):
                    total = total + cute.arch.shuffle_sync_bfly(
                        total, offset=1 << offset
                    )
                rescale = cute.math.exp2(previous - base, fastmath=True)
                cute.arch.sync_warp()
                if lane == 0:
                    row_max[head] = current
                    row_sum[head] = row_sum[head] * rescale + total
                    factor[head] = rescale
            cute.arch.barrier()

            # ---- O^T = O^T * factor + V^T P^T ----
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                acc_O[i] = acc_O[i] * factor[tOcO[i][1]]
            for k in cutlass.range_constexpr(cute.size(tOsP.shape[2])):
                cute.copy(copy_vt, tOsVt[None, None, k], tOrVt_view[None, None, k])
                cute.copy(copy_p, tOsP[None, None, k], tOrP_view[None, None, k])
                cute.gemm(
                    mma_o, acc_O, tOrVt[None, None, k], tOrP[None, None, k], acc_O
                )
            cute.arch.barrier()

        # ---- O^T to shared memory as [head, dim] (the key tile is free) ----
        sO = storage.k.get_tensor(
            cute.make_layout((head_rows, DIM), stride=(DIM + 4, 1)), dtype=Float32
        )
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            dim, head = tOcO[i]
            sO[head, dim] = acc_O[i]
        cute.arch.barrier()
        # Thread (head, l) takes the dim pairs 2 l + 2 lanes j of its head.
        lanes = THREADS // head_rows
        out_head = tid // lanes
        first_dim = (tid % lanes) * 2
        if splits == 1:
            # ---- the only split: finish the rows as SparseMlaMerge would ----
            if tid < head_rows and head0 + tid < heads:
                gSink = cute.make_tensor(sink, cute.make_layout(heads))
                top = row_max[tid]
                denominator = Float32(0.0)
                if top != -Float32.inf:
                    denominator = row_sum[tid] + cute.math.exp2(
                        gSink[head0 + tid] * LOG2E - top, fastmath=True
                    )
                factor[tid] = denominator
            cute.arch.barrier()
            if head0 + out_head < heads:
                gCosSin = _cos_sin_row(
                    positions, cos_sin, row, cos_sin_stride, self.inverse_rope
                )
                gOut = cute.make_tensor(
                    out
                    + Int64(row) * Int64(out_row_stride)
                    + Int64(head0 + out_head) * DIM,
                    cute.make_layout(DIM),
                )
                for j in cutlass.range_constexpr(DIM // (2 * lanes)):
                    dim = first_dim + 2 * lanes * j
                    _finish_pair(
                        sO[out_head, dim],
                        sO[out_head, dim + 1],
                        row_max[out_head],
                        factor[out_head],
                        dim,
                        gCosSin,
                        gOut,
                        self.inverse_rope,
                    )
        else:
            # ---- the item's partial output and row statistics ----
            item = (Int64(row) * Int64(splits) + Int64(split)) * Int64(heads)
            if head0 + out_head < heads:
                gPart = cute.make_tensor(
                    part_base + (item + Int64(head0 + out_head)) * DIM,
                    cute.make_layout(DIM),
                )
                for j in cutlass.range_constexpr(DIM // (2 * lanes)):
                    dim = first_dim + 2 * lanes * j
                    gPart[dim] = sO[out_head, dim]
                    gPart[dim + 1] = sO[out_head, dim + 1]
            if tid < head_rows and head0 + tid < heads:
                gStats = cute.make_tensor(
                    stats_base + (item + Int64(head0 + tid)) * 2, cute.make_layout(2)
                )
                gStats[0] = row_max[tid]
                gStats[1] = row_sum[tid]


class SparseMlaMerge:
    """Combine the splits of each (token, head), add the sink, normalize and
    (with `inverse_rope`) inverse-RoPE; see the module docstring."""

    def __init__(self, inverse_rope: bool = True):
        self.inverse_rope = inverse_rope

    @cute.jit
    def __call__(
        self,
        part: cute.Pointer,
        stats: cute.Pointer,
        sink: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        heads: Int32,
        splits: Int32,
        out_row_stride: Int32,
        cos_sin_stride: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            part,
            stats,
            sink,
            positions,
            cos_sin,
            out,
            heads,
            splits,
            out_row_stride,
            cos_sin_stride,
        ).launch(grid=(rows, heads, 1), block=[DIM // 4, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        part: cute.Pointer,
        stats: cute.Pointer,
        sink: cute.Pointer,
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        out: cute.Pointer,
        heads: Int32,
        splits: Int32,
        out_row_stride: Int32,
        cos_sin_stride: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, head, _ = cute.arch.block_idx()
        dim = tid * 4
        first = (Int64(row) * Int64(splits)) * Int64(heads) + Int64(head)
        gStats = cute.make_tensor(
            stats + first * 2, cute.make_layout((2, splits), stride=(1, heads * 2))
        )
        gPart = cute.make_tensor(
            part + first * DIM + dim,
            cute.make_layout((4, splits), stride=(1, heads * DIM)),
        )
        top = -Float32.inf
        for s in cutlass.range(splits, unroll=4):
            top = cute.arch.fmax(top, gStats[0, s])
        acc = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        acc.fill(0.0)
        denominator = Float32(0.0)
        if top != -Float32.inf:
            for s in cutlass.range(splits, unroll=4):
                weight = cute.math.exp2(gStats[0, s] - top, fastmath=True)
                denominator = denominator + gStats[1, s] * weight
                for i in cutlass.range_constexpr(4):
                    acc[i] = acc[i] + gPart[i, s] * weight
            gSink = cute.make_tensor(sink, cute.make_layout(heads))
            denominator = denominator + cute.math.exp2(
                gSink[head] * LOG2E - top, fastmath=True
            )
        gCosSin = _cos_sin_row(
            positions, cos_sin, row, cos_sin_stride, self.inverse_rope
        )
        gOut = cute.make_tensor(
            out + Int64(row) * Int64(out_row_stride) + Int64(head) * DIM,
            cute.make_layout(DIM),
        )
        for i in cutlass.range_constexpr(2):
            _finish_pair(
                acc[2 * i],
                acc[2 * i + 1],
                top,
                denominator,
                dim + 2 * i,
                gCosSin,
                gOut,
                self.inverse_rope,
            )


def _fake(dtype: type[cutlass.Numeric], align: int):
    """A placeholder pointer for compilation."""
    return make_ptr(dtype, align, cute.AddressSpace.gmem, assumed_align=align)


def _pointer(dtype: type[cutlass.Numeric], tensor: torch.Tensor, align: int):
    """A CuTe DSL global-memory pointer to `tensor`'s data."""
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


@cache
def _partial_kernel(
    device_index: int,
    has_compressed: bool,
    head_rows: int,
    v4_records: bool,
    inverse_rope: bool,
):
    """`SparseMlaPartial` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SparseMlaPartial(has_compressed, head_rows, v4_records, inverse_rope),
            _fake(BFloat16, 16),
            _fake(Float32, 16),
            _fake(Float32, 8),
            _fake(Float32, 4),
            _fake(cutlass.Int64, 8),
            _fake(Float32, 4),
            _fake(BFloat16, 4),
            *(Int64(0),) * 6,
            *(Int32(1),) * 6,
            Int64(1),
            Int32(1),
            Int64(1),
            Int32(1),
            Int32(1),
            Float32(1.0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


@cache
def _merge_kernel(device_index: int, inverse_rope: bool):
    """`SparseMlaMerge` compiled for a device."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            SparseMlaMerge(inverse_rope),
            _fake(Float32, 16),
            _fake(Float32, 8),
            _fake(Float32, 4),
            _fake(cutlass.Int64, 8),
            _fake(Float32, 4),
            _fake(BFloat16, 8),
            *(Int32(1),) * 5,
            make_fake_stream(),
        )


def _block_bytes(cache: torch.Tensor, record: int) -> tuple[int, int]:
    """(block size in records, block stride in bytes) of a paged record cache
    `[blocks, block_size, record]` (uint8)."""
    assert cache.dtype == torch.uint8 and cache.shape[-1] == record
    assert cache.stride(1) == record and cache.stride(2) == 1
    return cache.shape[1], cache.stride(0)


@cache
def _multiprocessors(device_index: int) -> int:
    """The SM count of a device."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def default_splits(
    items: int, chunks: int, head_rows: int, device: torch.device
) -> int:
    """CTAs per (token, head group) item: as many as fill one wave of resident
    CTAs (two per SM with 8-head CTAs, else one), at most one per chunk."""
    resident = _multiprocessors(device.index) * (HEADS // head_rows)
    return max(1, min(chunks, resident // items))


def sparse_mla(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    compressed_cache: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    compressed_lens: torch.Tensor | None,
    sink: torch.Tensor,
    scale: float,
    positions: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    out: torch.Tensor,
    splits: int | None = None,
    v4_records: bool = False,
) -> None:
    """Sparse MLA of `q [T, H, 512]` (BF16) into `out [T, H, 512]`, with
    the arguments of `reference.sparse_mla`.

    `splits` CTAs share one token's key list (default:
    `default_splits`). `v4_records` reads both caches as 584-byte V4
    records; without `positions` and `cos_sin_cache` the output's RoPE
    dims stay rotated.
    """
    tokens, heads, dim = q.shape
    if tokens == 0:
        return
    assert dim == DIM and q.dtype == torch.bfloat16 and q.stride(2) == 1
    assert q.stride(1) == DIM and out.stride(1) == DIM and out.stride(2) == 1
    assert q.stride(0) % 8 == 0 and q.data_ptr() % 16 == 0
    assert sink.dtype == torch.float32 and sink.is_contiguous()
    inverse_rope = cos_sin_cache is not None
    if positions is None or cos_sin_cache is None:
        assert positions is None and cos_sin_cache is None
        # Never read: any valid pointers.
        positions_ptr = _pointer(cutlass.Int64, q, 8)
        cos_sin_ptr = _pointer(Float32, q, 4)
        cos_sin_stride = 0
    else:
        assert positions.dtype == torch.int64 and positions.is_contiguous()
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.stride(1) == 1
        positions_ptr = _pointer(cutlass.Int64, positions, 8)
        cos_sin_ptr = _pointer(Float32, cos_sin_cache, 4)
        cos_sin_stride = cos_sin_cache.stride(0)
    assert swa_indices.dtype == torch.int32 and swa_indices.stride(1) == 1
    assert swa_lens.dtype == torch.int32 and swa_lens.is_contiguous()
    swa_record = V4_RECORD if v4_records else SWA_RECORD
    cmp_record = V4_RECORD if v4_records else NVFP4_RECORD
    has_compressed = compressed_cache is not None
    max_keys = swa_indices.shape[1]
    cmp_block_size, cmp_block_bytes = 1, 0
    cmp_cache_ptr = cmp_indices_ptr = cmp_lens_ptr = cmp_indices_stride = 0
    if compressed_cache is not None:
        assert compressed_indices is not None and compressed_lens is not None
        assert compressed_indices.dtype == torch.int32
        assert compressed_indices.stride(1) == 1
        assert compressed_lens.dtype == torch.int32
        assert compressed_lens.is_contiguous()
        max_keys += compressed_indices.shape[1]
        cmp_block_size, cmp_block_bytes = _block_bytes(compressed_cache, cmp_record)
        cmp_cache_ptr = compressed_cache.data_ptr()
        cmp_indices_ptr = compressed_indices.data_ptr()
        cmp_lens_ptr = compressed_lens.data_ptr()
        cmp_indices_stride = compressed_indices.stride(0)
    chunks = (max_keys + KEYS - 1) // KEYS
    head_rows = 8 if heads <= 8 else HEADS
    head_groups = (heads + head_rows - 1) // head_rows
    if splits is None:
        splits = default_splits(tokens * head_groups, chunks, head_rows, q.device)
    chunks_per_split = (chunks + splits - 1) // splits
    splits = (chunks + chunks_per_split - 1) // chunks_per_split
    swa_block_size, swa_block_bytes = _block_bytes(swa_cache, swa_record)
    # The kernel loads records as aligned 32-bit words.
    for cache_ptr, block_bytes in (
        (swa_cache.data_ptr(), swa_block_bytes),
        (cmp_cache_ptr, cmp_block_bytes),
    ):
        assert cache_ptr % 4 == 0 and block_bytes % 4 == 0
    # One split finishes its rows itself: no workspace and no merge.
    work = 0 if splits == 1 else tokens * splits * heads
    part = torch.empty((work, DIM), dtype=torch.float32, device=q.device)
    stats = torch.empty((work, 2), dtype=torch.float32, device=q.device)
    device = q.device.index
    stream = current_cu_stream()
    _partial_kernel(device, has_compressed, head_rows, v4_records, inverse_rope)(
        _pointer(BFloat16, q, 16),
        _pointer(Float32, part, 16),
        _pointer(Float32, stats, 8),
        _pointer(Float32, sink, 4),
        positions_ptr,
        cos_sin_ptr,
        _pointer(BFloat16, out, 4),
        Int64(swa_cache.data_ptr()),
        Int64(swa_indices.data_ptr()),
        Int64(swa_lens.data_ptr()),
        Int64(cmp_cache_ptr),
        Int64(cmp_indices_ptr),
        Int64(cmp_lens_ptr),
        Int32(tokens),
        Int32(heads),
        Int32(q.stride(0)),
        Int32(swa_indices.stride(0)),
        Int32(cmp_indices_stride),
        Int32(swa_block_size),
        Int64(swa_block_bytes),
        Int32(cmp_block_size),
        Int64(cmp_block_bytes),
        Int32(splits),
        Int32(chunks_per_split),
        Float32(scale * LOG2E),
        Int32(out.stride(0)),
        Int32(cos_sin_stride),
        stream,
    )
    if splits == 1:
        return
    _merge_kernel(device, inverse_rope)(
        _pointer(Float32, part, 16),
        _pointer(Float32, stats, 8),
        _pointer(Float32, sink, 4),
        positions_ptr,
        cos_sin_ptr,
        _pointer(BFloat16, out, 8),
        Int32(tokens),
        Int32(heads),
        Int32(splits),
        Int32(out.stride(0)),
        Int32(cos_sin_stride),
        stream,
    )
