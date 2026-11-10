# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed-expert front of a Qwen4Exp MoE layer in one CuTe DSL kernel.

Stock path. A Qwen3.8-Flash-Next MoE layer routes each token row to 10 of 512
experts (softmax over the router logits, top-10, the ten probabilities
renormalized to sum 1). At TP1 the stock decode runs, after the router GEMM:

1. vLLM's NVFP4 quantization of the layer input `x` (BF16 `[rows, 2560]`)
   with the layer's global input scale: per 16 values one E4M3 scale and
   eight bytes of E2M1 pairs (`cvt_fp16_to_fp4_sf_major`);
2. inside FlashInfer's TRT-LLM MoE, the routing kernels (one or two
   launches): the top-10 and its weights, then the routing metadata the
   expert GEMMs read (rows permuted expert by expert, each expert's rows
   padded to the GEMM's 8-row token tile, per tile its expert and row
   limit);
3. the expert GEMMs and the finalize.

`RoutedFront` replaces 1 and 2 with one launch whose outputs equal
theirs bit for bit, so the expert GEMMs can run on them unchanged
(FlashInfer's routing-metadata entry point, `runtime/moe.py`).

Bitwise contract. FlashInfer routes the model's softmax, top-10 and
renormalization (`RenormalizeNaive`) as its equivalent `Renormalize`,
whatever kernel the row count selects: the top-10 of the raw BF16 logits by
the key (logit order bits, then the lower expert id), then a softmax over
those ten in one warp, lane `k` holding the `k`-th pick: their maximum,
`exp` of each difference as `__expf`, the butterfly sum over the warp
(zero above lane 9) and `div.approx(e, sum)`, rounded to BF16. A softmax
over all 512 experts followed by the ten probabilities' renormalization is
the same function, but not the same rounding: about one weight in 10^5
then lands on the other side of a BF16 rounding boundary, so this kernel
repeats the stock arithmetic step by step.
The quantization repeats vLLM's arithmetic: amax, `scale = global *
(amax * rcp(6))` to E4M3, values times `rcp(scale * rcp(global))`,
`cvt.rn.satfinite` to E2M1. `_ptx.py` names each instruction.

Routing metadata. Experts in ascending id order, each one's rows in
ascending token order and padded with token -1 to a multiple of eight rows;
per 8-row tile its expert and its row limit (the segment end, at most the
tile end); the padded row count and the tile count. A row's rank among its
expert's rows comes from a bitmask of the rows that chose the expert (one
32-bit word per 32 rows) by population counts, and the experts' first tiles
from a scan over the 512 experts' tile counts. The order of rows within an
expert's segment may differ from FlashInfer's (its cluster kernel orders by
arrival); the GEMMs compute every row independently, so their outputs do
not depend on it.

Work split (1024 threads a CTA). The first four CTAs form a thread block
cluster and select: row `r` belongs to CTA `r % 4`, warp `r // 4`.
Each selecting warp sends its row's ten expert ids to CTA 0's shared memory,
then the cluster barrier; CTA 0 builds the metadata. The other CTAs
quantize, one thread per 16-value block.

Row cap. The kernel serves calls of up to `MAX_ROWS` (32) rows. One warp
selects one row and one CTA builds the metadata, so the kernel's time grows
with the rows while the stock quantization and routing stay nearly flat:
measured on GB300 against the stock chain (front, expert GEMMs and
finalize, per layer), it saves 2.1 and 4.7 us at 16 and 32 rows, 1.6 us at
64 rows, and is 6.6 us slower at 128 rows.

PDL. The kernel triggers its dependents on entry when `early_trigger`
(the expert GEMMs read its outputs only after their own grid-dependency
wait), and reads the router logits and `x` after its own wait, which
returns once the previous kernel (the router GEMM) has completed; that
kernel in turn completed only after its own wait, so `x` is complete too.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ._ptx import (
    add_ftz,
    bits_f32,
    clear_if_equal_u64,
    div_ftz,
    e2m1x8,
    e4m3_scale,
    exp_ftz,
    f32_bits,
    load_bf16,
    load_bf16x8,
    load_f32,
    load_shared_u32,
    max_ftz,
    max_u64,
    mul_rn,
    nanosleep,
    or_shared_u32,
    ordered_f32,
    rcp_ftz,
    store_bf16,
    store_cluster_u32,
    store_shared_u32,
    store_u8,
    store_u32,
    store_u32x2,
    sub_ftz,
    unordered_f32,
    warp_max_u32,
)

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
TILE = 8  # the expert GEMMs' token tile
MAX_ROWS = 32
SF_VECTOR = 16  # values per NVFP4 scale
BLOCKS = HIDDEN // SF_VECTOR  # scales per row
PER_LANE = EXPERTS // 32
THREADS = 1024
WARPS = THREADS // 32
CLUSTER = 4  # the selecting CTAs
MASK_WORDS = MAX_ROWS // 32  # one bit per row and expert
assert MAX_ROWS <= CLUSTER * WARPS


def max_tiles(rows: int) -> int:
    """The most 8-row tiles any routing of `rows` rows needs."""
    slots = rows * TOP_K
    filled = min(slots, EXPERTS)
    return filled + (slots - filled) // TILE


def quant_ctas(rows: int) -> int:
    """Quantizing CTAs for `rows` rows, whole clusters."""
    ctas = (rows * BLOCKS + THREADS - 1) // THREADS
    return (ctas + CLUSTER - 1) // CLUSTER * CLUSTER


def _warp_sum_bfly(value: Float32) -> Float32:
    """`cg::reduce(plus)` under fast math: the butterfly sum over the
    warp, the same bits in every lane."""
    for offset in (16, 8, 4, 2, 1):
        value = add_ftz(value, cute.arch.shuffle_sync_bfly(value, offset))
    return value


class RoutedFront:
    """Top-10 routing, routing metadata and NVFP4 quantization of a MoE
    layer's input (module docstring).

    Args:
        early_trigger: Let the next kernel launch at entry.
        stall_us: Tests only: every CTA sleeps this long after its
            grid-dependency wait, before its first global access. With
            `early_trigger` the dependent kernel has launched by then, so
            one that read the outputs before its own wait would read stale
            values; without, it launches only after the sleep.

    """

    def __init__(self, early_trigger: bool, stall_us: int = 0) -> None:
        self.early_trigger = early_trigger
        self.stall_us = stall_us
        words = MAX_ROWS * TOP_K + EXPERTS * MASK_WORDS + EXPERTS + WARPS
        self.shared_bytes = words * 4 + 64

    @cute.jit
    def __call__(
        self,
        logits: Int64,
        x: Int64,
        x_stride: Int32,
        global_scale: Int64,
        x_fp4: Int64,
        x_scale: Int64,
        topk_ids: Int64,
        topk_weights: Int64,
        padded_rows: Int64,
        permuted_of: Int64,
        token_of: Int64,
        tile_expert: Int64,
        tile_limit: Int64,
        tiles: Int64,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream` (module docstring for the outputs)."""
        quant = (rows * BLOCKS + THREADS - 1) // THREADS
        quant = (quant + CLUSTER - 1) // CLUSTER * CLUSTER
        self.kernel(
            logits,
            x,
            x_stride,
            global_scale,
            x_fp4,
            x_scale,
            topk_ids,
            topk_weights,
            padded_rows,
            permuted_of,
            token_of,
            tile_expert,
            tile_limit,
            tiles,
            rows,
        ).launch(
            grid=(CLUSTER + quant, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.shared_bytes,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        logits: Int64,
        x: Int64,
        x_stride: Int32,
        global_scale: Int64,
        x_fp4: Int64,
        x_scale: Int64,
        topk_ids: Int64,
        topk_weights: Int64,
        padded_rows: Int64,
        permuted_of: Int64,
        token_of: Int64,
        tile_expert: Int64,
        tile_limit: Int64,
        tiles: Int64,
        rows: Int32,
    ):
        """One CTA: selection and metadata (cluster 0) or quantization."""
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32

        allocator = cutlass.utils.SmemAllocator()
        ids_shared = allocator.allocate_array(
            Int32, MAX_ROWS * TOP_K, byte_alignment=16
        ).toint()
        masks = allocator.allocate_array(
            Int32, EXPERTS * MASK_WORDS, byte_alignment=16
        ).toint()
        first_tile = allocator.allocate_array(Int32, EXPERTS, byte_alignment=16).toint()
        warp_tiles = allocator.allocate_array(Int32, WARPS, byte_alignment=16).toint()

        if cutlass.const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        # Every CTA of the selecting cluster has started before any writes to
        # CTA 0's shared memory (the matching wait precedes those writes).
        if cta < CLUSTER:
            cute.arch.cluster_arrive_relaxed()

        if cta == 0:
            for word in cutlass.range(tid, EXPERTS * MASK_WORDS, THREADS, unroll=1):
                store_shared_u32(masks + word * 4, Int32(0))

        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.stall_us > 0):
            for _ in cutlass.range(self.stall_us, unroll=1):
                nanosleep(Int32(1000))

        if cta < CLUSTER:
            cute.arch.cluster_wait()
            row = warp * CLUSTER + cta
            if row < rows:
                self.select(logits, topk_ids, topk_weights, ids_shared, row, lane)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if cta == 0:
                self.build_metadata(
                    ids_shared,
                    masks,
                    first_tile,
                    warp_tiles,
                    padded_rows,
                    permuted_of,
                    token_of,
                    tile_expert,
                    tile_limit,
                    tiles,
                    rows,
                    tid,
                    warp,
                    lane,
                )
        else:
            block = (cta - CLUSTER) * THREADS + tid
            if block < rows * BLOCKS:
                self.quantize(x, x_stride, global_scale, x_fp4, x_scale, block)

    @cute.jit
    def select(
        self,
        logits: Int64,
        topk_ids: Int64,
        topk_weights: Int64,
        ids_shared: Int32,
        row: Int32,
        lane: Int32,
    ):
        """One warp: the top-10 of `row` and its weights."""
        base = logits + Int64(row) * (EXPERTS * 2)
        scores = [load_bf16(base + Int64(lane + 32 * i) * 2) for i in range(PER_LANE)]

        # Top-10 by (logit order bits, 65535 - expert id): FlashInfer's key.
        keys = [
            (Uint64(ordered_f32(scores[i])) << Uint64(32))
            | Uint64(65535 - (lane + 32 * i))
            for i in range(PER_LANE)
        ]
        mine_bits = Uint32(0)
        mine_low = Uint32(0)
        for k in cutlass.range_constexpr(TOP_K):
            best = keys[0]
            for i in cutlass.range_constexpr(1, PER_LANE):
                best = max_u64(best, keys[i])
            high = Uint32(best >> Uint64(32))
            low = Uint32(best & Uint64(0xFFFFFFFF))
            top_high = warp_max_u32(high)
            if high != top_high:
                low = Uint32(0)
            top_low = warp_max_u32(low)
            winner = (Uint64(top_high) << Uint64(32)) | Uint64(top_low)
            for i in cutlass.range_constexpr(PER_LANE):
                keys[i] = clear_if_equal_u64(keys[i], winner)
            if lane == k:
                mine_bits = top_high
                mine_low = top_low

        # Softmax over the ten: their maximum is the first pick's logit.
        top = cute.arch.shuffle_sync(unordered_f32(mine_bits), 0)
        score = Float32(0.0)
        if lane < TOP_K:
            score = exp_ftz(sub_ftz(unordered_f32(mine_bits), top))
        weight = div_ftz(score, _warp_sum_bfly(score))
        if lane < TOP_K:
            expert = Int32(65535) - Int32(mine_low)
            slot = row * TOP_K + lane
            store_bf16(topk_weights + Int64(slot) * 2, weight)
            store_u32(topk_ids + Int64(slot) * 4, expert)
            store_cluster_u32(ids_shared + slot * 4, Int32(0), expert)

    @cute.jit
    def build_metadata(
        self,
        ids_shared: Int32,
        masks: Int32,
        first_tile: Int32,
        warp_tiles: Int32,
        padded_rows: Int64,
        permuted_of: Int64,
        token_of: Int64,
        tile_expert: Int64,
        tile_limit: Int64,
        tiles: Int64,
        rows: Int32,
        tid: Int32,
        warp: Int32,
        lane: Int32,
    ):
        """CTA 0: the routing metadata from the selected ids."""
        slots = rows * TOP_K
        # Which rows chose each expert.
        for slot in cutlass.range(tid, slots, THREADS, unroll=1):
            expert = load_shared_u32(ids_shared + slot * 4)
            row = slot // TOP_K
            or_shared_u32(
                masks + (expert * MASK_WORDS + row // 32) * 4,
                Uint32(1) << Uint32(row % 32),
            )
        cute.arch.sync_threads()

        # Tiles per expert and their exclusive scan (one thread per expert).
        count = Int32(0)
        expert_tiles = Int32(0)
        inclusive = Int32(0)
        if tid < EXPERTS:
            for w in cutlass.range_constexpr(MASK_WORDS):
                count += Int32(
                    cute.arch.popc(load_shared_u32(masks + (tid * MASK_WORDS + w) * 4))
                )
            expert_tiles = (count + TILE - 1) // TILE
            inclusive = expert_tiles
            for offset in (1, 2, 4, 8, 16):
                other = cute.arch.shuffle_sync_up(inclusive, offset)
                if lane >= offset:
                    inclusive += other
            if lane == 31:
                store_shared_u32(warp_tiles + warp * 4, inclusive)
        cute.arch.sync_threads()
        if tid < EXPERTS:
            before = Int32(0)
            for w in cutlass.range(warp, unroll=1):
                before += load_shared_u32(warp_tiles + w * 4)
            first = before + inclusive - expert_tiles
            store_shared_u32(first_tile + tid * 4, first)
            segment_end = first * TILE + count
            for t in cutlass.range(expert_tiles, unroll=1):
                tile = first + t
                store_u32(tile_expert + Int64(tile) * 4, tid)
                limit = (tile + 1) * TILE
                if segment_end < limit:
                    limit = segment_end
                store_u32(tile_limit + Int64(tile) * 4, limit)
            for p in cutlass.range(
                segment_end, (first + expert_tiles) * TILE, unroll=1
            ):
                store_u32(token_of + Int64(p) * 4, Int32(-1))
            if tid == EXPERTS - 1:
                total = before + inclusive
                store_u32(padded_rows, total * TILE)
                store_u32(tiles, total)
        cute.arch.sync_threads()

        # Each slot's row in the permutation: its expert's first row plus the
        # number of earlier rows that chose the expert.
        for slot in cutlass.range(tid, slots, THREADS, unroll=1):
            expert = load_shared_u32(ids_shared + slot * 4)
            row = slot // TOP_K
            rank = Int32(0)
            for w in cutlass.range_constexpr(MASK_WORDS):
                word = Uint32(load_shared_u32(masks + (expert * MASK_WORDS + w) * 4))
                if w * 32 + 32 <= row:
                    rank += Int32(cute.arch.popc(word))
                elif w * 32 <= row:
                    below = (Uint32(1) << Uint32(row - w * 32)) - Uint32(1)
                    rank += Int32(cute.arch.popc(word & below))
            permuted = load_shared_u32(first_tile + expert * 4) * TILE + rank
            store_u32(permuted_of + Int64(slot) * 4, permuted)
            store_u32(token_of + Int64(permuted) * 4, row)

    @cute.jit
    def quantize(
        self,
        x: Int64,
        x_stride: Int32,
        global_scale: Int64,
        x_fp4: Int64,
        x_scale: Int64,
        block: Int32,
    ):
        """One thread: the NVFP4 scale and values of one 16-value block."""
        row = block // BLOCKS
        column = block % BLOCKS
        source = x + (Int64(row) * Int64(x_stride) + Int64(column * SF_VECTOR)) * 2
        values = list(load_bf16x8(source)) + list(load_bf16x8(source + 16))
        scale_global = load_f32(global_scale)
        amax = Float32(0.0)
        for v in values:
            amax = max_ftz(amax, _abs(v))
        scale = mul_rn(scale_global, mul_rn(amax, rcp_ftz(Float32(6.0))))
        scale_byte, scale_value = e4m3_scale(scale)
        out_scale = Float32(0.0)
        if scale_value != Float32(0.0):
            out_scale = rcp_ftz(mul_rn(scale_value, rcp_ftz(scale_global)))
        scaled = [mul_rn(v, out_scale) for v in values]
        lo = e2m1x8(tuple(scaled[:8]))
        hi = e2m1x8(tuple(scaled[8:]))
        store_u32x2(x_fp4 + Int64(row) * (HIDDEN // 2) + Int64(column) * 8, lo, hi)
        store_u8(x_scale + Int64(row) * BLOCKS + Int64(column), scale_byte)


def _abs(value: Float32) -> Float32:
    """`|value|`."""
    return f32_bits(bits_f32(value) & Uint32(0x7FFFFFFF))


@cache
def _compiled(device_index: int, early_trigger: bool, stall_us: int):
    """`RoutedFront` compiled for a device (the row count is a launch
    argument)."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            RoutedFront(early_trigger, stall_us),
            *(Int64(0),) * 2,
            Int32(HIDDEN),
            *(Int64(0),) * 11,
            Int32(1),
            make_fake_stream(),
        )


class RoutingMetadata:
    """The routing metadata buffers of calls of up to `max_rows` rows, in
    the order of FlashInfer's routing-metadata ABI."""

    def __init__(self, max_rows: int, device: torch.device) -> None:
        tiles = max_tiles(max_rows)
        i32 = dict(dtype=torch.int32, device=device)
        self.max_rows = max_rows
        self.padded_rows = torch.zeros(1, **i32)
        self.permuted_of = torch.zeros(max_rows * TOP_K, **i32)
        # One guard element past the most padded rows (FlashInfer's extent).
        self.token_of = torch.full((tiles * TILE + 1,), -1, **i32)
        self.weights = torch.zeros(max_rows, TOP_K, dtype=torch.bfloat16, device=device)
        self.histogram = torch.zeros(max(2 * EXPERTS, 512), **i32)
        self.per_expert = torch.zeros(EXPERTS, **i32)
        self.tile_expert = torch.zeros(tiles, **i32)
        self.tile_limit = torch.zeros(tiles, **i32)
        self.tiles = torch.zeros(1, **i32)
        self.ids = torch.zeros(max_rows, TOP_K, **i32)

    def tensors(self, rows: int) -> list[torch.Tensor]:
        """The nine tensors of FlashInfer's routing-metadata ABI for a call
        of `rows` rows."""
        return [
            self.padded_rows,
            self.permuted_of,
            self.token_of,
            self.weights[:rows],
            self.histogram,
            self.per_expert,
            self.tile_expert,
            self.tile_limit,
            self.tiles,
        ]


def routed_front(
    logits: torch.Tensor,
    x: torch.Tensor,
    global_scale: torch.Tensor,
    metadata: RoutingMetadata,
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    early_trigger: bool = True,
    stall_us: int = 0,
) -> None:
    """Route `rows` tokens and quantize their activations.

    Args:
        logits: The router logits, BF16 `[rows, 512]`, contiguous.
        x: The layer input, BF16 `[rows, 2560]` with contiguous rows.
        global_scale: The NVFP4 global input scale, FP32; its first element
            (vLLM stores the layer's scale once per expert, and its
            quantization reads the first).
        metadata: Filled with the routing (ids, weights and the GEMMs'
            metadata) of the `rows` rows.
        x_fp4: Filled with the E2M1 values, uint8 `[>= rows, 1280]`.
        x_scale: Filled with the E4M3 scales, uint8 `[>= rows, 160]`.
        early_trigger: Let the next kernel launch at entry.
        stall_us: Tests only (`RoutedFront`).

    """
    rows = logits.shape[0]
    assert 1 <= rows <= min(MAX_ROWS, metadata.max_rows)
    assert logits.dtype == torch.bfloat16 and logits.shape == (rows, EXPERTS)
    assert logits.is_contiguous()
    assert x.dtype == torch.bfloat16 and x.shape == (rows, HIDDEN) and x.stride(1) == 1
    assert x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    assert global_scale.dtype == torch.float32 and global_scale.numel() >= 1
    assert x_fp4.dtype == torch.uint8 and x_fp4.is_contiguous()
    assert x_fp4.shape[1] == HIDDEN // 2 and x_fp4.shape[0] >= rows
    assert x_scale.dtype == torch.uint8 and x_scale.is_contiguous()
    assert x_scale.shape[1] == BLOCKS and x_scale.shape[0] >= rows
    _compiled(logits.device.index, early_trigger, stall_us)(
        Int64(logits.data_ptr()),
        Int64(x.data_ptr()),
        Int32(x.stride(0)),
        Int64(global_scale.data_ptr()),
        Int64(x_fp4.data_ptr()),
        Int64(x_scale.data_ptr()),
        Int64(metadata.ids.data_ptr()),
        Int64(metadata.weights.data_ptr()),
        Int64(metadata.padded_rows.data_ptr()),
        Int64(metadata.permuted_of.data_ptr()),
        Int64(metadata.token_of.data_ptr()),
        Int64(metadata.tile_expert.data_ptr()),
        Int64(metadata.tile_limit.data_ptr()),
        Int64(metadata.tiles.data_ptr()),
        Int32(rows),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
