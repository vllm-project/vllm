# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SynthID-Text sampling.

Tokens whose g-bits agree on a chunk of layers get the same multipliers in
that chunk, so the chunk's layers can run on a 2**CHUNK_BITS-bucket histogram
of the mass reweighted by the earlier chunks. A depth-32 tournament then takes
three vocabulary passes, each building one histogram, and a Gumbel-max pass
that samples from the final distribution.
"""

import functools

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK as _UINT32_MASK_VALUE
from vllm.v1.worker.gpu.sample.gumbel import (
    _log1p_neg_stable,
    gumbel_noised_argmax,
    murmur3_uniform32,
    murmur3_uniform64,
)
from vllm.v1.worker.gpu.sample.watermark import (
    _CONTEXT_DOMAIN,
    _UINT32_MASK,
    _philox4x32_10,
    _philox_candidate_words,
    _philox_context_state,
)

_MAX_CHUNK_BITS = 11
_MAX_CHUNKS = 3
# One Philox call yields the words of 4 consecutive tokens.
_HIST_GROUPS = 512
_HIST_TILE = 4 * _HIST_GROUPS
# Each histogram word adds at most 2**16 per token; this many tokens per
# program cannot overflow its uint32 bins.
_HIST_MAX_TOKENS = (1 << 16) - 1

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    HAS_GLUON = HAS_TRITON
except ImportError:
    HAS_GLUON = False


if HAS_GLUON:

    @gluon.jit
    def _load_shared_table(
        log_tables_ptr,
        row,
        chunk: gl.constexpr,
        NUM_CHUNKS: gl.constexpr,
        NUM_BINS: gl.constexpr,
        layout: gl.constexpr,
        shared_layout: gl.constexpr,
    ):
        bins = gl.arange(0, NUM_BINS, layout=layout)
        return gl.allocate_shared_memory(
            gl.float32,
            [NUM_BINS],
            shared_layout,
            gl.load(log_tables_ptr + (row * NUM_CHUNKS + chunk) * NUM_BINS + bins),
        )

    @gluon.jit
    def _context_state(contexts_ptr, key_0, key_1, CONTEXT_WIDTH: gl.constexpr):
        """`_philox_context_state` with Gluon scalars."""
        state_0 = gl.full((), _CONTEXT_DOMAIN, gl.uint32, layout=None)
        state_1 = gl.full((), CONTEXT_WIDTH, gl.uint32, layout=None)
        state_2 = gl.full((), 0, gl.uint32, layout=None)
        state_3 = gl.full((), 0, gl.uint32, layout=None)
        for offset in gl.static_range(0, CONTEXT_WIDTH, 4):
            context_0 = gl.load(contexts_ptr + offset).to(gl.uint32)
            if offset + 1 < CONTEXT_WIDTH:
                context_1 = gl.load(contexts_ptr + offset + 1).to(gl.uint32)
            else:
                context_1 = gl.full((), _UINT32_MASK - 1, gl.uint32, layout=None)
            if offset + 2 < CONTEXT_WIDTH:
                context_2 = gl.load(contexts_ptr + offset + 2).to(gl.uint32)
            else:
                context_2 = gl.full((), _UINT32_MASK - 2, gl.uint32, layout=None)
            if offset + 3 < CONTEXT_WIDTH:
                context_3 = gl.load(contexts_ptr + offset + 3).to(gl.uint32)
            else:
                context_3 = gl.full((), _UINT32_MASK - 3, gl.uint32, layout=None)
            state_0, state_1, state_2, state_3 = _philox4x32_10(
                (state_0 ^ context_0) & _UINT32_MASK,
                (state_1 ^ context_1) & _UINT32_MASK,
                (state_2 ^ context_2) & _UINT32_MASK,
                (state_3 ^ context_3) & _UINT32_MASK,
                (key_0 ^ offset) & _UINT32_MASK,
                key_1,
            )
        return state_0, state_1, state_2, state_3

    @gluon.jit
    def _build_table(
        partial_ptr,
        partial_max_ptr,
        row_offset_ptr,
        log_tables_ptr,
        row,
        num_splits,
        DEPTH: gl.constexpr,
        NUM_CHUNKS: gl.constexpr,
        CHUNK_BITS: gl.constexpr,
        CHUNK: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """Run the CHUNK-th chunk's layers on the row's bucket masses."""
        NUM_BINS: gl.constexpr = 1 << CHUNK_BITS
        SPLIT_BLOCK: gl.constexpr = 2
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, num_warps], [1, 0])
        bins = gl.arange(0, NUM_BINS, layout=gl.SliceLayout(0, layout))
        splits = gl.arange(0, SPLIT_BLOCK, layout=gl.SliceLayout(1, layout))
        if CHUNK == 0:
            offset = float("-inf")
            for start in range(0, num_splits, SPLIT_BLOCK):
                split_max = gl.load(
                    partial_max_ptr + row * num_splits + start + splits,
                    mask=start + splits < num_splits,
                    other=float("-inf"),
                    cache_modifier=".cg",
                )
                offset = gl.maximum(offset, gl.max(split_max, axis=0))
        else:
            offset = gl.load(row_offset_ptr + row)
        hist = gl.zeros([NUM_BINS], gl.float32, gl.SliceLayout(0, layout))
        for start in range(0, num_splits, SPLIT_BLOCK):
            split_mask = start + splits < num_splits
            partial = gl.load(
                partial_ptr
                + gl.expand_dims((row * num_splits + start + splits) * NUM_BINS, 1)
                + gl.expand_dims(bins, 0),
                mask=gl.expand_dims(split_mask, 1),
                other=0.0,
                cache_modifier=".cg",
            )
            if CHUNK == 0:
                split_max = gl.load(
                    partial_max_ptr + row * num_splits + start + splits,
                    mask=split_mask,
                    other=float("-inf"),
                    cache_modifier=".cg",
                )
                scale = gl.where(
                    split_max == float("-inf"), 0.0, gl.exp(split_max - offset)
                )
                partial *= gl.expand_dims(scale, 1)
            hist += gl.sum(partial, axis=0)
        # A layer keeps the total mass, so each layer needs one reduction.
        total = gl.sum(hist, axis=0)
        weight = gl.full([NUM_BINS], 1.0, gl.float32, gl.SliceLayout(0, layout))
        for bit in gl.static_range(CHUNK_BITS):
            if CHUNK * CHUNK_BITS + bit < DEPTH:
                g = ((bins >> bit) & 1) != 0
                # 1 - m from the g = 0 mass: full precision when m is close to 1.
                factor_0 = gl.sum(gl.where(g, 0.0, hist * weight), axis=0) / total
                weight *= gl.where(g, 1.0 + factor_0, factor_0)
        gl.store(
            log_tables_ptr + (row * NUM_CHUNKS + CHUNK) * NUM_BINS + bins,
            gl.log(weight),
        )
        # Normalize the next chunk's masses so that they sum to 1.
        gl.store(row_offset_ptr + row, offset + gl.log(total))

    @gluon.jit
    def _is_last_program(counter_ptr, num_programs):
        """Count this program; true for the row's last one, which resets."""
        gl.barrier()
        ticket = gl.atomic_add(counter_ptr, 1, sem="acq_rel")
        is_last = ticket == num_programs - 1
        if is_last:
            gl.store(counter_ptr, 0)
        return is_last

    @gluon.jit
    def _tile_words(
        tile,
        state_0,
        state_1,
        state_2,
        state_3,
        key_0,
        key_1,
        GROUPS: gl.constexpr,
        layout: gl.constexpr,
    ):
        """Philox words of the tile's 4 * GROUPS tokens, in `layout`."""
        num_warps: gl.constexpr = layout.warps_per_cta[0]
        group_layout: gl.constexpr = gl.BlockedLayout(
            [1, 4], [32, 1], [num_warps, 1], [1, 0]
        )
        groups = tile * GROUPS + gl.arange(
            0, GROUPS, layout=gl.SliceLayout(1, group_layout)
        )
        word_0, word_1, word_2, word_3 = _philox_candidate_words(
            groups, state_0, state_1, state_2, state_3, key_0, key_1
        )
        word_idx = gl.expand_dims(
            gl.arange(0, 4, layout=gl.SliceLayout(0, group_layout)), 0
        )
        words = gl.where(
            word_idx == 0,
            gl.expand_dims(word_0, 1),
            gl.where(
                word_idx == 1,
                gl.expand_dims(word_1, 1),
                gl.where(
                    word_idx == 2,
                    gl.expand_dims(word_2, 1),
                    gl.expand_dims(word_3, 1),
                ),
            ),
        )
        return gl.convert_layout(gl.reshape(words, [4 * GROUPS]), layout)

    @gluon.jit
    def _load_fixed_hist(hist_hi, hist_mid, hist_lo, layout: gl.constexpr):
        """Read the three uint32 words of each bin as one float64 sum."""
        hi = hist_hi.load(layout).to(gl.uint32, bitcast=True).to(gl.float64)
        mid = hist_mid.load(layout).to(gl.uint32, bitcast=True).to(gl.float64)
        lo = hist_lo.load(layout).to(gl.uint32, bitcast=True).to(gl.float64)
        return hi * 4294967296.0 + mid * 65536.0 + lo

    @gluon.jit
    def _store_fixed_hist(hist_hi, hist_mid, hist_lo, value):
        hi = gl.floor(value * 2.3283064365386963e-10)
        value -= hi * 4294967296.0
        mid = gl.floor(value * 1.52587890625e-05)
        lo = value - mid * 65536.0
        hist_hi.store(hi.to(gl.uint32).to(gl.int32, bitcast=True))
        hist_mid.store(mid.to(gl.int32))
        hist_lo.store(lo.to(gl.int32))

    @gluon.jit(
        do_not_specialize=[
            "key_0_value",
            "key_1_value",
            "stream_0_value",
            "stream_1_value",
        ]
    )
    def _synthid_hist_kernel(
        partial_ptr,  # [B, num_splits, 2**CHUNK_BITS]
        partial_max_ptr,  # [B, num_splits]
        log_tables_ptr,  # [B, NUM_CHUNKS, 2**CHUNK_BITS]
        row_offset_ptr,  # [B]
        counter_ptr,  # [B], zero between launches
        logits_ptr,
        logits_stride,
        contexts_ptr,
        context_stride,
        skip_ptr,
        key_0_value,
        key_1_value,
        stream_0_value,
        stream_1_value,
        vocab_size,
        tiles_per_split,
        CONTEXT_WIDTH: gl.constexpr,
        DEPTH: gl.constexpr,
        NUM_CHUNKS: gl.constexpr,
        CHUNK_BITS: gl.constexpr,
        CHUNK: gl.constexpr,
        HAS_SKIP: gl.constexpr,
        GROUPS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """Histogram one split of a row's mass by the CHUNK-th g-bit pattern.

        The mass is exp(logit - offset) times the earlier chunks' multipliers,
        where the offset keeps every token's mass in [0, 1]: the running max
        of the split for chunk 0, else the row's log-normalizer. Bins
        accumulate it as 48-bit fixed point in three uint32 shared-memory
        words, as Hopper has native integer but not float shared atomics.
        The row's last program to finish runs the chunk's layers.
        """
        row = gl.program_id(0).to(gl.int64)
        split = gl.program_id(1)
        num_splits = gl.num_programs(1)
        watermark = True
        if HAS_SKIP:
            watermark = gl.load(skip_ptr + row) == 0
        if watermark:
            _histogram_split(
                partial_ptr,
                partial_max_ptr,
                log_tables_ptr,
                row_offset_ptr,
                logits_ptr,
                logits_stride,
                contexts_ptr,
                context_stride,
                key_0_value,
                key_1_value,
                stream_0_value,
                stream_1_value,
                vocab_size,
                tiles_per_split,
                row,
                split,
                num_splits,
                CONTEXT_WIDTH,
                DEPTH,
                NUM_CHUNKS,
                CHUNK_BITS,
                CHUNK,
                GROUPS,
                num_warps,
            )
        is_last = _is_last_program(counter_ptr + row, num_splits)
        if HAS_SKIP:
            is_last &= watermark
        if is_last:
            _build_table(
                partial_ptr,
                partial_max_ptr,
                row_offset_ptr,
                log_tables_ptr,
                row,
                num_splits,
                DEPTH,
                NUM_CHUNKS,
                CHUNK_BITS,
                CHUNK,
                num_warps,
            )

    @gluon.jit
    def _histogram_split(
        partial_ptr,
        partial_max_ptr,
        log_tables_ptr,
        row_offset_ptr,
        logits_ptr,
        logits_stride,
        contexts_ptr,
        context_stride,
        key_0_value,
        key_1_value,
        stream_0_value,
        stream_1_value,
        vocab_size,
        tiles_per_split,
        row,
        split,
        num_splits,
        CONTEXT_WIDTH: gl.constexpr,
        DEPTH: gl.constexpr,
        NUM_CHUNKS: gl.constexpr,
        CHUNK_BITS: gl.constexpr,
        CHUNK: gl.constexpr,
        GROUPS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        NUM_BINS: gl.constexpr = 1 << CHUNK_BITS
        BIN_MASK: gl.constexpr = NUM_BINS - 1
        NUM_LAYERS: gl.constexpr = min(CHUNK_BITS, DEPTH - CHUNK * CHUNK_BITS)
        layout: gl.constexpr = gl.BlockedLayout([4], [32], [num_warps], [0])
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

        zeros = gl.zeros([NUM_BINS], gl.int32, layout)
        hist_hi = gl.allocate_shared_memory(gl.int32, [NUM_BINS], shared_layout, zeros)
        hist_mid = gl.allocate_shared_memory(gl.int32, [NUM_BINS], shared_layout, zeros)
        hist_lo = gl.allocate_shared_memory(gl.int32, [NUM_BINS], shared_layout, zeros)
        if CHUNK >= 1:
            table_0 = _load_shared_table(
                log_tables_ptr, row, 0, NUM_CHUNKS, NUM_BINS, layout, shared_layout
            )
        if CHUNK >= 2:
            table_1 = _load_shared_table(
                log_tables_ptr, row, 1, NUM_CHUNKS, NUM_BINS, layout, shared_layout
            )
        offset = float("-inf") if CHUNK == 0 else gl.load(row_offset_ptr + row)

        key_0 = key_0_value.to(gl.uint32)
        key_1 = key_1_value.to(gl.uint32)
        state_0, state_1, state_2, state_3 = _context_state(
            contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
        )
        # The stream enters the candidate key as in `_philox_uint32`.
        key_0 ^= stream_1_value.to(gl.uint32)
        key_1 ^= stream_0_value.to(gl.uint32)
        gl.barrier()
        for tile_offset in range(tiles_per_split):
            tile = split * tiles_per_split + tile_offset
            words = _tile_words(
                tile, state_0, state_1, state_2, state_3, key_0, key_1, GROUPS, layout
            )
            candidate = tile * (4 * GROUPS) + gl.arange(0, 4 * GROUPS, layout=layout)
            logits = gl.load(
                logits_ptr + row * logits_stride + candidate,
                mask=candidate < vocab_size,
                other=float("-inf"),
            ).to(gl.float32)
            if CHUNK == 0:
                tile_max = gl.max(logits, axis=0)
                if tile_max > offset:
                    gl.barrier()
                    hist = _load_fixed_hist(hist_hi, hist_mid, hist_lo, layout)
                    scale = gl.exp(offset - tile_max).to(gl.float64)
                    _store_fixed_hist(
                        hist_hi, hist_mid, hist_lo, gl.floor(hist * scale)
                    )
                    gl.barrier()
                    offset = tile_max
            # A split may start with masked tiles, where offset is still -inf.
            log_mass = gl.where(logits > float("-inf"), logits - offset, logits)
            if CHUNK >= 1:
                log_mass += table_0.gather((words & BIN_MASK).to(gl.int32), axis=0)
            if CHUNK >= 2:
                log_mass += table_1.gather(
                    ((words >> CHUNK_BITS) & BIN_MASK).to(gl.int32), axis=0
                )
            # Rounds down: the 2**-48 per token lost is below fp32 resolution.
            fixed = (gl.exp(log_mass) * 281474976710656.0).to(gl.int64)
            pattern = (words >> (CHUNK * CHUNK_BITS)) & ((1 << NUM_LAYERS) - 1)
            pattern = pattern.to(gl.int32)
            hi = (fixed >> 32).to(gl.int32)
            mid = ((fixed >> 16) & 0xFFFF).to(gl.int32)
            lo = (fixed & 0xFFFF).to(gl.int32)
            hist_hi.atomic_scatter_add(hi, pattern, axis=0, mask=hi != 0)
            hist_mid.atomic_scatter_add(mid, pattern, axis=0, mask=mid != 0)
            hist_lo.atomic_scatter_add(lo, pattern, axis=0, mask=lo != 0)
        gl.barrier()
        hist = _load_fixed_hist(hist_hi, hist_mid, hist_lo, layout)
        bins = gl.arange(0, NUM_BINS, layout=layout)
        gl.store(
            partial_ptr + (row * num_splits + split) * NUM_BINS + bins,
            (hist * 3.552713678800501e-15).to(gl.float32),  # 2**-48
        )
        if CHUNK == 0:
            gl.store(partial_max_ptr + row * num_splits + split, offset)

    @gluon.jit(
        do_not_specialize=[
            "key_0_value",
            "key_1_value",
            "stream_0_value",
            "stream_1_value",
        ]
    )
    def _synthid_sample_kernel(
        sampled_ptr,  # [B]
        split_argmax_ptr,  # [B, num_splits]
        split_max_ptr,  # [B, num_splits]
        counter_ptr,  # [B], zero between launches
        log_tables_ptr,  # [B, NUM_CHUNKS, 2**CHUNK_BITS]
        logits_ptr,
        logits_stride,
        contexts_ptr,
        context_stride,
        skip_ptr,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        key_0_value,
        key_1_value,
        stream_0_value,
        stream_1_value,
        vocab_size,
        tiles_per_split,
        CONTEXT_WIDTH: gl.constexpr,
        NUM_CHUNKS: gl.constexpr,
        CHUNK_BITS: gl.constexpr,
        HAS_SKIP: gl.constexpr,
        USE_FP64: gl.constexpr,
        GROUPS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """Gumbel-max over one split of a row, after all chunks' multipliers."""
        row = gl.program_id(0).to(gl.int64)
        split = gl.program_id(1)
        num_splits = gl.num_programs(1)
        NUM_BINS: gl.constexpr = 1 << CHUNK_BITS
        BIN_MASK: gl.constexpr = NUM_BINS - 1
        layout: gl.constexpr = gl.BlockedLayout([4], [32], [num_warps], [0])
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
        watermark = True
        if HAS_SKIP:
            watermark = gl.load(skip_ptr + row) == 0
        table_0 = _load_shared_table(
            log_tables_ptr, row, 0, NUM_CHUNKS, NUM_BINS, layout, shared_layout
        )
        if NUM_CHUNKS >= 2:
            table_1 = _load_shared_table(
                log_tables_ptr, row, 1, NUM_CHUNKS, NUM_BINS, layout, shared_layout
            )
        if NUM_CHUNKS >= 3:
            table_2 = _load_shared_table(
                log_tables_ptr, row, 2, NUM_CHUNKS, NUM_BINS, layout, shared_layout
            )
        gl.barrier()

        key_0 = key_0_value.to(gl.uint32)
        key_1 = key_1_value.to(gl.uint32)
        state_0, state_1, state_2, state_3 = _context_state(
            contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
        )
        key_0 ^= stream_1_value.to(gl.uint32)
        key_1 ^= stream_0_value.to(gl.uint32)
        req_state_idx = gl.load(expanded_idx_mapping_ptr + row).to(gl.int64)
        is_valid_req = req_state_idx >= 0
        temp = gl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0)
        seed = gl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = gl.load(pos_ptr + row)
        best_value = gl.full(
            (), float("-inf"), gl.float64 if USE_FP64 else gl.float32, layout=None
        )
        best_token = gl.full((), 0, gl.int64, layout=None)
        for tile_offset in range(tiles_per_split):
            tile = split * tiles_per_split + tile_offset
            candidate = tile * (4 * GROUPS) + gl.arange(0, 4 * GROUPS, layout=layout)
            mask = candidate < vocab_size
            logits = gl.load(
                logits_ptr + row * logits_stride + candidate,
                mask=mask,
                other=float("-inf"),
            ).to(gl.float32)
            if watermark:
                words = _tile_words(
                    tile,
                    state_0,
                    state_1,
                    state_2,
                    state_3,
                    key_0,
                    key_1,
                    GROUPS,
                    layout,
                )
                logits += table_0.gather((words & BIN_MASK).to(gl.int32), axis=0)
                if NUM_CHUNKS >= 2:
                    logits += table_1.gather(
                        ((words >> CHUNK_BITS) & BIN_MASK).to(gl.int32), axis=0
                    )
                if NUM_CHUNKS >= 3:
                    logits += table_2.gather(
                        ((words >> (2 * CHUNK_BITS)) & BIN_MASK).to(gl.int32), axis=0
                    )
            # Same noise as `gumbel_noised_argmax`, whose indexed max Gluon
            # lacks.
            if USE_FP64:
                logits = logits.to(gl.float64)
            if temp != 0.0:
                if USE_FP64:
                    uniform = murmur3_uniform64(seed, pos, candidate)
                    noise = -gl.log(-gl.log(uniform))
                else:
                    uniform = murmur3_uniform32(seed, pos, candidate)
                    noise = -gl.log(-_log1p_neg_stable(uniform))
                logits = gl.where(mask, logits + noise, float("-inf"))
            value = gl.max(logits, axis=0)
            token = gl.min(gl.where(logits == value, candidate, vocab_size), axis=0)
            # Strictly greater keeps the lowest token on ties, as an argmax
            # over the whole row does.
            if value > best_value:
                best_value = value
                best_token = token.to(gl.int64)
        gl.store(split_argmax_ptr + row * num_splits + split, best_token)
        gl.store(split_max_ptr + row * num_splits + split, best_value)
        if _is_last_program(counter_ptr + row, num_splits):
            # Splits are in token order: the first maximal split holds the
            # lowest maximal token.
            SPLIT_BLOCK: gl.constexpr = 32 * num_warps
            split_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
            splits = gl.arange(0, SPLIT_BLOCK, layout=split_layout)
            for start in range(0, num_splits, SPLIT_BLOCK):
                split_mask = start + splits < num_splits
                values = gl.load(
                    split_max_ptr + row * num_splits + start + splits,
                    mask=split_mask,
                    other=float("-inf"),
                    cache_modifier=".cg",
                )
                tokens = gl.load(
                    split_argmax_ptr + row * num_splits + start + splits,
                    mask=split_mask,
                    other=0,
                    cache_modifier=".cg",
                )
                value = gl.max(values, axis=0)
                if (value > best_value) | (start == 0):
                    first = gl.min(
                        gl.where(values == value, splits, SPLIT_BLOCK), axis=0
                    )
                    best_value = value
                    best_token = gl.sum(gl.where(splits == first, tokens, 0), axis=0)
            gl.store(sampled_ptr + row, best_token)


_COUNTERS: dict[torch.device, torch.Tensor] = {}


def _row_counters(device: torch.device, num_rows: int) -> torch.Tensor:
    """Zeroed per-row counters; each kernel's last program resets its own."""
    counters = _COUNTERS.get(device)
    if counters is None or counters.shape[1] < num_rows:
        counters = torch.zeros(
            _MAX_CHUNKS + 1, max(num_rows, 1024), dtype=torch.int32, device=device
        )
        _COUNTERS[device] = counters
    return counters


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def synthid_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    depth: int,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    stream: int = 0,
    skip_mask: torch.Tensor | None = None,
    use_fp64: bool = False,
    programs_per_sm: int = 4,
) -> torch.Tensor:
    """Gumbel-max sample `logits` [B, V] after the SynthID-Text tournament.

    Draws the same noise as `gumbel_sample` on the log-probs that
    `synthid_watermark_logits` returns. Rows in `skip_mask` are sampled from
    `logits` unchanged.
    """
    assert 1 <= depth <= 32
    num_chunks = triton.cdiv(depth, _MAX_CHUNK_BITS)
    assert num_chunks <= _MAX_CHUNKS
    chunk_bits = triton.cdiv(depth, num_chunks)
    num_bins = 1 << chunk_bits
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    num_tokens, vocab_size = logits.shape
    num_tiles = triton.cdiv(vocab_size, _HIST_TILE)
    # At least two tiles per program, or reducing the partial histograms
    # costs more than the parallelism saves.
    num_splits = min(
        triton.cdiv(programs_per_sm * _num_sms(logits.device), max(num_tokens, 1)),
        triton.cdiv(num_tiles, 2),
    )
    tiles_per_split = min(
        triton.cdiv(num_tiles, num_splits), _HIST_MAX_TOKENS // _HIST_TILE
    )
    num_splits = triton.cdiv(num_tiles, tiles_per_split)

    sampled = logits.new_empty(num_tokens, dtype=torch.int64)
    if num_tokens == 0:
        return sampled
    partial = logits.new_empty(num_tokens, num_splits, num_bins, dtype=torch.float32)
    partial_max = logits.new_empty(num_tokens, num_splits, dtype=torch.float32)
    row_offset = logits.new_empty(num_tokens, dtype=torch.float32)
    log_tables = logits.new_empty(num_tokens, num_chunks, num_bins, dtype=torch.float32)
    split_argmax = logits.new_empty(num_tokens, num_splits, dtype=torch.int64)
    split_max = logits.new_empty(
        num_tokens, num_splits, dtype=torch.float64 if use_fp64 else torch.float32
    )
    counters = _row_counters(logits.device, num_tokens)

    grid = (num_tokens, num_splits)
    prf_args = (
        key & _UINT32_MASK_VALUE,
        key >> 32,
        stream & _UINT32_MASK_VALUE,
        stream >> 32,
        vocab_size,
        tiles_per_split,
    )
    for chunk in range(num_chunks):
        _synthid_hist_kernel[grid](
            partial,
            partial_max,
            log_tables,
            row_offset,
            counters[chunk],
            logits,
            logits.stride(0),
            contexts,
            contexts.stride(0),
            skip_mask,
            *prf_args,
            CONTEXT_WIDTH=contexts.shape[-1],
            DEPTH=depth,
            NUM_CHUNKS=num_chunks,
            CHUNK_BITS=chunk_bits,
            CHUNK=chunk,
            HAS_SKIP=skip_mask is not None,
            GROUPS=_HIST_GROUPS,
            num_warps=16,
        )
    _synthid_sample_kernel[grid](
        sampled,
        split_argmax,
        split_max,
        counters[_MAX_CHUNKS],
        log_tables,
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        skip_mask,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        *prf_args,
        CONTEXT_WIDTH=contexts.shape[-1],
        NUM_CHUNKS=num_chunks,
        CHUNK_BITS=chunk_bits,
        HAS_SKIP=skip_mask is not None,
        USE_FP64=use_fp64,
        GROUPS=_HIST_GROUPS,
        num_warps=8,
    )
    return sampled


@triton.jit(
    do_not_specialize=["key_0_value", "key_1_value", "stream_0_value", "stream_1_value"]
)
def _synthid_top_k_sample_kernel(
    sampled_ptr,
    values_ptr,  # [B, K]
    values_stride,
    token_ids_ptr,  # [B, K]
    token_ids_stride,
    contexts_ptr,
    context_stride,
    skip_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    key_0_value,
    key_1_value,
    stream_0_value,
    stream_1_value,
    num_kept,
    CONTEXT_WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    HAS_SKIP: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Run every layer on a row's kept tokens and Gumbel-max sample them."""
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < num_kept
    logits = tl.load(
        values_ptr + row * values_stride + offsets, mask=mask, other=float("-inf")
    ).to(tl.float32)
    token_ids = tl.load(
        token_ids_ptr + row * token_ids_stride + offsets, mask=mask, other=0
    )
    skip = False
    if HAS_SKIP:
        skip = tl.load(skip_ptr + row)
    if not skip:
        key_0 = key_0_value.to(tl.uint32)
        key_1 = key_1_value.to(tl.uint32)
        state_0, state_1, state_2, state_3 = _philox_context_state(
            contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
        )
        word_0, word_1, word_2, word_3 = _philox_candidate_words(
            token_ids >> 2,
            state_0,
            state_1,
            state_2,
            state_3,
            key_0 ^ stream_1_value.to(tl.uint32),
            key_1 ^ stream_0_value.to(tl.uint32),
        )
        word_idx = token_ids & 3
        words = tl.where(
            word_idx == 0,
            word_0,
            tl.where(word_idx == 1, word_1, tl.where(word_idx == 2, word_2, word_3)),
        )
        probs = tl.exp(logits - tl.max(logits, axis=0))
        total = tl.sum(probs, axis=0)
        for layer in tl.static_range(DEPTH):
            g = ((words >> layer) & 1) != 0
            factor_0 = tl.sum(tl.where(g, 0.0, probs), axis=0) / total
            probs *= tl.where(g, 1.0 + factor_0, factor_0)
        logits = tl.log(probs)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0)
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    pos = tl.load(pos_ptr + row)
    _, idx = gumbel_noised_argmax(
        logits,
        token_ids,
        mask,
        seed,
        pos,
        temp.to(tl.float32),
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    tl.store(sampled_ptr + row, tl.sum(tl.where(offsets == idx, token_ids, 0)))


def synthid_sample_top_k(
    values: torch.Tensor,
    token_ids: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    depth: int,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    stream: int = 0,
    skip_mask: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """`synthid_sample` for rows whose support is the kept `token_ids` [B, K].

    Masked entries of `values` are -inf. Draws the same noise as
    `synthid_sample` on the full logits with every other token masked.
    """
    assert 1 <= depth <= 32
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    num_tokens, num_kept = values.shape
    sampled = token_ids.new_empty(num_tokens)
    if num_tokens == 0:
        return sampled
    block_k = triton.next_power_of_2(num_kept)
    _synthid_top_k_sample_kernel[(num_tokens,)](
        sampled,
        values,
        values.stride(0),
        token_ids,
        token_ids.stride(0),
        contexts,
        contexts.stride(0),
        skip_mask,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        key & _UINT32_MASK_VALUE,
        key >> 32,
        stream & _UINT32_MASK_VALUE,
        stream >> 32,
        num_kept,
        CONTEXT_WIDTH=contexts.shape[-1],
        DEPTH=depth,
        HAS_SKIP=skip_mask is not None,
        USE_FP64=use_fp64,
        BLOCK_K=block_k,
        num_warps=4 if block_k <= 256 else 8,
    )
    return sampled
