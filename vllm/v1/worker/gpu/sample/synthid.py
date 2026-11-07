# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused SynthID-Text sampling.

Tokens whose g-bits agree on a chunk of layers get the same multipliers in
that chunk, so the chunk's layers can run on a 2**CHUNK_BITS-bucket histogram
of the mass reweighted by the earlier chunks. A depth-32 tournament takes
three vocabulary passes, each building one histogram into a table of
per-bucket log multipliers. A last pass adds the tables to the logits and
samples with Gumbel-max.
"""

import functools

import torch
from packaging import version

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, gl, gluon, tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noise
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
)

_UINT32_MASK = 0xFFFFFFFF

_MAX_CHUNK_BITS = 11
_MAX_CHUNKS = 3
# One counter per chunk's histogram pass plus one for the last pass.
_NUM_COUNTERS = _MAX_CHUNKS + 1
_CONSUME_WARPS = 8
# The last program reduces the splits in a single block of one lane each.
_MAX_SPLITS = 32 * _CONSUME_WARPS
_PROGRAMS_PER_SM = 4
# One Philox call yields the words of 4 consecutive tokens.
_HIST_GROUPS = 512
_HIST_TILE = 4 * _HIST_GROUPS
# Each histogram word adds at most 2**16 per token; this many tokens per
# program cannot overflow its uint32 bins.
_HIST_MAX_TOKENS = (1 << 16) - 1
# Triton 3.6 and 3.7 import Gluon but lack gl.barrier or
# shared_memory_descriptor.atomic_scatter_add.
HAS_GLUON = HAS_TRITON and version.parse(triton.__version__).release >= (3, 8)


def is_supported() -> bool:
    """Validated on H100 (sm90); expected to work on sm80 and newer.

    The layouts assume 32-thread warps, so ROCm cannot run the kernels.
    """
    return (
        HAS_GLUON
        and current_platform.is_cuda()
        and current_platform.has_device_capability(80)
    )


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _synthid_prepare_kernel(
    states_ptr,  # [B, 4]
    counters_ptr,  # [B, _NUM_COUNTERS]
    contexts_ptr,
    context_stride,
    key_0_value,
    key_1_value,
    CONTEXT_WIDTH: tl.constexpr,
    NUM_COUNTERS: tl.constexpr,
):
    """Hash each row's context once and zero its ticket counters."""
    row = tl.program_id(0).to(tl.int64)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride,
        key_0_value.to(tl.uint32),
        key_1_value.to(tl.uint32),
        CONTEXT_WIDTH,
    )
    tl.store(states_ptr + row * 4, state_0.to(tl.int32, bitcast=True))
    tl.store(states_ptr + row * 4 + 1, state_1.to(tl.int32, bitcast=True))
    tl.store(states_ptr + row * 4 + 2, state_2.to(tl.int32, bitcast=True))
    tl.store(states_ptr + row * 4 + 3, state_3.to(tl.int32, bitcast=True))
    counters = tl.arange(0, NUM_COUNTERS)
    tl.store(counters_ptr + row * NUM_COUNTERS + counters, 0)


if HAS_TRITON:

    @gluon.jit
    def _load_shared_table(
        log_tables_ptr,
        row,
        chunk: gl.constexpr,
        active,
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
            gl.load(
                log_tables_ptr + (row * NUM_CHUNKS + chunk) * NUM_BINS + bins,
                mask=(bins >= 0) & active,
                other=0.0,
            ),
        )

    @gluon.jit
    def _load_states(states_ptr, key_0, key_1, stream_0, stream_1):
        state_0 = gl.load(states_ptr).to(gl.uint32)
        state_1 = gl.load(states_ptr + 1).to(gl.uint32)
        state_2 = gl.load(states_ptr + 2).to(gl.uint32)
        state_3 = gl.load(states_ptr + 3).to(gl.uint32)
        # The stream enters the candidate key as in `_philox_uint32`.
        return state_0, state_1, state_2, state_3, key_0 ^ stream_1, key_1 ^ stream_0

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
        # A layer keeps the total mass: the normalizer is the first total.
        log_total = gl.log(gl.sum(hist, axis=0))
        weight = gl.full([NUM_BINS], 1.0, gl.float32, gl.SliceLayout(0, layout))
        for bit in gl.static_range(CHUNK_BITS):
            if CHUNK * CHUNK_BITS + bit < DEPTH:
                g = ((bins >> bit) & 1) != 0
                mass = hist * weight
                one_mass = gl.sum(gl.where(g, mass, 0.0), axis=0)
                zero_mass = gl.sum(gl.where(g, 0.0, mass), axis=0)
                # `_apply_layer`: the multiplier is 1 + g - m, taken as an
                # offset from 1 when m < 0.5 and from the g = 0 mass otherwise,
                # as rounding 1 - m moves every g = 0 token the same way.
                inv_total = 1.0 / (zero_mass + one_mass)
                mean = one_mass * inv_total
                offset_form = mean < 0.5
                factor = gl.where(offset_form, -mean, zero_mass * inv_total)
                weight = weight * factor + (
                    gl.where(offset_form, weight, 0.0) + gl.where(g, weight, 0.0)
                )
        gl.store(
            log_tables_ptr + (row * NUM_CHUNKS + CHUNK) * NUM_BINS + bins,
            gl.log(weight),
        )
        gl.store(row_offset_ptr + row, offset + log_total)

    @gluon.jit
    def _is_last_program(counter_ptr, num_programs):
        gl.barrier()
        return gl.atomic_add(counter_ptr, 1, sem="acq_rel") == num_programs - 1

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
            "key_0",
            "key_1",
            "stream_0",
            "stream_1",
            "vocab_size",
            "tiles_per_split",
        ]
    )
    def _synthid_hist_kernel(
        partial_ptr,  # [B, num_splits, 2**CHUNK_BITS]
        partial_max_ptr,  # [B, num_splits]
        log_tables_ptr,  # [B, NUM_CHUNKS, 2**CHUNK_BITS]
        row_offset_ptr,  # [B]
        counters_ptr,  # [B, NUM_COUNTERS]
        states_ptr,  # [B, 4]
        logits_ptr,
        logits_stride,
        skip_ptr,
        key_0,
        key_1,
        stream_0,
        stream_1,
        vocab_size,
        tiles_per_split,
        DEPTH: gl.constexpr,
        NUM_CHUNKS: gl.constexpr,
        NUM_COUNTERS: gl.constexpr,
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
        words, as integer adds give a deterministic sum.
        The row's last program to finish runs the chunk's layers.
        """
        row = gl.program_id(0).to(gl.int64)
        split = gl.program_id(1)
        num_splits = gl.num_programs(1)
        watermark = True
        if HAS_SKIP:
            watermark = gl.load(skip_ptr + row) == 0
        if watermark:
            NUM_BINS: gl.constexpr = 1 << CHUNK_BITS
            BIN_MASK: gl.constexpr = NUM_BINS - 1
            NUM_LAYERS: gl.constexpr = min(CHUNK_BITS, DEPTH - CHUNK * CHUNK_BITS)
            layout: gl.constexpr = gl.BlockedLayout([4], [32], [num_warps], [0])
            shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

            zeros = gl.zeros([NUM_BINS], gl.int32, layout)
            hist_hi = gl.allocate_shared_memory(
                gl.int32, [NUM_BINS], shared_layout, zeros
            )
            hist_mid = gl.allocate_shared_memory(
                gl.int32, [NUM_BINS], shared_layout, zeros
            )
            hist_lo = gl.allocate_shared_memory(
                gl.int32, [NUM_BINS], shared_layout, zeros
            )
            if CHUNK >= 1:
                table_0 = _load_shared_table(
                    log_tables_ptr,
                    row,
                    0,
                    True,
                    NUM_CHUNKS,
                    NUM_BINS,
                    layout,
                    shared_layout,
                )
            if CHUNK >= 2:
                table_1 = _load_shared_table(
                    log_tables_ptr,
                    row,
                    1,
                    True,
                    NUM_CHUNKS,
                    NUM_BINS,
                    layout,
                    shared_layout,
                )
            offset = float("-inf") if CHUNK == 0 else gl.load(row_offset_ptr + row)

            state_0, state_1, state_2, state_3, word_key_0, word_key_1 = _load_states(
                states_ptr + row * 4,
                key_0.to(gl.uint32),
                key_1.to(gl.uint32),
                stream_0.to(gl.uint32),
                stream_1.to(gl.uint32),
            )
            gl.barrier()
            for tile_offset in range(tiles_per_split):
                tile = split * tiles_per_split + tile_offset
                candidate = tile * (4 * GROUPS) + gl.arange(
                    0, 4 * GROUPS, layout=layout
                )
                logits = gl.load(
                    logits_ptr + row * logits_stride + candidate,
                    mask=candidate < vocab_size,
                    other=float("-inf"),
                ).to(gl.float32)
                tile_max = gl.max(logits, axis=0)
                # Masked tiles (top-k, top-p) add no mass.
                if tile_max > float("-inf"):
                    if CHUNK == 0:  # noqa: SIM102 (CHUNK is constexpr, offset is not)
                        if tile_max > offset:
                            gl.barrier()
                            hist = _load_fixed_hist(hist_hi, hist_mid, hist_lo, layout)
                            scale = gl.exp(offset - tile_max).to(gl.float64)
                            _store_fixed_hist(
                                hist_hi, hist_mid, hist_lo, gl.floor(hist * scale)
                            )
                            gl.barrier()
                            offset = tile_max
                    words = _tile_words(
                        tile,
                        state_0,
                        state_1,
                        state_2,
                        state_3,
                        word_key_0,
                        word_key_1,
                        GROUPS,
                        layout,
                    )
                    # A split may start with masked tiles, where offset is still -inf.
                    log_mass = gl.where(logits > float("-inf"), logits - offset, logits)
                    if CHUNK >= 1:
                        log_mass += table_0.gather(
                            (words & BIN_MASK).to(gl.int32), axis=0
                        )
                    if CHUNK >= 2:
                        log_mass += table_1.gather(
                            ((words >> CHUNK_BITS) & BIN_MASK).to(gl.int32), axis=0
                        )
                    # Rounds down: the 2**-48 per token lost is below fp32
                    # resolution.
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
            if _is_last_program(counters_ptr + row * NUM_COUNTERS + CHUNK, num_splits):
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

    @gluon.jit(
        do_not_specialize=[
            "key_0",
            "key_1",
            "stream_0",
            "stream_1",
            "vocab_size",
            "tiles_per_split",
        ]
    )
    def _synthid_consume_kernel(
        sampled_ptr,  # [B]
        split_argmax_ptr,  # [B, num_splits]
        split_max_ptr,  # [B, num_splits]
        counters_ptr,  # [B, NUM_COUNTERS]
        log_tables_ptr,  # [B, NUM_CHUNKS, 2**CHUNK_BITS]
        states_ptr,  # [B, 4]
        logits_ptr,
        logits_stride,
        skip_ptr,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        key_0,
        key_1,
        stream_0,
        stream_1,
        vocab_size,
        tiles_per_split,
        NUM_CHUNKS: gl.constexpr,
        NUM_COUNTERS: gl.constexpr,
        CHUNK_BITS: gl.constexpr,
        HAS_SKIP: gl.constexpr,
        USE_FP64: gl.constexpr,
        GROUPS: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        """Add the tables to one split of a row's logits and Gumbel-max it.

        The row's last program to finish reduces the splits.
        """
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
            log_tables_ptr,
            row,
            0,
            watermark,
            NUM_CHUNKS,
            NUM_BINS,
            layout,
            shared_layout,
        )
        if NUM_CHUNKS >= 2:
            table_1 = _load_shared_table(
                log_tables_ptr,
                row,
                1,
                watermark,
                NUM_CHUNKS,
                NUM_BINS,
                layout,
                shared_layout,
            )
        if NUM_CHUNKS >= 3:
            table_2 = _load_shared_table(
                log_tables_ptr,
                row,
                2,
                watermark,
                NUM_CHUNKS,
                NUM_BINS,
                layout,
                shared_layout,
            )
        gl.barrier()

        state_0, state_1, state_2, state_3, key_0, key_1 = _load_states(
            states_ptr + row * 4,
            key_0.to(gl.uint32),
            key_1.to(gl.uint32),
            stream_0.to(gl.uint32),
            stream_1.to(gl.uint32),
        )
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
            if gl.max(logits, axis=0) > float("-inf"):
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
                    watermarked = logits + table_0.gather(
                        (words & BIN_MASK).to(gl.int32), axis=0
                    )
                    if NUM_CHUNKS >= 2:
                        watermarked += table_1.gather(
                            ((words >> CHUNK_BITS) & BIN_MASK).to(gl.int32), axis=0
                        )
                    if NUM_CHUNKS >= 3:
                        watermarked += table_2.gather(
                            ((words >> (2 * CHUNK_BITS)) & BIN_MASK).to(gl.int32),
                            axis=0,
                        )
                    # Unmasked tokens that lost all their mass stay finite.
                    logits = gl.where(
                        (logits > float("-inf")) & (watermarked == float("-inf")),
                        -3.4028234663852886e38,
                        watermarked,
                    )
                # The same noise as `gumbel_noised_argmax`, whose indexed max
                # Gluon lacks.
                scores = logits.to(gl.float64) if USE_FP64 else logits
                if temp != 0.0:
                    noise = gumbel_noise(seed, pos, candidate, USE_FP64)
                    scores = gl.where(mask, scores + noise, float("-inf"))
                value = gl.max(scores, axis=0)
                token = gl.min(gl.where(scores == value, candidate, vocab_size), axis=0)
                # Strictly greater keeps the lowest token on ties, as an
                # argmax over the whole row does.
                if value > best_value:
                    best_value = value
                    best_token = token.to(gl.int64)
        gl.store(split_argmax_ptr + row * num_splits + split, best_token)
        gl.store(split_max_ptr + row * num_splits + split, best_value)
        if _is_last_program(
            counters_ptr + row * NUM_COUNTERS + NUM_COUNTERS - 1, num_splits
        ):
            # Splits are in token order: the first maximal split holds the
            # lowest maximal token.
            SPLIT_BLOCK: gl.constexpr = 32 * num_warps
            split_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])
            splits = gl.arange(0, SPLIT_BLOCK, layout=split_layout)
            split_mask = splits < num_splits
            values = gl.load(
                split_max_ptr + row * num_splits + splits,
                mask=split_mask,
                other=float("-inf"),
                cache_modifier=".cg",
            )
            tokens = gl.load(
                split_argmax_ptr + row * num_splits + splits,
                mask=split_mask,
                other=0,
                cache_modifier=".cg",
            )
            first = gl.min(
                gl.where(values == gl.max(values, axis=0), splits, SPLIT_BLOCK), axis=0
            )
            gl.store(
                sampled_ptr + row, gl.sum(gl.where(splits == first, tokens, 0), axis=0)
            )


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
) -> torch.Tensor:
    """Gumbel-max sample `logits` [B, V] after the SynthID-Text tournament.

    Draws the same noise as `gumbel_sample` on the log-probs of the torch
    tournament. Rows in `skip_mask` are sampled from `logits` unchanged. Rows
    of `logits` may be strided, but not their elements.
    """
    assert 1 <= depth <= 32
    assert logits.stride(-1) == 1 and contexts.stride(-1) == 1
    num_rows, vocab_size = logits.shape
    sampled = logits.new_empty(num_rows, dtype=torch.int64)
    if num_rows == 0:
        return sampled

    num_chunks = triton.cdiv(depth, _MAX_CHUNK_BITS)
    chunk_bits = triton.cdiv(depth, num_chunks)
    num_bins = 1 << chunk_bits
    num_tiles = triton.cdiv(vocab_size, _HIST_TILE)
    # At least two tiles per program, or reducing the partial histograms
    # costs more than the parallelism saves.
    num_splits = min(
        triton.cdiv(_PROGRAMS_PER_SM * _num_sms(logits.device), num_rows),
        triton.cdiv(num_tiles, 2),
    )
    tiles_per_split = min(
        triton.cdiv(num_tiles, num_splits), _HIST_MAX_TOKENS // _HIST_TILE
    )
    num_splits = triton.cdiv(num_tiles, tiles_per_split)
    assert num_splits <= _MAX_SPLITS, "vocabulary too large for one-pass reduction"

    partial = logits.new_empty(num_rows, num_splits, num_bins, dtype=torch.float32)
    partial_max = logits.new_empty(num_rows, num_splits, dtype=torch.float32)
    row_offset = logits.new_empty(num_rows, dtype=torch.float32)
    log_tables = logits.new_empty(num_rows, num_chunks, num_bins, dtype=torch.float32)
    states = logits.new_empty(num_rows, 4, dtype=torch.int32)
    counters = logits.new_empty(num_rows, _NUM_COUNTERS, dtype=torch.int32)
    split_argmax = logits.new_empty(num_rows, num_splits, dtype=torch.int64)
    split_max = logits.new_empty(
        num_rows, num_splits, dtype=torch.float64 if use_fp64 else torch.float32
    )

    key_args = (key & _UINT32_MASK, key >> 32)
    stream_args = (stream & _UINT32_MASK, stream >> 32)
    _synthid_prepare_kernel[(num_rows,)](
        states,
        counters,
        contexts,
        contexts.stride(0),
        *key_args,
        CONTEXT_WIDTH=contexts.shape[-1],
        NUM_COUNTERS=_NUM_COUNTERS,
    )
    grid = (num_rows, num_splits)
    for chunk in range(num_chunks):
        _synthid_hist_kernel[grid](
            partial,
            partial_max,
            log_tables,
            row_offset,
            counters,
            states,
            logits,
            logits.stride(0),
            skip_mask,
            *key_args,
            *stream_args,
            vocab_size,
            tiles_per_split,
            DEPTH=depth,
            NUM_CHUNKS=num_chunks,
            NUM_COUNTERS=_NUM_COUNTERS,
            CHUNK_BITS=chunk_bits,
            CHUNK=chunk,
            HAS_SKIP=skip_mask is not None,
            GROUPS=_HIST_GROUPS,
            num_warps=16,
        )
    _synthid_consume_kernel[grid](
        sampled,
        split_argmax,
        split_max,
        counters,
        log_tables,
        states,
        logits,
        logits.stride(0),
        skip_mask,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        *key_args,
        *stream_args,
        vocab_size,
        tiles_per_split,
        NUM_CHUNKS=num_chunks,
        NUM_COUNTERS=_NUM_COUNTERS,
        CHUNK_BITS=chunk_bits,
        HAS_SKIP=skip_mask is not None,
        USE_FP64=use_fp64,
        GROUPS=_HIST_GROUPS,
        num_warps=_CONSUME_WARPS,
    )
    return sampled
