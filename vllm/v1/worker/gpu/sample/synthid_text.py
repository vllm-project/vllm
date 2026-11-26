# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SynthID-Text tournament sampling.

A token's tournament multiplier depends only on its g-bit pattern, so the layers
run chunk by chunk on per-row histograms of the probability mass over the
`2**chunk_bits` patterns of the chunk, reweighted by the earlier chunks. Each
chunk takes one pass over the vocabulary; a last pass adds the per-pattern log
multipliers to the logits and samples with the Gumbel noise of `gumbel_sample`.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK as _UINT32_MASK_VALUE
from vllm.v1.worker.gpu.sample.gumbel import (
    _gumbel_combine_kernel,
    gumbel_noised_argmax,
)
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
    _select_max,
)

_MAX_CHUNK_BITS = 11
_TILE = 2048
# A fixed split keeps every row's reductions independent of the batch.
_TILES_PER_PROGRAM = 4
# Mass is summed as integers in units of 2**-40 of the row max: the sums are
# exact in any order, and the tokens it drops hold at most vocab * 2**-41.
_MASS_SCALE = tl.constexpr(2.0**40)


@triton.jit
def _tile_words(tile, state_0, state_1, state_2, state_3, key_0, key_1, TILE):
    groups = tile * (TILE // 4) + tl.arange(0, TILE // 4)
    word_0, word_1, word_2, word_3 = _philox_candidate_words(
        groups, state_0, state_1, state_2, state_3, key_0, key_1
    )
    return tl.interleave(tl.interleave(word_0, word_2), tl.interleave(word_1, word_3))


@triton.jit
def _patterns(words, CHUNK: tl.constexpr, CHUNK_BITS: tl.constexpr):
    return ((words >> (CHUNK * CHUNK_BITS)) & ((1 << CHUNK_BITS) - 1)).to(tl.int32)


@triton.jit
def _log_multiplier_kernel(
    log_multipliers_ptr,
    mass_ptr,
    skip_mask_ptr,
    CHUNK: tl.constexpr,
    CHUNK_BITS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    DEPTH: tl.constexpr,
):
    """Run the chunk's layers on the row's pattern histogram."""
    row = tl.program_id(0).to(tl.int64)
    if tl.load(skip_mask_ptr + row):
        return
    offset = (row * NUM_CHUNKS + CHUNK) * (1 << CHUNK_BITS)
    patterns = tl.arange(0, 1 << CHUNK_BITS)
    mass = tl.load(mass_ptr + offset + patterns).to(tl.float32)
    multiplier = tl.full([1 << CHUNK_BITS], 1.0, tl.float32)
    for bit in tl.static_range(CHUNK_BITS):
        if CHUNK * CHUNK_BITS + bit < DEPTH:
            g = ((patterns >> bit) & 1) != 0
            layer_mass = mass * multiplier
            zero = tl.sum(tl.where(g, 0.0, layer_mass), axis=0)
            one = tl.sum(tl.where(g, layer_mass, 0.0), axis=0)
            # 1 + g - G with G = one / (zero + one), written so that a G that
            # rounds to 1 cannot make the g = 0 multipliers negative.
            multiplier *= g.to(tl.float32) + zero / (zero + one)
    tl.store(log_multipliers_ptr + offset + patterns, tl.log(multiplier))


@triton.jit
def _gather_log_multiplier(
    log_multipliers_ptr, words, skip, CHUNK: tl.constexpr, CHUNK_BITS: tl.constexpr
):
    table = tl.load(
        log_multipliers_ptr + CHUNK * (1 << CHUNK_BITS) + tl.arange(0, 1 << CHUNK_BITS),
        mask=skip == 0,
        other=0.0,
    )
    return tl.gather(table, _patterns(words, CHUNK, CHUNK_BITS), 0)


@triton.jit
def _row_max_kernel(
    row_max_ptr,
    mass_ptr,
    logits_ptr,
    logits_stride,
    skip_mask_ptr,
    vocab_size,
    MASS_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    program = tl.program_id(1)
    if tl.load(skip_mask_ptr + row):
        return
    offsets = tl.arange(0, TILE)
    row_max = float("-inf")
    for tile_offset in tl.static_range(TILES_PER_PROGRAM):
        tokens = (program * TILES_PER_PROGRAM + tile_offset) * TILE + offsets
        logits = tl.load(
            logits_ptr + row * logits_stride + tokens,
            mask=tokens < vocab_size,
            other=float("-inf"),
        )
        row_max = tl.maximum(row_max, tl.max(logits.to(tl.float32), axis=0))
    tl.store(row_max_ptr + row * tl.num_programs(1) + program, row_max)
    if program == 0:
        for start in tl.static_range(0, MASS_SIZE, TILE):
            mass = start + offsets
            tl.store(mass_ptr + row * MASS_SIZE + mass, 0, mask=mass < MASS_SIZE)


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _synthid_text_kernel(
    local_argmax_ptr,
    local_max_ptr,
    mass_ptr,
    log_multipliers_ptr,
    row_max_ptr,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    skip_mask_ptr,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    DEPTH: tl.constexpr,
    CHUNK_BITS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    PROGRAMS_POW2: tl.constexpr,
    TILE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """Add the mass of chunk CHUNK, or sample once CHUNK == NUM_CHUNKS."""
    row = tl.program_id(0).to(tl.int64)
    program = tl.program_id(1)
    num_programs = tl.num_programs(1)
    skip = tl.load(skip_mask_ptr + row)
    if CHUNK < NUM_CHUNKS:  # noqa: SIM102
        if skip:
            return

    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )
    if CHUNK < NUM_CHUNKS:
        programs = tl.arange(0, PROGRAMS_POW2)
        row_max = tl.max(
            tl.load(
                row_max_ptr + row * num_programs + programs,
                mask=programs < num_programs,
                other=float("-inf"),
            ),
            axis=0,
        )
    row_mass_ptr = mass_ptr + row * (NUM_CHUNKS << CHUNK_BITS)
    row_log_multipliers_ptr = log_multipliers_ptr + row * (NUM_CHUNKS << CHUNK_BITS)
    if CHUNK == NUM_CHUNKS:
        req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
        valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=valid_req, other=0.0)
        seed = tl.load(seeds_ptr + req_state_idx, mask=valid_req, other=0)
        pos = tl.load(pos_ptr + row)

    for tile_offset in tl.static_range(TILES_PER_PROGRAM):
        tile = program * TILES_PER_PROGRAM + tile_offset
        tokens = tile * TILE + tl.arange(0, TILE)
        mask = tokens < vocab_size
        logits = tl.load(
            logits_ptr + row * logits_stride + tokens,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        words = _tile_words(
            tile, state_0, state_1, state_2, state_3, key_0, key_1, TILE
        )
        log_multiplier = tl.zeros([TILE], tl.float32)
        for chunk in tl.static_range(CHUNK):
            log_multiplier += _gather_log_multiplier(
                row_log_multipliers_ptr, words, skip, chunk, CHUNK_BITS
            )

        if CHUNK < NUM_CHUNKS:
            mass = tl.exp(logits + log_multiplier - row_max) * _MASS_SCALE + 0.5
            mass = mass.to(tl.int64)
            patterns = CHUNK * (1 << CHUNK_BITS) + _patterns(words, CHUNK, CHUNK_BITS)
            tl.atomic_add(row_mass_ptr + patterns, mass, mask=mass != 0, sem="relaxed")
        else:
            value, index = gumbel_noised_argmax(
                logits + log_multiplier,
                tokens,
                mask,
                seed,
                pos,
                temp,
                IS_DRAFTING=False,
                USE_FP64=USE_FP64,
                APPLY_TEMPERATURE=False,
            )
            if tile_offset == 0:
                best_value = value
                best_token = tile * TILE + index
            else:
                best_value, best_token = _select_max(
                    best_value, best_token, value, tile * TILE + index
                )

    if CHUNK == NUM_CHUNKS:
        tl.store(local_argmax_ptr + row * num_programs + program, best_token)
        tl.store(local_max_ptr + row * num_programs + program, best_value)


def synthid_text_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    depth: int,
    skip_mask: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Sample each row with `depth` tournament layers, or plainly where skipped.

    `logits` are processed (temperature, top-k/top-p already applied); skipped
    rows get the token `gumbel_sample` draws from them.
    """
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    skip_mask = skip_mask.contiguous()
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    positions = positions.contiguous()
    num_tokens, vocab_size = logits.shape
    num_chunks = triton.cdiv(depth, _MAX_CHUNK_BITS)
    chunk_bits = triton.cdiv(depth, num_chunks)
    num_programs = triton.cdiv(vocab_size, _TILE * _TILES_PER_PROGRAM)
    grid = (num_tokens, num_programs)

    row_max = logits.new_empty(num_tokens, num_programs, dtype=torch.float32)
    mass = logits.new_empty(num_tokens, num_chunks << chunk_bits, dtype=torch.int64)
    log_multipliers = torch.empty_like(mass, dtype=torch.float32)
    _row_max_kernel[grid](
        row_max,
        mass,
        logits,
        logits.stride(0),
        skip_mask,
        vocab_size,
        MASS_SIZE=num_chunks << chunk_bits,
        TILE=_TILE,
        TILES_PER_PROGRAM=_TILES_PER_PROGRAM,
    )
    local_argmax = logits.new_empty(num_tokens, num_programs, dtype=torch.int64)
    local_max = logits.new_empty(
        num_tokens,
        num_programs,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    for chunk in range(num_chunks + 1):
        _synthid_text_kernel[grid](
            local_argmax,
            local_max,
            mass,
            log_multipliers,
            row_max,
            logits,
            logits.stride(0),
            contexts,
            contexts.stride(0),
            skip_mask,
            expanded_idx_mapping,
            temperatures,
            seeds,
            positions,
            key & _UINT32_MASK_VALUE,
            key >> 32,
            vocab_size,
            CONTEXT_WIDTH=contexts.shape[-1],
            DEPTH=depth,
            CHUNK_BITS=chunk_bits,
            NUM_CHUNKS=num_chunks,
            CHUNK=chunk,
            PROGRAMS_POW2=triton.next_power_of_2(num_programs),
            TILE=_TILE,
            TILES_PER_PROGRAM=_TILES_PER_PROGRAM,
            USE_FP64=use_fp64,
            num_warps=8,
        )
        if chunk < num_chunks:
            _log_multiplier_kernel[(num_tokens,)](
                log_multipliers,
                mass,
                skip_mask,
                CHUNK=chunk,
                CHUNK_BITS=chunk_bits,
                NUM_CHUNKS=num_chunks,
                DEPTH=depth,
                num_warps=1,
            )
    sampled = logits.new_empty(num_tokens, dtype=torch.int64)
    _gumbel_combine_kernel[(num_tokens,)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        sampled,
        num_programs,
        PADDED_NUM_BLOCKS=triton.next_power_of_2(num_programs),
    )
    return sampled
