# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernels for the binomial-score (chi-squared and perplexity) watermarks."""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.watermarking.prfs.philox import _UINT32_MASK
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax
from vllm.v1.worker.gpu.sample.watermark import (
    _philox_candidate_words,
    _philox_context_state,
    _uint32_to_uniform,
)


@triton.jit
def _philox_uniforms(groups, state_0, state_1, state_2, state_3, key_0, key_1):
    """Detector-compatible uniforms of tokens 4g, ..., 4g + 3 of each group g,
    in token order."""
    output_0, output_1, output_2, output_3 = _philox_candidate_words(
        groups, state_0, state_1, state_2, state_3, key_0, key_1
    )
    return _uint32_to_uniform(
        tl.interleave(
            tl.interleave(output_0, output_2),
            tl.interleave(output_1, output_3),
        )
    )


# Each program covers one segment of a row's vocabulary.
_SEGMENT_SIZE = 8192


@triton.jit
def _logsumexp_partials_kernel(
    max_ptr,
    sum_ptr,
    logits_ptr,
    logits_stride,
    vocab_size,
    segment_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Per segment of a row: the largest logit and the sum of exp(logit - it)."""
    row = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1)
    segment_start = segment * segment_size
    segment_end = tl.minimum(segment_start + segment_size, vocab_size)
    running_max = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for block_start in range(segment_start, segment_end, BLOCK_SIZE):
        candidate = block_start + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + row * logits_stride + candidate,
            mask=candidate < segment_end,
            other=float("-inf"),
        )
        logits = tl.where(logits == logits, logits, float("-inf"))
        new_max = tl.maximum(running_max, logits)
        # Lanes without a finite logit yet keep a zero sum (-inf - -inf is NaN).
        running_sum = tl.where(
            new_max > float("-inf"),
            running_sum * tl.exp(running_max - new_max) + tl.exp(logits - new_max),
            0.0,
        )
        running_max = new_max
    segment_max = tl.max(running_max, axis=0)
    segment_sum = tl.sum(
        tl.where(
            running_max > float("-inf"),
            running_sum * tl.exp(running_max - segment_max),
            0.0,
        ),
        axis=0,
    )
    offset = row * tl.num_programs(1) + segment
    tl.store(max_ptr + offset, segment_max)
    tl.store(sum_ptr + offset, segment_sum)


def _logsumexp_partials(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, vocab_size = logits.shape
    num_segments = triton.cdiv(vocab_size, _SEGMENT_SIZE)
    maxima = logits.new_empty(num_rows, num_segments)
    sums = logits.new_empty(num_rows, num_segments)
    if num_rows > 0:
        _logsumexp_partials_kernel[(num_rows, num_segments)](
            maxima,
            sums,
            logits,
            logits.stride(0),
            vocab_size,
            _SEGMENT_SIZE,
            BLOCK_SIZE=1024,
        )
    return maxima, sums


@triton.jit
def _row_logsumexp(max_ptr, sum_ptr, row, num_segments, BLOCK_SEGMENTS: tl.constexpr):
    """The row's log-normalizer from the partials of _logsumexp_partials_kernel."""
    segments = tl.arange(0, BLOCK_SEGMENTS)
    valid = segments < num_segments
    maxima = tl.load(
        max_ptr + row * num_segments + segments, mask=valid, other=float("-inf")
    )
    sums = tl.load(sum_ptr + row * num_segments + segments, mask=valid, other=0.0)
    row_max = tl.max(maxima, axis=0)
    total = tl.sum(
        tl.where(maxima > float("-inf"), sums * tl.exp(maxima - row_max), 0.0),
        axis=0,
    )
    return row_max + tl.log(total)


@triton.jit
def _binomial_score(
    uniforms,
    thresholds_ptr,
    NUM_TRIALS: tl.constexpr,
    LOG2_BLOCK_LEVELS: tl.constexpr,
):
    """The number of thresholds below each uniform, by binary search: the
    score s with thresholds[s - 1] < uniform <= thresholds[s]."""
    score = tl.zeros(uniforms.shape, dtype=tl.int32)
    for i in tl.static_range(LOG2_BLOCK_LEVELS):
        probe = score + (1 << (LOG2_BLOCK_LEVELS - 1 - i))
        threshold = tl.load(
            thresholds_ptr + probe - 1, mask=probe <= NUM_TRIALS, other=float("inf")
        )
        score = tl.where(threshold < uniforms, probe, score)
    return score


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_score_levels_kernel(
    mass_ptr,
    log_prob_ptr,
    token_ids_ptr,
    top_score_ptr,
    least_log_prob_ptr,
    least_token_ids_ptr,
    plogp_ptr,
    plain_max_ptr,
    plain_token_ids_ptr,
    skip_mask_ptr,
    lse_ptr,
    levels_stride_0,
    levels_stride_1,
    logits_ptr,
    logits_stride,
    lse_max_ptr,
    lse_sum_ptr,
    cap_ptr,
    contexts_ptr,
    context_stride,
    thresholds_ptr,
    expanded_idx_mapping_ptr,
    temperatures_ptr,
    seeds_ptr,
    positions_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    segment_size,
    NUM_TRIALS: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr,
    LOG2_BLOCK_LEVELS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    PLOGP_FLOOR: tl.constexpr,
    PLOGP_SUPPORT_ONLY: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """One pass over a segment of a row's logits; outputs and inputs left as
    None are skipped.

    The row's log-normalizer, stored in ``lse_ptr``, is rebuilt from the
    partials of ``_logsumexp_partials_kernel``. Per segment, the kernel
    finds the probability mass and the most probable token of each score, the
    highest score and its least probable token, the sum of
    ``max(p, PLOGP_FLOOR) * log(max(p, PLOGP_FLOOR))`` over the tokens with
    probability (``PLOGP_SUPPORT_ONLY``) or over all tokens, and the token
    ``gumbel_sample`` would draw from the logits, with its noised logit, for
    the rows in ``skip_mask_ptr``. Tokens whose input is below
    ``cap_ptr[row]`` are left out of the most probable token of each score.
    """
    row = tl.program_id(0).to(tl.int64)
    segment = tl.program_id(1)
    num_segments = tl.num_programs(1)
    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )
    levels = tl.arange(0, BLOCK_LEVELS)
    lse = _row_logsumexp(lse_max_ptr, lse_sum_ptr, row, num_segments, BLOCK_SEGMENTS)
    tl.store(lse_ptr + row, lse, mask=segment == 0)
    cap = float("-inf")
    if cap_ptr is not None:
        cap = tl.load(cap_ptr + row)
    if plain_max_ptr is not None:
        sample_plain = tl.load(skip_mask_ptr + row)
        if USE_FP64:
            plain_max = tl.full((), float("-inf"), dtype=tl.float64)
        else:
            plain_max = tl.full((), float("-inf"), dtype=tl.float32)
        plain_token = tl.zeros((), dtype=tl.int64)

    # Accumulate per (lane, level) and reduce over lanes once at the end.
    mass = tl.zeros((BLOCK_SIZE, BLOCK_LEVELS), dtype=tl.float32)
    best = tl.full((BLOCK_SIZE, BLOCK_LEVELS), float("-inf"), dtype=tl.float32)
    best_token = tl.zeros((BLOCK_SIZE, BLOCK_LEVELS), dtype=tl.int32)
    # Per lane: the highest score seen and its least probable token.
    lane_top = tl.full((BLOCK_SIZE,), -1, dtype=tl.int32)
    lane_least = tl.full((BLOCK_SIZE,), float("inf"), dtype=tl.float32)
    lane_least_token = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    plogp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    segment_start = segment * segment_size
    segment_end = tl.minimum(segment_start + segment_size, vocab_size)
    for block_start in range(segment_start, segment_end, BLOCK_SIZE):
        candidate = block_start + tl.arange(0, BLOCK_SIZE)
        mask = candidate < segment_end
        logits = tl.load(
            logits_ptr + row * logits_stride + candidate,
            mask=mask,
            other=float("-inf"),
        )
        # The outer test is resolved at compile time, the inner one per row.
        if plain_max_ptr is not None:  # noqa: SIM102
            if sample_plain:
                # The noise of gumbel_sample, so the token is the one it would
                # draw.
                noised, index = gumbel_block_argmax(
                    logits,
                    candidate,
                    mask,
                    row,
                    expanded_idx_mapping_ptr,
                    temperatures_ptr,
                    seeds_ptr,
                    positions_ptr,
                    None,  # logits_cache_ptr
                    0,  # logits_cache_stride_0
                    0,  # logits_cache_stride_1
                    None,  # logits_cache_col_ptr
                    None,  # logits_cache_source_ptr
                    0,  # logits_cache_source_stride
                    vocab_size,
                    IS_DRAFTING=False,
                    APPLY_TEMPERATURE=False,
                    USE_FP64=USE_FP64,
                )
                # Strictly greater: blocks come in ascending token ids.
                better_plain = noised > plain_max
                plain_max = tl.where(better_plain, noised, plain_max)
                plain_token = tl.where(
                    better_plain, (block_start + index).to(tl.int64), plain_token
                )
        logits = tl.where(logits == logits, logits, float("-inf"))
        selectable = logits >= cap
        log_probs = logits - lse
        uniforms = _philox_uniforms(
            block_start // 4 + tl.arange(0, BLOCK_SIZE // 4),
            state_0,
            state_1,
            state_2,
            state_3,
            key_0,
            key_1,
        )
        # Lanes past the segment have -inf log-probabilities, so their scores
        # change nothing.
        score = _binomial_score(uniforms, thresholds_ptr, NUM_TRIALS, LOG2_BLOCK_LEVELS)
        at_level = score[:, None] == levels[None, :]
        if mass_ptr is not None:
            mass += tl.where(at_level, tl.exp(log_probs)[:, None], 0.0)
        if log_prob_ptr is not None:
            # Strictly greater: each lane meets ascending token ids.
            selected = tl.where(selectable, log_probs, float("-inf"))
            better = at_level & (selected[:, None] > best)
            best = tl.where(better, selected[:, None], best)
            best_token = tl.where(better, candidate[:, None], best_token)
        if top_score_ptr is not None:
            finite = log_probs > float("-inf")
            higher = finite & (score > lane_top)
            # Strictly less: each lane meets ascending token ids.
            less_probable = finite & (score == lane_top) & (log_probs < lane_least)
            take = higher | less_probable
            lane_top = tl.where(higher, score, lane_top)
            lane_least = tl.where(take, log_probs, lane_least)
            lane_least_token = tl.where(take, candidate, lane_least_token)
        if plogp_ptr is not None:
            p = tl.maximum(tl.exp(log_probs), PLOGP_FLOOR)
            counted = mask & (log_probs > float("-inf")) if PLOGP_SUPPORT_ONLY else mask
            plogp += tl.where(counted, p * tl.log(p), 0.0)

    levels_offset = row * levels_stride_0 + segment * levels_stride_1 + levels
    level_mask = levels <= NUM_TRIALS
    segment_offset = row * num_segments + segment
    if mass_ptr is not None:
        tl.store(mass_ptr + levels_offset, tl.sum(mass, axis=0), mask=level_mask)
    if log_prob_ptr is not None:
        level_best = tl.max(best, axis=0)
        token_id = tl.min(
            tl.where(best == level_best[None, :], best_token, vocab_size), axis=0
        )
        # A level without finite log-probabilities reports token 0.
        token_id = tl.where(level_best > float("-inf"), token_id, 0)
        tl.store(log_prob_ptr + levels_offset, level_best, mask=level_mask)
        tl.store(token_ids_ptr + levels_offset, token_id, mask=level_mask)
    if top_score_ptr is not None:
        top_score = tl.max(lane_top, axis=0)
        at_top = lane_top == top_score
        least = tl.min(tl.where(at_top, lane_least, float("inf")), axis=0)
        least_token = tl.min(
            tl.where(at_top & (lane_least == least), lane_least_token, vocab_size),
            axis=0,
        )
        tl.store(top_score_ptr + segment_offset, top_score)
        tl.store(least_log_prob_ptr + segment_offset, least)
        tl.store(least_token_ids_ptr + segment_offset, least_token)
    if plogp_ptr is not None:
        tl.store(plogp_ptr + segment_offset, tl.sum(plogp, axis=0))
    if plain_max_ptr is not None:
        tl.store(plain_max_ptr + segment_offset, plain_max)
        tl.store(plain_token_ids_ptr + segment_offset, plain_token)


class _ScoreLevelStats(NamedTuple):
    """Per-segment outputs of _philox_score_levels_kernel: [num_rows,
    num_segments, num_levels] for the levels, [num_rows, num_segments] for the
    rest, and [num_rows] for the log-normalizer; None when not requested."""

    mass: torch.Tensor | None
    log_probs: torch.Tensor | None
    token_ids: torch.Tensor | None
    top_score: torch.Tensor | None
    least: torch.Tensor | None
    least_token_ids: torch.Tensor | None
    plogp: torch.Tensor | None
    plain_max: torch.Tensor | None
    plain_token_ids: torch.Tensor | None
    lse: torch.Tensor


def _launch_score_levels(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    with_mass: bool,
    with_best: bool,
    with_least_probable_top: bool = False,
    *,
    cap: torch.Tensor | None = None,
    plogp_floor: float | None = None,
    plogp_support_only: bool = False,
    skip_mask: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> _ScoreLevelStats:
    """Run _philox_score_levels_kernel on float32 ``logits``; the rows in
    ``skip_mask`` get the plain Gumbel sample, drawn with the given sampler
    inputs."""
    assert logits.dtype == torch.float32
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    num_rows, vocab_size = logits.shape
    num_levels = len(thresholds) + 1
    block_levels = triton.next_power_of_2(num_levels)
    num_segments = triton.cdiv(vocab_size, _SEGMENT_SIZE)
    shape = (num_rows, num_segments, num_levels)
    mass = logits.new_empty(shape) if with_mass else None
    best = logits.new_empty(shape) if with_best else None
    token_ids = logits.new_empty(shape, dtype=torch.int32) if with_best else None
    top_score = least = least_token_ids = None
    if with_least_probable_top:
        top_score = logits.new_empty(shape[:2], dtype=torch.int32)
        least = logits.new_empty(shape[:2])
        least_token_ids = logits.new_empty(shape[:2], dtype=torch.int32)
    plogp = logits.new_empty(shape[:2]) if plogp_floor is not None else None
    plain_max = plain_token_ids = None
    if skip_mask is not None:
        assert expanded_idx_mapping is not None and positions is not None
        skip_mask = skip_mask.contiguous()
        expanded_idx_mapping = expanded_idx_mapping.contiguous()
        positions = positions.contiguous()
        plain_max = logits.new_empty(
            shape[:2], dtype=torch.float64 if use_fp64 else torch.float32
        )
        plain_token_ids = logits.new_empty(shape[:2], dtype=torch.int64)
    lse_max, lse_sum = _logsumexp_partials(logits)
    lse = logits.new_empty(num_rows)
    if num_rows > 0:
        _philox_score_levels_kernel[(num_rows, num_segments)](
            mass,
            best,
            token_ids,
            top_score,
            least,
            least_token_ids,
            plogp,
            plain_max,
            plain_token_ids,
            skip_mask,
            lse,
            num_segments * num_levels,
            num_levels,
            logits,
            logits.stride(0),
            lse_max,
            lse_sum,
            cap,
            contexts,
            contexts.stride(0),
            thresholds,
            expanded_idx_mapping,
            temperatures,
            seeds,
            positions,
            key & _UINT32_MASK,
            key >> 32,
            vocab_size,
            _SEGMENT_SIZE,
            NUM_TRIALS=num_levels - 1,
            CONTEXT_WIDTH=contexts.shape[-1],
            BLOCK_SIZE=128,
            BLOCK_LEVELS=block_levels,
            LOG2_BLOCK_LEVELS=block_levels.bit_length() - 1,
            BLOCK_SEGMENTS=triton.next_power_of_2(num_segments),
            PLOGP_FLOOR=0.0 if plogp_floor is None else plogp_floor,
            PLOGP_SUPPORT_ONLY=plogp_support_only,
            USE_FP64=use_fp64,
        )
    return _ScoreLevelStats(
        mass,
        best,
        token_ids,
        top_score,
        least,
        least_token_ids,
        plogp,
        plain_max,
        plain_token_ids,
        lse,
    )


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_level_bias_kernel(
    output_ptr,
    output_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    level_bias_ptr,
    level_bias_stride,
    skip_mask_ptr,
    thresholds_ptr,
    key_0_value,
    key_1_value,
    vocab_size,
    NUM_TRIALS: tl.constexpr,
    LOG2_BLOCK_LEVELS: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size
    logits = tl.load(
        logits_ptr + row * logits_stride + candidate, mask=mask, other=0.0
    ).to(tl.float32)
    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )
    uniforms = _philox_uniforms(
        block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4),
        state_0,
        state_1,
        state_2,
        state_3,
        key_0,
        key_1,
    )
    scores = _binomial_score(uniforms, thresholds_ptr, NUM_TRIALS, LOG2_BLOCK_LEVELS)
    bias = tl.load(level_bias_ptr + row * level_bias_stride + scores)
    if skip_mask_ptr is not None:
        bias = tl.where(tl.load(skip_mask_ptr + row) == 0, bias, 0.0)
    tl.store(output_ptr + row * output_stride + candidate, logits + bias, mask=mask)


def philox_level_bias(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    level_bias: torch.Tensor,
    *,
    skip_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add ``level_bias[row, score]`` to each token's logit, where ``score`` is
    its detector-compatible binomial score.

    Returns float32 logits; rows in ``skip_mask`` are returned unchanged.
    """
    assert level_bias.dtype == torch.float32
    assert level_bias.shape == (logits.shape[0], len(thresholds) + 1)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    if level_bias.stride(-1) != 1:
        level_bias = level_bias.contiguous()
    if skip_mask is not None:
        skip_mask = skip_mask.contiguous()
    num_rows, vocab_size = logits.shape
    output = logits.new_empty(num_rows, vocab_size, dtype=torch.float32)
    if num_rows == 0:
        return output
    block_size = 1024
    _philox_level_bias_kernel[(num_rows, triton.cdiv(vocab_size, block_size))](
        output,
        output.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        level_bias,
        level_bias.stride(0),
        skip_mask,
        thresholds,
        key & _UINT32_MASK,
        key >> 32,
        vocab_size,
        NUM_TRIALS=len(thresholds),
        LOG2_BLOCK_LEVELS=triton.next_power_of_2(len(thresholds) + 1).bit_length() - 1,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
    )
    return output


@triton.jit
def _log1m(r):
    # log(1 - r) without cancellation when r is small (Goldberg's log1p).
    # Unlike gumbel's _log1p_neg_stable it does not clamp: r = 1 gives -inf,
    # as in first_score_ranks.
    u = 1.0 - r
    return tl.where(u == 1.0, -r, tl.log(u) * (-r / (u - 1.0)))


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _monte_carlo_bisection_kernel(
    beta_ptr,
    nodes_ptr,
    nodes_stride_row,
    nodes_stride_draw,
    nodes_stride_level,
    ranked_ptr,
    ranked_stride,
    width,
    contexts_ptr,
    context_stride,
    key_0_value,
    key_1_value,
    level_probs_ptr,
    target_ptr,
    eps,
    beta_high_value,
    N_MC: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_ITER: tl.constexpr,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_MC: tl.constexpr,
):
    # One program per row. Each lane follows its own draws, so the work over
    # score levels stays in registers; the phases pass data through ``nodes``.
    row = tl.program_id(0).to(tl.int64)
    draws = tl.arange(0, BLOCK_MC)
    valid = draws < N_MC
    nodes = nodes_ptr + row * nodes_stride_row + draws * nodes_stride_draw
    ranked = ranked_ptr + row * ranked_stride
    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )

    # The rank where each score first appears, as in first_score_ranks.
    seen = tl.zeros([BLOCK_MC], dtype=tl.int64)
    rank = tl.zeros([BLOCK_MC], dtype=tl.float32) - 1.0
    for step in range(NUM_LEVELS):
        remaining = tl.zeros([BLOCK_MC], dtype=tl.float32)
        for level in tl.static_range(NUM_LEVELS):
            unseen = ((seen >> level) & 1) == 0
            remaining += tl.where(unseen, tl.load(level_probs_ptr + level), 0.0)
        remaining = tl.minimum(remaining, 1.0)
        # Uniforms 2 * step and 2 * step + 1 of each draw, as PRF tokens
        # draw * 2 * NUM_LEVELS + 2 * step and the next: one Philox output.
        uniform_index = draws * (2 * NUM_LEVELS) + 2 * step
        word_0, word_1, word_2, word_3 = _philox_candidate_words(
            uniform_index // 4, state_0, state_1, state_2, state_3, key_0, key_1
        )
        low_words = uniform_index % 4 == 0
        u_skip = _uint32_to_uniform(tl.where(low_words, word_0, word_2))
        u_level = _uint32_to_uniform(tl.where(low_words, word_1, word_3))
        skip = tl.log(u_skip) / _log1m(remaining)
        skip = tl.where((skip > 1.0) & (step > 0), tl.ceil(skip), 1.0)
        rank = tl.minimum(rank + skip, width)
        threshold = u_level * remaining
        cumulative = tl.zeros([BLOCK_MC], dtype=tl.float32)
        chosen = tl.full([BLOCK_MC], -1, dtype=tl.int32)
        last = tl.full([BLOCK_MC], -1, dtype=tl.int32)
        for level in tl.static_range(NUM_LEVELS):
            unseen = ((seen >> level) & 1) == 0
            cumulative += tl.where(unseen, tl.load(level_probs_ptr + level), 0.0)
            passed = unseen & (chosen < 0) & (cumulative > threshold)
            chosen = tl.where(passed, level, chosen)
            last = tl.where(unseen, level, last)
        chosen = tl.where(chosen < 0, last, chosen)
        seen = seen | (tl.full([BLOCK_MC], 1, dtype=tl.int64) << chosen.to(tl.int64))
        tl.store(nodes + chosen * nodes_stride_level, rank, mask=valid)
    tl.debug_barrier()

    # The log-probabilities at those ranks; -inf past the ranked tokens.
    for level in tl.static_range(NUM_LEVELS):
        r = tl.load(nodes + level * nodes_stride_level, mask=valid, other=width)
        node = tl.load(
            ranked + r.to(tl.int32), mask=valid & (r < width), other=float("-inf")
        )
        node = tl.where(
            node > float("-inf"),
            tl.log(tl.maximum(tl.exp(node), 1e-12)),
            float("-inf"),
        )
        tl.store(nodes + level * nodes_stride_level, node, mask=valid)
    tl.debug_barrier()

    # The reference's bisection. An iteration that moves neither bound leaves
    # the state unchanged, so stopping there returns what MAX_ITER would.
    target = tl.load(target_ptr + row)
    beta_low = tl.zeros((), dtype=tl.float32)
    beta_high = tl.zeros((), dtype=tl.float32) + beta_high_value
    iteration = tl.zeros((), dtype=tl.int32)
    moving = tl.full((), 1, dtype=tl.int1)
    while (iteration < MAX_ITER) & moving:
        mid = (beta_low + beta_high) * 0.5
        best = tl.full([BLOCK_MC], float("-inf"), dtype=tl.float32)
        selected = tl.zeros([BLOCK_MC], dtype=tl.float32)
        for level in tl.static_range(NUM_LEVELS):
            node = tl.load(
                nodes + level * nodes_stride_level, mask=valid, other=float("-inf")
            )
            s = tl.where(node > float("-inf"), level + mid * node, float("-inf"))
            selected = tl.where(s > best, node, selected)
            best = tl.maximum(s, best)
        val = tl.sum(tl.where(valid, selected, 0.0), axis=0) / N_MC - target + eps
        new_high = tl.where(val > 0, mid, beta_high)
        new_low = tl.where(val < 0, mid, beta_low)
        moving = (new_high != beta_high) | (new_low != beta_low)
        beta_high = new_high
        beta_low = new_low
        iteration += 1
    tl.store(beta_ptr + row, (beta_low + beta_high) * 0.5)


def monte_carlo_bisection(
    ranked_log_probs: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    n_mc: int,
    level_probs: torch.Tensor,
    target: torch.Tensor,
    eps: float,
    beta_max: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel version of ``first_score_ranks`` followed by ``_bisection_logic``
    in ``vllm.v1.watermarking.perplexity``, on the uniforms of
    ``PPLSoftWatermarker.monte_carlo_uniforms``.

    Args:
        ranked_log_probs: float32 [num_rows, width], each row's largest
            log-probabilities in decreasing order
        contexts: [num_rows, context_width], keying each row's draws
        key: the Philox key of the draws
        n_mc: the number of draws per row
        level_probs: float32 [num_levels], the probability of each score
        target: [num_rows], the mean log-probability sought, before ``eps``
        eps: the allowed shortfall from ``target``
        beta_max: the upper end of the bisection's range, which starts at 0
        max_iter: bisection iterations

    Returns:
        beta: [num_rows]
        nodes: [num_rows, n_mc, num_levels], the log-probability, at least
            1e-12, of the first token with each score in each draw, or -inf
            when it is not among the ranked tokens

    """
    num_rows, width = ranked_log_probs.shape
    num_levels = level_probs.shape[0]
    # The kernel tracks the scores seen in an int64 bitmask.
    assert num_levels < 64
    assert ranked_log_probs.dtype == level_probs.dtype == torch.float32
    ranked_log_probs = ranked_log_probs.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    target = target.to(torch.float32).contiguous()
    beta = target.new_empty(num_rows)
    # Draws contiguous, one per lane.
    nodes = target.new_empty(num_rows, num_levels, n_mc).transpose(1, 2)
    if num_rows == 0:
        return beta, nodes
    block_mc = triton.next_power_of_2(n_mc)
    _monte_carlo_bisection_kernel[(num_rows,)](
        beta,
        nodes,
        nodes.stride(0),
        nodes.stride(1),
        nodes.stride(2),
        ranked_log_probs,
        ranked_log_probs.stride(0),
        width,
        contexts,
        contexts.stride(0),
        key & _UINT32_MASK,
        key >> 32,
        level_probs,
        target,
        eps,
        beta_max,
        N_MC=n_mc,
        NUM_LEVELS=num_levels,
        MAX_ITER=max_iter,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_MC=block_mc,
        num_warps=min(max(block_mc // 32, 1), 8),
    )
    return beta, nodes


@triton.jit
def _reduce_segment_level_best(
    level_log_prob_ptr,
    level_token_ids_ptr,
    levels_stride_0,
    levels_stride_1,
    row,
    num_segments,
    NUM_LEVELS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The most probable token of each score over a row's segments, the
    earliest segment's on ties; lanes from NUM_LEVELS on get -inf and token 0."""
    segments = tl.arange(0, BLOCK_SEGMENTS)
    lanes = tl.arange(0, BLOCK)
    valid = (segments < num_segments)[:, None] & (lanes < NUM_LEVELS)[None, :]
    offsets = (
        row * levels_stride_0 + segments[:, None] * levels_stride_1 + lanes[None, :]
    )
    best = tl.load(level_log_prob_ptr + offsets, mask=valid, other=float("-inf"))
    tokens = tl.load(level_token_ids_ptr + offsets, mask=valid, other=0)
    level_best = tl.max(best, axis=0)
    # Ties go to the earliest segment, which holds the lowest token ids.
    first = tl.min(
        tl.where(
            valid & (best == level_best[None, :]), segments[:, None], BLOCK_SEGMENTS
        ),
        axis=0,
    )
    token_ids = tl.sum(tl.where(segments[:, None] == first[None, :], tokens, 0), axis=0)
    return level_best, token_ids


@triton.jit
def _plain_sample_token(
    plain_max_ptr,
    plain_token_ids_ptr,
    row,
    num_segments,
    BLOCK_SEGMENTS: tl.constexpr,
):
    """The plain sample from its per-segment partials: the segment with the
    largest noised logit, the earliest on ties, as gumbel_sample."""
    segments = tl.arange(0, BLOCK_SEGMENTS)
    segment_valid = segments < num_segments
    offsets = row * num_segments + segments
    plain = tl.load(plain_max_ptr + offsets, mask=segment_valid, other=float("-inf"))
    first = tl.min(
        tl.where(
            segment_valid & (plain == tl.max(plain, axis=0)), segments, BLOCK_SEGMENTS
        ),
        axis=0,
    )
    plain_tokens = tl.load(plain_token_ids_ptr + offsets, mask=segment_valid, other=0)
    return tl.sum(tl.where(segments == first, plain_tokens, 0), axis=0)


@triton.jit
def _ppl_hard_hull_kernel(
    best_q_ptr,
    token_ids_ptr,
    level_log_prob_ptr,
    level_token_ids_ptr,
    levels_stride_0,
    levels_stride_1,
    top_score_ptr,
    least_log_prob_ptr,
    least_token_ids_ptr,
    plogp_ptr,
    sampled_ptr,
    skip_mask_ptr,
    plain_max_ptr,
    plain_token_ids_ptr,
    expanded_idx_mapping_ptr,
    temperatures_ptr,
    seeds_ptr,
    positions_ptr,
    epsilon,
    vocab_size,
    num_segments,
    NUM_LEVELS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    segments = tl.arange(0, BLOCK_SEGMENTS)
    segment_valid = segments < num_segments
    segment_offsets = row * num_segments + segments
    lanes = tl.arange(0, BLOCK_CANDIDATES)
    num_candidates = NUM_LEVELS + 1

    # The most probable token of each score, and the least probable token of
    # the highest score, over the row's segments.
    level_log_probs, level_token_ids = _reduce_segment_level_best(
        level_log_prob_ptr,
        level_token_ids_ptr,
        levels_stride_0,
        levels_stride_1,
        row,
        num_segments,
        NUM_LEVELS,
        BLOCK_SEGMENTS,
        BLOCK_CANDIDATES,
    )
    top_scores = tl.load(top_score_ptr + segment_offsets, mask=segment_valid, other=-2)
    least = tl.load(
        least_log_prob_ptr + segment_offsets, mask=segment_valid, other=float("inf")
    )
    least_tokens = tl.load(
        least_token_ids_ptr + segment_offsets, mask=segment_valid, other=0
    )
    least = tl.where(
        segment_valid & (top_scores == tl.max(top_scores, axis=0)),
        least,
        float("inf"),
    )
    least_log_prob = tl.min(least, axis=0)
    least_token_id = tl.min(
        tl.where(segment_valid & (least == least_log_prob), least_tokens, vocab_size),
        axis=0,
    )
    least_token_id = tl.where(least_token_id == vocab_size, 0, least_token_id)

    # hull_candidates: the most probable token of each score, then the least
    # probable token of the highest score.
    is_level = lanes < NUM_LEVELS
    top_score = tl.max(
        tl.where(is_level & (level_log_probs > float("-inf")), lanes, -1), axis=0
    )
    top_log_prob = tl.max(
        tl.where(lanes == tl.maximum(top_score, 0), level_log_probs, float("-inf")),
        axis=0,
    )
    least_log_prob = tl.where(
        least_log_prob < top_log_prob, least_log_prob, float("-inf")
    )
    is_extra = lanes == NUM_LEVELS
    log_probs = tl.where(is_extra, least_log_prob, level_log_probs)
    scores = tl.where(is_extra, top_score, lanes)
    token_ids = tl.where(is_extra, least_token_id, level_token_ids)
    valid = lanes < num_candidates

    # convex_hull_algorithm_vectorized, in float64 as there. Sorting is
    # replaced by pairwise comparisons: before[k, a] says whether candidate a
    # comes before candidate k in the stable sort by cost.
    probs = tl.exp(log_probs.to(tl.float64))
    c = -tl.log(tl.maximum(probs, 1e-30))
    H_p = -tl.sum(
        tl.load(plogp_ptr + segment_offsets, mask=segment_valid, other=0.0), axis=0
    )
    C = (H_p + epsilon).to(tl.float64)
    # Candidates without probability are not tokens of the distribution.
    s = tl.where(probs > 0, scores, -1).to(tl.float64)
    pair = valid[:, None] & valid[None, :]
    same_cost = c[None, :] == c[:, None]
    before = pair & (
        (c[None, :] < c[:, None]) | (same_cost & (lanes[None, :] < lanes[:, None]))
    )
    after = pair & (
        (c[None, :] > c[:, None]) | (same_cost & (lanes[None, :] > lanes[:, None]))
    )
    # Pareto filtering: keep points scoring at least as high as every earlier one.
    dominated = tl.max(tl.where(before & (s[None, :] > s[:, None]), 1, 0), axis=1)
    is_pareto = valid & (s >= 0) & (dominated == 0)
    # Upper hull: the slope in from every earlier point exceeds the slope out
    # to every later point.
    both = is_pareto[:, None] & is_pareto[None, :]
    slopes = (s[:, None] - s[None, :]) / (c[:, None] - c[None, :])
    slope_in = tl.min(tl.where(both & before, slopes, float("inf")), axis=1)
    slope_out = tl.max(tl.where(both & after, slopes, float("-inf")), axis=1)
    on_hull = is_pareto & (slope_in > slope_out)

    # Find the hull segment [i, j] around C.
    hull_rank = tl.sum(tl.where(before & on_hull[None, :], 1, 0), axis=1)
    stack_ptr = tl.sum(on_hull.to(tl.int32), axis=0)
    idx_j = tl.minimum(
        tl.sum((on_hull & (c < C)).to(tl.int32), axis=0), num_candidates - 1
    )
    idx_i = tl.maximum(idx_j - 1, 0)
    # Handle C > max_cost (clamp to last valid point)
    last_valid = tl.maximum(stack_ptr - 1, 0)
    use_last = idx_j > last_valid
    idx_j = tl.where(use_last, last_valid, idx_j)
    idx_i = tl.where(use_last, last_valid, idx_i)
    at_i = on_hull & (hull_rank == idx_i)
    at_j = on_hull & (hull_rank == idx_j)
    c_i = tl.sum(tl.where(at_i, c, 0.0), axis=0)
    c_j = tl.sum(tl.where(at_j, c, 0.0), axis=0)

    denom = c_j - c_i
    alpha = (c_j - C) / (denom + 1e-30)
    alpha = tl.where(tl.abs(denom) < 1e-20, 1.0, alpha)
    alpha = tl.minimum(tl.maximum(alpha, 0.0), 1.0)
    best_q = tl.where(at_i, alpha, 0.0) + tl.where(at_j, 1.0 - alpha, 0.0)
    offsets = row * num_candidates + lanes
    tl.store(best_q_ptr + offsets, best_q, mask=valid)
    tl.store(token_ids_ptr + offsets, token_ids.to(tl.int64), mask=valid)

    if sampled_ptr is not None:
        # The token gumbel_sample would draw from log q, whose only finite
        # entries are the candidates: its noise is keyed by token id.
        _, index = gumbel_block_argmax(
            tl.log(best_q.to(tl.float32)),
            token_ids,
            valid & (best_q > 0),
            row,
            expanded_idx_mapping_ptr,
            temperatures_ptr,
            seeds_ptr,
            positions_ptr,
            None,  # logits_cache_ptr
            0,  # logits_cache_stride_0
            0,  # logits_cache_stride_1
            None,  # logits_cache_col_ptr
            None,  # logits_cache_source_ptr
            0,  # logits_cache_source_stride
            vocab_size,
            IS_DRAFTING=False,
            APPLY_TEMPERATURE=False,
            USE_FP64=USE_FP64,
        )
        token_id = tl.sum(tl.where(lanes == index, token_ids, 0), axis=0)
        plain_token = _plain_sample_token(
            plain_max_ptr, plain_token_ids_ptr, row, num_segments, BLOCK_SEGMENTS
        )
        token_id = tl.where(tl.load(skip_mask_ptr + row), plain_token, token_id)
        tl.store(sampled_ptr + row, token_id.to(tl.int64))


def _launch_ppl_hard(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    epsilon: float,
    skip_mask: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The candidates' watermarked probabilities and token ids, and, with a
    ``skip_mask``, each row's sampled token."""
    stats = _launch_score_levels(
        logits,
        contexts,
        key,
        thresholds,
        with_mass=False,
        with_best=True,
        with_least_probable_top=True,
        plogp_floor=1e-30,
        skip_mask=skip_mask,
        expanded_idx_mapping=expanded_idx_mapping,
        temperatures=temperatures,
        seeds=seeds,
        positions=positions,
        use_fp64=use_fp64,
    )
    num_rows, vocab_size = logits.shape
    num_levels = len(thresholds) + 1
    num_segments = triton.cdiv(vocab_size, _SEGMENT_SIZE)
    best_q = logits.new_empty(num_rows, num_levels + 1, dtype=torch.float64)
    token_ids = logits.new_empty(num_rows, num_levels + 1, dtype=torch.int64)
    sampled = None
    if skip_mask is not None:
        assert expanded_idx_mapping is not None and positions is not None
        sampled = logits.new_empty(num_rows, dtype=torch.int64)
        skip_mask = skip_mask.contiguous()
        expanded_idx_mapping = expanded_idx_mapping.contiguous()
        positions = positions.contiguous()
    if num_rows == 0:
        return best_q, token_ids, sampled
    assert stats.log_probs is not None and stats.token_ids is not None
    _ppl_hard_hull_kernel[(num_rows,)](
        best_q,
        token_ids,
        stats.log_probs,
        stats.token_ids,
        stats.log_probs.stride(0),
        stats.log_probs.stride(1),
        stats.top_score,
        stats.least,
        stats.least_token_ids,
        stats.plogp,
        sampled,
        skip_mask,
        stats.plain_max,
        stats.plain_token_ids,
        expanded_idx_mapping,
        temperatures,
        seeds,
        positions,
        epsilon,
        vocab_size,
        num_segments,
        NUM_LEVELS=num_levels,
        BLOCK_SEGMENTS=triton.next_power_of_2(num_segments),
        BLOCK_CANDIDATES=triton.next_power_of_2(num_levels + 1),
        USE_FP64=use_fp64,
    )
    return best_q, token_ids, sampled


def philox_ppl_hard_hull(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel version of ``hull_candidates`` and
    ``convex_hull_algorithm_vectorized`` in
    ``vllm.v1.watermarking.perplexity``, for float32 ``logits`` [num_rows,
    vocab_size]. One pass over the vocabulary finds the candidates and the
    entropy.

    Returns:
        best_q: float64 [num_rows, num_levels + 1], the watermarked
            probability of each candidate
        token_ids: [num_rows, num_levels + 1], the candidates' token ids

    """
    best_q, token_ids, _ = _launch_ppl_hard(logits, contexts, key, thresholds, epsilon)
    return best_q, token_ids


def philox_ppl_hard_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    epsilon: float,
    skip_mask: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    use_fp64: bool = False,
) -> torch.Tensor:
    """The token ``gumbel_sample`` draws, with the given sampler inputs, from
    ``PPLHardWatermarker.watermark_logits`` of float32 ``logits``, without
    building those logits: [num_rows]."""
    _, _, sampled = _launch_ppl_hard(
        logits,
        contexts,
        key,
        thresholds,
        epsilon,
        skip_mask,
        expanded_idx_mapping,
        temperatures,
        seeds,
        positions,
        use_fp64,
    )
    assert sampled is not None
    return sampled


@triton.jit
def _ppl_soft_select_kernel(
    token_ids_ptr,
    level_log_prob_ptr,
    level_token_ids_ptr,
    levels_stride_0,
    levels_stride_1,
    beta_ptr,
    plain_max_ptr,
    plain_token_ids_ptr,
    skip_mask_ptr,
    num_segments,
    NUM_LEVELS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    levels = tl.arange(0, BLOCK_LEVELS)
    level_log_probs, level_token_ids = _reduce_segment_level_best(
        level_log_prob_ptr,
        level_token_ids_ptr,
        levels_stride_0,
        levels_stride_1,
        row,
        num_segments,
        NUM_LEVELS,
        BLOCK_SEGMENTS,
        BLOCK_LEVELS,
    )
    # argmax_u scores_u + beta * log(max(p_u, 1e-12)) is the most probable
    # token of the score s maximizing s + beta * log(max(p_s, 1e-12)).
    beta = tl.load(beta_ptr + row)
    keys = levels.to(tl.float32) + beta * tl.log(
        tl.maximum(tl.exp(level_log_probs), 1e-12)
    )
    keys = tl.where(level_log_probs > float("-inf"), keys, float("-inf"))
    chosen = tl.min(
        tl.where(keys == tl.max(keys, axis=0), levels, BLOCK_LEVELS), axis=0
    )
    token_id = tl.sum(tl.where(levels == chosen, level_token_ids, 0), axis=0)
    token_id = token_id.to(tl.int64)
    if skip_mask_ptr is not None:
        plain_token = _plain_sample_token(
            plain_max_ptr, plain_token_ids_ptr, row, num_segments, BLOCK_SEGMENTS
        )
        token_id = tl.where(tl.load(skip_mask_ptr + row), plain_token, token_id)
    tl.store(token_ids_ptr + row, token_id)


def philox_ppl_soft_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    ranked_logits: torch.Tensor,
    monte_carlo_key: int,
    n_mc: int,
    level_probs: torch.Tensor,
    eps: float,
    beta_max: float,
    max_iter: int,
    *,
    skip_mask: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """The token ``PPLSoftWatermarker`` selects in each row of float32
    ``logits`` [num_rows, vocab_size], int64 [num_rows], from one pass over the
    vocabulary and ``monte_carlo_bisection``, whose arguments ``n_mc``,
    ``level_probs``, ``eps``, ``beta_max`` and ``max_iter`` are passed on, with
    ``monte_carlo_key`` as its ``key``.
    ``ranked_logits`` holds each row's ``min(vocab_size, 4096)`` largest logits
    in decreasing order. Rows in ``skip_mask`` get the token ``gumbel_sample``
    draws from ``logits`` with the given sampler inputs instead."""
    sample_plain = skip_mask is not None
    assert not sample_plain or expanded_idx_mapping is not None
    stats = _launch_score_levels(
        logits,
        contexts,
        key,
        thresholds,
        with_mass=False,
        with_best=True,
        cap=ranked_logits[:, -1].contiguous(),
        plogp_floor=1e-12,
        plogp_support_only=True,
        skip_mask=skip_mask,
        expanded_idx_mapping=expanded_idx_mapping,
        temperatures=temperatures,
        seeds=seeds,
        positions=positions,
        use_fp64=use_fp64,
    )
    assert stats.plogp is not None
    assert stats.log_probs is not None and stats.token_ids is not None
    beta, _ = monte_carlo_bisection(
        ranked_logits - stats.lse.unsqueeze(-1),
        contexts,
        monte_carlo_key,
        n_mc,
        level_probs,
        stats.plogp.sum(dim=1),
        eps,
        beta_max,
        max_iter,
    )
    num_rows, vocab_size = logits.shape
    num_levels = len(thresholds) + 1
    num_segments = triton.cdiv(vocab_size, _SEGMENT_SIZE)
    token_ids = logits.new_empty(num_rows, dtype=torch.int64)
    if num_rows == 0:
        return token_ids
    _ppl_soft_select_kernel[(num_rows,)](
        token_ids,
        stats.log_probs,
        stats.token_ids,
        stats.log_probs.stride(0),
        stats.log_probs.stride(1),
        beta,
        stats.plain_max,
        stats.plain_token_ids,
        skip_mask.contiguous() if skip_mask is not None else None,
        num_segments,
        NUM_LEVELS=num_levels,
        BLOCK_SEGMENTS=triton.next_power_of_2(num_segments),
        BLOCK_LEVELS=triton.next_power_of_2(num_levels),
    )
    return token_ids


@triton.jit
def _chi_squared_weights_kernel(
    adjusted_scores_ptr,
    mass_ptr,
    mass_stride_0,
    mass_stride_1,
    sigma_ptr,
    num_segments,
    NUM_LEVELS: tl.constexpr,
    BLOCK_SEGMENTS: tl.constexpr,
    BLOCK_LEVELS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    segments = tl.arange(0, BLOCK_SEGMENTS)
    lanes = tl.arange(0, BLOCK_LEVELS)
    is_level = lanes < NUM_LEVELS
    # tau = -1 - sigma * G decreases with the score G, so sorting tau puts the
    # scores in decreasing order: lane r holds score NUM_LEVELS - 1 - r.
    G_s = NUM_LEVELS - 1 - lanes
    valid = (segments < num_segments)[:, None] & is_level[None, :]
    mass = tl.load(
        mass_ptr
        + row * mass_stride_0
        + segments[:, None] * mass_stride_1
        + G_s[None, :],
        mask=valid,
        other=0.0,
    )
    # Segments summed in float32, then float64, as adjusted_scores takes
    # level_mass.
    p_s = tl.sum(mass, axis=0).to(tl.float64)
    G_s = G_s.to(tl.float64)
    sigma = tl.load(sigma_ptr)
    tol = tl.full((), 1e-12, dtype=tl.float64)
    tiny = tl.full((), 1e-18, dtype=tl.float64)

    # _find_mu
    tau_s = -1.0 - sigma * G_s
    P_cum = tl.cumsum(p_s, axis=0)
    A_cum = tl.cumsum(p_s * (1.0 + sigma * G_s), axis=0)
    U_candidates = (1.0 - A_cum) / (P_cum + tiny)
    tau_next = tl.where(
        lanes < NUM_LEVELS - 1, -1.0 - sigma * (G_s - 1.0), float("inf")
    )
    is_valid = (
        is_level
        & (P_cum > tiny)
        & (U_candidates >= tau_s - tol)
        & (U_candidates <= tau_next + tol)
    )
    first_valid_idx = tl.min(tl.where(is_valid, lanes, BLOCK_LEVELS), axis=0)
    final_idx = tl.where(
        first_valid_idx < BLOCK_LEVELS, first_valid_idx, NUM_LEVELS - 1
    )
    U_selected = tl.sum(tl.where(lanes == final_idx, U_candidates, 0.0), axis=0)
    mu = U_selected / sigma

    # adjusted_scores, in score order.
    min_value = tl.full((), 1e-30, dtype=tl.float64)
    scores = lanes.to(tl.float64)
    adjusted_scores = tl.log(tl.maximum(1.0 + sigma * (scores + mu), min_value))
    tl.store(
        adjusted_scores_ptr + row * NUM_LEVELS + lanes,
        adjusted_scores.to(tl.float32),
        mask=is_level,
    )


def philox_chi_squared_weights(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    thresholds: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Kernel version of ``ChiSquaredWatermarker.adjusted_scores`` of
    ``level_mass``, for float32 ``logits`` [num_rows, vocab_size] and
    ``sigma`` a one-element float64 tensor on their device: [num_rows,
    num_levels]."""
    stats = _launch_score_levels(
        logits,
        contexts,
        key,
        thresholds,
        with_mass=True,
        with_best=False,
    )
    num_rows, vocab_size = logits.shape
    num_levels = len(thresholds) + 1
    num_segments = triton.cdiv(vocab_size, _SEGMENT_SIZE)
    adjusted_scores = logits.new_empty(num_rows, num_levels)
    if num_rows == 0:
        return adjusted_scores
    assert stats.mass is not None
    _chi_squared_weights_kernel[(num_rows,)](
        adjusted_scores,
        stats.mass,
        stats.mass.stride(0),
        stats.mass.stride(1),
        sigma,
        num_segments,
        NUM_LEVELS=num_levels,
        BLOCK_SEGMENTS=triton.next_power_of_2(num_segments),
        BLOCK_LEVELS=triton.next_power_of_2(num_levels),
    )
    return adjusted_scores
