# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.watermarking.prfs.philox import (
    _CONTEXT_DOMAIN as _CONTEXT_DOMAIN_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_M0 as _PHILOX_M0_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_M1 as _PHILOX_M1_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_W0 as _PHILOX_W0_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _PHILOX_W1 as _PHILOX_W1_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _TOKEN_DOMAIN as _TOKEN_DOMAIN_VALUE,
)
from vllm.v1.watermarking.prfs.philox import (
    _UINT32_MASK as _UINT32_MASK_VALUE,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

if HAS_TRITON:
    from triton.language import math as tl_math
else:
    tl_math = tl

_UINT32_MASK = tl.constexpr(_UINT32_MASK_VALUE) if HAS_TRITON else _UINT32_MASK_VALUE
_PHILOX_M0 = tl.constexpr(_PHILOX_M0_VALUE) if HAS_TRITON else _PHILOX_M0_VALUE
_PHILOX_M1 = tl.constexpr(_PHILOX_M1_VALUE) if HAS_TRITON else _PHILOX_M1_VALUE
_PHILOX_W0 = tl.constexpr(_PHILOX_W0_VALUE) if HAS_TRITON else _PHILOX_W0_VALUE
_PHILOX_W1 = tl.constexpr(_PHILOX_W1_VALUE) if HAS_TRITON else _PHILOX_W1_VALUE
_CONTEXT_DOMAIN = (
    tl.constexpr(_CONTEXT_DOMAIN_VALUE) if HAS_TRITON else _CONTEXT_DOMAIN_VALUE
)
_TOKEN_DOMAIN = tl.constexpr(_TOKEN_DOMAIN_VALUE) if HAS_TRITON else _TOKEN_DOMAIN_VALUE


@triton.jit
def _repeated_context_mask_kernel(
    output_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    req_indices_ptr,
    prompt_lens_ptr,
    total_lens_ptr,
    history_offsets_ptr,
    history_offsets_stride,
    contexts_ptr,
    context_stride,
    local_positions_ptr,
    prior_contexts_ptr,
    prior_contexts_stride_0,
    prior_contexts_stride_1,
    steps_ptr,
    steps_stride,
    enabled_ptr,
    CONTEXT_WIDTH: tl.constexpr,
    NUM_SPECULATIVE_STEPS: tl.constexpr,
    MAX_HISTORY: tl.constexpr,
    INCLUDE_PROMPT: tl.constexpr,
    SKIP_PARTIAL_CONTEXT: tl.constexpr,
    HAS_HISTORY_OFFSETS: tl.constexpr,
    SPECULATIVE_CONTEXTS: tl.constexpr,
    DRAFT_CONTEXTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(req_indices_ptr + row).to(tl.int64)
    if req_idx < 0:
        tl.store(output_ptr + row, 0)
        return
    prompt_len = tl.load(prompt_lens_ptr + req_idx)
    total_len = tl.load(total_lens_ptr + req_idx)
    if SKIP_PARTIAL_CONTEXT and total_len - prompt_len < CONTEXT_WIDTH:
        tl.store(output_ptr + row, 1)
        return
    sequence_start = 0 if INCLUDE_PROMPT else prompt_len
    history_len = total_len - sequence_start
    history_offset = 0
    if HAS_HISTORY_OFFSETS:
        history_offset = tl.load(history_offsets_ptr + row * history_offsets_stride)

    offsets = tl.arange(0, BLOCK)
    repeated = tl.full((), 0, tl.int32)
    scan_start = 0
    if MAX_HISTORY > 0:
        remaining_history = tl.maximum(MAX_HISTORY - history_offset, 0)
        scan_start = tl.maximum(history_len - remaining_history, 0)
    # Aligned loop bounds let the per-lane offsets fold into immediates.
    aligned_start = (scan_start // BLOCK) * BLOCK
    for block_start in tl.range(aligned_start, history_len, BLOCK):
        previous_pos = block_start + offsets
        in_window = (previous_pos >= scan_start) & (previous_pos < history_len)
        matches = in_window
        for offset in range(CONTEXT_WIDTH):
            context_token = tl.load(contexts_ptr + row * context_stride + offset)
            historical_pos = previous_pos + offset - CONTEXT_WIDTH
            historical_token = tl.load(
                all_token_ids_ptr
                + req_idx * all_token_ids_stride
                + sequence_start
                + tl.maximum(historical_pos, 0),
                mask=in_window & (historical_pos >= 0),
                other=-1,
            )
            matches &= historical_token == context_token
        repeated |= tl.max(matches.to(tl.int32), axis=0)

    if SPECULATIVE_CONTEXTS:
        local_pos = tl.load(local_positions_ptr + row)
        if INCLUDE_PROMPT:
            partial_context = tl.full((), 0, tl.int32)
            for context_offset in range(CONTEXT_WIDTH):
                context_token = tl.load(
                    contexts_ptr + row * context_stride + context_offset
                )
                partial_context |= context_token < 0
            repeated |= partial_context
        for offset in range(1, NUM_SPECULATIVE_STEPS + 1):
            prior_row = row - offset
            in_history = (prior_row >= 0) & (local_pos >= offset)
            if MAX_HISTORY > 0:
                in_history &= offset <= MAX_HISTORY
            prior_req_idx = tl.load(
                req_indices_ptr + tl.maximum(prior_row, 0),
                mask=in_history,
                other=-1,
            )
            prior_local_pos = tl.load(
                local_positions_ptr + tl.maximum(prior_row, 0),
                mask=in_history,
                other=-1,
            )
            matches = (
                in_history
                & (req_idx == prior_req_idx)
                & (local_pos == prior_local_pos + offset)
            )
            for context_offset in range(CONTEXT_WIDTH):
                context_token = tl.load(
                    contexts_ptr + row * context_stride + context_offset
                )
                prior_token = tl.load(
                    contexts_ptr
                    + tl.maximum(prior_row, 0) * context_stride
                    + context_offset,
                    mask=in_history,
                    other=-2,
                )
                matches &= context_token == prior_token
            repeated |= matches

    if DRAFT_CONTEXTS:
        step = tl.load(steps_ptr + row * steps_stride).to(tl.int64)
        partial_context = tl.full((), 0, tl.int32)
        for context_offset in range(CONTEXT_WIDTH):
            context_token = tl.load(
                contexts_ptr + row * context_stride + context_offset
            )
            partial_context |= context_token < 0
        if INCLUDE_PROMPT:
            repeated |= partial_context

        for prior_step in range(NUM_SPECULATIVE_STEPS):
            in_history = prior_step < step
            if MAX_HISTORY > 0:
                in_history &= prior_step >= step - MAX_HISTORY
            matches = in_history
            for context_offset in range(CONTEXT_WIDTH):
                context_token = tl.load(
                    contexts_ptr + row * context_stride + context_offset
                )
                prior_token = tl.load(
                    prior_contexts_ptr
                    + row * prior_contexts_stride_0
                    + prior_step * prior_contexts_stride_1
                    + context_offset
                )
                matches &= context_token == prior_token
            repeated |= matches

        for context_offset in range(CONTEXT_WIDTH):
            context_token = tl.load(
                contexts_ptr + row * context_stride + context_offset
            )
            tl.store(
                prior_contexts_ptr
                + row * prior_contexts_stride_0
                + step * prior_contexts_stride_1
                + context_offset,
                context_token,
            )
        enabled = tl.load(enabled_ptr + row)
        repeated = enabled & (repeated == 0)
    tl.store(output_ptr + row, repeated)


def _repeated_context_mask_cpu(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    contexts: torch.Tensor,
    max_history: int | None = None,
    include_prompt: bool = False,
    skip_partial_context: bool = False,
    history_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation; the parity tests check the Triton kernel against it."""
    repeated = torch.zeros(len(req_indices), dtype=torch.bool)
    for row, req_idx_tensor in enumerate(req_indices):
        req_idx = int(req_idx_tensor)
        if req_idx < 0:
            continue
        prompt_len = int(prompt_lens[req_idx])
        total_len = int(total_lens[req_idx])
        if skip_partial_context and total_len - prompt_len < contexts.shape[-1]:
            repeated[row] = True
            continue
        sequence_start = 0 if include_prompt else prompt_len
        history_tokens = all_token_ids[req_idx, sequence_start:total_len].tolist()
        prefix = [-1] * contexts.shape[-1]
        current_context = tuple(contexts[row].tolist())
        history_offset = 0 if history_offsets is None else int(history_offsets[row])
        history_start = (
            max(0, len(history_tokens) - max(0, max_history - history_offset))
            if max_history is not None
            else 0
        )
        for history_pos, token_id in enumerate(history_tokens):
            if history_pos >= history_start and (
                tuple(prefix[-contexts.shape[-1] :]) == current_context
            ):
                repeated[row] = True
                break
            prefix.append(token_id)
    return repeated


def repeated_context_mask(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    contexts: torch.Tensor,
    max_history: int | None = None,
    include_prompt: bool = False,
    skip_partial_context: bool = False,
    history_offsets: torch.Tensor | None = None,
    local_positions: torch.Tensor | None = None,
    num_speculative_steps: int = 0,
) -> torch.Tensor:
    """Return, per row, whether the row's context already occurred in its history.

    Args:
        all_token_ids: `[max_num_reqs, max_model_len]` token ids of every request.
        req_indices: Request slot per sampled row; -1 marks a padding row, which
            is reported as not repeated.
        prompt_lens: Prompt length per request slot.
        total_lens: Prompt plus generated length per request slot.
        contexts: `[num_rows, context_width]` context per row, padded with -1
            before the start of the scanned history.
        max_history: Number of most recent history positions searched, or
            ``None`` for all of them. The window compared at each position
            reaches `context_width` tokens further back.
        include_prompt: Search the prompt as well as the generated tokens.
        skip_partial_context: Mark contexts containing start padding so they use
            ordinary sampling.
        history_offsets: Number of newer, non-committed positions preceding each
            context. These positions count toward `max_history`. Must be 1-D
            over rows; a stride-0 broadcast row is fine.
        local_positions: Position within each request's speculative block. When
            provided, compare each context with earlier contexts in that block.
        num_speculative_steps: Maximum number of draft tokens in the block.

    """
    if max_history is not None and max_history < 1:
        raise ValueError("max_history must be positive or None")
    if (local_positions is None) != (num_speculative_steps == 0):
        raise ValueError(
            "local_positions and positive num_speculative_steps must be "
            "provided together"
        )
    if all_token_ids.device.type == "cpu":
        repeated = _repeated_context_mask_cpu(
            all_token_ids,
            req_indices,
            prompt_lens,
            total_lens,
            contexts,
            max_history,
            include_prompt,
            skip_partial_context,
            history_offsets,
        )
        if local_positions is not None:
            max_offset = num_speculative_steps + 1
            if max_history is not None:
                max_offset = min(max_offset, max_history + 1)
            for offset in range(1, max_offset):
                prior_contexts = torch.cat(
                    (contexts[:offset], contexts[:-offset]), dim=0
                )
                prior_requests = torch.cat(
                    (req_indices[:offset], req_indices[:-offset]), dim=0
                )
                prior_local_pos = torch.cat(
                    (local_positions[:offset], local_positions[:-offset]), dim=0
                )
                repeated |= (
                    (req_indices >= 0)
                    & (local_positions >= offset)
                    & (req_indices == prior_requests)
                    & (local_positions == prior_local_pos + offset)
                    & (contexts == prior_contexts).all(dim=-1)
                )
            if include_prompt:
                repeated |= (contexts < 0).any(dim=-1)
        return repeated

    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    # The kernel reads these flat, one element per row. `history_offsets` is
    # not in the list because the kernel reads it as
    # `history_offsets_ptr + row * history_offsets_stride`, so a stride-0
    # broadcast row needs no copy. `local_positions` is read flat and is packed
    # even when the caller passes the same tensor for both.
    req_indices = req_indices.contiguous()
    if local_positions is not None:
        local_positions = local_positions.contiguous()
    # `prompt_lens` and `total_lens` are indexed flat by request slot, and
    # `all_token_ids` takes a row stride but is read flat within the row. Those
    # three are contiguous by contract; every caller owns them outright.
    repeated = torch.empty(len(req_indices), dtype=torch.bool, device=contexts.device)
    _repeated_context_mask_kernel[(len(req_indices),)](
        repeated,
        all_token_ids,
        all_token_ids.stride(0),
        req_indices,
        prompt_lens,
        total_lens,
        history_offsets,
        0 if history_offsets is None else history_offsets.stride(0),
        contexts,
        contexts.stride(0),
        local_positions,
        None,
        0,
        0,
        None,
        0,
        None,
        CONTEXT_WIDTH=contexts.shape[-1],
        NUM_SPECULATIVE_STEPS=num_speculative_steps,
        MAX_HISTORY=0 if max_history is None else max_history,
        INCLUDE_PROMPT=include_prompt,
        SKIP_PARTIAL_CONTEXT=skip_partial_context,
        HAS_HISTORY_OFFSETS=history_offsets is not None,
        SPECULATIVE_CONTEXTS=local_positions is not None,
        DRAFT_CONTEXTS=False,
        BLOCK=512,
    )
    return repeated


def draft_watermarking_mask(
    all_token_ids: torch.Tensor,
    req_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    total_lens: torch.Tensor,
    prior_contexts: torch.Tensor,
    contexts: torch.Tensor,
    steps: torch.Tensor,
    enabled: torch.Tensor,
    max_history: int | None,
    include_prompt: bool,
) -> torch.Tensor:
    if contexts.device.type == "cpu":
        repeated = repeated_context_mask(
            all_token_ids,
            req_indices,
            prompt_lens,
            total_lens,
            contexts,
            max_history,
            include_prompt=include_prompt,
            history_offsets=steps,
        )
        prior_steps = torch.arange(prior_contexts.shape[1]).unsqueeze(0)
        in_history = prior_steps < steps.unsqueeze(1)
        if max_history is not None:
            in_history &= prior_steps >= steps.unsqueeze(1) - max_history
        repeated |= (
            (prior_contexts[: len(contexts)] == contexts.unsqueeze(1)).all(dim=-1)
            & in_history
        ).any(dim=-1)
        if include_prompt:
            repeated |= (contexts < 0).any(dim=-1)
        row_indices = torch.arange(len(contexts))
        prior_contexts.index_put_((row_indices, steps), contexts)
        return enabled & ~repeated

    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    # The kernel indexes these by row without a stride. `steps` is left alone:
    # the kernel takes its stride, so a broadcast row needs no copy.
    req_indices = req_indices.contiguous()
    enabled = enabled.contiguous()
    # `prompt_lens`, `total_lens` and the last dimension of `all_token_ids` and
    # of `prior_contexts` are read flat and contiguous by contract:
    # `prior_contexts` is allocated by the draft watermarker and the other
    # three come from the runner's request state, which owns them outright.
    active = torch.empty(len(contexts), dtype=torch.bool, device=contexts.device)
    _repeated_context_mask_kernel[(len(contexts),)](
        active,
        all_token_ids,
        all_token_ids.stride(0),
        req_indices,
        prompt_lens,
        total_lens,
        steps,
        steps.stride(0),
        contexts,
        contexts.stride(0),
        None,
        prior_contexts,
        prior_contexts.stride(0),
        prior_contexts.stride(1),
        steps,
        steps.stride(0),
        enabled,
        NUM_SPECULATIVE_STEPS=prior_contexts.shape[1],
        CONTEXT_WIDTH=contexts.shape[1],
        MAX_HISTORY=0 if max_history is None else max_history,
        INCLUDE_PROMPT=include_prompt,
        SKIP_PARTIAL_CONTEXT=False,
        HAS_HISTORY_OFFSETS=True,
        SPECULATIVE_CONTEXTS=False,
        DRAFT_CONTEXTS=True,
        BLOCK=512,
    )
    return active


@triton.jit
def _mulhilo32(multiplier: tl.constexpr, value):
    high = tl_math.umulhi(multiplier, value)
    low = tl.mul(multiplier, value, sanitize_overflow=False)
    return high, low


@triton.jit
def _philox4x32_10(counter_0, counter_1, counter_2, counter_3, key_0, key_1):
    for round_index in range(10):
        high_0, low_0 = _mulhilo32(_PHILOX_M0, counter_0)
        high_1, low_1 = _mulhilo32(_PHILOX_M1, counter_2)
        counter_0, counter_1, counter_2, counter_3 = (
            (high_1 ^ counter_1 ^ key_0) & _UINT32_MASK,
            low_1,
            (high_0 ^ counter_3 ^ key_1) & _UINT32_MASK,
            low_0,
        )
        if round_index != 9:
            key_0 = (key_0 + _PHILOX_W0) & _UINT32_MASK
            key_1 = (key_1 + _PHILOX_W1) & _UINT32_MASK
    return counter_0, counter_1, counter_2, counter_3


@triton.jit
def _uint32_to_uniform(value):
    mantissa = (value & _UINT32_MASK) >> 8
    scaled = (mantissa + 1).to(tl.float32) * 2**-24
    return (scaled.to(tl.uint32, bitcast=True) - 1).to(tl.float32, bitcast=True)


@triton.jit
def _philox_context_state(
    contexts_row_ptr,
    key_0,
    key_1,
    CONTEXT_WIDTH: tl.constexpr,
):
    state_0 = tl.full((), _CONTEXT_DOMAIN, tl.uint32)
    state_1 = tl.full((), CONTEXT_WIDTH, tl.uint32)
    state_2 = tl.full((), 0, tl.uint32)
    state_3 = tl.full((), 0, tl.uint32)
    for offset in range(0, CONTEXT_WIDTH, 4):
        context_0 = tl.load(contexts_row_ptr + offset).to(tl.uint32)
        if offset + 1 < CONTEXT_WIDTH:
            context_1 = tl.load(contexts_row_ptr + offset + 1).to(tl.uint32)
        else:
            context_1 = tl.full((), _UINT32_MASK - 1, tl.uint32)
        if offset + 2 < CONTEXT_WIDTH:
            context_2 = tl.load(contexts_row_ptr + offset + 2).to(tl.uint32)
        else:
            context_2 = tl.full((), _UINT32_MASK - 2, tl.uint32)
        if offset + 3 < CONTEXT_WIDTH:
            context_3 = tl.load(contexts_row_ptr + offset + 3).to(tl.uint32)
        else:
            context_3 = tl.full((), _UINT32_MASK - 3, tl.uint32)
        state_0, state_1, state_2, state_3 = _philox4x32_10(
            (state_0 ^ context_0) & _UINT32_MASK,
            (state_1 ^ context_1) & _UINT32_MASK,
            (state_2 ^ context_2) & _UINT32_MASK,
            (state_3 ^ context_3) & _UINT32_MASK,
            (key_0 ^ offset) & _UINT32_MASK,
            key_1,
        )
    return state_0, state_1, state_2, state_3


@triton.jit
def _philox_candidate_words(groups, state_0, state_1, state_2, state_3, key_0, key_1):
    candidate_words = groups.to(tl.uint32)
    vector_zero = candidate_words * 0
    return _philox4x32_10(
        candidate_words & _UINT32_MASK,
        state_0 + vector_zero,
        state_1 + vector_zero,
        state_2 + vector_zero,
        ((key_0 ^ state_3) & _UINT32_MASK) + vector_zero,
        ((key_1 ^ _TOKEN_DOMAIN) & _UINT32_MASK) + vector_zero,
    )


@triton.jit
def _philox_gumbel_from_logits(logits, word):
    logits = tl.where(logits != logits, float("-inf"), logits)
    uniform = _uint32_to_uniform(word)
    return logits - tl.log(-tl.log(uniform))


@triton.jit
def _gumbel_value(logits_ptr, output, mask):
    logits = tl.load(logits_ptr, mask=mask, other=float("-inf")).to(tl.float32)
    return _philox_gumbel_from_logits(logits, output)


@triton.jit
def philox_gumbel_block_argmax(
    logits,
    mask,
    block_idx,
    contexts_row_ptr,
    key_0,
    key_1,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply the detector-compatible PRF to one vocabulary block."""
    tl.static_assert(BLOCK_SIZE % 4 == 0)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_row_ptr, key_0, key_1, CONTEXT_WIDTH
    )
    groups = block_idx * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
    output_0, output_1, output_2, output_3 = _philox_candidate_words(
        groups, state_0, state_1, state_2, state_3, key_0, key_1
    )
    # Lay the per-group words out in token order: 4g+0, 4g+1, 4g+2, 4g+3.
    words = tl.interleave(
        tl.interleave(output_0, output_2),
        tl.interleave(output_1, output_3),
    )
    values = tl.where(
        mask,
        _philox_gumbel_from_logits(logits.to(tl.float32), words),
        float("-inf"),
    )
    return tl.max(values, axis=0, return_indices=True)


@triton.jit
def _select_max(best_value, best_token, candidate_value, candidate):
    take_candidate = candidate_value > best_value
    return (
        tl.where(take_candidate, candidate_value, best_value),
        tl.where(take_candidate, candidate, best_token),
    )


@triton.jit(do_not_specialize=["key_0_value", "key_1_value"])
def _philox_gumbel_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    contexts_ptr,
    context_stride,
    skip_mask_ptr,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_cache_source_ptr,
    logits_cache_source_stride,
    key_0_value,
    key_1_value,
    vocab_size,
    CONTEXT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block_index = tl.program_id(1)
    groups = block_index * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
    candidate = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = candidate < vocab_size

    req_state_idx = row
    valid_req = True
    if expanded_idx_mapping_ptr is not None:
        req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
        valid_req = req_state_idx >= 0

    if logits_cache_ptr is not None:
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + row)
        else:
            col = tl.load(logits_cache_col_ptr)
        cached_logits = tl.load(
            logits_cache_source_ptr + row * logits_cache_source_stride + candidate,
            mask=mask,
        )
        tl.store(
            logits_cache_ptr
            + req_state_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
            + candidate,
            cached_logits,
            mask=mask & valid_req,
        )

    if skip_mask_ptr is not None:  # noqa: SIM102
        if tl.load(skip_mask_ptr + row):
            temp = tl.load(temp_ptr + req_state_idx, mask=valid_req, other=0.0).to(
                tl.float32
            )
            seed = tl.load(seeds_ptr + req_state_idx, mask=valid_req, other=0)
            pos = tl.load(pos_ptr + row)
            logits_row = logits_ptr + row * logits_stride
            logits = tl.load(logits_row + candidate, mask=mask, other=float("-inf")).to(
                tl.float32
            )
            value, index = gumbel_noised_argmax(
                logits,
                candidate,
                mask,
                seed,
                pos,
                temp,
                IS_DRAFTING=IS_DRAFTING,
                USE_FP64=USE_FP64,
                APPLY_TEMPERATURE=False,
            )
            token_id = block_index * BLOCK_SIZE + index
            tl.store(
                local_argmax_ptr + row * local_argmax_stride + block_index,
                token_id,
            )
            tl.store(local_max_ptr + row * local_max_stride + block_index, value)
            return

    key_0 = key_0_value.to(tl.uint32)
    key_1 = key_1_value.to(tl.uint32)
    state_0, state_1, state_2, state_3 = _philox_context_state(
        contexts_ptr + row * context_stride, key_0, key_1, CONTEXT_WIDTH
    )
    output_0, output_1, output_2, output_3 = _philox_candidate_words(
        groups, state_0, state_1, state_2, state_3, key_0, key_1
    )
    candidate_0 = groups * 4
    candidate_1 = candidate_0 + 1
    candidate_2 = candidate_0 + 2
    candidate_3 = candidate_0 + 3
    logits_row = logits_ptr + row * logits_stride
    value_0 = _gumbel_value(
        logits_row + candidate_0, output_0, candidate_0 < vocab_size
    )
    value_1 = _gumbel_value(
        logits_row + candidate_1, output_1, candidate_1 < vocab_size
    )
    value_2 = _gumbel_value(
        logits_row + candidate_2, output_2, candidate_2 < vocab_size
    )
    value_3 = _gumbel_value(
        logits_row + candidate_3, output_3, candidate_3 < vocab_size
    )
    best_value = value_0
    best_token = candidate_0
    best_value, best_token = _select_max(best_value, best_token, value_1, candidate_1)
    best_value, best_token = _select_max(best_value, best_token, value_2, candidate_2)
    best_value, best_token = _select_max(best_value, best_token, value_3, candidate_3)
    value = tl.max(best_value, axis=0)
    token_id = tl.min(tl.where(best_value == value, best_token, vocab_size), axis=0)
    tl.store(local_argmax_ptr + row * local_argmax_stride + block_index, token_id)
    tl.store(local_max_ptr + row * local_max_stride + block_index, value)


def philox_gumbel_sample(
    logits: torch.Tensor,
    contexts: torch.Tensor,
    key: int,
    *,
    skip_mask: torch.Tensor | None = None,
    expanded_idx_mapping: torch.Tensor | None = None,
    temperatures: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    use_fp64: bool = False,
    is_drafting: bool = False,
    logits_cache: torch.Tensor | None = None,
    logits_cache_col: torch.Tensor | None = None,
    logits_cache_source: torch.Tensor | None = None,
) -> torch.Tensor:
    sampling_state = (expanded_idx_mapping, temperatures, seeds, positions)
    if skip_mask is None and any(value is not None for value in sampling_state):
        raise ValueError("sampling state requires skip_mask")
    if skip_mask is not None and any(value is None for value in sampling_state):
        raise ValueError("skip_mask requires complete sampling state")
    if logits_cache is not None:
        if logits_cache_col is None or logits_cache_source is None:
            raise ValueError("logits cache requires its column and source logits")
        if expanded_idx_mapping is None:
            raise ValueError("logits cache requires expanded_idx_mapping")
        assert logits_cache_source.shape == logits.shape, (
            "logits cache source must match sampled logits shape"
        )
        assert logits_cache_source.device == logits.device, (
            "logits cache source must be on the sampled logits device"
        )
        assert logits_cache_source.dtype == logits_cache.dtype, (
            "logits cache source and destination must have the same dtype"
        )
        assert logits_cache.size(-1) >= logits.shape[-1], (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({logits.shape[-1]}). Cached logits would be "
            "truncated."
        )
    elif logits_cache_col is not None or logits_cache_source is not None:
        raise ValueError("logits cache column and source require logits_cache")

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if contexts.stride(-1) != 1:
        contexts = contexts.contiguous()
    if skip_mask is not None:
        skip_mask = skip_mask.contiguous()
    if expanded_idx_mapping is not None:
        expanded_idx_mapping = expanded_idx_mapping.contiguous()
    if positions is not None:
        positions = positions.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()
    if logits_cache_source is not None and logits_cache_source.stride(-1) != 1:
        logits_cache_source = logits_cache_source.contiguous()
    num_tokens, vocab_size = logits.shape
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max = logits.new_empty(
        num_tokens,
        num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _philox_gumbel_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        contexts,
        contexts.stride(0),
        skip_mask,
        expanded_idx_mapping,
        seeds,
        positions,
        temperatures,
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits_cache_source,
        logits_cache_source.stride(0) if logits_cache_source is not None else 0,
        key & _UINT32_MASK_VALUE,
        key >> 32,
        vocab_size,
        CONTEXT_WIDTH=contexts.shape[-1],
        BLOCK_SIZE=block_size,
        USE_FP64=use_fp64,
        IS_DRAFTING=is_drafting,
        PER_TOKEN_COL=logits_cache_col is not None and logits_cache_col.dim() > 0,
    )
    max_block_index = local_max.argmax(dim=-1, keepdim=True)
    return local_argmax.gather(dim=-1, index=max_block_index).view(-1)
