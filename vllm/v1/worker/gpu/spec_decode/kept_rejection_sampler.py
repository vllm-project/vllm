# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rejection sampling from each target row's kept tokens only.

With top-k (and top-p), the target distribution is zero outside the row's
kept tokens (`top_k_top_p_kept_tokens_triton`). So the acceptance test and the
residual `max(p - q, 0)` only need the target over those tokens, and the draft
probabilities gathered at them. Only the draft normalizer still reads the full
draft row. Covers standard (Leviathan) and greedy verification, with one-hot or
full draft distributions; the noise is keyed as in `rejection_sample`.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32
from vllm.v1.worker.gpu.spec_decode.draft_logits import (
    DraftLogitsCache,
    draft_logits_row,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp,
    _compute_max_and_sumexp,
)


def kept_rejection_sample(
    # [num_logits, max_kept], [num_logits, max_kept], [num_logits]
    kept: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    draft_logits_cache: DraftLogitsCache | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    vocab_size: int,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept_ids, kept_logits, num_kept = kept
    num_logits, max_kept = kept_ids.shape
    num_reqs = cu_num_logits.shape[0] - 1
    draft_logits, row_mapping = None, None
    if draft_logits_cache is not None:
        draft_logits = draft_logits_cache.values
        row_mapping = draft_logits_cache.row_mapping
        assert draft_logits.size(-1) >= vocab_size
    draft_stats, num_blocks = _draft_logits_stats(
        draft_logits,
        row_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        num_logits,
        vocab_size,
    )
    draft_local_max, draft_local_sumexp = draft_stats
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    _kept_rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        kept_ids,
        kept_logits,
        num_kept,
        max_kept,
        draft_sampled,
        draft_logits,
        draft_logits.stride(0) if draft_logits is not None else 0,
        draft_logits.stride(1) if draft_logits is not None else 0,
        row_mapping,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        num_blocks,
        BLOCK_SIZE=triton.next_power_of_2(max_kept),
        PADDED_VOCAB_NUM_BLOCKS=triton.next_power_of_2(num_blocks),
        HAS_DRAFT_LOGITS=draft_logits is not None,
        USE_FP64=use_fp64,
        num_warps=4,
    )
    return sampled, num_sampled


def _draft_logits_stats(
    draft_logits: torch.Tensor | None,
    row_mapping: torch.Tensor | None,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    num_logits: int,
    vocab_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
    """Per-vocab-block draft max and summed exponentials, as `rejection_sample`
    computes them, so the draft normalizer matches it bitwise."""
    VOCAB_BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    empty = expanded_local_pos.new_empty((num_logits, num_blocks), dtype=torch.float32)
    local_max, local_sumexp = empty, torch.empty_like(empty)
    if draft_logits is None:
        return (local_max, local_sumexp), num_blocks
    _draft_local_stats_kernel[(num_logits, num_blocks)](
        local_max,
        local_max.stride(0),
        local_sumexp,
        local_sumexp.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        row_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_logits,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
    )
    return (local_max, local_sumexp), num_blocks


@triton.jit(do_not_specialize=["num_logits"])
def _draft_local_stats_kernel(
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    temp_ptr,
    vocab_size,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    # The draft half of _compute_local_logits_stats_kernel.
    logit_idx = tl.program_id(0).to(tl.int64)
    next_local_pos = tl.load(
        expanded_local_pos_ptr + logit_idx + 1,
        mask=logit_idx + 1 < num_logits,
        other=0,
    )
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if next_local_pos == 0 or temp == 0.0:
        # Bonus or greedy rows don't need the draft normalizer.
        return
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr)
    draft_logits = (
        tl.load(
            draft_logits_ptr
            + tl.maximum(row, 0) * draft_logits_stride_0
            + draft_step_idx * draft_logits_stride_1
            + block_offsets,
            mask=block_offsets < vocab_size,
            other=float("-inf"),
        ).to(tl.float32)
        / temp
    )
    draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
    tl.store(
        draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
        draft_max,
    )
    tl.store(
        draft_local_sumexp_ptr + logit_idx * draft_local_sumexp_stride + block_idx,
        draft_sumexp,
    )


@triton.jit
def _kept_rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_logits, max_kept]
    kept_ids_ptr,
    kept_logits_ptr,
    # [num_logits]
    num_kept_ptr,
    max_kept,
    # [num_logits]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    draft_logits_row_mapping_ptr,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    seed_ptr,
    # [num_logits]
    pos_ptr,
    vocab_num_blocks,
    BLOCK_SIZE: tl.constexpr,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    is_greedy = temp == 0.0
    if HAS_DRAFT_LOGITS:
        draft_row = tl.maximum(
            draft_logits_row(req_state_idx, draft_logits_row_mapping_ptr), 0
        )
        draft_row_ptr = draft_logits_ptr + draft_row * draft_logits_stride_0

    # Verify the draft tokens up to the first rejection.
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        is_valid_draft = draft_token >= 0
        draft_token = tl.maximum(0, draft_token)
        if not is_greedy:
            verifying &= is_valid_draft
        if verifying:
            ids, logits, mask = _load_kept(
                kept_ids_ptr, kept_logits_ptr, num_kept_ptr, max_kept, logit_idx,
                BLOCK_SIZE,
            )  # fmt: skip
            sampled = draft_token
            if is_greedy:
                target_argmax = _kept_argmax(ids, logits, mask)
                accepted = (target_argmax == draft_token) & is_valid_draft
                sampled = tl.where(accepted, draft_token, target_argmax)
            else:
                u = tl_rand32(seed, tl.load(pos_ptr + logit_idx), includes_zero=False)
                target_max, target_sumexp = _compute_max_and_sumexp(logits)
                target_lse = target_max + tl.log(target_sumexp)
                is_draft = mask & (ids == draft_token)
                target_logit = tl.max(tl.where(is_draft, logits, float("-inf")))
                draft_logprob = 0.0
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.load(
                        draft_row_ptr + i * draft_logits_stride_1 + draft_token
                    ).to(tl.float32)
                    draft_lse = _compute_global_logsumexp(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        logit_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    )
                    draft_logprob = draft_logit / temp - draft_lse
                accepted = target_logit - target_lse > tl.log(u) + draft_logprob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, sampled)
            verifying = accepted
            accepted_length += accepted

    # Sample the token after the accepted prefix: the target argmax of a greedy
    # rejection is already stored, otherwise resample from the residual.
    resample_idx = start_idx + accepted_length
    is_bonus = resample_idx == end_idx - 1
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))
    if is_greedy and not is_bonus:
        return
    ids, logits, mask = _load_kept(
        kept_ids_ptr, kept_logits_ptr, num_kept_ptr, max_kept, resample_idx, BLOCK_SIZE
    )
    rejected_token = tl.load(
        draft_sampled_ptr + resample_idx + 1, mask=not is_bonus, other=0
    )
    residual_logits = logits
    if not is_bonus and rejected_token >= 0:
        if HAS_DRAFT_LOGITS:
            # r(x) = max(p(x) - q(x), 0), in log space as in _resample_kernel.
            target_max, target_sumexp = _compute_max_and_sumexp(logits)
            target_log_probs = logits - (target_max + tl.log(target_sumexp))
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                resample_idx,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
            draft_logits = tl.load(
                draft_row_ptr + accepted_length * draft_logits_stride_1 + ids,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            draft_log_probs = draft_logits / temp - draft_lse
            ratio = tl.exp(draft_log_probs - target_log_probs)
            residual_logits = tl.where(
                ratio < 1.0,
                target_log_probs + tldevice.log1p(-ratio),
                float("-inf"),
            ).to(tl.float32)
        else:
            residual_logits = tl.where(ids != rejected_token, logits, float("-inf"))
    _, idx = gumbel_noised_argmax(
        residual_logits,
        ids,
        mask,
        seed,
        tl.load(pos_ptr + resample_idx),
        temp,
        IS_DRAFTING=False,
        USE_FP64=USE_FP64,
        APPLY_TEMPERATURE=False,
    )
    resampled = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == idx, ids, 0))
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, resampled)


@triton.jit
def _load_kept(
    kept_ids_ptr,
    kept_logits_ptr,
    num_kept_ptr,
    max_kept,
    logit_idx,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < tl.minimum(tl.load(num_kept_ptr + logit_idx), max_kept)
    row = logit_idx * max_kept + offs
    ids = tl.load(kept_ids_ptr + row, mask=mask, other=0).to(tl.int64)
    logits = tl.load(kept_logits_ptr + row, mask=mask, other=float("-inf"))
    return ids, logits, mask


@triton.jit
def _kept_argmax(ids, logits, mask):
    """Lowest token id with the largest logit, as the vocab-block argmax."""
    logits = tl.where(logits != logits, float("-inf"), logits)
    value = tl.max(logits)
    argmax = tl.min(tl.where(mask & (logits == value), ids, 2**62))
    # A row without kept tokens (all -inf) gives token 0, as the dense argmax.
    return tl.where(argmax == 2**62, 0, argmax)
