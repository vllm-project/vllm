# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.flash_sampler_utils import _lm_head_matmul
from vllm.v1.worker.gpu.sample.gumbel import tl_rand64
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_lse,
    _insert_resampled_kernel,
    _resample_kernel,
)


@triton.jit
def _compute_stable_sumexp(logits, maxes):
    sumexp = tl.where(
        maxes[:, None] > float("-inf"),
        tl.exp(logits - maxes[:, None]),
        0.0,
    )
    return tl.sum(sumexp, axis=1)


@triton.jit
def _fused_lm_head_block_stats_kernel(
    target_local_max_ptr,
    target_local_max_stride,
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_tokens, num_vocab_blocks]
    target_logits_at_draft_sampled_ptr,
    target_logits_at_draft_sampled_stride,
    # [num_tokens, hidden_dim]
    target_hidden_states_ptr,
    target_hidden_states_stride,
    # [local_vocab_size, hidden_dim]
    lm_head_weights_ptr,
    lm_head_weights_stride,
    target_logits_scale_ptr,
    # [max_slots, shard_vocab_size]
    flash_target_logits_ptr,
    flash_target_logits_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_tokens]
    draft_sampled_ptr,
    # [num_tokens]
    expanded_idx_mapping_ptr,
    # [num_tokens]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    num_tokens,
    num_speculative_steps,
    hidden_dim,
    vocab_size,
    vocab_start,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    token_block_idx = tl.program_id(0)
    token_block = token_block_idx * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    token_mask = token_block < num_tokens
    draft_step_idxs = tl.load(
        expanded_local_pos_ptr + token_block, mask=token_mask, other=0
    )
    req_state_idxs = tl.load(
        expanded_idx_mapping_ptr + token_block, mask=token_mask, other=0
    )
    temps = tl.load(temp_ptr + req_state_idxs, mask=token_mask, other=1.0).to(
        tl.float32
    )

    vocab_block_idx = tl.program_id(1)
    vocab_block = vocab_block_idx * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
    vocab_mask = vocab_block < vocab_size

    # Compute target logits from hidden states and LM head.
    target_logits = _lm_head_matmul(
        target_hidden_states_ptr,
        target_hidden_states_stride,
        lm_head_weights_ptr,
        lm_head_weights_stride,
        target_logits_scale_ptr,
        token_block,
        token_mask,
        vocab_block,
        vocab_mask,
        hidden_dim,
        BLOCK_SIZE_T,
        BLOCK_SIZE_H,
        BLOCK_SIZE_V,
    )

    # Apply temperature to target logits.
    is_greedy = temps == 0.0
    safe_temps = tl.where(is_greedy, 1.0, temps)
    target_logits = target_logits / safe_temps[:, None]

    # Cache temperature-scaled logits in the persistent buffer.
    tl.store(
        flash_target_logits_ptr
        + token_block[:, None] * flash_target_logits_stride
        + vocab_block[None, :],
        target_logits,
        mask=token_mask[:, None] & vocab_mask[None, :],
    )

    is_draft = draft_step_idxs < num_speculative_steps
    draft_token_mask = token_mask & is_draft

    # Extract the target logits at the draft-sampled token positions.
    # Each vocab block writes its contribution (0 if the draft token
    # is not in this block, the logit value otherwise). The rejection
    # kernel sums across blocks to recover the scalar.
    draft_tokens = tl.load(draft_sampled_ptr + token_block + 1, mask=token_mask, other=0)
    local_draft_tokens = draft_tokens - vocab_start
    offset_in_block = local_draft_tokens - vocab_block_idx * BLOCK_SIZE_V
    extract_mask = tl.arange(0, BLOCK_SIZE_V)[None, :] == offset_in_block[:, None]
    target_logits_at_draft_sampled = tl.sum(
        tl.where(extract_mask, target_logits, 0.0),
        axis=1,
    )
    tl.store(
        target_logits_at_draft_sampled_ptr
        + token_block * target_logits_at_draft_sampled_stride
        + vocab_block_idx,
        target_logits_at_draft_sampled,
        mask=draft_token_mask,
    )

    # Compute the max and sum-exponentials for the target and draft logits.
    target_max = tl.max(target_logits, axis=1, return_indices=False)
    target_sumexp = _compute_stable_sumexp(target_logits, target_max)
    tl.store(
        target_local_max_ptr + token_block * target_local_max_stride + vocab_block_idx,
        target_max,
        mask=draft_token_mask,
    )
    tl.store(
        target_local_sumexp_ptr
        + token_block * target_local_sumexp_stride
        + vocab_block_idx,
        target_sumexp,
        mask=draft_token_mask,
    )
    if HAS_DRAFT_LOGITS:
        global_offsets = vocab_block + vocab_start
        draft_logits = tl.load(
            draft_logits_ptr
            + req_state_idxs[:, None] * draft_logits_stride_0
            + draft_step_idxs[:, None] * draft_logits_stride_1
            + global_offsets[None, :],
            mask=draft_token_mask[:, None] & vocab_mask[None, :],
            other=float("-inf"),
        ).to(tl.float32)
        draft_max = tl.max(draft_logits, axis=1)
        draft_sumexp = _compute_stable_sumexp(draft_logits, draft_max)
        tl.store(
            draft_local_max_ptr + token_block * draft_local_max_stride + vocab_block_idx,
            draft_max,
            mask=draft_token_mask,
        )
        tl.store(
            draft_local_sumexp_ptr
            + token_block * draft_local_sumexp_stride
            + vocab_block_idx,
            draft_sumexp,
            mask=draft_token_mask,
        )


@triton.jit
def _rejection_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    rejected_steps_ptr,
    # [num_reqs]
    target_rejected_logsumexp_ptr,
    # [num_reqs]
    draft_rejected_logsumexp_ptr,
    # [num_tokens, num_blocks]
    target_logits_at_draft_sampled_ptr,
    target_logits_at_draft_sampled_stride,
    # [num_tokens, num_blocks]
    target_local_max_ptr,
    target_local_max_stride,
    # [num_tokens, num_blocks]
    target_local_sumexp_ptr,
    target_local_sumexp_stride,
    # [num_tokens]
    draft_sampled_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_tokens, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_tokens, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # [max_num_reqs]
    seed_ptr,
    # [num_logits]
    pos_ptr,
    num_vocab_blocks,
    PADDED_NUM_VOCAB_BLOCKS: tl.constexpr,
    HAS_DRAFT_LOGITS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx
    seed = tl.load(seed_ptr + req_state_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    rejected_step = 0
    target_lse = 0.0
    draft_lse = 0.0
    accepted = True
    for i in range(num_tokens - 1):
        if accepted:
            logit_idx = start_idx + i
            draft_sampled = tl.load(draft_sampled_ptr + logit_idx + 1)
            vocab_blocks = tl.arange(0, PADDED_NUM_VOCAB_BLOCKS)
            vocab_blocks_mask = vocab_blocks < num_vocab_blocks
            target_logit = tl.sum(tl.load(
                target_logits_at_draft_sampled_ptr
                + logit_idx * target_logits_at_draft_sampled_stride
                + vocab_blocks,
                mask=vocab_blocks_mask,
                other=0.0,
            )).to(tl.float32)
            if temp == 0.0:
                # Greedy sampling. Accept IFF the target logit at the
                # draft sampled token matches the target max.
                target_local_max = tl.load(
                    target_local_max_ptr
                    + logit_idx * target_local_max_stride
                    + vocab_blocks,
                    mask=vocab_blocks_mask,
                    other=float("-inf"),
                )
                target_max = tl.max(target_local_max, axis=0)
                accepted &= target_logit == target_max
            else:
                target_lse = _compute_global_lse(
                    target_local_max_ptr,
                    target_local_max_stride,
                    target_local_sumexp_ptr,
                    target_local_sumexp_stride,
                    logit_idx,
                    num_vocab_blocks,
                    PADDED_NUM_VOCAB_BLOCKS,
                )
                target_log_prob = target_logit - target_lse
                pos = tl.load(pos_ptr + logit_idx)
                u = tl_rand64(seed, pos, includes_zero=False)
                if HAS_DRAFT_LOGITS:
                    draft_logit = tl.load(
                        draft_logits_ptr
                        + req_state_idx * draft_logits_stride_0
                        + i * draft_logits_stride_1
                        + draft_sampled
                    ).to(tl.float32)
                    draft_lse = _compute_global_lse(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        logit_idx,
                        num_vocab_blocks,
                        PADDED_NUM_VOCAB_BLOCKS,
                    )
                    draft_log_prob = draft_logit - draft_lse
                else:
                    # One-hot draft: q(draft_token) = 1, log_q = 0.
                    draft_log_prob = 0
                # Probability ratio test: p(x) > u * q(x)
                # Equivalent log form: log_p(x) > log(u) + log_q(x)
                accepted &= target_log_prob > tl.log(u) + draft_log_prob
            tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)
            rejected_step += accepted
    tl.store(rejected_steps_ptr + req_idx, rejected_step)
    tl.store(target_rejected_logsumexp_ptr + req_idx, target_lse)
    tl.store(draft_rejected_logsumexp_ptr + req_idx, draft_lse)


def flash_rejection_sample(
    # [num_logits, hidden_size]
    target_hidden_states: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    positions: torch.Tensor,
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
    # [local_vocab_size, hidden_size]
    lm_head_weight: torch.Tensor,
    vocab_shard_indices: VocabParallelEmbeddingShardIndices,
    # [num_logits, shard_vocab_size]
    target_logits_cache: torch.Tensor,
    num_speculative_steps: int,
    target_logits_scale: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    num_tokens, hidden_dim = target_hidden_states.shape
    vocab_size = vocab_shard_indices.num_org_elements
    vocab_start = vocab_shard_indices.org_vocab_start_index
    has_draft_logits = draft_logits is not None

    if draft_logits is None:
        # When draft_logits is None, create a dummy tensor so that Triton
        # kernel signatures receive valid pointers/strides. The kernels
        # will never read from it when HAS_DRAFT_LOGITS=False.
        draft_logits = target_hidden_states.new_empty(1, 1, 1)

    if num_tokens <= 16:
        BLOCK_SIZE_T = 16
        BLOCK_SIZE_H = 256
        BLOCK_SIZE_V = 128
    elif num_tokens <= 32:
        BLOCK_SIZE_T = 32
        BLOCK_SIZE_H = 128
        BLOCK_SIZE_V = 256
    else:
        BLOCK_SIZE_T = 64
        BLOCK_SIZE_H = 128
        BLOCK_SIZE_V = 128

    # Fused lm_head matmul + block-level logits stats.
    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE_V)
    padded_num_vocab_blocks = triton.next_power_of_2(num_vocab_blocks)
    # Row 0: target_logits_at_draft_sampled (per-block contributions)
    # Row 1: target_local_max
    # Row 2: target_local_sumexp
    # Row 3: draft_local_max       (only when has_draft_logits)
    # Row 4: draft_local_sumexp    (only when has_draft_logits)
    num_stat_rows = 5 if has_draft_logits else 3
    block_stats = target_hidden_states.new_empty(
        num_stat_rows, num_tokens, num_vocab_blocks, dtype=torch.float32
    )
    target_logits_at_draft_sampled = block_stats[0]
    target_local_max = block_stats[1]
    target_local_sumexp = block_stats[2]
    draft_local_max = block_stats[3] if has_draft_logits else target_local_max
    draft_local_sumexp = block_stats[4] if has_draft_logits else target_local_sumexp
    num_token_blocks = triton.cdiv(num_tokens, BLOCK_SIZE_T)
    _fused_lm_head_block_stats_kernel[(num_token_blocks, num_vocab_blocks)](
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits_at_draft_sampled,
        target_logits_at_draft_sampled.stride(0),
        target_hidden_states,
        target_hidden_states.stride(0),
        lm_head_weight,
        lm_head_weight.stride(0),
        target_logits_scale,
        target_logits_cache,
        target_logits_cache.stride(0),
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_sampled,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        num_tokens,
        num_speculative_steps,
        hidden_dim,
        vocab_size,
        vocab_start,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        HAS_DRAFT_LOGITS=has_draft_logits,
        num_warps=8,
        num_stages=4,
    )

    # TP all-gather block logits stats across ranks.
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        block_stats = tensor_model_parallel_all_gather(block_stats, dim=-1)
        target_logits_at_draft_sampled = block_stats[0]
        target_local_max = block_stats[1]
        target_local_sumexp = block_stats[2]
        if has_draft_logits:
            draft_local_max = block_stats[3]
            draft_local_sumexp = block_stats[4]
        # Update the number of vocab blocks.
        num_vocab_blocks = target_local_max.shape[-1]
        padded_num_vocab_blocks = triton.next_power_of_2(num_vocab_blocks)

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_hidden_states.new_empty(
        num_reqs, dtype=torch.float32
    )
    draft_rejected_logsumexp = target_hidden_states.new_empty(
        num_reqs, dtype=torch.float32
    )
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits_at_draft_sampled,
        target_logits_at_draft_sampled.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        positions,
        num_vocab_blocks,
        PADDED_NUM_VOCAB_BLOCKS=padded_num_vocab_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens using the cached target
    # logits in flash_target_logits.
    RESAMPLE_BLOCK_SIZE = 1024
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits_cache.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits_cache.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.float32
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits_cache,
        target_logits_cache.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        positions,
        vocab_size,
        vocab_start,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
    )

    # TP all-gather resampled max/argmax across ranks.
    if tp_size > 1:
        resampled_pair = torch.stack(
            [resampled_local_argmax.float(), resampled_local_max], dim=-1,
        )
        gathered = tensor_model_parallel_all_gather(resampled_pair, dim=-1)
        gathered = gathered.view(num_reqs, -1, 2)
        resampled_local_argmax = gathered[:, :, 0].to(torch.int64).contiguous()
        resampled_local_max = gathered[:, :, 1].contiguous()
        resample_num_blocks = resampled_local_max.shape[-1]
        padded_resample_num_blocks = triton.next_power_of_2(
            resample_num_blocks
        )

    # Insert the resampled tokens into the output sampled.
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
