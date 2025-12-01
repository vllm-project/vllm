# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton

_SAMPLING_EPS = 1e-5


def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return (
        sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
        or sampling_params.repetition_penalty != 1.0
        or sampling_params.min_p > _SAMPLING_EPS
        or sampling_params.logprobs is not None
    )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_reqs,  # tl.int32
):
    """
    Fused kernel for Eagle prepare_input_padded. This kernel computes the
    token index to sample for each request, taking into account the number
    of draft tokens and the number of valid sampled tokens (which is one more than
    the number of accepted tokens).
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    num_draft_tokens = 0
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
