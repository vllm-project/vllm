# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
)

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n. Pure Python implementation."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def eagle_prepare_inputs_padded_pytorch(
    cu_num_draft_tokens: torch.Tensor,  # [num_reqs]
    valid_sampled_tokens_count: torch.Tensor,  # [num_reqs]
    query_start_loc_gpu: torch.Tensor,  # [num_reqs + 1]
    token_indices_to_sample: torch.Tensor,  # [num_reqs] (output)
    num_rejected_tokens_gpu: torch.Tensor,  # [num_reqs] (output)
    num_reqs: int,
):
    """PyTorch implementation of eagle_prepare_inputs_padded_kernel."""
    # Compute num_draft_tokens from cumulative sum
    zero = torch.zeros(
        1, dtype=cu_num_draft_tokens.dtype, device=cu_num_draft_tokens.device
    )
    num_draft_tokens = torch.diff(cu_num_draft_tokens, prepend=zero)[:num_reqs]

    # Compute number of rejected tokens
    num_rejected = num_draft_tokens + 1 - valid_sampled_tokens_count[:num_reqs]
    num_rejected = torch.where(num_draft_tokens > 0, num_rejected, zero)

    # Compute token indices to sample
    q_last_tok_idx = query_start_loc_gpu[1 : num_reqs + 1] - 1
    index_to_sample = q_last_tok_idx - num_rejected

    # Write outputs
    token_indices_to_sample[:num_reqs] = index_to_sample.to(
        token_indices_to_sample.dtype
    )
    num_rejected_tokens_gpu[:num_reqs] = num_rejected.to(num_rejected_tokens_gpu.dtype)


def eagle_prepare_next_token_padded_pytorch(
    sampled_token_ids: torch.Tensor,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask: torch.Tensor,  # [num_reqs]
    backup_next_token_ids: torch.Tensor,  # [num_reqs]
    next_token_ids: torch.Tensor,  # [num_reqs] (output)
    valid_sampled_tokens_count: torch.Tensor,  # [num_reqs] (output)
    vocab_size: int,
    num_sampled_tokens_per_req: int,
    num_reqs: int,
):
    """PyTorch implementation of eagle_prepare_next_token_padded_kernel."""
    device = sampled_token_ids.device
    token_ids = sampled_token_ids[:num_reqs, :num_sampled_tokens_per_req]

    # Valid tokens are in [0, vocab_size) and not -1
    is_valid = (token_ids != -1) & (token_ids < vocab_size)
    valid_count = is_valid.sum(dim=1)

    # Find last valid index per row using index multiplication trick
    indices = torch.arange(num_sampled_tokens_per_req, device=device)
    # Set invalid positions to -1 so they don't win the max
    masked_indices = torch.where(is_valid, indices, -1)
    last_valid_idx = masked_indices.max(dim=1).values

    # Gather the token at the last valid index (clamp to 0 for invalid rows)
    gather_idx = last_valid_idx.clamp(min=0).unsqueeze(1)
    last_valid_token = token_ids.gather(1, gather_idx).squeeze(1)

    # Determine which requests have valid tokens
    has_valid = valid_count > 0

    # Set next_token_ids: use last_valid_token if has_valid and not discarded
    use_sampled = has_valid & ~discard_request_mask[:num_reqs]
    next_token_ids[:num_reqs] = torch.where(
        use_sampled, last_valid_token, backup_next_token_ids[:num_reqs]
    ).to(next_token_ids.dtype)

    # Set valid_sampled_tokens_count: 0 if discarded, else valid_count
    valid_sampled_tokens_count[:num_reqs] = torch.where(
        discard_request_mask[:num_reqs],
        torch.zeros_like(valid_count),
        valid_count,
    ).to(valid_sampled_tokens_count.dtype)


def copy_and_expand_eagle_inputs_pytorch(
    # (Padded) Inputs from the target model
    target_token_ids: torch.Tensor,  # [total_tokens_in_batch]
    target_positions: torch.Tensor,  # [total_tokens_in_batch]
    next_token_ids: torch.Tensor,  # [num_reqs]
    # Outputs to the drafting buffers
    out_input_ids: torch.Tensor,  # [total_draft_tokens_in_batch] (output)
    out_positions: torch.Tensor,  # [total_draft_tokens_in_batch] (output)
    out_is_rejected_token_mask: torch.Tensor,  # [total_draft_tokens_in_batch] (output)
    out_is_masked_token_mask: torch.Tensor,  # [total_draft_tokens_in_batch] (output)
    # [num_padding_slots_per_request * num_reqs] (output)
    out_new_token_indices: torch.Tensor,
    out_hidden_state_mapping: torch.Tensor,  # [total_tokens_in_batch]
    # Input metadata
    query_start_loc: torch.Tensor,  # [num_reqs + 1]
    query_end_loc: torch.Tensor,  # [num_reqs]
    padding_token_id: int,
    parallel_drafting_token_id: int,
    # Sizing info
    total_input_tokens: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
):
    """PyTorch implementation of copy_and_expand_eagle_inputs_kernel."""
    num_reqs = query_end_loc.shape[0]
    device = target_token_ids.device

    # Precompute per-request metadata using tensor operations
    query_starts = query_start_loc[:num_reqs]
    next_query_starts = query_start_loc[1 : num_reqs + 1]

    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_starts
        input_offset = 1
        req_indices = torch.arange(num_reqs, device=device)
        output_starts = query_starts + req_indices * (num_padding_slots_per_request - 1)
    else:
        num_valid_tokens = query_end_loc - query_starts + 1
        input_offset = 0
        req_indices = torch.arange(num_reqs, device=device)
        output_starts = query_starts + req_indices * num_padding_slots_per_request

    # Number of rejected tokens per request
    num_rejected = next_query_starts - query_end_loc - 1

    # Total output tokens per request
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # Get start positions for each request
    start_positions = target_positions[query_starts.long()]

    # Process each request using tensor slicing
    for request_idx in range(num_reqs):
        n_valid = num_valid_tokens[request_idx].long().item()
        n_total = total_output_tokens[request_idx].long().item()
        out_start = output_starts[request_idx].long().item()
        q_start = query_starts[request_idx].long().item()
        start_pos = start_positions[request_idx].item()
        bonus_token = next_token_ids[request_idx]

        # Create output index range for this request
        j_range = torch.arange(n_total, device=device)
        out_indices = out_start + j_range

        # Region masks
        is_valid = j_range < n_valid
        is_bonus = j_range == n_valid
        is_parallel = (j_range > n_valid) & (
            j_range < n_valid + num_padding_slots_per_request
        )
        is_rejected = j_range >= n_valid + num_padding_slots_per_request

        # Compute input indices for valid region (clamped)
        in_indices = torch.clamp(
            q_start + input_offset + j_range, max=total_input_tokens - 1
        )

        # Build token_ids tensor for this request
        token_ids = torch.full(
            (n_total,), padding_token_id, device=device, dtype=out_input_ids.dtype
        )
        # Valid tokens from input
        token_ids = torch.where(
            is_valid, target_token_ids[in_indices.long()], token_ids
        )
        # Bonus token
        token_ids = torch.where(is_bonus, bonus_token.to(token_ids.dtype), token_ids)
        # Parallel drafting tokens
        token_ids = torch.where(is_parallel, parallel_drafting_token_id, token_ids)
        # Rejected tokens already set to padding_token_id

        # Compute positions
        positions = torch.where(is_rejected, 0, start_pos + j_range)

        # Write outputs using tensor indexing
        out_input_ids[out_indices.long()] = token_ids.to(out_input_ids.dtype)
        out_positions[out_indices.long()] = positions.to(out_positions.dtype)
        out_is_rejected_token_mask[out_indices.long()] = is_rejected
        out_is_masked_token_mask[out_indices.long()] = is_parallel

        # Store new token indices (bonus + parallel drafting slots)
        is_new_token = (j_range >= n_valid) & (
            j_range < n_valid + num_padding_slots_per_request
        )
        new_token_local_idx = j_range[is_new_token] - n_valid
        new_token_out_idx = (
            request_idx * num_padding_slots_per_request + new_token_local_idx
        )
        out_new_token_indices[new_token_out_idx.long()] = out_indices[is_new_token].to(
            out_new_token_indices.dtype
        )

        # Compute hidden state mapping if shift_input_ids
        if shift_input_ids:
            n_input = (next_query_starts[request_idx] - q_start).long().item()
            src_indices = q_start + torch.arange(n_input, device=device)
            dst_indices = out_start + torch.arange(n_input, device=device)
            out_hidden_state_mapping[src_indices.long()] = dst_indices.to(
                out_hidden_state_mapping.dtype
            )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_rejected_tokens_gpu_ptr,  # [num_reqs] (output)
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
    tl.store(num_rejected_tokens_gpu_ptr + req_idx, num_rejected_tokens)


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


def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
):
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(
        req_indices,
        cad.naive_query_lens() + num_new_tokens,
        output_size=len(new_positions),
    )
    # Clamp the positions to prevent an out-of-bounds error when indexing
    # into block_table_tensor.
    clamped_positions = torch.clamp(new_positions, max=max_model_len - 1)
    block_table_indices = (
        req_indices * n_blocks_per_req + clamped_positions // block_size
    )
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = clamped_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
    # Mask out rejected tokens to prevent saves to the KV cache.
    new_slot_mapping.masked_fill_(is_rejected_token_mask, PADDING_SLOT_ID)
    return new_slot_mapping


def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig:
    """The vllm_config is configured for the target model, e.g.
    its quant_config and parallel_config. But the draft model is potentially
    quantized differently, and has potentially different tensor_parallel_size.
    This function creates a new vllm_config configured for the drafter.
    The vllm_config is useful when loading the draft model with get_model().
    """
    old = target_model_vllm_config
    assert old.speculative_config is not None, "speculative_config is not set"
    old_spec_config = old.speculative_config
    new_parallel_config = replace(
        old_spec_config.draft_parallel_config, rank=old.parallel_config.rank
    )
    new: VllmConfig = replace(
        old,
        quant_config=None,
        parallel_config=new_parallel_config,
        model_config=old_spec_config.draft_model_config,
    )
    return new


def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata:
    """
    Creates a new CommonAttentionMetadata with all query lengths increased by N.
    Also all seq lens are increased by N.
    This is useful e.g. in speculative decoding with parallel drafting, where we
    extend each sequence by N tokens and predict all tokens in one pass.
    The slot mapping is computed externally, as it requires more information.
    """
    cad = common_attn_metadata
    # query start loc must be increased by [+0, +N, +2N, ..., +batch_size * N]
    new_query_start_loc = cad.query_start_loc + N * arange[: len(cad.query_start_loc)]
    new_query_start_loc_cpu = cad.query_start_loc_cpu + N * torch.arange(
        len(cad.query_start_loc_cpu), dtype=torch.int32
    )
    new_cad = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=cad.seq_lens + N,
        # each request is extended by N tokens -> batch_size * N tokens are added
        num_actual_tokens=cad.num_actual_tokens + cad.batch_size() * N,
        # All query lens increase by N, so max query len increases by N
        max_query_len=cad.max_query_len + N,
        max_seq_len=cad.max_seq_len + N,
        slot_mapping=new_slot_mapping,
    )
    return new_cad


# Unified copy/expand kernel
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    # (Padded) Inputs from the target model
    target_token_ids_ptr,  # [total_tokens_in_batch]
    target_positions_ptr,  # [total_tokens_in_batch]
    next_token_ids_ptr,  # [num_reqs]
    # Outputs to the drafting buffers
    out_input_ids_ptr,  # [total_draft_tokens_in_batch] (output)
    out_positions_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_rejected_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_is_masked_token_mask_ptr,  # [total_draft_tokens_in_batch] (output)
    out_new_token_indices_ptr,  # [num_padding_slots_per_request * num_reqs] (output)
    out_hidden_state_mapping_ptr,  # [total_tokens_in_batch]
    # Input metadata
    query_start_loc_ptr,  # [num_reqs + 1], last value is the total num input tokens
    query_end_loc_ptr,  # [num_reqs]
    padding_token_id,  # tl.int32
    parallel_drafting_token_id,  # tl.int32
    # Sizing info
    total_input_tokens,  # tl.int32
    num_padding_slots_per_request,  # tl.int32
    shift_input_ids,  # tl.bool
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Blocks along token dim to handle prefills
):
    """
    Copy and expand inputs from the target model to the drafting buffers for Eagle
    speculative decoding. This kernel handles padding slots and parallel drafting
    tokens, if enabled.
    """
    request_idx = tl.program_id(axis=0)
    token_batch_idx = tl.program_id(axis=1)

    # Load query locations
    query_start_loc = tl.load(query_start_loc_ptr + request_idx)
    next_query_start_loc = tl.load(query_start_loc_ptr + request_idx + 1)
    query_end_loc = tl.load(query_end_loc_ptr + request_idx)

    # Calculate number of valid tokens to copy and input offset
    # With shift_input_ids=True, we skip the first token
    # Output layout: each request gets (input_len + num_padding_slots_per_request) slots
    # But with shift, we lose one token per request
    if shift_input_ids:
        num_valid_tokens = query_end_loc - query_start_loc
        input_offset = 1
        output_start = query_start_loc + request_idx * (
            num_padding_slots_per_request - 1
        )
    else:
        num_valid_tokens = query_end_loc - query_start_loc + 1
        input_offset = 0
        output_start = query_start_loc + request_idx * num_padding_slots_per_request

    # Number of rejected tokens from previous speculation
    num_rejected = next_query_start_loc - query_end_loc - 1

    # Total output tokens for this request
    total_output_tokens = (
        num_valid_tokens + num_padding_slots_per_request + num_rejected
    )

    # Process tokens in this block
    j = token_batch_idx * BLOCK_SIZE_TOKENS + tl.arange(0, BLOCK_SIZE_TOKENS)

    # Compute masks for different output regions:
    # [0, num_valid_tokens): valid tokens copied from input
    # [num_valid_tokens]: bonus token from next_token_ids
    # (num_valid_tokens, num_valid_tokens + num_padding_slots_per_request):
    #     parallel drafting slots
    # [num_valid_tokens + num_padding_slots_per_request, total_output_tokens):
    #     rejected slots
    in_bounds = j < total_output_tokens
    is_valid_region = j < num_valid_tokens
    is_bonus_region = j == num_valid_tokens
    is_parallel_draft_region = (j > num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    is_rejected_region = j >= num_valid_tokens + num_padding_slots_per_request

    # Compute output indices
    out_idx = output_start + j

    # For valid tokens, compute input index
    in_idx = query_start_loc + input_offset + j
    # Clamp to avoid out-of-bounds access (masked loads still need valid addresses)
    in_idx_clamped = tl.minimum(in_idx, total_input_tokens - 1)

    # Load input tokens (masked to valid region)
    token_ids = tl.load(
        target_token_ids_ptr + in_idx_clamped, mask=is_valid_region & in_bounds, other=0
    )

    # Load the starting position for this request (first position in the sequence)
    start_pos = tl.load(target_positions_ptr + query_start_loc)

    # Load bonus token for this request
    bonus_token = tl.load(next_token_ids_ptr + request_idx)

    # Build final token_ids based on region
    token_ids = tl.where(is_bonus_region, bonus_token, token_ids)
    token_ids = tl.where(
        is_parallel_draft_region, parallel_drafting_token_id, token_ids
    )
    token_ids = tl.where(is_rejected_region, padding_token_id, token_ids)

    # Build final positions:
    # Positions are NOT shifted - they start from the first input position and increment
    # Output position j gets start_pos + j
    # (e.g., input positions [5,6,7] -> output [5,6,7,8,9,...])
    positions = start_pos + j
    # Rejected positions are don't-care, set to 0
    positions = tl.where(is_rejected_region, 0, positions)

    # Compute output masks
    is_rejected_out = is_rejected_region & in_bounds
    is_masked_out = is_parallel_draft_region & in_bounds

    # Compute indices of new tokens (bonus + parallel drafting) for sampling
    # New tokens are at positions
    #     [num_valid_tokens, num_valid_tokens + num_padding_slots_per_request)
    is_new_token_region = (j >= num_valid_tokens) & (
        j < num_valid_tokens + num_padding_slots_per_request
    )
    new_token_local_idx = (
        j - num_valid_tokens
    )  # 0 for bonus, 1, 2, ... for parallel drafting
    new_token_out_idx = (
        request_idx * num_padding_slots_per_request + new_token_local_idx
    )

    # Compute hidden state mapping (source index -> destination index)
    # This maps each input position to its corresponding output position
    # Hidden states don't get shifted, so we map all input tokens (including rejected)
    if shift_input_ids:
        num_input_tokens_this_request = next_query_start_loc - query_start_loc
        is_input_region = j < num_input_tokens_this_request
        src_idx = query_start_loc + j
        tl.store(out_hidden_state_mapping_ptr + src_idx, out_idx, mask=is_input_region)

    # Store outputs
    tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
    tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
    tl.store(out_is_rejected_token_mask_ptr + out_idx, is_rejected_out, mask=in_bounds)
    tl.store(out_is_masked_token_mask_ptr + out_idx, is_masked_out, mask=in_bounds)
    tl.store(
        out_new_token_indices_ptr + new_token_out_idx,
        out_idx,
        mask=is_new_token_region & in_bounds,
    )
