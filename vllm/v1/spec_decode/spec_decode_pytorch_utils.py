# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PyTorch fallback implementations for Triton kernels used in speculative decoding.

This module contains pure PyTorch implementations of the Triton kernels
used in speculative decoding. These fallbacks are used when Triton is not
available (e.g., on CPU platforms where Triton kernels cannot execute).

Note: These implementations are functionally equivalent to their Triton
counterparts but may have different performance characteristics. They are
primarily intended for CPU execution where Triton is not supported.

"""

import torch

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


# =============================================================================
# EAGLE Speculative Decoding Fallbacks (from utils.py)
# =============================================================================


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


def eagle_step_update_slot_mapping_and_metadata_pytorch(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None:
    """PyTorch implementation of eagle_step_slot_mapping_metadata_kernel."""
    batch_size = positions_1d.shape[0]
    if input_batch_size is None:
        input_batch_size = batch_size

    # Handle padding slots (req_idx >= batch_size)
    if input_batch_size > batch_size:
        out_slot_mapping[batch_size:input_batch_size] = PADDING_SLOT_ID

    # Process actual requests
    new_position = positions_1d + 1
    exceeds_max = new_position >= max_model_len
    clamped_position = torch.where(
        exceeds_max, torch.zeros_like(new_position), new_position
    )

    # Block table lookup
    block_number = clamped_position // block_size
    n_blocks_per_req = block_table_tensor.shape[1]
    block_number = torch.clamp(block_number, max=n_blocks_per_req - 1)

    # Gather block_id from block_table
    block_id = block_table_tensor[
        torch.arange(batch_size, device=block_table_tensor.device), block_number.long()
    ]
    slot_id = block_id * block_size + (clamped_position % block_size)
    slot_id = torch.where(exceeds_max, PADDING_SLOT_ID, slot_id)

    # Update seq_lens
    new_seq_len = torch.where(exceeds_max, torch.ones_like(seq_lens), seq_lens + 1)
    new_seq_len = torch.clamp(new_seq_len, max=max_model_len)

    # Store outputs
    out_clamped_positions[:batch_size] = clamped_position.to(
        out_clamped_positions.dtype
    )
    out_slot_mapping[:batch_size] = slot_id.to(out_slot_mapping.dtype)
    seq_lens[:] = new_seq_len.to(seq_lens.dtype)


# =============================================================================
# Rejection Sampler Fallbacks (from rejection_sampler.py)
# =============================================================================


def rejection_greedy_sample_pytorch(
    output_token_ids: torch.Tensor,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    draft_token_ids: torch.Tensor,  # [num_tokens]
    target_argmax: torch.Tensor,  # [num_tokens]
    bonus_token_ids: torch.Tensor,  # [batch_size]
    is_greedy: torch.Tensor | None,  # [batch_size] or None
):
    """PyTorch implementation of rejection_greedy_sample_kernel."""
    batch_size = cu_num_draft_tokens.shape[0]
    max_spec_len = output_token_ids.shape[1] - 1
    device = draft_token_ids.device

    # Compute counts and start indices per request
    zero = torch.zeros(1, dtype=cu_num_draft_tokens.dtype, device=device)
    counts = torch.diff(cu_num_draft_tokens, prepend=zero)
    start_indices = torch.cat([zero, cu_num_draft_tokens[:-1]])

    # Create request indices for each token
    req_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), counts
    )

    # Create position indices within each request
    pos_in_req = (
        torch.arange(draft_token_ids.shape[0], device=device)
        - start_indices[req_indices]
    )

    # Find mismatches (rejections)
    is_mismatch = draft_token_ids != target_argmax

    # For each request, find the first rejection position
    # Set mismatches to their position, matches to max_spec_len + 1
    mismatch_pos = torch.where(is_mismatch, pos_in_req, max_spec_len + 1)

    # Scatter min to find first rejection per request
    first_reject_pos = torch.full(
        (batch_size,), max_spec_len + 1, dtype=torch.int64, device=device
    )
    first_reject_pos.scatter_reduce_(
        0, req_indices.long(), mismatch_pos.long(), reduce="amin"
    )

    # Create output mask: positions < first_reject_pos get target_argmax
    pos_range = torch.arange(max_spec_len, device=device).unsqueeze(0)
    valid_mask = pos_range < first_reject_pos.unsqueeze(1)
    valid_mask = valid_mask & (pos_range < counts.unsqueeze(1))

    # Apply greedy mask if provided
    if is_greedy is not None:
        valid_mask = valid_mask & is_greedy.unsqueeze(1)

    # Scatter target_argmax values to output
    for req_idx in range(batch_size):
        if is_greedy is not None and not is_greedy[req_idx]:
            continue
        n = counts[req_idx].item()
        start = start_indices[req_idx].long().item()
        first_rej = first_reject_pos[req_idx].item()

        # Copy tokens up to and including the first rejection
        copy_len = min(n, first_rej + 1)
        output_token_ids[req_idx, :copy_len] = target_argmax[start : start + copy_len]

        # Add bonus token if all accepted
        if first_rej > n - 1:
            output_token_ids[req_idx, n] = bonus_token_ids[req_idx, 0]


def rejection_random_sample_pytorch(
    output_token_ids: torch.Tensor,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    draft_token_ids: torch.Tensor,  # [num_tokens]
    draft_probs: torch.Tensor | None,  # [num_tokens, vocab_size] or None
    target_probs: torch.Tensor,  # [num_tokens, vocab_size]
    bonus_token_ids: torch.Tensor,  # [batch_size]
    recovered_token_ids: torch.Tensor,  # [num_tokens]
    uniform_probs: torch.Tensor,  # [num_tokens]
    is_greedy: torch.Tensor | None,  # [batch_size] or None
    no_draft_probs: bool = False,
):
    """PyTorch implementation of rejection_random_sample_kernel."""
    if not no_draft_probs and draft_probs is None:
        raise ValueError("draft_probs is required when no_draft_probs=False")

    batch_size = cu_num_draft_tokens.shape[0]
    num_tokens = draft_token_ids.shape[0]
    max_spec_len = output_token_ids.shape[1] - 1
    device = draft_token_ids.device

    # Compute counts and start indices per request
    zero = torch.zeros(1, dtype=cu_num_draft_tokens.dtype, device=device)
    counts = torch.diff(cu_num_draft_tokens, prepend=zero)
    start_indices = torch.cat([zero, cu_num_draft_tokens[:-1]])

    # Create request indices for each token
    req_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), counts
    )[:num_tokens]

    # Create position indices within each request
    pos_in_req = (
        torch.arange(num_tokens, device=device) - start_indices[req_indices].long()
    )

    # Gather draft and target probabilities for the draft tokens
    token_range = torch.arange(num_tokens, device=device)
    if no_draft_probs:
        draft_prob_vals = torch.ones(num_tokens, device=device)
    else:
        draft_prob_vals = draft_probs[token_range, draft_token_ids]  # type: ignore

    target_prob_vals = target_probs[token_range, draft_token_ids]

    # Compute acceptance: target_prob / draft_prob >= uniform_prob
    # Handle division by zero
    accept_ratio = torch.where(
        draft_prob_vals > 0,
        target_prob_vals / draft_prob_vals,
        torch.zeros_like(target_prob_vals),
    )
    is_accepted = accept_ratio >= uniform_probs

    # Find first rejection position per request
    reject_pos = torch.where(is_accepted, max_spec_len + 1, pos_in_req)
    first_reject_pos = torch.full(
        (batch_size,), max_spec_len + 1, dtype=torch.int64, device=device
    )
    first_reject_pos.scatter_reduce_(
        0, req_indices.long(), reject_pos.long(), reduce="amin"
    )

    # Process each request
    for req_idx in range(batch_size):
        if is_greedy is not None and is_greedy[req_idx]:
            continue

        n = counts[req_idx].item()
        start = int(start_indices[req_idx].item())
        first_rej = first_reject_pos[req_idx].item()

        # Copy accepted tokens (draft_token_ids) and first rejected (recovered)
        for pos in range(min(n, first_rej)):
            output_token_ids[req_idx, pos] = draft_token_ids[start + pos]

        # If there was a rejection, add the recovered token
        if first_rej < n:
            output_token_ids[req_idx, first_rej] = recovered_token_ids[
                start + first_rej
            ]

        # Add bonus token if all accepted
        if first_rej >= n:
            output_token_ids[req_idx, n] = bonus_token_ids[req_idx, 0]


def expand_pytorch(
    output: torch.Tensor,  # [num_tokens]
    input_val: torch.Tensor,  # [batch_size]
    cu_num_tokens: torch.Tensor,  # [batch_size]
    replace_from: int,
    replace_to: int,
):
    """PyTorch implementation of expand_kernel."""
    # Compute counts per batch from cumulative sum
    zero = torch.zeros(1, dtype=cu_num_tokens.dtype, device=cu_num_tokens.device)
    counts = torch.diff(cu_num_tokens, prepend=zero)

    # Replace values
    vals = torch.where(input_val == replace_from, replace_to, input_val)

    # Expand using repeat_interleave
    total_tokens = cu_num_tokens[-1].item() if len(cu_num_tokens) > 0 else 0
    output[:total_tokens] = vals.repeat_interleave(counts).to(output.dtype)


def sample_recovered_tokens_pytorch(
    output_token_ids: torch.Tensor,  # [num_tokens]
    cu_num_draft_tokens: torch.Tensor,  # [batch_size]
    draft_token_ids: torch.Tensor,  # [num_tokens]
    draft_probs: torch.Tensor | None,  # [num_tokens, vocab_size] or None
    target_probs: torch.Tensor,  # [num_tokens, vocab_size]
    inv_q: torch.Tensor,  # [batch_size, vocab_size] - reciprocal of q
    no_draft_probs: bool = False,
):
    """PyTorch implementation of sample_recovered_tokens_kernel."""
    if not no_draft_probs and draft_probs is None:
        raise ValueError("draft_probs is required when no_draft_probs=False")

    num_tokens = draft_token_ids.shape[0]
    device = target_probs.device

    # Map each token position to its request index
    zero = torch.zeros(1, dtype=cu_num_draft_tokens.dtype, device=device)
    counts = torch.diff(cu_num_draft_tokens, prepend=zero)
    req_indices = torch.repeat_interleave(
        torch.arange(len(counts), device=device), counts
    )[:num_tokens]

    # Gather inv_q values for each token position
    inv_q_expanded = inv_q[req_indices]  # [num_tokens, vocab_size]

    if no_draft_probs:
        # Zero out the draft token probability for each position
        prob = target_probs.clone()
        token_indices = torch.arange(num_tokens, device=device)
        prob[token_indices, draft_token_ids] = 0
    else:
        # Compute adjusted probability: max(target - draft, 0)
        diff = target_probs - draft_probs  # type: ignore[operator]
        prob = torch.clamp(diff, min=0)

    # Gumbel-max trick: argmax(prob * inv_q) equivalent to argmax(prob / q)
    recovered_ids = torch.argmax(prob * inv_q_expanded, dim=1)
    output_token_ids[:num_tokens] = recovered_ids.to(output_token_ids.dtype)
