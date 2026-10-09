# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch implementations of the model runner's input preparation Triton kernels."""

import torch

from vllm.v1.worker.cpu.kernels.utils import ptr_view, token_to_batch_idx


def prepare_prefill_inputs(
    input_ids_ptr: torch.Tensor,
    next_prefill_tokens_ptr: torch.Tensor,
    next_prefill_tokens_stride: int,
    num_lookahead: int,
    idx_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    all_token_ids_ptr: torch.Tensor,
    all_token_ids_stride: int,
    prefill_lens_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    BLOCK_SIZE: int,
    LOOKAHEAD_BLOCK: int,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    req = idx_mapping_ptr.long()
    num_computed = num_computed_tokens_ptr[req].long()
    prefill_len = prefill_lens_ptr[req].long()
    is_prefill = num_computed < prefill_len
    if not bool(is_prefill.any()):
        return

    query_start_loc = query_start_loc_ptr[: num_reqs + 1].long()
    num_tokens = int(query_start_loc[-1])
    batch_idx = token_to_batch_idx(query_start_loc, num_tokens)
    tok_req = req[batch_idx]
    local = torch.arange(num_tokens) - query_start_loc[batch_idx]
    col = num_computed[batch_idx] + local
    tokens = all_token_ids_ptr[tok_req, col.clamp_max(all_token_ids_ptr.shape[1] - 1)]
    input_ids = input_ids_ptr[:num_tokens]
    input_ids.copy_(torch.where(is_prefill[batch_idx], tokens, input_ids))

    if num_lookahead == 0:
        return
    query_len = query_start_loc[1:] - query_start_loc[:-1]
    # [num_lookahead, num_reqs]
    pos = (num_computed + query_len) + torch.arange(num_lookahead).unsqueeze(1)
    lookahead = all_token_ids_ptr[
        req.expand_as(pos), pos.clamp_max(all_token_ids_ptr.shape[1] - 1)
    ]
    lookahead = torch.where(pos < prefill_len, lookahead, 0)
    prefill_req = req[is_prefill]
    next_prefill_tokens_ptr[:, prefill_req] = lookahead[:, is_prefill].to(
        next_prefill_tokens_ptr.dtype
    )


def prepare_pos_seq_lens(
    pos_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    max_num_reqs: int,
    BLOCK_SIZE: int,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    seq_lens_ptr[num_reqs:max_num_reqs] = 0

    num_computed = num_computed_tokens_ptr[idx_mapping_ptr.long()]
    query_start_loc = query_start_loc_ptr[: num_reqs + 1]
    query_len = query_start_loc[1:] - query_start_loc[:-1]
    seq_lens_ptr[:num_reqs] = num_computed + query_len

    num_tokens = int(query_start_loc[-1])
    batch_idx = token_to_batch_idx(query_start_loc, num_tokens)
    pos_ptr[:num_tokens] = (
        num_computed[batch_idx] + torch.arange(num_tokens) - query_start_loc[batch_idx]
    )


def combine_sampled_and_draft_tokens(
    input_ids_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    last_sampled_tokens_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    prefill_len_ptr: torch.Tensor,
    draft_tokens_ptr: torch.Tensor,
    draft_tokens_stride: int,
    cu_num_logits_ptr: torch.Tensor,
    logits_indices_ptr: torch.Tensor,
    BLOCK_SIZE: int,
    NUM_NEW_SAMPLED_TOKENS: int = 1,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    req = idx_mapping_ptr.long()
    cu_num_logits = cu_num_logits_ptr[: num_reqs + 1].long()
    num_logits = cu_num_logits[1:] - cu_num_logits[:-1]
    query_end = query_start_loc_ptr[1 : num_reqs + 1].long()
    logits_start = query_end - num_logits

    # One entry per logit; slot i of a request holds the last sampled token
    # if i < NUM_NEW_SAMPLED_TOKENS, else draft token i - NUM_NEW_SAMPLED_TOKENS.
    total_num_logits = logits_indices_ptr.shape[0]
    batch_idx = token_to_batch_idx(cu_num_logits, total_num_logits)
    local = torch.arange(total_num_logits) - cu_num_logits[batch_idx]
    target = logits_start[batch_idx] + local
    logits_indices_ptr.copy_(target)

    seq_len = seq_lens_ptr[:num_reqs].long()
    prefill_len = prefill_len_ptr[req].long()
    is_generating = seq_len > prefill_len
    # Keep prompt-tail slots intact; only rewrite generated-token slots.
    writes_last_sampled = is_generating & (seq_len - num_logits >= prefill_len)

    tok_req = req[batch_idx]
    is_draft = local >= NUM_NEW_SAMPLED_TOKENS
    value = last_sampled_tokens_ptr.flatten()[tok_req]
    if draft_tokens_ptr.shape[-1] > 0:
        draft_col = (local - NUM_NEW_SAMPLED_TOKENS).clamp(
            0, draft_tokens_ptr.shape[-1] - 1
        )
        value = torch.where(is_draft, draft_tokens_ptr[tok_req, draft_col], value)
    write = torch.where(
        is_draft, is_generating[batch_idx], writes_last_sampled[batch_idx]
    )
    input_ids_ptr[target[write]] = value[write].to(input_ids_ptr.dtype)


def get_num_sampled_and_rejected(
    num_sampled_ptr: torch.Tensor,
    num_rejected_ptr: torch.Tensor,
    seq_lens_ptr: torch.Tensor,
    cu_num_logits_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    prefill_len_ptr: torch.Tensor,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    is_chunked_prefilling = (
        seq_lens_ptr[:num_reqs] < prefill_len_ptr[idx_mapping_ptr.long()]
    )
    num_sampled = num_sampled_ptr[:num_reqs]
    num_sampled.masked_fill_(is_chunked_prefilling, 0)
    num_logits = cu_num_logits_ptr[1 : num_reqs + 1] - cu_num_logits_ptr[:num_reqs]
    num_rejected_ptr[:num_reqs] = (num_logits - num_sampled).masked_fill_(
        is_chunked_prefilling, 0
    )


def post_update(
    idx_mapping_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    last_sampled_tokens_ptr: torch.Tensor,
    output_bin_counts_ptr: torch.Tensor | None,
    output_bin_counts_stride: int,
    sampled_tokens_ptr: torch.Tensor,
    sampled_tokens_stride: int,
    num_sampled_ptr: torch.Tensor,
    num_rejected_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor | None,
    all_token_ids_ptr: torch.Tensor,
    all_token_ids_stride: int,
    total_len_ptr: torch.Tensor,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    # Rows with negative index entries are skipped.
    batch_idx = (idx_mapping_ptr >= 0).nonzero().squeeze(1)
    req = idx_mapping_ptr[batch_idx].long()
    num_sampled = num_sampled_ptr[batch_idx].long()
    total_len = total_len_ptr[req].long()

    # [num_valid, max_sampled] mask of the tokens to append.
    sampled = sampled_tokens_ptr[batch_idx]
    is_sampled = torch.arange(sampled.shape[1]) < num_sampled.unsqueeze(1)
    rows = req.unsqueeze(1).expand_as(sampled)[is_sampled]
    tokens = sampled[is_sampled]
    cols = (total_len.unsqueeze(1) + torch.arange(sampled.shape[1]))[is_sampled]
    all_token_ids_ptr[rows, cols] = tokens.to(all_token_ids_ptr.dtype)
    if output_bin_counts_ptr is not None:
        output_bin_counts_ptr.index_put_(
            (rows, tokens.long()),
            torch.ones_like(tokens, dtype=output_bin_counts_ptr.dtype),
            accumulate=True,
        )

    has_sampled = num_sampled > 0
    last = sampled.gather(1, (num_sampled - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    last_sampled_tokens = last_sampled_tokens_ptr.flatten()
    last_sampled_tokens[req[has_sampled]] = last[has_sampled].to(
        last_sampled_tokens.dtype
    )
    total_len_ptr[req] = (total_len + num_sampled).to(total_len_ptr.dtype)

    num_rejected = num_rejected_ptr[batch_idx]
    if query_start_loc_ptr is None:
        computed_delta = -num_rejected
    else:
        query_start_loc = query_start_loc_ptr[: num_reqs + 1]
        query_len = query_start_loc[1:] - query_start_loc[:-1]
        computed_delta = query_len[batch_idx] - num_rejected
    num_computed_tokens_ptr.index_add_(
        0, req, computed_delta.to(num_computed_tokens_ptr.dtype)
    )


def post_update_num_computed_tokens(
    idx_mapping_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    query_start_loc = query_start_loc_ptr[: num_reqs + 1]
    query_len = query_start_loc[1:] - query_start_loc[:-1]
    num_computed_tokens_ptr.index_add_(
        0, idx_mapping_ptr.long(), query_len.to(num_computed_tokens_ptr.dtype)
    )


def expand_idx_mapping(
    idx_mapping_ptr: torch.Tensor,
    expanded_idx_mapping_ptr: torch.Tensor,
    expanded_local_pos_ptr: torch.Tensor,
    cu_num_logits_ptr: torch.Tensor,
    BLOCK_SIZE: int,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    cu_num_logits = cu_num_logits_ptr[: num_reqs + 1].long()
    total_num_logits = expanded_idx_mapping_ptr.shape[0]
    batch_idx = token_to_batch_idx(cu_num_logits, total_num_logits)
    expanded_idx_mapping_ptr.copy_(idx_mapping_ptr[batch_idx])
    expanded_local_pos_ptr.copy_(
        torch.arange(total_num_logits) - cu_num_logits[batch_idx]
    )


def prepare_rope_positions(
    positions_ptr: torch.Tensor,
    positions_stride: int,
    prefill_positions_ptr: torch.Tensor,
    prefill_positions_stride0: int,
    prefill_positions_stride1: int,
    prefill_delta_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    prefill_lens_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    BLOCK_SIZE: int,
    NUM_DIMS: int,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    query_start_loc = query_start_loc_ptr[: num_reqs + 1].long()
    num_tokens = int(query_start_loc[-1])
    batch_idx = token_to_batch_idx(query_start_loc, num_tokens)
    req = idx_mapping_ptr.long()[batch_idx]
    num_computed = num_computed_tokens_ptr[req].long()
    is_prefill = (num_computed < prefill_lens_ptr[req]).unsqueeze(0)
    orig_pos = num_computed + torch.arange(num_tokens) - query_start_loc[batch_idx]

    dims = torch.arange(NUM_DIMS).unsqueeze(1)
    prefill_offsets = (
        req * prefill_positions_stride0 + dims * prefill_positions_stride1 + orig_pos
    )
    prefill_positions = prefill_positions_ptr.flatten()
    prefill_offsets = prefill_offsets.clamp_max(prefill_positions.shape[0] - 1)
    decode_pos = orig_pos + prefill_delta_ptr[req].long()
    positions_ptr[:NUM_DIMS, :num_tokens] = torch.where(
        is_prefill, prefill_positions[prefill_offsets].long(), decode_pos
    )


def apply_prompt_embeds(
    inputs_embeds_ptr: torch.Tensor,
    inputs_embeds_stride: int,
    embeds_ptrs_ptr: torch.Tensor,
    mask_ptrs_ptr: torch.Tensor,
    embeds_lens_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    hidden_size: int,
    TOKEN_BLOCK: int,
    BLOCK_SIZE: int,
    *,
    grid: tuple[int, int],
) -> None:
    num_reqs = grid[0]
    query_start_loc = query_start_loc_ptr[: num_reqs + 1].tolist()
    reqs = idx_mapping_ptr[:num_reqs].tolist()
    embeds_addrs = embeds_ptrs_ptr.tolist()
    mask_addrs = mask_ptrs_ptr.tolist()
    for batch_idx, req in enumerate(reqs):
        embeds_len = int(embeds_lens_ptr[req])
        num_computed = int(num_computed_tokens_ptr[req])
        if num_computed >= embeds_len:
            continue
        query_start = query_start_loc[batch_idx]
        num_rows = min(
            query_start_loc[batch_idx + 1] - query_start, embeds_len - num_computed
        )
        src = ptr_view(
            embeds_addrs[req], inputs_embeds_ptr.dtype, embeds_len * hidden_size
        ).view(embeds_len, hidden_size)[num_computed : num_computed + num_rows]
        dst = inputs_embeds_ptr[query_start : query_start + num_rows]
        if mask_addrs[req] == 0:
            dst.copy_(src)
            continue
        is_token_id = ptr_view(mask_addrs[req], torch.int8, embeds_len)
        is_embed = is_token_id[num_computed : num_computed + num_rows] == 0
        dst[is_embed] = src[is_embed]
