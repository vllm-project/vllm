# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch implementations of the `gpu/sample` Triton kernels."""

import torch

from vllm.v1.worker.cpu.kernels.utils import token_to_batch_idx


def _compile(fn):
    return torch.compile(fn, dynamic=True, fullgraph=True)


@_compile
def _min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    max_val = logits.amax(dim=1, keepdim=True).to(torch.float32)
    threshold = max_val + torch.log(min_p)
    return torch.where((min_p != 0.0) & (logits < threshold), float("-inf"), logits)


@_compile
def _penalties(
    logits: torch.Tensor,
    output_bin_counts: torch.Tensor,
    packed_prompt_mask: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
) -> torch.Tensor:
    num_tokens, vocab_size = logits.shape
    bits = torch.arange(32, dtype=packed_prompt_mask.dtype)
    prompt_mask = (packed_prompt_mask.unsqueeze(-1) >> bits) & 1
    prompt_mask = prompt_mask.view(num_tokens, -1)[:, :vocab_size].bool()
    output_mask = output_bin_counts > 0
    # Identity for rows without penalties: scale 1 and zero frequency and
    # presence penalties leave the logits bitwise unchanged.
    scale = torch.where(prompt_mask | output_mask, repetition_penalty, 1.0)
    logits = logits.to(torch.float32)
    logits = logits * torch.where(logits > 0, 1.0 / scale, scale)
    logits = logits - frequency_penalty * output_bin_counts
    return logits - presence_penalty * output_mask


@_compile
def _token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logits = logits.to(torch.float32)
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    return logits.gather(1, token_ids) - lse


@_compile
def _ranks(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    x = logits.gather(1, token_ids.unsqueeze(1))
    return (logits >= x).sum(dim=1)


def apply_min_p(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    min_p_ptr: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int,
) -> None:
    num_tokens = logits_ptr.shape[0]
    req = expanded_idx_mapping_ptr[:num_tokens].long()
    min_p = min_p_ptr[req].to(torch.float32).unsqueeze(1)
    logits = logits_ptr[:, :vocab_size]
    logits.copy_(_min_p(logits, min_p))


def apply_penalties(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    token_ids_ptr: torch.Tensor,
    expanded_local_pos_ptr: torch.Tensor,
    repetition_penalty_ptr: torch.Tensor,
    frequency_penalty_ptr: torch.Tensor,
    presence_penalty_ptr: torch.Tensor,
    prompt_bin_mask_ptr: torch.Tensor,
    prompt_bin_mask_stride: int,
    output_bin_counts_ptr: torch.Tensor,
    output_bin_counts_stride: int,
    vocab_size: int,
    BLOCK_SIZE: int,
) -> None:
    num_tokens = logits_ptr.shape[0]
    req = expanded_idx_mapping_ptr[:num_tokens].long()
    output_bin_counts = output_bin_counts_ptr[req, :vocab_size]

    # Draft tokens preceding each position count as output tokens. Position
    # p of a request sees input tokens 1..p of its query.
    local_pos = expanded_local_pos_ptr[:num_tokens].long()
    max_pos = int(local_pos.max()) if num_tokens else 0
    if max_pos > 0:
        prev = torch.arange(1, max_pos + 1)
        is_prev = prev <= local_pos.unsqueeze(1)
        first = torch.arange(num_tokens) - local_pos
        rows = torch.arange(num_tokens).unsqueeze(1).expand_as(is_prev)[is_prev]
        cols = token_ids_ptr[(first.unsqueeze(1) + prev)[is_prev]].long()
        output_bin_counts.index_put_(
            (rows, cols),
            torch.ones_like(cols, dtype=output_bin_counts.dtype),
            accumulate=True,
        )

    logits = logits_ptr[:, :vocab_size]
    logits.copy_(
        _penalties(
            logits,
            output_bin_counts,
            prompt_bin_mask_ptr[req],
            repetition_penalty_ptr[req].to(torch.float32).unsqueeze(1),
            frequency_penalty_ptr[req].to(torch.float32).unsqueeze(1),
            presence_penalty_ptr[req].to(torch.float32).unsqueeze(1),
        )
    )


def bincount(
    expanded_idx_mapping_ptr: torch.Tensor,
    all_token_ids_ptr: torch.Tensor,
    all_token_ids_stride: int,
    prompt_len_ptr: torch.Tensor,
    prefill_len_ptr: torch.Tensor,
    prompt_bin_mask_ptr: torch.Tensor,
    prompt_bin_mask_stride: int,
    output_bin_counts_ptr: torch.Tensor,
    output_bin_counts_stride: int,
    BLOCK_SIZE: int,
) -> None:
    num_words = prompt_bin_mask_ptr.shape[1]
    bits = torch.arange(32, dtype=torch.int64)
    for req in expanded_idx_mapping_ptr.tolist():
        prompt_len = int(prompt_len_ptr[req])
        prefill_len = int(prefill_len_ptr[req])
        token_ids = all_token_ids_ptr[req, :prefill_len].long()

        prompt_mask = torch.zeros(num_words * 32, dtype=torch.int64)
        prompt_mask[token_ids[:prompt_len]] = 1
        packed = (prompt_mask.view(num_words, 32) << bits).sum(dim=1)
        # Bit 31 makes the word negative in int32; the cast wraps it.
        prompt_bin_mask_ptr[req] |= packed.to(prompt_bin_mask_ptr.dtype)

        output_tokens = token_ids[prompt_len:]
        output_bin_counts_ptr[req].index_add_(
            0,
            output_tokens,
            torch.ones_like(output_tokens, dtype=output_bin_counts_ptr.dtype),
        )


def _gather_ragged(
    req: torch.Tensor, num_ptr: torch.Tensor, ids_ptr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten each token's first `num_ptr[req]` entries of `ids_ptr[req]`.

    Returns the per-token counts, and the token index, value and column of
    every valid entry.
    """
    num = num_ptr[req].long()
    ids = ids_ptr[req]
    valid = torch.arange(ids.shape[1]) < num.unsqueeze(1)
    rows = torch.arange(req.shape[0]).unsqueeze(1).expand_as(ids)[valid]
    cols = torch.arange(ids.shape[1]).expand_as(ids)[valid]
    return num, rows, ids[valid].long(), cols


def apply_bias(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    vocab_size: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    num_allowed_token_ids_ptr: torch.Tensor,
    allowed_token_ids_ptr: torch.Tensor,
    allowed_token_ids_stride: int,
    num_logit_bias_ptr: torch.Tensor,
    bias_token_ids_ptr: torch.Tensor,
    bias_token_ids_stride: int,
    bias_ptr: torch.Tensor,
    bias_stride: int,
    pos_ptr: torch.Tensor,
    min_lens_ptr: torch.Tensor,
    num_stop_token_ids_ptr: torch.Tensor,
    restore_when_all_masked_ptr: torch.Tensor,
    stop_token_ids_ptr: torch.Tensor,
    stop_token_ids_stride: int,
    BLOCK_SIZE: int,
    LOGITS_BLOCK_SIZE: int,
    CHECK_ALL_MASKED_ROWS: bool,
) -> None:
    num_tokens = logits_ptr.shape[0]
    logits = logits_ptr[:, :vocab_size]
    req = expanded_idx_mapping_ptr[:num_tokens].long()

    num_allowed, rows, token_ids, _ = _gather_ragged(
        req, num_allowed_token_ids_ptr, allowed_token_ids_ptr
    )
    if rows.numel():
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep[rows, token_ids] = True
        logits.masked_fill_((num_allowed > 0).unsqueeze(1) & ~keep, float("-inf"))

    _, rows, token_ids, cols = _gather_ragged(
        req, num_logit_bias_ptr, bias_token_ids_ptr
    )
    if rows.numel():
        bias = bias_ptr[req[rows], cols].to(logits.dtype)
        logits.index_put_((rows, token_ids), bias, accumulate=True)

    num_stop, rows, token_ids, _ = _gather_ragged(
        req, num_stop_token_ids_ptr, stop_token_ids_ptr
    )
    is_active = (num_stop > 0) & (pos_ptr[:num_tokens] + 1 < min_lens_ptr[req])
    is_active = is_active[rows]
    rows, token_ids = rows[is_active], token_ids[is_active]
    if rows.numel() == 0:
        return
    stop_logits = logits[rows, token_ids]
    logits[rows, token_ids] = float("-inf")
    if CHECK_ALL_MASKED_ROWS:
        # Rows left with no finite logit get their stop tokens back.
        all_masked = logits.amax(dim=1) == float("-inf")
        restore = (
            restore_when_all_masked_ptr[req[rows]].bool()
            & all_masked[rows]
            & stop_logits.isfinite()
        )
        logits[rows[restore], token_ids[restore]] = stop_logits[restore]


def apply_bad_words(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    bad_word_token_ids_ptr: torch.Tensor,
    bad_word_token_ids_stride: int,
    bad_word_offsets_ptr: torch.Tensor,
    bad_word_offsets_stride: int,
    num_bad_words_ptr: torch.Tensor,
    all_token_ids_ptr: torch.Tensor,
    all_token_ids_stride: int,
    prompt_len_ptr: torch.Tensor,
    total_len_ptr: torch.Tensor,
    input_ids_ptr: torch.Tensor,
    expanded_local_pos_ptr: torch.Tensor,
) -> None:
    num_tokens = logits_ptr.shape[0]
    reqs = expanded_idx_mapping_ptr[:num_tokens].tolist()
    local_pos = expanded_local_pos_ptr[:num_tokens].tolist()
    for token_idx, (req, pos) in enumerate(zip(reqs, local_pos)):
        num_bad_words = int(num_bad_words_ptr[req])
        if num_bad_words == 0:
            continue
        prompt_len = int(prompt_len_ptr[req])
        output_len = int(total_len_ptr[req]) - prompt_len
        offsets = bad_word_offsets_ptr[req, : num_bad_words + 1].tolist()
        bad_word_tokens = bad_word_token_ids_ptr[req].tolist()
        # Generated tokens, followed by this position's committed and draft
        # inputs; input_ids at local position 0 is the last committed token.
        first_pos = token_idx - pos
        history = (
            all_token_ids_ptr[req, prompt_len : prompt_len + output_len].tolist()
            + input_ids_ptr[first_pos + 1 : first_pos + 1 + pos].tolist()
        )
        for start, end in zip(offsets[:-1], offsets[1:]):
            prefix = bad_word_tokens[start : end - 1]
            if len(prefix) > len(history):
                continue
            if history[len(history) - len(prefix) :] == prefix:
                logits_ptr[token_idx, bad_word_tokens[end - 1]] = float("-inf")


def topk_log_softmax(
    output_ptr: torch.Tensor,
    logits_ptr: torch.Tensor,
    logits_stride: int,
    topk_ids_ptr: torch.Tensor,
    topk_ids_stride: int,
    topk: int,
    vocab_size: int,
    BLOCK_SIZE: int,
    TOPK_BLOCK_SIZE: int,
) -> None:
    output_ptr.copy_(
        _token_logprobs(logits_ptr[:, :vocab_size], topk_ids_ptr[:, :topk])
    )


def ranks(
    output_ptr: torch.Tensor,
    logits_ptr: torch.Tensor,
    logits_stride: int,
    token_ids_ptr: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int,
) -> None:
    batch_size = output_ptr.shape[0]
    output_ptr.copy_(
        _ranks(logits_ptr[:batch_size, :vocab_size], token_ids_ptr[:batch_size].long())
    )


def fill_logprob_token_ids(
    out_token_ids_ptr: torch.Tensor,
    out_token_ids_stride: int,
    out_valid_mask_ptr: torch.Tensor,
    out_valid_mask_stride: int,
    sampled_token_ids_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    topk_indices_stride: int,
    expanded_idx_mapping_ptr: torch.Tensor,
    num_per_req_token_ids_ptr: torch.Tensor,
    per_req_token_ids_ptr: torch.Tensor,
    per_req_token_ids_stride: int,
    NUM_TOPK: int,
    PADDED_COLS: int,
) -> None:
    batch_size, num_cols = out_token_ids_ptr.shape
    num_cols -= 1
    out_token_ids_ptr[:, 0] = sampled_token_ids_ptr[:batch_size]
    out_valid_mask_ptr[:, 0] = True

    req = expanded_idx_mapping_ptr[:batch_size].long()
    num_custom = num_per_req_token_ids_ptr[req].unsqueeze(1)
    col = torch.arange(num_cols)
    tokens = torch.zeros(batch_size, num_cols, dtype=torch.int64)
    width = min(num_cols, per_req_token_ids_ptr.shape[1])
    custom = per_req_token_ids_ptr[req, :width].long()
    if NUM_TOPK > 0:
        tokens[:, :NUM_TOPK] = topk_indices_ptr[:batch_size, :NUM_TOPK]
    tokens[:, :width] = torch.where(num_custom > 0, custom, tokens[:, :width])
    valid = torch.where(num_custom > 0, col < num_custom, col < NUM_TOPK)
    out_token_ids_ptr[:, 1:] = torch.where(valid, tokens, out_token_ids_ptr[:, 1:])
    out_valid_mask_ptr[:, 1:] |= valid


def prompt_logprobs_token_ids(
    prompt_logprobs_token_ids_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    all_token_ids_ptr: torch.Tensor,
    all_token_ids_stride: int,
    BLOCK_SIZE: int,
) -> None:
    num_reqs = idx_mapping_ptr.shape[0]
    query_start_loc = query_start_loc_ptr[: num_reqs + 1].long()
    num_tokens = int(query_start_loc[-1])
    batch_idx = token_to_batch_idx(query_start_loc, num_tokens)
    req = idx_mapping_ptr.long()[batch_idx]
    # Shifted by one: the logprob at a position is for the next token.
    target_pos = (
        num_computed_tokens_ptr[req].long()
        + 1
        + torch.arange(num_tokens)
        - query_start_loc[batch_idx]
    )
    target_pos = target_pos.clamp_max(all_token_ids_ptr.shape[1] - 1)
    prompt_logprobs_token_ids_ptr[:num_tokens] = all_token_ids_ptr[req, target_pos]


def compact_sampling_mask(
    logits_ptr: torch.Tensor,
    logits_row_stride: int,
    logits_col_stride: int,
    cu_num_logits_ptr: torch.Tensor,
    num_sampled_tokens_ptr: torch.Tensor,
    token_ids_ptr: torch.Tensor,
    token_ids_row_stride: int,
    packed_mask_ptr: torch.Tensor,
    packed_mask_row_stride: int,
    counts_ptr: torch.Tensor,
    vocab_size: int,
    max_num_kept: int,
    ROWS_PER_REQUEST: int,
    BLOCK_SIZE: int,
) -> None:
    num_rows = counts_ptr.shape[0]
    row = torch.arange(num_rows)
    req = row // ROWS_PER_REQUEST
    slot = row % ROWS_PER_REQUEST
    is_active = slot < num_sampled_tokens_ptr[req]
    source_row = (cu_num_logits_ptr[req].long() + slot).clamp_max(
        logits_ptr.shape[0] - 1
    )
    logits = logits_ptr[source_row, :vocab_size]
    keep = logits.isfinite() & is_active.unsqueeze(1)
    counts_ptr.copy_(keep.sum(dim=1))

    pos = keep.cumsum(dim=1) - 1
    is_stored = keep & (pos < max_num_kept)
    rows = row.unsqueeze(1).expand_as(keep)[is_stored]
    token_ids = torch.arange(vocab_size).expand_as(keep)[is_stored]
    token_ids_ptr[rows, pos[is_stored]] = token_ids.to(token_ids_ptr.dtype)

    num_bytes = packed_mask_ptr.shape[1]
    padded = torch.zeros(num_rows, num_bytes * 8, dtype=torch.int32)
    padded[:, :vocab_size] = keep
    bits = padded.view(num_rows, num_bytes, 8) << torch.arange(8, dtype=torch.int32)
    packed_mask_ptr.copy_(bits.sum(dim=2).to(torch.uint8))


def apply_grammar_bitmask(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    logits_indices_ptr: torch.Tensor,
    cu_num_logits_ptr: torch.Tensor,
    bitmask_ptr: torch.Tensor,
    bitmask_stride: int,
    vocab_size: int,
    MASK_STRIDE: int,
    BLOCK_SIZE: int,
) -> None:
    num_masks, num_words = bitmask_ptr.shape
    mapping_idx = logits_indices_ptr[:num_masks].long()
    req = mapping_idx // MASK_STRIDE
    position = mapping_idx % MASK_STRIDE
    logits_start = cu_num_logits_ptr[req].long()
    is_active = position < cu_num_logits_ptr[req + 1].long() - logits_start
    rows = (logits_start + position)[is_active]

    bits = torch.arange(32, dtype=bitmask_ptr.dtype)
    allowed = (bitmask_ptr[is_active].unsqueeze(-1) >> bits) & 1
    allowed = allowed.view(rows.shape[0], num_words * 32)[:, :vocab_size].bool()
    logits = logits_ptr[rows, :vocab_size]
    logits_ptr[rows, :vocab_size] = logits.masked_fill(~allowed, float("-inf"))


def num_nans(
    logits_ptr: torch.Tensor,
    logits_stride: int,
    num_nans_ptr: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int,
) -> None:
    num_nans_ptr.copy_(logits_ptr[:, :vocab_size].isnan().sum(dim=1))
