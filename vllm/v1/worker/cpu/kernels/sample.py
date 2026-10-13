# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch implementations of the `gpu/sample` Triton kernels."""

import torch

from vllm.v1.worker.cpu.kernels.utils import numba_kernel


def _compile(fn):
    return torch.compile(fn, dynamic=True, fullgraph=True)


@_compile
def _token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logits = logits.to(torch.float32)
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    return logits.gather(1, token_ids) - lse


@_compile
def _ranks(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    x = logits.gather(1, token_ids.unsqueeze(1))
    return (logits >= x).sum(dim=1)


def topk_log_softmax(
    output_ptr: torch.Tensor,
    logits_ptr: torch.Tensor,
    topk_ids_ptr: torch.Tensor,
    topk: int,
    vocab_size: int,
) -> None:
    output_ptr.copy_(
        _token_logprobs(logits_ptr[:, :vocab_size], topk_ids_ptr[:, :topk])
    )


def ranks(
    output_ptr: torch.Tensor,
    logits_ptr: torch.Tensor,
    token_ids_ptr: torch.Tensor,
    vocab_size: int,
) -> None:
    batch_size = output_ptr.shape[0]
    output_ptr.copy_(
        _ranks(logits_ptr[:batch_size, :vocab_size], token_ids_ptr[:batch_size].long())
    )


def fill_logprob_token_ids(
    out_token_ids_ptr: torch.Tensor,
    out_valid_mask_ptr: torch.Tensor,
    sampled_token_ids_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    expanded_idx_mapping_ptr: torch.Tensor,
    num_per_req_token_ids_ptr: torch.Tensor,
    per_req_token_ids_ptr: torch.Tensor,
    NUM_TOPK: int,
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


@numba_kernel
def prompt_logprobs_token_ids(
    grid,
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
):
    for batch_idx in range(grid[0]):
        req = idx_mapping_ptr[batch_idx]
        start = query_start_loc_ptr[batch_idx]
        end = query_start_loc_ptr[batch_idx + 1]
        last_col = all_token_ids_ptr.shape[1] - 1
        for i in range(end - start):
            # Shifted by one: the logprob at a position is for the next token.
            col = min(num_computed_tokens_ptr[req] + 1 + i, last_col)
            prompt_logprobs_token_ids_ptr[start + i] = all_token_ids_ptr[req, col]


def apply_grammar_bitmask(
    logits_ptr: torch.Tensor,
    logits_indices_ptr: torch.Tensor,
    cu_num_logits_ptr: torch.Tensor,
    bitmask_ptr: torch.Tensor,
    vocab_size: int,
    MASK_STRIDE: int,
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
    num_nans_ptr: torch.Tensor,
    vocab_size: int,
) -> None:
    num_nans_ptr.copy_(logits_ptr[:, :vocab_size].isnan().sum(dim=1))
