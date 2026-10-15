# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numba ports of the model runner's input preparation Triton kernels."""

import numba
import torch

from vllm.v1.worker.cpu.kernels.utils import numba_kernel, ptr_view


@numba.njit(cache=True, nogil=True)
def prepare_inputs(
    input_ids,
    positions,
    seq_lens,
    next_prefill_tokens,
    idx_mapping,
    query_start_loc,
    all_token_ids,
    prefill_lens,
    num_computed_tokens,
    last_sampled_tokens,
):
    """Fills the inputs of a batch with one sampled token per request."""
    num_reqs = idx_mapping.shape[0]
    seq_lens[num_reqs:] = 0
    for batch_idx in range(num_reqs):
        req = idx_mapping[batch_idx]
        prefill_len = prefill_lens[req]
        num_computed = num_computed_tokens[req]
        start = query_start_loc[batch_idx]
        end = query_start_loc[batch_idx + 1]
        seq_len = num_computed + end - start
        seq_lens[batch_idx] = seq_len
        for i in range(end - start):
            positions[start + i] = num_computed + i
        if num_computed < prefill_len:
            input_ids[start:end] = all_token_ids[req, num_computed:seq_len]
            for i in range(next_prefill_tokens.shape[0]):
                pos = seq_len + i
                next_prefill_tokens[i, req] = (
                    all_token_ids[req, pos] if pos < prefill_len else 0
                )
        if seq_len > prefill_len:
            input_ids[end - 1] = last_sampled_tokens[req, 0]


@numba_kernel
def get_num_sampled_and_rejected(
    grid,
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
):
    for batch_idx in range(grid[0]):
        num_logits = cu_num_logits_ptr[batch_idx + 1] - cu_num_logits_ptr[batch_idx]
        if seq_lens_ptr[batch_idx] < prefill_len_ptr[idx_mapping_ptr[batch_idx]]:
            num_sampled_ptr[batch_idx] = 0
            num_rejected_ptr[batch_idx] = 0
        else:
            num_rejected_ptr[batch_idx] = num_logits - num_sampled_ptr[batch_idx]


@numba_kernel
def post_update(
    grid,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    sampled_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    total_len_ptr,
):
    for batch_idx in range(grid[0]):
        req = idx_mapping_ptr[batch_idx]
        if req < 0:
            continue
        total_len = total_len_ptr[req]
        num_sampled = num_sampled_ptr[batch_idx]
        if num_sampled > 0:
            last_sampled_tokens_ptr.flat[req] = sampled_tokens_ptr[
                batch_idx, num_sampled - 1
            ]
            total_len_ptr[req] = total_len + num_sampled
        for i in range(num_sampled):
            token_id = sampled_tokens_ptr[batch_idx, i]
            all_token_ids_ptr[req, total_len + i] = token_id
            if output_bin_counts_ptr is not None:
                output_bin_counts_ptr[req, token_id] += 1

        num_computed_tokens_ptr[req] -= num_rejected_ptr[batch_idx]
        if query_start_loc_ptr is not None:
            num_computed_tokens_ptr[req] += (
                query_start_loc_ptr[batch_idx + 1] - query_start_loc_ptr[batch_idx]
            )


@numba_kernel
def post_update_num_computed_tokens(
    grid, idx_mapping_ptr, num_computed_tokens_ptr, query_start_loc_ptr
):
    for batch_idx in range(grid[0]):
        num_computed_tokens_ptr[idx_mapping_ptr[batch_idx]] += (
            query_start_loc_ptr[batch_idx + 1] - query_start_loc_ptr[batch_idx]
        )


@numba_kernel
def prepare_rope_positions(
    grid,
    positions_ptr,
    prefill_positions_ptr,
    prefill_delta_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    prefill_lens_ptr,
    num_computed_tokens_ptr,
    NUM_DIMS,
):
    for batch_idx in range(grid[0]):
        req = idx_mapping_ptr[batch_idx]
        num_computed = num_computed_tokens_ptr[req]
        is_prefill = num_computed < prefill_lens_ptr[req]
        start = query_start_loc_ptr[batch_idx]
        end = query_start_loc_ptr[batch_idx + 1]
        for j in range(NUM_DIMS):
            for i in range(end - start):
                if is_prefill:
                    pos = prefill_positions_ptr[req * NUM_DIMS + j, num_computed + i]
                else:
                    pos = num_computed + i + prefill_delta_ptr[req]
                positions_ptr[j, start + i] = pos


def apply_prompt_embeds(
    inputs_embeds_ptr: torch.Tensor,
    embeds_ptrs_ptr: torch.Tensor,
    mask_ptrs_ptr: torch.Tensor,
    embeds_lens_ptr: torch.Tensor,
    idx_mapping_ptr: torch.Tensor,
    query_start_loc_ptr: torch.Tensor,
    num_computed_tokens_ptr: torch.Tensor,
    hidden_size: int,
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
