# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch


def apply_draft_offsets(
    tree_draft_offsets: list[int],
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    query_start_loc_np: np.array,
    token_positions_np: np.array,
):
    """
    Updates the draft token positions with their offsets (levels) in the tree.

    Args:
        tree_draft_offsets: Offsets to apply to the draft token positions.
        input_batch: The input batch.
        scheduler_output: The scheduler output.
        query_start_loc_np: Start locations of the queries for each request.
        token_positions_np: Token positions. Will be updated in-place.
    """

    draft_token_offsets = np.array(tree_draft_offsets)
    for (
            req_id,
            draft_token_ids,
    ) in scheduler_output.scheduled_spec_decode_tokens.items():
        if len(draft_token_ids) == 0:
            continue
        req_idx = input_batch.req_id_to_index[req_id]
        start = query_start_loc_np[req_idx]
        end = query_start_loc_np[req_idx + 1]
        token_positions_np[start + 1:end] = (token_positions_np[start] +
                                             draft_token_offsets)


def apply_accepted_draft_indices(
    sampled_token_indices: list[list[int]],
    query_start_loc_np: np.array,
    token_indices_np: np.array,
):
    """
    Updates token_indices_np with the sampled token indices.

    Args:
        sampled_token_indices: The indices of the accepted draft tokens.
        query_start_loc_np: Start locations of the queries for each request.
        token_indices_np: Token indices. Will be updated in-place.
    """

    for req_idx, seq in enumerate(sampled_token_indices):
        if len(seq) <= 1:
            continue
        start = query_start_loc_np[req_idx] + 1
        end = query_start_loc_np[req_idx + 1]
        token_indices_np[start:end] = token_indices_np[start] + np.array(
            seq[:-1])


def copy_kv_cache_slots(
    kv_caches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    from_token_indices: torch.Tensor,
    to_token_indices: torch.Tensor,
):
    """
    Copies K/Vs from from_token_indices to to_token_indices. Used for updating
    the KV cache after tree rejection sampling.

    Args:
        kv_caches: List of per-layer tensors, each with shape
                   (2, num_blocks, block_size, num_kv_heads, head_size)
        slot_mapping: Tensor mapping token indices to positions in the KV
                      cache.
        from_token_indices: Tensor containing token indices into slot_mapping
                            to copy from.
        to_token_indices: Tensor containing token indices into slot_mapping to
                            copy to.
    """
    if len(kv_caches) == 0:
        # Nothing to do.
        return

    # Get block size from first kv_cache tensor.
    first_kv_cache = kv_caches[0]
    _, _, block_size, _, _ = first_kv_cache.shape

    # Ignore pairs of token indices that are the same.
    is_moved = from_token_indices != to_token_indices
    from_token_indices = from_token_indices[is_moved]
    to_token_indices = to_token_indices[is_moved]

    # Get KV cache slots.
    from_slots = slot_mapping[from_token_indices]
    to_slots = slot_mapping[to_token_indices]

    # Convert slots to blocks and offsets.
    from_blocks = from_slots // block_size
    from_offsets = from_slots % block_size
    to_blocks = to_slots // block_size
    to_offsets = to_slots % block_size

    # TODO (TheEpicDolphin): Optimize the per-layer KV cache copy.
    for kv_cache in kv_caches:
        kv_cache[:, to_blocks, to_offsets, :, :] = kv_cache[:, from_blocks,
                                                            from_offsets, :, :]
