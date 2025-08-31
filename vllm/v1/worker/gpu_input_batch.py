# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any, Optional

import numba
import numpy as np
import torch
from numba import types

from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


@dataclass
class InputBatch:

    # batch_idx -> req_id
    req_ids: list[str]

    # req_id -> batch_idx
    req_id_to_batch_idx: dict[str, int]

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    total_num_tokens: int
    max_query_len: int
    num_reqs: int

    attn_metadata: dict[str, Any]
    spec_decode_common_attn_metadata: Optional[Any]
    spec_decode_metadata: Optional[SpecDecodeMetadata]

    logits_indices: torch.Tensor


# NOTE: With the type annotations, this function is pre-compiled
# before the first call.
@numba.jit(
    [
        types.none(
            types.int32[:],  # idx_mapping
            types.int32[:, :],  # token_ids
            types.int32[:],  # num_computed_tokens
            types.int32[:],  # num_scheduled_tokens
            types.int32[:],  # input_ids
            types.int32[:],  # query_start_loc
            types.int32[:],  # seq_lens
            types.int64[:],  # positions
        )
    ],
    nopython=True,
    cache=True,
)
def prepare_inputs(
        # Inputs
        idx_mapping: np.ndarray,  # batch_idx -> req_idx
        token_ids: np.ndarray,  # [N, max_model_len]
        num_computed_tokens: np.ndarray,  # [N]
        num_scheduled_tokens: np.ndarray,  # [B]
        # Outputs
    input_ids: np.ndarray,  # [num_input_tokens]
        query_start_loc: np.ndarray,  # [B + 1]
        seq_lens: np.ndarray,  # [B]
        positions: np.ndarray,  # [num_input_tokens]
) -> None:
    num_reqs = num_scheduled_tokens.shape[0]
    query_start_loc[0] = 0

    cu_num_tokens = 0
    for i in range(num_reqs):
        req_idx = idx_mapping[i]
        start = num_computed_tokens[req_idx]
        end = start + num_scheduled_tokens[i]
        seq_lens[i] = end

        start_idx = cu_num_tokens
        end_idx = start_idx + num_scheduled_tokens[i]
        input_ids[start_idx:end_idx] = token_ids[req_idx, start:end]
        positions[start_idx:end_idx] = np.arange(start, end)

        cu_num_tokens = end_idx
        query_start_loc[i + 1] = cu_num_tokens

    # Pad the inputs for CUDA graphs.
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    query_start_loc[num_reqs + 1:].fill(cu_num_tokens)
    # Fill unused with 0 for full cuda graph mode.
    seq_lens[num_reqs:].fill(0)


def prepare_spec_decode(
        # Inputs
        query_start_loc: np.ndarray,  # [B + 1]
        num_draft_tokens: np.ndarray,  # [B]
        # Outputs
    cu_num_draft_tokens: np.ndarray,  # [B]
        logits_indices: np.ndarray,  # [N + B]
        target_logits_indices: np.ndarray,  # [N]
        bonus_logits_indices: np.ndarray,  # [B]
) -> int:  # N
    # Inputs:
    # query_start_loc:          [  0,   4, 104, 107, 207, 209]
    # num_draft_tokens:         [  3,   0,   2,   0,   1]
    # Outputs:
    # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
    # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
    #                            206, 207, 208]
    # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
    # bonus_logits_indices:     [  3,   4,   7,   8,  10]
    # return:                   6 (total number of draft tokens)

    cu_num_draft = 0
    cu_num_sample = 0
    num_reqs = num_draft_tokens.shape[0]
    for i in range(num_reqs):
        q_end_idx = query_start_loc[i + 1]
        draft_len = num_draft_tokens[i]

        # The last draft_len + 1 query tokens are used for sampling.
        sample_len = draft_len + 1
        sample_start_idx = cu_num_sample
        sample_end_idx = sample_start_idx + sample_len
        logits_indices[sample_start_idx:sample_end_idx] = (np.arange(
            q_end_idx - sample_len, q_end_idx))

        # For each query, the first draft_len tokens need target logits for
        # rejection sampling. The draft_len + 1th token is used for bonus token.
        draft_start_idx = cu_num_draft
        draft_end_idx = draft_start_idx + draft_len
        target_logits_indices[draft_start_idx:draft_end_idx] = (np.arange(
            sample_start_idx, sample_end_idx - 1))
        bonus_logits_indices[i] = sample_end_idx - 1

        cu_num_draft += draft_len
        cu_num_draft_tokens[i] = cu_num_draft
        cu_num_sample += sample_len

    return cu_num_draft
