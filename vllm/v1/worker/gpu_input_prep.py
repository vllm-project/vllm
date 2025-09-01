# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numba
import numpy as np
from numba import types


# NOTE: With the type annotations, this function is pre-compiled
# before the first call.
@numba.jit(
    [
        types.int32(
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
        token_ids: np.ndarray,  # [num_reqs, max_model_len]
        num_computed_tokens: np.ndarray,  # [num_reqs]
        num_scheduled_tokens: np.ndarray,  # [num_reqs]
        # Outputs
    input_ids: np.ndarray,  # [num_input_tokens]
        query_start_loc: np.ndarray,  # [num_reqs + 1]
        seq_lens: np.ndarray,  # [num_reqs]
        positions: np.ndarray,  # [num_input_tokens]
) -> int:
    num_reqs = num_scheduled_tokens.shape[0]
    query_start_loc[0] = 0

    cu_num_tokens = 0
    for i in range(num_reqs):
        start = num_computed_tokens[i]
        end = start + num_scheduled_tokens[i]
        seq_lens[i] = end

        start_idx = cu_num_tokens
        end_idx = start_idx + num_scheduled_tokens[i]
        input_ids[start_idx:end_idx] = token_ids[i, start:end]
        positions[start_idx:end_idx] = np.arange(start, end)

        cu_num_tokens = end_idx
        query_start_loc[i + 1] = cu_num_tokens

    # Pad the inputs for CUDA graphs.
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    query_start_loc[num_reqs + 1:].fill(cu_num_tokens)
    # Fill unused with 0 for full cuda graph mode.
    seq_lens[num_reqs:].fill(0)

    return num_scheduled_tokens.max()


@numba.jit(
    [
        types.int32(
            types.int32[:],  # query_start_loc
            types.int32[:],  # num_draft_tokens
            types.int32[:],  # cu_num_draft_tokens
            types.int32[:],  # logits_indices
            types.int32[:],  # target_logits_indices
            types.int32[:],  # bonus_logits_indices
        )
    ],
    nopython=True,
    cache=True,
)
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
