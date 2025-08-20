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


# NOTE: With the type annotations, this function is pre-compiled
# before the first call.
@numba.jit(
    [
        types.none(
            types.int64[:],  # positions
            types.int32[:],  # query_start_loc
            types.int32[:, :],  # block_table
            types.int32,  # block_size
            types.int64[:],  # slot_mapping
        )
    ],
    nopython=True,
    cache=True,
)
def compute_slot_mapping(
    # Inputs
    positions: np.ndarray,  # [num_input_tokens]
    query_start_loc: np.ndarray,  # [num_reqs + 1]
    block_table: np.ndarray,  # [num_reqs, max_num_blocks_per_req]
    block_size: int,
    # Outputs
    slot_mapping: np.ndarray,  # [num_input_tokens]
) -> None:
    num_reqs = block_table.shape[0]
    for i in range(num_reqs):
        start_idx = query_start_loc[i]
        end_idx = query_start_loc[i + 1]

        pos = positions[start_idx:end_idx]
        block_ids = pos // block_size
        block_numbers = block_table[i, block_ids]
        block_offsets = pos % block_size
        slot_mapping[start_idx:end_idx] = (block_numbers * block_size +
                                           block_offsets)

    # Fill unused with -1.
    # Needed for reshape_and_cache in full cuda graph mode.
    slot_mapping[end_idx:].fill(-1)
