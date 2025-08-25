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
    max_num_tokens: int
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
