# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import torch
from numba import types

from vllm.v1.utils import CpuGpuBuffer


class InputBuffers:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device
        self.pin_memory = pin_memory

        self.idx_mapping = self._make_buffer(max_num_reqs, dtype=torch.int32)
        self.input_ids = self._make_buffer(max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(max_num_reqs + 1,
                                                 dtype=torch.int32)
        self.seq_lens = self._make_buffer(max_num_reqs, dtype=torch.int32)

    def _make_buffer(self, *args, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(*args,
                            dtype=dtype,
                            pin_memory=self.pin_memory,
                            device=self.device)


@dataclass
class InputBatch:

    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # sum(num_scheduled_tokens)
    num_tokens: int
    # [num_reqs]
    is_chunked_prefilling: np.ndarray

    # [max_num_batched_tokens]
    input_ids: torch.Tensor
    # [max_num_batched_tokens]
    positions: torch.Tensor

    # layer_name -> Metadata
    attn_metadata: dict[str, Any]

    # [num_reqs]
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
            types.int64[:],  # positions
            types.int32[:],  # query_start_loc
            types.int32[:],  # seq_lens
        )
    ],
    nopython=True,
    cache=True,
)
def prepare_inputs(
        idx_mapping: np.ndarray,  # batch_idx -> req_idx
        token_ids: np.ndarray,  # [N, max_model_len]
        num_computed_tokens: np.ndarray,  # [N]
        num_scheduled_tokens: np.ndarray,  # [B]
        input_ids: np.ndarray,  # [num_input_tokens]
        positions: np.ndarray,  # [num_input_tokens]
        query_start_loc: np.ndarray,  # [B + 1]
        seq_lens: np.ndarray,  # [B]
) -> None:
    num_reqs = num_scheduled_tokens.shape[0]
    query_start_loc[0] = 0

    cu_num_tokens = 0
    for i in range(num_reqs):
        req_idx = idx_mapping[i]
        query_len = num_scheduled_tokens[i]
        start = num_computed_tokens[req_idx]
        end = start + query_len
        seq_lens[i] = end

        start_idx = cu_num_tokens
        end_idx = start_idx + query_len
        input_ids[start_idx:end_idx] = token_ids[req_idx, start:end]
        positions[start_idx:end_idx] = np.arange(start, end, dtype=np.int64)

        cu_num_tokens = end_idx
        query_start_loc[i + 1] = cu_num_tokens

    # Pad the inputs for CUDA graphs.
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    query_start_loc[num_reqs + 1:].fill(cu_num_tokens)
    # Fill unused with 0 for full cuda graph mode.
    seq_lens[num_reqs:].fill(0)
