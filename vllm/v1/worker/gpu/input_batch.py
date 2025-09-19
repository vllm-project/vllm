# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numba
import numba.types as types
import numpy as np
import torch
import triton
import triton.language as tl

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
    num_tokens_after_padding: int
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

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        device: torch.device,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        req_ids = [f"req_{i}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.tensor(idx_mapping_np, device=device)
        num_scheduled_tokens = np.full(num_reqs,
                                       num_tokens // num_reqs,
                                       dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        is_chunked_prefilling = np.zeros(num_reqs, dtype=np.bool_)
        input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        attn_metadata = defaultdict(lambda: None)
        logits_indices = torch.arange(num_reqs,
                                      dtype=torch.int32,
                                      device=device)
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            is_chunked_prefilling=is_chunked_prefilling,
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            logits_indices=logits_indices,
        )


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
def _prepare_inputs(
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


def prepare_inputs(
    idx_mapping: np.ndarray,
    prompt_token_ids: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    input_ids: CpuGpuBuffer,
    positions: CpuGpuBuffer,
    query_start_loc: CpuGpuBuffer,
    seq_lens: CpuGpuBuffer,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    _prepare_inputs(
        idx_mapping,
        prompt_token_ids,
        num_computed_tokens,
        num_scheduled_tokens,
        input_ids.np,
        positions.np,
        query_start_loc.np,
        seq_lens.np,
    )
    input_ids.copy_to_gpu(num_tokens)
    positions.copy_to_gpu(num_tokens)
    # NOTE(woosuk): We should copy the whole query_start_loc and seq_lens
    # tensors from CPU to GPU, because they may include paddings needed
    # for full CUDA graph mode.
    query_start_loc.copy_to_gpu()
    seq_lens.copy_to_gpu()

    num_reqs = num_scheduled_tokens.shape[0]
    max_query_len = int(num_scheduled_tokens.max())
    max_seq_len = int(seq_lens.np[:num_reqs].max())
    return max_query_len, max_seq_len


@triton.jit
def _combine_last_token_ids_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_token_ids_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    num_tokens_ptr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    num_tokens = tl.load(num_tokens_ptr + req_state_idx)
    if seq_len < num_tokens:
        # Chunked prefilling.
        return

    last_token_id = tl.load(last_token_ids_ptr + req_state_idx)
    if last_token_id == -1:
        return

    end = tl.load(query_start_loc_ptr + batch_idx + 1)
    tl.store(input_ids_ptr + end - 1, last_token_id)


def combine_last_token_ids(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_tokens: torch.Tensor,
) -> torch.Tensor:
    num_reqs = seq_lens.shape[0]
    _combine_last_token_ids_kernel[(num_reqs, )](
        input_ids,
        idx_mapping,
        last_token_ids,
        query_start_loc,
        seq_lens,
        num_tokens,
    )
    return input_ids
