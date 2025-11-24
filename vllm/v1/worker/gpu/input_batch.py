# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils import random_uuid
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer


class InputBuffers:
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.device = device
        self.pin_memory = pin_memory

        self.idx_mapping = self._make_buffer(max_num_reqs, dtype=torch.int32)
        self.input_ids = self._make_buffer(max_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.query_start_loc = self._make_buffer(max_num_reqs + 1, dtype=torch.int32)
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.cu_num_logits = self._make_buffer(max_num_reqs + 1, dtype=torch.int32)

        # Spec decoding.
        self.next_prefill_tokens = self._make_buffer(max_num_reqs, dtype=torch.int32)

        # Structured outputs.
        self.bitmask_indices = self._make_buffer(max_num_reqs, dtype=torch.int32)
        self.grammar_bitmask = self._make_buffer(
            max_num_reqs, cdiv(vocab_size, 32), dtype=torch.int32
        )

    def _make_buffer(self, *args, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *args, dtype=dtype, pin_memory=self.pin_memory, device=self.device
        )


@dataclass
class InputBatch:
    # batch_idx -> req_id
    req_ids: list[str]
    num_reqs: int

    # batch_idx -> req_state_idx
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    # [num_reqs]
    # batch_idx -> num_scheduled_tokens
    num_scheduled_tokens: np.ndarray
    # sum(num_scheduled_tokens)
    num_tokens: int
    num_tokens_after_padding: int
    num_draft_tokens: int

    # [num_reqs + 1]
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    # [num_reqs]
    seq_lens: torch.Tensor
    seq_lens_np: np.ndarray

    # [num_tokens_after_padding]
    input_ids: torch.Tensor
    # [num_tokens_after_padding]
    positions: torch.Tensor

    # layer_name -> Metadata
    attn_metadata: dict[str, Any]

    # [total_num_logits]
    logits_indices: torch.Tensor
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor

    @classmethod
    def make_dummy(
        cls,
        num_reqs: int,
        num_tokens: int,
        input_buffers: InputBuffers,
        device: torch.device,
    ) -> "InputBatch":
        assert 0 < num_reqs <= num_tokens
        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
        num_scheduled_tokens = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        num_scheduled_tokens[-1] += num_tokens % num_reqs
        assert int(num_scheduled_tokens.sum()) == num_tokens

        input_buffers.query_start_loc.np[0] = 0
        input_buffers.query_start_loc.np[1 : num_reqs + 1] = np.cumsum(
            num_scheduled_tokens
        )
        input_buffers.query_start_loc.np[num_reqs + 1 :] = num_tokens
        query_start_loc_np = input_buffers.query_start_loc.np[: num_reqs + 1]
        query_start_loc = input_buffers.query_start_loc.copy_to_gpu()[: num_reqs + 1]
        # seq_len equals to query_len
        seq_lens_np = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
        seq_lens_np[-1] += num_tokens % num_reqs
        input_buffers.seq_lens[:num_reqs] = num_tokens // num_reqs
        input_buffers.seq_lens[num_reqs - 1] += num_tokens % num_reqs
        input_buffers.seq_lens[num_reqs:] = 0
        seq_lens = input_buffers.seq_lens[:num_reqs]

        input_ids = input_buffers.input_ids.copy_to_gpu(num_tokens)
        positions = input_buffers.positions[:num_tokens]
        # attn_metadata = defaultdict(lambda: None)
        logits_indices = query_start_loc[1:] - 1
        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)
        return cls(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=0,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_np=seq_lens_np,
            input_ids=input_ids,
            positions=positions,
            attn_metadata=None,  # type: ignore
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
        )


@numba.njit(cache=True)
def _prepare_prefill_inputs(
    idx_mapping: np.ndarray,  # [B]
    query_lens: np.ndarray,  # [B]
    query_start_loc: np.ndarray,  # [B + 1]
    prefill_token_ids: np.ndarray,  # [N, max_model_len]
    num_computed_prefill_tokens: np.ndarray,  # [N]
    input_ids: np.ndarray,  # [num_input_tokens]
) -> None:
    num_reqs = idx_mapping.shape[0]
    query_starts = query_start_loc[:num_reqs]
    query_ends = query_start_loc[1 : num_reqs + 1]
    starts = num_computed_prefill_tokens[idx_mapping]
    ends = starts + query_lens
    for i in range(num_reqs):
        input_ids[query_starts[i] : query_ends[i]] = prefill_token_ids[
            idx_mapping[i], starts[i] : ends[i]
        ]


def prepare_prefill_inputs(
    idx_mapping: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    query_start_loc: np.ndarray,
    prefill_token_ids: np.ndarray,
    num_computed_prefill_tokens: np.ndarray,
    input_ids: np.ndarray,
) -> None:
    _prepare_prefill_inputs(
        idx_mapping,
        num_scheduled_tokens,
        query_start_loc,
        prefill_token_ids,
        num_computed_prefill_tokens,
        input_ids,
    )


@triton.jit
def _prepare_pos_seq_lens_kernel(
    pos_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_computed_tokens_ptr,
    max_num_reqs,
    BLOCK_SIZE: tl.constexpr,
):
    req_id = tl.program_id(0)
    num_reqs = tl.num_programs(0) - 1
    if req_id == num_reqs:
        # Pad unused seq_lens as 0 for full CUDA graphs.
        for i in tl.range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(seq_lens_ptr + block, 0, mask=mask)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_id)
    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)

    start = tl.load(query_start_loc_ptr + req_id)
    end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = end - start

    seq_len = num_computed_tokens + query_len
    tl.store(seq_lens_ptr + req_id, seq_len)

    for i in tl.range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        pos = num_computed_tokens + block
        tl.store(pos_ptr + start + block, pos, mask=mask)


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    # NOTE(woosuk): We do +1 because the last thread block is used
    # to pad unused seq_lens as 0 for full CUDA graphs.
    _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
        pos,
        seq_lens,
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        seq_lens.shape[0],
        BLOCK_SIZE=1024,
    )


@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    cu_num_logits_ptr,
    logits_indices_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    # Get the number of logits and draft tokens.
    cu_num_logits_start = tl.load(cu_num_logits_ptr + batch_idx)
    cu_num_logits_end = tl.load(cu_num_logits_ptr + batch_idx + 1)
    num_logits = cu_num_logits_end - cu_num_logits_start
    num_draft_tokens = num_logits - 1

    # Compute the logits indices.
    block = tl.arange(0, BLOCK_SIZE)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    logits_start = query_end - num_logits
    tl.store(
        logits_indices_ptr + cu_num_logits_start + block,
        logits_start + block,
        mask=block < num_logits,
    )

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    if seq_len <= prefill_len:
        # Handling prefill tokens. No sampled or draft tokens.
        return

    # Write the last sampled token ID to input_ids.
    last_token_id = tl.load(last_sampled_tokens_ptr + req_state_idx)
    tl.store(input_ids_ptr + query_end - num_logits, last_token_id)

    # Write the draft tokens (if any) to input_ids.
    if num_draft_tokens > 0:
        mask = block < num_draft_tokens
        draft_tokens = tl.load(
            draft_tokens_ptr + req_state_idx * draft_tokens_stride + block,
            mask=mask,
        )
        tl.store(
            input_ids_ptr + query_end - num_draft_tokens + block,
            draft_tokens,
            mask=mask,
        )


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor:
    num_reqs = seq_lens.shape[0]
    num_speculative_steps = draft_tokens.shape[-1]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )
    _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        draft_tokens.stride(0),
        cu_num_logits,
        logits_indices,
        # NOTE(woosuk): Add 1 to ensure the block can cover the last sampled token
        # in addition to all draft tokens.
        BLOCK_SIZE=triton.next_power_of_2(num_speculative_steps + 1),
    )
    return logits_indices


@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    query_start_loc_ptr,
    cu_num_logits_ptr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)

    num_sampled = tl.load(num_sampled_ptr + req_id)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)

    query_start = tl.load(query_start_loc_ptr + req_id)
    query_end = tl.load(query_start_loc_ptr + req_id + 1)
    query_len = query_end - query_start

    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    num_computed += query_len
    # Consider the rejected tokens in spec decoding.
    if num_sampled > 0:
        # NOTE(woosuk): We must skip num_sampled == 0 to account for chunked prefills.
        logits_start = tl.load(cu_num_logits_ptr + req_id)
        logits_end = tl.load(cu_num_logits_ptr + req_id + 1)
        num_logits = logits_end - logits_start
        num_rejected = num_logits - num_sampled
        num_computed -= num_rejected
    tl.store(num_computed_tokens_ptr + req_state_idx, num_computed)


def post_update(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
) -> None:
    num_reqs = idx_mapping.shape[0]
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        sampled_tokens,
        sampled_tokens.stride(0),
        num_sampled,
        query_start_loc,
        cu_num_logits,
        num_warps=1,
    )
