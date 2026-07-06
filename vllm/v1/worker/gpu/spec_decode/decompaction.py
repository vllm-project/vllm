# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu


@dataclass
class SamplerDecompactionMetadata:
    cu_num_logits: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    draft_sampled: torch.Tensor
    pos: torch.Tensor
    target_logit_idx_mapping: torch.Tensor
    query_start_loc: torch.Tensor


@dataclass
class SamplerDecompactionBuffers:
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    target_logit_idx_mapping: torch.Tensor
    draft_sampled: torch.Tensor
    pos: torch.Tensor

    @classmethod
    def make(
        cls,
        max_num_tokens: int,
        device: torch.device,
    ) -> "SamplerDecompactionBuffers":
        expanded_idx_mapping = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        expanded_local_pos = torch.empty(
            max_num_tokens, dtype=torch.int32, device=device
        )
        target_logit_idx_mapping = torch.empty(
            max_num_tokens, dtype=torch.int64, device=device
        )
        return cls(
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            target_logit_idx_mapping=target_logit_idx_mapping,
            draft_sampled=torch.empty_like(target_logit_idx_mapping),
            pos=torch.empty_like(target_logit_idx_mapping),
        )


@triton.jit
def _prepare_sampler_decompaction_metadata_kernel(
    sampler_target_logit_idx_mapping_ptr,
    sampler_expanded_idx_mapping_ptr,
    sampler_expanded_local_pos_ptr,
    sampler_draft_sampled_ptr,
    sampler_pos_ptr,
    compact_cu_num_logits_ptr,
    full_cu_num_logits_ptr,
    compact_query_start_loc_ptr,
    idx_mapping_ptr,
    positions_ptr,
    last_sampled_tokens_ptr,
    draft_tokens_ptr,
    draft_tokens_stride,
    BLOCK_SIZE: tl.constexpr,
    NUM_NEW_SAMPLED_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    compact_start = tl.load(compact_cu_num_logits_ptr + req_idx)
    compact_end = tl.load(compact_cu_num_logits_ptr + req_idx + 1)
    compact_num_logits = compact_end - compact_start
    compact_last_local = compact_num_logits - 1
    compact_draft_tokens = compact_num_logits - NUM_NEW_SAMPLED_TOKENS

    full_start = tl.load(full_cu_num_logits_ptr + req_idx)
    full_end = tl.load(full_cu_num_logits_ptr + req_idx + 1)
    full_num_logits = full_end - full_start

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < full_num_logits
    compact_local = tl.minimum(block, compact_last_local)
    compact_logit_idx = compact_start + compact_local
    full_logit_idx = full_start + block
    tl.store(
        sampler_target_logit_idx_mapping_ptr + full_logit_idx,
        compact_logit_idx,
        mask=mask,
    )
    tl.store(
        sampler_expanded_idx_mapping_ptr + full_logit_idx,
        req_state_idx,
        mask=mask,
    )
    tl.store(
        sampler_expanded_local_pos_ptr + full_logit_idx,
        block,
        mask=mask,
    )

    compact_query_start = tl.load(compact_query_start_loc_ptr + req_idx)
    compact_input_idx = compact_query_start + compact_local
    pos = tl.load(positions_ptr + compact_input_idx, mask=mask, other=0)
    tl.store(sampler_pos_ptr + full_logit_idx, pos, mask=mask)

    last_sampled = tl.load(last_sampled_tokens_ptr + req_state_idx)
    draft_idx = block - NUM_NEW_SAMPLED_TOKENS
    is_sampled_token = block < NUM_NEW_SAMPLED_TOKENS
    is_draft_token = block >= NUM_NEW_SAMPLED_TOKENS
    is_valid_draft = is_draft_token & (draft_idx < compact_draft_tokens)
    draft_token = tl.load(
        draft_tokens_ptr + req_state_idx * draft_tokens_stride + draft_idx,
        mask=mask & is_valid_draft,
        other=0,
    )
    token = tl.where(is_sampled_token, last_sampled, draft_token)
    draft_sampled = tl.where(is_draft_token & ~is_valid_draft, -1, token)
    tl.store(sampler_draft_sampled_ptr + full_logit_idx, draft_sampled, mask=mask)


def _get_or_slice_buffer(
    buffer: torch.Tensor | None,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if buffer is None:
        return torch.empty(size, dtype=dtype, device=device)
    return buffer[:size]


def prepare_sampler_decompaction_metadata(
    compact_cu_num_logits: torch.Tensor,
    full_cu_num_logits: torch.Tensor,
    full_query_start_loc: torch.Tensor,
    compact_query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    positions: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    total_num_logits: int,
    max_num_logits_per_req: int,
    num_new_sampled_tokens: int = 1,
    expanded_idx_mapping: torch.Tensor | None = None,
    expanded_local_pos: torch.Tensor | None = None,
    target_logit_idx_mapping: torch.Tensor | None = None,
    sampler_draft_sampled: torch.Tensor | None = None,
    sampler_pos: torch.Tensor | None = None,
    buffers: SamplerDecompactionBuffers | None = None,
) -> SamplerDecompactionMetadata:
    assert num_new_sampled_tokens == 1, (
        "sampler decompaction is only supported for speculative decoding"
    )
    device = idx_mapping.device
    if buffers is not None:
        expanded_idx_mapping = buffers.expanded_idx_mapping
        expanded_local_pos = buffers.expanded_local_pos
        target_logit_idx_mapping = buffers.target_logit_idx_mapping
        sampler_draft_sampled = buffers.draft_sampled
        sampler_pos = buffers.pos
    expanded_idx_mapping = _get_or_slice_buffer(
        expanded_idx_mapping, total_num_logits, idx_mapping.dtype, device
    )
    expanded_local_pos = _get_or_slice_buffer(
        expanded_local_pos, total_num_logits, torch.int32, device
    )
    target_logit_idx_mapping = _get_or_slice_buffer(
        target_logit_idx_mapping, total_num_logits, torch.int64, device
    )
    sampler_draft_sampled = _get_or_slice_buffer(
        sampler_draft_sampled, total_num_logits, torch.int64, device
    )
    sampler_pos = _get_or_slice_buffer(
        sampler_pos, total_num_logits, positions.dtype, device
    )
    _prepare_sampler_decompaction_metadata_kernel[(idx_mapping.shape[0],)](
        target_logit_idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        sampler_draft_sampled,
        sampler_pos,
        compact_cu_num_logits,
        full_cu_num_logits,
        compact_query_start_loc,
        idx_mapping,
        positions,
        last_sampled_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(max_num_logits_per_req),
        NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
    )
    return SamplerDecompactionMetadata(
        cu_num_logits=full_cu_num_logits,
        expanded_idx_mapping=expanded_idx_mapping,
        expanded_local_pos=expanded_local_pos,
        draft_sampled=sampler_draft_sampled,
        pos=sampler_pos,
        target_logit_idx_mapping=target_logit_idx_mapping,
        query_start_loc=full_query_start_loc,
    )


def prepare_sampler_decompaction_from_counts(
    compact_cu_num_logits: torch.Tensor,
    compact_query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    positions: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_scheduled_tokens: np.ndarray,
    scheduled_draft_tokens_per_req: np.ndarray,
    num_bonus_tokens: int,
    num_reqs: int,
    num_reqs_padded: int,
    max_num_reqs: int,
    device: torch.device,
    buffers: SamplerDecompactionBuffers,
    num_new_sampled_tokens: int = 1,
) -> SamplerDecompactionMetadata:
    full_num_logits_np = scheduled_draft_tokens_per_req + num_bonus_tokens
    full_cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
    full_cu_num_logits_np[0] = 0
    np.cumsum(full_num_logits_np, out=full_cu_num_logits_np[1:])
    full_cu_num_logits = async_copy_to_gpu(full_cu_num_logits_np, device=device)

    full_query_start_loc_np = np.empty(max_num_reqs + 1, dtype=np.int32)
    full_query_start_loc_np[0] = 0
    np.cumsum(num_scheduled_tokens, out=full_query_start_loc_np[1 : num_reqs + 1])
    full_num_tokens = int(num_scheduled_tokens.sum())
    full_query_start_loc_np[num_reqs + 1 :] = full_num_tokens
    full_query_start_loc = async_copy_to_gpu(full_query_start_loc_np, device=device)[
        : num_reqs_padded + 1
    ]

    return prepare_sampler_decompaction_metadata(
        compact_cu_num_logits,
        full_cu_num_logits,
        full_query_start_loc,
        compact_query_start_loc,
        idx_mapping,
        positions,
        last_sampled_tokens,
        draft_tokens,
        int(full_cu_num_logits_np[-1]),
        int(full_num_logits_np.max()),
        num_new_sampled_tokens,
        buffers=buffers,
    )
