# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d_generic import (
    generic_causal_conv1d,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _resolve_apc_state_indices_kernel(
    cache_indices_ptr,
    initial_state_idx_ptr,
    last_scheduled_ptr,
    input_indices_ptr,
    output_indices_ptr,
    batch,
    stride_cache_batch,
    BLOCK: tl.constexpr,
):
    sequence = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = sequence < batch
    initial_offset = tl.load(initial_state_idx_ptr + sequence, mask=mask, other=0)
    final_offset = tl.load(last_scheduled_ptr + sequence, mask=mask, other=0)
    input_index = tl.load(
        cache_indices_ptr + sequence * stride_cache_batch + initial_offset,
        mask=mask,
        other=0,
    )
    output_index = tl.load(
        cache_indices_ptr + sequence * stride_cache_batch + final_offset,
        mask=mask,
        other=0,
    )
    tl.store(input_indices_ptr + sequence, input_index, mask=mask)
    tl.store(output_indices_ptr + sequence, output_index, mask=mask)


@triton.jit
def _store_apc_intermediate_states_kernel(
    x_ptr,
    states_ptr,
    query_start_ptr,
    cache_indices_ptr,
    first_scheduled_ptr,
    last_scheduled_ptr,
    num_computed_ptr,
    batch,
    dim,
    max_blocks,
    block_size,
    stride_x_dim,
    stride_x_token,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    stride_cache_batch,
    pad_slot_id,
    null_block_id,
    STATE_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    sequence = tl.program_id(0)
    block_offset = tl.program_id(1)
    channel = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    state_token = tl.arange(0, BLOCK_STATE)
    channel_mask = channel < dim
    state_mask = state_token < STATE_LEN

    sequence_start = tl.load(query_start_ptr + sequence)
    sequence_end = tl.load(query_start_ptr + sequence + 1)
    sequence_length = sequence_end - sequence_start
    first = tl.load(first_scheduled_ptr + sequence)
    last = tl.load(last_scheduled_ptr + sequence)
    blocks_to_fill = last - first
    computed = tl.load(num_computed_ptr + sequence)
    active_block = block_offset < blocks_to_fill

    completed_offset = block_size - (computed % block_size)
    end_offset = (sequence_length - completed_offset) % block_size
    last_full = sequence_end - end_offset
    last_full = tl.where(end_offset == 0, last_full - block_size, last_full)
    boundary = last_full - (blocks_to_fill - block_offset - 1) * block_size
    input_token = boundary - STATE_LEN + state_token

    cache_index = tl.load(
        cache_indices_ptr + sequence * stride_cache_batch + first + block_offset
    ).to(tl.int64)
    active = (
        active_block
        & (block_offset < max_blocks)
        & (cache_index != pad_slot_id)
        & (cache_index != null_block_id)
        & channel_mask[:, None]
        & state_mask[None, :]
        & (input_token[None, :] >= sequence_start)
        & (input_token[None, :] < sequence_end)
    )
    source = input_token[None, :] * stride_x_token + channel[:, None] * stride_x_dim
    destination = (
        cache_index * stride_state_seq
        + channel[:, None] * stride_state_dim
        + state_token[None, :] * stride_state_token
    )
    value = tl.load(x_ptr + source, mask=active, other=0.0)
    tl.store(states_ptr + destination, value, mask=active)


def apc_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | bool | None,
    pad_slot_id: int,
    null_block_id: int,
    block_idx_first_scheduled_token: torch.Tensor,
    block_idx_last_scheduled_token: torch.Tensor,
    initial_state_idx: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    block_size_to_align: int,
) -> torch.Tensor:
    """Self-owned prefix-cache/APC causal-conv implementation."""
    assert cache_indices.ndim == 2
    assert block_size_to_align > 0
    batch = query_start_loc.numel() - 1
    input_indices = torch.empty(batch, dtype=torch.int32, device=x.device)
    output_indices = torch.empty_like(input_indices)
    block = 128
    _resolve_apc_state_indices_kernel[(triton.cdiv(batch, block),)](
        cache_indices,
        initial_state_idx,
        block_idx_last_scheduled_token,
        input_indices,
        output_indices,
        batch,
        cache_indices.stride(0),
        BLOCK=block,
        num_warps=4,
    )
    output = generic_causal_conv1d(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        input_indices,
        has_initial_state,
        activation,
        pad_slot_id,
        null_block_id,
        output_cache_indices=output_indices,
    )

    state_len = weight.shape[1] - 1
    max_blocks = cache_indices.shape[1]
    block_d = 32
    _store_apc_intermediate_states_kernel[
        (batch, max_blocks, triton.cdiv(x.shape[0], block_d))
    ](
        x,
        conv_states,
        query_start_loc,
        cache_indices,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        num_computed_tokens,
        batch,
        x.shape[0],
        max_blocks,
        block_size_to_align,
        x.stride(0),
        x.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        pad_slot_id,
        null_block_id,
        STATE_LEN=state_len,
        BLOCK_D=block_d,
        BLOCK_STATE=triton.next_power_of_2(state_len),
        num_warps=4,
    )
    return output
