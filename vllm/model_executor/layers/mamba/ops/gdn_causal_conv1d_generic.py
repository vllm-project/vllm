# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _causal_conv1d_varlen_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    states_ptr,
    query_start_ptr,
    state_indices_ptr,
    has_initial_ptr,
    output_ptr,
    total_tokens,
    dim,
    stride_x_dim,
    stride_x_token,
    stride_weight_dim,
    stride_weight_width,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    stride_output_dim,
    stride_output_token,
    pad_slot_id,
    null_block_id,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    token_mask = token < total_tokens
    channel_mask = channel < dim

    sequence = tl.zeros((BLOCK_T,), dtype=tl.int32)
    for seq in tl.static_range(1, BATCH):
        start = tl.load(query_start_ptr + seq)
        sequence += (token >= start).to(tl.int32)

    sequence_start = tl.load(query_start_ptr + sequence, mask=token_mask, other=0)
    sequence_end = tl.load(query_start_ptr + sequence + 1, mask=token_mask, other=0)
    local_token = token - sequence_start
    sequence_length = sequence_end - sequence_start
    state_index = tl.load(
        state_indices_ptr + sequence, mask=token_mask, other=null_block_id
    ).to(tl.int64)
    has_initial = tl.load(has_initial_ptr + sequence, mask=token_mask, other=0).to(
        tl.int1
    )
    active = (
        token_mask
        & (local_token >= 0)
        & (local_token < sequence_length)
        & (state_index != pad_slot_id)
        & (state_index != null_block_id)
    )

    accumulator = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    state_len: tl.constexpr = WIDTH - 1
    for tap in tl.static_range(WIDTH):
        source_local = local_token - state_len + tap
        from_input = source_local >= 0
        input_token = sequence_start + source_local
        input_offsets = (
            input_token[:, None] * stride_x_token + channel[None, :] * stride_x_dim
        )
        input_value = tl.load(
            x_ptr + input_offsets,
            mask=active[:, None] & from_input[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        state_token = source_local + state_len
        state_offsets = (
            state_index[:, None] * stride_state_seq
            + channel[None, :] * stride_state_dim
            + state_token[:, None] * stride_state_token
        )
        state_value = tl.load(
            states_ptr + state_offsets,
            mask=(
                active[:, None]
                & (~from_input[:, None])
                & has_initial[:, None]
                & channel_mask[None, :]
            ),
            other=0.0,
        ).to(tl.float32)
        value = tl.where(from_input[:, None], input_value, state_value)
        weight = tl.load(
            weight_ptr + channel * stride_weight_dim + tap * stride_weight_width,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += value * weight[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel, mask=channel_mask, other=0.0)
        accumulator += bias[None, :].to(tl.float32)
    if ACTIVATION:
        accumulator *= tl.sigmoid(accumulator)

    output_offsets = (
        token[:, None] * stride_output_token + channel[None, :] * stride_output_dim
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=active[:, None] & channel_mask[None, :],
    )


@triton.jit
def _update_conv_state_kernel(
    x_ptr,
    states_ptr,
    new_states_ptr,
    query_start_ptr,
    state_indices_ptr,
    has_initial_ptr,
    dim,
    stride_x_dim,
    stride_x_token,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    stride_new_batch,
    stride_new_dim,
    stride_new_token,
    pad_slot_id,
    null_block_id,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    sequence = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    state_token = tl.arange(0, BLOCK_STATE)
    channel_mask = channel < dim
    state_len: tl.constexpr = WIDTH - 1
    state_mask = state_token < state_len

    sequence_start = tl.load(query_start_ptr + sequence)
    sequence_end = tl.load(query_start_ptr + sequence + 1)
    sequence_length = sequence_end - sequence_start
    state_index = tl.load(state_indices_ptr + sequence).to(tl.int64)
    has_initial = tl.load(has_initial_ptr + sequence).to(tl.int1)
    active = (
        (state_index != pad_slot_id)
        & (state_index != null_block_id)
        & channel_mask[:, None]
        & state_mask[None, :]
    )

    shift = state_len - tl.minimum(sequence_length, state_len)
    from_old_state = state_token < shift
    old_state_token = state_token + sequence_length
    input_token = sequence_end - (state_len - state_token)

    old_offsets = (
        state_index * stride_state_seq
        + channel[:, None] * stride_state_dim
        + old_state_token[None, :] * stride_state_token
    )
    old_value = tl.load(
        states_ptr + old_offsets,
        mask=active & from_old_state[None, :] & has_initial,
        other=0.0,
    )
    input_offsets = (
        input_token[None, :] * stride_x_token + channel[:, None] * stride_x_dim
    )
    input_value = tl.load(
        x_ptr + input_offsets,
        mask=active & (~from_old_state[None, :]),
        other=0.0,
    )
    value = tl.where(from_old_state[None, :], old_value, input_value)
    destination = (
        sequence * stride_new_batch
        + channel[:, None] * stride_new_dim
        + state_token[None, :] * stride_new_token
    )
    tl.store(new_states_ptr + destination, value, mask=active)


@triton.jit
def _scatter_conv_state_kernel(
    new_states_ptr,
    states_ptr,
    state_indices_ptr,
    total_elements,
    elements_per_state,
    stride_new_batch,
    stride_new_dim,
    stride_new_token,
    stride_state_seq,
    stride_state_dim,
    stride_state_token,
    state_len,
    pad_slot_id,
    null_block_id,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < total_elements
    sequence = offset // elements_per_state
    inner = offset % elements_per_state
    channel = inner // state_len
    state_token = inner % state_len
    state_index = tl.load(
        state_indices_ptr + sequence, mask=mask, other=null_block_id
    ).to(tl.int64)
    active = mask & (state_index != pad_slot_id) & (state_index != null_block_id)
    source = (
        sequence * stride_new_batch
        + channel * stride_new_dim
        + state_token * stride_new_token
    )
    destination = (
        state_index * stride_state_seq
        + channel * stride_state_dim
        + state_token * stride_state_token
    )
    value = tl.load(new_states_ptr + source, mask=active)
    tl.store(states_ptr + destination, value, mask=active)


def generic_causal_conv1d(
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
    output_cache_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Self-owned generic packed-varlen causal-conv implementation."""
    assert x.ndim == 2 and weight.ndim == 2 and conv_states.ndim == 3
    assert query_start_loc.ndim == 1
    batch = query_start_loc.numel() - 1
    assert cache_indices.numel() >= batch
    assert has_initial_state.numel() >= batch
    if output_cache_indices is None:
        output_cache_indices = cache_indices
    assert output_cache_indices.numel() >= batch
    dim, total_tokens = x.shape
    width = weight.shape[1]
    assert width in (2, 3, 4, 5)
    output = torch.empty_like(x)
    block_t = 16
    block_d = 32
    _causal_conv1d_varlen_kernel[
        (triton.cdiv(total_tokens, block_t), triton.cdiv(dim, block_d))
    ](
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        output,
        total_tokens,
        dim,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        output.stride(0),
        output.stride(1),
        pad_slot_id,
        null_block_id,
        BATCH=batch,
        WIDTH=width,
        HAS_BIAS=bias is not None,
        ACTIVATION=activation in ("silu", "swish", True),
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
    )
    state_len = width - 1
    new_states = torch.empty(
        (batch, dim, state_len), dtype=conv_states.dtype, device=conv_states.device
    )
    _update_conv_state_kernel[(batch, triton.cdiv(dim, block_d))](
        x,
        conv_states,
        new_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        dim,
        x.stride(0),
        x.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        new_states.stride(0),
        new_states.stride(1),
        new_states.stride(2),
        pad_slot_id,
        null_block_id,
        WIDTH=width,
        BLOCK_D=block_d,
        BLOCK_STATE=triton.next_power_of_2(state_len),
        num_warps=4,
    )
    elements_per_state = dim * state_len
    total_state_elements = batch * elements_per_state
    state_block = 256
    _scatter_conv_state_kernel[(triton.cdiv(total_state_elements, state_block),)](
        new_states,
        conv_states,
        output_cache_indices,
        total_state_elements,
        elements_per_state,
        new_states.stride(0),
        new_states.stride(1),
        new_states.stride(2),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        state_len,
        pad_slot_id,
        null_block_id,
        BLOCK=state_block,
        num_warps=4,
    )
    return output
