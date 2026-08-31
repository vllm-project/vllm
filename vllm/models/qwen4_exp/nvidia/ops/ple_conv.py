# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed varlen dilated short convolution for Qwen4Exp PLE."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit
def _varlen_dilated_conv1d_kernel(
    x_ptr,
    weight_ptr,
    conv_state_ptr,
    query_start_loc_ptr,
    sequence_indices_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    num_tokens,
    stride_x_token,
    stride_x_channel,
    stride_weight_channel,
    stride_weight_width,
    stride_state_sequence,
    stride_state_channel,
    stride_state_token,
    stride_output_token,
    stride_output_channel,
    CHANNELS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    DILATION: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_STATE_CACHE: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    channel_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    token_mask = token_offsets < num_tokens
    channel_mask = channel_offsets < CHANNELS

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    sequence_indices = tl.load(
        sequence_indices_ptr + token_offsets, mask=token_mask, other=0
    ).to(tl.int64)
    query_starts = tl.load(
        query_start_loc_ptr + sequence_indices, mask=token_mask, other=0
    ).to(tl.int64)
    local_positions = token_offsets - query_starts
    state_indices = tl.load(
        state_indices_ptr + sequence_indices,
        mask=token_mask,
        other=NULL_STATE_INDEX,
    ).to(tl.int64)
    valid_state = token_mask & (state_indices != NULL_STATE_INDEX)
    safe_state_indices = tl.maximum(state_indices, 0)

    if HAS_STATE_CACHE:
        has_initial_state = tl.load(
            has_initial_state_ptr + sequence_indices,
            mask=valid_state,
            other=False,
        ).to(tl.int1)

    acc = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for tap in tl.static_range(KERNEL_WIDTH):
        source_positions = local_positions - (KERNEL_WIDTH - 1 - tap) * DILATION
        from_input = valid_state & (source_positions >= 0)
        source_rows = tl.maximum(token_offsets - (KERNEL_WIDTH - 1 - tap) * DILATION, 0)
        x = tl.load(
            x_ptr
            + source_rows[:, None] * stride_x_token
            + channel_offsets[None, :] * stride_x_channel,
            mask=from_input[:, None] & channel_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        if HAS_STATE_CACHE:
            from_state = valid_state & has_initial_state & (source_positions < 0)
            state_positions = tl.maximum(STATE_LEN + source_positions, 0)
            state = tl.load(
                conv_state_ptr
                + safe_state_indices[:, None] * stride_state_sequence
                + channel_offsets[None, :] * stride_state_channel
                + state_positions[:, None] * stride_state_token,
                mask=from_state[:, None] & channel_mask[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            state = state.to(tl.float32)
        else:
            state = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

        weight = tl.load(
            weight_ptr
            + channel_offsets * stride_weight_channel
            + tap * stride_weight_width,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        acc += (x + state) * weight[None, :]

    acc *= tl.sigmoid(acc)
    acc = tl.where(valid_state[:, None], acc, 0.0)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output_ptr
        + token_offsets[:, None] * stride_output_token
        + channel_offsets[None, :] * stride_output_channel,
        acc,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _update_varlen_conv_state_kernel(
    x_ptr,
    conv_state_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    stride_x_token,
    stride_x_channel,
    stride_state_sequence,
    stride_state_channel,
    stride_state_token,
    CHANNELS: tl.constexpr,
    STATE_LEN: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    sequence_index = tl.program_id(0)
    channel_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    state_offsets = tl.arange(0, BLOCK_STATE)
    channel_mask = channel_offsets < CHANNELS
    state_mask = state_offsets < STATE_LEN

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    query_start = tl.load(query_start_loc_ptr + sequence_index).to(tl.int64)
    query_end = tl.load(query_start_loc_ptr + sequence_index + 1).to(tl.int64)
    query_len = query_end - query_start
    state_index = tl.load(state_indices_ptr + sequence_index).to(tl.int64)
    valid_sequence = (state_index != NULL_STATE_INDEX) & (query_len > 0)
    safe_state_index = tl.maximum(state_index, 0)
    has_initial_state = tl.load(
        has_initial_state_ptr + sequence_index,
        mask=valid_sequence,
        other=False,
    ).to(tl.int1)

    source_positions = query_len - STATE_LEN + state_offsets
    from_input = valid_sequence & state_mask & (source_positions >= 0)
    source_rows = tl.maximum(query_start + source_positions, 0)
    x = tl.load(
        x_ptr
        + source_rows[:, None] * stride_x_token
        + channel_offsets[None, :] * stride_x_channel,
        mask=from_input[:, None] & channel_mask[None, :],
        other=0.0,
    )

    from_state = (
        valid_sequence & state_mask & has_initial_state & (source_positions < 0)
    )
    prior_state_offsets = tl.maximum(STATE_LEN + source_positions, 0)
    prior_state = tl.load(
        conv_state_ptr
        + safe_state_index * stride_state_sequence
        + channel_offsets[None, :] * stride_state_channel
        + prior_state_offsets[:, None] * stride_state_token,
        mask=from_state[:, None] & channel_mask[None, :],
        other=0.0,
    ).to(x_ptr.dtype.element_ty)
    next_state = tl.where(from_input[:, None], x, prior_state)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        conv_state_ptr
        + safe_state_index * stride_state_sequence
        + channel_offsets[None, :] * stride_state_channel
        + state_offsets[:, None] * stride_state_token,
        next_state,
        mask=valid_sequence & state_mask[:, None] & channel_mask[None, :],
    )


def varlen_dilated_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Apply packed varlen dilated depthwise convolution and update its state."""
    num_tokens, channels = x.shape
    kernel_width = weight.shape[1]
    state_len = (kernel_width - 1) * dilation
    num_sequences = query_start_loc.numel() - 1

    assert x.ndim == 2 and x.stride(1) == 1
    assert weight.shape[0] == channels and weight.stride(1) == 1
    assert dilation > 0
    assert state_indices.numel() >= num_sequences
    assert has_initial_state.numel() >= num_sequences
    assert query_start_loc.device == x.device
    assert state_indices.device == x.device
    assert has_initial_state.device == x.device

    has_state_cache = conv_state.ndim == 3 and conv_state.shape[0] > 0
    if has_state_cache:
        assert conv_state.shape[1] == channels
        assert conv_state.shape[2] >= state_len
        state_strides = conv_state.stride()
    else:
        assert conv_state.numel() == 0
        state_strides = (0, 0, 0)

    output = torch.empty_like(x)
    if num_tokens == 0:
        return output

    positions = torch.arange(num_tokens, dtype=query_start_loc.dtype, device=x.device)
    sequence_indices = torch.searchsorted(query_start_loc[1:], positions, right=True)

    block_t = 8
    block_c = min(triton.next_power_of_2(channels), 256)
    grid = (triton.cdiv(num_tokens, block_t), triton.cdiv(channels, block_c))
    launch_pdl = current_platform.is_arch_support_pdl()
    _varlen_dilated_conv1d_kernel[grid](
        x,
        weight,
        conv_state,
        query_start_loc,
        sequence_indices,
        state_indices,
        has_initial_state,
        output,
        num_tokens,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        state_strides[0],
        state_strides[1],
        state_strides[2],
        output.stride(0),
        output.stride(1),
        CHANNELS=channels,
        KERNEL_WIDTH=kernel_width,
        DILATION=dilation,
        STATE_LEN=state_len,
        HAS_STATE_CACHE=has_state_cache,
        NULL_STATE_INDEX=NULL_BLOCK_ID,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        launch_pdl=launch_pdl,
        num_warps=4,
    )

    if has_state_cache and state_len > 0:
        block_state = triton.next_power_of_2(state_len)
        state_grid = (num_sequences, triton.cdiv(channels, block_c))
        _update_varlen_conv_state_kernel[state_grid](
            x,
            conv_state,
            query_start_loc,
            state_indices,
            has_initial_state,
            x.stride(0),
            x.stride(1),
            state_strides[0],
            state_strides[1],
            state_strides[2],
            CHANNELS=channels,
            STATE_LEN=state_len,
            NULL_STATE_INDEX=NULL_BLOCK_ID,
            BLOCK_STATE=block_state,
            BLOCK_C=block_c,
            launch_pdl=launch_pdl,
            num_warps=4,
        )
    return output
