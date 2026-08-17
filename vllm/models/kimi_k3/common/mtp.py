# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Kimi-K3 MTP input preparation."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm(x, weight, eps, hidden_size: tl.constexpr):
    x = x.to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    return x * tl.rsqrt(variance + eps) * weight.to(tl.float32)


@triton.jit
def _fused_mtp_input_kernel(
    positions_ptr,
    inputs_embeds_ptr,
    previous_hidden_states_ptr,
    enorm_weight_ptr,
    hnorm_weight_ptr,
    output_ptr,
    eps,
    inputs_embeds_stride,
    previous_hidden_states_stride,
    output_stride,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    input_idx = tl.program_id(1)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size

    if input_idx == 0:
        position = tl.load(positions_ptr + token_idx)
        values = tl.load(
            inputs_embeds_ptr + token_idx * inputs_embeds_stride + offsets,
            mask=mask & (position != 0),
            other=0.0,
        )
        weight = tl.load(enorm_weight_ptr + offsets, mask=mask, other=0.0)
    else:
        values = tl.load(
            previous_hidden_states_ptr
            + token_idx * previous_hidden_states_stride
            + offsets,
            mask=mask,
            other=0.0,
        )
        weight = tl.load(hnorm_weight_ptr + offsets, mask=mask, other=0.0)

    output = _rms_norm(values, weight, eps, hidden_size)
    tl.store(
        output_ptr + token_idx * output_stride + input_idx * hidden_size + offsets,
        output,
        mask=mask,
    )


def fused_mtp_input(
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Mask and normalize both MTP inputs into the projection layout."""
    num_tokens, hidden_size = inputs_embeds.shape
    output = torch.empty(
        num_tokens,
        2 * hidden_size,
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    if num_tokens == 0:
        return output

    _fused_mtp_input_kernel[(num_tokens, 2)](
        positions,
        inputs_embeds,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        output,
        eps,
        inputs_embeds.stride(0),
        previous_hidden_states.stride(0),
        output.stride(0),
        hidden_size,
        triton.next_power_of_2(hidden_size),
    )
    return output
