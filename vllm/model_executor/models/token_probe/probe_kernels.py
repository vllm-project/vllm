# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton

ACT_GELU = 1
ACT_RELU = 2


@triton.jit
def _tap_kernel(
    hidden_ptr,
    residual_ptr,
    output_ptr,
    hidden_stride,
    residual_stride,
    output_stride,
    column_offset,
    width,
    eps,
    has_residual: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < width
    values = tl.load(
        hidden_ptr + row * hidden_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if has_residual:
        values += tl.load(
            residual_ptr + row * residual_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    values *= tl.rsqrt(tl.sum(values * values, 0) / width + eps)
    tl.store(
        output_ptr + row * output_stride + column_offset + offsets,
        values.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


def tap_into(
    output: torch.Tensor,
    slot: int,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    eps: float = 1e-6,
) -> None:
    rows, width = hidden_states.shape
    if not hidden_states.is_cuda:
        values = hidden_states.float()
        if residual is not None:
            values = values + residual.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        output[:, slot * width : (slot + 1) * width] = values.to(output.dtype)
        return
    _tap_kernel[(rows,)](
        hidden_states,
        residual if residual is not None else hidden_states,
        output,
        hidden_states.stride(0),
        residual.stride(0) if residual is not None else 0,
        output.stride(0),
        slot * width,
        width,
        eps,
        has_residual=residual is not None,
        block=triton.next_power_of_2(width),
        num_warps=8,
    )


@triton.jit
def _classify_tail_kernel(
    hidden_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    hidden_stride,
    output_stride,
    hidden_width: tl.constexpr,
    hidden_block: tl.constexpr,
    output_width: tl.constexpr,
    output_block: tl.constexpr,
    activation: tl.constexpr,
    has_bias: tl.constexpr,
):
    row = tl.program_id(0)
    hidden_offsets = tl.arange(0, hidden_block)
    output_offsets = tl.arange(0, output_block)
    hidden_mask = hidden_offsets < hidden_width
    output_mask = output_offsets < output_width
    hidden = tl.load(
        hidden_ptr + row * hidden_stride + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    if activation == 1:
        hidden *= 0.5 * (1.0 + tl.erf(hidden * 0.7071067811865476))
    else:
        hidden = tl.maximum(hidden, 0.0)
    weight = tl.load(
        weight_ptr + output_offsets[:, None] * hidden_width + hidden_offsets[None, :],
        mask=output_mask[:, None] & hidden_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(hidden[None, :] * weight, 1)
    if has_bias:
        logits += tl.load(bias_ptr + output_offsets, mask=output_mask, other=0.0).to(
            tl.float32
        )
    tl.store(
        output_ptr + row * output_stride + output_offsets,
        tl.sigmoid(logits),
        mask=output_mask,
    )


def classify_tail(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: int,
) -> torch.Tensor:
    rows, hidden_width = hidden.shape
    output_width = weight.shape[0]
    if not hidden.is_cuda:
        hidden = F.gelu(hidden) if activation == ACT_GELU else F.relu(hidden)
        return torch.sigmoid(
            F.linear(
                hidden.float(),
                weight.float(),
                bias.float() if bias is not None else None,
            )
        )
    output = torch.empty(rows, output_width, device=hidden.device, dtype=torch.float32)
    _classify_tail_kernel[(rows,)](
        hidden,
        weight,
        bias,
        output,
        hidden.stride(0),
        output.stride(0),
        hidden_width=hidden_width,
        hidden_block=triton.next_power_of_2(hidden_width),
        output_width=output_width,
        output_block=triton.next_power_of_2(output_width),
        activation=activation,
        has_bias=bias is not None,
        num_warps=4,
    )
    return output


@triton.jit
def _attention_tail_kernel(
    projected_ptr,
    query_ptr,
    norm_weight_ptr,
    classifier_weight_ptr,
    bias_ptr,
    output_ptr,
    projected_stride,
    query_stride,
    output_stride,
    eps: tl.constexpr,
    hidden_width: tl.constexpr,
    hidden_block: tl.constexpr,
    output_width: tl.constexpr,
    output_block: tl.constexpr,
    has_bias: tl.constexpr,
):
    row = tl.program_id(0)
    hidden_offsets = tl.arange(0, hidden_block)
    output_offsets = tl.arange(0, output_block)
    hidden_mask = hidden_offsets < hidden_width
    output_mask = output_offsets < output_width
    hidden = tl.load(
        projected_ptr + row * projected_stride + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    hidden += tl.load(
        query_ptr + row * query_stride + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    hidden *= tl.rsqrt(tl.sum(hidden * hidden, 0) / hidden_width + eps)
    norm_weight = tl.load(
        norm_weight_ptr + hidden_offsets, mask=hidden_mask, other=0.0
    ).to(tl.float32)
    hidden *= norm_weight
    classifier_weight = tl.load(
        classifier_weight_ptr
        + output_offsets[:, None] * hidden_width
        + hidden_offsets[None, :],
        mask=output_mask[:, None] & hidden_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(hidden[None, :] * classifier_weight, 1)
    if has_bias:
        logits += tl.load(bias_ptr + output_offsets, mask=output_mask, other=0.0).to(
            tl.float32
        )
    tl.store(
        output_ptr + row * output_stride + output_offsets,
        tl.sigmoid(logits),
        mask=output_mask,
    )


def attention_tail(
    projected: torch.Tensor,
    query: torch.Tensor,
    norm_weight: torch.Tensor,
    classifier_weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    rows, hidden_width = projected.shape
    output_width = classifier_weight.shape[0]
    if not projected.is_cuda:
        hidden = projected.float() + query.float()
        hidden *= torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps)
        hidden *= norm_weight.float()
        return torch.sigmoid(
            F.linear(
                hidden,
                classifier_weight.float(),
                bias.float() if bias is not None else None,
            )
        )
    output = torch.empty(
        rows, output_width, device=projected.device, dtype=torch.float32
    )
    _attention_tail_kernel[(rows,)](
        projected,
        query,
        norm_weight,
        classifier_weight,
        bias,
        output,
        projected.stride(0),
        query.stride(0),
        output.stride(0),
        eps=eps,
        hidden_width=hidden_width,
        hidden_block=triton.next_power_of_2(hidden_width),
        output_width=output_width,
        output_block=triton.next_power_of_2(output_width),
        has_bias=bias is not None,
        num_warps=4,
    )
    return output
