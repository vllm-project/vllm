# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _pack_kda_mixed_inputs_kernel(
    mixed_qkv,
    g1,
    beta,
    non_spec_indices,
    spec_indices,
    packed_mixed_qkv,
    packed_g1,
    packed_beta,
    num_non_spec,
    mixed_qkv_stride_token,
    mixed_qkv_stride_dim,
    g1_stride_token,
    g1_stride_head,
    g1_stride_dim,
    beta_stride_token,
    beta_stride_head,
    QKV_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    packed_token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    is_non_spec = packed_token < num_non_spec
    non_spec_token = tl.load(
        non_spec_indices + packed_token,
        mask=is_non_spec,
        other=0,
    )
    spec_token = tl.load(
        spec_indices + packed_token - num_non_spec,
        mask=~is_non_spec,
        other=0,
    )
    source_token = tl.where(is_non_spec, non_spec_token, spec_token)

    qkv_mask = offsets < QKV_WIDTH
    qkv = tl.load(
        mixed_qkv
        + source_token * mixed_qkv_stride_token
        + offsets * mixed_qkv_stride_dim,
        mask=qkv_mask,
    )
    tl.store(
        packed_mixed_qkv + packed_token * QKV_WIDTH + offsets,
        qkv,
        mask=qkv_mask,
    )

    g1_width: tl.constexpr = NUM_HEADS * HEAD_DIM
    g1_mask = offsets < g1_width
    g1_head = offsets // HEAD_DIM
    g1_dim = offsets % HEAD_DIM
    g1_values = tl.load(
        g1
        + source_token * g1_stride_token
        + g1_head * g1_stride_head
        + g1_dim * g1_stride_dim,
        mask=g1_mask,
    )
    tl.store(
        packed_g1 + packed_token * g1_width + offsets,
        g1_values,
        mask=g1_mask,
    )

    beta_mask = offsets < NUM_HEADS
    beta_values = tl.load(
        beta + source_token * beta_stride_token + offsets * beta_stride_head,
        mask=beta_mask,
    )
    tl.store(
        packed_beta + packed_token * NUM_HEADS + offsets,
        beta_values,
        mask=beta_mask,
    )


def _pack_kda_mixed_inputs_impl(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_non_spec = non_spec_indices.numel()
    num_tokens = num_non_spec + spec_indices.numel()
    num_heads, head_dim = g1.shape[-2:]
    qkv_width = mixed_qkv.size(1)
    packed_mixed_qkv = torch.empty(
        (num_tokens, qkv_width),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    packed_g1 = torch.empty(
        (1, num_tokens, num_heads, head_dim),
        dtype=g1.dtype,
        device=g1.device,
    )
    packed_beta = torch.empty(
        (1, num_tokens, num_heads),
        dtype=beta.dtype,
        device=beta.device,
    )
    if num_tokens == 0:
        return packed_mixed_qkv, packed_g1, packed_beta

    max_width = max(qkv_width, num_heads * head_dim)
    block_size = 256
    _pack_kda_mixed_inputs_kernel[(num_tokens, cdiv(max_width, block_size))](
        mixed_qkv,
        g1,
        beta,
        non_spec_indices,
        spec_indices,
        packed_mixed_qkv,
        packed_g1,
        packed_beta,
        num_non_spec,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        g1.stride(1),
        g1.stride(2),
        g1.stride(3),
        beta.stride(1),
        beta.stride(2),
        QKV_WIDTH=qkv_width,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return packed_mixed_qkv, packed_g1, packed_beta


def _pack_kda_mixed_inputs_fake(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = non_spec_indices.numel() + spec_indices.numel()
    return (
        torch.empty(
            (num_tokens, mixed_qkv.size(1)),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        ),
        torch.empty(
            (1, num_tokens, g1.size(2), g1.size(3)),
            dtype=g1.dtype,
            device=g1.device,
        ),
        torch.empty(
            (1, num_tokens, beta.size(2)),
            dtype=beta.dtype,
            device=beta.device,
        ),
    )


direct_register_custom_op(
    op_name="kimi_k3_pack_kda_mixed_inputs",
    op_func=_pack_kda_mixed_inputs_impl,
    fake_impl=_pack_kda_mixed_inputs_fake,
)


@triton.jit
def _scatter_rms_norm_kda_mixed_outputs_kernel(
    non_spec_output,
    spec_output,
    non_spec_indices,
    spec_indices,
    gate,
    weight,
    output,
    num_non_spec,
    eps,
    non_spec_stride_token,
    non_spec_stride_head,
    non_spec_stride_dim,
    spec_stride_token,
    spec_stride_head,
    spec_stride_dim,
    gate_stride_token,
    gate_stride_head,
    gate_stride_dim,
    output_stride_token,
    output_stride_head,
    output_stride_dim,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    packed_token = row // NUM_HEADS
    head = row % NUM_HEADS
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HEAD_DIM

    is_non_spec = packed_token < num_non_spec
    non_spec_token = packed_token
    spec_token = packed_token - num_non_spec
    destination_token_non_spec = tl.load(
        non_spec_indices + non_spec_token,
        mask=is_non_spec,
        other=0,
    )
    destination_token_spec = tl.load(
        spec_indices + spec_token,
        mask=~is_non_spec,
        other=0,
    )
    destination_token = tl.where(
        is_non_spec,
        destination_token_non_spec,
        destination_token_spec,
    )

    non_spec_values = tl.load(
        non_spec_output
        + non_spec_token * non_spec_stride_token
        + head * non_spec_stride_head
        + offsets * non_spec_stride_dim,
        mask=is_non_spec & mask,
        other=0.0,
    )
    spec_values = tl.load(
        spec_output
        + spec_token * spec_stride_token
        + head * spec_stride_head
        + offsets * spec_stride_dim,
        mask=(~is_non_spec) & mask,
        other=0.0,
    )
    values = tl.where(is_non_spec, non_spec_values, spec_values).to(tl.float32)
    squared_values = tl.where(mask, values * values, 0.0)
    reciprocal_std = 1.0 / tl.sqrt(tl.sum(squared_values, axis=0) / HEAD_DIM + eps)

    norm_weight = tl.load(weight + offsets, mask=mask).to(tl.float32)
    gate_values = tl.load(
        gate
        + destination_token * gate_stride_token
        + head * gate_stride_head
        + offsets * gate_stride_dim,
        mask=mask,
    ).to(tl.float32)
    normalized = values * reciprocal_std * norm_weight * tl.sigmoid(gate_values)
    tl.store(
        output
        + destination_token * output_stride_token
        + head * output_stride_head
        + offsets * output_stride_dim,
        normalized,
        mask=mask,
    )


def _scatter_rms_norm_kda_mixed_outputs_impl(
    non_spec_output: torch.Tensor,
    spec_output: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> None:
    num_non_spec = non_spec_indices.numel()
    num_tokens = num_non_spec + spec_indices.numel()
    if num_tokens == 0:
        return

    num_heads, head_dim = output.shape[-2:]
    block_size = next_power_of_2(head_dim)
    _scatter_rms_norm_kda_mixed_outputs_kernel[(num_tokens * num_heads,)](
        non_spec_output,
        spec_output,
        non_spec_indices,
        spec_indices,
        gate,
        weight,
        output,
        num_non_spec,
        eps,
        non_spec_output.stride(1),
        non_spec_output.stride(2),
        non_spec_output.stride(3),
        spec_output.stride(1),
        spec_output.stride(2),
        spec_output.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


def _scatter_rms_norm_kda_mixed_outputs_fake(
    non_spec_output: torch.Tensor,
    spec_output: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> None:
    return


direct_register_custom_op(
    op_name="kimi_k3_scatter_rms_norm_kda_mixed_outputs",
    op_func=_scatter_rms_norm_kda_mixed_outputs_impl,
    mutates_args=["output"],
    fake_impl=_scatter_rms_norm_kda_mixed_outputs_fake,
)


def _pack_kda_mixed_inputs_native(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.cat((non_spec_indices, spec_indices))
    return (
        mixed_qkv.index_select(0, indices),
        g1.index_select(1, indices),
        beta.index_select(1, indices),
    )


def _can_pack_with_triton(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
) -> bool:
    tensors = (mixed_qkv, g1, beta, non_spec_indices, spec_indices)
    return (
        mixed_qkv.device.type == "cuda"
        and all(tensor.device == mixed_qkv.device for tensor in tensors)
        and mixed_qkv.dtype in (torch.float16, torch.bfloat16)
        and g1.dtype == mixed_qkv.dtype
        and beta.dtype == mixed_qkv.dtype
        and non_spec_indices.dtype in (torch.int32, torch.int64)
        and spec_indices.dtype == non_spec_indices.dtype
        and g1.size(3) == 128
        and mixed_qkv.size(1) == 3 * g1.size(2) * g1.size(3)
        and mixed_qkv.stride(1) == 1
        and g1.stride(3) == 1
        and beta.stride(2) == 1
        and non_spec_indices.stride(0) == 1
        and spec_indices.stride(0) == 1
    )


def pack_kda_mixed_inputs(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    beta: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack stable non-spec/spec partitions with one device launch.

    The returned tensors contain the non-spec partition first. Each partition
    preserves the scheduler's original token order. The index tensors must be
    a disjoint partition of all input rows.
    """
    num_tokens = non_spec_indices.numel() + spec_indices.numel()
    if (
        mixed_qkv.ndim != 2
        or g1.ndim != 4
        or beta.ndim != 3
        or non_spec_indices.ndim != 1
        or spec_indices.ndim != 1
        or g1.size(0) != 1
        or beta.size(0) != 1
        or mixed_qkv.size(0) != num_tokens
        or g1.size(1) != num_tokens
        or beta.size(1) != num_tokens
        or beta.size(2) != g1.size(2)
    ):
        raise ValueError("Invalid KDA mixed-input shapes")
    if _can_pack_with_triton(
        mixed_qkv,
        g1,
        beta,
        non_spec_indices,
        spec_indices,
    ):
        return torch.ops.vllm.kimi_k3_pack_kda_mixed_inputs(
            mixed_qkv,
            g1,
            beta,
            non_spec_indices,
            spec_indices,
        )
    return _pack_kda_mixed_inputs_native(
        mixed_qkv,
        g1,
        beta,
        non_spec_indices,
        spec_indices,
    )


def _scatter_rms_norm_kda_mixed_outputs_native(
    non_spec_output: torch.Tensor,
    spec_output: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> None:
    indices = torch.cat((non_spec_indices, spec_indices))
    output.index_copy_(1, non_spec_indices, non_spec_output)
    output.index_copy_(1, spec_indices, spec_output)
    selected = output.index_select(1, indices)
    selected_gate = gate.index_select(0, indices)
    selected_float = selected.float()
    variance = selected_float.square().mean(dim=-1, keepdim=True)
    normalized = selected_float * torch.rsqrt(variance + eps)
    normalized *= weight.float()
    normalized *= torch.sigmoid(selected_gate.float())
    output.index_copy_(1, indices, normalized.to(output.dtype))


def _can_scatter_rms_norm_with_triton(
    non_spec_output: torch.Tensor,
    spec_output: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    tensors = (
        non_spec_output,
        spec_output,
        non_spec_indices,
        spec_indices,
        gate,
        weight,
        output,
    )
    return (
        output.device.type == "cuda"
        and all(tensor.device == output.device for tensor in tensors)
        and output.dtype in (torch.float16, torch.bfloat16)
        and non_spec_output.dtype == output.dtype
        and spec_output.dtype == output.dtype
        and gate.dtype == output.dtype
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and non_spec_indices.dtype in (torch.int32, torch.int64)
        and spec_indices.dtype == non_spec_indices.dtype
        and non_spec_output.stride(3) == 1
        and spec_output.stride(3) == 1
        and gate.stride(2) == 1
        and weight.stride(0) == 1
        and output.stride(3) == 1
        and non_spec_indices.stride(0) == 1
        and spec_indices.stride(0) == 1
        and output.size(3) == 128
    )


def scatter_rms_norm_kda_mixed_outputs(
    non_spec_output: torch.Tensor,
    spec_output: torch.Tensor,
    non_spec_indices: torch.Tensor,
    spec_indices: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
) -> None:
    """Scatter and normalize indexed mixed-KDA rows in ``output``.

    Rows absent from both index tensors, including CUDA-graph padding, are
    intentionally left untouched. The index tensors must be disjoint and
    in-bounds.
    """
    num_non_spec = non_spec_indices.numel()
    num_spec = spec_indices.numel()
    if (
        non_spec_output.ndim != 4
        or spec_output.ndim != 4
        or non_spec_indices.ndim != 1
        or spec_indices.ndim != 1
        or gate.ndim != 3
        or output.ndim != 4
        or non_spec_output.size(0) != 1
        or spec_output.size(0) != 1
        or output.size(0) != 1
        or non_spec_output.size(1) != num_non_spec
        or spec_output.size(1) != num_spec
        or non_spec_output.shape[2:] != output.shape[2:]
        or spec_output.shape[2:] != output.shape[2:]
        or gate.shape != output.shape[1:]
        or weight.shape != (output.size(3),)
    ):
        raise ValueError("Invalid KDA mixed-output shapes")
    if _can_scatter_rms_norm_with_triton(
        non_spec_output,
        spec_output,
        non_spec_indices,
        spec_indices,
        gate,
        weight,
        output,
    ):
        torch.ops.vllm.kimi_k3_scatter_rms_norm_kda_mixed_outputs(
            non_spec_output,
            spec_output,
            non_spec_indices,
            spec_indices,
            gate,
            weight,
            output,
            eps,
        )
        return
    _scatter_rms_norm_kda_mixed_outputs_native(
        non_spec_output,
        spec_output,
        non_spec_indices,
        spec_indices,
        gate,
        weight,
        output,
        eps,
    )
