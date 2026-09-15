# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage LoRA matmul with explicit token-to-adapter routing."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_K = 64
_BLOCK_N = 64


@triton.jit
def _routed_matmul_kernel(
    x,
    weight,
    output,
    token_lora_mapping,
    num_heads: tl.constexpr,
    contraction_size: tl.constexpr,
    output_size: tl.constexpr,
    x_stride_m,
    x_stride_h,
    x_stride_k,
    weight_stride_l,
    weight_stride_h,
    weight_stride_k,
    weight_stride_n,
    output_stride_m,
    output_stride_h,
    output_stride_n,
    add_output: tl.constexpr,
    block_k: tl.constexpr,
    block_n: tl.constexpr,
):
    token_head = tl.program_id(0)
    token_id = token_head // num_heads
    head_id = token_head % num_heads
    output_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
    output_mask = output_offsets < output_size

    lora_id = tl.load(token_lora_mapping + token_id)
    if lora_id == -1:
        return

    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    contraction_offsets = tl.arange(0, block_k)
    for contraction_start in range(0, contraction_size, block_k):
        current_k = contraction_start + contraction_offsets
        contraction_mask = current_k < contraction_size
        x_values = tl.load(
            x + token_id * x_stride_m + head_id * x_stride_h + current_k * x_stride_k,
            mask=contraction_mask,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weight
            + lora_id * weight_stride_l
            + head_id * weight_stride_h
            + current_k[:, None] * weight_stride_k
            + output_offsets[None, :] * weight_stride_n,
            mask=contraction_mask[:, None] & output_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(x_values[:, None] * weight_values, axis=0)

    output_ptr = (
        output
        + token_id * output_stride_m
        + head_id * output_stride_h
        + output_offsets * output_stride_n
    )
    if add_output:
        accumulator += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, accumulator, mask=output_mask)


def _routed_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    add_output: bool,
) -> None:
    num_tokens, num_heads, contraction_size = x.shape
    output_size = output.shape[-1]
    grid = (num_tokens * num_heads, triton.cdiv(output_size, _BLOCK_N))
    _routed_matmul_kernel[grid](
        x,
        weight,
        output,
        token_lora_mapping,
        num_heads=num_heads,
        contraction_size=contraction_size,
        output_size=output_size,
        x_stride_m=x.stride(0),
        x_stride_h=x.stride(1),
        x_stride_k=x.stride(2),
        weight_stride_l=weight.stride(0),
        weight_stride_h=0 if weight.shape[1] == 1 else weight.stride(1),
        weight_stride_k=weight.stride(2),
        weight_stride_n=weight.stride(3),
        output_stride_m=output.stride(0),
        output_stride_h=output.stride(1),
        output_stride_n=output.stride(2),
        add_output=add_output,
        block_k=_BLOCK_K,
        block_n=_BLOCK_N,
    )


def _native_routed_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    add_output: bool,
) -> None:
    result = torch.zeros_like(output)
    for lora_id in range(weight.shape[0]):
        mask = token_lora_mapping[: x.shape[0]] == lora_id
        if torch.any(mask):
            adapter_weight = weight[lora_id].expand(x.shape[1], -1, -1)
            result[mask] = torch.einsum(
                "mhk,hkn->mhn", x[mask].float(), adapter_weight.float()
            ).to(output.dtype)
    if add_output:
        output.add_(result)
    else:
        output.copy_(result)


def _validate_shapes(
    x: torch.Tensor,
    first_weight: torch.Tensor,
    second_weight: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
) -> None:
    assert x.ndim == output.ndim == 3
    assert first_weight.ndim == second_weight.ndim == 4
    assert x.shape[:2] == output.shape[:2]
    assert first_weight.shape[0] == second_weight.shape[0]
    assert first_weight.shape[1] in (1, x.shape[1])
    assert second_weight.shape[1] in (1, x.shape[1])
    assert first_weight.shape[2] == x.shape[2]
    assert first_weight.shape[3] == second_weight.shape[2]
    assert second_weight.shape[3] == output.shape[2]
    assert token_lora_mapping.shape[0] >= x.shape[0]


@torch.inference_mode()
def _routed_lora_two_stage(
    x: torch.Tensor,
    first_weight: torch.Tensor,
    second_weight: torch.Tensor,
    output: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
) -> None:
    """Add a routed ``x @ first_weight @ second_weight`` to ``output``.

    Both weights use the logical layout ``(loras, heads, input, output)``.
    A singleton head dimension broadcasts the same matrix across all heads.
    """
    if no_lora_flag_cpu.item():
        return
    _validate_shapes(x, first_weight, second_weight, output, token_lora_mapping)
    intermediate = torch.empty(
        (*x.shape[:2], first_weight.shape[-1]),
        dtype=torch.float32,
        device=x.device,
    )
    matmul = _routed_matmul
    if not current_platform.is_cuda_alike():
        matmul = _native_routed_matmul
    matmul(
        x,
        first_weight,
        intermediate,
        token_lora_mapping,
        add_output=False,
    )
    matmul(
        intermediate,
        second_weight,
        output,
        token_lora_mapping,
        add_output=True,
    )


def _fake(*args, **kwargs) -> None:
    return None


try:
    direct_register_custom_op(
        op_name="routed_lora_two_stage",
        op_func=_routed_lora_two_stage,
        mutates_args=["output"],
        fake_impl=_fake,
        tags=(torch.Tag.flexible_layout,),
    )
    routed_lora_two_stage = torch.ops.vllm.routed_lora_two_stage
except AttributeError:
    routed_lora_two_stage = _routed_lora_two_stage
