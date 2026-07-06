# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def _bgmv_shrink_impl(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float,
) -> None:
    torch.ops._xpu_C.bgmv_shrink(
        output_tensor, inputs, lora_a_weights, lora_indices_tensor, scaling
    )


def _bgmv_expand_impl(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool,
) -> None:
    weight_out_dim = lora_b_weights.size(-2)
    output_dim = output_tensor.size(1)

    if weight_out_dim == output_dim:
        torch.ops._xpu_C.bgmv_expand(
            output_tensor,
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            add_inputs,
        )
    elif weight_out_dim < output_dim:
        # LoRA weight output dim can be smaller than the output tensor
        # (e.g. vocab_size vs padded logits). Use expand_slice to write
        # only the matching portion, mirroring torch_ops common_len logic.
        torch.ops._xpu_C.bgmv_expand_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            0,
            weight_out_dim,
            add_inputs,
        )
    else:
        # Weight output dim larger than output tensor: truncate weights.
        lora_b_weights = lora_b_weights[..., :output_dim, :].contiguous()
        torch.ops._xpu_C.bgmv_expand_slice(
            output_tensor,
            inputs,
            lora_b_weights,
            lora_indices_tensor,
            0,
            output_dim,
            add_inputs,
        )


def _bgmv_expand_slice_impl(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
) -> None:
    assert slice_size == lora_b_weights.size(-2)
    assert slice_offset + slice_size <= output_tensor.size(1)
    torch.ops._xpu_C.bgmv_expand_slice(
        output_tensor,
        inputs,
        lora_b_weights,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )


def _bgmv_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float,
) -> None:
    return None


def _bgmv_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool,
) -> None:
    return None


def _bgmv_expand_slice_fake(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="xpu_bgmv_shrink",
    op_func=_bgmv_shrink_impl,
    mutates_args=["output_tensor"],
    fake_impl=_bgmv_shrink_fake,
)

direct_register_custom_op(
    op_name="xpu_bgmv_expand",
    op_func=_bgmv_expand_impl,
    mutates_args=["output_tensor"],
    fake_impl=_bgmv_expand_fake,
)

direct_register_custom_op(
    op_name="xpu_bgmv_expand_slice",
    op_func=_bgmv_expand_slice_impl,
    mutates_args=["output_tensor"],
    fake_impl=_bgmv_expand_slice_fake,
)


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    torch.ops.vllm.xpu_bgmv_shrink(
        inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
) -> None:
    torch.ops.vllm.xpu_bgmv_expand(
        inputs, lora_b_weights, output_tensor, lora_indices_tensor, add_inputs
    )


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
) -> None:
    torch.ops.vllm.xpu_bgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )
