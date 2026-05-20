# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    torch.ops._xpu_C.bgmv_shrink(
        output_tensor, inputs, lora_a_weights, lora_indices_tensor, scaling
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
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


def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
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
