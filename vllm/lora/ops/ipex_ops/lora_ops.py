# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    raise e


def bgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
) -> None:
    ipex.llm.functional.bgmv_shrink(
        inputs, lora_a_weights, output_tensor, lora_indices_tensor, scaling
    )


def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
) -> None:
    ipex.llm.functional.bgmv_expand(
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
    ipex.llm.functional.bgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor,
        lora_indices_tensor,
        slice_offset,
        slice_size,
        add_inputs,
    )
