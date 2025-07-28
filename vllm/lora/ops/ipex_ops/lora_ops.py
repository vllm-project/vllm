# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except ImportError as e:
    logger.warning("Import error msg: %s", e.msg)


def _bgmv_shrink(inputs: torch.Tensor,
                 lora_a_weights: torch.Tensor,
                 output_tensor: torch.Tensor,
                 lora_indices_tensor: torch.Tensor,
                 scaling: float = 1.0) -> None:

    ipex.llm.functional.bgmv_shrink(inputs, lora_a_weights, output_tensor,
                                    lora_indices_tensor, scaling)


def _bgmv_shrink_fake(inputs: torch.Tensor,
                      lora_a_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      scaling: float = 1.0) -> None:
    pass


def _bgmv_expand(inputs: torch.Tensor,
                 lora_b_weights: torch.Tensor,
                 output_tensor: torch.Tensor,
                 lora_indices_tensor: torch.Tensor,
                 add_inputs: bool = True) -> None:
    ipex.llm.functional.bgmv_expand(inputs, lora_b_weights, output_tensor,
                                    lora_indices_tensor, add_inputs)


def _bgmv_expand_fake(inputs: torch.Tensor,
                      lora_b_weights: torch.Tensor,
                      output_tensor: torch.Tensor,
                      lora_indices_tensor: torch.Tensor,
                      add_inputs: bool = True) -> None:
    pass


def _bgmv_expand_slice(inputs: torch.Tensor,
                       lora_b_weights: torch.Tensor,
                       output_tensor: torch.Tensor,
                       lora_indices_tensor: torch.Tensor,
                       slice_offset: int,
                       slice_size: int,
                       add_inputs: bool = True) -> None:
    ipex.llm.functional.bgmv_expand_slice(inputs, lora_b_weights,
                                          output_tensor, lora_indices_tensor,
                                          slice_offset, slice_size, add_inputs)


def _bgmv_expand_slice_fake(inputs: torch.Tensor,
                            lora_b_weights: torch.Tensor,
                            output_tensor: torch.Tensor,
                            lora_indices_tensor: torch.Tensor,
                            slice_offset: int,
                            slice_size: int,
                            add_inputs: bool = True) -> None:
    pass


try:
    direct_register_custom_op(
        op_name="bgmv_shrink",
        op_func=_bgmv_shrink,
        mutates_args=["output_tensor"],
        fake_impl=_bgmv_shrink_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    bgmv_shrink = torch.ops.vllm.bgmv_shrink

    direct_register_custom_op(
        op_name="bgmv_expand",
        op_func=_bgmv_expand,
        mutates_args=["output_tensor"],
        fake_impl=_bgmv_expand_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    bgmv_expand = torch.ops.vllm.bgmv_expand

    direct_register_custom_op(
        op_name="bgmv_expand_slice",
        op_func=_bgmv_expand_slice,
        mutates_args=["output_tensor"],
        fake_impl=_bgmv_expand_slice_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    bgmv_expand_slice = torch.ops.vllm.bgmv_expand_slice

except AttributeError:
    bgmv_shrink = _bgmv_shrink
    bgmv_expand = _bgmv_expand
    bgmv_expand_slice = _bgmv_expand_slice
