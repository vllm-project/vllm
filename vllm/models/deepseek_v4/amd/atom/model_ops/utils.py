# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import logging
from functools import cache
from typing import List, Optional, Tuple, Union

import torch
from aiter import per_tensor_quant
from aiter.ops.shuffle import shuffle_weight
from torch import nn

from vllm.models.deepseek_v4.amd.atom.utils import envs

logger = logging.getLogger("atom")


def atom_parameter(data: torch.Tensor) -> nn.Parameter:
    """Create an ``nn.Parameter`` with gradient tracking controlled by
    the ``ATOM_REQUIRES_GRAD`` environment variable (default: disabled).

    Use this instead of ``nn.Parameter(...)`` everywhere in ATOM so that
    inference vs. training gradient behaviour is controlled from a single
    place.
    """
    requires_grad = envs.ATOM_REQUIRES_GRAD and (
        data.is_floating_point() or data.is_complex()
    )
    return nn.Parameter(data, requires_grad=requires_grad)


@cache
def _has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.

    The result is cached so that subsequent queries for the same module incur
    no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None


MXFP4_QUANT_BLOCK_SIZE = 32


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float)
    dq_weight = fake_qweight * inv_scale
    return dq_weight


def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale


def requantize_with_max_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_widths: List[int],
    normalize_e4m3fn_to_e4m3fnuz=False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    if normalize_e4m3fn_to_e4m3fnuz:
        quant_dtype = torch.float8_e4m3fnuz
        weight = weight.view(torch.float8_e4m3fn)
        max_w_scale = weight_scale.max() * 2.0
    else:
        quant_dtype = weight.dtype
        max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (
        weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint or normalize_e4m3fn_to_e4m3fnuz:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight.view(quant_dtype)[start:end, :], _ = per_tensor_quant(
                weight_dq, max_w_scale, quant_dtype=quant_dtype
            )
            start = end

    return max_w_scale, weight.view(quant_dtype)


def shuffle_weights(*tensors: torch.nn.Parameter, layout: tuple[int, int] = (16, 16)):
    """
    Applies shuffle_weight function from AITER to each
    input tensor and returns them.

    Rearranges (shuffles) the input tensor/s
    into a specified block layout for optimized computation.

    Args:
        *tensors: Variable number of torch.Tensor objects.
        layout: A pair of integers specifying the
        block sizes used to divide the tensors during shuffling.
        Default is (16, 16).

    Returns:
    A Tuple of shuffled tensors.
    """
    for tensor in tensors:
        if not isinstance(tensor, torch.nn.Parameter):
            raise TypeError(f"Expected torch.nn.Parameter, but got {type(tensor)}")

        weight = tensor.data
        if weight.dim() == 2:
            tensor.data = shuffle_weight(weight, layout=layout)
        elif weight.dim() == 3:
            # Split fully on dim0 and shuffle each 2D slice independently.
            for i in range(weight.shape[0]):
                weight[i].copy_(shuffle_weight(weight[i], layout=layout))
            tensor.data = weight
        else:
            raise ValueError(
                f"Expected weight dim to be 2 or 3 for shuffle, got {weight.dim()}"
            )

        tensor.is_shuffled = True

