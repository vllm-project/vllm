# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
        QuantizationConfig)

__all__ = ["is_activation_quantization_format", "get_linear_quantization"]


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


def get_linear_quantization(
        config: "QuantizationConfig"
) -> tuple[QuantizationArgs, QuantizationArgs]:
    for scheme in config.quantization_config.config_groups:
        if scheme.targets == "Linear":
            return (scheme.input_activations, scheme.weights)

    raise ValueError(
        "Could not find a quantization scheme applied to all linears")
