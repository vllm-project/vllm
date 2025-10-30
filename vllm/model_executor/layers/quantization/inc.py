# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Intel Gaudi supports quantization of various modules and functions,
# including, but not limited to `Linear`, `KVCache`, `Matmul` and `Softmax`.
# During model loading,
# INC will patch layers with quantization/dequantization operators.
# Meanwhile, INC will convert original weight to target datatype
# and loading to target device.
# static scaling should be provided through Quant_CONFIG:
# `QUANT_CONFIG` is an environment variable,
# that points to the measurement or quantization JSON config file.
# The measurement configuration file is used during the calibration procedure,
# to collect measurements for a given model.
# The quantization configuration is used during inference.
# For more information, please refer to:
# https://docs.habana.ai/en/v1.21.1/PyTorch/vLLM_Inference/vLLM_FP8_Inference.html

from typing import Any, Optional

import torch

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


class INCConfig(QuantizationConfig):
    """Config class for FP8 using Intel Neural Compressor."""

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "INCConfig":
        raise AssertionError

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            return UnquantizedFusedMoEMethod(layer.moe_config)
        return None

    @classmethod
    def get_min_capability(cls) -> int:
        raise AssertionError

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []
