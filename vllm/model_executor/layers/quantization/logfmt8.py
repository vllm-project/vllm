# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.logfmt8_utils import (
    logfmt8_dequantize, logfmt8_quantize)


class LogFMT8Config(QuantizationConfig):
    """Config class for LogFMT-8bit quantization."""

    def __init__(self,
                 n_bits: int = 8,
                 ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.n_bits = n_bits
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "logfmt8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70  # Assume works on most modern GPUs

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LogFMT8Config":
        n_bits = config.get("n_bits", 8)
        ignored_layers = config.get("ignored_layers")
        return cls(n_bits=n_bits, ignored_layers=ignored_layers)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            # Optionally skip layers
            if prefix in (self.ignored_layers or []):
                return UnquantizedLinearMethod()
            return LogFMT8LinearMethod(self)
        return None


class LogFMT8LinearMethod(LinearMethodBase):
    """Linear method for LogFMT-8bit quantization."""

    def __init__(self, quant_config: LogFMT8Config):
        self.quant_config = quant_config
        self.n_bits = quant_config.n_bits

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # Create quantized weight and scale parameters
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.float32),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        # Store quantization parameters (min, step, etc.) as attributes
        layer.register_buffer("logfmt8_min", torch.zeros(1,
                                                         dtype=torch.float32))
        layer.register_buffer("logfmt8_step",
                              torch.zeros(1, dtype=torch.float32))
        layer.register_buffer("logfmt8_sign",
                              torch.zeros(1, dtype=torch.float32))
        # Optionally add more buffers as needed

    def process_weights_after_loading(self, layer: Module) -> None:
        # Quantize the weights using LogFMT-8bit
        weight_fp32 = layer.weight.data
        qweight, min_val, step_val, sign_val = logfmt8_quantize(
            weight_fp32, n_bits=self.n_bits)
        layer.weight.data = qweight
        layer.logfmt8_min.copy_(min_val)
        layer.logfmt8_step.copy_(step_val)
        layer.logfmt8_sign.copy_(sign_val)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Dequantize weights and perform linear operation
        qweight = layer.weight
        min_val = layer.logfmt8_min
        step_val = layer.logfmt8_step
        sign_val = layer.logfmt8_sign
        weight_fp32 = logfmt8_dequantize(qweight,
                                         min_val,
                                         step_val,
                                         sign_val,
                                         n_bits=self.n_bits)
        output = torch.nn.functional.linear(x, weight_fp32, bias)
        return output
