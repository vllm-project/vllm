# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.parameter import ModelWeightParameter

ACTIVATION_SCHEMES = ["none"]


class Int8TpuConfig(QuantizationConfig):
    """Int8 Quantization Config class for TPU Backend."""

    def __init__(
        self,
        activation_scheme: str = "none",
    ) -> None:
        super().__init__()
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme

    def get_name(self) -> QuantizationMethods:
        return "tpu_int8"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "This function should not be called with TPU Backend")

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Int8TpuConfig":
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(activation_scheme=activation_scheme)

    def get_quant_method(self, layer: Module,
                         prefix: str) -> Optional["TPUInt8LinearMethod"]:
        if isinstance(layer, LinearBase):
            return TPUInt8LinearMethod(self)
        return None


class TPUInt8LinearMethod(LinearMethodBase):
    """Int8 Linear method for TPU Quant. """

    def __init__(self, quant_config: Int8TpuConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: Module, input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):

        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

    def _quantize_weight(
            self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_dtype = weight.dtype
        weight = weight.cpu().to(torch.float32)
        n_bit = 8
        eps = 1e-5
        max_int = 2**(n_bit - 1) - 1
        min_int = -(2**(n_bit - 1))
        max_val = weight.abs().amax(dim=-1, keepdim=True)
        max_val = max_val.clamp(min=eps)
        qscale = max_val / max_int
        qweight = torch.clamp(torch.round(weight * (1.0 / qscale)), min_int,
                              max_int).to(torch.int8)
        qscale = qscale.squeeze().to(weight_dtype)
        return qweight, qscale

    def process_weights_after_loading(self, layer: Module) -> None:
        layer.weight = Parameter(layer.weight.data, requires_grad=False)
        device = layer.weight.device
        qweight, qscale = self._quantize_weight(layer.weight)
        qweight = qweight.to(device)
        qscale = qscale.to(device)
        layer.weight = Parameter(qweight, requires_grad=False)
        layer.scale = Parameter(qscale, requires_grad=False)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            import torch_xla.experimental.xla_quantized_matmul  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "Please install torch_xla by following the instructions at "
                "https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html "  # noqa: E501
                "to run vLLM on TPU.") from err
        weight = layer.weight
        scale = layer.scale
        out = torch.ops.xla.quantized_matmul(x, weight, scale)
        if bias is not None:
            out = out + bias
        return out
