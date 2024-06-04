from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class GGMLConfig(QuantizationConfig):
    """Config class for GGML."""

    def __init__(self, ) -> None:
        pass

    def __repr__(self) -> str:
        return ("GGMLConfig()")

    def get_name(self) -> str:
        return "ggml"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGMLConfig":
        return cls()

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["GGMLLinearMethod"]:
        if isinstance(layer, LinearBase):
            return GGMLLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class GGMLLinearMethod(LinearMethodBase):
    """Linear method for GGML.

    Args:
        quant_config: The GGML quantization config.
    """

    def __init__(self, quant_config: GGMLConfig):
        self.quant_config = quant_config
        self.block_size = 32

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)
        quants = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=torch.int8),
                           requires_grad=False)
        set_weight_attrs(quants, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(quants, extra_weight_attrs)
        layer.register_parameter("quants", quants)

        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.block_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 1,
            "output_dim": 0,
            "ggml_scales": True
        })
        set_weight_attrs(scales, extra_weight_attrs)
        layer.register_parameter("scales", scales)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # dequantized for Q4_0 and Q8_0
        shape = layer.quants.shape
        out = layer.quants.reshape(-1, self.block_size) * layer.scales.reshape(
            -1, 1)
        out = torch.matmul(x, out.reshape(shape).T)
        if bias is not None:
            out.add_(bias)
        return out
