from typing import Any, Dict, List, Optional

import torch
from gguf.constants import GGML_QUANT_SIZES
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, ) -> None:
        pass

    def __repr__(self) -> str:
        return ("GGUFConfig()")
    
    @property
    def merge_weight(self) -> bool:
        return False

    @property
    def is_neox_style(self) -> bool:
        return False

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GGUFConfig":
        return cls()

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["GGUFLinearMethod"]:
        if isinstance(layer, LinearBase):
            return GGUFLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        output_size_per_partition = sum(output_partition_sizes)

        qweight = UninitializedParameter(requires_grad=False)
        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(torch.empty(1, dtype=torch.uint8), requires_grad=False)
        set_weight_attrs(qweight_type, {"ignore_warning": True})
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.data.item()
        # use dequantize mulmat for IQmatrix, mmq for k-quants
        if qweight_type >= 16:
            block_size, type_size = GGML_QUANT_SIZES[qweight_type]
            shape = (qweight.shape[0], qweight.shape[1]//type_size*block_size)
            weight = ops.ggml_dequantize(qweight.contiguous(), qweight_type, *shape)
            out = x @ weight.T
        else:
            out = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
        if bias:
            out.add_(bias)
        return out
