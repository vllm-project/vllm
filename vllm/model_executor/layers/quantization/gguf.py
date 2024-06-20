from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_min_capability(self) -> int:
        return 70

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
        set_weight_attrs(qweight, {"use_gguf": True})
        set_weight_attrs(qweight, {
            "tensor_shape":
            (output_size_per_partition, input_size_per_partition)
        })
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(torch.empty(1, dtype=torch.uint8),
                                 requires_grad=False)
        set_weight_attrs(qweight_type, {"ignore_warning": True})
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        shard_id = getattr(layer.qweight, "shard_id", None)
        shard_size = getattr(layer.qweight, "shard_size", None)
        qweight_type = layer.qweight_type.data.item()
        # we need to slice merged weight to pass right data
        if shard_id and shard_size:
            out = []
            block_size, type_size = GGML_QUANT_SIZES[qweight_type]
            offset = 0
            # dequantize shard weights respectively
            for id in shard_id:
                shard_weight = layer.qweight[offset:offset + shard_size[id], :]
                # the shape of dequantized shard weight
                shard_shape = (shard_weight.shape[0],
                               shard_weight.shape[1] // type_size * block_size)
                out.append(ops.ggml_dequantize(shard_weight, qweight_type, *shard_shape))
                offset += shard_size[id]
            out = torch.cat(out, axis=0)
        else:
            shape = layer.qweight.tensor_shape
            out = ops.ggml_dequantize(layer.qweight, qweight_type, *shape)
        out = F.linear(x, out, bias)
        return out
