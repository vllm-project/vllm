import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from torch.nn.parameter import Parameter
from torchao.quantization import (
    int4_weight_only,
    int8_dynamic_activation_int4_weight,
    int8_dynamic_activation_int8_semi_sparse_weight,
    int8_dynamic_activation_int8_weight,
    int8_weight_only,
    quantize_,
)
from vllm.model_executor.utils import set_weight_attrs
from typing import List, Dict, Any, Optional
import torch.nn.functional as F


class TorchAOConfig(QuantizationConfig):
    """Config class for torchao.
    """

    # TODO
    def __init__(self) -> None:
        print("in torchao config init")
        pass

    def __repr__(self) -> str:
        return "TorchAOConfig()"

    def get_name(self) -> str:
        return "torchao"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        # TODO
        print("get config file names in torchao")
        return ["quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TorchAOConfig":
        # TODO
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["TorchAOLinearMethod"]:
        if isinstance(layer, LinearBase):
            return TorchAOLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

class TorchAOLinearMethod(LinearMethodBase):
    """Linear method for torchao.

    Args:
        quant_config: The torchao quantization config.
    """

    def __init__(self, quant_config: TorchAOConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        # unquantized weights
        weight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        # print("quantized weights:", weight)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # dummy_linear = torch.nn.Linear(layer.weight.shape[1], layer.weight.shape[0], bias=False)
        # dummy_linear.weight = layer.weight
        # quantize_(dummy_linear, int4_weight_only())
        # layer.weight = dummy_linear.weight
        return F.linear(x, layer.weight, bias)
