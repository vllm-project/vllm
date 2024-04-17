from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.logger import init_logger

logger = init_logger(__name__)

HAS_EETQ = False
try:
    from EETQ import w8_a16_gemm
    HAS_EETQ = True
except ImportError:
    logger.warning("You have not installed EETQ. Please refer to https://github.com/NetEase-FuXi/EETQ/tree/main")
    

class EETQConfig(QuantizationConfig):
    """Config class for eetq.
    https://github.com/NetEase-FuXi/EETQ/tree/main
    """
    def __init__(
        self,
        weight_bits: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.zero_point = zero_point
        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only 8-bit weight quantization is supported for "
                f"EETQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"EETQConfig(weight_bits={self.weight_bits}, \
            zero_point={self.zero_point})")

    def get_name(self) -> str:
        return "eetq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The EETQ kernel only supports Turing or newer GPUs.
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EETQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, zero_point)

    def get_linear_method(self) -> "EETQLinearMethod":
        return EETQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class EETQLinearMethod(LinearMethodBase):
    """Linear method for EETQ.

    Args:
        quant_config: The EETQ quantization config.
    """

    def __init__(self, quant_config: EETQConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        qweight = Parameter(torch.empty(input_size_per_partition,
                                        output_size_per_partition,
                                       dtype=torch.int8),
                           requires_grad=False)
        weight_scales = Parameter(torch.empty(output_size_per_partition,
                                       dtype=torch.float16),
                           requires_grad=False)
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
            })
        set_weight_attrs(
            weight_scales, {
                "input_dim": 0,
                "output_dim": 0
            }
        )
        return {"qweight": qweight, "weight_scales": weight_scales}

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = weights["qweight"].data
        weight_scales = weights["weight_scales"].data
        
        if HAS_EETQ:
            output = w8_a16_gemm(x, qweight, weight_scales)
        else:
            raise ImportError("You have not installed EETQ. Please refer to https://github.com/NetEase-FuXi/EETQ/tree/main")

        if bias is not None:
            output = output + bias
        return output
