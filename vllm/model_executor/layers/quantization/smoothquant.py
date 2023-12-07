from typing import Any, Dict, List, Optional

import torch
from torch._tensor import Tensor
from torch.nn.parameter import Parameter
import threading

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class SmoothQuantConfig(QuantizationConfig):
    """Config class for SmoothQuant

    Reference: https://github.com/mit-han-lab/smoothquant
    """

    def __init__(self,
                 weight_bits: int = 8,
                 quant_type: str = "tensor") -> None:
        self.weight_bits = weight_bits
        self.quant_type = quant_type

        if self.weight_bits != 8:
            raise ValueError(
                "Currently, only w8a8 quantization is supported for "
                f"SmoothQuant, but got {self.weight_bits} bits.")
        if self.quant_type != "tensor":
            raise ValueError(
                "Currently, only tensor wise quantization is supported for "
                f"SmoothQuant, but got {self.quant_type} type quantization.")

    def __repr__(self) -> str:
        return (f"SmoothQuantConfig(weight_bits={self.weight_bits}, "
                f"quant_type={self.quant_type})")

    def get_name(self) -> str:
        return "smoothquant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.float]

    def get_min_capability(self) -> int:
        # The smoothquant kernel only supports Ampere or newer GPUs.
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """List of filenames to search for in the model directory."""
        return [
            "quant_config.json",
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SmoothQuantConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        quant_type = cls.get_from_keys(config, ["quant_type", "q_type"])
        return cls(weight_bits, quant_type)
    
    def get_linear_method(self) -> "SQLinearMethod":
        return SQLinearMethod(Int8GEMM)

    def get_scaled_act_names(self) -> List[str]:
        return []

class Int8GEMM(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "i8cugemm"):
            self.i8cugemm = ops.I8CUGEMM()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Int8GEMM, "_instance"):
            with Int8GEMM._instance_lock:
                if not hasattr(Int8GEMM, "_instance"):
                    Int8GEMM._instance = object.__new__(cls)  
        return Int8GEMM._instance
    
    def get_i8cugemm(self):
        return self.i8cugemm
        

class SQLinearMethod(LinearMethodBase):
    """Linear method for SmoothQuant.
    """
    def __init__(self, gemm):
        i8_gemm = gemm()
        self.i8cugemm = i8_gemm.get_i8cugemm()

    def create_weights(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Tensor]:
        weight = Parameter(
            torch.empty(
                output_size,
                input_size,
                device="cuda",
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {
                "input_dim": 1,
                "output_dim": 0,
            })
        return {"weight": weight}
    
    def apply_weights(self, weights: Dict[str, Tensor], x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Tensor:
        assert bias is None
        weight = weights["weight"]
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        y = torch.empty((x.shape[0], weight.shape[0]),
                        dtype=torch.int32,
                        device=x.device)
        self.i8cugemm.linear_a8_w8_o32_(x, weight, y)
        y = y.view(*x_shape[:-1], -1)
        return y