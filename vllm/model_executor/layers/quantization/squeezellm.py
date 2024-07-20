from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import is_hip


class SqueezeLLMConfig(QuantizationConfig):
    """Config class for SqueezeLLM.

    Reference: https://arxiv.org/pdf/2306.07629
    """

    def __init__(
        self,
        weight_bits: int,
    ) -> None:
        self.weight_bits = weight_bits

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"SqueezeLLM, but got {self.weight_bits} bits.")

        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return f"SqueezeLLMConfig(weight_bits={self.weight_bits})"

    def get_name(self) -> str:
        return "squeezellm"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SqueezeLLMConfig":
        weight_bits = cls.get_from_keys(config, ["wbits"])
        return cls(weight_bits)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            return SqueezeLLMLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SqueezeLLMLinearMethod(QuantizeMethodBase):
    """Linear method for SqueezeLLM.

    Args:
        quant_config: The SqueezeLLM quantization config.
    """

    def __init__(self, quant_config: SqueezeLLMConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        output_size_per_partition = sum(output_partition_sizes)
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            })
        lookup_table = Parameter(
            torch.empty(
                output_size,
                self.quant_config.weight_bits**2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(lookup_table, {
            "output_dim": 0,
        })

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("lookup_table", lookup_table)
        set_weight_attrs(lookup_table, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        lookup_table = layer.lookup_table
        out_shape = x.shape[:-1] + (qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        if is_hip():
            out_f = torch.zeros(out_shape, dtype=torch.float)
            ops.squeezellm_gemm(reshaped_x, qweight, out_f, lookup_table)
            out = out_f.to(dtype=torch.float16)
        else:
            # NOTE: The output tensor should be zero-initialized.
            out = torch.zeros(out_shape, dtype=torch.float16)
            ops.squeezellm_gemm(reshaped_x, qweight, out, lookup_table)

        if bias is not None:
            out.add_(bias)
        return out.reshape(out_shape)
