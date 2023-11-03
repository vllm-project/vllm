from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.quantization_utils.base import QuantizationConfig


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

    @classmethod
    def get_name(cls) -> str:
        return "squeezellm"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SqueezeLLMConfig":
        weight_bits = cls.get_from_keys(config, ["wbits"])
        return cls(weight_bits)

    @classmethod
    def get_packed_tensors(cls) -> Dict[str, int]:
        return {"qweight": 0}

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["qweight"]

    @classmethod
    def get_col_parallel_tensor_names(cls) -> List[str]:
        return ["qweight", "lookup_table"]

    @classmethod
    def get_row_parallel_tensor_names(cls) -> List[str]:
        return ["qweight"]

    def get_linear_method(self) -> "SqueezeLLMLinearMethod":
        return SqueezeLLMLinearMethod(self)


class SqueezeLLMLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: SqueezeLLMConfig):
        self.quant_config = quant_config

    def create_weights(self, module: torch.nn.Module, input_size: int,
                       output_size: int, params_dtype: torch.dtype) -> None:
        if input_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        qweight = Parameter(
            torch.empty(
                input_size // self.quant_config.pack_factor,
                output_size,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        lookup_table = Parameter(
            torch.empty(
                output_size,
                self.quant_config.weight_bits**2,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        module.register_parameter("qweight", qweight)
        module.register_parameter("lookup_table", lookup_table)

    def apply_weights(self,
                      module: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (module.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        # NOTE: The output tensor should be zero-initialized.
        out = torch.zeros(out_shape, device="cuda", dtype=torch.float16)
        quantization_ops.squeezellm_gemm(reshaped_x, module.qweight, out,
                                         module.lookup_table)

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)


class SqueezeLLMColumnParallelLinear:
    pass


class SqueezeLLMRowParallelLinear:
    pass
