from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import quantization_ops
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.quantization_utils.base import QuantizationConfig


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        return (f"AWQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"zero_point={self.zero_point})")

    @classmethod
    def get_name(cls) -> str:
        return "awq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            "quantize_config.json",  # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq  # pylint: disable=line-too-long
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    @classmethod
    def get_packed_tensors(cls) -> Dict[str, int]:
        return {"qweight": 1, "qzeros": 1}

    @classmethod
    def get_transposed_tensor_names(cls) -> List[str]:
        return ["qweight", "qzeros", "scales"]

    @classmethod
    def get_col_parallel_tensor_names(cls) -> List[str]:
        return ["qweight", "qzeros", "scales"]

    @classmethod
    def get_row_parallel_tensor_names(cls) -> List[str]:
        return ["qweight", "qzeros", "scales"]

    def get_linear_method(self) -> "AWQLinearMethod":
        return AWQLinearMethod(self)


class AWQLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, module: torch.nn.Module, input_size: int,
                       output_size: int, params_dtype: torch.dtype) -> None:
        if input_size % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        qweight = Parameter(
            torch.empty(
                input_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        qzeros = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size // self.quant_config.pack_factor,
                device="cuda",
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        scales = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size,
                device="cuda",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        module.register_parameter("qweight", qweight)
        module.register_parameter("qzeros", qzeros)
        module.register_parameter("scales", scales)

    def apply_weights(self,
                      module: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (module.qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = quantization_ops.awq_gemm(reshaped_x, module.qweight,
                                        module.scales, module.qzeros,
                                        pack_factor)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)


class AWQColumnParallelLinear:
    pass


class AWQRowParallelLinear:
    pass
