from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm._C import ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


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

    def get_name(self) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half]

    def get_min_capability(self) -> int:
        # The AWQ kernel only supports Turing or newer GPUs.
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",  # E.g., casperhansen/vicuna-7b-v1.5-awq
            # E.g., abhinavkulkarni/mosaicml-mpt-7b-instruct-w4-g128-awq
            "quantize_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def get_linear_method(self) -> "AWQLinearMethod":
        return AWQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config

    def create_weights(self, input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })
        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
        }

    def apply_weights(self,
                      weights: Dict[str, Any],
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = weights["qweight"]
        scales = weights["scales"]
        qzeros = weights["qzeros"]
        pack_factor = self.quant_config.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                               pack_factor)
        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
