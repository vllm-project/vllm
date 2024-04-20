from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class AWQConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(self, weight_bits: int, group_size: int, zero_point: bool,
                 version: str) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version

        if self.weight_bits != 4:
            raise ValueError(
                "Currently, only 4-bit weight quantization is supported for "
                f"AWQ, but got {self.weight_bits} bits.")
        self.pack_factor_int32 = 32 // self.weight_bits
        self.pack_factor_int16 = 16 // self.weight_bits
        self.interleave = 4

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
        VERSION_MIN_CAPABILITY_MAP = {
            "gemm": 75,
            "gemv_fast": 80,
        }

        return VERSION_MIN_CAPABILITY_MAP[self.version]

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
        version = cls.get_from_keys(config, ["version"])
        return cls(weight_bits, group_size, zero_point, version)

    def get_linear_method(self) -> "AWQLinearMethod":
        return AWQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQGemm(torch.nn.Module):

    def __init__(self, quant_config: AWQConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
            self, input_size_per_partition: int,
            output_size_per_partition: int, params_dtype: torch.dtype
    ) -> Tuple[Parameter, Parameter, Parameter]:
        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition //
                self.quant_config.pack_factor_int32,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor_int32,
            })
        qzeros = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.group_size,
                output_size_per_partition //
                self.quant_config.pack_factor_int32,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor_int32,
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

        return qweight, qzeros, scales

    def forward(self, qweight, scales, qzeros, x: torch.Tensor, reshaped_x,
                out_shape) -> torch.Tensor:

        pack_factor = self.quant_config.pack_factor_int32

        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256

        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
            out = torch.matmul(reshaped_x, out)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, scales, qzeros,
                               pack_factor)

        return out.reshape(out_shape)


class AWQGemvFast(torch.nn.Module):

    def __init__(self, quant_config: AWQConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
            self, input_size_per_partition: int,
            output_size_per_partition: int, params_dtype: torch.dtype
    ) -> Tuple[Parameter, Parameter, Parameter]:
        qweight = Parameter(
            torch.empty(
                output_size_per_partition // self.quant_config.interleave,
                input_size_per_partition //
                self.quant_config.pack_factor_int16 *
                self.quant_config.interleave,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor_int16,
                "awq_interleave": self.quant_config.interleave,
            })
        qzeros = Parameter(
            torch.empty(
                calculate_zeros_width(input_size_per_partition,
                                      self.quant_config.group_size) *
                self.quant_config.pack_factor_int32,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qzeros, {
            "input_dim": 0,
            "output_dim": 1,
        })
        scales = Parameter(
            torch.empty(
                calculate_zeros_width(input_size_per_partition,
                                      self.quant_config.group_size) *
                self.quant_config.pack_factor_int32,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })

        return qweight, qzeros, scales

    def forward(self, qweight, scales, qzeros, x: torch.Tensor, reshaped_x,
                out_shape) -> torch.Tensor:

        if x.shape[:-1].numel() < 8:
            out_shape = (x.shape[:-1] +
                         (qweight.shape[0] * self.quant_config.interleave, ))
            out = ops.awq_gemv_fast(input=reshaped_x,
                                    qweight=qweight,
                                    scales=scales,
                                    qzeros=qzeros,
                                    out_features=out_shape[-1],
                                    in_features=reshaped_x.shape[1],
                                    group_size=self.quant_config.group_size)
        else:
            out = ops.awq_gemm_fast(reshaped_x, qweight, scales, qzeros)

        return out


class AWQLinearMethod(LinearMethodBase):
    """Linear method for AWQ.

    Args:
        quant_config: The AWQ quantization config.
    """

    def __init__(self, quant_config: AWQConfig):
        self.quant_config = quant_config
        self.pack_factor = quant_config.pack_factor_int16 if (
            quant_config.version
            == "gemv_fast") else quant_config.pack_factor_int32

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        if output_size_per_partition % self.quant_config.pack_factor_int32 != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        VERSION_TO_AWQ_MODULE_MAP = {
            "gemm": AWQGemm,
            "gemv_fast": AWQGemvFast,
        }
        self.awq_module = VERSION_TO_AWQ_MODULE_MAP[self.quant_config.version](
            self.quant_config)
        qweight, qzeros, scales = self.awq_module.create_weights(
            input_size_per_partition, output_size_per_partition, params_dtype)

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        return {
            "qweight": qweight,
            "qzeros": qzeros,
            "scales": scales,
        }

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.qzeros
        pack_factor = self.pack_factor
        out_shape = (x.shape[:-1] + (qweight.shape[-1] * pack_factor, ))
        reshaped_x = x.reshape(-1, x.shape[-1])

        out = self.awq_module.forward(
            qweight=qweight,
            scales=scales,
            qzeros=qzeros,
            x=x,
            reshaped_x=reshaped_x,
            out_shape=out_shape,
        )

        if bias is not None:
            out.add_(bias)
        return out
