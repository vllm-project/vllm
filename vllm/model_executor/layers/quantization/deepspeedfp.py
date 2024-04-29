import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)


class DeepSpeedFPConfig(QuantizationConfig):
    """Config for DeepSpeed FP quantizer. It supports fp6 and fp8."""

    def __init__(
        self,
        weight_bits: int = 8,
        rounding: str = "nearest",
        mantissa_bits: int = 3,
        q_range=480.0,
        group_size: int = 512,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.rounding = rounding
        self.mantissa_bits = mantissa_bits
        self.q_range = q_range
        self.valid_types = [torch.bfloat16, torch.float16]

        if self.weight_bits not in [6, 8]:
            raise ValueError(
                "Currently, only 6-bit or 8-bit weight quantization are "
                f"supported for DeepSpeed FP quantizaiton, but got "
                f"{self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"DeepSpeedFPConfig(weight_bits={self.weight_bits}), "
                f"group_size={self.group_size}, "
                f"rounding={self.rounding}, "
                f"mantissa_bits={self.mantissa_bits}, "
                f"")

    @classmethod
    def get_name(cls) -> str:
        return "DeepSpeedFP"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepSpeedFPConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits=weight_bits, group_size=group_size)

    def get_linear_method(self) -> "DeepSpeedFPLinearMethod":
        return DeepSpeedFPLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "quant_config.json",
            "quantize_config.json",
        ]


class DeepSpeedFPLinearMethod(LinearMethodBase):
    """Linear method for DeepSpeedFP quantizer.

    Args:
        quant_config: the DeepSpeedFP quantization config.
    """

    def __init__(self, quant_config: DeepSpeedFPConfig):
        self.quant_config = quant_config
        self.weight = None

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_size_per_partition: int, input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del output_size
        weight = DeepSpeedFPParameter(
            torch.Size(output_size_per_partition,
                       input_size_per_partition),
            quant_config=self.quant_config,
        )
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
        })
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = layer.weight
        y = weight.ds_dequantize()
        return F.linear(x, y, bias)


class DeepSpeedFPParameter(nn.Parameter):
    """
    DeepSpeedFP quantized parameter class that implements fp8/fp6
    quantization deepspeed. Weights are stored in quantized form on
    GPUs, and can be dequantized on-the-fly when needed by the model.
    """

    def __new__(cls, orig_shape: torch.Size, quant_config: DeepSpeedFPConfig):
        from deepspeed.ops.fp_quantizer import FP_Quantize
        data = torch.empty((
            orig_shape.numel() // quant_config.group_size,
            quant_config.group_size * quant_config.weight_bits // 8 + 4,
        ))
        self = torch.Tensor._make_subclass(cls, data, requires_grad=False)
        self.orig_shape = orig_shape
        self.quant_config = quant_config
        self.fp_quantizer = FP_Quantize(group_size=quant_config.group_size)
        return self

    def ds_quantize_(self, tensor: torch.Tensor):
        assert tensor.device.type == "cuda" and tensor.dtype != torch.int8
        return self.data.copy_(
            self.fp_quantizer.quantize(
                tensor.data,
                q_bits=self.quant_config.weight_bits,
            )
        )

    def ds_dequantize(self, fp_out=None) -> torch.Tensor:
        """
        Return a tensor containing the dequantized weights of this parameter.
        """
        assert self.data.device.type == "cuda" and self.data.dtype == torch.int8
        return self.fp_quantizer.dequantize(
            self.data, fp_out=fp_out,
            q_bits=self.quant_config.weight_bits)

    def ds_selective_dequantize(self, indices, fp_out=None) -> torch.Tensor:
        """
        Return a tensor where only the weights at `indices` are dequantized
        (to save HBM -> SRAM bandwidth).
        """
        assert self.data.device.type == "cuda" and self.data.dtype == torch.int8
        return self.fp_quantizer.selective_dequantize(
            self.data, indices, fp_out=fp_out,
            q_bits=self.quant_config.weight_bits)
