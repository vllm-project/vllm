# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025, Oracle and/or its affiliates.

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

logger = init_logger(__name__)
"""By default, use 8 bit as target precision, but it can be 
overridden by setting the RTN_NUM_BITS envvar
"""
NUM_BITS = os.getenv('RTN_NUM_BITS', "8")
"""By default, use group size of 128 parameters, but it can be 
overridden by setting the RTN_GROUP_SIZE envvar
"""
GROUP_SIZE = os.getenv('RTN_GROUP_SIZE', "128")


class RTNConfig(QuantizationConfig):
    """Config class for RTN.
    """

    def __init__(
            self,
            weight_bits: int = int(NUM_BITS),
            group_size: int = int(GROUP_SIZE),
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size

        if self.weight_bits != 4 and self.weight_bits != 8:
            raise ValueError(
                "Currently, only 4-bit or 8-bit weight quantization is "
                f"supported for RTN, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"RTNConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "rtn"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RTNConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["RTNLinearMethod"]:
        if isinstance(layer, LinearBase):
            return RTNLinearMethod(self)
        return None


class RTNTensor:
    """A wrapper over Tensor that enables quantization on-the-fly by
    overloading the copy_ method.
    """

    def __init__(self, data: torch.Tensor, scale: torch.Tensor,
                 quant_config: RTNConfig) -> None:
        self.data = data
        self.scale = scale
        self.quant_config = quant_config

    def narrow(self, dim, start, length):
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        return RTNTensor(
            self.data.narrow(dim, start // factor, length // factor),
            self.scale.narrow(dim, start, length), self.quant_config)

    @property
    def shape(self):
        shape = self.data.shape
        factor = 1 if self.quant_config.weight_bits == 8 else 2
        return torch.Size((shape[0] * factor, shape[1]))

    def copy_(self, loaded_weight: torch.Tensor) -> None:
        qweight, weight_scale = rtn_quantize(loaded_weight.cuda(),
                                             self.quant_config.weight_bits,
                                             self.quant_config.group_size)

        self.data.copy_(qweight)
        self.scale.data.copy_(weight_scale)


class RTNParameter(Parameter):
    """A wrapper over Parameter that returns RTNTensor (a wrapper over Tensor)
    when its data is accessed. We need this wrapper for the data loading phase
    only, so we can intercept a weight copying function (torch.Tensor.copy_)
    and apply quantization on-the-fly.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, scale: torch.Tensor,
                 quant_config: RTNConfig) -> None:
        self.scale = scale
        self.quant_config = quant_config

    @property
    def data(self):
        return RTNTensor(super().data, self.scale, self.quant_config)


class RTNLinearMethod(LinearMethodBase):
    """Linear method for RTN.

    Args:
        quant_config: The RTN quantization config.
    """

    def __init__(self, quant_config: RTNConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        num_groups_per_col = (input_size_per_partition //
                              self.quant_config.group_size
                              if self.quant_config.group_size != -1 else 1)

        scale = Parameter(
            torch.empty(output_size_per_partition,
                        num_groups_per_col,
                        dtype=params_dtype),
            requires_grad=False,
        )
        factor = 1 if self.quant_config.weight_bits == 8 else 2

        weight = RTNParameter(data=torch.empty(output_size_per_partition //
                                               factor,
                                               input_size_per_partition,
                                               dtype=torch.int8),
                              scale=scale,
                              quant_config=self.quant_config)

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

        layer.register_parameter("scale", scale)
        layer.output_size_per_partition = output_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """torch.compile does not know how to deal with a Parameter subclass
        (aka RTNParameter). As we don't really need RTNParameters for the
        forward pass, we replace them with equivalent instances of Parameters.
        """
        old_weight = layer.weight
        assert isinstance(old_weight, RTNParameter)
        data = old_weight.data.data

        delattr(layer, "weight")

        new_weight = Parameter(data=data, requires_grad=False)
        layer.register_parameter("weight", new_weight)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.weight
        scale = layer.scale

        weight = rtn_dequantize(qweight, scale)
        out = F.linear(x, weight)
        del weight
        if bias is not None:
            out.add_(bias)

        return out


def rtn_quantize(tensor: torch.Tensor, num_bits: int,
                 group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor using per-group static scaling factor.

    Args:
        tensor: The input tensor.
        num_bits: Target precision for the result (supported values are
                  8 or 4).
        group_size: Quantization granularity. 
                    If equal to -1, each row in the input tensor is treated
                    as one group.
    """

    q_range = 2**num_bits
    num_groups = (tensor.shape[0] * tensor.shape[1] //
                  group_size if group_size != -1 else tensor.shape[0])
    """Calculate a scaling factor per input group.
    """
    input_flat = tensor.reshape(num_groups, -1)
    input_min = torch.min(input_flat, dim=1, keepdim=True)[0]
    input_max = torch.max(input_flat, dim=1, keepdim=True)[0]
    input_max_abs = torch.max(input_min.abs(), input_max.abs())
    scale = (input_max_abs * 2.0 / (q_range - 1))
    """Scale each input group, truncate and round to the nearest integer.
    """
    scaled_input = input_flat / scale
    scaled_input = scaled_input.clamp(-q_range // 2, q_range // 2 - 1)
    scaled_input = scaled_input.round()

    scale = scale.reshape(tensor.shape[0], -1).contiguous()
    inputs_q = scaled_input.reshape(tensor.shape).to(torch.int8)
    inputs_q = inputs_q.contiguous()

    if num_bits == 4:
        """Pack two 4-bit values into each byte.
        """
        inputs_q = (inputs_q[:, 1::2] << 4) | (inputs_q[:, ::2] & 0xf)
        inputs_q = inputs_q.reshape(tensor.shape[0] // 2, tensor.shape[1])
        inputs_q = inputs_q.contiguous()

    return inputs_q, scale


def rtn_dequantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize a tensor using per-group static scaling factors.

    Args:
        tensor: The input tensor.
        scale: The tensor with per-group scale factors.
    """

    num_groups = scale.size(0) * scale.size(1)
    input_dim, output_dim = tensor.shape

    num_bits = 8 if input_dim == scale.size(0) else 4
    if num_bits == 4:
        input_dim *= 2

    data = torch.empty((input_dim, output_dim),
                       dtype=scale.dtype,
                       device=tensor.device)

    if num_bits == 8:
        data.copy_(tensor)
    else:
        """Unpack two 4-bit values from each byte.
        """
        tensor = tensor.reshape(input_dim, output_dim // 2)
        for i in range(2):
            data[:, i::2] = (tensor << 4 * (1 - i)) >> 4
    """Scale each input group with its scaling factor.
    """
    scale = scale.reshape(num_groups, -1)
    data = data.reshape(num_groups, -1)
    data = torch.mul(data, scale)

    input_deq = data.reshape((input_dim, output_dim)).contiguous()
    return input_deq
