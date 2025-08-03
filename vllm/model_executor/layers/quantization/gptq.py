# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from enum import Enum
from fractions import Fraction
from typing import Any, Optional, Union

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)


class GPTQConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, Union[int, bool]]],
    ) -> None:
        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is dict[str, dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        super().__init__()
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}), "
                f"lm_head_quantized={self.lm_head_quantized}), "
                f"dynamic={self.dynamic}")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized,
                   dynamic)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[Union["GPTQLinearMethod", "QuantizeMethodBase"]]:
        if isinstance(layer, FusedMoE):
            # GPTQ MoE support: fall back to MoeWNA16 for broad compatibility
            from .moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "gptq",
                "bits": self.weight_bits,
                "group_size": self.group_size,
                "sym": True,  # GPTQ typically uses symmetric quantization
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(
                layer, prefix)

        return get_linear_quant_method(self, layer, prefix, GPTQLinearMethod)


class ExllamaState(Enum):

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
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
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        output_size_per_partition = sum(output_partition_sizes)
        if (output_size_per_partition % self.quant_config.pack_factor.numerator
                != 0):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        g_idx = RowvLLMParameter(data=torch.tensor(
            [
                i // self.quant_config.group_size
                for i in range(input_size_per_partition)
            ],
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)
        qzeros_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }
        if scale_and_zero_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            scales = GroupQuantScaleParameter(output_dim=1,
                                              input_dim=0,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.exllama_state = exllama_state

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # for torch.compile
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if layer.exllama_state == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty((0, ),
                                               dtype=torch.int,
                                               device=layer.g_idx.device)
            layer.exllama_state = ExllamaState.READY
            ops.gptq_shuffle(layer.qweight, layer.g_idx,
                             self.quant_config.weight_bits)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        output = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                               layer.scales, layer.g_idx,
                               layer.exllama_state == ExllamaState.READY,
                               self.quant_config.weight_bits)
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
