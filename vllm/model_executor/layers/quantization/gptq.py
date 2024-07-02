import enum
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.utils import set_weight_attrs


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
    ) -> None:
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
                f"desc_act={self.desc_act}),"
                f"lm_head_quantized={self.lm_head_quantized}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["GPTQLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return GPTQLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


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
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
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
        g_idx = Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(input_size_per_partition)
                ],
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})
        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": scale_and_zero_input_dim,
            "output_dim": 1,
        })

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("g_idx", g_idx)
        set_weight_attrs(g_idx, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        layer.exllama_state = exllama_state

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qweight = layer.qweight
        out_shape = x.shape[:-1] + (qweight.shape[-1], )
        reshaped_x = x.reshape(-1, x.shape[-1])
        # exllama needs to shuffle the weight after the weight is loaded
        # here we do the shuffle on first forward pass
        if layer.exllama_state == ExllamaState.UNINITIALIZED:
            if self.quant_config.desc_act:
                layer.g_idx.data = torch.argsort(layer.g_idx).to(torch.int)
            else:
                layer.g_idx.data = torch.empty((0, ),
                                               device=layer.g_idx.device)
            layer.exllama_state = ExllamaState.READY
            ops.gptq_shuffle(layer.qweight, layer.g_idx,
                             self.quant_config.weight_bits)
        output = ops.gptq_gemm(reshaped_x, layer.qweight, layer.qzeros,
                               layer.scales, layer.g_idx,
                               layer.exllama_state == ExllamaState.READY,
                               self.quant_config.weight_bits)
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
