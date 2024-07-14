import enum
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.fused_moe_gptq import fused_moe_gptq
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
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

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return GPTQLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return GPTQFusedMoEMethod(self)
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
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


class GPTQFusedMoEMethod(FusedMoEMethodBase):
    """MoE method with gptq quantization."""

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        group_size = self.quant_config.group_size
        assert hidden_size % group_size == 0
        assert intermediate_size % group_size == 0
        assert not self.quant_config.desc_act
        assert self.quant_config.weight_bits in [
            4
        ], "Only 4-bit weight is supported for GPTQ fused MoE"
        pack_factor = self.quant_config.pack_factor

        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                                     hidden_size //
                                                     pack_factor,
                                                     2 * intermediate_size,
                                                     dtype=torch.int32),
                                         requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(
            w13_qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": pack_factor,
            })
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w13_qzeros = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // group_size,
            2 * (intermediate_size // pack_factor),
            dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(
            w13_qzeros, {
                "input_dim": 1,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": pack_factor,
            })
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(torch.empty(num_experts,
                                                    hidden_size // group_size,
                                                    2 * intermediate_size,
                                                    dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, {
            "input_dim": 0,
            "output_dim": 1,
        })
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(2 * hidden_size)
                ],
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, {"input_dim": 0, "ignore_warning": True})
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                                    intermediate_size //
                                                    pack_factor,
                                                    hidden_size,
                                                    dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(
            w2_qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": pack_factor,
            })
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(torch.empty(num_experts,
                                                   intermediate_size //
                                                   group_size,
                                                   hidden_size // pack_factor,
                                                   dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(
            w2_qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": pack_factor,
            })
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.empty(num_experts,
                                                   intermediate_size //
                                                   group_size,
                                                   hidden_size,
                                                   dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, {
            "input_dim": 0,
            "output_dim": 1,
        })
        set_weight_attrs(w2_scales, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.tensor(
                [
                    i // self.quant_config.group_size
                    for i in range(intermediate_size)
                ],
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, {"input_dim": 0, "ignore_warning": True})
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        assert not self.quant_config.desc_act, \
            "desc_act is not supported for GPTQ fused MoE"

    def _pack_int8_tensor_to_packed_int4(self, unpacked_tensor):
        length = unpacked_tensor.shape[1]
        first_half, second_half = unpacked_tensor[:, 0:length:
                                                  2, :], unpacked_tensor[:, 1:
                                                                         length:
                                                                         2, :]
        packed_first_half = first_half & 0x0F  # even
        packed_second_half = second_half & 0x0F  # odd
        # even in low 4 bits
        packed_tensor = packed_first_half | (packed_second_half << 4).to(
            torch.int8)
        return packed_tensor

    def _convert_weight_from_gptq(self, weight: torch.Tensor,
                                  bit: int) -> torch.Tensor:
        wf = torch.tensor(list(range(0, 32, bit)),
                          dtype=torch.int32).unsqueeze(0).to(weight.device)
        weight = torch.bitwise_right_shift(
            input=torch.unsqueeze(weight, 2).expand(-1, -1, 32 // bit, -1),
            other=wf.unsqueeze(-1),
        ).to(torch.int16 if bit == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**bit) - 1)
        weight = weight.reshape(weight.shape[0], -1, weight.shape[-1])
        weight = weight - (1 << (bit - 1))  # unsigned -> signed
        weight = weight.transpose(1, 2).contiguous()
        if bit == 4:
            weight = self._pack_int8_tensor_to_packed_int4(weight)
        return weight

    def _convert_zero_point_from_gptq(self, zero_point: torch.Tensor,
                                      bit: int) -> torch.Tensor:
        wf = torch.tensor(list(range(0, 32, bit)),
                          dtype=torch.int32).unsqueeze(0).to(zero_point.device)
        zero_point = torch.bitwise_right_shift(
            input=torch.unsqueeze(zero_point, 3).expand(-1, -1, -1, 32 // bit),
            other=wf.unsqueeze(0),
        ).to(torch.int16 if bit == 8 else torch.int8)
        zero_point = torch.bitwise_and(zero_point, (2**bit) - 1)
        zero_point = zero_point + 1
        zero_point = zero_point.reshape(
            zero_point.shape[0], -1, zero_point.shape[2] * zero_point.shape[3])
        zero_point = zero_point - (1 << (bit - 1))  # unsigned -> signed
        zero_point = zero_point.transpose(1, 2).contiguous()
        return zero_point

    def _process_weight_after_loading(self, layer: torch.nn.Module,
                                      key: str) -> None:
        bit = self.quant_config.weight_bits

        qweight_key = f'{key}_qweight'
        weight = getattr(layer, qweight_key).data
        device = weight.device
        weight = self._convert_weight_from_gptq(weight,
                                                bit).to(torch.int8).cpu()
        weight = weight.to(device)
        delattr(layer, qweight_key)
        layer.register_parameter(
            qweight_key, torch.nn.Parameter(weight, requires_grad=False))

        qzeros_key = f'{key}_qzeros'
        zero_point = getattr(layer, qzeros_key).data
        device = zero_point.device
        zero_point = self._convert_zero_point_from_gptq(zero_point, bit)
        delattr(layer, qzeros_key)
        layer.register_parameter(
            qzeros_key, torch.nn.Parameter(zero_point, requires_grad=False))

        scales_key = f'{key}_scales'
        scales = getattr(layer, scales_key).data
        device = scales.device
        scales = scales.transpose(1, 2).contiguous()
        delattr(layer, scales_key)
        layer.register_parameter(
            scales_key, torch.nn.Parameter(scales, requires_grad=False))

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._process_weight_after_loading(layer, 'w13')
        self._process_weight_after_loading(layer, 'w2')

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool = True,
              use_grouped_topk: bool = False,
              num_expert_group: Optional[int] = None,
              topk_group: Optional[int] = None) -> torch.Tensor:

        return fused_moe_gptq(x,
                              layer.w13_qweight,
                              layer.w13_qzeros,
                              layer.w13_scales,
                              layer.w2_qweight,
                              layer.w2_qzeros,
                              layer.w2_scales,
                              router_logits,
                              top_k,
                              renormalize=renormalize,
                              inplace=True,
                              use_grouped_topk=use_grouped_topk,
                              num_expert_group=num_expert_group,
                              topk_group=topk_group,
                              quantize_bits=self.quant_config.weight_bits)
