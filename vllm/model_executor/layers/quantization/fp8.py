from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import MoEMethodBase, fused_moe
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs


class FP8Config(QuantizationConfig):
    """Config class for FP8."""

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO: PyTorch 2.3.0+ is required to run FP8 on
        # SM 89 (e.g. Ada) GPUs. Specifically, this PR has to
        # be included: https://github.com/pytorch/pytorch/pull/118881
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FP8Config":
        return cls()

    def get_quantize_method(
            self, layer: torch.nn.Module) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        if "MoE" in layer.__class__.__name__:
            return Fp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    We now support common FP16/BF16 model checkpoints ONLY. The weight
    scaling factor will be initialized after the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)
       
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP8Config):
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
        output_size_per_partition = sum(output_partition_sizes)
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=params_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)

        w_scale = Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("weight_scaling_factor", w_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Although the quant_method is propagated to all layers,
        # only linear layers invoke "create_weights". So we check
        # whether "weight_scaling_facor" is registered to determine
        # whether the layer is a linear layer that requires quantization.
        if not hasattr(layer, "weight_scaling_factor"):
            return

        qweight, weight_scale = ops.scaled_fp8_quant(layer.weight)
        # torch._scaled_mm requires column-major in the second
        # input (weight), so we transpose the quantized weight.
        layer.weight = Parameter(qweight.t(), requires_grad=False)
        layer.weight_scaling_factor.data.copy_(weight_scale)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        qinput, x_scale = ops.scaled_fp8_quant(x)
        output, _ = torch._scaled_mm(
            qinput,
            layer.weight,
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=layer.weight_scaling_factor,
            bias=bias,
        )
        return output


class Fp8MoEMethod(MoEMethodBase):
    """MoE method for FP8.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: FP8Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_total_experts: int,
                       intermediate_size: int, hidden_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        ws = Parameter(
            torch.empty(num_total_experts,
                        2 * intermediate_size,
                        hidden_size,
                        dtype=params_dtype))
        w2s = Parameter(
            torch.empty(num_total_experts,
                        hidden_size,
                        intermediate_size,
                        dtype=params_dtype))
        layer.register_parameter("ws", ws)
        layer.register_parameter("w2s", w2s)
        set_weight_attrs(ws, extra_weight_attrs)
        set_weight_attrs(w2s, extra_weight_attrs)

        # Scaling factors for FP8 weights
        ws_scale = Parameter(torch.ones(num_total_experts,
                                        dtype=torch.float32),
                             requires_grad=False)
        w2s_scale = Parameter(torch.ones(num_total_experts,
                                         dtype=torch.float32),
                              requires_grad=False)
        layer.register_parameter("ws_scale", ws_scale)
        layer.register_parameter("w2s_scale", w2s_scale)

    def process_weights_after_loading(self, layer: Module):
        ws = torch.empty_like(layer.ws.data, dtype=torch.float8_e4m3fn)
        w2s = torch.empty_like(layer.w2s.data, dtype=torch.float8_e4m3fn)
        for expert in range(layer.num_total_experts):
            ws[expert, :, :], layer.ws_scale[expert] = ops.scaled_fp8_quant(
                layer.ws.data[expert, :, :])
            w2s[expert, :, :], layer.w2s_scale[expert] = ops.scaled_fp8_quant(
                layer.w2s.data[expert, :, :])
        layer.ws = Parameter(ws, requires_grad=False)
        layer.w2s = Parameter(w2s, requires_grad=False)

    def apply(self, layer: torch.nn.Module, hidden_states: torch.Tensor,
              router_logits: torch.Tensor) -> torch.Tensor:
        return fused_moe(hidden_states,
                         layer.ws,
                         layer.w2s,
                         router_logits,
                         layer.top_k,
                         renormalize=True,
                         inplace=True,
                         use_fp8=True,
                         w1_scale=layer.ws_scale,
                         w2_scale=layer.w2s_scale)
