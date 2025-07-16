# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details.layout import (HopperMXScaleLayout,
                                                  HopperMXValueLayout,
                                                  StridedLayout)

from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.utils import set_weight_attrs


def swizzle_mxfp4(quant_tensor, scale):
    value_layout = StridedLayout
    scale_layout = StridedLayout
    if torch.cuda.get_device_capability()[0] == 9:
        value_layout = HopperMXValueLayout
        scale_layout = HopperMXScaleLayout
    # import pdb; pdb.set_trace()
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(wrap_torch_tensor(quant_tensor, dtype=FP4),
                                  value_layout)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


class Mxfp4Config(QuantizationConfig):

    def __init__(self):
        super().__init__()

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix=prefix,
                                ignored_layers=self.ignored_layers,
                                fused_mapping=self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            raise NotImplementedError("Mxfp4 linear layer is not implemented")
        elif isinstance(layer, FusedMoE):
            return Mxfp4MoEMethod()
        elif isinstance(layer, Attention):
            return NotImplementedError(
                "Mxfp4 attention layer is not implemented")
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # is_torch_mxfp4_available = hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e8m0fnu")
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        smallest_even_divide_number = lambda x, n: (x // n + 1
                                                    ) * n if x % n != 0 else x
        intermediate_size_per_partition_after_pad = smallest_even_divide_number(
            intermediate_size_per_partition, 128)
        hidden_size_after_pad = smallest_even_divide_number(hidden_size, 256)

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition_after_pad,
            hidden_size_after_pad // 2,
            dtype=weight_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition_after_pad,
            hidden_size_after_pad // mxfp4_block,
            dtype=scale_dtype),
                                              requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition_after_pad,
            dtype=torch.bfloat16),
                                      requires_grad=False)
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size_after_pad,
            intermediate_size_per_partition_after_pad // 2,
            dtype=weight_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size_after_pad,
            intermediate_size_per_partition_after_pad // mxfp4_block,
            dtype=scale_dtype),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                 hidden_size_after_pad,
                                                 dtype=torch.bfloat16),
                                     requires_grad=False)
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):

        w13_bias = layer.w13_bias.to(torch.float32)
        w2_bias = layer.w2_bias.to(torch.float32)

        layer.w13_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(w2_bias, requires_grad=False)

        w13_weight, w13_flex, w13_scale = swizzle_mxfp4(
            layer.w13_weight, layer.w13_weight_scale)
        w2_weight, w2_flex, w2_scale = swizzle_mxfp4(layer.w2_weight,
                                                     layer.w2_weight_scale)

        self.w13_precision_config = PrecisionConfig(
            weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex))
        self.w2_precision_config = PrecisionConfig(
            weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex))

        del layer.w13_weight
        del layer.w2_weight

        layer.w13_weight_triton_tensor = w13_weight
        layer.w2_weight_triton_tensor = w2_weight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # avoid import error when triton_kernel is not installed
        from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
            triton_kernel_moe_forward)

        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        return triton_kernel_moe_forward(
            hidden_states=x,
            w1=layer.w13_weight_triton_tensor,
            w2=layer.w2_weight_triton_tensor,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_precision=self.w13_precision_config,
            w2_precision=self.w2_precision_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
