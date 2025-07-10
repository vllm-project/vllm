# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.fused_moe import FusedMoE, FusedMoEMethodBase
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.layers.utils import shuffle_weight
from vllm.model_executor.utils import set_weight_attrs

from triton_kernels.matmul_ogs import MicroscalingCtx, PrecisionConfig, FlexCtx
from triton_kernels.numerics import InFlexData
from triton_kernels.numerics_details.mxfp import (SwizzlingType, swizzle_mxfp4_value_hopper, 
                                                  perm_tensor_from_contig, perm_tuple_from_contig,
                                                  swizzle_mx_scale_bw, swizzle_mxfp4_scale_hopper)

def swizzle_mxfp4(quant_tensor, scale):
    swizzle_value = None
    swizzle_scale = None
    axis = 1
    swizzle_axis = 2
    # Swizzling
    if swizzle_value == SwizzlingType.HOPPER:
        quant_tensor = swizzle_mxfp4_value_hopper(quant_tensor, op_idx=0, mma_version=3)
    assert quant_tensor.is_contiguous()
    quant_tensor = perm_tensor_from_contig(quant_tensor, axis, swizzle_axis)

    orig_scale_shape = scale.shape
    if swizzle_scale == SwizzlingType.BLACKWELL:
        scale = swizzle_mx_scale_bw(scale, allow_pad=True)
    elif swizzle_scale == SwizzlingType.HOPPER:
        scale = swizzle_mxfp4_scale_hopper(scale, num_warps=8)
    assert scale.is_contiguous()
    scale = perm_tensor_from_contig(scale, axis, swizzle_axis)
    return quant_tensor, InFlexData(), MicroscalingCtx(weight_scale=scale, swizzle_scale=swizzle_scale,
                                                swizzle_value=swizzle_value,
                                                actual_weight_scale_shape=perm_tuple_from_contig(orig_scale_shape, axis, swizzle_axis))

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
        # TODO: we only register parameter here
        # and do not pre-allocate tensor
        # since they need to be transformed when loading
        # allocating will cause pytorch OOM
        # these dummy weight will be
        # replace in func FusedMoE::_load_weights_oai_mlp

        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # is_torch_mxfp4_available = hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch, "float8_e8m0fnu")
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2,
            dtype=weight_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // mxfp4_block,
            dtype=scale_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_bias = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=torch.bfloat16
        ), 
                                        requires_grad=False)
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // 2,
            dtype=weight_dtype),
                                        requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // mxfp4_block,
            dtype=scale_dtype),
                                        requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            dtype=torch.bfloat16
        ), 
                                        requires_grad=False)
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):

        w13_bias = layer.w13_bias.to(torch.float32)
        # w13_bias = F.pad(w13_bias, (0, layer.w13_right_pad, 0, 0),
        #                  mode="constant",
        #                  value=0)

        w2_bias = layer.w2_bias.to(torch.float32)
        # w2_bias = F.pad(w2_bias, (0, layer.w2_right_pad, 0, 0),
        #                 mode="constant",
        #                 value=0)

        layer.w13_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(w2_bias, requires_grad=False)

        w13_weight, w13_flex, w13_mx = swizzle_mxfp4(layer.w13_weight, layer.w13_weight_scale)
        w2_weight, w2_flex, w2_mx = swizzle_mxfp4(layer.w2_weight, layer.w2_weight_scale)

        self.w13_precision_config = PrecisionConfig(mx_ctx=w13_mx, flex_ctx=FlexCtx(rhs_data=w13_flex))
        self.w2_precision_config = PrecisionConfig(mx_ctx=w2_mx, flex_ctx=FlexCtx(rhs_data=w2_flex))

        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)



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
        # aviod import error when triton_kernel is not installed
        from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
            triton_kernel_moe_forward)
        
        if enable_eplb:
            raise NotImplementedError("EPLB is not supported for mxfp4")

        return triton_kernel_moe_forward(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
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
