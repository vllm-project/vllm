# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Callable, Optional

import torch
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
from torch.nn.parameter import Parameter
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig
from triton_kernels.numerics import InFlexData
from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
from triton_kernels.tensor_details import layout

from vllm import envs
from vllm.model_executor.layers.fused_moe import (
    FusedMoE, FusedMoEActivationFormat, FusedMoEConfig, FusedMoEMethodBase,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils import round_up


def swizzle_mxfp4(quant_tensor, scale, num_warps):
    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1)
    scale_layout, scale_layout_opts = (
        layout.make_default_matmul_mxfp4_w_scale_layout(mx_axis=1,
                                                        num_warps=num_warps))
    if current_platform.is_cuda() and \
        torch.cuda.get_device_capability()[0] == 10:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(wrap_torch_tensor(quant_tensor, dtype=FP4),
                                  value_layout, **value_layout_opts)
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout,
                           **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


class Mxfp4Config(QuantizationConfig):

    def __init__(self, ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

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
            return Mxfp4MoEMethod(layer.moe_config)
        elif isinstance(layer, Attention):
            raise NotImplementedError(
                "Mxfp4 attention layer is not implemented")
        return None


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.topk_indices_dtype = None
        self.moe = moe

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        # FIXME (zyongye): ship after torch and safetensors support mxfp4
        # is_torch_mxfp4_available = (
        #     hasattr(torch, "float4_e2m1fn_x2") and
        #     hasattr(torch, "float8_e8m0fnu"))
        # if is_torch_mxfp4_available:
        #     weight_dtype = torch.float4_e2m1fn_x2
        #     scale_dtype = torch.float8_e8m0fnu

        mxfp4_block = 32

        # pad the intermediate size to be a multiple of 2 * mxfp4_block
        # for to hold non-uniform sharded tensor as well as swizzling
        intermediate_size_per_partition_after_pad = round_up(
            intermediate_size_per_partition, 64)

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition_after_pad,
            hidden_size // 2,
            dtype=weight_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition_after_pad,
            hidden_size // mxfp4_block,
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
            hidden_size,
            intermediate_size_per_partition_after_pad // 2,
            dtype=weight_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size_per_partition_after_pad // mxfp4_block,
            dtype=scale_dtype),
                                             requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                 hidden_size,
                                                 dtype=torch.bfloat16),
                                     requires_grad=False)
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):

        w13_bias = layer.w13_bias.to(torch.float32)
        w2_bias = layer.w2_bias.to(torch.float32)

        layer.w13_bias = Parameter(w13_bias, requires_grad=False)
        layer.w2_bias = Parameter(w2_bias, requires_grad=False)

        # FIXME warp need to be adjusted based on batch size
        # only apply to  batched mode
        if self.moe.use_ep:
            num_warps = 4 if envs.VLLM_MOE_DP_CHUNK_SIZE <= 512 else 8
        else:
            num_warps = 8

        w13_weight, w13_flex, w13_scale = swizzle_mxfp4(
            layer.w13_weight, layer.w13_weight_scale, num_warps)
        w2_weight, w2_flex, w2_scale = swizzle_mxfp4(layer.w2_weight,
                                                     layer.w2_weight_scale,
                                                     num_warps)

        self.w13_precision_config = PrecisionConfig(
            weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex))
        self.w2_precision_config = PrecisionConfig(
            weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex))

        self.w13_weight_triton_tensor = w13_weight
        self.w2_weight_triton_tensor = w2_weight

        # need to delete the original weights to save memory on single GPU
        del layer.w13_weight
        del layer.w2_weight
        layer.w13_weight = None
        layer.w2_weight = None
        torch.cuda.empty_cache()

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        moe: FusedMoEConfig,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # this is happen after the weights are loaded
        # so we can safely use precision config
        from vllm.model_executor.layers.fused_moe.triton_kernels_moe import (
            BatchedOAITritonExperts)
        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            max_num_tokens_per_rank = (
                prepare_finalize.max_num_tokens_per_rank())
            return BatchedOAITritonExperts(
                None,
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                w1_precision=self.w13_precision_config,
                w2_precision=self.w2_precision_config,
            )
        else:
            raise NotImplementedError(
                "Mxfp4 does not support non-batched experts format for EP")

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

        if self.moe.use_ep:
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                indices_type=self.topk_indices_dtype,
                enable_eplb=enable_eplb,
                expert_map=expert_map,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

            return self.fused_experts(
                x,
                self.w13_weight_triton_tensor,
                self.w2_weight_triton_tensor,
                topk_weights,
                topk_ids,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
            )
        else:
            return triton_kernel_moe_forward(
                hidden_states=x,
                w1=self.w13_weight_triton_tensor,
                w2=self.w2_weight_triton_tensor,
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
