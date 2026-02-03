# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch.nn import Module
from vllm_xpu_kernels.fused_moe_interface import xpu_fused_moe

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform


class XPUFp8LinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
        self.use_w8a8_gemm = envs.VLLM_XPU_USE_W8A8_GEMM

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale

        if self.use_w8a8_gemm:
            out_dtype = self.out_dtype if self.out_dtype is not None else x.dtype
            output_shape = [*x.shape[:-1], weight.shape[1]]
            input_2d = x.view(-1, x.shape[-1])
            if input_2d.dtype != current_platform.fp8_dtype():
                qinput, x_scale = self.fp8_linear.quant_fp8(
                    input_2d,
                    layer.input_scale,
                )
            else:
                qinput, x_scale = input_2d, layer.input_scale
            output = torch.ops._xpu_C.fp8_gemm(
                qinput, weight, out_dtype, x_scale, weight_scale, bias
            )
            return torch.narrow(output, 0, 0, qinput.shape[0]).view(*output_shape)
        else:
            return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)


class XPUFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # INPUT_SCALES
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        if not self.quant_config.is_checkpoint_fp8_serialized:
            fp8_dtype = current_platform.fp8_dtype()
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.local_num_experts,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )
            for expert in range(layer.local_num_experts):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        M, _ = x.size()
        routing_weights = torch.empty(
            M, layer.top_k, dtype=torch.float32, device=x.device
        )
        selected_experts = torch.empty(
            M, layer.top_k, dtype=torch.int32, device=x.device
        )
        token_expert_indices = torch.empty(
            M, layer.top_k, dtype=torch.int32, device=x.device
        )
        if layer.use_grouped_topk:
            routing_weights, selected_experts = torch.ops._moe_C.fused_grouped_topk(
                x,
                router_logits,
                layer.top_k,
                layer.renormalize,
                n_expert_group=layer.num_expert_group,
                n_topk_group=layer.topk_group,
                scoring_func=layer.scoring_func,
                routed_scaling_factor=layer.routed_scaling_factor,
                bias=layer.e_score_correction_bias,
            )
        else:
            torch.ops._moe_C.topk_softmax(
                routing_weights,
                selected_experts,
                token_expert_indices,
                router_logits,
                layer.renormalize,
            )

        return xpu_fused_moe(
            hidden_states=x,
            w13=layer.w13_weight,
            w13_scales=layer.w13_weight_scale,
            w13_bias=layer.w13_bias if self.moe.has_bias else None,
            w2=layer.w2_weight,
            w2_scales=layer.w2_weight_scale,
            w2_bias=layer.w2_bias if self.moe.has_bias else None,
            topk_weights=routing_weights,
            topk_ids=selected_experts,
            n_experts_per_token=layer.top_k,
            activation=layer.activation,
            num_experts=layer.global_num_experts
            // self.moe.moe_parallel_config.ep_size,
            ep_rank=self.moe.moe_parallel_config.ep_rank,
            ep_size=self.moe.moe_parallel_config.ep_size,
            is_fp8=True,
        )
