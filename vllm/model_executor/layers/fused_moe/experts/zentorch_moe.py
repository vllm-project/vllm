# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zentorch fused MoE experts for AMD Zen CPUs."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import select_experts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static,
    kInt4Static32,
)


class ZentorchExpertsInt4(mk.FusedMoEExpertsMonolithic):
    """DA8W4 (W4A8) group-quantized monolithic MoE experts.

    Runs the same int4 checkpoint as the W4A16 CPU experts, repacked to s4, but
    quantizes activations to int8 per token so the expert GEMMs run as int8 x
    int8 on ZenDNN.
    """

    # swiglu_oai_mul reads gate/up interleaved, not in the half-split order the
    # weight loader leaves behind.
    requires_interleaved_w13 = True

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return has_zentorch_op(["zentorch_fused_moe", "zentorch_woq_repack_weight"])

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return supported, reason
        if moe_config.in_dtype != torch.bfloat16:
            return False, "kernel requires bfloat16 activations"
        # The grouped GEMM needs tokens spread over at least two experts, which
        # top_k=1 cannot satisfy once decode drops to a single token.
        if moe_config.experts_per_token < 2:
            return False, "kernel requires experts_per_token >= 2"
        return True, None

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SWIGLUOAI,
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in [
            (kInt4Static, None),
            (kInt4Static32, None),
        ]

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "ZentorchExpertsInt4 does not support "
                "apply_router_weight_on_input=True."
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method
            in (
                RoutingMethodType.Renormalize,
                RoutingMethodType.RenormalizeNaive,
            ),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )

        output = torch.empty_like(hidden_states)
        torch.ops.zentorch.zentorch_fused_moe(
            output,
            hidden_states,
            w1,
            w2,
            self.w1_bias,
            self.w2_bias,
            topk_weights,
            topk_ids,
            False,  # skip_weighted
            str(activation.value).lower(),
            self.w1_scale,
            self.w2_scale,
        )
        return output


__all__ = ["ZentorchExpertsInt4"]
