# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static32,
)
from vllm.platforms import current_platform


class TrtLlmMxint4ExpertsMonolithic(mk.FusedMoEExpertsMonolithic):
    """
    FlashInfer TRT-LLM MxInt4 MoE kernel. Monolithic interface
    (fused router + experts).

    Wraps flashinfer_trtllm_mxint4_moe().
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self.topk = moe_config.experts_per_token
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.local_num_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.ep_rank
        self.routing_method = moe_config.routing_method

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (  # noqa: E501
            is_flashinfer_mxint4_moe_available,
        )

        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(100)
            and is_flashinfer_mxint4_moe_available()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kInt4Static32, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # FlashInfer MxInt4 uses a fused SwiGLU activation.
        return activation == MoEActivation.SWIGLUOAI

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        ) and not moe_parallel_config.enable_eplb

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Simulated,
        ]

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        if router_logits_dtype == torch.float32:
            # DeepSeekV3 routing handles float32 logits internally.
            # Simulated routing generates synthetic decisions.
            return routing_method in (
                RoutingMethodType.DeepSeekV3,
                RoutingMethodType.Simulated,
            )
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return False

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The kernel handles quantization internally.
        return True

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
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (  # noqa: E501
            flashinfer_trtllm_mxint4_moe,
        )

        assert self.w1_scale is not None
        assert self.w2_scale is not None
        return flashinfer_trtllm_mxint4_moe(
            x=hidden_states,
            router_logits=router_logits,
            w13_weight_packed=w1,
            w13_weight_scale=self.w1_scale,
            w2_weight_packed=w2,
            w2_weight_scale=self.w2_scale,
            global_num_experts=global_num_experts,
            top_k=self.topk,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            local_num_experts=self.local_num_experts,
            ep_rank=self.ep_rank,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            e_score_correction_bias=e_score_correction_bias,
            routing_method_type=self.routing_method,
        )
