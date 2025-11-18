# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
    FusedMoEPrepareAndFinalize,
)

logger = init_logger(__name__)


@CustomOp.register("modular_fused_moe")
class FusedMoEModularMethod(FusedMoEMethodBase, CustomOp):
    def __init__(
        self, old_quant_method: FusedMoEMethodBase, experts: FusedMoEModularKernel
    ):
        super().__init__(old_quant_method.moe)
        self.moe_quant_config = old_quant_method.moe_quant_config
        self.fused_experts = experts
        self.disable_expert_map = getattr(
            old_quant_method,
            "disable_expert_map",
            not self.fused_experts.supports_expert_map(),
        )
        self.old_quant_method = old_quant_method
        logger.debug("Swapping out %s", self.old_quant_method.__class__.__name__)

    @staticmethod
    def make(
        moe_layer: torch.nn.Module,
        old_quant_method: FusedMoEMethodBase,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        shared_experts: torch.nn.Module | None,
    ) -> "FusedMoEModularMethod":
        return FusedMoEModularMethod(
            old_quant_method,
            FusedMoEModularKernel(
                prepare_finalize,
                old_quant_method.select_gemm_impl(prepare_finalize, moe_layer),
                shared_experts,
            ),
        )

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return self.fused_experts.prepare_finalize.topk_indices_dtype()

    @property
    def supports_eplb(self) -> bool:
        return self.old_quant_method.supports_eplb

    @property
    def allow_inplace(self) -> bool:
        return self.old_quant_method.allow_inplace

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return self.moe_quant_config

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Is getattr needed?
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        if enable_eplb:
            if self.supports_eplb:
                assert expert_load_view is not None
                assert logical_to_physical_map is not None
                assert logical_replica_count is not None
            else:
                raise NotImplementedError(
                    "EPLB is not supported for "
                    f"{self.old_quant_method.__class__.__name__}."
                )

        topk_weights, topk_ids, zero_expert_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )

        result = self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=self.allow_inplace,
            activation=activation,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=None if self.disable_expert_map else expert_map,
        )

        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result
