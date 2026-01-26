# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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


# --8<-- [start:modular_fused_moe]
@CustomOp.register("modular_fused_moe")
class FusedMoEModularMethod(FusedMoEMethodBase, CustomOp):
    # --8<-- [end:modular_fused_moe]

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
        assert not self.old_quant_method.is_monolithic
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
                moe_parallel_config=moe_layer.moe_parallel_config,
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

    @property
    def method_name(self) -> str:
        return self.old_quant_method.method_name

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
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=self.allow_inplace,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=None if self.disable_expert_map else layer.expert_map,
        )
