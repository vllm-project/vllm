# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_substitution import (
    ConstantExpertSubstitution,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

logger = init_logger(__name__)


class SubstitutedRoutedExperts(RoutedExperts):
    """Compact expert weights with constant contributions outside the backend."""

    _non_expert_parameter_prefixes = ("expert_substitution.",)

    def __init__(
        self, *args: Any, expert_substitution: ConstantExpertSubstitution, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        if self.quant_method.is_monolithic:
            raise NotImplementedError(
                "expert substitution requires a decomposed MoE backend"
            )
        self.expert_substitution = expert_substitution

    def get_expert_mapping(
        self,
        ckpt_gate_proj_name: str | None = None,
        ckpt_down_proj_name: str | None = None,
        ckpt_up_proj_name: str | None = None,
        include_fused: bool = False,
    ) -> list[tuple[str, str, int, str]]:
        if include_fused:
            logger.warning_once(
                "Fused expert checkpoint tensors are not supported with "
                "expert substitution"
            )
        return self.expert_substitution.make_expert_params_mapping(
            moe_prefix=self.layer_name,
            ckpt_gate_proj_name=ckpt_gate_proj_name or self.ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name or self.ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name or self.ckpt_up_proj_name,
            routed_experts_prefix="",
            base_layer=self.lora_base_layer_prefix,
        )

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids, contribution = (
            self.expert_substitution.transform_routes(x, topk_weights, topk_ids)
        )
        output = super().forward_modular(
            x, topk_weights, topk_ids, shared_experts, shared_experts_input
        )
        kernel = self.quant_method.moe_kernel
        # Unreduced outputs are summed by the runner: contribute the replicated
        # constant only once. Already-reduced outputs need it on every TP rank.
        if self.moe_config.tp_rank == 0 or (
            kernel is not None and kernel.output_is_reduced()
        ):
            output.add_(contribution[..., : output.shape[-1]])
        return output
