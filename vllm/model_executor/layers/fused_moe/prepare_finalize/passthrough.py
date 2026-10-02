# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare/finalize stages that do no expert-parallel communication.

Selected by ``all2all_backend="passthrough"``: the experts implementation
dispatches, combines and reduces itself, so nothing is left for these stages.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)


class PassThroughPrepareAndFinalize(MoEPrepareAndFinalizeNoDPEPModular):
    """Hands activations and routing straight to experts that own their EP
    communication."""

    def output_is_reduced(self) -> bool:
        return True

    def supports_deferred_moe_finalize(self) -> bool:
        # The experts apply the top-k weights and combine internally; there
        # is no finalize left for a consumer to take over.
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        return a1, None, None, None, None
