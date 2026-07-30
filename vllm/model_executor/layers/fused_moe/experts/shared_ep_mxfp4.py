# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native GB200 MXFP4 experts over SharedEP VMM objects."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.shared_ep import (
    SharedEPPrepareAndFinalize,
)


class SharedEPMxfp4Experts(TrtLlmMxfp4ExpertsModular):
    """Run local MXFP4 experts, then directly peer-store owner contributions."""

    def __init__(
        self,
        *,
        prepare_finalize: SharedEPPrepareAndFinalize,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if prepare_finalize.memory.quant_dtype != "mxfp8":
            raise ValueError("SharedEPMxfp4Experts requires MXFP8 SharedEP state")
        self.shared_ep = prepare_finalize

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        super().apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        self.shared_ep.memory.publish_partial_output(output)
