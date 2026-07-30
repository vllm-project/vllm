# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native GB200 NVFP4 experts over SharedEP VMM objects."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_moe import (
    FlashInferCuteDSLExperts,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.shared_ep import (
    SharedEPPrepareAndFinalize,
)
from vllm.utils.flashinfer import (
    has_flashinfer_cutedsl_moe_nvfp4_direct_output,
)


class SharedEPNvFP4Experts(FlashInferCuteDSLExperts):
    """Run CuTeDSL NVFP4 experts directly into canonical owner slots."""

    def __init__(
        self,
        *,
        prepare_finalize: SharedEPPrepareAndFinalize,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if prepare_finalize.memory.quant_dtype != "nvfp4":
            raise ValueError("SharedEPNvFP4Experts requires NVFP4 SharedEP state")
        if not has_flashinfer_cutedsl_moe_nvfp4_direct_output():
            raise RuntimeError(
                "Native NVFP4 SharedEP requires FlashInfer CuTeDSL direct-output "
                "support; refusing to materialize rank-partial W2 outputs"
            )
        self.shared_ep = prepare_finalize

    def _extra_flashinfer_kwargs(self) -> dict[str, int | bool]:
        memory = self.shared_ep.memory
        return {
            "direct_output": True,
            "output_rows_per_owner": memory.max_tokens * memory.top_k,
            "output_physical_rows_per_owner": (
                memory.direct_output_physical_rows_per_owner
            ),
        }

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        direct_output = self.shared_ep.memory.direct_output
        if direct_output is None:
            raise RuntimeError("Direct SharedEP output storage is unavailable")
        super().apply(
            output=direct_output,
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
        self.shared_ep.memory.publish_output()
