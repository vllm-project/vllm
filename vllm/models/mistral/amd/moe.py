# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from torch import nn
from transformers import PreTrainedConfig

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
)
from vllm.model_executor.layers.fused_moe.utils import (
    resolve_layer_fused_shared_expert,
)
from vllm.models.mistral.amd.mlp import MistralMLP


class MistralMoE(nn.Module):
    """Softmax top-k routed experts plus always-on shared experts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PreTrainedConfig,
        prefix: str,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        assert config.scoring_func == "softmax" and config.topk_method is None
        assert (config.n_group or 1) == 1, "Grouped top-k is not supported"
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        # FP32 router logits, as in training; the low-latency ll_bf16 GEMM
        # produces them directly at decode batch sizes.
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # On ROCm, AITER can fuse the always-on shared expert into the routed
        # MoE call (one kernel instead of a separate shared MLP), enabled with
        # VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1. The routing is gate-less
        # softmax, so this requires the gate-less shared-expert router fix.
        self.is_fused_shared_expert_enabled = False
        if config.n_shared_experts:
            self.is_fused_shared_expert_enabled = resolve_layer_fused_shared_expert(
                quant_config, prefix
            )

        if not config.n_shared_experts or self.is_fused_shared_expert_enabled:
            # The fused path builds the shared expert inside the MoE module.
            self.shared_experts = None
        else:
            self.shared_experts = MistralMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func="softmax",
            routed_scaling_factor=config.routed_scaling_factor,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            gate=self.gate,
            router_logits_dtype=torch.float32,
            reduce_results=reduce_results,
            shared_experts=self.shared_experts,
            n_shared_experts=config.n_shared_experts
            if self.is_fused_shared_expert_enabled
            else None,
            fuse_shared_experts=self.is_fused_shared_expert_enabled,
            shared_expert_prefix=f"{prefix}.shared_experts",
        )
        # A deferred reduction must not be applied by the MoE layer itself.
        assert reduce_results or self.experts.moe_config.skip_final_all_reduce

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden_states=hidden_states, router_logits=hidden_states)
