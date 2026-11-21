# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash latent experts with HYV4's routing, shared experts and weight format."""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.block_fp8_mega_moe import (
    BlockFp8MegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import MegaMoEActivation
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_grouped_topk,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.hy_v4.nvidia.moe import HYV4FeedForward, HYV4MoEFused
from vllm.models.hy_v4_flash.configs import HYV4FlashConfig


class HYV4FlashFeedForward(HYV4FeedForward):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.post_fc2_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x)
        return self.post_fc2_norm(result) if self.post_fc2_norm is not None else result


class HYV4FlashMoE(HYV4MoEFused):
    def __init__(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        vllm_config = get_current_vllm_config()
        use_mega_moe = vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        latent = config.routed_expert_hidden_size
        down: nn.Module | None = None
        up: nn.Module | None = None
        if latent is not None:
            down = ReplicatedLinear(
                config.hidden_size,
                latent,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            up = ReplicatedLinear(
                latent,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
        if use_mega_moe:
            nn.Module.__init__(self)
            self._init_mega_moe(config, quant_config, prefix)
        else:
            super().__init__(
                config,
                quant_config=quant_config,
                prefix=prefix,
                routed_hidden_size=latent,
                routed_input_transform=down,
                routed_output_transform=up,
            )
        self.use_mega_moe = use_mega_moe
        self.routed_expert_down_proj = down
        self.routed_expert_up_proj = up
        self.combine_output_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.sandwich_norm
            else None
        )

    def _init_mega_moe(
        self,
        config: HYV4FlashConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        vllm_config = get_current_vllm_config()
        self.tp_size = 1
        self.use_sequence_parallel = False
        self.enable_eplb = False
        self.n_routed_experts = self.n_logical_experts = config.num_experts
        self.n_physical_experts = self.n_routed_experts
        self.n_redundant_experts = 0
        ep_group = get_ep_group()
        self.ep_size, self.ep_rank = ep_group.world_size, ep_group.rank_in_group
        self.ep_group = ep_group.device_group
        self.prefix = prefix
        if self.n_routed_experts % self.ep_size:
            raise ValueError("HYV4 Flash MegaMoE requires experts evenly divided by EP")
        self.n_local_physical_experts = self.n_routed_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )
        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )
        self.shared_experts = (
            HYV4FeedForward(
                config.hidden_size,
                config.expert_hidden_dim * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
            )
            if config.num_shared_experts
            else None
        )
        self.route_norm = config.route_norm
        self.routed_scaling_factor = config.router_scaling_factor
        self.experts = BlockFp8MegaMoEExperts(
            vllm_config,
            weight_block_size=(128, 128),
            num_experts=self.n_routed_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=self.physical_expert_start,
            top_k=config.num_experts_per_tok,
            hidden_size=config.routed_expert_hidden_size or config.hidden_size,
            intermediate_size=config.expert_hidden_dim,
            activation=MegaMoEActivation("swiglu", clamp=config.swiglu_limit or None),
            prefix=f"{prefix}.experts",
        )

    def process_weights_after_loading(self) -> None:
        if self.use_mega_moe:
            self.experts.finalize_block_fp8_weights()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_mega_moe:
            orig_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, orig_shape[-1])
            router_logits, _ = self.gate(hidden_states)
            topk_weights, topk_ids = fused_grouped_topk(
                hidden_states,
                router_logits,
                self.experts.top_k,
                self.route_norm,
                self.expert_bias,
                num_expert_group=1,
                topk_group=1,
                scoring_func="sigmoid",
                routed_scaling_factor=self.routed_scaling_factor,
            )
            routed = hidden_states
            if self.routed_expert_down_proj is not None:
                routed, _ = self.routed_expert_down_proj(routed)
            result = self.experts(routed, topk_weights, topk_ids)
            if self.routed_expert_up_proj is not None:
                result, _ = self.routed_expert_up_proj(result)
            if self.shared_experts is not None:
                result = result + self.shared_experts(hidden_states)
            result = result.view(orig_shape)
        else:
            result = super().forward(hidden_states)
        return (
            self.combine_output_norm(result)
            if self.combine_output_norm is not None
            else result
        )
