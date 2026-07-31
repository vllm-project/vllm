# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi MoE variants implemented with ordinary PyTorch routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from .layers import KimiMLP, RMSNorm


class Router(nn.Module):
    def __init__(self, config: KimiLinearConfig) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.num_experts_per_token is not None
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.scoring = config.moe_router_activation_func
        self.renormalize = config.moe_renormalize
        self.scaling_factor = config.routed_scaling_factor
        self.num_groups = config.num_expert_group
        self.topk_groups = config.topk_group
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size),
        )
        set_weight_attrs(self.weight, {"weight_loader": default_weight_loader})
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states.float(), self.weight.float())
        if self.scoring == "sigmoid":
            scores = logits.sigmoid()
        elif self.scoring == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported router scoring function {self.scoring!r}")

        choice_scores = scores + self.e_score_correction_bias.float()
        if self.num_groups > self.topk_groups:
            if self.num_experts % self.num_groups:
                raise ValueError("num_experts must be divisible by num_expert_group")
            grouped = choice_scores.view(*choice_scores.shape[:-1], self.num_groups, -1)
            group_scores = grouped.topk(min(2, grouped.shape[-1]), dim=-1).values.sum(
                dim=-1
            )
            selected_groups = group_scores.topk(self.topk_groups, dim=-1).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(-1, selected_groups, True)
            expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).flatten(-2)
            choice_scores = choice_scores.masked_fill(~expert_mask, -torch.inf)

        expert_ids = choice_scores.topk(self.top_k, dim=-1).indices
        expert_weights = scores.gather(-1, expert_ids)
        if self.top_k > 1 and self.renormalize:
            expert_weights = expert_weights / expert_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-20)
        return expert_ids, expert_weights * self.scaling_factor


class Expert(KimiMLP):
    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        expert_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        assert config.moe_intermediate_size is not None
        super().__init__(
            hidden_size,
            config.moe_intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.{expert_id}",
            reduce_results=False,
            situ_beta=config.activation_situ_beta,
            situ_linear_beta=config.activation_situ_linear_beta,
        )
        self.w1 = self.gate_proj
        self.w3 = self.up_proj
        self.w2 = self.down_proj
        del self.gate_proj
        del self.up_proj
        del self.down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, _ = self.w1(hidden_states)
        up, _ = self.w3(hidden_states)
        hidden_states = self.activate(gate, up)
        hidden_states, _ = self.w2(hidden_states)
        return hidden_states


class KimiMoE(nn.Module):
    """Standard Kimi MoE, optionally using K3's latent expert space."""

    def __init__(
        self,
        config: KimiLinearConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.num_experts is not None
        assert config.moe_intermediate_size is not None

        self.tp_size = get_tensor_model_parallel_world_size()
        self.gate = Router(config)
        expert_hidden_size = (
            config.routed_expert_hidden_size
            if config.routed_expert_hidden_size is not None
            else config.hidden_size
        )
        self.routed_expert_down_proj = (
            ReplicatedLinear(
                config.hidden_size,
                expert_hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_down_proj",
            )
            if config.routed_expert_hidden_size is not None
            else None
        )
        self.experts = nn.ModuleList(
            [
                Expert(
                    config,
                    expert_hidden_size,
                    expert_id,
                    quant_config,
                    f"{prefix}.experts",
                )
                for expert_id in range(config.num_experts)
            ]
        )
        self.routed_expert_norm = (
            RMSNorm(expert_hidden_size, config.rms_norm_eps)
            if self.routed_expert_down_proj is not None and config.latent_moe_use_norm
            else None
        )
        self.routed_expert_up_proj = (
            ReplicatedLinear(
                expert_hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.routed_expert_up_proj",
            )
            if self.routed_expert_down_proj is not None
            else None
        )
        self.shared_experts = (
            KimiMLP(
                config.hidden_size,
                config.moe_intermediate_size * config.num_shared_experts,
                config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                situ_beta=config.activation_situ_beta,
                situ_linear_beta=config.activation_situ_linear_beta,
            )
            if config.num_shared_experts > 0
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        expert_ids, expert_weights = self.gate(hidden_states)
        if self.routed_expert_down_proj is None:
            routed_inputs = hidden_states
        else:
            routed_inputs, _ = self.routed_expert_down_proj(hidden_states)
        routed_output = torch.zeros_like(routed_inputs)

        for expert_id, expert in enumerate(self.experts):
            token_indices, slots = torch.where(expert_ids == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(routed_inputs[token_indices])
            weights = expert_weights[token_indices, slots, None].to(expert_output.dtype)
            routed_output.index_add_(0, token_indices, expert_output * weights)

        if self.tp_size > 1:
            routed_output = tensor_model_parallel_all_reduce(routed_output)
        if self.routed_expert_norm is not None:
            routed_output = self.routed_expert_norm(routed_output)
        if self.routed_expert_up_proj is None:
            output = routed_output
        else:
            output, _ = self.routed_expert_up_proj(routed_output)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output
