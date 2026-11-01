# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared EPLB MoE registration helpers for DeepSeek V4 family models."""

from collections.abc import Iterable
from typing import Any

import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.interfaces import MixtureOfExperts


def dspark_draft_supports_eplb(draft_model_config: ModelConfig) -> bool:
    """Return whether a DSpark draft can share EPLB state with the target.

    Only DeepSeek-V4 DSpark drafts reuse the target expert layout. V4.1 drafts
    use a smaller routed-expert count and cannot share EPLB state.
    """
    return getattr(draft_model_config.hf_config, "model_type", None) == "deepseek_v4"


def collect_moe_layers(
    moe_model: Any,
    layers: Iterable[nn.Module],
    config: object,
    *,
    decoder_layer_type: type[nn.Module],
    moe_type: type[nn.Module],
) -> None:
    """Populate ``MixtureOfExperts`` fields from decoder layers for EPLB.

    Works for both target backbones and DSpark draft stacks. Draft models pass
    their shorter ``model.layers`` list; pipeline-parallel placeholder layers
    are skipped via ``decoder_layer_type``.
    """
    moe_model.num_expert_groups = getattr(config, "n_group", 1)
    moe_model.moe_layers = []
    moe_model.moe_mlp_layers = []
    example_moe = None
    for layer in layers:
        if not isinstance(layer, decoder_layer_type):
            continue
        if not isinstance(layer.ffn, moe_type):
            continue
        example_moe = layer.ffn
        moe_model.moe_mlp_layers.append(layer.ffn)
        moe_model.moe_layers.append(layer.ffn.experts)

    moe_model.num_moe_layers = len(moe_model.moe_layers)
    moe_model.extract_moe_parameters(example_moe)


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[Any]

    def extract_moe_parameters(self, example_moe: Any) -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()
