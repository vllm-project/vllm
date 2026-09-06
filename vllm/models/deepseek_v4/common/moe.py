# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import MutableSequence, Sequence
from typing import Any

import torch
import torch.nn as nn

from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.model_executor.models.utils import PPMissingLayer


def _supports_eplb_experts(experts: object) -> bool:
    return all(
        hasattr(experts, name)
        for name in ("get_expert_weights", "set_eplb_state", "update_expert_map")
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    """Register DeepSeek V4 FusedMoE layers with the EPLB control plane."""

    decoder_layer_cls: type[nn.Module]
    moe_layer_cls: type[nn.Module]
    moe_mlp_layers: list[nn.Module]

    def set_moe_parameters(self) -> None:
        self.expert_weights: MutableSequence[Sequence[torch.Tensor]] = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None

        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, self.decoder_layer_cls):
                continue

            moe: Any = layer.ffn
            if not isinstance(moe, self.moe_layer_cls):
                continue
            if not _supports_eplb_experts(moe.experts):
                continue

            example_moe = moe
            self.moe_mlp_layers.append(moe)
            self.moe_layers.append(moe.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def extract_moe_parameters(self, example_moe: Any | None) -> None:
        if example_moe is None:
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
