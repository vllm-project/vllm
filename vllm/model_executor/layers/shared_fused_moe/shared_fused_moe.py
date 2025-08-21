# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


# TODO(bnell): Add shared + fused combo function? e.g. +
class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def __init__(self, shared_experts: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts

    @property
    def shared_experts(self) -> Optional[torch.nn.Module]:
        return self._shared_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out, fused_out = super().forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, fused_out
