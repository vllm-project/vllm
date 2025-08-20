# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

#@CustomOp.register("shared_fused_moe") ???


# TODO: add shared + fused combo function?
class SharedFusedMoE(FusedMoE):

    def __init__(self, shared_experts: torch.nn.Module, **kwargs):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts

    @property
    def shared_experts(self) -> Optional[torch.nn.Module]:
        return self._shared_experts

    def forward(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out, fused_out = super().forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # If shared_out is None, it means the shared experts have not
        #if shared_out is None:
        #    shared_out = self.shared_experts_fn(hidden_states)

        return shared_out, fused_out
