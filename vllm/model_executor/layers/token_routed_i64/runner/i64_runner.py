# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
I64 Expert Runner — dispatches tokens to the fused I64 kernel.

Analogous to fused_moe/runner/default_moe_runner.py but for
deterministic token-routed I64 experts.
"""

import torch

from vllm.model_executor.layers.token_routed_i64.fused_i64_moe import (
    fused_i64_experts,
)


class I64ExpertRunner:
    """
    Runs I64 token-routed expert forward pass.

    Wraps fused_i64_experts with the same interface pattern
    as vLLM's DefaultMoERunner.
    """

    def __init__(
        self,
        num_experts: int,
        intermediate_per_tp: int,
    ):
        self.num_experts = num_experts
        self.intermediate_per_tp = intermediate_per_tp

    def forward(
        self,
        x: torch.Tensor,
        gate_up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run fused I64 expert forward with pre-computed expert_ids.

        For direct use outside CUDA graphs (e.g. EP path).
        """
        return fused_i64_experts(
            x,
            gate_up_proj,
            down_proj,
            expert_ids,
            self.num_experts,
            self.intermediate_per_tp,
        )
