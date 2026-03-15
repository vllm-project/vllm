# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Configuration for I64 token-routed expert layers.
"""

from dataclasses import dataclass


@dataclass
class I64MoEConfig:
    """Configuration for an I64 token-routed MoE layer."""

    hidden_size: int
    intermediate_size: int
    num_experts: int
    vocab_size: int
    ep_size: int = 1
    ep_rank: int = 0

    @property
    def expert_intermediate_size(self) -> int:
        return self.intermediate_size // self.num_experts
