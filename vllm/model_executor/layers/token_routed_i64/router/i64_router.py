# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
I64 Deterministic Router — expert_id = token_id % num_experts.

No learned gate, no softmax, no top-k.
Mu-guided routing: INL Dynamics mu vector can bias expert selection.

CUDA graph safe: all operations are pure tensor ops, no CPU sync.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class I64Router(nn.Module):
    """
    Deterministic I64 token router.

    Routes tokens to experts based on: expert_id = token_id % num_experts
    Optional mu-guided bias from INL Dynamics.
    """

    _BASE_ROUTING_SCALE = 10.0

    def __init__(self, num_experts: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.vocab_size = vocab_size

        # Mu-guided routing (replicated — small tensor)
        self.mu_router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None,
        mu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute expert IDs for each token (global expert IDs).

        Args:
            x: (num_tokens, hidden_size)
            token_ids: (num_tokens,) — token IDs for modulo routing
            mu: (num_tokens, hidden_size) — optional mu from INL Dynamics

        Returns:
            expert_ids: (num_tokens,) long
        """
        num_tokens = x.shape[0]

        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=x.device)

        token_ids_clamped = token_ids.clamp(0, self.vocab_size - 1)
        base_expert_ids = (token_ids_clamped % self.num_experts).long()

        if mu is not None:
            mu_logits = self.mu_router(mu)
            base_one_hot = F.one_hot(base_expert_ids, self.num_experts).float()
            combined_logits = base_one_hot * self._BASE_ROUTING_SCALE + mu_logits
            return combined_logits.argmax(dim=-1)

        return base_expert_ids
