# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drex pointer readout.

Drex scores a question's options without generating text: the backbone hidden
state at ``<decide>`` is the query, the hidden state at each ``</opt>`` is a
key, and the score is their scaled dot product projected through the trained
head. This module is torch-only so the math can be tested without the engine.
"""

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    """Two biased projections whose scaled dot product scores each option."""

    def __init__(self, hidden_size: int, head_dim: int = 256) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(hidden_size, head_dim)
        self.k = nn.Linear(hidden_size, head_dim)
        self.scale = head_dim**-0.5
        # Fitted at inference only; the reference leaves training at T=1.
        self.temperature = 1.0

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """Score options against the decide state.

        Args:
            h_decide: Hidden state at ``<decide>``, shape `[hidden]`.
            h_opts: Hidden states at each ``</opt>``, shape `[K, hidden]`.

        Returns:
            One score per option, shape `[K]`.

        """
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


def read_pointer_scores(
    hidden_states: torch.Tensor,
    opt_positions: list[int] | None,
    decide_position: int | None,
    head: PointerHead,
) -> torch.Tensor:
    """Score one packed row's options from its marker positions.

    Args:
        hidden_states: Row hidden states, shape `[seq, hidden]`.
        opt_positions: Row indices of the ``</opt>`` tokens, one per option.
        decide_position: Row index of the ``<decide>`` token.
        head: The pointer head.

    Returns:
        One score per option, shape `[K]`. Empty when a marker is missing or
        out of range, so a malformed row (such as a profiling dummy) cannot
        fail the engine core.

    """
    empty = hidden_states.new_empty(0, dtype=torch.float32)
    length = hidden_states.shape[0]
    if decide_position is None or not 0 <= decide_position < length:
        return empty
    if not opt_positions:
        return empty
    if any(not 0 <= position < length for position in opt_positions):
        return empty
    positions = torch.tensor(opt_positions, device=hidden_states.device)
    return head(hidden_states[decide_position], hidden_states[positions])
