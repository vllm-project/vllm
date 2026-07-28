# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router-lookahead (PILOT) expert prefetch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.model_executor.offloader.hierarchical.manager import ExpertTierManager

logger = init_logger(__name__)


class PilotPrefetcher:
    """Prefetch next-layer experts using a cheap routing heuristic.

    Colibri's PILOT applies layer L+1's gate to layer L's post-attention
    state. Here we approximate with: reuse the current layer's selected
    experts as a hint for L+1 (same-token affinity), optionally refined
    when a next-layer gate module is registered.
    """

    def __init__(self, manager: ExpertTierManager, *, real: bool = False):
        self.manager = manager
        self.real = real
        self._gates: dict[int, torch.nn.Module] = {}

    def register_gate(self, layer_id: int, gate: torch.nn.Module) -> None:
        self._gates[layer_id] = gate

    @torch.inference_mode()
    def prefetch_next(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        current_expert_ids: list[int],
    ) -> None:
        next_id = layer_id + 1
        if next_id not in self.manager.layers:
            return

        predicted = list(current_expert_ids)
        gate = self._gates.get(next_id)
        if gate is not None and hidden_states is not None:
            try:
                # Use last token only for a cheap lookahead.
                h = hidden_states[-1:] if hidden_states.dim() >= 2 else hidden_states
                logits = gate(h)
                if isinstance(logits, tuple):
                    logits = logits[0]
                topk = min(8, logits.shape[-1])
                _, idx = torch.topk(logits.reshape(-1, logits.shape[-1]), topk, dim=-1)
                predicted = idx.reshape(-1).tolist()
            except Exception as e:
                logger.debug("PILOT gate failed for layer %d: %s", next_id, e)

        # Fire-and-forget async ensure into RAM/device without blocking.
        self.manager.prefetch_experts(next_id, predicted, block=not self.real)
