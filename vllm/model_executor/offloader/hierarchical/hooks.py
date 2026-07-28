# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoERunner integration hooks for hierarchical expert staging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.offloader.hierarchical.manager import get_tier_manager

if TYPE_CHECKING:
    pass


def maybe_ensure_and_remap(
    layer_id: int,
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor | None = None,
) -> torch.Tensor:
    """If hierarchical staging is active, ensure experts and remap ids."""
    mgr = get_tier_manager()
    if mgr is None or not mgr._initialized:
        return topk_ids
    # Remap using original ids first for pilot, then ensure.
    original = topk_ids
    remapped = mgr.ensure_and_remap(layer_id, topk_ids)
    if hidden_states is not None:
        mgr.maybe_pilot_prefetch(layer_id, hidden_states, original)
    return remapped


def register_routed_experts(layer_id: int, module) -> None:
    """Register a RoutedExperts module with the active tier manager."""
    mgr = get_tier_manager()
    if mgr is None:
        return
    mgr.register_moe_module(layer_id, module)
