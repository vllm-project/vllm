# ABOUTME: Helpers to build EPS forward execution context.
# ABOUTME: Collects per-request state used by KV write hooks.

from __future__ import annotations

from typing import Dict, Sequence, Tuple, Optional

import torch

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.context import (
    EpsForwardContext,
    EpsGroupRuntime,
    EpsLayerInfo,
    EpsRequestRuntime,
)
from vllm.v1.eps.state import EpsJLState


LayerLookup = Dict[str, Tuple[int, int]]


def _build_group_runtimes(
    request_ids: Sequence[str],
    request_states: Sequence[Sequence[EpsJLState | None]],
    request_block_ids: Sequence[Tuple[Sequence[int], ...]],
    group_block_sizes: Sequence[int],
) -> Sequence[EpsGroupRuntime]:
    group_runtimes: list[EpsGroupRuntime] = []
    num_groups = len(group_block_sizes)
    for group_idx in range(num_groups):
        block_size = group_block_sizes[group_idx]
        request_entries: list[EpsRequestRuntime] = []
        for req_id, states, blocks in zip(request_ids, request_states, request_block_ids):
            state = states[group_idx] if group_idx < len(states) else None
            block_ids = blocks[group_idx] if group_idx < len(blocks) else []
            block_mapping = {int(block_id): idx for idx, block_id in enumerate(block_ids)}
            request_entries.append(
                EpsRequestRuntime(
                    request_id=req_id,
                    state=state,
                    block_mapping=block_mapping,
                )
            )
        group_runtimes.append(
            EpsGroupRuntime(block_size=block_size, request_runtimes=request_entries)
        )
    return group_runtimes


def _build_layer_map(layer_lookup: LayerLookup) -> Dict[str, EpsLayerInfo]:
    layer_map: Dict[str, EpsLayerInfo] = {}
    for layer_name, (group_id, layer_index) in layer_lookup.items():
        layer_map[layer_name] = EpsLayerInfo(
            group_id=group_id,
            layer_index=layer_index,
            layer_name=layer_name,
        )
    return layer_map


def build_eps_forward_context(
    *,
    cfg: EpsRuntimeConfig,
    layer_lookup: LayerLookup,
    group_block_sizes: Sequence[int],
    request_ids: Sequence[str],
    request_states: Sequence[Sequence[EpsJLState | None]],
    request_block_ids: Sequence[Tuple[Sequence[int], ...]],
    token_request_indices: Optional[torch.Tensor] = None,
    cudagraph_capture: bool = False,
) -> EpsForwardContext:
    group_runtimes = _build_group_runtimes(
        request_ids=request_ids,
        request_states=request_states,
        request_block_ids=request_block_ids,
        group_block_sizes=group_block_sizes,
    )
    layer_map = _build_layer_map(layer_lookup)

    if token_request_indices is None:
        token_request_indices = torch.empty(0, dtype=torch.int32)

    return EpsForwardContext(
        enabled=cfg.enabled,
        cfg=cfg,
        layer_map=layer_map,
        group_runtimes=list(group_runtimes),
        token_request_indices=token_request_indices.clone(),
        cudagraph_capture=cudagraph_capture,
    )


__all__ = ["build_eps_forward_context"]
