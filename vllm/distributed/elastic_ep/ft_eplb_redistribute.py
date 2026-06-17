# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB placement-table redistribution and weight reload after peer death.

All redistribution operations are deterministic with stable iteration
order, so all surviving ranks running the same function with the same
args produce bit-identical results — no cross-rank communication during
recovery.
"""

from __future__ import annotations

from collections.abc import Generator
from itertools import zip_longest

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def compute_dead_ep_ranks(
    dead_dp_ranks: set[int] | list[int],
    tp_size: int,
) -> set[int]:
    """Expand dead DP ranks to the corresponding dead EP ranks."""
    dead_ep: set[int] = set()
    for dp_rank in dead_dp_ranks:
        for tp_offset in range(tp_size):
            dead_ep.add(dp_rank * tp_size + tp_offset)
    return dead_ep


def mark_dead_columns_inplace(
    physical_to_logical_map: torch.Tensor,
    dead_ep_ranks: set[int],
    num_local_experts: int,
) -> None:
    """Mark dead EP ranks' columns as -1 in place.

    physical_to_logical_map is [num_moe_layers, num_physical]; each EP
    rank owns num_local_experts consecutive columns starting at
    rank * num_local_experts. Shape stays constant (no topology change).
    """
    num_physical = physical_to_logical_map.shape[1]
    ep_world_size = num_physical // num_local_experts
    for ep_rank in dead_ep_ranks:
        if ep_rank < 0 or ep_rank >= ep_world_size:
            raise ValueError(
                f"ep_rank={ep_rank} out of bounds for ep_world_size={ep_world_size}"
            )
        start = ep_rank * num_local_experts
        end = start + num_local_experts
        physical_to_logical_map[:, start:end] = -1


def redistribute_expert_placement(
    physical_to_logical_map: torch.Tensor,
    num_logical: int,
    num_local_experts: int,
) -> set[tuple[int, int]]:
    """Steals slots from over-replicated experts (donors with >1 replica)
    to recover missing logical experts, interleaving across EP ranks
    for balanced weight loading. Modifies physical_to_logical_map in place.

    Returns:
        Set of (layer_idx, logical_id) pairs needing weight reload.

    Raises:
        RuntimeError: If not enough slots to cover missing experts.
    """
    if physical_to_logical_map.ndim != 2:
        raise ValueError(
            f"physical_to_logical_map must be 2D; got shape "
            f"{tuple(physical_to_logical_map.shape)}"
        )

    num_layers, num_physical = physical_to_logical_map.shape
    all_logical = set(range(num_logical))
    reassignments: set[tuple[int, int]] = set()

    for layer_idx in range(num_layers):
        layer = physical_to_logical_map[layer_idx]

        replica_count: dict[int, int] = {}
        for phys_idx in range(num_physical):
            lid = int(layer[phys_idx].item())
            if lid >= 0:
                replica_count[lid] = replica_count.get(lid, 0) + 1

        missing = sorted(all_logical - set(replica_count.keys()))
        if not missing:
            continue

        # Group donors by EP rank, most-redundant first within each
        rank_donors: dict[int, list[tuple[int, int]]] = {}
        for phys_idx in range(num_physical):
            lid = int(layer[phys_idx].item())
            if lid >= 0 and replica_count.get(lid, 0) > 1:
                ep_rank = phys_idx // num_local_experts
                rank_donors.setdefault(ep_rank, []).append(
                    (replica_count[lid], phys_idx)
                )
        for bucket in rank_donors.values():
            bucket.sort(reverse=True)

        # Interleave across ranks (round-robin) so reassignments
        # spread evenly — each rank loads ~equal new weights.
        interleaved: list[tuple[int, int]] = []
        for items in zip_longest(*[rank_donors[r] for r in sorted(rank_donors)]):
            for item in items:
                if item is not None:
                    interleaved.append(item)
        slot_iter = iter(interleaved)

        for logical_id in missing:
            while True:
                candidate = next(slot_iter, None)
                if candidate is None:
                    raise RuntimeError(
                        f"Layer {layer_idx}: no redundant slot for "
                        f"missing expert {logical_id}. "
                        f"EPLB redundancy insufficient."
                    )
                _, global_slot = candidate
                old_lid = int(layer[global_slot].item())
                if replica_count.get(old_lid, 0) > 1:
                    layer[global_slot] = logical_id
                    replica_count[old_lid] -= 1
                    reassignments.add((layer_idx, logical_id))
                    break

    return reassignments


def rebuild_logical_expert_maps(
    physical_to_logical_map: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> None:
    """Rebuild logical_to_physical_map and logical_replica_count from
    physical_to_logical_map, in place.

    Layouts:
    - physical_to_logical_map: [num_layers, num_physical]
    - logical_to_physical_map: [num_layers, num_logical, max_replicas]
    - logical_replica_count: [num_layers, num_logical]
    """
    num_layers, num_physical = physical_to_logical_map.shape
    logical_replica_count.zero_()
    logical_to_physical_map.fill_(-1)
    for layer_idx in range(num_layers):
        for phys_idx in range(num_physical):
            lid = int(physical_to_logical_map[layer_idx, phys_idx].item())
            if lid < 0:
                continue
            c = int(logical_replica_count[layer_idx, lid].item())
            if c < logical_to_physical_map.shape[2]:
                logical_to_physical_map[layer_idx, lid, c] = phys_idx
            logical_replica_count[layer_idx, lid] += 1


def _parse_layer_expert(name: str) -> tuple[int, int] | None:
    """Extract (layer_idx, logical_expert_id) from a tensor name.

    Returns None for non-expert tensors. Expects HF convention:
    model.layers.{L}.mlp.experts.{E}.{shard}.weight
    """
    parts = name.split(".")
    try:
        layer_pos = parts.index("layers")
        expert_pos = parts.index("experts")
    except ValueError:
        return None
    try:
        layer_idx = int(parts[layer_pos + 1])
        expert_id = int(parts[expert_pos + 1])
    except (IndexError, ValueError):
        return None
    return layer_idx, expert_id


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reload_set: set[tuple[int, int]],
) -> int:
    """Reload specific (layer, logical_expert) weights from disk.

    Args:
        reload_set: {(layer_idx, logical_expert_id), ...} from
            redistribute_expert_placement.

    Returns:
        Number of parameter names loaded.
    """
    if not reload_set:
        return 0

    from vllm.model_executor.model_loader.default_loader import (
        DefaultModelLoader,
    )

    loader = DefaultModelLoader(vllm_config.load_config)
    loader.local_expert_ids = None

    all_weights = loader.get_all_weights(vllm_config.model_config, model)

    def filtered_iter() -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in all_weights:
            parsed = _parse_layer_expert(name)
            if parsed is not None and parsed in reload_set:
                yield name, tensor

    logger.info("[FT] Reloading %d (layer, expert) pair(s) from disk.", len(reload_set))

    loaded = model.load_weights(filtered_iter())

    logger.info(
        "[FT] Expert weight reload complete: loaded=%d tensors.",
        len(loaded) if loaded else 0,
    )
    return len(loaded) if loaded else 0
