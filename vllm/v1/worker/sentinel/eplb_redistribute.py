# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB fault-tolerance helpers: Gloo group recovery, expert redistribution,
and weight reload after peer death.

All redistribution operations are deterministic with stable iteration
order, so all surviving ranks running the same function with the same
args produce bit-identical results — no cross-rank communication during
recovery.
"""

from __future__ import annotations

from collections.abc import Generator
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    reinit_gloo_group,
)
from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

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


def mark_dead_expert_slots_inplace(
    physical_to_logical_map: torch.Tensor,
    dead_ep_ranks: set[int],
    num_local_experts: int,
) -> None:
    """Mark dead EP ranks' physical expert slots as -1 in place.

    physical_to_logical_map is [num_moe_layers, num_physical]; each EP
    rank owns num_local_experts consecutive physical slots starting at
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
        layer_p2l = physical_to_logical_map[layer_idx]

        # Replica count per logical expert across the surviving slots.
        replica_count: dict[int, int] = {}
        for phys_idx in range(num_physical):
            logical_id = int(layer_p2l[phys_idx].item())
            if logical_id >= 0:
                replica_count[logical_id] = replica_count.get(logical_id, 0) + 1

        missing_logical = sorted(all_logical - set(replica_count.keys()))
        if not missing_logical:
            continue

        # Reusable slots grouped by owning EP rank: an over-replicated
        # (>1) expert can spare a slot without losing coverage. Each entry
        # is (replica_count, phys_idx).
        spare_slots_by_rank: dict[int, list[tuple[int, int]]] = {}
        for phys_idx in range(num_physical):
            logical_id = int(layer_p2l[phys_idx].item())
            if logical_id >= 0 and replica_count.get(logical_id, 0) > 1:
                ep_rank = phys_idx // num_local_experts
                spare_slots_by_rank.setdefault(ep_rank, []).append(
                    (replica_count[logical_id], phys_idx)
                )
        for bucket in spare_slots_by_rank.values():
            bucket.sort(reverse=True)

        # Interleave across ranks (round-robin) so reassignments
        # spread evenly — each rank loads ~equal new weights.
        interleaved: list[tuple[int, int]] = []
        for items in zip_longest(
            *[spare_slots_by_rank[r] for r in sorted(spare_slots_by_rank)]
        ):
            for item in items:
                if item is not None:
                    interleaved.append(item)
        slot_iter = iter(interleaved)

        for logical_id in missing_logical:
            while True:
                candidate = next(slot_iter, None)
                if candidate is None:
                    raise RuntimeError(
                        f"Layer {layer_idx}: no redundant slot for "
                        f"missing expert {logical_id}. "
                        f"EPLB redundancy insufficient."
                    )
                _, spare_slot = candidate
                prev_logical = int(layer_p2l[spare_slot].item())
                # Re-check: an earlier reuse may have left prev_logical
                # with a single replica, making this slot no longer spare.
                if replica_count.get(prev_logical, 0) > 1:
                    layer_p2l[spare_slot] = logical_id
                    replica_count[prev_logical] -= 1
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
    """Extract (layer_idx, logical_expert_id) from a checkpoint tensor name.

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

    matched: set[tuple[int, int]] = set()

    def filtered_iter() -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in all_weights:
            parsed = _parse_layer_expert(name)
            if parsed is not None and parsed in reload_set:
                matched.add(parsed)
                yield name, tensor

    logger.info("[FT] Reloading %d (layer, expert) pair(s) from disk.", len(reload_set))

    loaded = model.load_weights(filtered_iter())

    unmatched = reload_set - matched
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {sorted(unmatched)[:5]}. The model's "
            f"expert weights likely use a layout _parse_layer_expert does not "
            f"support (fused experts or non-'layers'/'experts' tokens)."
        )

    logger.info(
        "[FT] Expert weight reload complete: loaded=%d tensors.",
        len(loaded) if loaded else 0,
    )
    return len(loaded) if loaded else 0


def reinit_eplb_gloo_groups(
    params: dict[str, Any],
    master_ip: str,
) -> None:
    """Reinit EP and EPLB Gloo cpu_groups from FT params."""
    for port_key, get_group in [
        ("new_ep_group_port", get_ep_group),
        ("new_eplb_group_port", get_eplb_group),
    ]:
        port = params.get(port_key)
        if port is None:
            continue
        group = get_group()
        if group is None or group.cpu_group is None:
            continue
        reinit_gloo_group(group, master_ip, port, group.rank_in_group, group.world_size)
        logger.info("[FT] Reinited %s Gloo group on port %d", port_key, port)


def refresh_eplb_communicator_group(model_runner: GPUModelRunner) -> None:
    """Update EPLB communicator's _cpu_group after Gloo reinit."""
    eplb_state = getattr(model_runner, "eplb_state", None)
    if eplb_state is None:
        return
    eplb_group = get_eplb_group()
    if eplb_group is None:
        return
    new_cpu_group = eplb_group.cpu_group
    for ms in eplb_state.model_states.values():
        if hasattr(ms.communicator, "_cpu_group"):
            ms.communicator._cpu_group = new_cpu_group
