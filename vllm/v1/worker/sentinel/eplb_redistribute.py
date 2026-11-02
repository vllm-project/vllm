# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB fault-tolerance helpers: expert redistribution and weight reload
after peer death.

All redistribution operations are deterministic with stable iteration
order, so all surviving ranks running the same function with the same
args produce bit-identical results — no cross-rank communication during
recovery.
"""

from __future__ import annotations

from collections.abc import Generator
from inspect import signature
from itertools import zip_longest
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.sentinel.expert_reload import _placement_weight_loader

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def compute_dead_ep_ranks(
    dead_dp_ranks: set[int] | list[int],
    tp_size: int,
    pcp_size: int = 1,
) -> set[int]:
    """Expand dead DP ranks to the corresponding dead EP ranks.

    Each DP rank owns a contiguous block of tp_size * pcp_size EP ranks
    """
    block = tp_size * pcp_size
    return {
        dp_rank * block + offset for dp_rank in dead_dp_ranks for offset in range(block)
    }


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
    if not dead_ep_ranks:
        return
    slots_by_rank = physical_to_logical_map.view(
        physical_to_logical_map.shape[0], -1, num_local_experts
    )
    slots_by_rank[:, sorted(dead_ep_ranks)] = -1


def redistribute_expert_placement(
    physical_to_logical_map: torch.Tensor,
    num_logical: int,
    num_local_experts: int,
) -> set[tuple[int, int]]:
    """Steals slots from over-replicated experts (donors with >1 replica)
    to recover missing logical experts, interleaving across EP ranks
    for balanced weight loading. Modifies physical_to_logical_map in place.

    Returns:
        Set of (moe_layer_idx, logical_id) pairs needing weight reload.

    Raises:
        RuntimeError: If not enough slots to cover missing experts.

    """
    num_layers = physical_to_logical_map.shape[0]
    all_logical = set(range(num_logical))
    reassignments: set[tuple[int, int]] = set()

    for layer_idx in range(num_layers):
        layer_p2l = physical_to_logical_map[layer_idx].cpu().tolist()

        # Replica count per logical expert across the surviving slots.
        replica_count: dict[int, int] = {}
        for logical_id in layer_p2l:
            if logical_id >= 0:
                replica_count[logical_id] = replica_count.get(logical_id, 0) + 1

        missing_logical = sorted(all_logical - set(replica_count.keys()))
        if not missing_logical:
            continue

        # Spare slots by owning EP rank: an expert with >1 replica can give
        # up a slot without losing coverage. Values: (replica_count, phys_idx).
        spare_slots_by_rank: dict[int, list[tuple[int, int]]] = {}
        for phys_idx, logical_id in enumerate(layer_p2l):
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
                prev_logical = layer_p2l[spare_slot]
                # Re-check: an earlier reuse may have left prev_logical
                # with a single replica, making this slot no longer spare.
                if replica_count.get(prev_logical, 0) > 1:
                    layer_p2l[spare_slot] = logical_id
                    replica_count[prev_logical] -= 1
                    reassignments.add((layer_idx, logical_id))
                    break

        physical_to_logical_map[layer_idx].copy_(
            torch.tensor(layer_p2l, dtype=physical_to_logical_map.dtype)
        )

    return reassignments


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reload_set: set[tuple[int, int]],
    physical_to_logical_map: torch.Tensor,
) -> None:
    """Reload complete MoE layers affected by expert reassignment from disk.

    Args:
        model: The model whose FusedMoE layers receive the reloaded weights.
        vllm_config: Engine config, used for the model loader and model config.
        reload_set: {(moe_layer_idx, logical_expert_id), ...} from
            redistribute_expert_placement.
        physical_to_logical_map: Recovered placement, including all replicas.

    """
    if not reload_set:
        return

    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
    )
    from vllm.model_executor.model_loader.reload.utils import get_layer_tensors

    moe_layers = list(model.moe_layers)
    prefix_renames = getattr(
        getattr(model, "hf_to_vllm_mapper", None), "orig_to_new_prefix", {}
    )

    def ckpt_prefix(layer_idx: int) -> str:
        prefix = moe_layers[layer_idx].layer_name + "."
        for orig, new in prefix_renames.items():
            if new and prefix.startswith(new):
                return orig + prefix[len(new) :]
        return prefix

    # Runtime-format expert reload: processing can merge scales or repack weights.
    # Reload every expert in an affected layer so finalization never consumes
    # uninitialized checkpoint storage for unchanged local experts.
    layer_indices = sorted({i for i, _ in reload_set})
    prefixes = tuple(ckpt_prefix(i) for i in layer_indices)
    layers = [
        getattr(moe_layers[i], "routed_experts", moe_layers[i]) for i in layer_indices
    ]
    loader = get_model_loader(vllm_config.load_config)
    all_weights = loader.get_all_weights(vllm_config.model_config, model)
    matched: set[int] = set()
    rename_checkpoint_weight = getattr(
        model, "_rename_weight_for_modelopt_checkpoint", lambda name: name
    )

    def filtered_iter() -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, tensor in all_weights:
            # Honor model-specific checkpoint renaming before filtering, while
            # passing the original name to load_weights for its normal conversion.
            renamed = rename_checkpoint_weight(name)
            for layer_idx, prefix in zip(layer_indices, prefixes):
                runtime_prefix = moe_layers[layer_idx].layer_name + "."
                if name.startswith(prefix) or renamed.startswith(runtime_prefix):
                    matched.add(layer_idx)
                    yield name, tensor
                    break

    # Reload recovered placement: remap before the layerwise loader buffers its
    # arguments, so deferred processing replays physical IDs against runtime maps.
    for layer_idx, layer in zip(layer_indices, layers):
        if getattr(layer, "quant_config", None) is not None and (
            layer.quant_config.get_name() == "gpt_oss_mxfp4"
        ):
            raise NotImplementedError(
                "Scale-down reload of fused GPT-OSS MXFP4 weights"
            )
        initialize_layerwise_reload(layer, defer_processing=True)
        quant_method = layer.quant_method
        quant_method = (
            getattr(quant_method, "requantization_source", None) or quant_method
        )
        local_ids: dict[int, list[int]] = {}
        for physical_id, logical_id in enumerate(
            physical_to_logical_map[layer_idx].cpu().tolist()
        ):
            if (
                logical_id >= 0
                and layer._map_global_expert_id_to_local_expert_id(physical_id) >= 0
            ):
                local_ids.setdefault(logical_id, []).append(physical_id)
        for param in get_layer_tensors(layer).values():
            weight_loader = getattr(param, "weight_loader", None)
            if (
                weight_loader is not None
                and "expert_id" in signature(weight_loader).parameters
            ):
                param.weight_loader = _placement_weight_loader(
                    weight_loader,
                    local_ids,
                    layer.moe_config.num_logical_experts,
                    physical_to_logical_map.shape[1],
                    use_global_sf=getattr(quant_method, "use_global_sf", False),
                )

    logger.info("[FT] Reloading %d (layer, expert) pair(s) from disk.", len(reload_set))

    loaded = model.load_weights(filtered_iter())

    unmatched = set(layer_indices) - matched
    if unmatched:
        raise RuntimeError(
            f"[FT] MoE layers {sorted(unmatched)} had no matching checkpoint weights."
        )
    for layer in layers:
        finalize_layerwise_reload(layer, vllm_config.model_config)

    logger.info(
        "[FT] Expert weight reload complete: loaded=%d tensors.",
        len(loaded) if loaded else 0,
    )


def reset_eplb_async_state(model_runner: GPUModelRunner) -> None:
    """Clear stale EPLB async state after fault or scale-down."""
    eplb_state = getattr(model_runner, "eplb_state", None)
    if eplb_state is None:
        return

    # Increment the epoch first, so an async worker that publishes a result
    # after this point sees the change and discards it.
    eplb_state.ft_reset_epoch += 1
    for ms in eplb_state.model_states.values():
        ms.rebalanced = False
        result = ms.pending_result
        ms.pending_result = None
        if result is not None:
            # Wake an async worker waiting on consumed_event; the in-flight
            # layer result is discarded, not applied.
            result.consumed_event.record()
        ms.expert_load_pass.zero_()
        ms.expert_load_window.zero_()

    eplb_state.expert_rearrangement_step = 0
    eplb_state.expert_load_window_step = 0
