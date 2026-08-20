# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from functools import cache
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.models.utils import extract_layer_index


def get_num_expert_map_layers(config: Any) -> int:
    num_draft_layers = getattr(config, "num_nextn_predict_layers", None)
    num_draft_layers = num_draft_layers or getattr(config, "n_mtp_layers", 0) or 0
    return config.num_hidden_layers + num_draft_layers


def get_expert_map_layer_index(weight_name: str) -> int:
    """Extract the model layer index while ignoring checkpoint expert IDs."""
    layer_name, separator, _ = weight_name.partition(".ffn.")
    if not separator:
        raise ValueError(f"DSV4 MoE weight name has no .ffn. component: {weight_name}")
    return extract_layer_index(layer_name)


@cache
def load_static_expert_map(
    path: str,
    *,
    num_layers: int,
    num_experts: int,
    num_physical_experts: int | None = None,
    num_expert_groups: int = 1,
) -> torch.Tensor:
    """Load a DSV4 physical-to-logical expert map from JSON."""
    try:
        raw_map = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to load DSV4 expert map from {path}: {exc}") from exc

    if not isinstance(raw_map, list) or not raw_map:
        raise ValueError("DSV4 expert map must be a non-empty JSON list.")

    has_nested_list = any(isinstance(value, list) for value in raw_map)
    if has_nested_list and not all(isinstance(value, list) for value in raw_map):
        raise ValueError(
            "DSV4 expert map must contain expert IDs or per-layer lists of IDs."
        )
    layer_maps = raw_map if has_nested_list else [raw_map]

    if len(layer_maps) not in (1, num_layers):
        raise ValueError(
            "DSV4 expert map must define either one shared permutation or "
            f"{num_layers} layer permutations, got {len(layer_maps)}."
        )

    if num_physical_experts is None:
        num_physical_experts = num_experts
    if num_physical_experts < num_experts:
        raise ValueError(
            "DSV4 num_physical_experts must be at least num_experts, got "
            f"{num_physical_experts} and {num_experts}."
        )

    expected_experts = set(range(num_experts))
    for layer_idx, layer_map in enumerate(layer_maps):
        if not all(_is_int(expert_id) for expert_id in layer_map):
            raise ValueError(
                f"DSV4 expert map layer {layer_idx} contains a non-integer ID."
            )
        if len(layer_map) != num_physical_experts:
            raise ValueError(
                f"DSV4 expert map layer {layer_idx} must define "
                f"{num_physical_experts} physical slots, got {len(layer_map)}."
            )
        actual_experts = set(layer_map)
        if actual_experts != expected_experts:
            missing = sorted(expected_experts - actual_experts)
            invalid = sorted(actual_experts - expected_experts)
            raise ValueError(
                f"DSV4 expert map layer {layer_idx} must assign every logical "
                f"expert ID in [0, {num_experts}); missing={missing}, "
                f"invalid={invalid}."
            )
        if num_physical_experts == num_experts:
            _validate_expert_groups(
                layer_map,
                layer_idx=layer_idx,
                num_experts=num_experts,
                num_expert_groups=num_expert_groups,
            )

    expert_map = torch.tensor(layer_maps, dtype=torch.long, device="cpu")
    if len(layer_maps) == 1:
        expert_map = expert_map.expand(num_layers, -1).contiguous()
    return expert_map


def remap_router_weight(
    loaded_weight: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
) -> torch.Tensor:
    """Reorder an expert-indexed router tensor into physical expert order."""
    if loaded_weight.shape[0] != physical_to_logical_map.numel():
        raise ValueError(
            "DSV4 router weight expert dimension does not match the expert map: "
            f"{loaded_weight.shape[0]} vs {physical_to_logical_map.numel()}."
        )
    return loaded_weight.index_select(
        0, physical_to_logical_map.to(device=loaded_weight.device)
    )


def remap_router_expert_ids(
    loaded_weight: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
) -> torch.Tensor:
    """Translate logical expert IDs in a hash router to physical IDs."""
    logical_to_physical_map = torch.empty_like(physical_to_logical_map)
    logical_to_physical_map[physical_to_logical_map] = torch.arange(
        physical_to_logical_map.numel(), dtype=torch.long
    )
    return logical_to_physical_map.to(device=loaded_weight.device)[
        loaded_weight.long()
    ].to(dtype=loaded_weight.dtype)


def remap_expert_params_mapping(
    mapping: list[tuple[str, str, int, str]],
    physical_to_logical_map: torch.Tensor,
) -> list[tuple[str, str, int, str]]:
    """Point each physical expert slot at its mapped checkpoint expert."""
    remapped = []
    for param_name, weight_name, physical_id, shard_id in mapping:
        logical_id = physical_to_logical_map[physical_id].item()
        weight_name = weight_name.replace(
            f"experts.{physical_id}.", f"experts.{logical_id}."
        )
        remapped.append((param_name, weight_name, physical_id, shard_id))
    return remapped


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_expert_groups(
    layer_map: list[int],
    *,
    layer_idx: int,
    num_experts: int,
    num_expert_groups: int,
) -> None:
    if num_expert_groups <= 0:
        raise ValueError("DSV4 num_expert_groups must be greater than zero.")
    if num_experts % num_expert_groups != 0:
        raise ValueError(
            f"DSV4 has {num_experts} experts, which is not divisible by "
            f"{num_expert_groups} expert groups."
        )

    group_size = num_experts // num_expert_groups
    for physical_group in range(num_expert_groups):
        start = physical_group * group_size
        logical_groups = {
            expert_id // group_size
            for expert_id in layer_map[start : start + group_size]
        }
        if len(logical_groups) != 1:
            raise ValueError(
                f"DSV4 expert map layer {layer_idx} mixes logical expert groups "
                f"in physical group {physical_group}."
            )
