# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as re
import torch


@dataclass(frozen=True)
class PrunedExpertsProfile:
    pruned_experts_by_layer: dict[int, frozenset[int]]


@lru_cache(maxsize=8)
def load_pruned_experts_profile(profile_path: str) -> PrunedExpertsProfile:
    """Load a profile describing experts whose weights should be skipped.

    The profile uses original logical expert ids. Pruned experts are removed
    from the local expert map (mapped to -1), so the MoE kernel never selects
    them.  Their weight tensors are allocated but left unloaded.
    """
    path = Path(profile_path)
    with path.open() as f:
        profile = json.load(f)

    if profile.get("version") != 1:
        raise ValueError(
            "MoE pruned experts profile must contain version 1, "
            f"got {profile.get('version')!r}."
        )

    entries = profile.get("pruned_experts")
    if not isinstance(entries, list):
        raise ValueError(
            "MoE pruned experts profile must contain a 'pruned_experts' list."
        )

    pruned_by_layer: dict[int, set[int]] = {}
    for entry in entries:
        if not (
            isinstance(entry, list | tuple)
            and len(entry) == 2
            and all(isinstance(value, int) for value in entry)
        ):
            raise ValueError("Each pruned expert entry must be a [layer, expert] pair.")
        layer_idx, expert_idx = entry
        if layer_idx < 0 or expert_idx < 0:
            raise ValueError("Layer and expert ids must be non-negative.")
        pruned_by_layer.setdefault(layer_idx, set()).add(expert_idx)

    return PrunedExpertsProfile(
        {
            layer_idx: frozenset(expert_ids)
            for layer_idx, expert_ids in pruned_by_layer.items()
        }
    )


def build_pruned_expert_map(
    num_experts: int,
    pruned_experts: frozenset[int],
    base_expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Build a global-to-local expert map with pruned experts mapped to -1."""
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
    next_local_expert = 0

    for expert_idx in range(num_experts):
        if expert_idx in pruned_experts:
            continue
        if base_expert_map is not None and int(base_expert_map[expert_idx]) == -1:
            continue
        expert_map[expert_idx] = next_local_expert
        next_local_expert += 1

    return expert_map, next_local_expert


def extract_moe_layer_index(prefix: str) -> int | None:
    """Extract the decoder layer index from a vLLM module prefix."""
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        return None
    return int(match.group(1))
