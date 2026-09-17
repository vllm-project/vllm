# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-round FP8 expert loading plans, independent of backend conversion."""

import inspect
from dataclasses import dataclass, field

from .trace import ReloadError, ReloadState, SlotKey, SlotTable


@dataclass
class RoutedExpertsReloadPlan:
    mapping: tuple[tuple[str, str, int, str], ...] = ()
    local_experts: tuple[int, ...] = ()
    all_slots: set[SlotKey] = field(default_factory=set)

    @staticmethod
    def _local_experts(state: ReloadState) -> tuple[int, ...]:
        module = state.module
        return tuple(
            module.expert_map_manager.map_global_to_local(expert)
            for expert in range(module.global_num_experts)
        )

    @staticmethod
    def _key(role: str, expert: int, shard: str) -> SlotKey:
        return SlotKey(role, (("expert_id", expert), ("shard_id", shard)))

    def build(self, state: ReloadState) -> SlotTable:
        module = state.module
        if module.moe_config.moe_parallel_config.enable_eplb and (
            module.eplb_state is None
            or module.eplb_state.logical_to_physical_map is None
        ):
            raise ReloadError(
                "EPLB runtime placement must be initialized before reload"
            )
        self.mapping = tuple(module.get_expert_mapping())
        self.local_experts = self._local_experts(state)
        local_ids = sorted(expert for expert in self.local_experts if expert != -1)
        if local_ids != list(range(module.local_num_experts)) or not local_ids:
            raise ReloadError("Unsupported local expert ownership")
        if any(
            state.metadata[role].shape[0] != module.local_num_experts
            for role in state.roles
        ):
            raise ReloadError("Local expert capacity changed since cold load")
        self.all_slots = {
            self._key(role, expert, shard)
            for prefix, _, expert, shard in self.mapping
            for role in state.roles
            if f"experts.{role}".startswith(prefix)
            and (shard != "w3" or module.moe_config.is_act_and_mul)
        }
        slots = SlotTable(
            expected={key for key in self.all_slots if self.is_local(key)}
        )
        if any(
            not any(key.role == role for key in slots.expected) for role in state.roles
        ):
            raise ReloadError("Expert mapping does not cover all reload roles")
        return slots

    def validate(self, state: ReloadState) -> None:
        if (
            tuple(state.module.get_expert_mapping()) != self.mapping
            or self._local_experts(state) != self.local_experts
        ):
            raise ReloadError("Expert mapping changed during reload; quiesce EPLB")

    def slot_key(self, role: str, bound: inspect.BoundArguments) -> SlotKey:
        if bound.arguments["loaded_weight"].ndim == 3:
            raise ReloadError("Use RoutedExperts.load_weights to unpack fused experts")
        return self._key(
            role, bound.arguments["expert_id"], bound.arguments["shard_id"]
        )

    def is_local(self, key: SlotKey) -> bool:
        if key not in self.all_slots:
            raise ReloadError(f"Unknown expert reload slot: {key}")
        expert = dict(key.arguments)["expert_id"]
        return self.local_experts[expert] != -1
