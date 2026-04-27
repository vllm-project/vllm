# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import torch

ExpertState = Literal["loading", "resident", "executing", "evicting"]
GPU_MEMORY_RETRY_SECONDS = 5
GPU_MEMORY_RETRY_LIMIT = 10


@dataclass
class ActiveExpertEntry:
    layer_id: int
    expert_id: int
    gpu_slot_id: int
    state: ExpertState
    weight_bytes: int
    loaded_step: int
    last_used_step: int
    recent_token_count: int


@dataclass
class _ExpertTensor:
    name: str
    source: torch.Tensor
    target: torch.Tensor


class ExpertCache:
    """Synchronous Stage-2 MoE expert cache.

    The CPU tensors are the source of truth. The layer parameters remain the
    execution tensors; when an expert is demanded, its slice is copied from CPU
    to the current parameter device before the fused MoE kernel runs.
    """

    def __init__(
        self,
        *,
        layer_id: int,
        active_expert_budget: int | None,
        expert_tensors: list[_ExpertTensor],
        use_identity_slots: bool = True,
    ) -> None:
        if active_expert_budget is not None and active_expert_budget < 1:
            raise ValueError("active_expert_budget must be at least 1")
        if not expert_tensors:
            raise ValueError("ExpertCache requires at least one expert tensor")

        self.layer_id = layer_id
        self.expert_tensors = expert_tensors
        self.use_identity_slots = use_identity_slots
        self.active_experts: dict[int, ActiveExpertEntry] = {}
        self.step = 0

        self._num_experts = self.expert_tensors[0].source.shape[0]
        self.auto_active_expert_budget = (
            active_expert_budget is None and not self.use_identity_slots
        )
        self.active_expert_budget = active_expert_budget or self._num_experts
        self._slot_count = (
            self.expert_tensors[0].target.shape[0]
            if self.use_identity_slots
            else self.active_expert_budget
        )
        self._free_slots = list(range(self._slot_count))
        self._bytes_by_expert = self._compute_bytes_by_expert()
        self._target_device = self.expert_tensors[0].target.device

    @classmethod
    def from_layer(
        cls,
        layer: torch.nn.Module,
        *,
        active_expert_budget: int | None,
        layer_id: int,
    ) -> ExpertCache:
        expert_tensors: list[_ExpertTensor] = []
        local_num_experts = int(layer.local_num_experts)

        for name, param in layer.named_parameters(recurse=False):
            if param.ndim == 0 or param.shape[0] != local_num_experts:
                continue
            source = cls._make_cpu_source(param.detach())
            expert_tensors.append(
                _ExpertTensor(name=name, source=source, target=param.data)
            )

        return cls(
            layer_id=layer_id,
            active_expert_budget=active_expert_budget,
            expert_tensors=expert_tensors,
            use_identity_slots=True,
        )

    @classmethod
    def from_cpu_sources(
        cls,
        *,
        layer_id: int,
        active_expert_budget: int | None,
        sources: dict[str, torch.Tensor],
        device: torch.device,
    ) -> ExpertCache:
        expert_tensors: list[_ExpertTensor] = []
        for name, source in sources.items():
            source = cls._make_cpu_source(source)
            target = torch.empty((0, *source.shape[1:]), dtype=source.dtype)
            expert_tensors.append(
                _ExpertTensor(name=name, source=source, target=target)
            )

        cache = cls(
            layer_id=layer_id,
            active_expert_budget=active_expert_budget,
            expert_tensors=expert_tensors,
            use_identity_slots=False,
        )
        cache._target_device = device
        return cache

    @staticmethod
    def _make_cpu_source(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach().to(device="cpu", copy=True).contiguous()
        if torch.cuda.is_available():
            source = source.pin_memory()
        return source

    def _compute_bytes_by_expert(self) -> dict[int, int]:
        bytes_by_expert: dict[int, int] = {}
        for expert_id in range(self._num_experts):
            bytes_by_expert[expert_id] = sum(
                int(t.source[expert_id].numel() * t.source.element_size())
                for t in self.expert_tensors
            )
        return bytes_by_expert

    def _required_cache_bytes(self, slot_count: int) -> int:
        if not self._bytes_by_expert:
            return 0
        return max(self._bytes_by_expert.values()) * slot_count

    def _set_slot_count(self, slot_count: int) -> None:
        slot_count = max(1, min(slot_count, self._num_experts))
        if slot_count == self._slot_count:
            return
        self.active_expert_budget = slot_count
        self._slot_count = slot_count
        self.release_targets_to_cpu()

    def _fit_auto_budget_to_available_memory(self, active_expert_count: int) -> None:
        if not self.auto_active_expert_budget:
            return
        desired_slots = max(1, min(active_expert_count, self._num_experts))
        if self._target_device.type == "cuda":
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(self._target_device)
            max_expert_bytes = max(self._bytes_by_expert.values())
            if max_expert_bytes > 0:
                desired_slots = min(
                    desired_slots,
                    max(1, int(free_bytes // max_expert_bytes)),
                )
        self._set_slot_count(desired_slots)

    def _wait_for_gpu_memory(self, expert_id: int, device: torch.device) -> None:
        if device.type != "cuda":
            return

        required_bytes = self._bytes_by_expert[expert_id]
        for attempt in range(GPU_MEMORY_RETRY_LIMIT + 1):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes >= required_bytes:
                return
            if attempt == GPU_MEMORY_RETRY_LIMIT:
                raise RuntimeError(
                    "Insufficient free GPU memory to load MoE expert "
                    f"{expert_id} for layer {self.layer_id}: required "
                    f"{required_bytes} bytes, free {free_bytes} bytes after "
                    f"{GPU_MEMORY_RETRY_LIMIT} retries."
                )
            torch.cuda.empty_cache()
            time.sleep(GPU_MEMORY_RETRY_SECONDS)

    def _targets_are_allocated_on(self, device: torch.device) -> bool:
        return all(
            expert_tensor.target.device == device
            and expert_tensor.target.shape[0] == self._slot_count
            for expert_tensor in self.expert_tensors
        )

    def ensure_targets_on_device(self, device: torch.device) -> None:
        if self._targets_are_allocated_on(device):
            return
        if device.type == "cuda":
            required_bytes = self._required_cache_bytes(self._slot_count)
            for attempt in range(GPU_MEMORY_RETRY_LIMIT + 1):
                free_bytes, _ = torch.cuda.mem_get_info(device)
                if free_bytes >= required_bytes:
                    break
                if attempt == GPU_MEMORY_RETRY_LIMIT:
                    raise RuntimeError(
                        "Insufficient free GPU memory to allocate MoE active "
                        f"expert cache for layer {self.layer_id}: required "
                        f"{required_bytes} bytes, free {free_bytes} bytes after "
                        f"{GPU_MEMORY_RETRY_LIMIT} retries."
                    )
                torch.cuda.empty_cache()
                time.sleep(GPU_MEMORY_RETRY_SECONDS)

        for expert_tensor in self.expert_tensors:
            source = expert_tensor.source
            expert_tensor.target = torch.empty(
                (self._slot_count, *source.shape[1:]),
                dtype=source.dtype,
                device=device,
            )
        self._target_device = device
        self._free_slots = list(range(self._slot_count))
        self.active_experts.clear()

    def release_targets_to_cpu(self) -> None:
        if self.use_identity_slots:
            return
        for expert_tensor in self.expert_tensors:
            source = expert_tensor.source
            expert_tensor.target = torch.empty(
                (0, *source.shape[1:]),
                dtype=source.dtype,
                device="cpu",
            )
        self._free_slots = list(range(self._slot_count))
        self.active_experts.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_expert(self, expert_id: int, slot_id: int) -> None:
        self.ensure_targets_on_device(self._target_device)
        self._wait_for_gpu_memory(expert_id, self.expert_tensors[0].target.device)
        for expert_tensor in self.expert_tensors:
            source = expert_tensor.source[expert_id]
            target = expert_tensor.target[slot_id]
            target.copy_(source, non_blocking=True)

    def _allocate_slot(self, expert_id: int) -> int:
        if self.use_identity_slots:
            return expert_id
        if not self._free_slots:
            raise RuntimeError("No free MoE offload GPU expert slots are available")
        return self._free_slots.pop(0)

    def _evict_one(self) -> None:
        candidates = [
            entry
            for entry in self.active_experts.values()
            if entry.state != "executing"
        ]
        if not candidates:
            raise RuntimeError("No evictable MoE experts are available")

        victim = min(
            candidates,
            key=lambda entry: (entry.recent_token_count, entry.last_used_step),
        )
        victim.state = "evicting"
        if not self.use_identity_slots:
            self._free_slots.append(victim.gpu_slot_id)
            self._free_slots.sort()
        del self.active_experts[victim.expert_id]

    def _evict_expert(self, expert_id: int) -> None:
        entry = self.active_experts[expert_id]
        entry.state = "evicting"
        if not self.use_identity_slots:
            self._free_slots.append(entry.gpu_slot_id)
            self._free_slots.sort()
        del self.active_experts[expert_id]

    def _ensure_capacity_for(self, expert_id: int) -> None:
        if expert_id in self.active_experts:
            return
        while len(self.active_experts) >= self.active_expert_budget:
            self._evict_one()

    def ensure_experts_resident(
        self,
        expert_token_counts: dict[int, int],
    ) -> None:
        """Load demanded experts synchronously and update LRU/hotness stats."""
        if not expert_token_counts:
            return
        self.ensure_targets_on_device(self._target_device)
        if (
            not self.use_identity_slots
            and len(expert_token_counts) > self.active_expert_budget
        ):
            raise RuntimeError(
                "MoE CPU offload Stage 2 cannot execute a batch that routes to "
                f"{len(expert_token_counts)} local experts with "
                f"active_expert_staging_slots={self.active_expert_budget}. "
                "Use smaller routed waves or reduce batch size."
            )
        if not self.use_identity_slots:
            required_experts = set(expert_token_counts)
            for expert_id in list(self.active_experts):
                if expert_id not in required_experts:
                    self._evict_expert(expert_id)

        self.step += 1
        for expert_id, token_count in sorted(
            expert_token_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            self._ensure_capacity_for(expert_id)
            entry = self.active_experts.get(expert_id)
            if entry is None:
                slot_id = self._allocate_slot(expert_id)
                self._load_expert(expert_id, slot_id)
                entry = ActiveExpertEntry(
                    layer_id=self.layer_id,
                    expert_id=expert_id,
                    gpu_slot_id=slot_id,
                    state="resident",
                    weight_bytes=self._bytes_by_expert[expert_id],
                    loaded_step=self.step,
                    last_used_step=self.step,
                    recent_token_count=token_count,
                )
                self.active_experts[expert_id] = entry
            else:
                entry.state = "resident"
                entry.last_used_step = self.step
                entry.recent_token_count = token_count

    def resident_expert_ids(self) -> set[int]:
        return set(self.active_experts)

    def retire_experts(self, expert_ids: set[int] | None = None) -> None:
        """Mark loaded experts non-resident and free their execution slots."""
        if expert_ids is None:
            expert_ids = set(self.active_experts)
        for expert_id in list(expert_ids):
            if expert_id in self.active_experts:
                self._evict_expert(expert_id)

    def target_for(self, name: str) -> torch.Tensor:
        for expert_tensor in self.expert_tensors:
            if expert_tensor.name == name:
                return expert_tensor.target
        raise KeyError(name)

    def move_targets_to(self, device: torch.device) -> None:
        self.ensure_targets_on_device(device)

    def remap_topk_ids(
        self,
        topk_ids: torch.Tensor,
        *,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_identity_slots:
            return topk_ids

        remapped = torch.empty_like(topk_ids)
        flat_in = topk_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        flat_out = remapped.reshape(-1)
        cpu_expert_map = None
        if expert_map is not None:
            cpu_expert_map = expert_map.detach().to(device="cpu", dtype=torch.long)

        for index, raw_expert_id in enumerate(flat_in.tolist()):
            if raw_expert_id < 0:
                flat_out[index] = raw_expert_id
                continue
            expert_id = raw_expert_id
            if cpu_expert_map is not None:
                if raw_expert_id >= cpu_expert_map.numel():
                    flat_out[index] = -1
                    continue
                expert_id = int(cpu_expert_map[raw_expert_id].item())
            entry = self.active_experts.get(expert_id)
            if entry is None:
                raise RuntimeError(
                    f"Expert {expert_id} is not resident in the MoE offload cache"
                )
            flat_out[index] = entry.gpu_slot_id
        return remapped

    def ensure_experts_resident_and_remap(
        self,
        topk_ids: torch.Tensor,
        *,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor:
        token_counts = local_expert_token_counts(
            topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
        )
        self.ensure_experts_resident(token_counts)
        return self.remap_topk_ids(topk_ids, expert_map=expert_map)

    def expert_batches_for_counts(
        self,
        expert_token_counts: dict[int, int],
    ) -> list[dict[int, int]]:
        if not expert_token_counts:
            return []
        self._fit_auto_budget_to_available_memory(len(expert_token_counts))
        sorted_counts = sorted(
            expert_token_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if self.use_identity_slots:
            return [dict(sorted_counts)]
        return [
            dict(sorted_counts[index : index + self.active_expert_budget])
            for index in range(0, len(sorted_counts), self.active_expert_budget)
        ]

    def make_wave_tensors(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        local_expert_ids: set[int],
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_identity_slots:
            return topk_ids, topk_weights

        remapped = torch.zeros_like(topk_ids)
        weights = torch.zeros_like(topk_weights)
        flat_in = topk_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        flat_remapped = remapped.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_source_weights = topk_weights.reshape(-1)
        cpu_expert_map = None
        if expert_map is not None:
            cpu_expert_map = expert_map.detach().to(device="cpu", dtype=torch.long)

        for index, raw_expert_id in enumerate(flat_in.tolist()):
            if raw_expert_id < 0:
                continue
            expert_id = raw_expert_id
            if cpu_expert_map is not None:
                if raw_expert_id >= cpu_expert_map.numel():
                    continue
                expert_id = int(cpu_expert_map[raw_expert_id].item())
            if expert_id not in local_expert_ids:
                continue
            entry = self.active_experts.get(expert_id)
            if entry is None:
                raise RuntimeError(
                    f"Expert {expert_id} is not resident in the MoE offload cache"
                )
            flat_remapped[index] = entry.gpu_slot_id
            flat_weights[index] = flat_source_weights[index]

        return remapped, weights


def local_expert_token_counts(
    topk_ids: torch.Tensor,
    *,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
) -> dict[int, int]:
    """Return token counts keyed by local expert id."""
    ids = topk_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    counts: dict[int, int] = {}

    cpu_expert_map = None
    if expert_map is not None:
        cpu_expert_map = expert_map.detach().to(device="cpu", dtype=torch.long)

    for raw_expert_id in ids.tolist():
        if raw_expert_id < 0:
            continue
        expert_id = raw_expert_id
        if cpu_expert_map is not None:
            if raw_expert_id >= cpu_expert_map.numel():
                continue
            expert_id = int(cpu_expert_map[raw_expert_id].item())
            if expert_id < 0:
                continue
        if expert_id >= local_num_experts:
            continue
        counts[expert_id] = counts.get(expert_id, 0) + 1

    return counts
