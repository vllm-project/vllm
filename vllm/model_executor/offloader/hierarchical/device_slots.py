# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed device expert slot pool with async H2D row copies."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass
class SlotMeta:
    expert_id: int = -1
    generation: int = 0
    last_access: float = 0.0
    heat: float = 0.0
    event: torch.Event | None = None
    pending: bool = False


class ExpertSlotPool:
    """Per-layer device slot pool for packed MoE expert rows."""

    def __init__(
        self,
        layer_id: int,
        weight_templates: list[torch.Tensor],
        num_slots: int,
        copy_stream,
        device: torch.device | None = None,
    ):
        assert weight_templates, "need at least one weight tensor"
        self.layer_id = layer_id
        self.num_slots = num_slots
        self.copy_stream = copy_stream
        self.num_experts_full = weight_templates[0].shape[0]

        if device is None:
            device = torch.device(f"{current_platform.device_type}:0")
        self.device = device

        self.slot_weights: list[torch.Tensor] = []
        for tmpl in weight_templates:
            slot_shape = (num_slots, *tmpl.shape[1:])
            buf = torch.empty(slot_shape, dtype=tmpl.dtype, device=device)
            self.slot_weights.append(buf)

        self.slots: list[SlotMeta] = [SlotMeta() for _ in range(num_slots)]
        self.expert_to_slot: dict[int, int] = {}
        self._generation = 0
        self._free: list[int] = list(range(num_slots))

        logger.debug(
            "ExpertSlotPool layer=%d slots=%d full_E=%d device=%s",
            layer_id,
            num_slots,
            self.num_experts_full,
            device,
        )

    def contains(self, expert_id: int) -> bool:
        return expert_id in self.expert_to_slot

    def slot_of(self, expert_id: int) -> int | None:
        return self.expert_to_slot.get(expert_id)

    def _touch(self, slot: SlotMeta) -> None:
        now = time.monotonic()
        slot.last_access = now
        slot.heat = slot.heat * 0.99 + 1.0

    def _alloc_slot(self, protect: set[int] | None = None) -> int:
        protect = protect or set()
        if self._free:
            return self._free.pop()
        candidates = [
            i
            for i in range(self.num_slots)
            if self.slots[i].expert_id not in protect
        ]
        if not candidates:
            raise RuntimeError(
                f"layer {self.layer_id}: cannot allocate a slot without "
                f"evicting a same-batch expert (slots={self.num_slots}, "
                f"protect={sorted(protect)})"
            )
        best = min(
            candidates,
            key=lambda i: (self.slots[i].heat, -self.slots[i].last_access),
        )
        old = self.slots[best].expert_id
        if old in self.expert_to_slot and self.expert_to_slot[old] == best:
            del self.expert_to_slot[old]
        self.slots[best] = SlotMeta()
        return best

    def ensure_from_host_rows(
        self,
        expert_ids: list[int],
        host_rows: dict[int, list[torch.Tensor]],
    ) -> tuple[dict[int, int], list[torch.Event]]:
        """Ensure experts are resident; return remap and events to wait on.

        Experts in ``expert_ids`` are protected from eviction by other
        allocations in this same call (batch-union safety).
        """
        remap: dict[int, int] = {}
        events: list[torch.Event] = []
        protect = {e for e in expert_ids if e >= 0}

        for eid in expert_ids:
            if eid < 0:
                continue
            if eid in self.expert_to_slot:
                sid = self.expert_to_slot[eid]
                self._touch(self.slots[sid])
                remap[eid] = sid
                slot = self.slots[sid]
                if slot.pending and slot.event is not None:
                    events.append(slot.event)
                continue

            rows = host_rows.get(eid)
            if rows is None:
                raise KeyError(
                    f"layer {self.layer_id}: missing host row for expert {eid}"
                )
            sid = self._alloc_slot(protect=protect)
            slot = self.slots[sid]
            event = torch.Event()

            with current_platform.stream(self.copy_stream):
                for dst, src in zip(self.slot_weights, rows):
                    dst[sid].copy_(src, non_blocking=True)
                event.record(self.copy_stream)

            self._generation += 1
            slot.expert_id = eid
            slot.generation = self._generation
            slot.event = event
            slot.pending = True
            self._touch(slot)
            self.expert_to_slot[eid] = sid
            remap[eid] = sid
            events.append(event)
            protect.add(eid)
        return remap, events

    def mark_ready(self, expert_ids: list[int]) -> None:
        for eid in expert_ids:
            sid = self.expert_to_slot.get(eid)
            if sid is not None:
                self.slots[sid].pending = False
