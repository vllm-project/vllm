# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic admission and resident slots for data-parallel MoE LoRA."""

import msgspec

from vllm.lora.request import LoRARequest


class LoRASlot(msgspec.Struct):
    request: LoRARequest
    generation: int


def adapter_identity(request: LoRARequest) -> tuple[int, str, bool]:
    return request.lora_int_id, request.lora_path, request.is_3d_lora_weight


class DPLoRASlots:
    def __init__(self, capacity: int):
        self.slots: list[LoRASlot | None] = [None] * capacity
        self.adapters: dict[int, LoRASlot] = {}
        self.pinned: set[int] = set()
        self.last_used: dict[int, int] = {}
        self.clock = 0
        self.generation = 0

    def validate(self, request: LoRARequest) -> None:
        if request.is_3d_lora_weight:
            raise ValueError("DP LoRA synchronization currently requires 2D adapters")
        previous = self.adapters.get(request.lora_int_id)
        if previous and adapter_identity(previous.request) != adapter_identity(request):
            raise ValueError("A LoRA ID must identify the same adapter on every rank")

    def acquire(self, requests: list[LoRARequest], protected: set[int]) -> None:
        protected = protected | self.pinned | {r.lora_int_id for r in requests}
        if len(protected) > len(self.slots):
            raise ValueError("The global LoRA working set exceeds max_loras")
        self.clock += 1
        for request in requests:
            self.validate(request)
            adapter_id = request.lora_int_id
            if adapter_id not in self.adapters:
                self.generation += 1
                self.adapters[adapter_id] = LoRASlot(
                    msgspec.structs.replace(request, load_inplace=False),
                    self.generation,
                )
            slot = self.adapters[adapter_id]
            index = next(
                (
                    i
                    for i, old in enumerate(self.slots)
                    if old is not None and old.request.lora_int_id == adapter_id
                ),
                None,
            )
            if index is None:
                available = [
                    i
                    for i, old in enumerate(self.slots)
                    if old is None or old.request.lora_int_id not in protected
                ]
                if not available:
                    raise ValueError("All LoRA slots are in use or pinned")

                def eviction_key(i: int) -> tuple[int, int]:
                    old = self.slots[i]
                    age = (
                        -1
                        if old is None
                        else self.last_used.get(old.request.lora_int_id, -1)
                    )
                    return age, i

                index = min(available, key=eviction_key)
            self.slots[index] = slot
            self.last_used[adapter_id] = self.clock

    def update(
        self, kind: str, payload: bytes, referenced: set[int], running: set[int]
    ) -> bool:
        if kind == "add":
            request = msgspec.msgpack.decode(payload, type=LoRARequest)
            adapter_id = request.lora_int_id
            if request.load_inplace:
                if adapter_id in referenced:
                    raise ValueError("Cannot replace a LoRA with unfinished requests")
                if request.is_3d_lora_weight:
                    raise ValueError("DP LoRA synchronization requires 2D adapters")
                self.generation += 1
                self.adapters[adapter_id] = LoRASlot(
                    msgspec.structs.replace(request, load_inplace=False),
                    self.generation,
                )
            self.acquire([request], running)
            return True
        adapter_id = msgspec.msgpack.decode(payload, type=int)
        if kind == "remove":
            if adapter_id in referenced:
                raise ValueError("Cannot remove a LoRA with unfinished requests")
            existed = adapter_id in self.adapters
            self.adapters.pop(adapter_id, None)
            self.pinned.discard(adapter_id)
            for i, slot in enumerate(self.slots):
                if slot is not None and slot.request.lora_int_id == adapter_id:
                    self.slots[i] = None
            return existed
        if kind == "pin":
            if not any(s and s.request.lora_int_id == adapter_id for s in self.slots):
                raise ValueError("Load the LoRA before pinning it")
            self.pinned.add(adapter_id)
            return True
        raise ValueError(f"Unknown LoRA update: {kind}")

    def payload(self) -> bytes:
        return msgspec.msgpack.encode(self.slots)


class LoRAUsage(msgspec.Struct):
    request: LoRARequest
    running: bool
    ready: bool
    priority: int
    arrival_time: float
    request_id: str
    client_index: int
