# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import math
from collections import OrderedDict, deque
from dataclasses import dataclass


class _LayerLocalLRU:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("LRU cache capacity must be positive.")
        self.capacity = capacity
        # block_id -> (slot_id, pinned)
        self.entries: OrderedDict[int, tuple[int, bool]] = OrderedDict()
        self.slot_to_block: dict[int, int] = {}
        self.free_slots = deque(range(capacity))

    def get(self, block_id: int, hot_score: float = 0.0) -> tuple[int, bool]:
        del hot_score
        if block_id in self.entries:
            slot_id, pinned = self.entries.pop(block_id)
            self.entries[block_id] = (slot_id, pinned)
            return slot_id, True

        if self.free_slots:
            slot_id = self.free_slots.popleft()
        else:
            slot_id = self._evict_one()

        self.entries[block_id] = (slot_id, False)
        self.slot_to_block[slot_id] = block_id
        return slot_id, False

    def pin(self, block_id: int) -> None:
        if block_id not in self.entries:
            raise KeyError(f"Cannot pin missing block {block_id}.")
        slot_id, _ = self.entries.pop(block_id)
        self.entries[block_id] = (slot_id, True)

    def unpin(self, block_id: int) -> None:
        if block_id not in self.entries:
            return
        slot_id, _ = self.entries.pop(block_id)
        self.entries[block_id] = (slot_id, False)

    def _evict_one(self) -> int:
        for block_id in list(self.entries.keys()):
            slot_id, pinned = self.entries[block_id]
            if pinned:
                continue
            self.entries.pop(block_id)
            self.slot_to_block.pop(slot_id, None)
            return slot_id
        raise RuntimeError("No evictable GPU cache block is available.")


@dataclass
class _HotEntry:
    slot_id: int
    hot_score: float
    version: int
    pinned: bool
    timer: int


class _LayerLocalHotLRU:
    def __init__(self, capacity: int, decay_factor: float, window_size: int):
        if capacity <= 0:
            raise ValueError("Hot cache capacity must be positive.")
        self.capacity = capacity
        self.decay_factor = decay_factor
        self.window_size = window_size

        self.entries: dict[int, _HotEntry] = {}
        self.slot_to_block: dict[int, int] = {}
        self.free_slots = deque(range(capacity))
        # min-heap: (hot_score, version, block_id, timer)
        self.heap: list[tuple[float, int, int, int]] = []
        self.global_version = 0
        self.timer = 0

    def add_timer(self) -> None:
        self.timer += 1

    @staticmethod
    def _scale_hot_score(hot_score: float) -> float:
        if math.isinf(hot_score):
            hot_score = 1e6 if hot_score > 0 else -1e6
        sign = 1.0 if hot_score >= 0 else -1.0
        return sign * math.tanh(math.log1p(abs(hot_score)))

    def get(self, block_id: int, hot_score: float = 0.0) -> tuple[int, bool]:
        scaled = self._scale_hot_score(hot_score)
        if block_id in self.entries:
            entry = self.entries[block_id]
            time_weight = max(0.0, 1.0 - (self.timer - entry.timer) / self.window_size)
            new_score = (
                entry.hot_score * self.decay_factor
                + scaled * (1.0 - self.decay_factor) * time_weight
            )
            self.global_version += 1
            entry.hot_score = new_score
            entry.version = self.global_version
            entry.timer = self.timer
            heapq.heappush(
                self.heap, (entry.hot_score, entry.version, block_id, self.timer)
            )
            return entry.slot_id, True

        if self.free_slots:
            slot_id = self.free_slots.popleft()
        else:
            slot_id = self._evict_one()

        self.global_version += 1
        entry = _HotEntry(
            slot_id=slot_id,
            hot_score=scaled,
            version=self.global_version,
            pinned=False,
            timer=self.timer,
        )
        self.entries[block_id] = entry
        self.slot_to_block[slot_id] = block_id
        heapq.heappush(self.heap, (entry.hot_score, entry.version, block_id, self.timer))
        return slot_id, False

    def pin(self, block_id: int) -> None:
        if block_id not in self.entries:
            raise KeyError(f"Cannot pin missing block {block_id}.")
        self.entries[block_id].pinned = True

    def unpin(self, block_id: int) -> None:
        if block_id not in self.entries:
            return
        self.entries[block_id].pinned = False

    def _evict_one(self) -> int:
        while self.heap:
            score, version, block_id, _timer = heapq.heappop(self.heap)
            del score
            if block_id not in self.entries:
                continue
            entry = self.entries[block_id]
            if entry.version != version:
                continue
            if entry.pinned:
                continue
            slot_id = entry.slot_id
            del self.entries[block_id]
            self.slot_to_block.pop(slot_id, None)
            return slot_id
        raise RuntimeError("No evictable GPU cache block is available.")


class LayerWiseLRUCache:
    def __init__(self, num_layers: int, capacity_per_layer: int):
        self.layers = [_LayerLocalLRU(capacity_per_layer) for _ in range(num_layers)]

    def add_timer(self) -> None:
        return

    def get(self, layer_idx: int, block_id: int, hot_score: float = 0.0) -> tuple[int, bool]:
        return self.layers[layer_idx].get(block_id, hot_score=hot_score)

    def pin_block(self, layer_idx: int, block_id: int) -> None:
        self.layers[layer_idx].pin(block_id)

    def unpin_block(self, layer_idx: int, block_id: int) -> None:
        self.layers[layer_idx].unpin(block_id)


class LRUCache(LayerWiseLRUCache):
    """Fallback policy for per-layer hot caches.

    The current sparse hot-cache path uses per-layer GPU KV tensors, so this
    behaves equivalently to layer-wise LRU.
    """


class LRUWithHotCache:
    def __init__(
        self,
        num_layers: int,
        capacity_per_layer: int,
        decay_factor: float = 0.44,
        window_size: int = 110,
    ) -> None:
        self.layers = [
            _LayerLocalHotLRU(capacity_per_layer, decay_factor, window_size)
            for _ in range(num_layers)
        ]

    def add_timer(self) -> None:
        for layer in self.layers:
            layer.add_timer()

    def get(self, layer_idx: int, block_id: int, hot_score: float = 0.0) -> tuple[int, bool]:
        return self.layers[layer_idx].get(block_id, hot_score=hot_score)

    def pin_block(self, layer_idx: int, block_id: int) -> None:
        self.layers[layer_idx].pin(block_id)

    def unpin_block(self, layer_idx: int, block_id: int) -> None:
        self.layers[layer_idx].unpin(block_id)

