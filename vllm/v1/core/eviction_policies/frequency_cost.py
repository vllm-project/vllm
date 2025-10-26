# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import math
import time

from vllm.v1.core.kv_cache_utils import KVCacheBlock


class FrequencyCostEvictionPolicy:
    """Min-heap policy over cached-free blocks by retention score.

    Implementation notes:
    - Uses lazy deletion with an `entry_finder` dict to avoid in-place heap edits.
    - Scores are computed lazily when a block becomes cached-free.
    - `block_size` is provided once at init; not stored on each block.
    - This class tracks only blocks that are both free (ref_cnt==0) and cached
      (i.e., have a non-None block_hash).
    """

    def __init__(
        self,
        block_size: int,
        alpha: float = 2.0,
        time_decay_factor: float = 0.0,
        min_time_window: float = 1.0,
    ) -> None:
        self.block_size = block_size
        self.alpha = alpha
        self.time_decay_factor = time_decay_factor
        self.min_time_window = min_time_window

        # Heap entries: (score, counter, block_id)
        self._heap: list[tuple[float, int, int]] = []
        self._entry_finder: dict[int, tuple[float, int, int]] = {}
        self._counter = 0

    def _score(self, block: KVCacheBlock) -> float:
        # If the block was never accessed through prefix hits, treat as lowest value.
        first_ts = getattr(block, "first_access_ts", None)
        access_count = getattr(block, "access_count", 0) or 0
        if first_ts is None:
            return 0.0
        now = time.monotonic()
        dt = max(self.min_time_window, now - first_ts)
        if self.time_decay_factor > 0.0:
            w = math.exp(-self.time_decay_factor * dt)
            freq = (access_count * w) / dt
        else:
            freq = access_count / dt
        cost = float(self.block_size) ** self.alpha
        return min(freq * cost, 1e15)

    def _add(self, block: KVCacheBlock) -> None:
        # Only track cached-free blocks
        if block.ref_cnt != 0 or block.block_hash is None:
            return
        score = self._score(block)
        self._counter += 1
        entry = (score, self._counter, block.block_id)
        self._entry_finder[block.block_id] = entry
        heapq.heappush(self._heap, entry)

    def on_block_access(self, block: KVCacheBlock) -> None:
        # Minimal tracking on access for frequency stats
        first_ts = getattr(block, "first_access_ts", None)
        if first_ts is None:
            block.first_access_ts = time.monotonic()
        block.access_count = (getattr(block, "access_count", 0) or 0) + 1

    def on_block_release(self, block: KVCacheBlock) -> None:
        # Block became cached-free
        self._add(block)

    def get_eviction_candidates(self, num_blocks: int) -> list[int]:
        out: list[int] = []
        while self._heap and len(out) < num_blocks:
            score, counter, block_id = heapq.heappop(self._heap)
            if self._entry_finder.get(block_id) == (score, counter, block_id):
                self._entry_finder.pop(block_id, None)
                out.append(block_id)
        return out

    def remove_block(self, block: KVCacheBlock) -> None:
        # Lazy deletion: ensure future pops skip this block
        self._entry_finder.pop(block.block_id, None)

    def reset(self) -> None:
        self._heap.clear()
        self._entry_finder.clear()

    @property
    def name(self) -> str:
        return f"FrequencyCost(alpha={self.alpha})"
