# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import random
import time
from collections import deque
from collections.abc import Container, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

from vllm.v1.metrics.stats import KVCacheEvictionEvent
from vllm.v1.request import Request, RequestStatus

KV_CACHE_BLOCK_STATES_INTERVAL_S = 10.0

# Marks for blocks owned by a request or pending a deferred free. A block with
# several owners keeps the highest mark.
_PENDING_FREE = 1
_WAITING_HOLD = 2
_HELD_FOR_SEND = 3
_LOADING_KV = 4
_ACTIVE_PRIVATE = 5
_ACTIVE_SHARED = 6


class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling."""

    def __init__(self, sample_rate: float = 0.01):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """Clear all state on cache reset."""
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events


def count_kv_cache_blocks_by_state(
    block_pool: "BlockPool",
    managers: Sequence["SingleTypeKVCacheManager"],
    requests: Mapping[str, Request],
    finished_recving_req_ids: Container[str],
    deferred_frees: Iterable[tuple[int, Sequence["KVCacheBlock"]]],
) -> dict[str, int]:
    """Count the usable blocks of a pool by exclusive state.

    Every block except the null block gets exactly one state, so the counts
    sum to `block_pool.num_gpu_blocks - 1`. A block owned by several
    requests takes the highest-priority state, in the order of
    `KV_CACHE_BLOCK_STATES`. In-use blocks without an owner request or a
    deferred free (e.g. connector or copy-on-write pins) are `unattributed`.

    Args:
        block_pool: The pool to count. Blocks of other pools are skipped.
        managers: Single-type managers whose `req_to_blocks` own blocks.
        requests: Scheduler requests by ID.
        finished_recving_req_ids: Requests whose async KV load finished.
        deferred_frees: `(step_id, blocks)` pairs withheld from the pool until
            the step finishes.

    Returns:
        Block count for every state in `KV_CACHE_BLOCK_STATES`.

    """
    marks = bytearray(block_pool.num_gpu_blocks)
    for manager in managers:
        if manager.block_pool is not block_pool:
            continue
        residency_changes = getattr(manager, "residency_changes", None)
        for req_id, blocks in manager.req_to_blocks.items():
            request = requests.get(req_id)
            if request is None:
                continue
            if residency_changes and req_id in residency_changes:
                blocks = manager.get_residency_row(req_id)  # type: ignore[attr-defined]
            status = request.status
            if status == RequestStatus.RUNNING:
                for block in blocks:
                    if block.ref_cnt == 0:
                        continue
                    mark = _ACTIVE_SHARED if block.ref_cnt > 1 else _ACTIVE_PRIVATE
                    if marks[block.block_id] < mark:
                        marks[block.block_id] = mark
                continue
            if RequestStatus.is_finished(status):
                mark = _HELD_FOR_SEND
            elif (
                status == RequestStatus.WAITING_FOR_REMOTE_KVS
                and req_id not in finished_recving_req_ids
            ):
                mark = _LOADING_KV
            else:
                mark = _WAITING_HOLD
            for block in blocks:
                if block.ref_cnt > 0 and marks[block.block_id] < mark:
                    marks[block.block_id] = mark
    for _, freed_blocks in deferred_frees:
        for block in freed_blocks:
            if (
                block.pool is block_pool
                and block.ref_cnt > 0
                and not marks[block.block_id]
            ):
                marks[block.block_id] = _PENDING_FREE
    marks[block_pool.null_block.block_id] = 0

    marked = {
        "active_shared": marks.count(_ACTIVE_SHARED),
        "active_private": marks.count(_ACTIVE_PRIVATE),
        "loading_kv": marks.count(_LOADING_KV),
        "held_for_send": marks.count(_HELD_FOR_SEND),
        "waiting_hold": marks.count(_WAITING_HOLD),
        "pending_free": marks.count(_PENDING_FREE),
    }
    num_free = block_pool.get_num_free_blocks()
    num_cached_free = (
        sum(
            1
            for block in block_pool.blocks
            if block.ref_cnt == 0 and block.block_hash is not None
        )
        if num_free > 0 and block_pool.cached_block_hash_to_block
        else 0
    )
    return {
        **marked,
        "unattributed": block_pool.num_gpu_blocks - 1 - num_free - sum(marked.values()),
        "cached_evictable": num_cached_free,
        "empty_free": num_free - num_cached_free,
    }
