# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_hints.actions import KvHintResult, SetPriority

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

_MAX_CLAIMS_PER_BLOCK = 16


@dataclass(frozen=True)
class _Claim:
    revision: int
    value: int | None
    ttl_seconds: float | None
    expires_at: float


class KVCachePriority:
    """Index eviction priorities while preserving the pool's free list and LRU."""

    def __init__(
        self,
        pool: BlockPool,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pool = pool
        self.clock = clock
        self._claims: dict[int, dict[str, _Claim]] = {}
        self._priorities: dict[int, int] = {}
        self._expiry: list[tuple[float, int, str, int]] = []
        self._free_order: dict[int, int] | None = None
        self._candidates: list[tuple[int, int, int]] = []
        self._left = self._right = 0

    def apply(
        self, action: SetPriority, blocks: Iterable[KVCacheBlock]
    ) -> KvHintResult:
        now = self.clock()
        self._expire(now)
        selected = dict.fromkeys(
            b.block_id
            for b in blocks
            if b.pool is self.pool and not b.is_null and b.block_hash is not None
        )
        if not selected:
            return KvHintResult("rejected", reason="No cached local G1 copies")
        changed = []
        for block_id in selected:
            claims = self._claims.get(block_id, {})
            previous = claims.get(action.claim_id)
            if previous is not None:
                if action.revision < previous.revision:
                    continue
                if action.revision == previous.revision:
                    if (action.value, action.ttl_seconds) != (
                        previous.value,
                        previous.ttl_seconds,
                    ):
                        return KvHintResult(
                            "rejected", reason="Conflicting payload for claim revision"
                        )
                    continue
            elif len(claims) >= _MAX_CLAIMS_PER_BLOCK:
                return KvHintResult("rejected", reason="Per-block claim limit reached")
            changed.append(block_id)

        expires_at = now + (action.ttl_seconds or 0)
        claim = _Claim(action.revision, action.value, action.ttl_seconds, expires_at)
        for block_id in changed:
            self._claims.setdefault(block_id, {})[action.claim_id] = claim
            if action.value is not None:
                heapq.heappush(
                    self._expiry,
                    (expires_at, block_id, action.claim_id, action.revision),
                )
            self._refresh(block_id, now)
        self._compact()
        return KvHintResult("applied" if changed else "duplicate", tuple(changed))

    def _refresh(self, block_id: int, now: float) -> None:
        value = max(
            (
                c.value
                for c in self._claims.get(block_id, {}).values()
                if c.value is not None and c.expires_at > now
            ),
            default=0,
        )
        if value == self._priorities.get(block_id, 0):
            return
        if value:
            self._priorities[block_id] = value
        else:
            self._priorities.pop(block_id, None)
        if self._free_order is not None and block_id in self._free_order:
            heapq.heappush(
                self._candidates, (value, self._free_order[block_id], block_id)
            )

    def _expire(self, now: float) -> None:
        while self._expiry and self._expiry[0][0] <= now:
            _, block_id, claim_id, revision = heapq.heappop(self._expiry)
            claim = self._claims.get(block_id, {}).get(claim_id)
            if claim is not None and claim.revision == revision:
                self._refresh(block_id, now)
        if not self._priorities:
            self._free_order = None
            self._candidates.clear()

    def take_blocks(self, count: int) -> list[KVCacheBlock]:
        if count == 0:
            return []
        self._expire(self.clock())
        queue = self.pool.free_block_queue
        if not self._priorities:
            return queue.popleft_n(count)
        if self._free_order is None:
            self._free_order = {}
            self.on_free(queue.iter_blocks_after(None))
        result: list[KVCacheBlock] = []
        while len(result) < count:
            priority, order, block_id = heapq.heappop(self._candidates)
            if self._free_order.get(block_id) != order or priority != (
                self._priorities.get(block_id, 0)
            ):
                continue
            block = self.pool.blocks[block_id]
            assert block.ref_cnt == 0 and not block.is_null
            queue.remove(block)
            self.on_touch(block)
            result.append(block)
        self._compact()
        return result

    def on_free(self, blocks: Iterable[KVCacheBlock], *, prepend: bool = False) -> None:
        if self._free_order is None:
            return
        if prepend:
            blocks = reversed(list(blocks))
        for block in blocks:
            if prepend:
                self._left -= 1
                order = self._left
            else:
                self._right += 1
                order = self._right
            self._free_order[block.block_id] = order
            heapq.heappush(
                self._candidates,
                (self._priorities.get(block.block_id, 0), order, block.block_id),
            )
        self._compact()

    def on_touch(self, block: KVCacheBlock) -> None:
        if self._free_order is not None:
            self._free_order.pop(block.block_id, None)

    def forget(self, block: KVCacheBlock) -> None:
        if self._claims.pop(block.block_id, None) is None:
            return
        self._refresh(block.block_id, self.clock())
        self._compact()

    def reset(self) -> None:
        self._claims.clear()
        self._priorities.clear()
        self._expiry.clear()
        self._free_order = None
        self._candidates.clear()

    def _compact(self) -> None:
        # Bound lazy heap entries after repeated updates, hits, and reuse.
        if self._free_order is not None and len(self._candidates) > (
            2 * len(self._free_order) + 64
        ):
            self._candidates = [
                (self._priorities.get(block_id, 0), order, block_id)
                for block_id, order in self._free_order.items()
            ]
            heapq.heapify(self._candidates)
        if len(self._expiry) > 2 * len(self._claims) * _MAX_CLAIMS_PER_BLOCK + 64:
            self._expiry = [
                (claim.expires_at, block_id, claim_id, claim.revision)
                for block_id, claims in self._claims.items()
                for claim_id, claim in claims.items()
                if claim.value is not None
            ]
            heapq.heapify(self._expiry)


def apply_g1_priority(
    manager: KVCacheManager, action: SetPriority, request: Request
) -> KvHintResult:
    pool = manager.block_pool
    if not manager.enable_caching:
        return KvHintResult("unsupported", reason="Prefix caching is disabled")
    if pool.kv_cache_priority is None:
        pool.kv_cache_priority = KVCachePriority(pool)
    return pool.kv_cache_priority.apply(
        action,
        (
            block
            for group in manager.get_blocks(request.request_id).blocks
            for block in group
        ),
    )
