# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Revisioned copy-lifetime claims feeding the existing retention queue.

Claim/TTL semantics follow vllm-project/vllm#59714 (7966f717dd4e). Eviction
continues through RetentionFreeBlockQueue, shared with logical declarations.
"""

import heapq
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_hints.actions import SetPriority
from vllm.v1.kv_hints.control import ControlError


@dataclass(frozen=True)
class _Claim:
    revision: int
    value: int | None
    ttl: float | None
    expires_at: float
    policy_name: str | None
    owners: tuple[str, ...]


class KVCachePriority:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_claims_per_copy: int = 16,
    ):
        self.clock = clock
        self.max_claims_per_copy = max_claims_per_copy
        self._claims: dict[int, dict[str, _Claim]] = {}
        self._expiry: list[tuple[float, int, int, str]] = []
        self._priorities: dict[int, int] = {}
        self._changed: set[int] = set()
        self._sequence = 0

    def apply(
        self,
        action: SetPriority,
        blocks: Iterable[KVCacheBlock],
        *,
        policy_name: str | None = None,
        owners: tuple[str, ...] = (),
    ) -> tuple[int, ...]:
        self.expire()
        now = self.clock()
        selected = {
            b.block_id for b in blocks if not b.is_null and b.block_hash is not None
        }
        if not selected:
            raise ControlError("missing", "No cached local G1 copies")
        changed = []
        for block_id in selected:
            claims = self._claims.get(block_id, {})
            old = claims.get(action.claim_id)
            if old is not None:
                if policy_name is not None and old.policy_name != policy_name:
                    raise ControlError(
                        "busy", "A policy cannot replace another caller's claim"
                    )
                if action.revision < old.revision:
                    continue
                if action.revision == old.revision:
                    if (action.value, action.ttl_seconds) != (old.value, old.ttl):
                        raise ControlError(
                            "conflict", "Conflicting payload for claim revision"
                        )
                    continue
            elif len(claims) >= self.max_claims_per_copy:
                raise ControlError("capacity", "Per-copy claim limit reached")
            changed.append(block_id)
        claim = _Claim(
            action.revision,
            action.value,
            action.ttl_seconds,
            now + (action.ttl_seconds or 0),
            policy_name,
            owners,
        )
        for block_id in changed:
            self._claims.setdefault(block_id, {})[action.claim_id] = claim
            if action.value is not None:
                self._sequence += 1
                heapq.heappush(
                    self._expiry,
                    (claim.expires_at, self._sequence, block_id, action.claim_id),
                )
            self._refresh(block_id, now)
        self._compact()
        return tuple(sorted(changed))

    def priority(self, block: KVCacheBlock) -> int | None:
        return self._priorities.get(block.block_id)

    def has_explicit(self, blocks: Iterable[KVCacheBlock]) -> bool:
        now = self.clock()
        return any(
            c.policy_name is None and c.value is not None and c.expires_at > now
            for block in blocks
            for c in self._claims.get(block.block_id, {}).values()
        )

    def _refresh(self, block_id: int, now: float) -> None:
        values = [
            c.value
            for c in self._claims.get(block_id, {}).values()
            if c.value is not None and c.expires_at > now
        ]
        old = self._priorities.get(block_id)
        if values:
            self._priorities[block_id] = max(values)
        else:
            self._priorities.pop(block_id, None)
        if old != self._priorities.get(block_id):
            self._changed.add(block_id)

    def expire(self) -> None:
        now = self.clock()
        while self._expiry and self._expiry[0][0] <= now:
            _, _, block_id, _ = heapq.heappop(self._expiry)
            self._refresh(block_id, now)
        self._compact()

    def take_changes(self) -> set[int]:
        self.expire()
        changed, self._changed = self._changed, set()
        return changed

    def forget(self, block: KVCacheBlock) -> None:
        self._claims.pop(block.block_id, None)
        self._priorities.pop(block.block_id, None)
        self._changed.discard(block.block_id)
        self._compact()

    def finish_owners(self, owners: tuple[str, ...]) -> None:
        for block_id, claims in self._claims.items():
            for identity, claim in tuple(claims.items()):
                if claim.policy_name is not None and set(claim.owners).intersection(
                    owners
                ):
                    # Retain the revision tombstone until this copy is forgotten.
                    claims[identity] = _Claim(
                        claim.revision, None, None, 0, claim.policy_name, claim.owners
                    )
            self._refresh(block_id, self.clock())

    def _compact(self) -> None:
        bound = max(64, 2 * len(self._claims) * self.max_claims_per_copy)
        if len(self._expiry) > bound:
            self._expiry = []
            for block_id, claims in self._claims.items():
                for identity, claim in claims.items():
                    if claim.value is not None:
                        self._sequence += 1
                        self._expiry.append(
                            (claim.expires_at, self._sequence, block_id, identity)
                        )
            heapq.heapify(self._expiry)
