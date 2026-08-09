# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PromotionPolicy: decides whether a landed secondary->primary promotion
should be protected from eviction until it's actually read.

Mirrors the CachePolicy split (vllm/v1/kv_offload/cpu/policies/base.py):
the policy owns the decision and its own bookkeeping; TieringOffloadingManager
owns the mechanics (ref_cnt manipulation via the primary tier) and calls the
policy at the same lifecycle points it previously handled inline.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


class PromotionPolicy(ABC):
    """Decides which landed promotions get protected from eviction, and
    tracks which blocks are currently protected."""

    @abstractmethod
    def on_promotion_landed(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        request_finished: bool,
    ) -> bool:
        """A promotion job for `keys` (owned by req_context) just completed
        successfully. Return True if the manager should pin these blocks
        (hold them non-evictable) until consumed or released."""

    @abstractmethod
    def on_consumed(self, keys: Collection[OffloadKey]) -> Collection[OffloadKey]:
        """`keys` are about to be read (prepare_load). Return the subset
        that were pinned, so the manager can release their hold."""

    @abstractmethod
    def on_request_finished(self, req_id: str) -> Collection[OffloadKey]:
        """`req_id` finished. Return any still-pinned keys it owned so the
        manager can release their holds."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracked state (used by reset_cache())."""


class AlwaysPromotePolicy(PromotionPolicy):
    """Never pins anything: every promotion lands as a normal evictable
    block immediately, matching behavior before this policy existed."""

    def on_promotion_landed(self, keys, req_context, request_finished) -> bool:
        return False

    def on_consumed(self, keys) -> Collection[OffloadKey]:
        return ()

    def on_request_finished(self, req_id) -> Collection[OffloadKey]:
        return ()

    def reset(self) -> None:
        return


class PinUnreadPromotionsPolicy(PromotionPolicy):
    """Pins landed-but-unread promotions, capped by number of distinct
    pinned requests (not blocks, so a request's batch is never pinned
    partially). Unbounded by default."""

    def __init__(self, max_pinned: int | None = None):
        self._max_pinned = max_pinned
        self._owner: dict[OffloadKey, str] = {}
        self._by_req: dict[str, set[OffloadKey]] = {}

    def on_promotion_landed(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        request_finished: bool,
    ) -> bool:
        if request_finished:
            return False
        req_id = req_context.req_id
        if (
            self._max_pinned is not None
            and req_id not in self._by_req
            and len(self._by_req) >= self._max_pinned
        ):
            return False
        for key in keys:
            self._owner[key] = req_id
        self._by_req.setdefault(req_id, set()).update(keys)
        return True

    def on_consumed(self, keys: Collection[OffloadKey]) -> Collection[OffloadKey]:
        held = [key for key in keys if key in self._owner]
        for key in held:
            req_id = self._owner.pop(key)
            owned = self._by_req[req_id]
            owned.discard(key)
            if not owned:
                del self._by_req[req_id]
        return held

    def on_request_finished(self, req_id: str) -> Collection[OffloadKey]:
        abandoned = self._by_req.pop(req_id, None)
        if not abandoned:
            return ()
        for key in abandoned:
            del self._owner[key]
        return abandoned

    def reset(self) -> None:
        self._owner.clear()
        self._by_req.clear()
