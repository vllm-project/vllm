# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reuse-frequency gating for CPU KV-cache offload stores.

FilterReusedOffloadingManager — OffloadingManager decorator that skips
    storing blocks that have not yet been seen enough times.
"""

from collections import OrderedDict
from collections.abc import Iterable, Sequence

from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)


class FilterReusedOffloadingManager(OffloadingManager):
    """An :class:`OffloadingManager` decorator that skips storing blocks
    whose reuse frequency is below *store_threshold*.

    All methods are delegated to the *backing* manager.  Two methods are
    intercepted:

    * ``prepare_store`` — filters out keys that have not yet
    * ``lookup`` — records the visited key in an internal LRU
      counter, then delegates to the backing manager.
      crossed the threshold *before* calling the backing
      ``prepare_store``.

    Args:
        backing: The underlying ``OffloadingManager`` to delegate to.
        store_threshold: A block must be seen at least this many times in
            ``lookup()`` before it is eligible for offloading.  Must be >= 2
            (a value of 1 would be equivalent to no filtering).
        max_tracker_size: Maximum entries in the internal tracker's LRU table.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ):
        if store_threshold < 2:
            raise ValueError(
                "FilterReusedOffloadingManager store_threshold must be >= 2, "
                f"got {store_threshold}"
            )
        if max_tracker_size < 1:
            raise ValueError(
                "FilterReusedOffloadingManager max_tracker_size must be >= 1, "
                f"got {max_tracker_size}"
            )
        self._backing = backing
        self.store_threshold = store_threshold
        self.max_tracker_size = max_tracker_size
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] = OrderedDict()

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """Record the key, then delegate lookup to backing manager."""
        if key in self.counts:
            self.counts.move_to_end(key)
            self.counts[key] += 1
        else:
            if len(self.counts) >= self.max_tracker_size:
                self.counts.popitem(last=False)  # evict LRU
            self.counts[key] = 1
        return self._backing.lookup(key, req_context)

    def prepare_store(
        self, keys: Sequence[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """Filter out blocks below threshold, then delegate to backing.

        Filtering is evaluated *before* calling the backing manager's
        ``prepare_store`` so that blocks that would be skipped do not
        consume any CPU offload capacity.
        """
        eligible = [
            key for key in keys if self.counts.get(key, 0) >= self.store_threshold
        ]

        # Passing an empty list is intentional and safe — CPUOffloadingManager
        # handles it correctly, returning a PrepareStoreOutput with empty lists.
        # Delegate to the backing manager with only the eligible keys.
        return self._backing.prepare_store(eligible, req_context)

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def prepare_load(
        self, keys: Sequence[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        return self._backing.prepare_load(keys, req_context)

    def touch(self, keys: Sequence[OffloadKey]) -> None:
        return self._backing.touch(keys)

    def complete_load(self, keys: Iterable[OffloadKey]) -> None:
        return self._backing.complete_load(keys)

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True) -> None:
        return self._backing.complete_store(keys, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()
