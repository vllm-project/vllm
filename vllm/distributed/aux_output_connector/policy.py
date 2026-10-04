# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy interfaces for routed-expert block retention and eviction."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal


class ExpertCachePolicy(ABC):
    """Track reference changes and select blocks for eviction.

    This extension point lets deployments align expert retention with a KV
    connector's cache state and eviction policy. The goal is to keep expert
    data available for as long as the corresponding KV can still be reused,
    reducing KV cache hits whose expert data has already been evicted.
    Connector state is not synchronized automatically; alignment depends on
    the policy implementation and the information supplied to it.

    Each store owns one policy instance. The store manages block data and
    reference counts; the policy maintains its own bookkeeping and recommends
    which eligible blocks to evict.

    Policy methods run as part of the store operation, which waits for them to
    return. With BackgroundBlockObjectStore, they run on its writer thread.
    Keep these methods lightweight and do not call back into the store: nested
    reads or writes can deadlock or interfere with an unfinished update.
    Exceptions propagate to the caller; earlier updates may already be applied.

    The optional context accepts caller-defined metadata. The built-in store
    currently calls these methods without context.
    """

    def on_reference_change(
        self,
        key: str,
        old_count: int,
        new_count: int,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Handle a reference increment or decrement after it has been applied.

        old_count and new_count are the counts before and after the change.
        A key may acquire references before its data is stored. A new count
        of zero removes reference protection but does not delete the data.
        """
        return None

    @abstractmethod
    def select_victims(
        self, candidates: Iterable[str], count: int
    ) -> Sequence[str] | None:
        """Select exactly count distinct keys, or return None if unable to do so.

        candidates yields eligible keys in LRU order, oldest first. Referenced
        keys and keys protected by the current write are excluded. Consume the
        iterable during this call; do not retain it for later use.

        Returning keys does not delete them. The store validates the complete
        selection before eviction and calls on_remove after deleting entries.
        """

    def on_remove(
        self,
        keys: Sequence[str],
        *,
        reason: Literal["eviction", "rollback"],
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Update policy state after the store has removed the given keys.

        reason is "eviction" for capacity-driven removal or "rollback" when
        cleaning up entries from a failed insertion. Rolled-back entries may
        never have received data. Neither event changes reference counts.
        """
        return None
