# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery units and the prefix graph that relates them.

A recovery unit is a recurrent state that is cached in every Mamba group, so
a request can resume from it: a block-boundary state (cached by align mode or
published at a decode boundary) or a frontier (the exact end of a finished
request, inside a block). A unit's parent is the nearest unit before it on
its prefix path, the one its request resumed from or cached before it.
"""

from dataclasses import dataclass, field
from typing import Protocol

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(slots=True, eq=False)
class RecoveryUnit:
    key: BlockHash
    num_tokens: int
    frontier: bool
    parent: BlockHash | None = None
    children: set[BlockHash] = field(default_factory=set)
    hits: int = 0
    linked: bool = False
    # Eviction bookkeeping, owned by GreedyDualSizeEviction.
    path: tuple[BlockHash, ...] = ()
    stamp: float = 0.0
    fallback: BlockHash | None = None
    size: int = 1
    dirty: bool = True
    dead: bool = False
    version: int = -1
    # Leaf-last tiers: the tier of the last score (-1 before any) and the
    # time of the last use.
    tier: int = -1
    used_at: float = 0.0

    @property
    def is_branch_point(self) -> bool:
        """Two cached continuations diverge here, or a request other than its
        own session's next turn has resumed here."""
        return len(self.children) > 1 or self.hits > 1


class UnitObserver(Protocol):
    def unit_added(self, unit: RecoveryUnit) -> None: ...

    def unit_recached(self, unit: RecoveryUnit) -> None: ...

    def unit_removed(self, unit: RecoveryUnit) -> None: ...

    def unit_used(self, unit: RecoveryUnit) -> None: ...

    def unit_linked(self, unit: RecoveryUnit) -> None: ...


class CheckpointGraph:
    """The recovery units by key, their prefix-graph links, the published
    session leaves and the unit each running request resumed from."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.units: dict[BlockHash, RecoveryUnit] = {}
        # Request -> the last unit on its path: the one it resumed from, then
        # each unit it caches.
        self.last: dict[str, BlockHash | None] = {}
        self.leaves: set[BlockHash] = set()
        self.resumed: dict[str, BlockHash | None] = {}
        self.observer: UnitObserver | None = None

    def inserted(self, key: BlockHash, num_tokens: int) -> None:
        unit = self.units.get(key)
        if unit is not None:
            if self.observer is not None:
                self.observer.unit_recached(unit)
            return
        unit = RecoveryUnit(key, num_tokens, bool(num_tokens % self.block_size))
        self.units[key] = unit
        if self.observer is not None:
            self.observer.unit_added(unit)

    def removed(self, key: BlockHash) -> None:
        """The last cached copy of `key` is gone. Its children move up to
        its parent."""
        unit = self.units.pop(key, None)
        if unit is None:
            return
        parent = self.units.get(unit.parent) if unit.parent is not None else None
        if parent is not None:
            parent.children.discard(key)
        for child_key in unit.children:
            child = self.units.get(child_key)
            if child is not None:
                child.parent = unit.parent
                if parent is not None:
                    parent.children.add(child_key)
        for request_id, last in self.last.items():
            if last == key:
                self.last[request_id] = unit.parent
        self.leaves.discard(key)
        for request_id, resumed in self.resumed.items():
            if resumed == key:
                self.resumed[request_id] = None
        if self.observer is not None:
            self.observer.unit_removed(unit)

    def touched(self, key: BlockHash) -> None:
        unit = self.units.get(key)
        if unit is None:
            return
        unit.hits += 1
        if self.observer is not None:
            self.observer.unit_used(unit)

    def admitted(self, request_id: str, resume: BlockHash | None) -> None:
        parent = resume if resume in self.units else None
        self.last[request_id] = parent
        self.resumed[request_id] = parent

    def link(self, request_id: str, key: BlockHash) -> None:
        """`request_id` cached `key` after the previous unit on its path."""
        unit = self.units.get(key)
        if unit is None:
            return
        if not unit.linked:
            unit.linked = True
            parent_key = self.last.get(request_id)
            parent = (
                self.units.get(parent_key)
                if parent_key is not None and parent_key != key
                else None
            )
            if parent is not None:
                unit.parent = parent_key
                parent.children.add(key)
        self.last[request_id] = key
        if self.observer is not None:
            self.observer.unit_linked(unit)

    def forget(self, request_id: str) -> None:
        self.last.pop(request_id, None)
        self.resumed.pop(request_id, None)

    def clear(self) -> None:
        self.units.clear()
        self.last.clear()
        self.leaves.clear()
        self.resumed.clear()
