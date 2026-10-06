# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent retention declarations; no physical cache references or leases."""

import heapq
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TypeAlias

from vllm.v1.kv_hints.control import ControlError
from vllm.v1.kv_hints.native import HashTarget, RetainPayload, SessionTarget
from vllm.v1.kv_hints.state import (
    ContentKey,
    ContinuationKey,
    LogicalResolution,
    SessionPrefixIndex,
    content_key,
)

DeclarationKey: TypeAlias = tuple[str, str]


@dataclass(frozen=True)
class RetainDeclaration:
    owner_id: str
    declaration_id: str
    generation: int
    priority: int
    expires_at: float | None
    target: HashTarget | SessionTarget
    follows_growth: bool
    continuation_incarnation: int | None
    snapshot: LogicalResolution
    policy_name: str | None = None


class RetentionDeclarations:
    """Eviction biases shared by logical content, never correctness references.

    Follow-growth lookup uses the common index's current membership. It does not
    copy or periodically rescan every continuation as requests extend prefixes.
    """

    def __init__(
        self,
        index: SessionPrefixIndex,
        *,
        max_declarations: int = 4096,
        max_snapshot_entries: int = 262144,
        clock: Callable[[], float] = time.monotonic,
    ):
        if any(
            type(n) is not int or n < 1
            for n in (max_declarations, max_snapshot_entries)
        ):
            raise ValueError("Declaration limits must be positive integers")
        self.index = index
        self.max_declarations = max_declarations
        self.max_snapshot_entries = max_snapshot_entries
        self._clock = clock
        self._generation = 0
        self._snapshot_count = 0
        self._declarations: dict[DeclarationKey, RetainDeclaration] = {}
        self._owners: dict[str, set[DeclarationKey]] = {}
        self._snapshot_keys: dict[DeclarationKey, set[ContentKey]] = {}
        self._snapshot_sizes: dict[DeclarationKey, int] = {}
        self._by_content: dict[ContentKey, dict[DeclarationKey, int]] = {}
        self._following: dict[ContinuationKey, set[DeclarationKey]] = {}
        self._expiry: list[tuple[float, int, DeclarationKey]] = []
        self._projection_version: tuple[int, int] | None = None
        self._projection: Mapping[ContentKey, int] = MappingProxyType({})

    def retain(
        self,
        payload: RetainPayload,
        *,
        policy_name: str | None = None,
        resolution: LogicalResolution | None = None,
    ) -> RetainDeclaration:
        """Create/update a declaration after native validation and deduplication."""
        if payload.tier != "gpu":
            raise ControlError(
                "unsupported", "Local declarations only bias GPU eviction"
            )
        if not isinstance(payload.target, (HashTarget, SessionTarget)):
            raise ControlError(
                "unresolved_target", "Resolve active requests before retaining"
            )
        follows = payload.scope == "follow_growth"
        if follows and not (
            isinstance(payload.target, SessionTarget)
            and payload.target.continuation_id is not None
        ):
            raise ControlError(
                "invalid_payload", "Growth requires a named continuation"
            )
        resolution = (
            resolution if resolution is not None else self.index.resolve(payload.target)
        )
        if resolution.namespace != self.index.namespace:
            raise ControlError(
                "incompatible", "Retention snapshot belongs to another namespace"
            )
        self.expire()
        key = payload.owner_id, payload.declaration_id
        snapshot_keys = {content_key(b) for p in resolution.prefixes for b in p.blocks}
        size = self._resolution_size(resolution)
        if isinstance(payload.target, HashTarget):
            size += len(payload.target.blocks)
        incarnation = None
        # Following declarations address the live common index, so they retain
        # no historical snapshot of its potentially unbounded future growth.
        if follows:
            assert isinstance(payload.target, SessionTarget)
            assert payload.target.continuation_id is not None
            incarnation = self.index.get(
                payload.target.session_id, payload.target.continuation_id
            ).incarnation
            snapshot_keys = set()
            size = 0
            resolution = LogicalResolution(
                resolution.namespace, resolution.start_token, resolution.end_token, ()
            )
        count = self._snapshot_count - self._snapshot_sizes.get(key, 0) + size
        if (
            key not in self._declarations
            and len(self._declarations) >= self.max_declarations
        ) or count > self.max_snapshot_entries:
            raise ControlError("capacity", "Retention declaration capacity exhausted")
        self._remove(key)
        self._generation += 1
        expiry = (
            None if payload.ttl_seconds is None else self._clock() + payload.ttl_seconds
        )
        declaration = RetainDeclaration(
            payload.owner_id,
            payload.declaration_id,
            self._generation,
            payload.priority,
            expiry,
            payload.target,
            follows,
            incarnation,
            resolution,
            policy_name,
        )
        self._declarations[key] = declaration
        self._owners.setdefault(payload.owner_id, set()).add(key)
        if follows:
            assert isinstance(payload.target, SessionTarget)
            assert payload.target.continuation_id is not None
            continuation = payload.target.session_id, payload.target.continuation_id
            self._following.setdefault(continuation, set()).add(key)
        else:
            self._snapshot_keys[key] = snapshot_keys
            self._snapshot_sizes[key] = size
            for content in snapshot_keys:
                self._by_content.setdefault(content, {})[key] = payload.priority
        self._snapshot_count = count
        if expiry is not None:
            heapq.heappush(self._expiry, (expiry, declaration.generation, key))
        self._compact_expiry()
        return declaration

    def get(self, owner_id: str, declaration_id: str) -> RetainDeclaration | None:
        self.expire()
        return self._declarations.get((owner_id, declaration_id))

    def coverage(self, owner_id: str, declaration_id: str) -> LogicalResolution:
        declaration = self.get(owner_id, declaration_id)
        if declaration is None:
            raise ControlError("missing", "Retention declaration is unknown or expired")
        if declaration.follows_growth:
            self._check_continuation(declaration)
            return self.index.resolve(declaration.target)
        return declaration.snapshot

    def priority(self, group_id: int, block_hash: str) -> int:
        self.expire()
        priority = max(
            self._by_content.get((group_id, block_hash), {}).values(), default=0
        )
        for continuation, spans in self.index.content_associations(
            group_id, block_hash
        ):
            for key in self._following.get(continuation, ()):
                declaration = self._declarations[key]
                target = declaration.target
                state = self.index.get(*continuation)
                if state.incarnation != declaration.continuation_incarnation:
                    continue
                if any(
                    b.start_token >= target.start_token
                    and (target.end_token is None or b.end_token <= target.end_token)
                    for b in spans
                ):
                    priority = max(priority, declaration.priority)
        return priority

    def priority_snapshot(self) -> tuple[tuple[int, int], Mapping[ContentKey, int]]:
        """Return a cached eviction view, invalidated by expiry or followed growth."""
        self.expire()
        version = (
            self._generation,
            self.index.content_revision if self._following else 0,
        )
        if version != self._projection_version:
            priorities = {
                key: max(values.values()) for key, values in self._by_content.items()
            }
            for continuation, keys in self._following.items():
                try:
                    state = self.index.get(*continuation)
                except ControlError:
                    continue
                for key in keys:
                    declaration = self._declarations[key]
                    if state.incarnation != declaration.continuation_incarnation:
                        continue
                    target = declaration.target
                    for block in state.blocks:
                        if block.start_token >= target.start_token and (
                            target.end_token is None
                            or block.end_token <= target.end_token
                        ):
                            content = content_key(block)
                            priorities[content] = max(
                                priorities.get(content, 0), declaration.priority
                            )
            self._projection = MappingProxyType(priorities)
            self._projection_version = version
        return version, self._projection

    def active_declarations(self) -> tuple[RetainDeclaration, ...]:
        self.expire()
        return tuple(self._declarations.values())

    def release(self, owner_id: str, declaration_id: str) -> bool:
        self.expire()
        removed = self._remove((owner_id, declaration_id))
        self._compact_expiry()
        return removed

    def finish_task(
        self, session_id: str, *, expected_revision: int
    ) -> tuple[str, ...]:
        """Release the addressed owners before dropping their logical associations."""
        if type(
            expected_revision
        ) is not int or expected_revision != self.index.session_revision(session_id):
            raise ControlError("stale_revision", "Task tree changed before cleanup")
        self.expire()
        owners = self.index.task_tree(session_id)
        owned = {key for owner in owners for key in self._owners.get(owner, ())}
        closed = set(owners)
        frozen: dict[DeclarationKey, LogicalResolution] = {}
        for continuation, keys in self._following.items():
            if continuation[0] in closed:
                for key in keys - owned:
                    declaration = self._declarations[key]
                    if (
                        self.index.get(*continuation).incarnation
                        == declaration.continuation_incarnation
                    ):
                        frozen[key] = self.index.resolve(declaration.target)
        count = self._snapshot_count - sum(
            self._snapshot_sizes.get(k, 0) for k in owned
        )
        if (
            count + sum(self._resolution_size(r) for r in frozen.values())
            > self.max_snapshot_entries
        ):
            raise ControlError(
                "capacity", "Cannot preserve another owner's closing prefix"
            )
        for owner in owners:
            for key in tuple(self._owners.get(owner, ())):
                self._remove(key)
        for key, resolution in frozen.items():
            declaration = self._declarations[key]
            # Preserve another owner's effective scope and original expiry.
            self._remove(key)
            self._declarations[key] = replace(
                declaration, follows_growth=False, snapshot=resolution
            )
            self._owners.setdefault(declaration.owner_id, set()).add(key)
            self._snapshot_keys[key] = {
                content_key(b) for p in resolution.prefixes for b in p.blocks
            }
            size = self._resolution_size(resolution)
            self._snapshot_sizes[key] = size
            self._snapshot_count += size
            for content in self._snapshot_keys[key]:
                self._by_content.setdefault(content, {})[key] = declaration.priority
        self._compact_expiry()
        return self.index.finish_task(session_id, expected_revision=expected_revision)

    def prune(
        self, session_id: str, continuation_id: str, *, expected_revision: int
    ) -> None:
        self.expire()
        state = self.index.get(session_id, continuation_id)
        if any(
            self._declarations[k].continuation_incarnation == state.incarnation
            for k in self._following.get((session_id, continuation_id), ())
        ):
            raise ControlError(
                "busy", "A following declaration still uses this continuation"
            )
        self.index.prune(
            session_id, continuation_id, expected_revision=expected_revision
        )

    def expire(self) -> None:
        now = self._clock()
        while self._expiry and self._expiry[0][0] <= now:
            _, generation, key = heapq.heappop(self._expiry)
            current = self._declarations.get(key)
            if current is not None and current.generation == generation:
                self._remove(key)
        self._compact_expiry()

    def _remove(self, key: DeclarationKey) -> bool:
        declaration = self._declarations.pop(key, None)
        if declaration is None:
            return False
        self._generation += 1
        owner = self._owners[declaration.owner_id]
        owner.remove(key)
        if not owner:
            del self._owners[declaration.owner_id]
        if declaration.follows_growth:
            target = declaration.target
            assert (
                isinstance(target, SessionTarget) and target.continuation_id is not None
            )
            continuation = target.session_id, target.continuation_id
            following = self._following[continuation]
            following.remove(key)
            if not following:
                del self._following[continuation]
        else:
            snapshot_keys = self._snapshot_keys.pop(key)
            self._snapshot_count -= self._snapshot_sizes.pop(key)
            for content in snapshot_keys:
                priorities = self._by_content[content]
                del priorities[key]
                if not priorities:
                    del self._by_content[content]
        return True

    def _check_continuation(self, declaration: RetainDeclaration) -> None:
        target = declaration.target
        assert isinstance(target, SessionTarget) and target.continuation_id is not None
        state = self.index.get(target.session_id, target.continuation_id)
        if state.incarnation != declaration.continuation_incarnation:
            raise ControlError(
                "missing_association", "Retained continuation was replaced"
            )

    @staticmethod
    def _resolution_size(resolution: LogicalResolution) -> int:
        return sum(
            max(1, len(p.blocks) + len(p.partial_blocks) + len(p.missing_ranges))
            for p in resolution.prefixes
        )

    def _compact_expiry(self) -> None:
        if len(self._expiry) > max(64, 2 * len(self._declarations)):
            self._expiry = [
                (d.expires_at, d.generation, key)
                for key, d in self._declarations.items()
                if d.expires_at is not None
            ]
            heapq.heapify(self._expiry)
