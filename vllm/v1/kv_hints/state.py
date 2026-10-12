# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-local logical associations, independent of physical block residency."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from itertools import islice
from typing import Literal, TypeAlias

from vllm.v1.kv_hints.control import ControlError, _validate_id
from vllm.v1.kv_hints.native import (
    ContentSpan,
    HashTarget,
    ParentPrefix,
    SessionTarget,
    validate_target,
)

ContinuationKey: TypeAlias = tuple[str, str]
ContentKey: TypeAlias = tuple[int, str]


@dataclass(frozen=True)
class TokenPrefixCheckpoint:
    token_count: int
    token_digest: str
    context_digest: str


@dataclass(frozen=True)
class ContinuationState:
    session_id: str
    continuation_id: str
    revision: int
    incarnation: int
    blocks: tuple[ContentSpan, ...]
    parent_prefix: ParentPrefix | None = None
    checkpoint: TokenPrefixCheckpoint | None = None


@dataclass(frozen=True)
class AssociationEvent:
    engine_epoch: str
    sequence: int
    revision: int
    session_id: str
    continuation_id: str
    kind: Literal[
        "associated", "removed", "updated", "pruned", "finished", "observation_failed"
    ]
    content: ContentSpan | None = None
    reason: str = ""


@dataclass(frozen=True)
class AssociationPage:
    revision: int
    items: tuple[ContinuationState, ...]
    next_after: ContinuationKey | None
    engine_epoch: str
    namespace: str
    event_sequence: int
    session_revisions: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RangeGap:
    group_id: int | None
    start_token: int
    end_token: int


@dataclass(frozen=True)
class ResolvedPrefix:
    continuation: ContinuationKey | None
    revision: int | None
    blocks: tuple[ContentSpan, ...]
    partial_blocks: tuple[ContentSpan, ...]
    missing_ranges: tuple[RangeGap, ...] = ()


@dataclass(frozen=True)
class LogicalResolution:
    namespace: str
    start_token: int
    end_token: int | None
    prefixes: tuple[ResolvedPrefix, ...]


@dataclass
class _Session:
    parent: str | None
    revision: int


def content_key(block: ContentSpan) -> ContentKey:
    return block.group_id, block.block_hash


def _ordered(blocks: tuple[ContentSpan, ...]) -> tuple[ContentSpan, ...]:
    return tuple(sorted(blocks, key=lambda b: (b.group_id, b.start_token, b.end_token)))


class SessionPrefixIndex:
    """Scheduler-thread index. Membership never establishes physical residency.

    Revisions are engine-wide and never reused, including after explicit pruning.
    Capacity exhaustion is observable; live associations are not silently evicted.
    """

    def __init__(
        self,
        namespace: str,
        engine_epoch: str,
        *,
        max_sessions: int = 4096,
        max_continuations: int = 16384,
        max_associations: int = 262144,
        max_events: int = 4096,
        max_tombstones: int = 16384,
    ):
        _validate_id(namespace, "namespace")
        _validate_id(engine_epoch, "engine_epoch")
        limits = (
            max_sessions,
            max_continuations,
            max_associations,
            max_events,
            max_tombstones,
        )
        if any(type(value) is not int or value < 1 for value in limits):
            raise ValueError("Index limits must be positive integers")
        self.namespace = namespace
        self.engine_epoch = engine_epoch
        self.max_sessions = max_sessions
        self.max_continuations = max_continuations
        self.max_associations = max_associations
        self.max_tombstones = max_tombstones
        self.revision = 0
        self.content_revision = 0
        self._sequence = 0
        self._association_count = 0
        self._sessions: dict[str, _Session] = {}
        self._continuations: dict[ContinuationKey, ContinuationState] = {}
        self._members: dict[
            ContentKey, dict[ContinuationKey, tuple[ContentSpan, ...]]
        ] = {}
        self._tombstones: OrderedDict[ContinuationKey, str] = OrderedDict()
        self._events: deque[AssociationEvent] = deque(maxlen=max_events)

    def get(self, session_id: str, continuation_id: str) -> ContinuationState:
        key = session_id, continuation_id
        record = self._continuations.get(key)
        if record is None:
            reason = self._tombstones.get(key, "unknown")
            raise ControlError("missing_association", f"Continuation is {reason}")
        return record

    def session_revision(self, session_id: str) -> int:
        session = self._sessions.get(session_id)
        if session is None:
            raise ControlError("missing_association", "Session is unknown")
        return session.revision

    def associate(
        self,
        session_id: str,
        continuation_id: str,
        blocks: tuple[ContentSpan, ...],
        *,
        expected_revision: int,
        task_parent_id: str | None = None,
        parent_prefix: ParentPrefix | None = None,
        checkpoint: TokenPrefixCheckpoint | None = None,
    ) -> ContinuationState:
        """Record an append-only branch, including prefix-cache-hit observations."""
        _validate_id(session_id, "session_id")
        _validate_id(continuation_id, "continuation_id")
        key = session_id, continuation_id
        previous = self._continuations.get(key)
        self._check_revision(
            expected_revision, 0 if previous is None else previous.revision
        )
        same_blocks = previous is not None and blocks is previous.blocks
        if not same_blocks:
            validate_target(HashTarget(self.namespace, blocks))
        session = self._sessions.get(session_id)
        if task_parent_id is not None:
            if task_parent_id == session_id:
                raise ControlError("invalid_payload", "A task cannot parent itself")
            if task_parent_id not in self._sessions:
                raise ControlError("missing_association", "Parent task is unknown")
            if session is not None and session.parent != task_parent_id:
                raise ControlError("conflict", "Task reparenting is not supported")
        if previous is not None and parent_prefix != previous.parent_prefix:
            raise ControlError("conflict", "Continuation parent prefix cannot change")
        if parent_prefix is not None and not same_blocks:
            if (parent_prefix.session_id, parent_prefix.continuation_id) == key:
                raise ControlError(
                    "invalid_payload", "A continuation cannot parent itself"
                )
            # The shared prefix is captured when the branch is created. Later
            # updates never read a newer version of its parent's continuation.
            parent = (
                previous
                if previous is not None
                else self.get(parent_prefix.session_id, parent_prefix.continuation_id)
            )
            if parent_prefix.end_token > max(
                (b.end_token for b in parent.blocks), default=0
            ):
                raise ControlError(
                    "missing_association", "Parent prefix is not tracked"
                )
            if any(
                b.start_token < parent_prefix.end_token < b.end_token
                and not any(
                    alias.group_id == b.group_id
                    and alias.start_token == b.start_token
                    and alias.end_token == parent_prefix.end_token
                    for alias in parent.blocks
                )
                for b in parent.blocks
            ):
                raise ControlError(
                    "invalid_payload", "Parent boundary splits tracked content"
                )
            inherited = [
                b for b in parent.blocks if b.end_token <= parent_prefix.end_token
            ]
            own = {(b.group_id, b.start_token, b.end_token): b for b in blocks}
            for block in inherited:
                position = block.group_id, block.start_token, block.end_token
                if position in own and own[position] != block:
                    raise ControlError(
                        "conflict", "Branch changes its shared parent prefix"
                    )
                own[position] = block
            blocks = tuple(own.values())
        if not same_blocks:
            blocks = _ordered(blocks)
            validate_target(HashTarget(self.namespace, blocks))
        old_blocks = () if previous is None else previous.blocks
        old_checkpoint = None if previous is None else previous.checkpoint
        if checkpoint is None:
            checkpoint = old_checkpoint
        if (
            old_checkpoint is not None
            and checkpoint is not None
            and (
                checkpoint.token_count < old_checkpoint.token_count
                or checkpoint.context_digest != old_checkpoint.context_digest
                or (
                    checkpoint.token_count == old_checkpoint.token_count
                    and checkpoint.token_digest != old_checkpoint.token_digest
                )
            )
        ):
            raise ControlError("conflict", "Continuation token prefix is append-only")
        if not same_blocks and not set(old_blocks).issubset(blocks):
            raise ControlError(
                "conflict",
                "Continuation content is append-only; use a new continuation ID "
                "to rewrite or truncate its prefix",
            )
        count = self._association_count - len(old_blocks) + len(blocks)
        if (
            (session is None and len(self._sessions) >= self.max_sessions)
            or (previous is None and len(self._continuations) >= self.max_continuations)
            or count > self.max_associations
        ):
            raise ControlError("capacity", "Logical index capacity exhausted")
        if (
            previous is not None
            and blocks == previous.blocks
            and checkpoint == old_checkpoint
        ):
            return previous
        self.revision += 1
        if previous is None or blocks != old_blocks:
            self.content_revision = self.revision
        state = ContinuationState(
            session_id,
            continuation_id,
            self.revision,
            self.revision if previous is None else previous.incarnation,
            blocks,
            parent_prefix,
            checkpoint,
        )
        self._sessions[session_id] = _Session(
            session.parent if session is not None else task_parent_id, self.revision
        )
        self._touch_ancestors(session_id)
        self._continuations[key] = state
        self._tombstones.pop(key, None)
        self._association_count = count
        if blocks == old_blocks:
            self._emit(key, "updated")
            return state
        removed = set(old_blocks) - set(blocks)
        added = set(blocks) - set(old_blocks)
        # Membership is by content, not by position or a reusable allocator ID.
        for content in {content_key(block) for block in old_blocks}:
            members = self._members[content]
            members.pop(key, None)
            if not members:
                del self._members[content]
        spans_by_content: dict[ContentKey, list[ContentSpan]] = {}
        for block in blocks:
            spans_by_content.setdefault(content_key(block), []).append(block)
        for content, spans in spans_by_content.items():
            self._members.setdefault(content, {})[key] = tuple(spans)
        for block in _ordered(tuple(removed)):
            self._emit(key, "removed", block)
        for block in _ordered(tuple(added)):
            self._emit(key, "associated", block)
        self._emit(key, "updated")
        return state

    def members(self, group_id: int, block_hash: str) -> tuple[ContinuationKey, ...]:
        return tuple(sorted(self._members.get((group_id, block_hash), ())))

    def report_observation_failure(
        self, session_id: str, continuation_id: str, *, reason: str
    ) -> None:
        self.get(session_id, continuation_id)
        self._emit(
            (session_id, continuation_id), "observation_failed", reason=reason[:4096]
        )

    def content_associations(
        self, group_id: int, block_hash: str
    ) -> tuple[tuple[ContinuationKey, tuple[ContentSpan, ...]], ...]:
        return tuple(sorted(self._members.get((group_id, block_hash), {}).items()))

    def resolve(self, target: HashTarget | SessionTarget) -> LogicalResolution:
        """Resolve logical coverage; KV managers must still validate group reuse."""
        validate_target(target)
        sources: list[
            tuple[
                ContinuationKey | None, int | None, tuple[ContentSpan, ...], int | None
            ]
        ]
        if isinstance(target, HashTarget):
            if target.namespace != self.namespace:
                raise ControlError("incompatible", "Target has another cache namespace")
            sources = [(None, None, target.blocks, None)]
        elif target.continuation_id is not None:
            state = self.get(target.session_id, target.continuation_id)
            sources = [
                (
                    (state.session_id, state.continuation_id),
                    state.revision,
                    state.blocks,
                    state.checkpoint.token_count if state.checkpoint else None,
                )
            ]
        else:
            self.session_revision(target.session_id)
            sources = [
                (
                    key,
                    state.revision,
                    state.blocks,
                    state.checkpoint.token_count if state.checkpoint else None,
                )
                for key, state in sorted(self._continuations.items())
                if key[0] == target.session_id
            ]
            if not sources:
                raise ControlError(
                    "missing_association", "Session has no tracked continuations"
                )
        prefixes = []
        for key, revision, blocks, known_end in sources:
            end_token = target.end_token
            if end_token is None and known_end is not None:
                end_token = max(
                    known_end, max((b.end_token for b in blocks), default=0)
                )
            full, partial = [], []
            for block in blocks:
                if block.end_token <= target.start_token or (
                    end_token is not None and block.start_token >= end_token
                ):
                    continue
                if block.start_token < target.start_token or (
                    end_token is not None and block.end_token > end_token
                ):
                    partial.append(block)
                else:
                    full.append(block)
            # A full partial-prefix alias can cover the requested interval even
            # when the same page also has a longer, unselected prefix entry.
            partial = [
                b
                for b in partial
                if not any(
                    f.group_id == b.group_id
                    and f.start_token <= max(b.start_token, target.start_token)
                    and f.end_token >= min(b.end_token, end_token or b.end_token)
                    for f in full
                )
            ]
            gaps = []
            if end_token is not None and end_token > target.start_token:
                groups = {b.group_id for b in blocks}
                if not groups:
                    gaps.append(RangeGap(None, target.start_token, end_token))
                for group in sorted(groups):
                    cursor = target.start_token
                    for block in (b for b in blocks if b.group_id == group):
                        start = max(block.start_token, target.start_token)
                        end = min(block.end_token, end_token)
                        if start >= end:
                            continue
                        if start > cursor:
                            gaps.append(RangeGap(group, cursor, start))
                        cursor = max(cursor, end)
                    if cursor < end_token:
                        gaps.append(RangeGap(group, cursor, end_token))
            prefixes.append(
                ResolvedPrefix(key, revision, tuple(full), tuple(partial), tuple(gaps))
            )
        return LogicalResolution(
            self.namespace, target.start_token, target.end_token, tuple(prefixes)
        )

    def query(
        self,
        *,
        session_id: str | None = None,
        after: ContinuationKey | None = None,
        revision: int | None = None,
        limit: int = 256,
    ) -> AssociationPage:
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("Query limit must be between 1 and 4096")
        if revision is not None and revision != self.revision:
            raise ControlError("stale_revision", "Index changed; restart pagination")
        if after is not None and revision is None:
            raise ControlError(
                "invalid_cursor", "Pagination requires an index revision"
            )
        if session_id is not None:
            self.session_revision(session_id)
        keys = [
            key
            for key in sorted(self._continuations)
            if (session_id is None or key[0] == session_id)
            and (after is None or key > after)
        ]
        selected = keys[:limit]
        cursor = selected[-1] if len(keys) > limit else None
        return AssociationPage(
            self.revision,
            tuple(self._continuations[k] for k in selected),
            cursor,
            self.engine_epoch,
            self.namespace,
            self._sequence,
            tuple(
                (name, self._sessions[name].revision)
                for name in sorted({key[0] for key in selected})
            ),
        )

    def task_tree(self, session_id: str) -> tuple[str, ...]:
        self.session_revision(session_id)
        children: dict[str, list[str]] = {}
        for name, session in self._sessions.items():
            if session.parent is not None:
                children.setdefault(session.parent, []).append(name)
        todo = [session_id]
        result = []
        while todo:
            name = todo.pop()
            result.append(name)
            todo.extend(children.get(name, ()))
        return tuple(result)

    def finish_task(
        self, session_id: str, *, expected_revision: int
    ) -> tuple[str, ...]:
        """Remove logical state; the controller must drain operation interests."""
        self._check_revision(expected_revision, self.session_revision(session_id))
        removed = self.task_tree(session_id)
        selected = set(removed)
        self.revision += 1
        self._touch_ancestors(session_id)
        for key in tuple(self._continuations):
            if key[0] in selected:
                self._remove(key, "finished")
        for name in removed:
            del self._sessions[name]
        return removed

    def prune(
        self, session_id: str, continuation_id: str, *, expected_revision: int
    ) -> None:
        """Forget selected history after the caller checks active interests."""
        state = self.get(session_id, continuation_id)
        self._check_revision(expected_revision, state.revision)
        self.revision += 1
        self._remove((session_id, continuation_id), "pruned")
        self._touch_ancestors(session_id)

    def events(
        self, after: int, *, epoch: str, limit: int = 256
    ) -> tuple[AssociationEvent, ...]:
        if epoch != self.engine_epoch:
            raise ControlError(
                "stale_epoch", "Association events belong to another epoch"
            )
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("Event limit must be between 1 and 4096")
        if type(after) is not int or not 0 <= after <= self._sequence:
            raise ControlError("event_gap", "Invalid association event position")
        if self._events and after < self._events[0].sequence - 1:
            raise ControlError(
                "event_gap", "Association history was pruned; query state"
            )
        return tuple(islice((e for e in self._events if e.sequence > after), limit))

    @staticmethod
    def _check_revision(expected: int, actual: int) -> None:
        if type(expected) is not int or expected != actual:
            raise ControlError("stale_revision", f"Expected current revision {actual}")

    def _emit(
        self,
        key: ContinuationKey,
        kind: Literal[
            "associated",
            "removed",
            "updated",
            "pruned",
            "finished",
            "observation_failed",
        ],
        content: ContentSpan | None = None,
        reason: str = "",
    ) -> None:
        self._sequence += 1
        self._events.append(
            AssociationEvent(
                self.engine_epoch,
                self._sequence,
                self.revision,
                *key,
                kind,
                content,
                reason,
            )
        )

    def _remove(
        self, key: ContinuationKey, reason: Literal["pruned", "finished"]
    ) -> None:
        state = self._continuations.pop(key)
        self.content_revision = self.revision
        self._association_count -= len(state.blocks)
        for content in {content_key(block) for block in state.blocks}:
            members = self._members[content]
            members.pop(key, None)
            if not members:
                del self._members[content]
        for block in state.blocks:
            self._emit(key, "removed", block)
        self._tombstones[key] = reason
        self._tombstones.move_to_end(key)
        while len(self._tombstones) > self.max_tombstones:
            self._tombstones.popitem(last=False)
        self._emit(key, reason)

    def _touch_ancestors(self, session_id: str) -> None:
        current: str | None = session_id
        while current is not None:
            session = self._sessions[current]
            session.revision = self.revision
            current = session.parent
