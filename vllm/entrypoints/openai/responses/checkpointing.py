# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from dataclasses import dataclass, field


class ContextSessionNotFoundError(KeyError):
    """Raised when a context session does not exist."""


class ContextCheckpointNotFoundError(KeyError):
    """Raised when a checkpoint label does not exist in a session."""


@dataclass(slots=True)
class ContextCheckpoint:
    label: str
    response_id: str | None
    engine_checkpoint_id: str | None
    created_at: float
    last_access_at: float


@dataclass(slots=True)
class ContextSession:
    current_response_id: str | None = None
    current_engine_checkpoint_id: str | None = None
    last_access_at: float = field(default_factory=time.monotonic)
    checkpoints: dict[str, ContextCheckpoint] = field(default_factory=dict)
    summary_queue: deque[str] = field(default_factory=deque)


class ResponseContextManager:
    """Tracks response-based context sessions for checkpoint and rewind flows."""

    def __init__(
        self,
        *,
        max_checkpoints_per_session: int | None = 64,
        checkpoint_ttl_s: float | None = 24 * 60 * 60,
    ) -> None:
        self._sessions: dict[str, ContextSession] = {}
        self._engine_checkpoint_refcounts: dict[str, int] = {}
        self.max_checkpoints_per_session = max_checkpoints_per_session
        self.checkpoint_ttl_s = checkpoint_ttl_s

    def _now(self) -> float:
        return time.monotonic()

    def _touch_session(self, session: ContextSession, now: float | None = None) -> None:
        session.last_access_at = self._now() if now is None else now

    def _touch_checkpoint(
        self, checkpoint: ContextCheckpoint, now: float | None = None
    ) -> None:
        checkpoint.last_access_at = self._now() if now is None else now

    def _incref_engine_checkpoint(self, checkpoint_id: str | None) -> None:
        if checkpoint_id is None:
            return
        self._engine_checkpoint_refcounts[checkpoint_id] = (
            self._engine_checkpoint_refcounts.get(checkpoint_id, 0) + 1
        )

    def _decref_engine_checkpoint(self, checkpoint_id: str | None) -> str | None:
        if checkpoint_id is None:
            return None
        current = self._engine_checkpoint_refcounts.get(checkpoint_id)
        if current is None:
            return None
        if current <= 1:
            self._engine_checkpoint_refcounts.pop(checkpoint_id, None)
            return checkpoint_id
        self._engine_checkpoint_refcounts[checkpoint_id] = current - 1
        return None

    def _drop_checkpoint_record(
        self,
        session: ContextSession,
        checkpoint_label: str,
    ) -> tuple[ContextCheckpoint | None, list[str]]:
        checkpoint = session.checkpoints.pop(checkpoint_label, None)
        if checkpoint is None:
            return None, []
        stale_checkpoint_ids: list[str] = []
        stale = self._decref_engine_checkpoint(checkpoint.engine_checkpoint_id)
        if stale is not None:
            stale_checkpoint_ids.append(stale)
        return checkpoint, stale_checkpoint_ids

    def _prune_session_checkpoints(
        self,
        session: ContextSession,
        *,
        now: float | None = None,
    ) -> list[str]:
        stale_checkpoint_ids: list[str] = []
        current_time = self._now() if now is None else now

        if self.checkpoint_ttl_s is not None and self.checkpoint_ttl_s >= 0:
            expired_labels = [
                label
                for label, checkpoint in session.checkpoints.items()
                if (current_time - checkpoint.last_access_at) >= self.checkpoint_ttl_s
            ]
            for label in expired_labels:
                _, stale = self._drop_checkpoint_record(session, label)
                stale_checkpoint_ids.extend(stale)

        if (
            self.max_checkpoints_per_session is not None
            and self.max_checkpoints_per_session >= 0
            and len(session.checkpoints) > self.max_checkpoints_per_session
        ):
            overflow = len(session.checkpoints) - self.max_checkpoints_per_session
            eviction_order = sorted(
                session.checkpoints.items(),
                key=lambda item: (item[1].last_access_at, item[1].created_at),
            )
            for label, _ in eviction_order[:overflow]:
                _, stale = self._drop_checkpoint_record(session, label)
                stale_checkpoint_ids.extend(stale)

        return stale_checkpoint_ids

    def get_or_create_session(self, session_id: str) -> ContextSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = ContextSession()
            self._sessions[session_id] = session
        self._touch_session(session)
        return session

    def get_session(self, session_id: str) -> ContextSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ContextSessionNotFoundError(session_id)
        self._touch_session(session)
        return session

    def get_current_response_id(self, session_id: str) -> str | None:
        return self.get_or_create_session(session_id).current_response_id

    def get_current_engine_checkpoint_id(self, session_id: str) -> str | None:
        return self.get_or_create_session(session_id).current_engine_checkpoint_id

    def set_current_response_id(self, session_id: str, response_id: str) -> None:
        self.get_or_create_session(session_id).current_response_id = response_id

    def resolve_request_session_state(
        self,
        session_id: str,
    ) -> tuple[str | None, str | None, list[str], list[str]]:
        now = self._now()
        session = self.get_or_create_session(session_id)
        self._touch_session(session, now)
        stale_checkpoint_ids = self._prune_session_checkpoints(session, now=now)
        summaries = list(session.summary_queue)
        session.summary_queue.clear()
        return (
            session.current_response_id,
            session.current_engine_checkpoint_id,
            summaries,
            stale_checkpoint_ids,
        )

    def set_current_state(
        self,
        session_id: str,
        response_id: str,
        engine_checkpoint_id: str,
    ) -> list[str]:
        session = self.get_or_create_session(session_id)
        now = self._now()
        self._touch_session(session, now)
        stale_checkpoint_ids: list[str] = []
        if session.current_engine_checkpoint_id != engine_checkpoint_id:
            stale = self._decref_engine_checkpoint(session.current_engine_checkpoint_id)
            if stale is not None:
                stale_checkpoint_ids.append(stale)
            self._incref_engine_checkpoint(engine_checkpoint_id)
        session.current_response_id = response_id
        session.current_engine_checkpoint_id = engine_checkpoint_id
        stale_checkpoint_ids.extend(self._prune_session_checkpoints(session, now=now))
        return stale_checkpoint_ids

    def drop_checkpoint(
        self,
        session_id: str,
        checkpoint_label: str,
        response_id: str | None = None,
        engine_checkpoint_id: str | None = None,
    ) -> tuple[str | None, str | None, list[str]]:
        session = self.get_or_create_session(session_id)
        now = self._now()
        self._touch_session(session, now)
        target_response_id = (
            response_id if response_id is not None else session.current_response_id
        )
        target_engine_checkpoint_id = (
            engine_checkpoint_id
            if engine_checkpoint_id is not None
            else session.current_engine_checkpoint_id
        )
        stale_checkpoint_ids: list[str] = []
        old_checkpoint = session.checkpoints.get(checkpoint_label)
        old_engine_checkpoint_id = (
            old_checkpoint.engine_checkpoint_id if old_checkpoint is not None else None
        )
        if old_engine_checkpoint_id != target_engine_checkpoint_id:
            stale = self._decref_engine_checkpoint(old_engine_checkpoint_id)
            if stale is not None:
                stale_checkpoint_ids.append(stale)
            self._incref_engine_checkpoint(target_engine_checkpoint_id)

        session.checkpoints[checkpoint_label] = ContextCheckpoint(
            label=checkpoint_label,
            response_id=target_response_id,
            engine_checkpoint_id=target_engine_checkpoint_id,
            created_at=now,
            last_access_at=now,
        )
        stale_checkpoint_ids.extend(self._prune_session_checkpoints(session, now=now))
        return target_response_id, target_engine_checkpoint_id, stale_checkpoint_ids

    def peek_checkpoint(
        self,
        session_id: str,
        checkpoint_label: str,
    ) -> ContextCheckpoint:
        session = self.get_session(session_id)
        checkpoint = session.checkpoints.get(checkpoint_label)
        if checkpoint is None:
            raise ContextCheckpointNotFoundError(checkpoint_label)
        self._touch_checkpoint(checkpoint)
        return checkpoint

    def revert_and_queue_summary(
        self,
        session_id: str,
        checkpoint_label: str,
        summary: str,
    ) -> tuple[str | None, str | None, int, list[str]]:
        session = self.get_session(session_id)
        now = self._now()
        self._touch_session(session, now)
        checkpoint = session.checkpoints.get(checkpoint_label)
        if checkpoint is None:
            raise ContextCheckpointNotFoundError(checkpoint_label)
        self._touch_checkpoint(checkpoint, now)

        stale_checkpoint_ids: list[str] = []
        if session.current_engine_checkpoint_id != checkpoint.engine_checkpoint_id:
            stale = self._decref_engine_checkpoint(session.current_engine_checkpoint_id)
            if stale is not None:
                stale_checkpoint_ids.append(stale)
            self._incref_engine_checkpoint(checkpoint.engine_checkpoint_id)

        session.current_response_id = checkpoint.response_id
        session.current_engine_checkpoint_id = checkpoint.engine_checkpoint_id
        session.summary_queue.append(summary)
        return (
            checkpoint.response_id,
            checkpoint.engine_checkpoint_id,
            len(session.summary_queue),
            stale_checkpoint_ids + self._prune_session_checkpoints(session, now=now),
        )

    def delete_checkpoint(
        self,
        session_id: str,
        checkpoint_label: str,
    ) -> tuple[ContextCheckpoint, list[str]]:
        session = self.get_session(session_id)
        checkpoint, stale_checkpoint_ids = self._drop_checkpoint_record(
            session,
            checkpoint_label,
        )
        if checkpoint is None:
            raise ContextCheckpointNotFoundError(checkpoint_label)
        stale_checkpoint_ids.extend(self._prune_session_checkpoints(session))
        return checkpoint, stale_checkpoint_ids

    def consume_summary_queue(self, session_id: str) -> list[str]:
        session = self.get_or_create_session(session_id)
        if not session.summary_queue:
            return []
        summaries = list(session.summary_queue)
        session.summary_queue.clear()
        return summaries
