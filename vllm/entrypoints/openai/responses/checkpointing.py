# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


@dataclass(slots=True)
class ContextSession:
    current_response_id: str | None = None
    current_engine_checkpoint_id: str | None = None
    checkpoints: dict[str, ContextCheckpoint] = field(default_factory=dict)
    summary_queue: deque[str] = field(default_factory=deque)


class ResponseContextManager:
    """Tracks response-based context sessions for checkpoint and rewind flows."""

    def __init__(self) -> None:
        self._sessions: dict[str, ContextSession] = {}
        self._engine_checkpoint_refcounts: dict[str, int] = {}

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

    def get_or_create_session(self, session_id: str) -> ContextSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = ContextSession()
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ContextSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ContextSessionNotFoundError(session_id)
        return session

    def get_current_response_id(self, session_id: str) -> str | None:
        return self.get_or_create_session(session_id).current_response_id

    def get_current_engine_checkpoint_id(self, session_id: str) -> str | None:
        return self.get_or_create_session(session_id).current_engine_checkpoint_id

    def set_current_response_id(self, session_id: str, response_id: str) -> None:
        self.get_or_create_session(session_id).current_response_id = response_id

    def set_current_state(
        self,
        session_id: str,
        response_id: str,
        engine_checkpoint_id: str,
    ) -> list[str]:
        session = self.get_or_create_session(session_id)
        stale_checkpoint_ids: list[str] = []
        if session.current_engine_checkpoint_id != engine_checkpoint_id:
            stale = self._decref_engine_checkpoint(session.current_engine_checkpoint_id)
            if stale is not None:
                stale_checkpoint_ids.append(stale)
            self._incref_engine_checkpoint(engine_checkpoint_id)
        session.current_response_id = response_id
        session.current_engine_checkpoint_id = engine_checkpoint_id
        return stale_checkpoint_ids

    def drop_checkpoint(
        self,
        session_id: str,
        checkpoint_label: str,
        response_id: str | None = None,
        engine_checkpoint_id: str | None = None,
    ) -> tuple[str | None, str | None, list[str]]:
        session = self.get_or_create_session(session_id)
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
        )
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
        return checkpoint

    def revert_and_queue_summary(
        self,
        session_id: str,
        checkpoint_label: str,
        summary: str,
    ) -> tuple[str | None, str | None, int, list[str]]:
        session = self.get_session(session_id)
        checkpoint = session.checkpoints.get(checkpoint_label)
        if checkpoint is None:
            raise ContextCheckpointNotFoundError(checkpoint_label)

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
            stale_checkpoint_ids,
        )

    def delete_checkpoint(
        self,
        session_id: str,
        checkpoint_label: str,
    ) -> tuple[ContextCheckpoint, list[str]]:
        session = self.get_session(session_id)
        checkpoint = session.checkpoints.pop(checkpoint_label, None)
        if checkpoint is None:
            raise ContextCheckpointNotFoundError(checkpoint_label)
        stale_checkpoint_ids: list[str] = []
        stale = self._decref_engine_checkpoint(checkpoint.engine_checkpoint_id)
        if stale is not None:
            stale_checkpoint_ids.append(stale)
        return checkpoint, stale_checkpoint_ids

    def consume_summary_queue(self, session_id: str) -> list[str]:
        session = self.get_or_create_session(session_id)
        if not session.summary_queue:
            return []
        summaries = list(session.summary_queue)
        session.summary_queue.clear()
        return summaries
