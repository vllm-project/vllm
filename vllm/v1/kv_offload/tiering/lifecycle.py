# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session lifecycle metadata for tiered KV offloading."""

from __future__ import annotations

import os
import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    IDLE_RETAINED = "idle_retained"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass(slots=True)
class SessionKVState:
    session_id: str
    status: LifecycleStatus
    created_at: float
    last_access_at: float
    ttl_deadline: float | None = None
    active_req_ids: set[str] = field(default_factory=set)
    retained_req_ids: set[str] = field(default_factory=set)
    block_keys: set[OffloadKey] = field(default_factory=set)

    @property
    def is_idle(self) -> bool:
        return self.status is LifecycleStatus.IDLE_RETAINED


@dataclass(slots=True)
class LifecycleConfig:
    idle_ttl_sec: float = 0.0
    delete_expired_secondary: bool = False

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> LifecycleConfig:
        return cls(
            idle_ttl_sec=float(extra_config.get("lifecycle_idle_ttl_sec", 0.0)),
            delete_expired_secondary=bool(
                extra_config.get("lifecycle_delete_expired_secondary", False)
            ),
        )


def get_session_id(req_context: ReqContext) -> str:
    """Resolve a stable conversation identifier, falling back to req_id."""
    params = req_context.kv_transfer_params or {}
    for key in ("session_id", "conversation_id", "kv_session_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return req_context.req_id


class SessionLifecycleManager:
    """Track active, retained, and expired KV sessions.

    TTL cleanup is disabled by default. Optional secondary deletion is limited
    to tiers exposing a FileMapper-compatible ``file_mapper`` attribute.
    """

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._sessions: dict[str, SessionKVState] = {}
        self._req_to_session: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self.config.idle_ttl_sec > 0

    def on_new_request(self, req_context: ReqContext) -> str:
        session_id = get_session_id(req_context)
        if not self.enabled:
            return session_id

        now = time.monotonic()
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionKVState(
                session_id=session_id,
                status=LifecycleStatus.ACTIVE,
                created_at=now,
                last_access_at=now,
            )
            self._sessions[session_id] = state
        else:
            state.status = LifecycleStatus.ACTIVE
            state.last_access_at = now
            state.ttl_deadline = None

        state.active_req_ids.add(req_context.req_id)
        state.retained_req_ids.discard(req_context.req_id)
        self._req_to_session[req_context.req_id] = session_id
        return session_id

    def record_request_keys(
        self, req_context: ReqContext, keys: Iterable[OffloadKey]
    ) -> None:
        if not self.enabled:
            return
        state = self._get_by_req(req_context)
        if state is None:
            self.on_new_request(req_context)
            state = self._get_by_req(req_context)
        assert state is not None
        state.block_keys.update(keys)
        state.last_access_at = time.monotonic()

    def on_request_finished(self, req_context: ReqContext) -> None:
        if not self.enabled:
            return
        state = self._get_by_req(req_context)
        if state is None:
            self.on_new_request(req_context)
            state = self._get_by_req(req_context)
        assert state is not None

        now = time.monotonic()
        state.active_req_ids.discard(req_context.req_id)
        state.retained_req_ids.add(req_context.req_id)
        state.last_access_at = now
        if not state.active_req_ids:
            state.status = LifecycleStatus.IDLE_RETAINED
            state.ttl_deadline = (
                now + self.config.idle_ttl_sec if self.config.idle_ttl_sec > 0 else None
            )

    def expire_idle_sessions(
        self,
        secondary_tiers: Iterable[object],
        *,
        protected_keys: Collection[OffloadKey] = (),
        protected_req_ids: Collection[str] = (),
    ) -> int:
        """Expire due sessions and optionally remove unshared FS blocks.

        When file deletion is enabled, sessions with in-flight request or block
        transfers remain retained and are retried on a later scheduler tick.
        """
        if not self.enabled:
            return 0

        now = time.monotonic()
        protected_key_set = set(protected_keys)
        protected_req_set = set(protected_req_ids)
        expired_count = 0
        for state in list(self._sessions.values()):
            if not state.is_idle or state.ttl_deadline is None:
                continue
            if state.ttl_deadline > now:
                continue
            if self.config.delete_expired_secondary and (
                state.block_keys.intersection(protected_key_set)
                or state.retained_req_ids.intersection(protected_req_set)
            ):
                continue

            state.status = LifecycleStatus.EXPIRED
            if self.config.delete_expired_secondary:
                self._delete_secondary_blocks(state, secondary_tiers)
            self._delete_state(state)
            expired_count += 1
        return expired_count

    def has_pending_expiration(self) -> bool:
        return self.enabled and any(
            state.is_idle and state.ttl_deadline is not None
            for state in self._sessions.values()
        )

    def snapshot(self) -> dict[str, int]:
        active = 0
        idle = 0
        retained_blocks = 0
        for state in self._sessions.values():
            retained_blocks += len(state.block_keys)
            if state.status is LifecycleStatus.ACTIVE:
                active += 1
            elif state.status is LifecycleStatus.IDLE_RETAINED:
                idle += 1
        return {
            "sessions": len(self._sessions),
            "active_sessions": active,
            "idle_sessions": idle,
            "retained_blocks": retained_blocks,
        }

    def _get_by_req(self, req_context: ReqContext) -> SessionKVState | None:
        session_id = self._req_to_session.get(req_context.req_id)
        if session_id is None:
            session_id = get_session_id(req_context)
        return self._sessions.get(session_id)

    def _delete_state(self, state: SessionKVState) -> None:
        state.status = LifecycleStatus.DELETED
        self._sessions.pop(state.session_id, None)
        for req_id in state.active_req_ids | state.retained_req_ids:
            self._req_to_session.pop(req_id, None)

    def _delete_secondary_blocks(
        self, state: SessionKVState, secondary_tiers: Iterable[object]
    ) -> None:
        for tier in secondary_tiers:
            file_mapper = getattr(tier, "file_mapper", None)
            if file_mapper is None:
                continue
            for key in state.block_keys:
                if self._is_key_referenced_by_other_session(key, state.session_id):
                    continue
                path = file_mapper.get_file_name(key)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.warning(
                        "Failed to delete expired lifecycle KV block %s",
                        path,
                        exc_info=True,
                    )

    def _is_key_referenced_by_other_session(
        self, key: OffloadKey, session_id: str
    ) -> bool:
        return any(
            other.session_id != session_id
            and other.status is not LifecycleStatus.DELETED
            and key in other.block_keys
            for other in self._sessions.values()
        )
