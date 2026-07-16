# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session lifecycle policy layered over block-level residency metadata."""

from __future__ import annotations

import math
import time
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.residency import BlockResidencyTracker


class LifecycleStatus(str, Enum):
    ACTIVE = "active"
    IDLE_RETAINED = "idle_retained"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass(slots=True)
class SessionKVState:
    session_id: str
    has_stable_id: bool
    status: LifecycleStatus
    created_at: float
    last_access_at: float
    ttl_deadline: float | None = None
    active_req_ids: set[str] = field(default_factory=set)
    retained_req_ids: set[str] = field(default_factory=set)
    block_keys: set[OffloadKey] = field(default_factory=set)
    request_count: int = 0
    reuse_request_count: int = 0
    reuse_hit_blocks: int = 0
    last_reuse_at: float | None = None
    last_reuse_req_id: str | None = None

    @property
    def is_idle(self) -> bool:
        return self.status is LifecycleStatus.IDLE_RETAINED


@dataclass(slots=True)
class LifecycleConfig:
    idle_ttl_sec: float = 0.0
    delete_expired_secondary: bool = False
    cpu_demote_after_sec: float = 0.0
    cpu_high_watermark: float = 1.0
    cpu_low_watermark: float = 1.0
    reclaim_batch_size: int = 64
    residency_max_entries: int = 64_000
    max_sessions: int = 4_096
    residency_tracking_enabled: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("lifecycle_idle_ttl_sec", self.idle_ttl_sec),
            ("lifecycle_cpu_demote_after_sec", self.cpu_demote_after_sec),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not 0 < self.cpu_low_watermark <= self.cpu_high_watermark <= 1:
            raise ValueError(
                "lifecycle_cpu_low_watermark and "
                "lifecycle_cpu_high_watermark must satisfy "
                "0 < low <= high <= 1"
            )
        if self.reclaim_batch_size <= 0:
            raise ValueError("lifecycle_reclaim_batch_size must be greater than zero")
        if self.residency_max_entries <= 0:
            raise ValueError("residency_max_entries must be greater than zero")
        if self.max_sessions <= 0:
            raise ValueError("lifecycle_max_sessions must be greater than zero")

    @property
    def enabled(self) -> bool:
        return (
            self.residency_tracking_enabled
            or self.idle_ttl_sec > 0
            or self.cpu_demote_after_sec > 0
            or self.cpu_high_watermark < 1
        )

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> LifecycleConfig:
        return cls(
            idle_ttl_sec=float(extra_config.get("lifecycle_idle_ttl_sec", 0.0)),
            delete_expired_secondary=bool(
                extra_config.get("lifecycle_delete_expired_secondary", False)
            ),
            cpu_demote_after_sec=float(
                extra_config.get("lifecycle_cpu_demote_after_sec", 0.0)
            ),
            cpu_high_watermark=float(
                extra_config.get("lifecycle_cpu_high_watermark", 0.9)
            ),
            cpu_low_watermark=float(
                extra_config.get("lifecycle_cpu_low_watermark", 0.7)
            ),
            reclaim_batch_size=int(
                extra_config.get("lifecycle_reclaim_batch_size", 64)
            ),
            residency_max_entries=int(
                extra_config.get("residency_max_entries", 64_000)
            ),
            max_sessions=int(extra_config.get("lifecycle_max_sessions", 4_096)),
            residency_tracking_enabled=bool(
                extra_config.get("residency_tracking_enabled", False)
            ),
        )


@dataclass(slots=True)
class ExpirationResult:
    expired_sessions: int = 0
    pruned_sessions: int = 0
    unreferenced_keys: set[OffloadKey] = field(default_factory=set)


def get_session_id(req_context: ReqContext) -> str:
    """Resolve a stable conversation identifier, falling back to req_id."""
    params = req_context.kv_transfer_params or {}
    for key in ("session_id", "conversation_id", "kv_session_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return req_context.req_id


def has_stable_session_id(req_context: ReqContext) -> bool:
    """Return whether the request carries a reusable conversation ID."""
    params = req_context.kv_transfer_params or {}
    return any(
        isinstance(params.get(key), str) and bool(params[key])
        for key in ("session_id", "conversation_id", "kv_session_id")
    )


class SessionLifecycleManager:
    """Manage Session state and its reverse Block ownership index."""

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self.residency = BlockResidencyTracker(config.residency_max_entries)
        self._sessions: dict[str, SessionKVState] = {}
        self._req_to_session: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def on_new_request(
        self,
        req_context: ReqContext,
        *,
        track_heat: bool = False,
    ) -> str:
        session_id = get_session_id(req_context)
        if not self.enabled and not track_heat:
            return session_id

        now = time.monotonic()
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionKVState(
                session_id=session_id,
                has_stable_id=has_stable_session_id(req_context),
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
        state.request_count += 1
        self._req_to_session[req_context.req_id] = session_id
        if self.enabled:
            self.residency.on_session_active(session_id)
        return session_id

    def record_reuse_hit(
        self,
        req_context: ReqContext,
        num_blocks: int = 1,
    ) -> None:
        """Record an external-tier hit as evidence of session reuse."""
        state = self._get_by_req(req_context)
        if state is None:
            return
        now = time.monotonic()
        if state.last_reuse_req_id != req_context.req_id:
            state.reuse_request_count += 1
            state.last_reuse_req_id = req_context.req_id
        state.reuse_hit_blocks += max(0, num_blocks)
        state.last_reuse_at = now
        state.last_access_at = now

    def get_session_heat(self, req_context: ReqContext) -> tuple[int, float]:
        """Return request count and empirical probability of session return.

        A stable session that has appeared ``n`` times has returned ``n - 1``
        times, so ``(n - 1) / n`` is used as a bounded, explainable reuse
        estimate. External-tier hits are tracked separately for observability.
        """
        state = self._get_by_req(req_context)
        if state is None or state.request_count <= 0:
            return 0, 0.0
        return state.request_count, (state.request_count - 1) / state.request_count

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
        key_list = list(keys)
        state.block_keys.update(key_list)
        state.last_access_at = time.monotonic()
        self.residency.record_access(state.session_id, key_list)

    def on_request_finished(
        self,
        req_context: ReqContext,
        *,
        track_heat: bool = False,
    ) -> None:
        if not self.enabled and not track_heat:
            return
        state = self._get_by_req(req_context)
        if state is None:
            self.on_new_request(req_context, track_heat=track_heat)
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
            if self.enabled:
                self.residency.on_session_idle(state.session_id)

    def on_request_finalized(self, req_context: ReqContext) -> None:
        """Release per-request metadata after all asynchronous stores finish."""
        state = self._get_by_req(req_context)
        self._req_to_session.pop(req_context.req_id, None)
        if state is None:
            return
        state.retained_req_ids.discard(req_context.req_id)
        if state.active_req_ids or state.retained_req_ids:
            return
        if state.has_stable_id:
            return

        self.residency.release_session(state.session_id)
        self._delete_state(state)

    def expire_idle_sessions(
        self,
        *,
        protected_keys: Collection[OffloadKey] = (),
        protected_req_ids: Collection[str] = (),
    ) -> ExpirationResult:
        if not self.enabled or self.config.idle_ttl_sec <= 0:
            return ExpirationResult()

        now = time.monotonic()
        protected_key_set = set(protected_keys)
        protected_req_set = set(protected_req_ids)
        result = ExpirationResult()
        for state in list(self._sessions.values()):
            if not state.is_idle or state.ttl_deadline is None:
                continue
            if state.ttl_deadline > now:
                continue
            if state.block_keys.intersection(protected_key_set):
                continue
            if state.retained_req_ids.intersection(protected_req_set):
                continue

            state.status = LifecycleStatus.EXPIRED
            result.unreferenced_keys.update(
                self.residency.release_session(state.session_id)
            )
            self._delete_state(state)
            result.expired_sessions += 1
        return result

    def prune_idle_sessions(self) -> ExpirationResult:
        """Bound retained Session metadata when TTL expiration is disabled."""
        excess = len(self._sessions) - self.config.max_sessions
        if excess <= 0:
            return ExpirationResult()

        candidates = sorted(
            (
                state
                for state in self._sessions.values()
                if state.is_idle
                and not state.active_req_ids
                and not state.retained_req_ids
            ),
            key=lambda state: state.last_access_at,
        )
        result = ExpirationResult()
        for state in candidates[:excess]:
            result.unreferenced_keys.update(
                self.residency.release_session(state.session_id)
            )
            self._delete_state(state)
            result.pruned_sessions += 1
        return result

    def get_idle_cpu_candidates(
        self,
        *,
        limit: int,
        require_idle_age: bool,
    ) -> list[OffloadKey]:
        idle_before = None
        if require_idle_age:
            if self.config.cpu_demote_after_sec <= 0:
                return []
            idle_before = time.monotonic() - self.config.cpu_demote_after_sec
        return self.residency.get_idle_cpu_candidates(
            limit=limit,
            idle_before=idle_before,
        )

    def has_pending_expiration(self) -> bool:
        return self.config.idle_ttl_sec > 0 and any(
            state.is_idle and state.ttl_deadline is not None
            for state in self._sessions.values()
        )

    def has_pending_cpu_demotion(self) -> bool:
        return (
            self.config.cpu_demote_after_sec > 0
            and self.residency.has_idle_cpu_blocks()
        )

    def snapshot(self) -> dict[str, int]:
        active = 0
        idle = 0
        retained_blocks: set[OffloadKey] = set()
        for state in self._sessions.values():
            retained_blocks.update(state.block_keys)
            if state.status is LifecycleStatus.ACTIVE:
                active += 1
            elif state.status is LifecycleStatus.IDLE_RETAINED:
                idle += 1
        return {
            "sessions": len(self._sessions),
            "active_sessions": active,
            "idle_sessions": idle,
            "retained_blocks": len(retained_blocks),
        }

    def heat_snapshot(self) -> dict[str, int]:
        return {
            "reuse_requests": sum(
                state.reuse_request_count for state in self._sessions.values()
            ),
            "reuse_hit_blocks": sum(
                state.reuse_hit_blocks for state in self._sessions.values()
            ),
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
