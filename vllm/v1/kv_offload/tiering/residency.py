# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-level residency and ownership metadata for KV tiering."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
)


class TieringMetrics:
    RESIDENT_BLOCKS = "vllm:kv_tiering_resident_blocks"
    SESSION_STATES = "vllm:kv_tiering_sessions"
    TRACKED_BLOCKS = "vllm:kv_tiering_tracked_blocks"
    SHARED_BLOCKS = "vllm:kv_tiering_shared_blocks"
    ACTIVE_BLOCKS = "vllm:kv_tiering_active_blocks"
    LOOKUPS = "vllm:kv_tiering_lookups"
    MIGRATION_BLOCKS = "vllm:kv_tiering_migration_blocks"
    MIGRATION_BYTES = "vllm:kv_tiering_migration_bytes"
    MIGRATION_LATENCY = "vllm:kv_tiering_migration_latency_seconds"
    EXPIRED_SESSIONS = "vllm:kv_tiering_expired_sessions"
    PRUNED_SESSIONS = "vllm:kv_tiering_pruned_sessions"
    DELETED_BLOCKS = "vllm:kv_tiering_deleted_blocks"
    PRIMARY_RECLAIMED_BLOCKS = "vllm:kv_tiering_primary_reclaimed_blocks"
    PRUNED_TRACKING_ENTRIES = "vllm:kv_tiering_pruned_tracking_entries"
    DEVICE_CACHE_USAGE = "vllm:kv_tiering_device_cache_usage"
    PRESSURE_STATE = "vllm:kv_tiering_pressure_state"
    STORE_DECISIONS = "vllm:kv_tiering_store_decisions"
    MIGRATION_BUDGET = "vllm:kv_tiering_migration_budget_blocks"
    REUSE_SIGNALS = "vllm:kv_tiering_reuse_signals"


def build_tiering_metric_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        TieringMetrics.RESIDENT_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of KV blocks observed in each offload tier.",
            labelnames=("tier",),
        ),
        TieringMetrics.SESSION_STATES: OffloadingGaugeMetadata(
            documentation="Number of lifecycle sessions in each state.",
            labelnames=("state",),
        ),
        TieringMetrics.TRACKED_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of block residency entries tracked by tiering.",
        ),
        TieringMetrics.SHARED_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of KV blocks referenced by multiple sessions.",
        ),
        TieringMetrics.ACTIVE_BLOCKS: OffloadingGaugeMetadata(
            documentation="Number of tracked KV blocks referenced by active sessions.",
        ),
        TieringMetrics.LOOKUPS: OffloadingCounterMetadata(
            documentation="Tiering lookup outcomes grouped by tier and result.",
            labelnames=("tier", "result"),
        ),
        TieringMetrics.MIGRATION_BLOCKS: OffloadingCounterMetadata(
            documentation="Number of KV blocks migrated between storage tiers.",
            labelnames=("source", "target"),
        ),
        TieringMetrics.MIGRATION_BYTES: OffloadingCounterMetadata(
            documentation="Bytes of KV data migrated between storage tiers.",
            labelnames=("source", "target"),
        ),
        TieringMetrics.MIGRATION_LATENCY: OffloadingHistogramMetadata(
            documentation="KV migration latency between storage tiers in seconds.",
            labelnames=("source", "target"),
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        ),
        TieringMetrics.EXPIRED_SESSIONS: OffloadingCounterMetadata(
            documentation="Number of idle KV lifecycle sessions that expired.",
        ),
        TieringMetrics.PRUNED_SESSIONS: OffloadingCounterMetadata(
            documentation=(
                "Number of oldest idle lifecycle sessions removed to bound metadata."
            ),
        ),
        TieringMetrics.DELETED_BLOCKS: OffloadingCounterMetadata(
            documentation="Number of expired KV block copies deleted from a tier.",
            labelnames=("tier",),
        ),
        TieringMetrics.PRIMARY_RECLAIMED_BLOCKS: OffloadingCounterMetadata(
            documentation=(
                "Number of idle CPU-primary blocks reclaimed after a secondary "
                "copy became durable."
            ),
            labelnames=("reason",),
        ),
        TieringMetrics.PRUNED_TRACKING_ENTRIES: OffloadingCounterMetadata(
            documentation="Number of stale residency metadata entries pruned.",
        ),
        TieringMetrics.DEVICE_CACHE_USAGE: OffloadingGaugeMetadata(
            documentation=(
                "Fraction of scheduler-managed device KV cache blocks in use."
            ),
        ),
        TieringMetrics.PRESSURE_STATE: OffloadingGaugeMetadata(
            documentation="Whether a tier is currently above its pressure threshold.",
            labelnames=("tier",),
        ),
        TieringMetrics.STORE_DECISIONS: OffloadingCounterMetadata(
            documentation="Pressure-policy decisions for newly computed KV blocks.",
            labelnames=("decision", "reason"),
        ),
        TieringMetrics.MIGRATION_BUDGET: OffloadingGaugeMetadata(
            documentation="Current device-to-CPU migration budget consumption.",
            labelnames=("scope",),
        ),
        TieringMetrics.REUSE_SIGNALS: OffloadingGaugeMetadata(
            documentation="Current lifecycle heat and external reuse observations.",
            labelnames=("signal",),
        ),
    }


@dataclass(slots=True)
class BlockResidencyState:
    key: OffloadKey
    cpu_resident: bool = False
    secondary_tiers: set[str] = field(default_factory=set)
    session_ids: set[str] = field(default_factory=set)
    inflight_transfers: Counter[tuple[str, str]] = field(default_factory=Counter)
    created_at: float = field(default_factory=time.monotonic)
    last_access_at: float = field(default_factory=time.monotonic)
    idle_since: float | None = None
    access_count: int = 0

    @property
    def is_shared(self) -> bool:
        return len(self.session_ids) > 1

    @property
    def has_secondary_copy(self) -> bool:
        return bool(self.secondary_tiers)

    @property
    def has_inflight_transfer(self) -> bool:
        return bool(self.inflight_transfers)


class BlockResidencyTracker:
    """Track residency, transfer state, and Session-to-Block ownership."""

    def __init__(self, max_entries: int = 64_000):
        if max_entries <= 0:
            raise ValueError("residency_max_entries must be greater than zero")
        self.max_entries = max_entries
        self._blocks: dict[OffloadKey, BlockResidencyState] = {}
        self._session_keys: dict[str, set[OffloadKey]] = {}
        self._active_sessions: set[str] = set()

    def on_session_active(self, session_id: str) -> None:
        self._active_sessions.add(session_id)
        now = time.monotonic()
        for key in self._session_keys.get(session_id, ()):
            state = self._blocks.get(key)
            if state is not None:
                state.idle_since = None
                state.last_access_at = now

    def on_session_idle(self, session_id: str) -> None:
        self._active_sessions.discard(session_id)
        now = time.monotonic()
        for key in self._session_keys.get(session_id, ()):
            state = self._blocks.get(key)
            if state is not None and not self._has_active_session(state):
                state.idle_since = now

    def record_access(self, session_id: str, keys: Iterable[OffloadKey]) -> None:
        now = time.monotonic()
        session_keys = self._session_keys.setdefault(session_id, set())
        for key in keys:
            state = self._get_or_create(key, now)
            state.session_ids.add(session_id)
            state.last_access_at = now
            state.access_count += 1
            if session_id in self._active_sessions:
                state.idle_since = None
            session_keys.add(key)

    def release_session(self, session_id: str) -> set[OffloadKey]:
        self._active_sessions.discard(session_id)
        keys = self._session_keys.pop(session_id, set())
        unreferenced: set[OffloadKey] = set()
        now = time.monotonic()
        for key in keys:
            state = self._blocks.get(key)
            if state is None:
                continue
            state.session_ids.discard(session_id)
            if not state.session_ids:
                state.idle_since = state.idle_since or now
                unreferenced.add(key)
            elif not self._has_active_session(state):
                state.idle_since = state.idle_since or now
        return unreferenced

    def mark_cpu_resident(
        self, keys: Iterable[OffloadKey], resident: bool = True
    ) -> None:
        now = time.monotonic()
        for key in keys:
            state = self._get_or_create(key, now)
            state.cpu_resident = resident

    def mark_secondary_resident(
        self, keys: Iterable[OffloadKey], tier: str, resident: bool = True
    ) -> None:
        now = time.monotonic()
        for key in keys:
            state = self._get_or_create(key, now)
            if resident:
                state.secondary_tiers.add(tier)
            else:
                state.secondary_tiers.discard(tier)

    def start_transfer(
        self, keys: Iterable[OffloadKey], source: str, target: str
    ) -> None:
        now = time.monotonic()
        transfer = (source, target)
        for key in keys:
            state = self._get_or_create(key, now)
            state.inflight_transfers[transfer] += 1

    def finish_transfer(
        self, keys: Iterable[OffloadKey], source: str, target: str
    ) -> None:
        transfer = (source, target)
        for key in keys:
            state = self._blocks.get(key)
            if state is None:
                continue
            count = state.inflight_transfers.get(transfer, 0)
            if count <= 1:
                state.inflight_transfers.pop(transfer, None)
            else:
                state.inflight_transfers[transfer] = count - 1

    def get_idle_cpu_candidates(
        self,
        *,
        limit: int,
        idle_before: float | None = None,
    ) -> list[OffloadKey]:
        candidates = [
            state
            for state in self._blocks.values()
            if state.cpu_resident
            and state.has_secondary_copy
            and not state.has_inflight_transfer
            and not self._has_active_session(state)
            and not state.is_shared
            and state.idle_since is not None
            and (idle_before is None or state.idle_since <= idle_before)
        ]
        candidates.sort(
            key=lambda state: (state.idle_since or 0.0, state.last_access_at)
        )
        return [state.key for state in candidates[:limit]]

    def get_cpu_pressure_candidates(
        self,
        *,
        limit: int,
        idle_before: float | None = None,
        include_active: bool = True,
    ) -> list[OffloadKey]:
        """Return CPU-resident blocks ordered from coldest to hottest.

        Idle, unshared, low-access blocks are preferred. Active-session blocks
        are considered only as a last resort when CPU pressure cannot be
        relieved using idle state alone.
        """
        candidates = [
            state
            for state in self._blocks.values()
            if state.cpu_resident
            and not state.has_inflight_transfer
            and (
                idle_before is None
                or (state.idle_since is not None and state.idle_since <= idle_before)
            )
            and (include_active or not self._has_active_session(state))
        ]
        candidates.sort(
            key=lambda state: (
                self._has_active_session(state),
                state.is_shared,
                state.access_count,
                state.idle_since is None,
                state.idle_since or state.last_access_at,
                state.last_access_at,
            )
        )
        return [state.key for state in candidates[:limit]]

    def has_secondary_copy(self, key: OffloadKey) -> bool:
        state = self._blocks.get(key)
        return state is not None and state.has_secondary_copy

    def is_known(self, key: OffloadKey) -> bool:
        state = self._blocks.get(key)
        return state is not None and (
            state.cpu_resident
            or state.has_secondary_copy
            or state.has_inflight_transfer
        )

    def has_idle_cpu_blocks(self) -> bool:
        return any(
            state.cpu_resident
            and state.has_secondary_copy
            and not state.has_inflight_transfer
            and not self._has_active_session(state)
            and not state.is_shared
            and state.idle_since is not None
            for state in self._blocks.values()
        )

    def get_inflight_keys(self) -> set[OffloadKey]:
        return {
            state.key for state in self._blocks.values() if state.has_inflight_transfer
        }

    def clear_cpu_residency(self) -> None:
        for state in self._blocks.values():
            state.cpu_resident = False

    def get_session_keys(self, session_id: str) -> set[OffloadKey]:
        return set(self._session_keys.get(session_id, ()))

    def snapshot(self) -> dict[str, object]:
        secondary_counts: Counter[str] = Counter()
        cpu_blocks = shared_blocks = active_blocks = 0
        for state in self._blocks.values():
            cpu_blocks += int(state.cpu_resident)
            shared_blocks += int(state.is_shared)
            active_blocks += int(self._has_active_session(state))
            secondary_counts.update(state.secondary_tiers)
        return {
            "tracked_blocks": len(self._blocks),
            "cpu_blocks": cpu_blocks,
            "secondary_blocks": dict(secondary_counts),
            "shared_blocks": shared_blocks,
            "active_blocks": active_blocks,
        }

    def prune(self) -> int:
        excess = len(self._blocks) - self.max_entries
        if excess <= 0:
            return 0
        candidates = sorted(
            (
                state
                for state in self._blocks.values()
                if not state.session_ids
                and not state.cpu_resident
                and not state.has_inflight_transfer
            ),
            key=lambda state: state.last_access_at,
        )
        for state in candidates[:excess]:
            self._blocks.pop(state.key, None)
        return min(excess, len(candidates))

    def _get_or_create(self, key: OffloadKey, now: float) -> BlockResidencyState:
        state = self._blocks.get(key)
        if state is None:
            state = BlockResidencyState(
                key=key,
                created_at=now,
                last_access_at=now,
            )
            self._blocks[key] = state
        return state

    def _has_active_session(self, state: BlockResidencyState) -> bool:
        return bool(state.session_ids.intersection(self._active_sessions))
