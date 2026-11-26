# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import heapq
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus

KV_HINTS_PROTOCOL_VERSION = "0.1"
SUPPORTED_ACTION_MAJOR_VERSION = "1"
RETAIN_ACTION_TYPE = "cpu.retain"
RELEASE_ACTION_TYPE = "cpu.release"
UNKNOWN_ACTION_LABEL = "cpu.unknown"
ENVELOPE_ACTION_LABEL = "envelope"

DEFAULT_RETENTION_TTL_S = 300.0
MAX_RETENTION_TTL_S = 3600.0
MAX_LEASES = 64_000


class HintOutcome:
    APPLIED = "applied"
    NOOP = "noop"
    REJECTED = "rejected"
    IGNORED = "ignored"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class _Retain:
    lease_id: str
    ttl_s: float


@dataclass(slots=True)
class _Lease:
    # Keys granted to this lease; some may have been evicted since.
    keys: set[OffloadKey]
    expires_at: float


class CPUHintRetention:
    """Applies `cpu.retain` / `cpu.release` KV hints as best-effort leases.

    A lease raises `lease_cnt` on its resident chunks; the policy only learns
    about 0<->1 transitions. Evicted keys stay in their lease until it is
    refreshed, so a re-store while the lease is live is retained again.
    """

    def __init__(self, policy: CachePolicy) -> None:
        self._policy = policy
        self._leases: dict[str, _Lease] = {}
        self._key_to_leases: dict[OffloadKey, set[str]] = {}
        # (expires_at, lease_id); entries older than the lease are skipped.
        self._expiry_heap: list[tuple[float, str]] = []
        # Retains of active requests, applied in on_request_finished.
        self._pending_retains: dict[str, list[_Retain]] = {}
        self._action_counts: dict[tuple[str, str], int] = {}
        self._keys_skipped = 0

    def _bump(self, action_type: str, outcome: str) -> None:
        key = (action_type, outcome)
        self._action_counts[key] = self._action_counts.get(key, 0) + 1

    def _unlink(self, key: OffloadKey, lease_id: str) -> None:
        lease_ids = self._key_to_leases[key]
        lease_ids.remove(lease_id)
        if not lease_ids:
            del self._key_to_leases[key]

    def _compact_expiry_heap(self) -> None:
        # Refreshes and releases leave stale entries behind until they expire.
        if len(self._expiry_heap) >= max(64, 2 * len(self._leases)):
            self._expiry_heap = [
                (lease.expires_at, lease_id) for lease_id, lease in self._leases.items()
            ]
            heapq.heapify(self._expiry_heap)

    def _remove_lease(self, lease_id: str) -> None:
        for key in self._leases.pop(lease_id).keys:
            self._unlink(key, lease_id)
            if (chunk := self._policy.get(key)) is not None:
                # Update lease_cnt first: policies read chunk.retained in the hook.
                chunk.lease_cnt -= 1
                if chunk.lease_cnt == 0:
                    self._policy.on_retention_changed(key, False)

    def _grant(
        self,
        lease_id: str,
        resident: Sequence[tuple[OffloadKey, ChunkStatus]],
        expires_at: float,
    ) -> str:
        lease = self._leases.get(lease_id)
        if lease is None:
            if len(self._leases) >= MAX_LEASES:
                return HintOutcome.REJECTED
            if not resident:
                return HintOutcome.NOOP
            lease = self._leases[lease_id] = _Lease(set(), expires_at)
            heapq.heappush(self._expiry_heap, (expires_at, lease_id))
        else:
            # Drop keys evicted since the last grant, which bounds a refreshed
            # lease by the cache size.
            evicted = [key for key in lease.keys if self._policy.get(key) is None]
            for key in evicted:
                lease.keys.remove(key)
                self._unlink(key, lease_id)
            # Refresh extends but never shortens the lease.
            if expires_at > lease.expires_at:
                lease.expires_at = expires_at
                heapq.heappush(self._expiry_heap, (expires_at, lease_id))
                self._compact_expiry_heap()
        # A refresh also covers keys this turn added to the session.
        for key, chunk in resident:
            if key not in lease.keys:
                lease.keys.add(key)
                self._key_to_leases.setdefault(key, set()).add(lease_id)
                chunk.lease_cnt += 1
                if chunk.lease_cnt == 1:
                    self._policy.on_retention_changed(key, True)
        return HintOutcome.APPLIED

    def sweep_expired(self, now: float | None = None) -> None:
        if not self._expiry_heap:
            return
        if now is None:
            now = time.monotonic()
        while self._expiry_heap and self._expiry_heap[0][0] <= now:
            expires_at, lease_id = heapq.heappop(self._expiry_heap)
            lease = self._leases.get(lease_id)
            if lease is not None and lease.expires_at == expires_at:
                self._remove_lease(lease_id)

    def on_new_request(self, req_context: ReqContext) -> None:
        """Validate CPU actions, apply releases now and defer retains."""
        hints = req_context.kv_hints
        if hints is None:
            return
        if hints.protocol_version != KV_HINTS_PROTOCOL_VERSION:
            self._bump(ENVELOPE_ACTION_LABEL, HintOutcome.IGNORED)
            return
        retains: list[_Retain] = []
        action_ids: set[str] = set()
        for action in hints.actions:
            action_type = action.action_type
            if not action_type.startswith("cpu."):
                continue
            if action_type not in (RETAIN_ACTION_TYPE, RELEASE_ACTION_TYPE):
                self._bump(UNKNOWN_ACTION_LABEL, HintOutcome.IGNORED)
                continue
            major_version = action.action_version.split(".")[0]
            if major_version != SUPPORTED_ACTION_MAJOR_VERSION:
                self._bump(action_type, HintOutcome.IGNORED)
                continue
            if action.action_id in action_ids:
                self._bump(action_type, HintOutcome.DUPLICATE)
                continue
            action_ids.add(action.action_id)
            lease_id = action.payload.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id:
                self._bump(action_type, HintOutcome.REJECTED)
                continue
            if action_type == RELEASE_ACTION_TYPE:
                # Envelope order: a release cancels earlier retains of its lease.
                kept = [retain for retain in retains if retain.lease_id != lease_id]
                for _ in range(len(retains) - len(kept)):
                    self._bump(RETAIN_ACTION_TYPE, HintOutcome.NOOP)
                retains = kept
                self.sweep_expired()
                released = lease_id in self._leases
                if released:
                    self._remove_lease(lease_id)
                    self._compact_expiry_heap()
                self._bump(
                    action_type, HintOutcome.APPLIED if released else HintOutcome.NOOP
                )
                continue
            ttl_s = action.payload.get("ttl_s")
            if ttl_s is None:
                ttl_s = DEFAULT_RETENTION_TTL_S
            if (
                isinstance(ttl_s, bool)
                or not isinstance(ttl_s, (int, float))
                or not 0 < ttl_s < math.inf
            ):
                self._bump(action_type, HintOutcome.REJECTED)
                continue
            retains.append(_Retain(lease_id, float(min(ttl_s, MAX_RETENTION_TTL_S))))
        if retains:
            self._pending_retains[req_context.req_id] = retains

    def on_request_finished(self, req_context: ReqContext) -> None:
        """Grant deferred retains over the request's resident keys."""
        retains = self._pending_retains.pop(req_context.req_id, None)
        if retains is None:
            return
        now = time.monotonic()
        self.sweep_expired(now)
        # Includes GPU prefix-cache hits that never reached CPU load/store.
        resident: list[tuple[OffloadKey, ChunkStatus]] = []
        for key in req_context._offload_key_positions:
            if (chunk := self._policy.get(key)) is not None:
                resident.append((key, chunk))
            else:
                self._keys_skipped += 1
        for retain in retains:
            outcome = self._grant(retain.lease_id, resident, now + retain.ttl_s)
            self._bump(RETAIN_ACTION_TYPE, outcome)

    def on_key_inserted(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        """Restore lease_cnt before a re-stored leased key reaches the policy."""
        if key not in self._key_to_leases:
            return
        self.sweep_expired()
        if (lease_ids := self._key_to_leases.get(key)) is not None:
            chunk.lease_cnt = len(lease_ids)

    def reset_cache(self) -> None:
        self._leases.clear()
        self._key_to_leases.clear()
        self._expiry_heap.clear()
        self._pending_retains.clear()

    def drain_stats(self, stats: OffloadingConnectorStats) -> None:
        for (action_type, outcome), count in self._action_counts.items():
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_HINT_ACTIONS,
                count,
                labelvalues=(action_type, outcome),
            )
        self._action_counts.clear()
        if self._keys_skipped:
            stats.increase_counter(
                CPUOffloadingMetrics.CPU_HINT_KEYS_SKIPPED, self._keys_skipped
            )
            self._keys_skipped = 0
