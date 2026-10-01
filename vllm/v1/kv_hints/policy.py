# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only policy observations and proposal-only startup extension points."""

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from vllm.v1.kv_hints.native import (
    DemotePayload,
    EvictPayload,
    PrefetchPayload,
    ReleasePayload,
    RequestTarget,
    RetainPayload,
    SessionTarget,
    SharePayload,
    SignalPayload,
    Target,
)

PolicyAction: TypeAlias = (
    RetainPayload
    | ReleasePayload
    | EvictPayload
    | DemotePayload
    | PrefetchPayload
    | SharePayload
)


@dataclass(frozen=True)
class RequestDemand:
    request_id: str
    session_id: str | None
    waiting_seconds: float
    arrival_order: int
    waiting: bool
    held_blocks: int
    estimated_blocks: int


@dataclass(frozen=True)
class CacheObservation:
    now: float
    total_blocks: int
    free_blocks: int
    referenced_blocks: int
    load_reservations: int
    transfer_source_blocks: int
    retention_declarations: int
    requests: tuple[RequestDemand, ...]
    session_blocks: tuple[tuple[str | None, int], ...]
    truncated_requests: bool = False
    cached_blocks: int = 0
    reclaimable_cached_blocks: int = 0
    pending_policy_actions: int = 0


@dataclass(frozen=True)
class PolicySignal:
    payload: SignalPayload
    target: Target
    owner_id: str
    request_id: str | None


@dataclass(frozen=True)
class PolicyProposal:
    action: PolicyAction
    not_before: float = 0.0
    content_keys: frozenset[tuple[int, str]] = frozenset()


@dataclass(frozen=True)
class PolicyDecision:
    proposals: tuple[PolicyProposal, ...] = ()


@dataclass(frozen=True)
class RequestGuidance:
    request_id: str
    admit: bool = True
    score: float = 0.0
    reason: str = ""


class KVPolicy:
    """Policies receive values, never a scheduler, allocator or connector."""

    uses_scheduler_guidance = False

    def on_signal(
        self,
        signal: PolicySignal,
        view: CacheObservation,
    ) -> PolicyDecision | None:
        return None

    def guidance(self, view: CacheObservation) -> tuple[RequestGuidance, ...]:
        return ()

    def cleanup(self, owners: tuple[str, ...]) -> None:
        pass


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


class RetentionPolicy(KVPolicy):
    def __init__(self, priority: int = 50, ttl_seconds: float = 30.0):
        if type(priority) is not int or not 0 <= priority <= 100:
            raise ValueError("Retention priority must be between 0 and 100")
        _positive(ttl_seconds, "ttl_seconds")
        self.priority = priority
        self.ttl_seconds = ttl_seconds

    def on_signal(
        self, signal: PolicySignal, view: CacheObservation
    ) -> PolicyDecision | None:
        payload = signal.payload
        if payload.signal not in ("importance", "resume"):
            return None
        priority = (
            payload.importance if payload.importance is not None else self.priority
        )
        return PolicyDecision(
            (
                PolicyProposal(
                    RetainPayload(
                        signal.target,
                        signal.owner_id,
                        "retention",
                        priority=priority,
                        ttl_seconds=self.ttl_seconds,
                    )
                ),
            )
        )


class ToolWaitPolicy(KVPolicy):
    def __init__(
        self, tier: Literal["cpu", "mooncake"] = "mooncake", lead_seconds: float = 0.1
    ):
        if tier not in ("cpu", "mooncake"):
            raise ValueError("Movement policy requires CPU or Mooncake")
        if not math.isfinite(lead_seconds) or lead_seconds < 0:
            raise ValueError("lead_seconds must be nonnegative and finite")
        self.tier = tier
        self.lead_seconds = lead_seconds

    def on_signal(
        self, signal: PolicySignal, view: CacheObservation
    ) -> PolicyDecision | None:
        payload = signal.payload
        if payload.signal == "tool_wait":
            return PolicyDecision(
                (PolicyProposal(DemotePayload(signal.target, self.tier)),)
            )
        if payload.signal == "resume":
            return PolicyDecision(
                (PolicyProposal(PrefetchPayload(signal.target, self.tier)),)
            )
        if (
            payload.signal == "resume_prediction"
            and payload.resume_after_seconds is not None
        ):
            return PolicyDecision(
                (
                    PolicyProposal(
                        PrefetchPayload(signal.target, self.tier),
                        view.now
                        + max(0.0, payload.resume_after_seconds - self.lead_seconds),
                    ),
                )
            )
        return None


class QuotaPolicy(KVPolicy):
    uses_scheduler_guidance = True

    def __init__(
        self,
        shares: dict[str, float] | None = None,
        max_wait_seconds: float = 1.0,
        aging_seconds: float = 1.0,
        importance_ttl: float = 60.0,
        max_states: int = 4096,
    ):
        self.shares = dict(shares or {})
        for share in self.shares.values():
            _positive(share, "share")
        for value, name in (
            (max_wait_seconds, "max_wait_seconds"),
            (aging_seconds, "aging_seconds"),
            (importance_ttl, "importance_ttl"),
        ):
            _positive(value, name)
        if type(max_states) is not int or max_states < 1:
            raise ValueError("max_states must be positive")
        self.max_wait_seconds = max_wait_seconds
        self.aging_seconds = aging_seconds
        self.importance_ttl = importance_ttl
        self.max_states = max_states
        self._importance: dict[tuple[str, str], tuple[int, float]] = {}

    def _expire(self, now: float) -> None:
        self._importance = {
            key: item for key, item in self._importance.items() if item[1] > now
        }

    def on_signal(
        self, signal: PolicySignal, view: CacheObservation
    ) -> PolicyDecision | None:
        if signal.payload.signal != "importance" or signal.payload.importance is None:
            return None
        target = signal.payload.target
        if isinstance(target, SessionTarget):
            key = "session", target.session_id
        elif isinstance(target, RequestTarget) and signal.request_id is not None:
            key = "request", signal.request_id
        else:
            return None
        self._expire(view.now)
        if key not in self._importance and len(self._importance) >= self.max_states:
            raise ValueError("Quota importance state capacity exhausted")
        self._importance[key] = (
            signal.payload.importance,
            view.now + self.importance_ttl,
        )
        return PolicyDecision()

    def guidance(self, view: CacheObservation) -> tuple[RequestGuidance, ...]:
        self._expire(view.now)
        weights = dict(self.shares)
        for request in view.requests:
            weights.setdefault(request.session_id or "", 1.0)
        total_weight = sum(weights.values()) or 1.0
        occupied = dict(view.session_blocks)
        result = []
        for request in view.requests:
            if not request.waiting:
                continue
            share = weights[request.session_id or ""] / total_weight
            demand = occupied.get(request.session_id, 0) + max(
                0, request.estimated_blocks - request.held_blocks
            )
            admit = request.waiting_seconds >= self.max_wait_seconds or demand <= max(
                1, int(view.total_blocks * share)
            )
            importance = max(
                self._importance.get(("request", request.request_id), (0, 0))[0],
                self._importance.get(("session", request.session_id or ""), (0, 0))[0],
            )
            score = importance / 100 + request.waiting_seconds / self.aging_seconds
            result.append(
                RequestGuidance(
                    request.request_id,
                    admit,
                    score,
                    "" if admit else "Session quota",
                )
            )
        return tuple(result)

    def cleanup(self, owners: tuple[str, ...]) -> None:
        for owner in owners:
            self._importance.pop(("session", owner), None)


class KVPlanComposer:
    """Compose advisory scores; physical conflicts remain engine-checked."""

    def compose(
        self, proposals: tuple[PolicyProposal, ...], view: CacheObservation
    ) -> tuple[bool, ...]:
        """Select compatible proposals in configured policy order."""
        accepted: list[tuple[frozenset[tuple[int, str]], str]] = []
        choices = []
        for proposal in proposals:
            action = proposal.action
            keys = proposal.content_keys
            placement = (
                "gpu"
                if isinstance(action, (RetainPayload, PrefetchPayload))
                else "cold"
                if isinstance(action, DemotePayload)
                else "evict"
                if isinstance(action, EvictPayload)
                else ""
            )
            conflict = bool(placement) and any(
                keys.intersection(previous) and placement != destination
                for previous, destination in accepted
            )
            choices.append(not conflict)
            if placement and not conflict:
                accepted.append((keys, placement))
        return tuple(choices)

    def combine(
        self, guidance: tuple[RequestGuidance, ...]
    ) -> tuple[RequestGuidance, ...]:
        merged: dict[str, RequestGuidance] = {}
        for proposal in guidance:
            if not math.isfinite(proposal.score):
                raise ValueError("Policy scores must be finite")
            previous = merged.get(proposal.request_id)
            merged[proposal.request_id] = (
                proposal
                if previous is None
                else RequestGuidance(
                    proposal.request_id,
                    previous.admit and proposal.admit,
                    previous.score + proposal.score,
                    previous.reason or proposal.reason,
                )
            )
        return tuple(merged.values())
