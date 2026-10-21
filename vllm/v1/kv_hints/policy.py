# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only policy observations and proposal-only startup extension points."""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, TypeAlias

from vllm.v1.kv_hints.actions import (
    CacheLocation,
    CopyTo,
    MoveTo,
    Primitive,
    SetPriority,
)
from vllm.v1.kv_hints.control import OperationResult
from vllm.v1.kv_hints.native import (
    ContentSpan,
    DemotePayload,
    EvictPayload,
    HashTarget,
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
    | Primitive
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
    remaining_output_tokens: int = 0
    cached_tokens: int = 0


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
    owners: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyProposal:
    action: PolicyAction
    not_before: float = 0.0
    content_keys: frozenset[tuple[int, str]] = frozenset()
    wait_for_admission: bool = False
    skip_if_running: bool = False


@dataclass(frozen=True)
class PolicyDecision:
    proposals: tuple[PolicyProposal, ...] = ()


@dataclass(frozen=True)
class RequestGuidance:
    request_id: str
    admit: bool = True
    score: float = 0.0
    reason: str = ""
    reserve_running_growth: bool = False


class KVPolicy:
    """Policies receive values, never a scheduler, allocator or connector."""

    uses_scheduler_guidance = False
    # None preserves delivery of every signal to third-party policies.
    consumed_signals: frozenset[str] | None = None

    def needs_request_observation(self, payload: SignalPayload) -> bool:
        """Opt out only when a signal needs no request or session demand values."""
        return True

    def needs_request_observation_for_view(
        self, payload: SignalPayload, view: CacheObservation
    ) -> bool:
        """Refine request reads using physical counters before demand queries."""
        return self.needs_request_observation(payload)

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

    def on_result(
        self,
        action: PolicyAction,
        result: OperationResult,
        *,
        owner_id: str,
        owners: tuple[str, ...],
        elapsed_seconds: float,
    ) -> None:
        """Observe completed proposals without obtaining allocator access."""
        pass


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


class RetentionPolicy(KVPolicy):
    consumed_signals = frozenset({"importance", "resume"})

    def needs_request_observation(self, payload: SignalPayload) -> bool:
        return False

    def __init__(self, priority: int = 50, ttl_seconds: float = 30.0):
        if type(priority) is not int or not 0 <= priority <= 100:
            raise ValueError("Retention priority must be between 0 and 100")
        _positive(ttl_seconds, "ttl_seconds")
        self.priority = priority
        self.ttl_seconds = ttl_seconds
        self._revision = 0

    def on_signal(
        self, signal: PolicySignal, view: CacheObservation
    ) -> PolicyDecision | None:
        payload = signal.payload
        if payload.signal not in ("importance", "resume"):
            return None
        priority = (
            payload.importance if payload.importance is not None else self.priority
        )
        self._revision += 1
        return PolicyDecision(
            (
                PolicyProposal(
                    SetPriority(
                        signal.target,
                        "local_g1",
                        signal.owner_id,
                        self._revision,
                        priority,
                        self.ttl_seconds,
                    ),
                    skip_if_running=payload.signal == "resume",
                ),
            )
        )


class ToolWaitPolicy(KVPolicy):
    consumed_signals = frozenset({"tool_wait", "resume", "resume_prediction"})
    signal_updates_guidance = False

    def __init__(
        self,
        tier: Literal["cpu", "mooncake"] = "mooncake",
        lead_seconds: float = 0.1,
        min_pressure: float = 0.0,
        max_stored_blocks: int = 16384,
        min_pause_seconds: float = 0.5,
        prefetch_on_admission: bool = False,
        consider_pause_duration: bool = True,
    ):
        if tier not in ("cpu", "mooncake"):
            raise ValueError("Movement policy requires CPU or Mooncake")
        if not math.isfinite(lead_seconds) or lead_seconds < 0:
            raise ValueError("lead_seconds must be nonnegative and finite")
        if not math.isfinite(min_pressure) or not 0 <= min_pressure <= 1:
            raise ValueError("min_pressure must be between zero and one")
        self.tier = tier
        self.lead_seconds = lead_seconds
        self.min_pressure = min_pressure
        if not math.isfinite(min_pause_seconds) or min_pause_seconds < 0:
            raise ValueError("min_pause_seconds must be nonnegative and finite")
        self.min_pause_seconds = min_pause_seconds
        self.prefetch_on_admission = prefetch_on_admission
        self.uses_scheduler_guidance = prefetch_on_admission
        self.consider_pause_duration = consider_pause_duration
        if type(max_stored_blocks) is not int or max_stored_blocks < 1:
            raise ValueError("max_stored_blocks must be positive")
        self.max_stored_blocks = max_stored_blocks
        self._stored: OrderedDict[str, tuple[HashTarget, tuple[str, ...], int]] = (
            OrderedDict()
        )
        self._known_stored: OrderedDict[tuple[str, ContentSpan], None] = OrderedDict()
        self._stored_blocks = 0
        self._copy_seconds_per_byte: float | None = None
        self._store_seconds = 0.0
        self._load_seconds = 0.0
        self._kept: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    def _worth_moving(
        self, pause: float | None, view: CacheObservation | None = None
    ) -> bool:
        if (
            view is not None
            and self.consider_pause_duration
            and view.referenced_blocks < self.min_pressure * view.total_blocks
        ):
            return False
        minimum_window = max(
            self.min_pause_seconds, self._store_seconds + self._load_seconds
        )
        return (
            not self.consider_pause_duration
            or pause is None
            or pause > minimum_window
            or (
                view is not None
                and self._has_admission_pressure(view)
                and any(
                    r.waiting and r.waiting_seconds > minimum_window
                    for r in view.requests
                )
            )
        )

    def _has_admission_pressure(self, view: CacheObservation) -> bool:
        return (
            self.prefetch_on_admission
            and view.referenced_blocks > 0
            and view.referenced_blocks >= self.min_pressure * view.total_blocks
        )

    def needs_request_observation(self, payload: SignalPayload) -> bool:
        return payload.signal == "tool_wait" and self._worth_moving(
            payload.resume_after_seconds
        )

    def needs_request_observation_for_view(
        self, payload: SignalPayload, view: CacheObservation
    ) -> bool:
        if (
            payload.signal == "tool_wait"
            and self.consider_pause_duration
            and view.referenced_blocks < self.min_pressure * view.total_blocks
        ):
            return False
        return self.needs_request_observation(payload) or (
            payload.signal == "tool_wait" and self._has_admission_pressure(view)
        )

    def on_result(
        self,
        action: PolicyAction,
        result: OperationResult,
        *,
        owner_id: str,
        owners: tuple[str, ...],
        elapsed_seconds: float,
    ) -> None:
        if (
            isinstance(action, CopyTo)
            and action.dst.tier == "gpu"
            and result.transferred_bytes
            and "gpu_content_registered" in result.completed_steps
            and elapsed_seconds > 0
        ):
            observed = elapsed_seconds / result.transferred_bytes
            previous = self._copy_seconds_per_byte
            self._copy_seconds_per_byte = (
                observed if previous is None else 0.75 * previous + 0.25 * observed
            )
            self._load_seconds = max(
                elapsed_seconds, 0.75 * self._load_seconds + 0.25 * elapsed_seconds
            )
        if not isinstance(action, MoveTo) or result.namespace is None:
            return
        if result.transferred_bytes and "cold_copy_stored" in result.completed_steps:
            self._store_seconds = max(
                elapsed_seconds, 0.75 * self._store_seconds + 0.25 * elapsed_seconds
            )
        intended = tuple(
            ContentSpan(o.block_hash, o.group_id, o.start_token, o.end_token)
            for o in result.target_outcomes
            if (o.block_hash is not None and o.group_id is not None)
        )
        for outcome in result.target_outcomes:
            if (
                outcome.block_hash is None
                or outcome.group_id is None
                or "cold_copy_stored" not in outcome.completed_steps
            ):
                continue
            block = ContentSpan(
                outcome.block_hash,
                outcome.group_id,
                outcome.start_token,
                outcome.end_token,
            )
            content_key = result.namespace, block
            self._known_stored[content_key] = None
            self._known_stored.move_to_end(content_key)
            while len(self._known_stored) > self.max_stored_blocks:
                self._known_stored.popitem(last=False)
        blocks = tuple(
            block
            for block in intended
            if (result.namespace, block) in self._known_stored
        )
        if not blocks or len(blocks) > self.max_stored_blocks:
            return
        key = owner_id
        previous_stored = self._stored.pop(key, None)
        if previous_stored is not None:
            self._stored_blocks -= len(previous_stored[0].blocks)
        while self._stored_blocks + len(blocks) > self.max_stored_blocks:
            _, entry = self._stored.popitem(last=False)
            self._stored_blocks -= len(entry[0].blocks)
        self._stored[key] = (
            HashTarget(result.namespace, blocks),
            owners,
            result.transferred_bytes,
        )
        self._stored_blocks += len(blocks)

    def cleanup(self, owners: tuple[str, ...]) -> None:
        for key, entry in tuple(self._stored.items()):
            if set(owners).intersection(entry[1]):
                del self._stored[key]
                self._stored_blocks -= len(entry[0].blocks)
        for key, kept_owners in tuple(self._kept.items()):
            if set(owners).intersection(kept_owners):
                del self._kept[key]

    def on_signal(
        self, signal: PolicySignal, view: CacheObservation
    ) -> PolicyDecision | None:
        payload = signal.payload
        if payload.signal == "tool_wait":
            projected = view.referenced_blocks + sum(
                max(0, r.estimated_blocks - r.held_blocks) for r in view.requests
            )
            worthwhile = self._worth_moving(payload.resume_after_seconds, view)
            if not worthwhile or projected < self.min_pressure * view.total_blocks:
                self._kept[signal.owner_id] = signal.owners
                self._kept.move_to_end(signal.owner_id)
                while len(self._kept) > self.max_stored_blocks:
                    self._kept.popitem(last=False)
                return PolicyDecision()
            self._kept.pop(signal.owner_id, None)
            return PolicyDecision(
                (
                    PolicyProposal(
                        MoveTo(
                            signal.target,
                            CacheLocation("gpu"),
                            CacheLocation(self.tier),
                        )
                    ),
                )
            )
        if signal.owner_id in self._kept:
            return PolicyDecision()
        stored = self._stored.get(signal.owner_id)
        load_target = signal.target if stored is None else stored[0]
        lead = self.lead_seconds
        if stored is not None and self._copy_seconds_per_byte is not None:
            lead = max(lead, stored[2] * self._copy_seconds_per_byte)
        if payload.signal == "resume":
            return PolicyDecision(
                (
                    PolicyProposal(
                        CopyTo(
                            load_target,
                            CacheLocation(self.tier),
                            CacheLocation("gpu"),
                        ),
                        wait_for_admission=self.prefetch_on_admission,
                        skip_if_running=self.prefetch_on_admission,
                    ),
                )
            )
        if (
            payload.signal == "resume_prediction"
            and payload.resume_after_seconds is not None
        ):
            return PolicyDecision(
                (
                    PolicyProposal(
                        CopyTo(
                            load_target,
                            CacheLocation(self.tier),
                            CacheLocation("gpu"),
                        ),
                        view.now + max(0.0, payload.resume_after_seconds - lead),
                        wait_for_admission=self.prefetch_on_admission,
                        skip_if_running=self.prefetch_on_admission,
                    ),
                )
            )
        return None


class QuotaPolicy(KVPolicy):
    uses_scheduler_guidance = True
    consumed_signals = frozenset({"importance"})

    def needs_request_observation(self, payload: SignalPayload) -> bool:
        return False

    def __init__(
        self,
        shares: dict[str, float] | None = None,
        max_wait_seconds: float = 1.0,
        aging_seconds: float = 1.0,
        importance_ttl: float = 60.0,
        max_states: int = 4096,
        reserve_running_growth: bool = False,
        borrow_idle_capacity: bool = False,
        reuse_score_weight: float = 0.0,
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
        self.reserve_running_growth = reserve_running_growth
        self.borrow_idle_capacity = borrow_idle_capacity
        if not math.isfinite(reuse_score_weight) or reuse_score_weight < 0:
            raise ValueError("reuse_score_weight must be nonnegative and finite")
        self.reuse_score_weight = reuse_score_weight
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
        growth = sum(
            max(0, r.estimated_blocks - r.held_blocks)
            for r in view.requests
            if not r.waiting
        )
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
            available = max(0, view.free_blocks - growth)
            additional = max(0, request.estimated_blocks - request.held_blocks)
            if (
                self.borrow_idle_capacity
                and not occupied.get(request.session_id, 0)
                and additional <= available
            ):
                admit = True
            if self.reserve_running_growth:
                admit = admit and additional <= available
            importance = max(
                self._importance.get(("request", request.request_id), (0, 0))[0],
                self._importance.get(("session", request.session_id or ""), (0, 0))[0],
            )
            score = importance / 100 + request.waiting_seconds / self.aging_seconds
            score += self.reuse_score_weight * request.cached_tokens / 1024
            result.append(
                RequestGuidance(
                    request.request_id,
                    admit,
                    score,
                    "" if admit else "Session quota",
                    self.reserve_running_growth,
                )
            )
        return tuple(result)

    def cleanup(self, owners: tuple[str, ...]) -> None:
        for owner in owners:
            self._importance.pop(("session", owner), None)


def placement(action: object) -> str:
    if isinstance(action, (RetainPayload, PrefetchPayload)):
        return "gpu"
    if isinstance(action, SetPriority):
        return "gpu" if action.value is not None else ""
    if isinstance(action, CopyTo):
        return "gpu" if action.dst.tier == "gpu" else ""
    if isinstance(action, (DemotePayload, MoveTo)):
        return "cold"
    return "evict" if isinstance(action, EvictPayload) else ""


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
            destination = placement(action)
            conflict = bool(destination) and any(
                keys.intersection(previous) and destination != previous_destination
                for previous, previous_destination in accepted
            )
            choices.append(not conflict)
            if destination and not conflict:
                accepted.append((keys, destination))
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
                    previous.reserve_running_growth or proposal.reserve_running_growth,
                )
            )
        return tuple(merged.values())
