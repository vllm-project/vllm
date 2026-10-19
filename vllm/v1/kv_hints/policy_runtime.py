# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded policy planning through the native controller's execution boundary."""

import hashlib
import math
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import islice
from typing import TYPE_CHECKING

import msgspec

from vllm.config.kv_control import KVControlConfig
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.kv_hints.actions import (
    PRIMITIVE_VERSION,
    CopyTo,
    MoveTo,
    Primitive,
    SetPriority,
)
from vllm.v1.kv_hints.control import (
    ControlContext,
    ControlError,
    OperationOrigin,
    OperationReference,
    OperationResult,
    OperationStatus,
)
from vllm.v1.kv_hints.native import (
    DemotePayload,
    EvictPayload,
    HashTarget,
    NativePayload,
    PrefetchPayload,
    ReleasePayload,
    RequestTarget,
    RetainPayload,
    SessionTarget,
    SharePayload,
    SignalPayload,
)
from vllm.v1.kv_hints.policy import (
    CacheObservation,
    KVPlanComposer,
    KVPolicy,
    PolicyDecision,
    PolicyProposal,
    PolicySignal,
    QuotaPolicy,
    RequestDemand,
    RequestGuidance,
    RetentionPolicy,
    ToolWaitPolicy,
    placement,
)
from vllm.v1.kv_hints.protocol import KvHintAction, KvHintsEnvelope
from vllm.v1.kv_hints.state import LogicalResolution
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.request_queue import RequestQueue
    from vllm.v1.kv_hints.controller import NativeKVController

logger = init_logger(__name__)
_BUILTINS = {
    "retention": RetentionPolicy,
    "tool_wait": ToolWaitPolicy,
    "quota": QuotaPolicy,
}
_ACTIONS = {
    SetPriority: "vllm.set_priority",
    CopyTo: "vllm.copy_to",
    MoveTo: "vllm.move_to",
    RetainPayload: "retain",
    ReleasePayload: "release",
    EvictPayload: "evict",
    DemotePayload: "demote",
    PrefetchPayload: "prefetch",
    SharePayload: "share",
}


def _keys(scope: LogicalResolution | None) -> frozenset[tuple[int, str]]:
    return frozenset(
        (block.group_id, block.block_hash)
        for prefix in (scope.prefixes if scope is not None else ())
        for block in (*prefix.blocks, *prefix.partial_blocks)
    )


@dataclass
class _Work:
    operation: OperationResult
    proposal: PolicyProposal
    scope: LogicalResolution | None
    policy_name: str
    identity: str
    owners: tuple[str, ...]
    generation: int
    entries: int
    started_at: float = 0.0


class KVPolicyRuntime:
    def __init__(
        self,
        settings: KVControlConfig,
        controller: "NativeKVController",
        manager: KVCacheManager,
        requests: Callable[[], Iterable[Request]],
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.settings = settings
        self.controller = controller
        self.manager = manager
        manager.block_pool.track_cached_block_counts()
        self.requests = requests
        self.clock = clock
        self.wall_clock = wall_clock
        self.policies: dict[str, KVPolicy] = {}
        for config in settings.policies:
            if config.name in self.policies:
                raise ValueError("Policy names must be unique")
            cls = _BUILTINS.get(config.implementation)
            if cls is None:
                cls = resolve_obj_by_qualname(config.implementation)
            if not isinstance(cls, type) or not issubclass(cls, KVPolicy):
                raise ValueError("Configured policy must implement KVPolicy")
            self.policies[config.name] = cls(**config.options)
        composer_cls = (
            KVPlanComposer
            if settings.composer == "priority"
            else resolve_obj_by_qualname(settings.composer)
        )
        if not isinstance(composer_cls, type) or not issubclass(
            composer_cls, KVPlanComposer
        ):
            raise ValueError("Configured composer must implement KVPlanComposer")
        self.composer = composer_cls()
        self.has_guidance = any(
            p.uses_scheduler_guidance for p in self.policies.values()
        )
        self._pending: dict[int, _Work] = {}
        self._running: dict[int, _Work] = {}
        self._target_entries = 0
        self._arrivals: dict[str, tuple[float, int]] = {}
        self._waiting_since: dict[str, float] = {}
        self._arrival_sequence = 0
        self._request_views: dict[str, RequestDemand] = {}
        self._guidance: dict[str, RequestGuidance] = {}
        self._declaration_owners: dict[str, tuple[str, ...]] = {}
        self._last_tick = -math.inf
        self._view: CacheObservation | None = None
        self._view_generation = -1
        self.errors: dict[str, str] = {}

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def _generation(self) -> int:
        resources = self.manager.transfer_manager
        return resources.generation if resources is not None else 0

    def observation(self, *, include_requests: bool = True) -> CacheObservation:
        now, wall = self.clock(), self.wall_clock()
        requests = (
            list(islice(self.requests(), self.settings.max_policy_requests + 1))
            if include_requests
            else []
        )
        truncated = len(requests) > self.settings.max_policy_requests
        requests = requests[: self.settings.max_policy_requests]
        active_ids = {r.request_id for r in requests}
        if include_requests:
            self._arrivals = {
                key: value for key, value in self._arrivals.items() if key in active_ids
            }
            self._waiting_since = {
                key: value
                for key, value in self._waiting_since.items()
                if key in active_ids
            }
        session_blocks: dict[str | None, set[int]] = {}
        demands = []
        for request in requests:
            if request.is_finished():
                continue
            new_request = request.request_id not in self._arrivals
            if new_request:
                self._arrival_sequence += 1
                self._arrivals[request.request_id] = (
                    now - max(0.0, wall - request.arrival_time),
                    self._arrival_sequence,
                )
            arrival, order = self._arrivals[request.request_id]
            blocks = {
                b.block_id
                for group in self.manager.get_blocks(request.request_id).blocks
                for b in group
                if not b.is_null and b.pool is self.manager.block_pool
            }
            session_blocks.setdefault(request.session_id, set()).update(blocks)
            waiting = request.status != RequestStatus.RUNNING
            computed = self.manager.empty_kv_cache_blocks
            computed_tokens = request.num_computed_tokens
            cache_hits = 0
            if waiting and not blocks:
                computed, cache_hits, _ = self.manager.get_computed_blocks(
                    request, emit_events=False
                )
                computed_tokens = cache_hits
            remaining = max(0, request.max_tokens - request.num_output_tokens)
            estimated = len(blocks) + self._allocation_demand(
                request, computed, computed_tokens, computed_tokens
            )
            if waiting:
                since = self._waiting_since.setdefault(
                    request.request_id, arrival if new_request else now
                )
                waited = max(0.0, now - since)
            else:
                self._waiting_since.pop(request.request_id, None)
                waited = 0.0
            demands.append(
                RequestDemand(
                    request.request_id,
                    request.session_id,
                    waited,
                    order,
                    waiting,
                    len(blocks),
                    estimated,
                    remaining,
                    cache_hits,
                )
            )
        resources = self.manager.transfer_manager
        loads, sources = (
            resources.held_block_counts() if resources is not None else (0, 0)
        )
        pool = self.manager.block_pool
        total, free = pool.num_gpu_blocks - 1, pool.get_num_free_blocks()
        cached, reclaimable = pool.get_cached_block_counts()
        view = CacheObservation(
            now,
            total,
            free,
            total - free,
            loads,
            sources,
            len(self.controller.declarations.active_declarations()),
            tuple(demands),
            tuple((session, len(blocks)) for session, blocks in session_blocks.items()),
            truncated,
            cached_blocks=cached,
            reclaimable_cached_blocks=reclaimable,
            pending_policy_actions=len(self._pending) + len(self._running),
        )
        return view

    def observe_requests(self) -> None:
        if (
            self._view is not None
            and self._view_generation == self._generation()
            and self.clock() - self._view.now < self.settings.policy_interval_seconds
        ):
            return
        view = self.observation()
        self._view = view
        self._view_generation = self._generation()
        self._request_views = {r.request_id: r for r in view.requests}
        guidance: list[RequestGuidance] = []
        if not view.truncated_requests:
            for name, policy in self.policies.items():
                if not policy.uses_scheduler_guidance:
                    continue
                try:
                    offered = tuple(
                        islice(
                            policy.guidance(view), self.settings.max_policy_requests + 1
                        )
                    )
                    if len(offered) > self.settings.max_policy_requests:
                        raise ValueError("Guidance limit exceeded")
                    if any(
                        not isinstance(g, RequestGuidance) or type(g.admit) is not bool
                        for g in offered
                    ):
                        raise ValueError("Invalid scheduler guidance")
                    guidance.extend(offered)
                except Exception as error:
                    self.errors[name] = str(error)[:512]
                    logger.exception("KV policy %s guidance failed", name)
        try:
            combined = self.composer.combine(tuple(guidance))
            if len(combined) > self.settings.max_policy_requests or any(
                not isinstance(g, RequestGuidance)
                or type(g.admit) is not bool
                or not math.isfinite(g.score)
                for g in combined
            ):
                raise ValueError("Invalid composed scheduler guidance")
            self._guidance = {g.request_id: g for g in combined}
        except Exception as error:
            self.errors["composer"] = str(error)[:512]
            self._guidance = {}
            logger.exception("KV policy composition failed")

    def select_request(self, queue: "RequestQueue", *, priority: bool) -> Request:
        if self._view is None or self._view.truncated_requests or not self._guidance:
            return queue.peek_request()

        def ordering(request: Request) -> tuple[float, ...]:
            observed = self._request_views.get(request.request_id)
            if observed is None:
                return (request.priority if priority else 0, 1, 0, math.inf)
            expired = (
                self._waiting_seconds(observed) >= self.settings.max_policy_wait_seconds
            )
            guidance = self._guidance.get(request.request_id)
            score = guidance.score if guidance is not None else 0.0
            return (
                request.priority if priority else 0,
                0 if expired else 1,
                observed.arrival_order if expired else -score,
                observed.arrival_order,
            )

        return min(queue, key=ordering)

    def allows_request(self, request: Request) -> bool:
        observed = self._request_views.get(request.request_id)
        if (
            self._view is None
            or observed is None
            or self._waiting_seconds(observed) >= self.settings.max_policy_wait_seconds
        ):
            return True
        guidance = self._guidance.get(request.request_id)
        return guidance.admit if guidance is not None else True

    def _waiting_seconds(self, request: RequestDemand) -> float:
        assert self._view is not None
        return request.waiting_seconds + max(0.0, self.clock() - self._view.now)

    def _allocation_demand(
        self,
        request: Request,
        computed: KVCacheBlocks,
        computed_tokens: int,
        local_computed_tokens: int,
    ) -> int:
        """Estimate capacity using the scheduler's current cache hit."""
        remaining = max(0, request.max_tokens - request.num_output_tokens)
        tokens = min(
            self.manager.max_model_len,
            request.num_tokens
            + min(remaining, self.settings.policy_decode_lookahead_tokens),
        )
        return sum(
            group.get_num_blocks_to_allocate(
                request.request_id,
                tokens,
                computed.blocks[group_id],
                computed_tokens,
                local_computed_tokens,
                tokens,
                apply_admission_cap=True,
            )
            for group_id, group in enumerate(
                self.manager.coordinator.single_type_managers
            )
            if group.block_pool is self.manager.block_pool
        )

    def allows_allocation(
        self,
        request: Request,
        computed: KVCacheBlocks,
        computed_tokens: int,
        local_computed_tokens: int,
    ) -> bool:
        observed = self._request_views.get(request.request_id)
        guidance = self._guidance.get(request.request_id)
        if (
            self._view is None
            or observed is None
            or guidance is None
            or not guidance.reserve_running_growth
            or self._waiting_seconds(observed) >= self.settings.max_policy_wait_seconds
        ):
            return True
        growth = self._running_growth()
        additional = self._allocation_demand(
            request, computed, computed_tokens, local_computed_tokens
        )
        return additional <= max(
            0, self.manager.block_pool.get_num_free_blocks() - growth
        )

    def _running_growth(self) -> int:
        requests = tuple(islice(self.requests(), self.settings.max_policy_requests + 1))
        if len(requests) > self.settings.max_policy_requests:
            return self.manager.block_pool.num_gpu_blocks
        return sum(
            self._allocation_demand(
                running,
                self.manager.empty_kv_cache_blocks,
                running.num_computed_tokens,
                running.num_computed_tokens,
            )
            for running in requests
            if running.status == RequestStatus.RUNNING
        )

    def before_admission(self, request: Request) -> bool:
        """Start a selected session's preparation and await its publication."""
        if request.session_id is None or not (self._pending or self._running):
            return True
        hashes = {block_hash.hex() for block_hash in request.block_hashes}

        def matches(work: _Work) -> bool:
            return (
                work.proposal.wait_for_admission
                and request.session_id in work.owners
                and isinstance(work.proposal.action, CopyTo)
                and work.proposal.action.dst.tier == "gpu"
                and any(key[1] in hashes for key in work.proposal.content_keys)
            )

        if not any(
            matches(w) for w in (*self._pending.values(), *self._running.values())
        ):
            return True
        observed = self._request_views.get(request.request_id)
        if (
            observed is not None
            and self._view is not None
            and self._waiting_seconds(observed) >= self.settings.max_policy_wait_seconds
        ):
            for work in tuple(self._pending.values()):
                if matches(work):
                    self._defer(work, "Preparation exceeded the admission wait budget")
        self.tick(
            force=True,
            admission_operations=frozenset(
                w.operation.operation_id for w in self._pending.values() if matches(w)
            ),
        )
        return not any(
            matches(w) for w in (*self._pending.values(), *self._running.values())
        )

    def _defer(self, work: _Work, reason: str) -> None:
        self.controller.registry.update(
            work.operation, OperationStatus.DEFERRED, reason=reason
        )
        self._pending.pop(work.operation.operation_id)
        self._target_entries -= work.entries

    def _queue_preparation(self, work: _Work) -> None:
        self.controller.registry.update(
            work.operation,
            OperationStatus.QUEUED,
            reason="Awaiting admission with complete preparation capacity",
        )

    def _preparation_fits(self, work: _Work) -> bool:
        action = work.proposal.action
        assert isinstance(action, CopyTo) and work.scope is not None
        resources = self.manager.get_transfer_manager()
        requirements = resources.load_requirements(
            (
                (
                    make_block_hash_with_group_id(
                        BlockHash(bytes.fromhex(b.block_hash)), b.group_id
                    ),
                    b.end_token,
                )
                for prefix in work.scope.prefixes
                for b in prefix.blocks
            ),
            scope=f"{action.src.tier}/{self.controller.index.namespace}",
        )
        if requirements is None:
            self._queue_preparation(work)
            return False
        loads, committed = requirements
        total = (
            self.manager.block_pool.num_gpu_blocks - 1 - self.manager.watermark_blocks
        )
        if (
            resources.max_load_blocks and loads > resources.max_load_blocks
        ) or committed > total:
            self._defer(work, "Complete preparation exceeds its capacity budget")
            return False
        available = max(
            0,
            self.manager.block_pool.get_num_free_blocks()
            - self.manager.watermark_blocks
            - self._running_growth(),
        )
        if (
            resources.max_load_blocks
            and loads + resources.reserved_load_blocks > resources.max_load_blocks
        ) or committed > available:
            self._queue_preparation(work)
            return False
        return True

    def on_signal(
        self, operation: OperationResult, payload: SignalPayload, request_id: str | None
    ) -> None:
        consumers = [
            (name, policy)
            for name, policy in self.policies.items()
            if (signals := type(policy).__dict__.get("consumed_signals")) is None
            or payload.signal in signals
        ]
        if not consumers:
            self.controller.registry.update(
                operation,
                OperationStatus.UNCONSUMED,
                reason="No enabled policy consumes this signal",
            )
            return
        self._reap()
        if any(
            policy.uses_scheduler_guidance
            and type(policy).__dict__.get("signal_updates_guidance", True)
            for _, policy in consumers
        ):
            self._view = None
        view = self.observation(include_requests=False)
        include_requests = any(
            policy.needs_request_observation_for_view(payload, view)
            if "needs_request_observation_for_view" in type(policy).__dict__
            else "needs_request_observation" not in type(policy).__dict__
            or policy.needs_request_observation(payload)
            for _, policy in consumers
        )
        if include_requests:
            view = self.observation()
        binding = (
            request_id
            if isinstance(payload.target, RequestTarget)
            and payload.target.request_id == "$request"
            else payload.target.request_id
            if isinstance(payload.target, RequestTarget)
            else None
        )
        owners = (
            (payload.target.session_id,)
            if isinstance(payload.target, SessionTarget)
            else ()
        )
        identity = hashlib.sha256(
            msgspec.json.encode((payload.target, binding))
        ).hexdigest()[:32]
        observer = self.controller.request_observer
        if isinstance(payload.target, RequestTarget) and observer is not None:
            session_id = observer.session_for(payload.target, request_id)
            if session_id is not None:
                owners = (session_id,)
        proposed: list[tuple[str, str, PolicyProposal]] = []
        consumed = 0
        for name, policy in consumers:
            owner = f"policy:{name}:{identity}"
            decision = policy.on_signal(
                PolicySignal(payload, payload.target, owner, binding, owners), view
            )
            if decision is None:
                continue
            if not isinstance(decision, PolicyDecision):
                raise ControlError("invalid_payload", "Invalid policy decision")
            consumed += 1
            if consumed == 1:
                self.controller.registry.update(
                    operation,
                    OperationStatus.ACCEPTED,
                    completed_steps=("policy_context_updated",),
                )
            offered = tuple(
                islice(decision.proposals, self.settings.max_pending_policy_actions + 1)
            )
            self._supersede(name, identity, offered)
            for proposal in offered:
                if not isinstance(proposal, PolicyProposal) or not math.isfinite(
                    proposal.not_before
                ):
                    raise ControlError("invalid_payload", "Invalid policy proposal")
                if proposal.wait_for_admission and not (
                    isinstance(proposal.action, CopyTo)
                    and proposal.action.dst.tier == "gpu"
                ):
                    raise ControlError(
                        "invalid_payload", "Admission preparation requires a GPU COPY"
                    )
                if (
                    proposal.not_before - view.now
                    > self.settings.max_policy_delay_seconds
                ):
                    raise ControlError(
                        "capacity", "Policy prediction exceeds its horizon"
                    )
                if (
                    isinstance(proposal.action, (RetainPayload, ReleasePayload))
                    and proposal.action.owner_id != owner
                ):
                    raise ControlError(
                        "invalid_payload",
                        "Policy declaration owner must match its context",
                    )
                proposed.append((name, owner, proposal))
                if (
                    isinstance(proposal.action, SetPriority)
                    and proposal.action.claim_id != owner
                ):
                    raise ControlError(
                        "invalid_payload", "Policy claim must match its context"
                    )
        if not consumed:
            self.controller.registry.update(
                operation,
                OperationStatus.UNCONSUMED,
                reason="No enabled policy consumes this signal",
            )
            return
        if len(proposed) > min(
            128,
            self.settings.max_pending_policy_actions
            - len(self._pending)
            - len(self._running),
        ):
            raise ControlError("capacity", "Policy proposal capacity exhausted")
        prepared: list[_Work] = []
        prepared_entries = 0
        try:
            for name, owner, proposal in proposed:
                action = proposal.action
                action_type = _ACTIONS.get(type(action))
                if action_type is None:
                    raise ControlError(
                        "unsupported", "Policies may propose cache operations only"
                    )
                hint = KvHintAction(
                    "action",
                    action_type,
                    PRIMITIVE_VERSION
                    if isinstance(action, (SetPriority, CopyTo, MoveTo))
                    else "1",
                    msgspec.json.decode(msgspec.json.encode(action)),
                )
                origin = OperationOrigin(
                    operation.message_id,
                    operation.action_id,
                    operation.operation_id,
                    name,
                )
                context = ControlContext(
                    operation.engine_id,
                    operation.engine_epoch,
                    "native",
                    binding_id=operation.binding_id,
                    namespace=self.controller.index.namespace,
                    origin=origin,
                )
                child = self.controller.registry.submit(
                    KvHintsEnvelope("0.1", f"policy-{uuid.uuid4().hex}", [hint]),
                    context,
                ).results[0]
                scope = None
                entries = 0
                try:
                    self.controller.dispatcher.parse(hint)
                    if not isinstance(action, ReleasePayload):
                        if isinstance(action.target, str):
                            raise ControlError(
                                "unsupported",
                                "Policy proposals require a logical target",
                            )
                        target = self.controller._target(action.target, request_id)
                        scope = self.controller.index.resolve(target)
                        action = msgspec.structs.replace(action, target=target)
                    entries = (
                        sum(
                            max(
                                1,
                                len(p.blocks)
                                + len(p.partial_blocks)
                                + len(p.missing_ranges),
                            )
                            for p in scope.prefixes
                        )
                        if scope is not None
                        else 0
                    )
                    if not isinstance(action, ReleasePayload) and isinstance(
                        action.target, HashTarget
                    ):
                        entries += len(action.target.blocks)
                    if (
                        self._target_entries + prepared_entries + entries
                        > self.settings.max_policy_target_entries
                    ):
                        raise ControlError(
                            "capacity", "Policy snapshot capacity exhausted"
                        )
                except ControlError as error:
                    self.controller.execute_registered(child, error)
                    scope, entries = None, 0
                except Exception as error:
                    self.controller._record_failure(
                        child, OperationStatus.FAILED, type(error).__name__
                    )
                    scope, entries = None, 0
                prepared_entries += entries
                prepared.append(
                    _Work(
                        child,
                        replace(proposal, action=action, content_keys=_keys(scope)),
                        scope,
                        name,
                        identity,
                        owners,
                        self._generation(),
                        entries,
                    )
                )
        except Exception:
            for work in prepared:
                current = self.controller.registry.get(
                    work.operation.message_id, work.operation.action_id
                )
                if current is not None and not current.status.terminal:
                    self.controller.registry.update(
                        work.operation,
                        OperationStatus.DEFERRED,
                        reason="Policy plan admission was incomplete",
                    )
            self._record_derived(operation, prepared)
            raise
        self._record_derived(operation, prepared)
        eligible = []
        for work in prepared:
            current = self.controller.registry.get(
                work.operation.message_id, work.operation.action_id
            )
            if current is not None and not current.status.terminal:
                eligible.append(work)
        try:
            choices = self.composer.compose(tuple(w.proposal for w in eligible), view)
            if (
                not isinstance(choices, tuple)
                or len(choices) != len(eligible)
                or any(type(choice) is not bool for choice in choices)
            ):
                raise ControlError("invalid_payload", "Invalid composer decisions")
        except Exception:
            for work in eligible:
                self.controller.registry.update(
                    work.operation,
                    OperationStatus.FAILED,
                    reason="Policy composition failed",
                )
            raise
        for work, accepted in zip(eligible, choices, strict=True):
            if accepted:
                self._pending[work.operation.operation_id] = work
                self._target_entries += work.entries
            else:
                self.controller.registry.update(
                    work.operation,
                    OperationStatus.DEFERRED,
                    reason="Conflicting policy proposal",
                )
        self.controller.registry.update(
            operation,
            OperationStatus.COMPLETED,
            completed_steps=("policy_context_updated",),
            derived_operations=tuple(
                OperationReference(
                    w.operation.message_id,
                    w.operation.action_id,
                    w.operation.operation_id,
                )
                for w in prepared
            ),
        )
        self.tick(force=True)

    def _record_derived(self, parent: OperationResult, works: list[_Work]) -> None:
        self.controller.registry.update(
            parent,
            OperationStatus.ACCEPTED,
            completed_steps=("policy_context_updated",),
            derived_operations=tuple(
                OperationReference(
                    w.operation.message_id,
                    w.operation.action_id,
                    w.operation.operation_id,
                )
                for w in works
            ),
        )

    def _cancel(self, work: _Work, reason: str) -> None:
        if work.operation.operation_id in self._running:
            executor = self.controller.executor
            if executor is not None:
                executor.cancel_operations({work.operation.operation_id}, reason)
        else:
            self.controller.registry.update(
                work.operation, OperationStatus.CANCELLED, reason=reason
            )
        self._pending.pop(work.operation.operation_id, None)
        self._running.pop(work.operation.operation_id, None)
        self._target_entries -= work.entries

    def _supersede(
        self, name: str, identity: str, proposals: tuple[PolicyProposal, ...]
    ) -> None:
        for work in (*self._pending.values(), *self._running.values()):
            if work.policy_name == name and work.identity == identity:
                if (
                    work.operation.operation_id in self._running
                    and isinstance(work.proposal.action, CopyTo)
                    and work.generation == self._generation()
                    and any(p.action == work.proposal.action for p in proposals)
                ):
                    continue
                self._cancel(work, "Newer policy observation")

    def before_explicit(
        self, payload: NativePayload | Primitive, scope: LogicalResolution
    ) -> None:
        destination = placement(payload)
        if not destination:
            return
        keys = _keys(scope)
        for work in (*self._pending.values(), *self._running.values()):
            other = placement(work.proposal.action)
            if (
                other
                and other != destination
                and keys.intersection(work.proposal.content_keys)
            ):
                self._cancel(work, "Explicit command takes precedence")

    def cleanup(self, owners: tuple[str, ...]) -> None:
        for work in (*self._pending.values(), *self._running.values()):
            if set(work.owners).intersection(owners):
                self._cancel(work, "Task finished")
        declarations = self.controller.declarations
        for declaration in declarations.active_declarations():
            source_owners = self._declaration_owners.get(declaration.owner_id, ())
            if declaration.policy_name in self.policies and set(
                source_owners
            ).intersection(owners):
                declarations.release(declaration.owner_id, declaration.declaration_id)
        for policy in self.policies.values():
            policy.cleanup(owners)

    def tick(
        self,
        *,
        force: bool = False,
        admission_operations: frozenset[int] = frozenset(),
    ) -> None:
        now = self.clock()
        if not force and now - self._last_tick < self.settings.policy_interval_seconds:
            return
        self._last_tick = now
        self._reap()
        registry = self.controller.registry
        for work in tuple(self._pending.values()):
            if work.generation != self._generation():
                self._cancel(work, "Cache reset invalidated the policy plan")
        ready = [
            w
            for w in self._pending.values()
            if w.proposal.not_before <= now
            or w.operation.operation_id in admission_operations
        ]
        watch_admission = any(
            w.proposal.wait_for_admission or w.proposal.skip_if_running for w in ready
        )
        requests = (
            tuple(islice(self.requests(), self.settings.max_policy_requests + 1))
            if watch_admission
            else ()
        )
        truncated = len(requests) > self.settings.max_policy_requests
        waiting = {
            r.session_id
            for r in requests
            if not r.is_finished() and r.status != RequestStatus.RUNNING
        }
        running = {r.session_id for r in requests if r.status == RequestStatus.RUNNING}
        active = set() if truncated else running - waiting
        queued_demand = truncated or bool(waiting)
        copies: dict[int, list[_Work]] = {}
        for candidate in (*self._pending.values(), *self._running.values()):
            if (
                isinstance(candidate.proposal.action, CopyTo)
                and candidate.proposal.action.dst.tier == "gpu"
                and candidate.operation.origin is not None
            ):
                copies.setdefault(candidate.operation.origin.operation_id, []).append(
                    candidate
                )
        executed = 0
        for work in ready:
            if executed >= self.settings.max_policy_actions_per_step:
                break
            action = work.proposal.action
            if work.proposal.skip_if_running and active.intersection(work.owners):
                self._defer(work, "Session has already passed cache admission")
                continue
            if (
                work.proposal.wait_for_admission
                and work.owners
                and queued_demand
                and work.operation.operation_id not in admission_operations
            ):
                self._queue_preparation(work)
                continue
            if work.proposal.wait_for_admission and not self._preparation_fits(work):
                continue
            if isinstance(action, SetPriority) and action.value is not None:
                origin = work.operation.origin
                assert origin is not None
                waiting_for_copy = any(
                    (
                        other.operation.operation_id in self._pending
                        or other.operation.operation_id in self._running
                    )
                    and work.proposal.content_keys.intersection(
                        other.proposal.content_keys
                    )
                    for other in copies.get(origin.operation_id, ())
                )
                if waiting_for_copy:
                    registry.update(
                        work.operation,
                        OperationStatus.QUEUED,
                        reason="Awaiting compatible policy COPY",
                    )
                    continue
            del self._pending[work.operation.operation_id]
            executed += 1
            work.started_at = now
            conflict = False
            if work.scope is not None:
                executor = self.controller.executor
                conflict = executor is not None and executor.conflicts_with_explicit(
                    action, work.scope
                )
                if isinstance(action, (DemotePayload, MoveTo, EvictPayload)):
                    for (
                        declaration
                    ) in self.controller.declarations.active_declarations():
                        if (
                            declaration.policy_name is None
                            and work.proposal.content_keys.intersection(
                                _keys(
                                    self.controller.declarations.coverage(
                                        declaration.owner_id, declaration.declaration_id
                                    )
                                )
                            )
                        ):
                            conflict = True
                            break
            if conflict:
                registry.update(
                    work.operation,
                    OperationStatus.DEFERRED,
                    reason="Explicit command takes precedence",
                )
            else:
                self.controller.execute_registered(
                    work.operation,
                    action,
                    resolution=work.scope,
                    policy_owners=work.owners,
                )
            result = registry.get(work.operation.message_id, work.operation.action_id)
            if (
                isinstance(action, RetainPayload)
                and result is not None
                and result.status
                in (OperationStatus.COMPLETED, OperationStatus.PARTIAL)
            ):
                self._declaration_owners[action.owner_id] = work.owners
            if result is not None and not result.status.terminal:
                self._running[work.operation.operation_id] = work
            else:
                if result is not None:
                    self._notify_result(work, result)
                self._target_entries -= work.entries

    def _notify_result(self, work: _Work, result: OperationResult) -> None:
        try:
            self.policies[work.policy_name].on_result(
                work.proposal.action,
                result,
                owner_id=f"policy:{work.policy_name}:{work.identity}",
                owners=work.owners,
                elapsed_seconds=max(0.0, self.clock() - work.started_at),
            )
        except Exception as error:
            self.errors[work.policy_name] = str(error)[:512]
            logger.exception("KV policy %s result observation failed", work.policy_name)

    def before_acknowledge(self) -> None:
        """Consume final outcomes before a caller releases their detailed records."""
        self._reap()

    def cancel_operation(self, operation_id: int) -> None:
        work = self._pending.get(operation_id) or self._running.get(operation_id)
        if work is not None:
            self._cancel(work, "Caller cancelled the operation")

    def _reap(self) -> None:
        registry = self.controller.registry
        for identifier, work in tuple(self._running.items()):
            current = registry.get(work.operation.message_id, work.operation.action_id)
            if current is None or current.status.terminal:
                if current is not None:
                    self._notify_result(work, current)
                del self._running[identifier]
                self._target_entries -= work.entries
        if self._declaration_owners:
            active_owners = {
                d.owner_id for d in self.controller.declarations.active_declarations()
            }
            self._declaration_owners = {
                key: value
                for key, value in self._declaration_owners.items()
                if key in active_owners
            }
