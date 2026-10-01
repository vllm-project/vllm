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
from vllm.v1.core.kv_cache_manager import KVCacheManager
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
    NativePayload,
    PrefetchPayload,
    ReleasePayload,
    RequestTarget,
    RetainPayload,
    SessionTarget,
    SharePayload,
    SignalPayload,
    parse_native_action,
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


def _placement(action: NativePayload) -> str:
    if isinstance(action, (RetainPayload, PrefetchPayload)):
        return "gpu"
    if isinstance(action, DemotePayload):
        return "cold"
    if isinstance(action, EvictPayload):
        return "evict"
    return ""


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
        self.errors: dict[str, str] = {}

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def _generation(self) -> int:
        resources = self.manager.transfer_manager
        return resources.generation if resources is not None else 0

    def observation(self) -> CacheObservation:
        now, wall = self.clock(), self.wall_clock()
        requests = list(islice(self.requests(), self.settings.max_policy_requests + 1))
        truncated = len(requests) > self.settings.max_policy_requests
        requests = requests[: self.settings.max_policy_requests]
        active_ids = {r.request_id for r in requests}
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
            estimated = sum(
                math.ceil(request.num_tokens / group.block_size)
                for group in self.manager.coordinator.single_type_managers
                if group.block_pool is self.manager.block_pool
            )
            waiting = request.status != RequestStatus.RUNNING
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
                )
            )
        resources = self.manager.transfer_manager
        loads, sources = (
            resources.held_block_counts() if resources is not None else (0, 0)
        )
        pool = self.manager.block_pool
        total, free = pool.num_gpu_blocks - 1, pool.get_num_free_blocks()
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
            cached_blocks=sum(b.block_hash is not None for b in pool.blocks),
            reclaimable_cached_blocks=sum(
                b.block_hash is not None and b.ref_cnt == 0 for b in pool.blocks
            ),
            pending_policy_actions=len(self._pending) + len(self._running),
        )
        self._view = view
        self._request_views = {r.request_id: r for r in demands}
        return view

    def observe_requests(self) -> None:
        view = self.observation()
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
            expired = observed.waiting_seconds >= self.settings.max_policy_wait_seconds
            score = self._guidance.get(
                request.request_id, RequestGuidance(request.request_id)
            ).score
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
            observed is None
            or observed.waiting_seconds >= self.settings.max_policy_wait_seconds
        ):
            return True
        return self._guidance.get(
            request.request_id, RequestGuidance(request.request_id)
        ).admit

    def on_signal(
        self, operation: OperationResult, payload: SignalPayload, request_id: str | None
    ) -> None:
        self._reap()
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
        for name, policy in self.policies.items():
            owner = f"policy:{name}:{identity}"
            decision = policy.on_signal(
                PolicySignal(payload, payload.target, owner, binding), view
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
            self._supersede(name, identity)
            for proposal in islice(
                decision.proposals, self.settings.max_pending_policy_actions + 1
            ):
                if not isinstance(proposal, PolicyProposal) or not math.isfinite(
                    proposal.not_before
                ):
                    raise ControlError("invalid_payload", "Invalid policy proposal")
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
                    "1",
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
                    parse_native_action(hint)
                    if not isinstance(action, ReleasePayload):
                        target = self.controller._target(action.target, request_id)
                        scope = self.controller.index.resolve(target)
                        action = msgspec.structs.replace(action, target=target)
                    entries = (
                        sum(
                            len(p.blocks)
                            + len(p.partial_blocks)
                            + len(p.missing_ranges)
                            for p in scope.prefixes
                        )
                        if scope is not None
                        else 0
                    )
                    if (
                        self._target_entries
                        + sum(w.entries for w in prepared)
                        + entries
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
        except Exception:
            for work in eligible:
                self.controller.registry.update(
                    work.operation,
                    OperationStatus.FAILED,
                    reason="Policy composition failed",
                )
            raise
        if len(choices) != len(eligible) or any(
            type(choice) is not bool for choice in choices
        ):
            for work in eligible:
                self.controller.registry.update(
                    work.operation,
                    OperationStatus.FAILED,
                    reason="Composer returned invalid decisions",
                )
            raise ControlError("invalid_payload", "Invalid composer decisions")
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

    def _supersede(self, name: str, identity: str) -> None:
        for work in (*self._pending.values(), *self._running.values()):
            if work.policy_name == name and work.identity == identity:
                self._cancel(work, "Newer policy observation")

    def before_explicit(self, payload: NativePayload, scope: LogicalResolution) -> None:
        placement = _placement(payload)
        if not placement:
            return
        keys = _keys(scope)
        for work in (*self._pending.values(), *self._running.values()):
            other = _placement(work.proposal.action)
            if (
                other
                and other != placement
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

    def tick(self, *, force: bool = False) -> None:
        now = self.clock()
        self._reap()
        if not force and now - self._last_tick < self.settings.policy_interval_seconds:
            return
        self._last_tick = now
        registry = self.controller.registry
        for work in tuple(self._pending.values()):
            if work.generation != self._generation():
                self._cancel(work, "Cache reset invalidated the policy plan")
        ready = [w for w in self._pending.values() if w.proposal.not_before <= now]
        for work in ready[: self.settings.max_policy_actions_per_step]:
            del self._pending[work.operation.operation_id]
            action = work.proposal.action
            conflict = False
            if work.scope is not None:
                executor = self.controller.executor
                conflict = executor is not None and executor.conflicts_with_explicit(
                    action, work.scope
                )
                if isinstance(action, (DemotePayload, EvictPayload)):
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
                    work.operation, action, resolution=work.scope
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
                self._target_entries -= work.entries

    def _reap(self) -> None:
        registry = self.controller.registry
        for identifier, work in tuple(self._running.items()):
            current = registry.get(work.operation.message_id, work.operation.action_id)
            if current is None or current.status.terminal:
                del self._running[identifier]
                self._target_entries -= work.entries
        active_owners = {
            d.owner_id for d in self.controller.declarations.active_declarations()
        }
        self._declaration_owners = {
            key: value
            for key, value in self._declaration_owners.items()
            if key in active_owners
        }
