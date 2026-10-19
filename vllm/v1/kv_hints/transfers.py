# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlate native movement with engine-owned buffers and adapter completion."""

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_transfer import KVCacheTransferError, KVCacheTransferLease
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_hints.actions import CopyTo, MoveTo, Primitive, SetPriority
from vllm.v1.kv_hints.control import (
    ControlError,
    OperationRegistry,
    OperationResult,
    OperationStatus,
    TargetOutcome,
)
from vllm.v1.kv_hints.native import (
    DemotePayload,
    EvictPayload,
    NativePayload,
    PrefetchPayload,
    RetainPayload,
    SharePayload,
    StorePayload,
)
from vllm.v1.kv_hints.state import LogicalResolution


@dataclass(frozen=True)
class KVTransferCompletion:
    transfer_id: int
    generation: int
    success: bool
    reason: str = ""
    missing: bool = False
    bytes_transferred: int = 0


class KVTransferAdapter(Protocol):
    tier: Literal["cpu", "mooncake"]
    actions: tuple[str, ...]
    limitations: tuple[str, ...]

    def validate(
        self,
        direction: Literal["load", "store"],
        key: BlockHashWithGroupId,
        num_tokens: int,
    ) -> None: ...

    def submit(self, lease: KVCacheTransferLease) -> None:
        """Queue local metadata; exceptions must leave no queued or worker access."""
        ...

    def poll(self) -> tuple[KVTransferCompletion, ...]:
        """Report completion only after every worker has ceased buffer access."""
        ...


@dataclass
class _Movement:
    operation: OperationResult
    payload: DemotePayload | PrefetchPayload | SharePayload | StorePayload
    owners: frozenset[str]
    keys: set[BlockHashWithGroupId]
    pending: set[int] = field(default_factory=set)
    succeeded: set[BlockHashWithGroupId] = field(default_factory=set)
    errors: Counter[str] = field(default_factory=Counter)
    reasons: list[str] = field(default_factory=list)
    outcomes: dict[BlockHashWithGroupId, TargetOutcome] = field(default_factory=dict)
    unresolved: tuple[TargetOutcome, ...] = ()
    transferred_bytes: int = 0
    completed_holds: set[KVCacheTransferLease] = field(default_factory=set)


@dataclass
class _Job:
    lease: KVCacheTransferLease
    callers: set[int] = field(default_factory=set)


class KVTransferOperations:
    def __init__(
        self,
        manager: KVCacheManager,
        registry: OperationRegistry,
        namespace: str,
        adapters: tuple[KVTransferAdapter, ...],
    ):
        self.manager = manager
        self.registry = registry
        self.namespace = namespace
        self.adapters = {adapter.tier: adapter for adapter in adapters}
        if len(self.adapters) != len(adapters):
            raise ValueError("Only one native transfer adapter may own each tier")
        self.resources = manager.get_transfer_manager()
        self._operations: dict[int, _Movement] = {}
        self._jobs: dict[int, _Job] = {}
        self.statistics = {
            "load_bytes": 0,
            "store_bytes": 0,
            "completed_transfers": 0,
            "failed_transfers": 0,
            "missing_transfers": 0,
            "discarded_transfers": 0,
        }

    @property
    def has_pending(self) -> bool:
        """Whether request-free work still needs an engine progress tick."""
        return bool(self._operations or self._jobs)

    def submit(
        self,
        operation: OperationResult,
        payload: DemotePayload | PrefetchPayload | SharePayload | StorePayload,
        target: LogicalResolution,
    ) -> None:
        if isinstance(payload, PrefetchPayload):
            tier, action, direction = payload.source, "prefetch", "load"
        elif isinstance(payload, (DemotePayload, StorePayload)):
            tier, action, direction = payload.destination, "demote", "store"
        else:
            tier, action, direction = "mooncake", "share", "store"
        adapter = self.adapters.get(tier)
        if adapter is None or action not in adapter.actions:
            raise ControlError("unsupported", "No adapter supports this movement")
        direction_value: Literal["load", "store"] = (
            "load" if direction == "load" else "store"
        )
        selected: dict[BlockHashWithGroupId, int] = {}
        outcomes = {}
        for prefix in target.prefixes:
            for block in prefix.blocks:
                key = make_block_hash_with_group_id(
                    BlockHash(bytes.fromhex(block.block_hash)), block.group_id
                )
                if key in selected and selected[key] != block.end_token:
                    raise ControlError("invalid_payload", "Conflicting content spans")
                adapter.validate(direction_value, key, block.end_token)
                selected[key] = block.end_token
                outcomes[key] = TargetOutcome(
                    block.block_hash,
                    block.group_id,
                    block.start_token,
                    block.end_token,
                    OperationStatus.RECEIVED,
                )
                if len(outcomes) > self.registry.max_result_targets:
                    raise ControlError(
                        "capacity", "Operation target result capacity exceeded"
                    )
        unresolved = tuple(
            TargetOutcome(
                b.block_hash,
                b.group_id,
                b.start_token,
                b.end_token,
                OperationStatus.UNSUPPORTED,
                reason="Range splits content",
            )
            for p in target.prefixes
            for b in p.partial_blocks
        ) + tuple(
            TargetOutcome(
                None,
                gap.group_id,
                gap.start_token,
                gap.end_token,
                OperationStatus.MISSING,
                reason="Unrepresented range",
            )
            for p in target.prefixes
            for gap in p.missing_ranges
        )
        if len(outcomes) + len(unresolved) > self.registry.max_result_targets:
            raise ControlError("capacity", "Operation target result capacity exceeded")
        movement = _Movement(
            operation,
            payload,
            frozenset(p.continuation[0] for p in target.prefixes if p.continuation),
            set(selected),
            outcomes=outcomes,
            unresolved=unresolved,
        )
        movement.errors["missing"] = sum(len(p.missing_ranges) for p in target.prefixes)
        movement.errors["unsupported"] = sum(
            len(p.partial_blocks) for p in target.prefixes
        )
        self._operations[operation.operation_id] = movement
        approved = False
        for key, num_tokens in selected.items():
            try:
                lease, created = self.resources.acquire(
                    direction_value,
                    key,
                    num_tokens,
                    operation.operation_id,
                    scope=f"{tier}/{self.namespace}",
                )
            except KVCacheTransferError as error:
                self._record_error(movement, error.code, str(error))
                self._target_error(movement, key, error.code, str(error))
                continue
            if lease is None:
                movement.completed_holds.add(
                    self.resources.hold_resident(
                        key,
                        num_tokens,
                        operation.operation_id,
                        scope=f"{tier}/{self.namespace}",
                    )
                )
                movement.succeeded.add(key)
                self._target_success(movement, key)
                movement.outcomes[key] = replace(
                    movement.outcomes[key], reason="Already resident"
                )
                continue
            movement.outcomes[key] = replace(
                movement.outcomes[key],
                transfer_id=lease.transfer_id,
                generation=lease.generation,
            )
            if not approved:
                self.registry.update(operation, OperationStatus.ACCEPTED)
                approved = True
            if created:
                assert self.resources.start(lease)
                try:
                    adapter.submit(lease)
                except Exception as error:
                    self.resources.complete(lease, success=False)
                    code = error.code if isinstance(error, ControlError) else "failed"
                    self._record_error(movement, code, str(error))
                    self._target_error(movement, key, code, str(error))
                    continue
                self._jobs[lease.transfer_id] = _Job(lease)
            job = self._jobs.get(lease.transfer_id)
            if job is None:
                self.resources.cancel(lease, operation.operation_id)
                self._record_error(
                    movement, "busy", "Transfer belongs to another executor"
                )
                self._target_error(
                    movement, key, "busy", "Transfer belongs to another executor"
                )
                continue
            job.callers.add(operation.operation_id)
            movement.pending.add(lease.transfer_id)
        if movement.pending:
            self.registry.update(
                operation,
                OperationStatus.RUNNING,
                target_outcomes=tuple(
                    replace(o, status=OperationStatus.RUNNING)
                    if o.status == OperationStatus.RECEIVED
                    else o
                    for o in movement.outcomes.values()
                )
                + movement.unresolved,
            )
        else:
            self._finish(movement)

    def progress(self) -> None:
        for movement in tuple(self._operations.values()):
            if any(
                self._jobs[job_id].lease.generation != self.resources.generation
                for job_id in movement.pending
            ):
                self._cancel(movement, "Cache reset invalidated the operation")
        for adapter in self.adapters.values():
            for completion in adapter.poll():
                job = self._jobs.get(completion.transfer_id)
                if (
                    job is None
                    or job.lease.generation != completion.generation
                    or job.lease.scope != f"{adapter.tier}/{self.namespace}"
                ):
                    continue
                valid = self.resources.complete(
                    job.lease,
                    success=completion.success,
                    retain=job.lease.direction == "load",
                )
                self.statistics[f"{job.lease.direction}_bytes"] += (
                    completion.bytes_transferred
                )
                outcome = (
                    "completed_transfers"
                    if valid
                    else "missing_transfers"
                    if completion.missing
                    else "discarded_transfers"
                    if completion.success
                    else "failed_transfers"
                )
                self.statistics[outcome] += 1
                del self._jobs[completion.transfer_id]
                for caller in job.callers:
                    movement = self._operations[caller]
                    movement.transferred_bytes += completion.bytes_transferred
                    movement.pending.remove(completion.transfer_id)
                    if valid:
                        if job.lease.direction == "load":
                            movement.completed_holds.add(job.lease)
                        movement.succeeded.add(job.lease.key)
                        self._target_success(movement, job.lease.key)
                    else:
                        movement.errors[
                            "missing" if completion.missing else "failed"
                        ] += 1
                        if completion.reason and len(movement.reasons) < 3:
                            movement.reasons.append(completion.reason[:512])
                        self._target_error(
                            movement,
                            job.lease.key,
                            "missing" if completion.missing else "failed",
                            completion.reason,
                        )
                    if not movement.pending:
                        self._finish(movement)

    def cancel_owners(self, owners: tuple[str, ...]) -> None:
        for movement in tuple(self._operations.values()):
            if movement.owners.intersection(owners):
                self._cancel(movement, "Task finished")

    def cancel_operations(self, operation_ids: set[int], reason: str) -> None:
        for operation_id in operation_ids:
            movement = self._operations.get(operation_id)
            if movement is not None:
                self._cancel(movement, reason)

    def conflicts_with_explicit(
        self,
        payload: NativePayload | Primitive,
        keys: set[BlockHashWithGroupId],
    ) -> bool:
        for movement in self._operations.values():
            if movement.operation.origin is not None or not movement.keys.intersection(
                keys
            ):
                continue
            if isinstance(payload, EvictPayload):
                return True
            if (
                isinstance(payload, (RetainPayload, PrefetchPayload, SetPriority))
                or (isinstance(payload, CopyTo) and payload.dst.tier == "gpu")
            ) and isinstance(movement.payload, DemotePayload):
                return True
            if isinstance(payload, (DemotePayload, MoveTo)) and isinstance(
                movement.payload, PrefetchPayload
            ):
                return True
        return False

    def _cancel(self, movement: _Movement, reason: str) -> None:
        caller = movement.operation.operation_id
        for job_id in movement.pending:
            job = self._jobs[job_id]
            job.callers.discard(caller)
            self.resources.cancel(job.lease, caller)
        for lease in movement.completed_holds:
            self.resources.cancel(lease, caller)
        self.registry.update(
            movement.operation,
            OperationStatus.CANCELLED,
            reason=reason,
            completed_steps=self._completed_steps(movement),
            affected_blocks=len(movement.succeeded),
            transferred_bytes=movement.transferred_bytes,
            target_outcomes=tuple(
                replace(outcome, status=OperationStatus.CANCELLED, reason=reason)
                if outcome.status == OperationStatus.RECEIVED
                else outcome
                for outcome in movement.outcomes.values()
            )
            + movement.unresolved,
        )
        del self._operations[caller]

    @staticmethod
    def _completed_steps(movement: _Movement) -> tuple[str, ...]:
        if not movement.succeeded:
            return ()
        return (
            ("gpu_content_registered",)
            if isinstance(movement.payload, PrefetchPayload)
            else ("cold_copy_stored",)
        )

    def _finish(self, movement: _Movement) -> None:
        errors = movement.errors
        pool = self.manager.block_pool
        affected = len(movement.succeeded)
        blocked = errors["capacity"] + errors["busy"]
        if isinstance(movement.payload, PrefetchPayload):
            lost = {
                key
                for key in movement.succeeded
                if not pool.cached_block_hash_to_block.get_blocks(key)
            }
            movement.succeeded.difference_update(lost)
            for key in lost:
                self._target_error(
                    movement,
                    key,
                    "missing",
                    "Destination was displaced before operation completion",
                )
            errors["missing"] += len(lost)
            affected = len(movement.succeeded)
        steps = self._completed_steps(movement)
        if isinstance(movement.payload, DemotePayload) and movement.succeeded:
            eviction = pool.evict_unreferenced(movement.succeeded)
            affected = eviction.num_evicted_blocks
            blocked += len(eviction.blocked)
            if eviction.evicted:
                steps += ("gpu_copies_reclaimed",)
            for key in movement.succeeded:
                outcome = movement.outcomes[key]
                movement.outcomes[key] = replace(
                    outcome,
                    status=OperationStatus.PARTIAL
                    if key in eviction.blocked
                    else OperationStatus.COMPLETED,
                    completed_steps=outcome.completed_steps
                    + (("gpu_copies_reclaimed",) if key in eviction.evicted else ()),
                    reason="Source remains referenced"
                    if key in eviction.blocked
                    else "",
                )
        if movement.succeeded:
            status = (
                OperationStatus.PARTIAL
                if sum(errors.values()) or blocked
                else OperationStatus.COMPLETED
            )
        elif errors["failed"]:
            status = OperationStatus.FAILED
        elif blocked:
            status = OperationStatus.DEFERRED
        elif errors["missing"]:
            status = OperationStatus.MISSING
        elif errors["unsupported"]:
            status = OperationStatus.UNSUPPORTED
        elif sum(errors.values()):
            status = OperationStatus.FAILED
        else:
            status = OperationStatus.COMPLETED
        self.registry.update(
            movement.operation,
            status,
            completed_steps=steps,
            affected_blocks=affected,
            transferred_bytes=movement.transferred_bytes,
            blocked_targets=blocked,
            missing_targets=errors["missing"] + errors["unsupported"],
            target_outcomes=tuple(movement.outcomes.values()) + movement.unresolved,
            reason="; ".join(movement.reasons)
            or (
                "Target coverage is incomplete"
                if status != OperationStatus.COMPLETED
                else ""
            ),
        )
        for lease in movement.completed_holds:
            self.resources.cancel(lease, movement.operation.operation_id)
        del self._operations[movement.operation.operation_id]

    @staticmethod
    def _target_success(movement: _Movement, key: BlockHashWithGroupId) -> None:
        steps = (
            ("gpu_content_registered",)
            if isinstance(movement.payload, PrefetchPayload)
            else ("cold_copy_stored",)
        )
        movement.outcomes[key] = replace(
            movement.outcomes[key],
            status=OperationStatus.COMPLETED,
            completed_steps=steps,
        )

    @staticmethod
    def _target_error(
        movement: _Movement, key: BlockHashWithGroupId, code: str, reason: str
    ) -> None:
        status = {
            "capacity": OperationStatus.DEFERRED,
            "busy": OperationStatus.DEFERRED,
            "missing": OperationStatus.MISSING,
            "unsupported": OperationStatus.UNSUPPORTED,
        }.get(code, OperationStatus.FAILED)
        movement.outcomes[key] = replace(
            movement.outcomes[key], status=status, reason=reason[:512]
        )

    @staticmethod
    def _record_error(movement: _Movement, code: str, reason: str) -> None:
        movement.errors[code] += 1
        if len(movement.reasons) < 3:
            movement.reasons.append(reason[:512])
