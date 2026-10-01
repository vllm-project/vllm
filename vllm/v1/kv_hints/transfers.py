# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correlate native movement with engine-owned buffers and adapter completion."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Protocol

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_transfer import KVCacheTransferError, KVCacheTransferLease
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_hints.control import (
    ControlError,
    OperationRegistry,
    OperationResult,
    OperationStatus,
)
from vllm.v1.kv_hints.native import DemotePayload, PrefetchPayload, SharePayload
from vllm.v1.kv_hints.state import LogicalResolution


@dataclass(frozen=True)
class KVTransferCompletion:
    transfer_id: int
    generation: int
    success: bool
    reason: str = ""
    missing: bool = False


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
    payload: DemotePayload | PrefetchPayload | SharePayload
    owners: frozenset[str]
    keys: set[BlockHashWithGroupId]
    pending: set[int] = field(default_factory=set)
    succeeded: set[BlockHashWithGroupId] = field(default_factory=set)
    errors: Counter[str] = field(default_factory=Counter)
    reasons: list[str] = field(default_factory=list)


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

    def submit(
        self,
        operation: OperationResult,
        payload: DemotePayload | PrefetchPayload | SharePayload,
        target: LogicalResolution,
    ) -> None:
        if isinstance(payload, PrefetchPayload):
            tier, action, direction = payload.source, "prefetch", "load"
        elif isinstance(payload, DemotePayload):
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
        for prefix in target.prefixes:
            for block in prefix.blocks:
                key = make_block_hash_with_group_id(
                    BlockHash(bytes.fromhex(block.block_hash)), block.group_id
                )
                if key in selected and selected[key] != block.end_token:
                    raise ControlError("invalid_payload", "Conflicting content spans")
                adapter.validate(direction_value, key, block.end_token)
                selected[key] = block.end_token
        movement = _Movement(
            operation,
            payload,
            frozenset(p.continuation[0] for p in target.prefixes if p.continuation),
            set(selected),
        )
        movement.errors["missing"] = sum(len(p.missing_ranges) for p in target.prefixes)
        movement.errors["unsupported"] = sum(
            len(p.partial_blocks) for p in target.prefixes
        )
        self.registry.update(operation, OperationStatus.ACCEPTED)
        self._operations[operation.operation_id] = movement
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
                continue
            if lease is None:
                movement.succeeded.add(key)
                continue
            if created:
                assert self.resources.start(lease)
                try:
                    adapter.submit(lease)
                except Exception as error:
                    self.resources.complete(lease, success=False)
                    code = error.code if isinstance(error, ControlError) else "failed"
                    self._record_error(movement, code, str(error))
                    continue
                self._jobs[lease.transfer_id] = _Job(lease)
            job = self._jobs.get(lease.transfer_id)
            if job is None:
                self.resources.cancel(lease, operation.operation_id)
                self._record_error(
                    movement, "busy", "Transfer belongs to another executor"
                )
                continue
            job.callers.add(operation.operation_id)
            movement.pending.add(lease.transfer_id)
        if movement.pending:
            self.registry.update(operation, OperationStatus.RUNNING)
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
                valid = self.resources.complete(job.lease, success=completion.success)
                del self._jobs[completion.transfer_id]
                for caller in job.callers:
                    movement = self._operations[caller]
                    movement.pending.remove(completion.transfer_id)
                    if valid:
                        movement.succeeded.add(job.lease.key)
                    else:
                        movement.errors[
                            "missing" if completion.missing else "failed"
                        ] += 1
                        if completion.reason and len(movement.reasons) < 3:
                            movement.reasons.append(completion.reason[:512])
                    if not movement.pending:
                        self._finish(movement)

    def cancel_owners(self, owners: tuple[str, ...]) -> None:
        for movement in tuple(self._operations.values()):
            if movement.owners.intersection(owners):
                self._cancel(movement, "Task finished")

    def _cancel(self, movement: _Movement, reason: str) -> None:
        caller = movement.operation.operation_id
        for job_id in movement.pending:
            job = self._jobs[job_id]
            job.callers.discard(caller)
            self.resources.cancel(job.lease, caller)
        self.registry.update(
            movement.operation,
            OperationStatus.CANCELLED,
            reason=reason,
            completed_steps=self._completed_steps(movement),
            affected_blocks=len(movement.succeeded),
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
        steps = self._completed_steps(movement)
        affected = len(movement.succeeded)
        blocked = errors["capacity"] + errors["busy"]
        if isinstance(movement.payload, PrefetchPayload):
            lost = {
                key
                for key in movement.succeeded
                if not pool.cached_block_hash_to_block.get_blocks(key)
            }
            movement.succeeded.difference_update(lost)
            errors["missing"] += len(lost)
            affected = len(movement.succeeded)
        elif isinstance(movement.payload, DemotePayload) and movement.succeeded:
            eviction = pool.evict_unreferenced(movement.succeeded)
            affected = eviction.num_evicted_blocks
            blocked += len(eviction.blocked)
            if eviction.evicted:
                steps += ("gpu_copies_reclaimed",)
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
            blocked_targets=blocked,
            missing_targets=errors["missing"] + errors["unsupported"],
            reason="; ".join(movement.reasons)
            or (
                "Target coverage is incomplete"
                if status != OperationStatus.COMPLETED
                else ""
            ),
        )
        del self._operations[movement.operation.operation_id]

    @staticmethod
    def _record_error(movement: _Movement, code: str, reason: str) -> None:
        movement.errors[code] += 1
        if len(movement.reasons) < 3:
            movement.reasons.append(reason[:512])
