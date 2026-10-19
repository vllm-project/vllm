# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side resource integration for programmable cache operations."""

import time
from collections.abc import Callable
from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_priority import KVCachePriority
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_hints.actions import MoveTo, Primitive, SetPriority
from vllm.v1.kv_hints.control import (
    ControlError,
    OperationRegistry,
    OperationResult,
    OperationStatus,
)
from vllm.v1.kv_hints.controller import PhysicalPayload
from vllm.v1.kv_hints.native import (
    NATIVE_ACTION_VERSION,
    ActionCapability,
    ContentSpan,
    DemotePayload,
    EvictPayload,
    NativePayload,
    PrefetchPayload,
    RequestTarget,
    SharePayload,
    StorePayload,
)
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import LogicalResolution, ResolvedPrefix
from vllm.v1.kv_hints.transfers import KVTransferAdapter, KVTransferOperations
from vllm.v1.request import Request


class KVRetentionPriority:
    """Project declaration changes into the existing pool's eviction ordering."""

    def __init__(self, pool: BlockPool, declarations: RetentionDeclarations):
        self.declarations = declarations
        self.pool = pool
        self.copies = KVCachePriority()
        self._version: tuple[int, int] | None = None
        self._priorities: dict[BlockHashWithGroupId, int] = {}
        self.synchronize()
        pool.configure_retention(
            self.priority,
            self.synchronize,
            copy_priority=self.copies.priority,
            forget_priority=self.copies.forget,
        )

    def priority(self, key: BlockHashWithGroupId) -> int | None:
        return self._priorities.get(key)

    def synchronize(self) -> tuple[BlockHashWithGroupId, ...]:
        changed_copies = self.copies.take_changes()
        if self.pool._retention_queue is not None:
            for block_id in changed_copies:
                self.pool._retention_queue.cache_changed(self.pool.blocks[block_id])
        version, snapshot = self.declarations.priority_snapshot()
        if version == self._version:
            return ()
        updated = {
            make_block_hash_with_group_id(
                BlockHash(bytes.fromhex(block_hash)), group
            ): p
            for (group, block_hash), p in snapshot.items()
        }
        changed = tuple(
            key
            for key in self._priorities.keys() | updated.keys()
            if self._priorities.get(key) != updated.get(key)
        )
        self._priorities = updated
        self._version = version
        return changed


class KVResourceController:
    """Approve physical work against the existing KV manager and block pool."""

    def __init__(
        self,
        manager: KVCacheManager,
        registry: OperationRegistry,
        declarations: RetentionDeclarations,
        adapters: tuple[KVTransferAdapter, ...] = (),
        transfer_allowed: Callable[[], bool] | None = None,
    ):
        self.manager = manager
        self.registry = registry
        self.declarations = declarations
        self.pool = manager.block_pool
        self.transfer_allowed = transfer_allowed
        self.retention_priority = KVRetentionPriority(self.pool, declarations)
        self.transfers = (
            KVTransferOperations(
                manager, registry, declarations.index.namespace, adapters
            )
            if adapters
            else None
        )

    def rebind(
        self,
        registry: OperationRegistry,
        declarations: RetentionDeclarations,
        adapters: tuple[KVTransferAdapter, ...],
    ) -> None:
        """Replace weight-scoped state after idle cache invalidation."""
        if self.has_pending or self.manager.get_transfer_manager().has_pending:
            raise ControlError("busy", "Transfers must drain before namespace change")
        self.registry = registry
        self.declarations = declarations
        self.retention_priority.declarations = declarations
        self.retention_priority._version = None
        for block in self.pool.blocks:
            self.retention_priority.copies.forget(block)
        self.retention_priority.synchronize()
        self.transfers = (
            KVTransferOperations(
                self.manager, registry, declarations.index.namespace, adapters
            )
            if adapters
            else None
        )

    def capabilities(self) -> tuple[ActionCapability, ...]:
        local = (
            ActionCapability(
                "evict",
                NATIVE_ACTION_VERSION,
                ("hashes", "session"),
                ("gpu",),
                True,
                ("Referenced copies and unaddressed aliases are preserved",),
            ),
        )
        if self.transfers is None:
            return local
        return local + tuple(
            ActionCapability(
                action,
                NATIVE_ACTION_VERSION,
                ("hashes", "session"),
                ("gpu", adapter.tier),
                True,
                adapter.limitations,
            )
            for adapter in self.transfers.adapters.values()
            for action in adapter.actions
        )

    def validate_scope(self, target: LogicalResolution) -> None:
        if target.namespace != self.declarations.index.namespace:
            raise ControlError(
                "incompatible", "Target belongs to another cache namespace"
            )
        if target.start_token > self.manager.max_model_len or (
            target.end_token is not None
            and target.end_token > self.manager.max_model_len
        ):
            raise ControlError(
                "invalid_payload", "Target exceeds the engine context limit"
            )
        for prefix in target.prefixes:
            for block in (*prefix.blocks, *prefix.partial_blocks):
                if not 0 <= block.group_id < self.manager.num_kv_cache_groups:
                    raise ControlError("unsupported", "Cache group is not configured")
                spec = self.manager.kv_cache_config.kv_cache_groups[
                    block.group_id
                ].kv_cache_spec
                if not spec.prefix_cacheable:
                    raise ControlError(
                        "unsupported", "Cache group has no reusable prefix entries"
                    )
                if block.end_token > self.manager.max_model_len:
                    raise ControlError(
                        "invalid_payload", "Content exceeds the engine context limit"
                    )
                block_size = self.manager.coordinator.single_type_managers[
                    block.group_id
                ].block_size
                if (
                    block.end_token % self.pool.hash_block_size
                    or block.start_token
                    != ((block.end_token - 1) // block_size) * block_size
                ):
                    raise ControlError(
                        "unsupported", "Content does not match a cache-group boundary"
                    )
                key = make_block_hash_with_group_id(
                    BlockHash(bytes.fromhex(block.block_hash)), block.group_id
                )
                for copy in self.pool.cached_block_hash_to_block.get_blocks(key):
                    if (
                        copy.block_hash == key
                        and copy.block_hash_num_tokens != block.end_token
                    ):
                        raise ControlError(
                            "invalid_payload",
                            "Content coordinates disagree with the cached copy",
                        )

    def residency(self, target: LogicalResolution, limit: int) -> dict[str, Any]:
        """Observe current GPU copies without acquiring a residency guarantee."""
        self.validate_scope(target)
        entries = sum(
            len(p.blocks) + len(p.partial_blocks) + len(p.missing_ranges)
            for p in target.prefixes
        )
        if entries > limit:
            raise ControlError("capacity", "Residency scope exceeds query limit")
        transfers = self.manager.transfer_manager
        rows = []
        for prefix in target.prefixes:
            for block in prefix.blocks:
                key = make_block_hash_with_group_id(
                    BlockHash(bytes.fromhex(block.block_hash)), block.group_id
                )
                copies = self.pool.cached_block_hash_to_block.get_blocks(key)
                rows.append(
                    {
                        "block_hash": block.block_hash,
                        "group_id": block.group_id,
                        "start_token": block.start_token,
                        "end_token": block.end_token,
                        "resident_copies": len(copies),
                        "referenced_copies": sum(b.ref_cnt > 0 for b in copies),
                        "eviction_priorities": tuple(
                            self.pool.eviction_priority(b) for b in copies
                        ),
                        "loading": transfers is not None and transfers.is_loading(key),
                    }
                )
        return {
            "tier": "gpu",
            "namespace": target.namespace,
            "observed_at": time.monotonic(),
            "cache_generation": transfers.generation if transfers is not None else 0,
            "total_blocks": self.pool.num_gpu_blocks - 1,
            "free_blocks": self.pool.get_num_free_blocks(),
            "items": rows,
            "partial_blocks": tuple(
                b for p in target.prefixes for b in p.partial_blocks
            ),
            "missing_ranges": tuple(
                g for p in target.prefixes for g in p.missing_ranges
            ),
        }

    def request_scope(
        self, request: Request, target: RequestTarget | str
    ) -> LogicalResolution:
        if (
            request.resumable
            or request.mm_features
            or request.lora_request is not None
            or request.prompt_embeds is not None
        ):
            raise ControlError(
                "unsupported", "Primitive request scope requires finite text tokens"
            )
        start = target.start_token if isinstance(target, RequestTarget) else 0
        end = target.end_token if isinstance(target, RequestTarget) else None
        blocks: set[ContentSpan] = set()
        partial: set[ContentSpan] = set()
        for group_id, group in enumerate(
            self.manager.get_blocks(request.request_id).blocks
        ):
            size = self.manager.coordinator.single_type_managers[group_id].block_size
            for block in group:
                key, boundary = block.block_hash, block.block_hash_num_tokens
                if (
                    block.pool is not self.pool
                    or block.is_null
                    or key is None
                    or boundary is None
                ):
                    continue
                begin = (boundary - 1) // size * size
                if boundary <= start or (end is not None and begin >= end):
                    continue
                span = ContentSpan(get_block_hash(key).hex(), group_id, begin, boundary)
                (
                    partial
                    if begin < start or (end is not None and boundary > end)
                    else blocks
                ).add(span)
        order = lambda b: (b.group_id, b.start_token, b.end_token, b.block_hash)
        return LogicalResolution(
            self.declarations.index.namespace,
            start,
            end,
            (
                ResolvedPrefix(
                    None,
                    None,
                    tuple(sorted(blocks, key=order)),
                    tuple(sorted(partial, key=order)),
                ),
            ),
        )

    def apply_priority(
        self,
        operation: OperationResult,
        action: SetPriority,
        scope: LogicalResolution,
        request: Request | None,
        owners: tuple[str, ...] = (),
    ) -> None:
        self.validate_scope(scope)
        keys = {
            make_block_hash_with_group_id(
                BlockHash(bytes.fromhex(b.block_hash)), b.group_id
            )
            for p in scope.prefixes
            for b in p.blocks
        }
        if action.target == "request_cached" and request is not None:
            blocks = [
                b
                for group in self.manager.get_blocks(request.request_id).blocks
                for b in group
                if b.pool is self.pool and b.block_hash in keys
            ]
        else:
            blocks = [
                b
                for key in keys
                for b in self.pool.cached_block_hash_to_block.get_blocks(key)
            ]
        changed = self.retention_priority.copies.apply(
            action,
            blocks,
            policy_name=operation.origin.policy_name if operation.origin else None,
            owners=tuple(
                set(owners)
                | {p.continuation[0] for p in scope.prefixes if p.continuation}
            ),
        )
        missing = sum(
            len(p.partial_blocks) + len(p.missing_ranges) for p in scope.prefixes
        )
        missing += sum(
            not self.pool.cached_block_hash_to_block.get_blocks(key) for key in keys
        )
        self.registry.update(
            operation,
            OperationStatus.PARTIAL if missing else OperationStatus.COMPLETED,
            affected_blocks=len(changed),
            missing_targets=missing,
            completed_steps=(
                ("priority_applied" if action.value is not None else "priority_cleared")
                if changed
                else "priority_unchanged",
            ),
            reason="Duplicate/stale claim revision" if not changed else "",
        )

    def submit(
        self,
        operation: OperationResult,
        payload: PhysicalPayload,
        target: LogicalResolution,
    ) -> None:
        self.validate_scope(target)
        if isinstance(
            payload, (DemotePayload, PrefetchPayload, SharePayload, StorePayload)
        ):
            if self.transfer_allowed is not None and not self.transfer_allowed():
                raise ControlError("busy", "Engine is paused or sleeping")
            if self.transfers is None:
                raise ControlError("unsupported", "No transfer adapter is configured")
            self.transfers.submit(operation, payload, target)
            return
        if not isinstance(payload, EvictPayload) or payload.tier != "gpu":
            raise ControlError(
                "unsupported", "No transfer adapter is configured for this operation"
            )
        selected = {
            make_block_hash_with_group_id(
                BlockHash(bytes.fromhex(block.block_hash)), block.group_id
            )
            for prefix in target.prefixes
            for block in prefix.blocks
        }
        partial_keys = {
            make_block_hash_with_group_id(
                BlockHash(bytes.fromhex(block.block_hash)), block.group_id
            )
            for prefix in target.prefixes
            for block in prefix.partial_blocks
        }
        gaps = sum(len(prefix.missing_ranges) for prefix in target.prefixes)
        partial = bool(partial_keys or gaps)
        outcome = self.pool.evict_unreferenced(selected)
        if outcome.evicted:
            status = (
                OperationStatus.PARTIAL
                if outcome.blocked or outcome.missing or partial
                else OperationStatus.COMPLETED
            )
        elif outcome.blocked:
            status = OperationStatus.DEFERRED
        elif outcome.missing:
            status = OperationStatus.MISSING
        elif partial:
            status = OperationStatus.UNSUPPORTED
        else:
            status = OperationStatus.COMPLETED
        self.registry.update(
            operation,
            status,
            reason="Range or copy coverage is incomplete"
            if partial or outcome.blocked
            else "",
            completed_steps=("gpu_copies_reclaimed",) if outcome.evicted else (),
            affected_blocks=outcome.num_evicted_blocks,
            blocked_targets=len(outcome.blocked | partial_keys),
            missing_targets=len(outcome.missing) + gaps,
        )

    def cancel_owners(self, owners: tuple[str, ...]) -> None:
        self.retention_priority.copies.finish_owners(owners)
        if self.transfers is not None:
            self.transfers.cancel_owners(owners)

    def progress(self) -> None:
        if self.transfers is not None:
            self.transfers.progress()

    @property
    def has_pending(self) -> bool:
        """Whether a request-free transfer still needs engine progress."""
        return self.transfers is not None and self.transfers.has_pending

    def cancel_operations(self, operation_ids: set[int], reason: str) -> None:
        if self.transfers is not None:
            self.transfers.cancel_operations(operation_ids, reason)

    def conflicts_with_explicit(
        self,
        payload: NativePayload | Primitive,
        scope: LogicalResolution,
    ) -> bool:
        keys = {
            make_block_hash_with_group_id(
                BlockHash(bytes.fromhex(b.block_hash)), b.group_id
            )
            for prefix in scope.prefixes
            for b in prefix.blocks
        }
        if isinstance(
            payload, (DemotePayload, MoveTo, EvictPayload)
        ) and self.retention_priority.copies.has_explicit(
            b
            for key in keys
            for b in self.pool.cached_block_hash_to_block.get_blocks(key)
        ):
            return True
        return self.transfers is not None and self.transfers.conflicts_with_explicit(
            payload, keys
        )
