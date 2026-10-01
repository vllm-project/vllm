# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side resource integration for programmable cache operations."""

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
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
from vllm.v1.kv_hints.controller import PhysicalPayload
from vllm.v1.kv_hints.native import (
    NATIVE_ACTION_VERSION,
    ActionCapability,
    DemotePayload,
    EvictPayload,
    PrefetchPayload,
    SharePayload,
)
from vllm.v1.kv_hints.retention import RetentionDeclarations
from vllm.v1.kv_hints.state import LogicalResolution
from vllm.v1.kv_hints.transfers import KVTransferAdapter, KVTransferOperations


class KVRetentionPriority:
    """Project declaration changes into the existing pool's eviction ordering."""

    def __init__(self, pool: BlockPool, declarations: RetentionDeclarations):
        self.declarations = declarations
        self._version: tuple[int, int] | None = None
        self._priorities: dict[BlockHashWithGroupId, int] = {}
        self.synchronize()
        pool.configure_retention(self.priority, self.synchronize)

    def priority(self, key: BlockHashWithGroupId) -> int:
        return self._priorities.get(key, 0)

    def synchronize(self) -> tuple[BlockHashWithGroupId, ...]:
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
            if self._priorities.get(key, 0) != updated.get(key, 0)
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
    ):
        self.manager = manager
        self.registry = registry
        self.declarations = declarations
        self.pool = manager.block_pool
        self.retention_priority = KVRetentionPriority(self.pool, declarations)
        self.transfers = (
            KVTransferOperations(
                manager, registry, declarations.index.namespace, adapters
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

    def submit(
        self,
        operation: OperationResult,
        payload: PhysicalPayload,
        target: LogicalResolution,
    ) -> None:
        self.validate_scope(target)
        if isinstance(payload, (DemotePayload, PrefetchPayload, SharePayload)):
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
        if self.transfers is not None:
            self.transfers.cancel_owners(owners)

    def progress(self) -> None:
        if self.transfers is not None:
            self.transfers.progress()
