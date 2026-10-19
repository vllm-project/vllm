# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-independent transfer references owned by the KV cache manager."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    get_group_id,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager


class KVCacheTransferError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class KVCacheTransferLease:
    transfer_id: int
    generation: int
    direction: Literal["load", "store"]
    scope: str
    key: BlockHashWithGroupId
    num_tokens: int
    block_id: int


@dataclass
class _Transfer:
    lease: KVCacheTransferLease
    block: KVCacheBlock
    interests: set[int] = field(default_factory=set)
    started: bool = False
    completed: bool = False


class KVCacheTransferManager:
    """Hold buffers until all worker access ends, independently of callers.

    Callers join one physical transfer per direction/content key. The executor
    must call ``complete`` only after every participating worker has stopped
    accessing the buffer, including workers whose transfer failed or was cancelled.
    """

    def __init__(self, manager: "KVCacheManager"):
        self.manager = manager
        self.pool = manager.block_pool
        self.generation = 0
        self._next_id = 0
        self._transfers: dict[int, _Transfer] = {}
        self._by_key: dict[tuple[str, str, BlockHashWithGroupId], int] = {}
        self._loads: dict[BlockHashWithGroupId, int] = {}
        self.max_load_blocks = 0

    @property
    def has_pending(self) -> bool:
        return bool(self._transfers)

    def held_block_counts(self) -> tuple[int, int]:
        """Return destination and source counts, already charged in pool usage."""
        loads = {
            t.block.block_id
            for t in self._transfers.values()
            if t.lease.direction == "load"
        }
        stores = {
            t.block.block_id
            for t in self._transfers.values()
            if t.lease.direction == "store"
        }
        return len(loads), len(stores)

    @property
    def reserved_load_blocks(self) -> int:
        return len(self._loads)

    def load_requirements(
        self,
        blocks: Iterable[tuple[BlockHashWithGroupId, int]],
        *,
        scope: str,
    ) -> tuple[int, int] | None:
        """Return new load slots and free blocks a complete preparation needs."""
        loads = committed = 0
        for key, num_tokens in blocks:
            existing = self._by_key.get(("load", scope, key))
            if existing is not None:
                transfer = self._transfers[existing]
                if (
                    transfer.lease.generation != self.generation
                    or not transfer.interests
                ):
                    return None
                continue
            copies = self.pool.cached_block_hash_to_block.get_blocks(key)
            source = next(
                (
                    b
                    for b in copies
                    if b.block_hash == key and b.block_hash_num_tokens == num_tokens
                ),
                None,
            )
            if source is not None:
                committed += source.ref_cnt == 0
            elif key in self._loads:
                return None
            elif not copies:
                loads += 1
                committed += 1
        return loads, committed

    def is_loading(self, key: BlockHashWithGroupId) -> bool:
        identifier = self._loads.get(key)
        transfer = self._transfers.get(identifier) if identifier is not None else None
        return bool(
            transfer is not None
            and transfer.lease.generation == self.generation
            and transfer.interests
            and not transfer.completed
        )

    def acquire(
        self,
        direction: Literal["load", "store"],
        key: BlockHashWithGroupId,
        num_tokens: int,
        interest: int,
        *,
        scope: str,
    ) -> tuple[KVCacheTransferLease | None, bool]:
        """Approve one buffer, returning its lease and whether work is new.

        A load of already resident content returns ``(None, False)``. Capacity,
        missing sources, invalid coordinates and draining work raise an error
        without acquiring resources. Repeated interests do not add references.
        """
        group = get_group_id(key)
        if direction not in ("load", "store"):
            raise KVCacheTransferError("unsupported", "Unknown transfer direction")
        if (
            not self.manager.enable_caching
            or not 0 <= group < self.manager.num_kv_cache_groups
            or not self.manager.kv_cache_config.kv_cache_groups[
                group
            ].kv_cache_spec.prefix_cacheable
            or self.manager.coordinator.single_type_managers[group].block_pool
            is not self.pool
        ):
            raise KVCacheTransferError("unsupported", "Group is not prefix cacheable")
        if not 0 < num_tokens <= self.manager.max_model_len or (
            num_tokens % self.pool.hash_block_size
        ):
            raise KVCacheTransferError("invalid_payload", "Invalid content boundary")

        lookup = (direction, scope, key)
        existing = self._by_key.get(lookup)
        if existing is not None:
            transfer = self._transfers[existing]
            if transfer.lease.num_tokens != num_tokens:
                raise KVCacheTransferError("invalid_payload", "Conflicting coordinates")
            if transfer.lease.generation != self.generation or not transfer.interests:
                raise KVCacheTransferError("busy", "Previous transfer is draining")
            if transfer.completed:
                return None, False
            transfer.interests.add(interest)
            return transfer.lease, False

        copies = self.pool.cached_block_hash_to_block.get_blocks(key)
        # A primary entry has an exact boundary. Aliases lack separate boundary
        # metadata, so adapters must request the primary entry for movement.
        source = next(
            (
                block
                for block in copies
                if block.block_hash == key and block.block_hash_num_tokens == num_tokens
            ),
            None,
        )
        if copies and source is None:
            raise KVCacheTransferError(
                "unsupported", "Transfer requires an exact primary cache boundary"
            )
        if direction == "load":
            if source is not None:
                return None, False
            if key in self._loads:
                raise KVCacheTransferError("busy", "Another source is loading this key")
            if self.max_load_blocks and len(self._loads) >= self.max_load_blocks:
                raise KVCacheTransferError(
                    "capacity", "In-flight load budget exhausted"
                )
            if self.pool.get_num_free_blocks() <= self.manager.watermark_blocks:
                raise KVCacheTransferError("capacity", "No destination capacity")
            block = self.pool.get_new_blocks(1)[0]
        else:
            if source is None:
                raise KVCacheTransferError("missing", "Source is not resident")
            block = source
            self.pool.touch([block])

        self._next_id += 1
        lease = KVCacheTransferLease(
            self._next_id,
            self.generation,
            direction,
            scope,
            key,
            num_tokens,
            block.block_id,
        )
        self._transfers[lease.transfer_id] = _Transfer(lease, block, {interest})
        self._by_key[lookup] = lease.transfer_id
        if direction == "load":
            self._loads[key] = lease.transfer_id
        return lease, True

    def hold_resident(
        self, key: BlockHashWithGroupId, num_tokens: int, interest: int, *, scope: str
    ) -> KVCacheTransferLease:
        """Keep an already validated resident member of a preparation stable."""
        lookup = ("load", scope, key)
        existing = self._by_key.get(lookup)
        if existing is not None:
            transfer = self._transfers[existing]
            if (
                not transfer.completed
                or transfer.lease.generation != self.generation
                or transfer.lease.num_tokens != num_tokens
            ):
                raise KVCacheTransferError("busy", "Resident copy is not ready")
            transfer.interests.add(interest)
            return transfer.lease
        source = next(
            (
                block
                for block in self.pool.cached_block_hash_to_block.get_blocks(key)
                if block.block_hash == key and block.block_hash_num_tokens == num_tokens
            ),
            None,
        )
        if source is None:
            raise KVCacheTransferError(
                "missing", "Resident copy is no longer available"
            )
        self.pool.touch([source])
        self._next_id += 1
        lease = KVCacheTransferLease(
            self._next_id,
            self.generation,
            "load",
            scope,
            key,
            num_tokens,
            source.block_id,
        )
        self._transfers[lease.transfer_id] = _Transfer(
            lease, source, {interest}, started=True, completed=True
        )
        self._by_key[lookup] = lease.transfer_id
        return lease

    def start(self, lease: KVCacheTransferLease) -> bool:
        """Mark dispatch before a worker can access the buffer, exactly once."""
        transfer = self._get(lease)
        if transfer is None or transfer.started or not transfer.interests:
            return False
        if lease.generation != self.generation:
            return False
        transfer.started = True
        return True

    def cancel(self, lease: KVCacheTransferLease, interest: int) -> None:
        transfer = self._get(lease)
        if transfer is None:
            return
        transfer.interests.discard(interest)
        if not transfer.interests:
            if transfer.completed:
                self._release(transfer)
            elif not transfer.started:
                self.complete(lease, success=False)

    def invalidate(self) -> None:
        """Reject stale publication while retaining every dispatched buffer."""
        self.generation += 1
        for transfer in tuple(self._transfers.values()):
            transfer.interests.clear()
            if transfer.completed:
                self._release(transfer)
            elif not transfer.started:
                self.complete(transfer.lease, success=False)

    def complete(
        self, lease: KVCacheTransferLease, *, success: bool, retain: bool = False
    ) -> bool:
        """Drain a physical transfer; return whether its outcome is still valid."""
        transfer = self._get(lease)
        if transfer is None or transfer.completed:
            return False
        valid = (
            success
            and transfer.started
            and bool(transfer.interests)
            and lease.generation == self.generation
        )
        if valid and lease.direction == "load":
            self.pool.cache_transferred_block(
                transfer.block, lease.key, lease.num_tokens
            )
            if retain:
                transfer.completed = True
                return True
        self._release(transfer)
        return valid

    def _release(self, transfer: _Transfer) -> None:
        lease = transfer.lease
        del self._transfers[lease.transfer_id]
        del self._by_key[(lease.direction, lease.scope, lease.key)]
        if self._loads.get(lease.key) == lease.transfer_id:
            del self._loads[lease.key]
        self.pool.free_blocks([transfer.block])

    def _get(self, lease: KVCacheTransferLease) -> _Transfer | None:
        transfer = self._transfers.get(lease.transfer_id)
        return transfer if transfer is not None and transfer.lease is lease else None
