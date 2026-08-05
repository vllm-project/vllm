# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vllm.distributed.kv_events import KVCacheEvent
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_unique_kv_cache_group_id
from vllm.v1.core.single_type_kv_cache_manager import (
    HiSparseHotManager,
    HiSparseResidentManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import (
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
)
from vllm.v1.kv_offload.sparse.base import (
    SparseKVOffloadCommand,
    SparseKVPageTransfer,
)


@dataclass
class _PendingSpill:
    transfer_id: int
    page: tuple[str, int]
    host_block: KVCacheBlock
    resident_blocks: tuple[KVCacheBlock, ...]
    release_after: bool
    copy_enqueued: bool = False


class HiSparseManager:
    """Own HiSparse host allocation, prefix state, and GPU residency transitions."""

    @staticmethod
    def create_host_block_pool(
        kv_cache_config: KVCacheConfig,
        *,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool,
        metrics_collector: KVCacheMetricsCollector | None,
    ) -> BlockPool | None:
        num_blocks = kv_cache_config.hisparse_host_num_blocks
        if num_blocks is None:
            return None
        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
            block_pool_id=None,
        )

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        managers: tuple[SingleTypeKVCacheManager, ...],
        max_model_len: int,
    ) -> None:
        self.managers = managers
        groups = kv_cache_config.kv_cache_groups

        resident_managers: list[HiSparseResidentManager] = []
        hot_managers: list[HiSparseHotManager] = []
        hisparse_pool_ids: set[int] = set()
        for group, manager in zip(groups, managers):
            if isinstance(group.kv_cache_spec, HiSparseResidentSpec):
                if not isinstance(manager, HiSparseResidentManager):
                    raise TypeError(
                        "HiSparse resident specs require resident cache managers."
                    )
                resident_managers.append(manager)
                assert group.block_pool_id is not None
                hisparse_pool_ids.add(group.block_pool_id)
            if isinstance(manager, HiSparseHotManager):
                hot_managers.append(manager)
                assert group.block_pool_id is not None
                hisparse_pool_ids.add(group.block_pool_id)

        if len(hisparse_pool_ids) > 1:
            raise ValueError(
                "HiSparse resident and hot groups must share one GPU block pool."
            )
        self.block_pool_id = next(iter(hisparse_pool_ids), None)
        self.resident_managers = tuple(resident_managers)
        self.hot_managers = tuple(hot_managers)
        self.num_growing_managers = sum(
            group.block_pool_id == self.block_pool_id
            and not isinstance(manager, HiSparseHotManager)
            for group, manager in zip(groups, managers)
        )
        self.host_manager: SingleTypeKVCacheManager | None = None
        self.pages_per_host_block = 0
        self.max_spill_pages = 0
        if kv_cache_config.hisparse_host_num_blocks is not None:
            host_group_id = get_unique_kv_cache_group_id(
                kv_cache_config, KVCacheGroupRole.HISPARSE_SOURCE
            )
            self.host_manager = managers[host_group_id]
        self.has_host_cache = self.host_manager is not None
        if self.resident_managers:
            assert self.host_manager is not None
            resident_block_sizes = {
                manager.block_size for manager in self.resident_managers
            }
            if len(resident_block_sizes) != 1:
                raise ValueError(
                    "HiSparse resident cache groups must use one block size."
                )
            resident_block_size = resident_block_sizes.pop()
            if self.host_manager.block_size % resident_block_size != 0:
                raise ValueError(
                    "HiSparse host and resident block sizes are incompatible."
                )
            self.pages_per_host_block = (
                self.host_manager.block_size // resident_block_size
            )
            self.max_spill_pages = max_model_len // resident_block_size

        self.block_table_updates: set[str] = set()
        self.spills_to_send: list[SparseKVPageTransfer] = []
        self.pending_spills: dict[int, _PendingSpill] = {}
        self.pending_pages: dict[tuple[str, int], int] = {}
        self.host_valid_pages: dict[str, set[int]] = {}
        self.transition_target_pages: dict[str, int] = {}
        self.next_spill_id = 0

    def require_hot_if_needed(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
    ) -> None:
        has_cpu_history = bool(new_computed_blocks) or (
            total_computed_tokens > num_local_computed_tokens
        )
        if self.resident_managers and has_cpu_history:
            block_size = self.resident_managers[0].block_size
            self.host_valid_pages.setdefault(request_id, set()).update(
                range(cdiv(total_computed_tokens, block_size))
            )
        if not self.resident_managers or has_cpu_history:
            for manager in self.hot_managers:
                manager.require_hot(request_id)

    def get_host_block_pool(self) -> BlockPool | None:
        manager = self.host_manager
        return manager.block_pool if manager is not None else None

    def owns_block(self, block: KVCacheBlock) -> bool:
        pool = self.get_host_block_pool()
        return (
            pool is not None
            and block.pool_id is None
            and 0 <= block.block_id < len(pool.blocks)
            and pool.blocks[block.block_id] is block
        )

    def has_host_capacity(self, num_blocks: int) -> bool:
        pool = self.get_host_block_pool()
        return (
            num_blocks == 0
            if pool is None
            else num_blocks <= pool.get_num_free_blocks()
        )

    def get_num_host_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        manager = self.host_manager
        if manager is None:
            return 0
        return manager.get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks[manager.kv_cache_group_id],
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )

    def free_host_blocks(self, blocks: Iterable[KVCacheBlock]) -> list[KVCacheBlock]:
        """Free owned host blocks and return blocks from other pools."""
        host_blocks: list[KVCacheBlock] = []
        other_blocks: list[KVCacheBlock] = []
        for block in blocks:
            (host_blocks if self.owns_block(block) else other_blocks).append(block)
        if host_blocks:
            pool = self.get_host_block_pool()
            assert pool is not None
            pool.free_blocks(host_blocks)
        return other_blocks

    def evict_host_blocks(self, block_ids: set[int]) -> bool:
        pool = self.get_host_block_pool()
        if pool is None:
            return False
        pool.evict_blocks(block_ids)
        return True

    def reset_prefix_cache(self) -> bool:
        pool = self.get_host_block_pool()
        return pool is None or pool.reset_prefix_cache()

    def take_events(self) -> list[KVCacheEvent]:
        pool = self.get_host_block_pool()
        return [] if pool is None else pool.take_events()

    def reclaim_resident_blocks(self, block_pool_id: int, num_blocks: int) -> int:
        """Reclaim host-valid pages and enqueue copies for GPU-only pages."""
        if (
            block_pool_id != self.block_pool_id
            or not self.resident_managers
            or num_blocks <= 0
        ):
            return 0
        resident_managers = self.resident_managers
        spill_plan_budget = max(
            self.max_spill_pages - len(self.spills_to_send),
            0,
        )
        candidates = resident_managers[0].reclaimable_pages()
        for manager in resident_managers[1:]:
            candidates.intersection_update(manager.reclaimable_pages())

        hot_managers = self.hot_managers
        by_request: dict[str, list[int]] = {}
        for request_id, block_idx in candidates:
            by_request.setdefault(request_id, []).append(block_idx)
        num_blocks = max(
            num_blocks,
            len(by_request) * self.num_growing_managers,
        )

        reclaimed = 0
        eventual = sum(
            len(pending.resident_blocks)
            for pending in self.pending_spills.values()
            if pending.release_after
        )
        for request_id, pages in sorted(
            by_request.items(), key=lambda item: len(item[1]), reverse=True
        ):
            pages.sort()
            has_hot = all(manager.has_hot(request_id) for manager in hot_managers)
            if has_hot:
                for block_idx in pages:
                    if eventual + reclaimed >= num_blocks:
                        break
                    if block_idx in self.host_valid_pages.get(request_id, ()):
                        for manager in resident_managers:
                            block = manager.release_resident_page(request_id, block_idx)
                            assert block is not None
                            reclaimed += 1
                        self.block_table_updates.add(request_id)
                    else:
                        pending = (request_id, block_idx) in self.pending_pages
                        if not pending and spill_plan_budget == 0:
                            continue
                        if self._mark_or_plan_release(request_id, block_idx):
                            eventual += len(resident_managers)
                            if not pending:
                                spill_plan_budget -= 1
                continue

            if request_id in self.transition_target_pages:
                continue
            hot_cost = sum(manager.blocks_per_request for manager in hot_managers)
            remaining = max(num_blocks - reclaimed - eventual, 1)
            pages_needed = cdiv(remaining + hot_cost, len(resident_managers))
            pages_needed = min(len(pages), pages_needed, spill_plan_budget)
            if pages_needed * len(resident_managers) <= hot_cost:
                continue
            for hot_manager in hot_managers:
                hot_manager.require_hot(request_id)
            self.transition_target_pages[request_id] = pages_needed
            for block_idx in pages[:pages_needed]:
                pending = (request_id, block_idx) in self.pending_pages
                if self._mark_or_plan_release(request_id, block_idx):
                    eventual += len(resident_managers)
                    if not pending:
                        spill_plan_budget -= 1
            eventual -= hot_cost
            if reclaimed + eventual >= num_blocks:
                break
        return reclaimed

    def _mark_or_plan_release(
        self,
        request_id: str,
        block_idx: int,
    ) -> bool:
        pending_id = self.pending_pages.get((request_id, block_idx))
        if pending_id is not None:
            pending = self.pending_spills[pending_id]
            if pending.release_after:
                return False
            pending.release_after = True
            return True
        return self._plan_spill(
            request_id,
            block_idx,
            release_after=True,
            after_forward=False,
        )

    def plan_prefix_materialization(
        self, request_id: str, num_computed_tokens: int
    ) -> None:
        if not self.resident_managers:
            return
        assert self.host_manager is not None
        host_block_size = self.host_manager.block_size
        num_pages = num_computed_tokens // host_block_size * self.pages_per_host_block
        for page_idx in range(num_pages):
            key = (request_id, page_idx)
            if key in self.pending_pages:
                continue
            if page_idx in self.host_valid_pages.get(request_id, ()):
                continue
            if not all(
                manager.get_resident_page(request_id, page_idx) is not None
                for manager in self.resident_managers
            ):
                continue
            self._plan_spill(
                request_id,
                page_idx,
                release_after=False,
                after_forward=True,
            )

    def _plan_spill(
        self,
        request_id: str,
        page_idx: int,
        *,
        release_after: bool,
        after_forward: bool,
    ) -> bool:
        assert self.host_manager is not None
        host_block_idx = page_idx // self.pages_per_host_block
        host_blocks = self.host_manager.req_to_blocks.get(request_id)
        if host_blocks is None or host_block_idx >= len(host_blocks):
            return False
        host_block = host_blocks[host_block_idx]
        if host_block.is_null:
            return False

        blocks: list[KVCacheBlock] = []
        for manager in self.resident_managers:
            block = manager.get_resident_page(request_id, page_idx)
            if block is None:
                return False
            blocks.append(block)

        self.host_manager.block_pool.touch([host_block])
        for manager, block in zip(self.resident_managers, blocks):
            manager.block_pool.touch([block])
        spill_id = self.next_spill_id
        self.next_spill_id += 1
        plan = SparseKVPageTransfer(
            transfer_id=spill_id,
            destination_block_id=host_block.block_id,
            destination_page_offset=page_idx % self.pages_per_host_block,
            source_block_ids=tuple(block.block_id for block in blocks),
            after_forward=after_forward,
        )
        self.pending_spills[spill_id] = _PendingSpill(
            transfer_id=spill_id,
            page=(request_id, page_idx),
            host_block=host_block,
            resident_blocks=tuple(blocks),
            release_after=release_after,
        )
        self.pending_pages[(request_id, page_idx)] = spill_id
        self.spills_to_send.append(plan)
        return True

    def build_offload_command(
        self,
        request_ids: Sequence[str],
    ) -> SparseKVOffloadCommand | None:
        """Package residency decisions behind the worker store boundary."""
        if not self.resident_managers:
            return None
        block_table_updates = {
            request_id: tuple(
                [block.block_id for block in manager.req_to_blocks.get(request_id, [])]
                for manager in self.managers
            )
            for request_id in self.block_table_updates
        }
        self.block_table_updates.clear()
        command = SparseKVOffloadCommand(
            block_table_updates=block_table_updates,
            page_transfers=self.spills_to_send,
            fully_resident=self.are_requests_fully_resident(request_ids),
        )
        self.spills_to_send = []
        return command

    def has_pending_reclamation(self) -> bool:
        return bool(self.transition_target_pages) or any(
            pending.release_after for pending in self.pending_spills.values()
        )

    def are_requests_fully_resident(self, request_ids: Sequence[str]) -> bool:
        return (
            bool(request_ids)
            and bool(self.resident_managers)
            and all(
                manager.is_fully_resident(request_id)
                for request_id in request_ids
                for manager in self.resident_managers
            )
        )

    def complete_spills(self, spill_ids: Sequence[int] | None) -> None:
        if not spill_ids:
            return
        for spill_id in spill_ids:
            pending = self.pending_spills.get(spill_id)
            if pending is not None:
                pending.copy_enqueued = True

        ready_by_request: dict[str, list[_PendingSpill]] = {}
        for pending in self.pending_spills.values():
            if pending.copy_enqueued:
                ready_by_request.setdefault(pending.page[0], []).append(pending)

        for request_id, ready in ready_by_request.items():
            transition_target = self.transition_target_pages.get(request_id)
            transition_ready = transition_target is not None and (
                sum(pending.release_after for pending in ready) >= transition_target
            )
            for pending in ready:
                if (
                    transition_target is None
                    or transition_ready
                    or not pending.release_after
                ):
                    self._finalize_spill(pending)
            if not transition_ready:
                continue

            del self.transition_target_pages[request_id]
            if any(
                request_id in manager.req_to_blocks
                for manager in self.resident_managers
            ):
                for manager in self.hot_managers:
                    manager.activate_hot(request_id)
                self.block_table_updates.add(request_id)

    def _finalize_spill(self, pending: _PendingSpill) -> None:
        request_id, page_idx = pending.page
        self.pending_spills.pop(pending.transfer_id, None)
        self.pending_pages.pop(pending.page, None)
        table_changed = False
        for manager, block in zip(self.resident_managers, pending.resident_blocks):
            current = manager.get_resident_page(request_id, page_idx)
            if current is block:
                self.host_valid_pages.setdefault(request_id, set()).add(page_idx)
                if pending.release_after:
                    released = manager.release_resident_page(
                        request_id,
                        page_idx,
                        expected_block=block,
                    )
                    assert released is block
                    table_changed = True
            manager.block_pool.free_blocks([block])
        assert self.host_manager is not None
        self.host_manager.block_pool.free_blocks([pending.host_block])
        if table_changed:
            self.block_table_updates.add(request_id)

    def free(self, request_id: str) -> None:
        self.host_valid_pages.pop(request_id, None)
